"""Query-atoms stage: one raw user utterance -> flat atom list, in a single LLM call."""

import json
from collections.abc import Callable
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from ._llm_json import loads_llm_json
from .query_prompt import QUERY_ATOMS_PROMPT


class QueryAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    type: Literal["E", "P"]
    role: Literal["agent", "predicate", "patient", "modifier"]


class QueryAtoms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atoms: list[QueryAtom]


def create_noesis_query_extractor(
    *, llm_client: BaseChatModel, json_mode: bool = False
) -> Callable[[str], object]:
    """Same provider constraints as the closure extractor: JSON Schema, or JSON mode."""
    response_format = (
        {"type": "json_object"}
        if json_mode
        else {
            "type": "json_schema",
            "json_schema": {
                "name": "noesis_query_atoms",
                "strict": True,
                "schema": QueryAtoms.model_json_schema(),
            },
        }
    )
    chain = ChatPromptTemplate.from_messages(
        [SystemMessage(content=QUERY_ATOMS_PROMPT), ("human", "{query}")]
    ) | llm_client.bind(response_format=response_format, max_tokens=32768)

    def extract_once(text: str) -> object:
        response = chain.invoke({"query": json.dumps({"query": text}, ensure_ascii=False)})
        if isinstance(response, BaseMessage):
            if response.response_metadata.get("finish_reason") not in (None, "stop", "end_turn"):
                raise ValueError("Noesis query extraction did not finish")
            content = response.text
        else:
            content = response
        if not isinstance(content, str):
            raise TypeError("Noesis query response content must be text")
        return loads_llm_json(content)

    return extract_once


def extract_noesis_query_atoms(text: str, *, extract_once: Callable[[str], object]) -> list[dict[str, str]]:
    """Validate the model output; retry call/format failures once, never return unchecked atoms.

    An empty list is a valid answer (nothing to extract). Atoms whose text is not a substring
    of the input, or whose type contradicts the role, are dropped; if every atom is dropped the
    attempt counts as a failure.
    shortcut: exact substring check, so a model that normalizes whitespace loses that atom;
    relax to whitespace-insensitive matching if real traffic shows it.
    """
    for attempt in (1, 2):
        try:
            parsed = QueryAtoms.model_validate(extract_once(text))
            atoms = [
                atom
                for atom in parsed.atoms
                if atom.text in text
                and (atom.role not in ("agent", "patient") or atom.type == "E")
                and (atom.role != "predicate" or atom.type == "P")
            ]
            if parsed.atoms and not atoms:
                raise ValueError("every atom failed validation")
            return [atom.model_dump() for atom in atoms]
        except Exception:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")
