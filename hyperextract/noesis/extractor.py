"""Noesis Stage 1 extractor: one retry around the LLM extraction call."""

import json
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from .models import ExtractionAlert, ExtractionOutcome
from .prompt import NOESIS_CANONICAL_PROMPT
from .validation import validate_components


def create_noesis_extractor(
    *,
    llm_client: BaseChatModel,
) -> Callable[[str], object]:
    """Build the production raw-JSON call for Noesis event closures.

    Noesis deliberately bypasses ``AutoModel.function_calling`` because tool
    schemas cannot reliably preserve both the authoritative root array and the
    recursive semantic tree. The external contract remains the JSON root array;
    schema and semantic validation happen in ``extract_noesis_components``.
    """
    prompt = ChatPromptTemplate.from_template(NOESIS_CANONICAL_PROMPT)
    chain = prompt | llm_client

    def extract_once(text: str) -> object:
        response = chain.invoke({"source_text": text})
        content = response.text if isinstance(response, BaseMessage) else response
        if not isinstance(content, str):
            raise TypeError("Noesis LLM response content must be text")
        return json.loads(content)

    return extract_once


def extract_noesis_components(
    text: str,
    *,
    extract_once: Callable[[str], object],
) -> ExtractionOutcome:
    """Extract event closure components with at most one retry.

    Call-level failures and schema-level failures retry the whole input once.
    Validator programming errors are never masked: only the pydantic
    ``ValidationError`` from the schema layer counts as an extraction failure.
    Component semantic failures are dropped inside validation without any
    extra call.
    """
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            raw = extract_once(text)
        except Exception as error:
            last_error = error
            continue
        try:
            result = validate_components(raw, source_text=text)
        except ValidationError as error:
            last_error = error
            continue
        return ExtractionOutcome(
            components=result.components,
            alerts=result.alerts,
            attempts=attempt,
        )
    alert = ExtractionAlert(
        stage="hyper_extract",
        alert_code="extraction_failed",
        severity="error",
        message="extraction failed after two attempts",
        details={"error_type": type(last_error).__name__},
    )
    return ExtractionOutcome(components=[], alerts=[alert], attempts=2)
