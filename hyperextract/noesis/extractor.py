"""Noesis Stage 1 extractor: one retry around the LLM extraction call."""

import json
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from .models import ExtractionAlert, ExtractionOutcome, NoesisExtraction
from .prompt import NOESIS_CANONICAL_PROMPT
from .validation import validate_components


def create_noesis_extractor(
    *,
    llm_client: BaseChatModel,
) -> Callable[..., object]:
    """Build the production strict-JSON-Schema call for Noesis closures.

    Noesis deliberately bypasses ``AutoModel.function_calling`` because tool
    schemas cannot reliably preserve both the authoritative root array and the
    recursive semantic tree. The provider constrains the response to the
    authoritative root-array schema; application-level semantic validation
    remains in ``extract_noesis_components``.
    """
    prompt = ChatPromptTemplate.from_template(NOESIS_CANONICAL_PROMPT)
    schema_client = llm_client.bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "noesis_extraction",
                "strict": True,
                "schema": NoesisExtraction.model_json_schema(),
            },
        }
    )
    chain = prompt | schema_client

    def extract_once(text: str, *, retry_feedback: str = "") -> object:
        response = chain.invoke(
            {
                "source_text": text,
                "retry_feedback": retry_feedback or "无；这是首次抽取。",
            }
        )
        content = response.text if isinstance(response, BaseMessage) else response
        if not isinstance(content, str):
            raise TypeError("Noesis LLM response content must be text")
        return json.loads(content)

    return extract_once


def extract_noesis_components(
    text: str,
    *,
    extract_once: Callable[..., object],
) -> ExtractionOutcome:
    """Extract event closure components with at most one retry.

    Call-level failures, schema-level failures and unrepairable-but-
    systematic semantic errors (04A §5: pos gaps, self references, cycles,
    ambiguous targets) retry the whole input once. A second attempt with the
    same problem is accepted as per-component drops. Validator programming
    errors are never masked: only the pydantic ``ValidationError`` from the
    schema layer counts as an extraction failure.
    """
    last_error: Exception | None = None
    retry_feedback = ""
    for attempt in (1, 2):
        try:
            if retry_feedback:
                raw = extract_once(text, retry_feedback=retry_feedback)
            else:
                raw = extract_once(text)
        except Exception as error:
            last_error = error
            continue
        try:
            result = validate_components(raw, source_text=text)
        except ValidationError as error:
            last_error = error
            continue
        if result.retry_needed and attempt == 1:
            retry_feedback = _build_entity_retry_feedback(raw, result.alerts)
            last_error = None
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


def _build_entity_retry_feedback(
    raw: object,
    alerts: list[ExtractionAlert],
) -> str:
    """Build transient, targeted feedback for proposition-shaped entities."""
    if not isinstance(raw, list):
        return ""
    issues: list[str] = []
    for alert in alerts:
        details = alert.details or {}
        if details.get("rule") != "entity_clause_shape":
            continue
        component_index = details.get("component_index")
        positions = details.get("atom_positions")
        if not isinstance(component_index, int) or not isinstance(positions, list):
            continue
        if component_index < 0 or component_index >= len(raw):
            continue
        component = raw[component_index]
        atoms = component.get("atoms") if isinstance(component, dict) else None
        if not isinstance(atoms, list):
            continue
        for position in positions:
            text = next(
                (
                    atom.get("text")
                    for atom in atoms
                    if isinstance(atom, dict) and atom.get("pos") == position
                ),
                None,
            )
            if isinstance(text, str):
                issues.append(f'component {component_index}, pos {position}: "{text}"')
    if not issues:
        return ""
    return (
        "上次输出把完整动作或状态命题错误地放进了 E：\n- "
        + "\n- ".join(issues)
        + "\n请将其拆为 P、论元和 modifier，并重建 atoms、tree 与 target_occ；"
        "如果无法确定正确结构，就省略对应 component。"
    )
