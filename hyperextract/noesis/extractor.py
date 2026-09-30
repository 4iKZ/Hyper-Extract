"""Noesis Stage 1 extractor: one retry around the LLM extraction call."""

import json
import os
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from .critic import (
    ClauseCriticResult,
    build_clause_retry_feedback,
    create_clause_critic,
    leakage_decisions,
)
from .models import (
    ExtractionAlert,
    ExtractionOutcome,
    FactComponent,
    HypothesisComponent,
    NoesisExtraction,
)
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
                "retry_feedback": retry_feedback or "无。",
            }
        )
        content = response.text if isinstance(response, BaseMessage) else response
        if not isinstance(content, str):
            raise TypeError("Noesis LLM response content must be text")
        return json.loads(content)

    # The fail-closed critic shares the exact same configured LLM client. It is only
    # activated by extract_noesis_components when the explicit experiment flag
    # is enabled, so production behavior remains unchanged by default.
    extract_once._noesis_clause_critic = create_clause_critic(llm_client=llm_client)  # type: ignore[attr-defined]
    return extract_once


def extract_noesis_components(
    text: str,
    *,
    extract_once: Callable[..., object],
    critic_once: Callable[
        [str, list[FactComponent | HypothesisComponent]], ClauseCriticResult
    ]
    | None = None,
) -> ExtractionOutcome:
    """Extract event closure components with at most one retry.

    Call-level failures, schema-level failures and unrepairable-but-
    systematic semantic errors (04A §5: pos gaps, self references, cycles,
    ambiguous targets) retry the whole input once. A second attempt with the
    same problem is accepted as per-component drops. Validator programming
    errors are never masked: only the pydantic ``ValidationError`` from the
    schema layer counts as an extraction failure.

    When enabled, the critic reviews every E occurrence. Failed reviews and
    leakage remaining after the retry reject whole affected components; an
    unlocalized critic failure rejects all unreviewed components.
    """
    if critic_once is None and os.getenv(
        "HYPEREXTRACT_NOESIS_CLAUSE_CRITIC", ""
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        critic_once = getattr(extract_once, "_noesis_clause_critic", None)

    last_error: Exception | None = None
    retry_feedback = ""
    critic_retry_alert: ExtractionAlert | None = None

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
            last_error = None
            continue

        if critic_once is not None:
            try:
                critic_result = critic_once(text, result.components)
            except Exception as error:
                alert = ExtractionAlert(
                    stage="hyper_extract",
                    alert_code="clause_critic_failed",
                    severity="warning",
                    message="clause critic failed; unreviewed components dropped",
                    details={
                        "error_type": type(error).__name__,
                        "dropped_component_indices": list(
                            range(len(result.components))
                        ),
                    },
                )
                return ExtractionOutcome(
                    components=[],
                    alerts=[*result.alerts, alert],
                    attempts=attempt,
                )

            uncertain_indices = {
                decision.component_index
                for decision in critic_result.decisions
                if decision.classification == "UNCERTAIN"
            }
            critic_alerts = []
            if uncertain_indices:
                critic_alerts.append(
                    ExtractionAlert(
                        stage="hyper_extract",
                        alert_code="clause_critic_failed",
                        severity="warning",
                        message="candidate review failed; affected components dropped",
                        details={
                            "dropped_component_indices": sorted(uncertain_indices)
                        },
                    )
                )
            issues = [
                issue
                for issue in leakage_decisions(critic_result)
                if issue.component_index not in uncertain_indices
            ]
            if issues and attempt == 1:
                retry_feedback = build_clause_retry_feedback(critic_result)
                critic_retry_alert = ExtractionAlert(
                    stage="hyper_extract",
                    alert_code="clause_critic_retry",
                    severity="info",
                    message="clause critic requested one targeted extraction retry",
                    details={
                        "issue_count": len(issues),
                        "atom_positions": [
                            {
                                "component_index": issue.component_index,
                                "atom_pos": issue.atom_pos,
                            }
                            for issue in issues
                        ],
                    },
                )
                last_error = None
                continue
            if issues:
                remaining = ExtractionAlert(
                    stage="hyper_extract",
                    alert_code="clause_leakage_remaining",
                    severity="warning",
                    message="clause-shaped Entity Atom remained; affected components dropped",
                    details={
                        "issue_count": len(issues),
                        "dropped_component_indices": sorted(
                            {issue.component_index for issue in issues}
                        ),
                        "atom_positions": [
                            {
                                "component_index": issue.component_index,
                                "atom_pos": issue.atom_pos,
                            }
                            for issue in issues
                        ],
                    },
                )
                alerts = [*result.alerts, *critic_alerts]
                if critic_retry_alert is not None:
                    alerts.append(critic_retry_alert)
                alerts.append(remaining)
                dropped_indices = uncertain_indices | {
                    issue.component_index for issue in issues
                }
                return ExtractionOutcome(
                    components=[
                        component
                        for index, component in enumerate(result.components)
                        if index not in dropped_indices
                    ],
                    alerts=alerts,
                    attempts=attempt,
                )
            result.components = [
                component
                for index, component in enumerate(result.components)
                if index not in uncertain_indices
            ]
            result.alerts.extend(critic_alerts)

        alerts = [*result.alerts]
        if critic_retry_alert is not None:
            alerts.append(critic_retry_alert)
        return ExtractionOutcome(
            components=result.components,
            alerts=alerts,
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
