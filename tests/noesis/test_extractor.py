"""Unit tests for the Noesis extractor retry/alert flow and golden cases (15.5/15.6)."""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from hyperextract.noesis import (
    create_noesis_extractor,
    extract_noesis_components,
    validate_components,
)
from hyperextract.noesis.prompt import NOESIS_CANONICAL_PROMPT

GOLDEN_CASES_PATH = Path(__file__).parent / "fixtures" / "golden_cases.json"
SOURCE_TEXT = "昨天妈妈在超市买了苹果。"


def atom(pos, text, type_, role, target_occ, resolved=None):
    return {
        "pos": pos,
        "text": text,
        "type": type_,
        "role": role,
        "target_occ": target_occ,
        "resolved": resolved,
    }


def valid_fact():
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "买", "P", "predicate", None),
            atom(3, "苹果", "E", "patient", 2),
        ],
        "tree": {
            "predicate": "买",
            "agent": [{"text": "妈妈", "modifier": [], "implied": False}],
            "patient": [{"text": "苹果", "modifier": [], "implied": False}],
            "modifier": [],
            "nested": [],
            "conditional": [],
        },
    }


def invalid_fact():
    """Schema-valid fact that violates the closure: two root predicates."""
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "买", "P", "predicate", None),
            atom(2, "吃", "P", "predicate", None),
            atom(3, "苹果", "E", "patient", 1),
        ],
        "tree": {
            "predicate": "买",
            "agent": [],
            "patient": [{"text": "苹果", "modifier": [], "implied": False}],
            "modifier": [],
            "nested": [],
            "conditional": [],
        },
    }


class FakeExtractOnce:
    """Deterministic offline stand-in for one LLM extraction call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class RecordingRawJSONChatModel(BaseChatModel):
    """Chat model that fails immediately if production uses structured tools."""

    responses: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-raw-json"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    def with_structured_output(self, schema, **kwargs):
        raise AssertionError("Noesis must not use function-calling structured output")


class TestProductionRawJSONAdapter:
    """The production adapter preserves the root array without tool schemas."""

    def test_invokes_chat_model_directly_and_parses_root_array(self):
        expected = [valid_fact()]
        llm = RecordingRawJSONChatModel(
            responses=[json.dumps(expected, ensure_ascii=False)]
        )

        extract_once = create_noesis_extractor(llm_client=llm)
        raw = extract_once(SOURCE_TEXT)

        assert raw == expected
        assert len(llm.calls) == 1
        rendered_prompt = "\n".join(str(message.content) for message in llm.calls[0])
        assert SOURCE_TEXT in rendered_prompt

    def test_invalid_json_is_retried_by_public_extraction_flow(self):
        llm = RecordingRawJSONChatModel(
            responses=["not-json", json.dumps([valid_fact()], ensure_ascii=False)]
        )
        extract_once = create_noesis_extractor(llm_client=llm)

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=extract_once)

        assert len(llm.calls) == 2
        assert outcome.attempts == 2
        assert len(outcome.components) == 1
        assert outcome.alerts == []

    def test_json_object_does_not_get_unwrapped(self):
        llm = RecordingRawJSONChatModel(
            responses=[json.dumps({"components": [valid_fact()]}, ensure_ascii=False)]
        )

        extract_once = create_noesis_extractor(llm_client=llm)
        raw = extract_once(SOURCE_TEXT)

        assert raw == {"components": [valid_fact()]}


class TestExtractorRetry:
    """Retry and alert behavior for schema/call-level failures (requirement 15.5)."""

    def test_valid_first_call_single_attempt(self):
        fake = FakeExtractOnce([[valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert fake.calls == [SOURCE_TEXT]
        assert outcome.attempts == 1
        assert len(outcome.components) == 1
        assert outcome.alerts == []

    def test_empty_array_no_retry_no_alert(self):
        fake = FakeExtractOnce([[]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 1
        assert outcome.attempts == 1
        assert outcome.components == []
        assert outcome.alerts == []

    def test_call_exception_retried_then_success(self):
        fake = FakeExtractOnce([RuntimeError("llm unavailable"), [valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 2
        assert outcome.attempts == 2
        assert len(outcome.components) == 1
        assert outcome.alerts == []

    def test_non_array_response_retried_then_success(self):
        fake = FakeExtractOnce([{"components": [valid_fact()]}, [valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 2
        assert outcome.attempts == 2
        assert len(outcome.components) == 1
        assert outcome.alerts == []

    def test_schema_failure_retried_then_success(self):
        broken = valid_fact()
        broken["unexpected_field"] = True
        fake = FakeExtractOnce([[broken], [valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 2
        assert outcome.attempts == 2
        assert len(outcome.components) == 1
        assert outcome.alerts == []

    def test_both_attempts_fail_returns_error_alert(self):
        fake = FakeExtractOnce([RuntimeError("first"), RuntimeError("second")])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 2
        assert outcome.attempts == 2
        assert outcome.components == []
        assert len(outcome.alerts) == 1
        alert = outcome.alerts[0]
        assert alert.alert_code == "extraction_failed"
        assert alert.severity == "error"
        assert alert.stage == "hyper_extract"

    def test_semantic_failure_does_not_trigger_retry(self):
        fake = FakeExtractOnce([[invalid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 1
        assert outcome.attempts == 1
        assert outcome.components == []
        assert len(outcome.alerts) == 1
        assert outcome.alerts[0].alert_code == "invalid_component_dropped"
        assert outcome.alerts[0].severity == "warning"

    def test_only_invalid_component_dropped_valid_kept_without_extra_call(self):
        fake = FakeExtractOnce([[invalid_fact(), valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 1
        assert outcome.attempts == 1
        assert len(outcome.components) == 1
        assert outcome.components[0].model_dump() == valid_fact()
        assert len(outcome.alerts) == 1
        assert outcome.alerts[0].alert_code == "invalid_component_dropped"
        assert outcome.alerts[0].severity == "warning"

    def test_validator_internal_error_propagates(self, monkeypatch):
        """Only call-level and schema-level failures retry; a validator
        programming error must surface instead of being masked as an
        extraction failure."""
        import hyperextract.noesis.extractor as extractor_module

        def broken_validator(raw, *, source_text):
            raise RuntimeError("validator bug")

        monkeypatch.setattr(extractor_module, "validate_components", broken_validator)
        fake = FakeExtractOnce([[valid_fact()]])

        with pytest.raises(RuntimeError, match="validator bug"):
            extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 1


class TestGoldenCases:
    """The four authoritative examples must pass field-by-field (requirement 15.6)."""

    @pytest.fixture(scope="class")
    def golden_cases(self):
        with open(GOLDEN_CASES_PATH, encoding="utf-8") as file:
            return json.load(file)

    def test_fixture_contains_four_cases(self, golden_cases):
        assert len(golden_cases) == 4
        for case in golden_cases:
            assert set(case.keys()) == {"input", "expected"}
            assert isinstance(case["expected"], list)

    @pytest.mark.parametrize("case_index", [0, 1, 2, 3])
    def test_golden_case_validates_field_by_field(self, golden_cases, case_index):
        case = golden_cases[case_index]

        result = validate_components(case["expected"], source_text=case["input"])

        assert result.alerts == []
        dumped = [component.model_dump() for component in result.components]
        assert dumped == case["expected"]


class TestCanonicalPromptTerminology:
    """The latest +1 terminology and four authoritative examples are frozen."""

    def test_uses_cogneme_terms_and_epg_storage_codes(self):
        assert "概元（Cogneme）" in NOESIS_CANONICAL_PROMPT
        assert "实元（Enteme）" in NOESIS_CANONICAL_PROMPT
        assert "谓元（Prediceme）" in NOESIS_CANONICAL_PROMPT
        assert "构元（Geneme）" in NOESIS_CANONICAL_PROMPT
        assert "E、P、G" in NOESIS_CANONICAL_PROMPT
        assert "总称 C 只用于文档和讨论" in NOESIS_CANONICAL_PROMPT

    def test_contains_fourth_coordinate_predicate_example(self):
        assert NOESIS_CANONICAL_PROMPT.count("### 示例 ") == 4
        assert "以下四个示例" in NOESIS_CANONICAL_PROMPT
        assert "小明坐在沙发上，吃着苹果，玩着苹果手机。" in NOESIS_CANONICAL_PROMPT
        assert "### 示例 4：共享主语的并列事实拆分" in NOESIS_CANONICAL_PROMPT

    def test_examples_keep_confirmed_target_and_resolved_contract(self):
        assert '"text": "太阳", "type": "E", "role": "agent", "target_occ": 3, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "每天", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "没写", "type": "P", "role": "predicate", "target_occ": 4, "resolved": null' in NOESIS_CANONICAL_PROMPT

    def test_explicitly_splits_coordinate_predicates_even_with_shared_context(self):
        assert "共享同一主语、时间或语境" in NOESIS_CANONICAL_PROMPT
        assert "并列且互不从属" in NOESIS_CANONICAL_PROMPT
        assert "必须拆成多个 component" in NOESIS_CANONICAL_PROMPT
        assert "共享主语或同时发生本身不构成从属关系" in NOESIS_CANONICAL_PROMPT
        assert "论元、修饰事件或条件事件" in NOESIS_CANONICAL_PROMPT
        assert "不得再额外输出包含这些并列动作的聚合 component" in NOESIS_CANONICAL_PROMPT
