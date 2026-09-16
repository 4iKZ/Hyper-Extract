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
    NoesisExtraction,
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


def uncorrectable_fact():
    """Semantic failure no bounded repair can fix: text absent from the source."""
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "买", "P", "predicate", None),
            atom(3, "榴莲", "E", "patient", 2),
        ],
        "tree": {
            "predicate": "买",
            "agent": [{"text": "妈妈", "modifier": [], "implied": False}],
            "patient": [{"text": "榴莲", "modifier": [], "implied": False}],
            "modifier": [],
            "nested": [],
            "conditional": [],
        },
    }


def pos_gap_fact():
    """pos numbers skip 3: repair is forbidden, the whole input retries once."""
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "买", "P", "predicate", None),
            atom(4, "苹果", "E", "patient", 2),
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


class RecordingJSONSchemaChatModel(BaseChatModel):
    """Chat model that records the Noesis structured-decoding request."""

    responses: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    call_kwargs: list[dict[str, Any]] = Field(default_factory=list)

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
        self.call_kwargs.append(kwargs)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    def with_structured_output(self, schema, **kwargs):
        raise AssertionError("Noesis must not use function-calling structured output")


class TestProductionJSONSchemaAdapter:
    """The production adapter constrains and preserves the JSON root array."""

    def test_requests_strict_root_array_schema_and_parses_response(self):
        expected = [valid_fact()]
        llm = RecordingJSONSchemaChatModel(
            responses=[json.dumps(expected, ensure_ascii=False)]
        )

        extract_once = create_noesis_extractor(llm_client=llm)
        raw = extract_once(SOURCE_TEXT)

        assert raw == expected
        assert len(llm.calls) == 1
        rendered_prompt = "\n".join(str(message.content) for message in llm.calls[0])
        assert SOURCE_TEXT in rendered_prompt
        response_format = llm.call_kwargs[0]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "noesis_extraction"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == NoesisExtraction.model_json_schema()
        assert response_format["json_schema"]["schema"]["type"] == "array"

    def test_invalid_json_is_retried_by_public_extraction_flow(self):
        llm = RecordingJSONSchemaChatModel(
            responses=["not-json", json.dumps([valid_fact()], ensure_ascii=False)]
        )
        extract_once = create_noesis_extractor(llm_client=llm)

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=extract_once)

        assert len(llm.calls) == 2
        assert outcome.attempts == 2
        assert len(outcome.components) == 1
        assert outcome.alerts == []

    def test_json_object_does_not_get_unwrapped(self):
        llm = RecordingJSONSchemaChatModel(
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
        fake = FakeExtractOnce([[uncorrectable_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 1
        assert outcome.attempts == 1
        assert outcome.components == []
        assert len(outcome.alerts) == 1
        assert outcome.alerts[0].alert_code == "invalid_component_dropped"
        assert outcome.alerts[0].severity == "warning"

    def test_only_invalid_component_dropped_valid_kept_without_extra_call(self):
        fake = FakeExtractOnce([[uncorrectable_fact(), valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 1
        assert outcome.attempts == 1
        assert len(outcome.components) == 1
        assert outcome.components[0].model_dump() == valid_fact()
        assert len(outcome.alerts) == 1
        assert outcome.alerts[0].alert_code == "invalid_component_dropped"
        assert outcome.alerts[0].severity == "warning"

    def test_repairable_coordinate_predicates_fixed_without_retry(self):
        """Two competing root predicates: the tree uniquely picks 买, the
        non-clause verb 吃 demotes to modifier (04A §5, no extra LLM call)."""
        source = "妈妈买苹果吃。"
        fake = FakeExtractOnce([[invalid_fact()]])

        outcome = extract_noesis_components(source, extract_once=fake)

        assert fake.calls == [source]
        assert outcome.attempts == 1
        assert len(outcome.components) == 1
        assert [a["role"] for a in outcome.components[0].model_dump()["atoms"]] == [
            "predicate",
            "modifier",
            "patient",
        ]
        assert len(outcome.alerts) == 1
        assert outcome.alerts[0].severity == "warning"

    def test_pos_gap_retries_whole_input_once_then_drops(self):
        """pos gaps are never renumbered: one full retry, then the component
        is dropped with an alert (04A §5)."""
        fake = FakeExtractOnce([[pos_gap_fact()], [pos_gap_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 2
        assert outcome.attempts == 2
        assert outcome.components == []
        assert len(outcome.alerts) == 1
        assert outcome.alerts[0].alert_code == "invalid_component_dropped"

    def test_pos_gap_retry_recovers_when_second_attempt_is_valid(self):
        fake = FakeExtractOnce([[pos_gap_fact()], [valid_fact()]])

        outcome = extract_noesis_components(SOURCE_TEXT, extract_once=fake)

        assert len(fake.calls) == 2
        assert outcome.attempts == 2
        assert len(outcome.components) == 1
        assert outcome.components[0].model_dump() == valid_fact()
        assert outcome.alerts == []

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
    """The three authoritative examples must pass field-by-field (requirement 15.6)."""

    @pytest.fixture(scope="class")
    def golden_cases(self):
        with open(GOLDEN_CASES_PATH, encoding="utf-8") as file:
            return json.load(file)

    def test_fixture_contains_three_cases(self, golden_cases):
        assert len(golden_cases) == 3
        for case in golden_cases:
            assert set(case.keys()) == {"input", "expected"}
            assert isinstance(case["expected"], list)

    @pytest.mark.parametrize("case_index", [0, 1, 2])
    def test_golden_case_validates_field_by_field(self, golden_cases, case_index):
        case = golden_cases[case_index]

        result = validate_components(case["expected"], source_text=case["input"])

        assert result.alerts == []
        dumped = [component.model_dump() for component in result.components]
        assert dumped == case["expected"]

    def test_coordinate_split_regression_not_in_prompt(self):
        """坐/吃/玩 stays a plain regression case: three independent components,
        never a fourth prompt example (04A §4)."""
        with open(GOLDEN_CASES_PATH, encoding="utf-8") as file:
            golden_cases = json.load(file)

        assert not any("坐" in case["input"] for case in golden_cases)
        assert not any("没写" in case["input"] for case in golden_cases)


class TestCanonicalPromptTerminology:
    """The latest +1 terminology and three authoritative examples are frozen."""

    def test_uses_cogneme_terms_and_epg_storage_codes(self):
        assert "概元（Cogneme）" in NOESIS_CANONICAL_PROMPT
        assert "实元（Enteme）" in NOESIS_CANONICAL_PROMPT
        assert "谓元（Prediceme）" in NOESIS_CANONICAL_PROMPT
        assert "构元（Geneme）" in NOESIS_CANONICAL_PROMPT
        assert "E、P、G" in NOESIS_CANONICAL_PROMPT
        assert "总称 C 只用于文档和讨论" in NOESIS_CANONICAL_PROMPT

    def test_contains_exactly_three_authoritative_examples(self):
        assert NOESIS_CANONICAL_PROMPT.count("### 示例 ") == 3
        assert "以下三个示例" in NOESIS_CANONICAL_PROMPT
        # The project-added fourth example is gone from the production prompt.
        assert "小明坐在沙发上" not in NOESIS_CANONICAL_PROMPT
        assert "示例 4" not in NOESIS_CANONICAL_PROMPT
        # The old third example is gone too.
        assert "没写作业" not in NOESIS_CANONICAL_PROMPT

    def test_third_example_is_the_composite_ones(self):
        """The composite example: fact 让/打 chain plus resolved hypothesis."""
        assert "妈妈让小明打酱油，否则就揍他。" in NOESIS_CANONICAL_PROMPT
        assert '"text": "让", "type": "P", "role": "predicate", "target_occ": null, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "打", "type": "P", "role": "predicate", "target_occ": 3, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "酱油", "type": "E", "role": "patient", "target_occ": 4, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": true' in NOESIS_CANONICAL_PROMPT
        # The nested 打 clause records the shared 小明 as an implied agent.
        assert '"text": "小明", "modifier": [], "implied": true' in NOESIS_CANONICAL_PROMPT
        # The implied 小明 occurrence is never duplicated in the fact atoms.
        after_input = NOESIS_CANONICAL_PROMPT.split("妈妈让小明打酱油，否则就揍他。")[1]
        fact_atoms = after_input.split('"utterance_type": "fact"')[1].split('"utterance_type": "hypothesis"')[0]
        fact_atoms = fact_atoms.split('"tree"')[0]
        assert fact_atoms.count('"text": "小明"') == 1
        # The rule track keeps the composite expression and the negation.
        assert '"打酱油"' in NOESIS_CANONICAL_PROMPT
        assert '"NOT 打酱油"' in NOESIS_CANONICAL_PROMPT

    def test_subordinate_predicate_points_to_the_modified_occurrence(self):
        """需求 08 §4.4: 打 points at 小明, not at the parent predicate 让."""
        assert "从属句 predicate 指向它在父句中实际修饰的 agent 或 patient occurrence" in NOESIS_CANONICAL_PROMPT
        assert "从属句 predicate 指向上级 SPO 的核心 predicate" not in NOESIS_CANONICAL_PROMPT

    def test_examples_keep_confirmed_target_and_resolved_contract(self):
        assert '"text": "太阳", "type": "E", "role": "agent", "target_occ": 3, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "每天", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null' in NOESIS_CANONICAL_PROMPT

    def test_explicitly_splits_coordinate_predicates_even_with_shared_context(self):
        assert "共享同一主语、时间或语境" in NOESIS_CANONICAL_PROMPT
        assert "并列且互不从属" in NOESIS_CANONICAL_PROMPT
        assert "必须拆成多个 component" in NOESIS_CANONICAL_PROMPT
        assert "共享主语或同时发生本身不构成从属关系" in NOESIS_CANONICAL_PROMPT
        assert "论元、修饰事件或条件事件" in NOESIS_CANONICAL_PROMPT
        assert "不得再额外输出包含这些并列动作的聚合 component" in NOESIS_CANONICAL_PROMPT
