"""Unit tests for the Noesis extractor retry/alert flow and golden cases (15.5/15.6)."""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
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

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def entity_fact(clause):
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "监控", "E", "agent", 2),
            atom(2, "显示", "P", "predicate", None),
            atom(3, clause, "E", "patient", 2),
        ],
        "tree": {
            "predicate": "显示",
            "agent": [{"text": "监控", "modifier": [], "implied": False}],
            "patient": [{"text": clause, "modifier": [], "implied": False}],
            "modifier": [],
            "nested": [],
            "conditional": [],
        },
    }


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

        assert fake.calls == [(SOURCE_TEXT, {})]
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

        assert fake.calls == [(source, {})]
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

    def test_error_literal_entity_is_kept_without_retry(self):
        literal = "Failed to parse multiline timeout"
        source = f"监控显示 {literal}。"
        component = entity_fact(literal)
        fake = FakeExtractOnce([[component]])

        outcome = extract_noesis_components(source, extract_once=fake)

        assert len(fake.calls) == 1
        assert outcome.attempts == 1
        assert [item.model_dump() for item in outcome.components] == [component]
        assert outcome.alerts == []

    def test_valid_literal_does_not_resample_other_components(self):
        literal = "Failed to parse multiline timeout"
        source = f"监控显示 {literal}。昨天妈妈在超市买了苹果。"
        components = [entity_fact(literal), valid_fact()]
        fake = FakeExtractOnce([components])

        outcome = extract_noesis_components(source, extract_once=fake)

        assert outcome.attempts == 1
        assert len(fake.calls) == 1
        assert [item.model_dump() for item in outcome.components] == components
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

    def test_uses_epg_storage_codes(self):
        assert "type 只有 E、P、G 三个值" in NOESIS_CANONICAL_PROMPT
        assert "G 由系统构造，你不要输出" in NOESIS_CANONICAL_PROMPT

    def test_contains_the_three_documented_examples(self):
        assert NOESIS_CANONICAL_PROMPT.count("### 示例 ") == 3
        assert "以下三个示例" in NOESIS_CANONICAL_PROMPT
        headings = [
            "纯事实",
            "纯规律",
            "### 示例 3：事实＋规律复合",
        ]
        positions = [NOESIS_CANONICAL_PROMPT.index(heading) for heading in headings]
        assert positions == sorted(positions)
        assert "昨天妈妈在超市买了苹果。" in NOESIS_CANONICAL_PROMPT
        assert "太阳每天从东边升起。" in NOESIS_CANONICAL_PROMPT
        # The previously removed project-added example stays absent.
        assert "小明坐在沙发上" not in NOESIS_CANONICAL_PROMPT
        # The old third example is gone too.
        assert "没写作业" not in NOESIS_CANONICAL_PROMPT

    def test_third_example_is_the_composite_ones(self):
        """The composite example: fact 让/打 chain plus resolved hypothesis."""
        assert "妈妈让小明打酱油，否则就揍他。" in NOESIS_CANONICAL_PROMPT
        assert '"text": "让", "type": "P", "role": "predicate", "target_occ": null, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "打", "type": "P", "role": "predicate", "target_occ": 2, "resolved": null' in NOESIS_CANONICAL_PROMPT
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

    def test_examples_keep_confirmed_target_and_resolved_contract(self):
        assert '"text": "打", "type": "P", "role": "predicate", "target_occ": 2, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "小明", "modifier": [], "implied": true' in NOESIS_CANONICAL_PROMPT
        assert '"text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": true' in NOESIS_CANONICAL_PROMPT

    def test_rendered_examples_validate_and_demonstrate_alignment(self):
        rendered = ChatPromptTemplate.from_template(NOESIS_CANONICAL_PROMPT).format(
            source_text="User: 验证输入。", retry_feedback="无。"
        )
        examples = []
        for section in rendered.split("### 示例 ")[1:]:
            input_part = section.split("输入：", 1)[1].split("输出：", 1)[0]
            source = input_part.split("判定边界：", 1)[0].strip()
            raw, _ = json.JSONDecoder().raw_decode(
                section.split("输出：", 1)[1].lstrip()
            )
            result = validate_components(raw, source_text=source)
            assert len(result.components) == len(raw), result.alerts
            assert not result.retry_needed
            examples.append(raw)
        assert len(examples) == 3
        assert [atom["target_occ"] for atom in examples[0][0]["atoms"]] == [4, 4, 4, None, 4]
        assert examples[1][0]["utterance_type"] == "hypothesis"
        assert examples[1][0]["rule_template"]["conclusion"]["predicate"] == "升起"
        assert [component["utterance_type"] for component in examples[2]] == ["fact", "hypothesis"]
        assert examples[2][0]["atoms"][3]["target_occ"] == 2
        assert examples[2][1]["atoms"][2]["resolved"] is True

    def test_defines_minimal_entity_granularity_without_dropping_values(self):
        assert "最小且可独立指称" in NOESIS_CANONICAL_PROMPT
        assert "完整命题不得作为单个 E" in NOESIS_CANONICAL_PROMPT
        assert "时间、数量、地点、标识符、路径、版本号、百分比和数值都作为 E 保留" in NOESIS_CANONICAL_PROMPT

    def test_relation_words_keep_both_ends_in_one_closure(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "关系两端必须在同一个 component 里" in prompt
        assert "它本身是 P，作为这一层的根" in prompt
        assert "以结果事件的谓词为根" in prompt
        assert "连接词本身保留为一个 modifier" in prompt
        assert "conclusion 的 predicate 必须等于根谓词" in prompt

    def test_examples_are_positive_only(self):
        assert "错误：" not in NOESIS_CANONICAL_PROMPT
        assert "反例" not in NOESIS_CANONICAL_PROMPT

    def test_relation_examples_validate_and_follow_root_rules(self):
        rendered = ChatPromptTemplate.from_template(NOESIS_CANONICAL_PROMPT).format(
            source_text="验证输入", retry_feedback="无。"
        )
        outputs = []
        for section in rendered.split("### 关系示例 ")[1:]:
            source = section.split("输入：", 1)[1].split("\n", 1)[0].strip()
            raw, _ = json.JSONDecoder().raw_decode(section.split("输出：", 1)[1].lstrip())
            checked = validate_components(raw, source_text=source)
            assert not checked.alerts and not checked.retry_needed
            assert [c.model_dump(mode="json") for c in checked.components] == raw
            atoms = [atom for component in raw for atom in component["atoms"]]
            assert not {a["text"] for a in atoms} & {"但", "了", "被", "并"}  # function words never become atoms
            assert all(a["role"] == "modifier" for a in atoms if a["text"] in {"所以", "后"})  # connectors stay as modifiers
            outputs.append(raw)
        assert [len(raw) for raw in outputs] == [1, 1, 1, 1, 1, 3]
        assert [raw[0]["tree"]["predicate"] for raw in outputs] == ["降到", "拦截", "判断为", "造成", "引发", "补"]
        assert outputs[3][0]["rule_template"]["conclusion"]["predicate"] == "造成"
        # "后" modifies the antecedent, "所以" the result
        assert outputs[0][0]["tree"]["nested"][0]["modifier"] == ["后"]
        assert outputs[1][0]["tree"]["modifier"] == ["所以", "没有"]
        assert outputs[4][0]["tree"]["nested"][0]["predicate"] == "导致"
        # coordinate actions repeat the shared subject; negation is a modifier, never inside P
        assert [c["atoms"][0]["text"] for c in outputs[5][:2]] == ["运维组", "运维组"]
        assert outputs[5][2]["tree"] == {"predicate": "中断", "agent": [], "patient": [{"text": "备份任务", "modifier": [], "implied": False}], "modifier": ["没有"], "nested": [], "conditional": []}
