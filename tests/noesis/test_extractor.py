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

    def test_uses_cogneme_terms_and_epg_storage_codes(self):
        assert "概元（Cogneme）" in NOESIS_CANONICAL_PROMPT
        assert "实元（Enteme）" in NOESIS_CANONICAL_PROMPT
        assert "谓元（Prediceme）" in NOESIS_CANONICAL_PROMPT
        assert "构元（Geneme）" in NOESIS_CANONICAL_PROMPT
        assert "E、P、G" in NOESIS_CANONICAL_PROMPT
        assert "总称 C 只用于文档和讨论" in NOESIS_CANONICAL_PROMPT

    def test_defines_minimal_entity_granularity_without_dropping_values(self):
        assert "最小且可独立指称" in NOESIS_CANONICAL_PROMPT
        assert "完整命题不得作为单个 E" in NOESIS_CANONICAL_PROMPT
        assert "拆成 P 及其论元或修饰语" in NOESIS_CANONICAL_PROMPT
        assert "从属命题通过 nested 或 conditional" in NOESIS_CANONICAL_PROMPT
        assert "时间、数量、地点、标识符、路径、版本号、百分比和标量值仍可作为 E" in NOESIS_CANONICAL_PROMPT

    def test_concrete_observation_does_not_require_an_explicit_time_word(self):
        assert "不得仅因缺少显式时间词而过滤" in NOESIS_CANONICAL_PROMPT
        assert "完成态或当前状态的一次具体观察" in NOESIS_CANONICAL_PROMPT

    def test_predicates_are_classified_before_atoms_are_constructed(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert prompt.index("先逐个列出谓词候选") < prompt.index("逐个判定其证据类型")
        assert prompt.index("逐个判定其证据类型") < prompt.index("先为每个保留的谓词建立 P")
        assert prompt.index("先为每个保留的谓词建立 P") < prompt.index("再把剩余成分构造成 E")
        assert "DIRECT、REPORTED、LITERAL、NONASSERTIVE" in prompt
        assert "每个 DIRECT 或 REPORTED 谓词候选都已有对应的 P" in prompt
        assert "每个 LITERAL 谓词候选都没有被提升为事实" in prompt
        assert "完整命题不得作为单个 E" in prompt

    def test_literal_exemption_is_local_to_an_identifiable_payload(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "引用豁免只作用于可明确定位的字面量范围" in prompt
        assert "前面出现命令或日志，不会使后续自然语言陈述自动变成 LITERAL" in prompt
        assert "普通陈述不能仅因包含技术术语、数值或状态词而使用引用豁免" in prompt

    def test_contrastive_diagnosis_keeps_the_positive_state_predicate(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "否定前项不能使转折后的肯定状态消失" in prompt
        assert "“被阻塞”“等”均为 DIRECT" in prompt

    def test_standalone_machine_error_line_is_literal(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "程序名、错误级别、错误码和消息" in prompt
        assert "整行均为 LITERAL 并返回 []" in prompt
        assert "消息中的 changed、failed 或其他动作词" in prompt

    def test_quoted_diagnostics_do_not_create_unasserted_events(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "机器原始输出没有外层陈述时才返回 []" in prompt
        assert "不从字面量中的动作词推断事件发生" in prompt
        assert "只抽取字面量出现这一事件" in prompt
        assert "保留报告谓词与被报告事件的从属关系" in prompt
        assert "问题、指令和命令输出本身不构成事件断言" in prompt
        assert "引号本身不授予 LITERAL 豁免" in prompt
        assert "直接陈述的状态变化属于 fact 候选" in prompt

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

    def test_subordinate_predicate_points_to_the_modified_occurrence(self):
        """An action content clause targets its parent action; attributes target objects."""
        assert "从属句 predicate 指向它在父句中实际修饰的 agent 或 patient occurrence" in NOESIS_CANONICAL_PROMPT
        assert "安排的动作内容指向安排谓词" in NOESIS_CANONICAL_PROMPT
        assert "从属句 predicate 指向上级 SPO 的核心 predicate" not in NOESIS_CANONICAL_PROMPT

    def test_examples_keep_confirmed_target_and_resolved_contract(self):
        assert '"text": "打", "type": "P", "role": "predicate", "target_occ": 2, "resolved": null' in NOESIS_CANONICAL_PROMPT
        assert '"text": "小明", "modifier": [], "implied": true' in NOESIS_CANONICAL_PROMPT
        assert '"text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": true' in NOESIS_CANONICAL_PROMPT

    def test_reported_arrangement_content_is_not_a_completed_child_action(self):
        assert "已发生的安排与安排内容不能一起丢弃" in NOESIS_CANONICAL_PROMPT
        assert "嵌套内容不等于另行断言该动作已经完成" in NOESIS_CANONICAL_PROMPT

    def test_explicitly_splits_coordinate_predicates_even_with_shared_context(self):
        assert (
            "相同对象、时间、话题、相邻出现或纯时间先后不构成从属关系"
            in NOESIS_CANONICAL_PROMPT
        )
        assert "才将纯并列动作分别输出" in NOESIS_CANONICAL_PROMPT

    def test_explicit_relations_take_precedence_over_clause_boundaries(self):
        assert "不得先给每个谓词分配 component" in NOESIS_CANONICAL_PROMPT
        assert "独立成句不等于独立事件" in NOESIS_CANONICAL_PROMPT
        assert "不为减少 component 数量制造关系" in NOESIS_CANONICAL_PROMPT
        assert "仅将语义互不依赖的独立事件拆成不同 component" in NOESIS_CANONICAL_PROMPT
        assert "共享实体本身既不能证明合并，也不能证明拆分" in NOESIS_CANONICAL_PROMPT
        assert "真实从属的动作、状态、内容或证据事件" in NOESIS_CANONICAL_PROMPT
        assert "不额外输出聚合副本" in NOESIS_CANONICAL_PROMPT

    def test_summarized_input_is_not_preverified_fact(self):
        assert "输入是第一阶段整理后的叙述" in NOESIS_CANONICAL_PROMPT
        assert "整理不等于已核实事实或完成语义拆分" in NOESIS_CANONICAL_PROMPT
        assert "关系表达就是本次要保留的关系" in NOESIS_CANONICAL_PROMPT
        assert "未证实猜测、建议、计划和规范要求不能借 hypothesis 重新准入" in NOESIS_CANONICAL_PROMPT

    def test_uncertainty_scope_is_checked_before_admission(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "先判定模态的作用域" in prompt
        assert "不得先删去不确定表达" in prompt
        assert "不扩散到同句中独立明确陈述的已发生操作" in prompt

    def test_closure_partition_precedes_building_predicate_atoms(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "## 闭包优先的任务顺序" in prompt
        assert prompt.index("再识别输入已经表达的关系、划定闭包") < prompt.index(
            "先为每个保留的谓词建立 P"
        )
        assert "一个 component 可以有多个 P" in prompt
        assert "不能因各自能成句或各有谓词而分别输出孤立 component" in prompt

    def test_closure_partition_uses_local_occurrences_not_global_text_uniqueness(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert (
            "不同 component 独立建立局部 pos，不共享 occurrence；允许重复实体字面"
            in prompt
        )
        assert "明确作为判断依据的多项观察可共同从属该判断" in prompt
        assert "不为减少 component 数量制造关系" in prompt

    def test_unasserted_configuration_condition_is_omitted_not_literal(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "配置了某个告警、触发或筛选条件，不等于条件已经满足" in prompt
        assert "只保留已经发生且能独立成立的外层操作、对象和数量" in prompt
        assert "不得省略明确断言的事件、明确转述或一般规律" in prompt

    def test_recoverable_patient_is_preserved_when_splitting(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "省略的受事在本次输入中明确且唯一" in prompt
        assert "不能用目的地或体系名称替代受事" in prompt
        assert "全部应修饰手工证书" in prompt

    def test_admission_precedes_structural_success_and_separates_norms(self):
        prompt = NOESIS_CANONICAL_PROMPT
        assert "描述性规律与规范性要求不是同一类" in prompt
        assert "不能因为能构造合法 atoms/tree 而获得准入资格" in prompt
        assert "普通明确状态不需要带“已经”" in prompt
        assert "结论事件的核心谓词作为根" in prompt

    def test_complete_admission_and_split_contrasts_validate(self):
        rendered = ChatPromptTemplate.from_template(NOESIS_CANONICAL_PROMPT).format(
            source_text="User: 验证输入。", retry_feedback="无。"
        )
        contrasts = rendered.split("## 准入完整对照", 1)[1].split(
            "## 关系闭包完整对照", 1
        )[0]
        outputs = []
        for section in contrasts.split("输入：")[1:]:
            source = section.split("输出：", 1)[0].split("判定边界：", 1)[0].strip()
            raw, _ = json.JSONDecoder().raw_decode(section.split("输出：", 1)[1].lstrip())
            checked = validate_components(raw, source_text=source)
            assert not checked.alerts
            assert [component.model_dump(mode="json") for component in checked.components] == raw
            outputs.append(raw)
        assert [len(raw) for raw in outputs] == [0, 0, 1, 1, 1, 0, 0, 0, 1]
        assert outputs[3][0]["tree"]["nested"][0]["predicate"] == "超过"
        rule = outputs[4][0]
        assert rule["tree"]["predicate"] == rule["rule_template"]["conclusion"]["predicate"]
        assert rule["tree"]["nested"][0]["patient"][0]["text"] == "连接池"
        assert outputs[-1][0]["tree"]["predicate"] == "违约"
        assert outputs[-1][0]["rule_template"]["conclusion"]["predicate"] == "违约"

    def test_relation_closure_examples_validate_without_dropping(self):
        rendered = NOESIS_CANONICAL_PROMPT.format(
            source_text="验证输入", retry_feedback="无。"
        )
        sections = rendered.split("## 重点反例", 1)[1].split("## 闭包优先", 1)[0]
        sections += rendered.split("## 关系闭包完整对照", 1)[1].split("## 权威示例", 1)[
            0
        ]
        outputs = []
        for section in sections.split("输入：")[1:]:
            source = section.split("\n", 1)[0].strip()
            marker = "正确输出：" if "正确输出：" in section else "输出："
            raw, _ = json.JSONDecoder().raw_decode(section.split(marker, 1)[1].lstrip())
            checked = validate_components(raw, source_text=source)
            assert not checked.alerts
            assert not checked.retry_needed
            assert [c.model_dump(mode="json") for c in checked.components] == raw
            outputs.append(raw)
        assert [len(raw) for raw in outputs] == [1, 1, 1, 1, 2]
        assert outputs[0][0]["tree"]["predicate"] == "表明"
        assert outputs[1][0]["utterance_type"] == "hypothesis"

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
