"""Clause critic PoC: advisory semantic review plus targeted retry feedback."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from hyperextract.noesis import (
    ClauseCriticDecision,
    ClauseCriticResult,
    collect_clause_critic_candidates,
    create_clause_critic,
    extract_noesis_components,
)
from hyperextract.noesis.prompt import NOESIS_CANONICAL_PROMPT


def atom(pos, text, type_, role, target_occ, resolved=None):
    return {
        "pos": pos,
        "text": text,
        "type": type_,
        "role": role,
        "target_occ": target_occ,
        "resolved": resolved,
    }


SOURCE = "把「TTL 禁止整点对齐」写进缓存规范。"


def leaked_fact():
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "把「TTL 禁止整点对齐」写进缓存规范", "E", "modifier", 2),
            atom(2, "写进", "P", "predicate", None),
            atom(3, "缓存规范", "E", "patient", 2),
        ],
        "tree": {
            "predicate": "写进",
            "agent": [],
            "patient": [{"text": "缓存规范", "modifier": [], "implied": False}],
            "modifier": ["把「TTL 禁止整点对齐」写进缓存规范"],
            "nested": [],
            "conditional": [],
        },
    }


def repaired_fact():
    return {
        "utterance_type": "fact",
        "atoms": [
            atom(1, "「TTL 禁止整点对齐」", "E", "patient", 2),
            atom(2, "写进", "P", "predicate", None),
            atom(3, "缓存规范", "E", "modifier", 2),
        ],
        "tree": {
            "predicate": "写进",
            "agent": [],
            "patient": [
                {"text": "「TTL 禁止整点对齐」", "modifier": [], "implied": False}
            ],
            "modifier": ["缓存规范"],
            "nested": [],
            "conditional": [],
        },
    }


class FakeExtractOnce:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


class RecordingChatModel(BaseChatModel):
    responses: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    call_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-clause-critic"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))]
        )


def leakage_result():
    return ClauseCriticResult(
        decisions=[
            ClauseCriticDecision(
                component_index=0,
                atom_pos=1,
                classification="CLAUSE_LEAKAGE",
                missing_predicates=["写进"],
            )
        ]
    )


def test_candidate_prefilter_is_broad_but_does_not_decide_semantics():
    from hyperextract.noesis.validation import validate_components

    result = validate_components([leaked_fact()], source_text=SOURCE)
    candidates = collect_clause_critic_candidates(result.components)

    assert [(c.atom_pos, c.role) for c in candidates] == [(1, "modifier")]


def test_critic_uses_strict_json_schema_and_source_context():
    from hyperextract.noesis.validation import validate_components

    validated = validate_components([leaked_fact()], source_text=SOURCE)
    payload = {"leak": True, "predicates": ["写进"]}
    llm = RecordingChatModel(responses=[json.dumps(payload, ensure_ascii=False)])
    critic = create_clause_critic(llm_client=llm)

    result = critic(SOURCE, validated.components)

    assert result.decisions[0].classification == "CLAUSE_LEAKAGE"
    assert result.decisions[0].missing_predicates == ["写进"]
    rendered = "\n".join(str(message.content) for message in llm.calls[0])
    assert SOURCE in rendered
    assert "把「TTL 禁止整点对齐」写进缓存规范" in rendered
    response_format = llm.call_kwargs[0]["response_format"]
    assert response_format["json_schema"]["name"] == "noesis_clause_critic"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"leak", "predicates"}


def test_env_flag_enables_bound_critic_without_hindsight_api_changes(monkeypatch):
    fake = FakeExtractOnce([[leaked_fact()], [repaired_fact()]])
    calls = []

    def bound_critic(source, components):
        calls.append(components)
        return leakage_result() if len(calls) == 1 else ClauseCriticResult(decisions=[])

    fake._noesis_clause_critic = bound_critic
    monkeypatch.setenv("HYPEREXTRACT_NOESIS_CLAUSE_CRITIC", "true")

    outcome = extract_noesis_components(SOURCE, extract_once=fake)

    assert outcome.attempts == 2
    assert len(calls) == 2
    assert outcome.components[0].model_dump() == repaired_fact()


def test_env_flag_is_off_by_default(monkeypatch):
    fake = FakeExtractOnce([[leaked_fact()]])
    fake._noesis_clause_critic = lambda source, components: leakage_result()
    monkeypatch.delenv("HYPEREXTRACT_NOESIS_CLAUSE_CRITIC", raising=False)

    outcome = extract_noesis_components(SOURCE, extract_once=fake)

    assert outcome.attempts == 1
    assert outcome.components[0].model_dump() == leaked_fact()


def test_critic_hit_retries_with_targeted_feedback_and_does_not_rewrite_atoms():
    fake = FakeExtractOnce([[leaked_fact()], [repaired_fact()]])
    critic_calls = []

    def critic(source, components):
        critic_calls.append(components)
        return (
            leakage_result()
            if len(critic_calls) == 1
            else ClauseCriticResult(decisions=[])
        )

    outcome = extract_noesis_components(SOURCE, extract_once=fake, critic_once=critic)

    assert outcome.attempts == 2
    assert len(fake.calls) == 2
    assert fake.calls[0] == (SOURCE, {})
    assert "retry_feedback" in fake.calls[1][1]
    assert "Entity Atom 从句泄漏" in fake.calls[1][1]["retry_feedback"]
    assert "写进" in fake.calls[1][1]["retry_feedback"]
    assert outcome.components[0].model_dump() == repaired_fact()
    assert [a.alert_code for a in outcome.alerts] == ["clause_critic_retry"]


def test_second_critic_hit_warns_without_third_call_or_local_repair():
    fake = FakeExtractOnce([[leaked_fact()], [leaked_fact()]])

    outcome = extract_noesis_components(
        SOURCE,
        extract_once=fake,
        critic_once=lambda source, components: leakage_result(),
    )

    assert len(fake.calls) == 2
    assert outcome.components[0].model_dump() == leaked_fact()
    assert [a.alert_code for a in outcome.alerts] == [
        "clause_critic_retry",
        "clause_leakage_remaining",
    ]


def test_critic_failure_keeps_validated_result_instead_of_failing_extraction():
    fake = FakeExtractOnce([[leaked_fact()]])

    def broken_critic(source, components):
        raise RuntimeError("critic unavailable")

    outcome = extract_noesis_components(
        SOURCE,
        extract_once=fake,
        critic_once=broken_critic,
    )

    assert outcome.attempts == 1
    assert outcome.components[0].model_dump() == leaked_fact()
    assert [a.alert_code for a in outcome.alerts] == ["clause_critic_failed"]


def test_prompt_keeps_retry_feedback_without_global_clause_contrasts():
    assert "局部粒度对照" not in NOESIS_CANONICAL_PROMPT
    assert "把规则写进缓存规范" not in NOESIS_CANONICAL_PROMPT
    assert "cannot evict pod as it would violate PDB" not in NOESIS_CANONICAL_PROMPT
    assert "服务注册接入规范" not in NOESIS_CANONICAL_PROMPT
    assert "{retry_feedback}" in NOESIS_CANONICAL_PROMPT
    assert NOESIS_CANONICAL_PROMPT.count("### 示例 ") == 3
