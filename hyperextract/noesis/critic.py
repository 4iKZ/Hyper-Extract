"""Semantic critic for proposition-shaped Entity Atoms.

The critic never rewrites atoms. It identifies leakage for a targeted retry
and marks failed candidate reviews as UNCERTAIN so the extractor can reject
their components rather than silently accept unreviewed data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict

from .models import FactComponent, HypothesisComponent


class ClauseCriticCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_index: int
    atom_pos: int
    text: str
    role: Literal["agent", "patient", "modifier"]
    component_predicates: list[str]


class ClauseCriticDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_index: int
    atom_pos: int
    classification: Literal["KEEP", "CLAUSE_LEAKAGE", "UNCERTAIN"]
    missing_predicates: list[str]


class ClauseCriticVerdict(BaseModel):
    """Minimal model-facing response for exactly one candidate E atom."""

    model_config = ConfigDict(extra="forbid")

    leak: bool
    predicates: list[str]


class ClauseCriticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[ClauseCriticDecision]


CLAUSE_CRITIC_PROMPT = """你是 Noesis 的 Entity Atom 语义审查器。一次只审查一个候选 E atom。
判断它是否把本应继续处理的自然语言命题或从句整体包进了 E。

判定原则：
1. 不能按长度判断；长错误消息、SQL、配置项、DNS 名、路径、指标名都可以 KEEP。
2. 不能只因 E 中出现某个 predicate 的字符串就判错；“服务注册接入规范”仍可能是合法名词。
3. 使用 source 判断字面量边界。比如“日志里出现 cannot evict pod as it would violate PDB”时，整个错误字符串可以 KEEP，内部 evict/violate 不建立新事实。
4. 自然语言动作链不能作为一个 E。例如“实际把旧值写回 V1”中的“写回”是谓词；“取两边状态比较后写 V2 的值到 V1”包含“取/比较/写”等动作，不应整体作为 E。
5. 建议或操作步骤也不能为了填充论元而整体塞进 E；它们仍属于 leakage，后续抽取器会重新判断是否应输出事实。
6. 只有确定是实体、复合名词或技术字面量时才返回 leak=false；不确定时按 fail-closed 返回 leak=true。

只返回 {{"leak": boolean, "predicates": string[]}}。predicates 只填写候选 E 内逐字出现的最小谓词；leak=false 时必须为 []。不要解释。

source:
{source_text}

candidate:
{candidate_json}
"""


def collect_clause_critic_candidates(
    components: list[FactComponent | HypothesisComponent],
) -> list[ClauseCriticCandidate]:
    """Review every Entity Atom; length is not a semantic safety criterion."""

    candidates: list[ClauseCriticCandidate] = []
    for component_index, component in enumerate(components):
        predicates = [
            atom.text
            for atom in component.atoms
            if atom.type == "P" and atom.role == "predicate"
        ]
        for atom in component.atoms:
            if atom.type != "E" or atom.role not in {"agent", "patient", "modifier"}:
                continue
            candidates.append(
                ClauseCriticCandidate(
                    component_index=component_index,
                    atom_pos=atom.pos,
                    text=atom.text,
                    role=atom.role,
                    component_predicates=predicates,
                )
            )
    return candidates


def create_clause_critic(
    *,
    llm_client: BaseChatModel,
) -> Callable[[str, list[FactComponent | HypothesisComponent]], ClauseCriticResult]:
    """Create the optional semantic critic used after deterministic validation."""

    prompt = ChatPromptTemplate.from_template(CLAUSE_CRITIC_PROMPT)
    schema_client = llm_client.bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "noesis_clause_critic",
                "strict": True,
                "schema": ClauseCriticVerdict.model_json_schema(),
            },
        }
    )
    chain = prompt | schema_client

    def critique(
        source_text: str,
        components: list[FactComponent | HypothesisComponent],
    ) -> ClauseCriticResult:
        candidates = collect_clause_critic_candidates(components)
        if not candidates:
            return ClauseCriticResult(decisions=[])

        decisions = []
        for candidate in candidates:
            try:
                response = chain.invoke(
                    {
                        "source_text": source_text,
                        "candidate_json": json.dumps(
                            candidate.model_dump(),
                            ensure_ascii=False,
                        ),
                    }
                )
                content = (
                    response.text if isinstance(response, BaseMessage) else response
                )
                if not isinstance(content, str):
                    raise TypeError(
                        "Noesis clause critic response content must be text"
                    )
                verdict = ClauseCriticVerdict.model_validate_json(content)
            except Exception:
                decisions.append(
                    ClauseCriticDecision(
                        component_index=candidate.component_index,
                        atom_pos=candidate.atom_pos,
                        classification="UNCERTAIN",
                        missing_predicates=[],
                    )
                )
                continue
            decisions.append(
                ClauseCriticDecision(
                    component_index=candidate.component_index,
                    atom_pos=candidate.atom_pos,
                    classification=("CLAUSE_LEAKAGE" if verdict.leak else "KEEP"),
                    missing_predicates=verdict.predicates if verdict.leak else [],
                )
            )
        return ClauseCriticResult(decisions=decisions)

    return critique


def leakage_decisions(result: ClauseCriticResult) -> list[ClauseCriticDecision]:
    return [
        decision
        for decision in result.decisions
        if decision.classification == "CLAUSE_LEAKAGE"
    ]


def build_clause_retry_feedback(result: ClauseCriticResult) -> str:
    """Render narrow feedback for the canonical extractor; never rewrite atoms."""

    issues = leakage_decisions(result)
    if not issues:
        return ""

    lines = [
        (
            "上一次输出存在 Entity Atom 从句泄漏。请重新执行完整的谓词候选与断言边界判断；"
            "不要机械拆词，也不要把建议/指令改造成事实。"
        )
    ]
    for issue in issues:
        predicates = (
            "、".join(issue.missing_predicates)
            if issue.missing_predicates
            else "未确定"
        )
        lines.append(
            f"- component[{issue.component_index}] atom pos={issue.atom_pos}: "
            f"遗漏/被包裹谓词候选={predicates}"
        )
    return "\n".join(lines)
