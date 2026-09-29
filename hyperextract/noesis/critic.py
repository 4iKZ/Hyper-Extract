"""Semantic critic for proposition-shaped Entity Atoms.

The critic is deliberately advisory: it never rewrites atoms or drops a
component. It only identifies E atoms that appear to contain a proposition so
the canonical extractor can retry with targeted feedback.
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
    reason: str


class ClauseCriticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[ClauseCriticDecision]


CLAUSE_CRITIC_PROMPT = """你是 Noesis 的 Entity Atom 语义审查器。你不重写 atoms、不补 tree，也不判断事实真假。
你的任务只是在给定 source 与候选 E atom 后，判断这个 E 是否把一个本应继续处理的自然语言命题/从句整体包进了 Entity Atom。

分类只能是：
- KEEP：这是可独立指称的实体、复合名词、时间/数量/路径/配置名/指标名/代码/SQL/命令、或可明确定位的技术字面量。即使其中出现看起来像动词的英文词，也不能因此拆开。
- CLAUSE_LEAKAGE：该 E 自身包含一个自然语言动作、状态变化、判断或多个串行动作，当前抽取把命题内容整体塞进了 E。命中后，主抽取器需要重新检查：若这是 DIRECT/REPORTED 断言，应把真正的动作/状态建立为 P；若其实是问题、建议、指令、操作步骤或设想，应按 NONASSERTIVE 处理而不是制造事实。
- UNCERTAIN：上下文不足，无法可靠区分 KEEP 与 CLAUSE_LEAKAGE。

判定原则：
1. 不能按长度判断；长错误消息、SQL、配置项、DNS 名、路径、指标名都可以 KEEP。
2. 不能只因 E 中出现某个 predicate 的字符串就判错；“服务注册接入规范”仍可能是合法名词。
3. 使用 source 判断字面量边界。比如“日志里出现 cannot evict pod as it would violate PDB”时，整个错误字符串可以 KEEP，内部 evict/violate 不建立新事实。
4. 自然语言动作链不能作为一个 E。例如“实际把旧值写回 V1”中的“写回”是谓词；“取两边状态比较后写 V2 的值到 V1”包含“取/比较/写”等动作，不应整体作为 E。
5. 建议/操作步骤同样不能为了填充论元而整体塞进 E；例如“把规则写进缓存规范”若只是建议，应由主抽取器在 retry 时重新判为 NONASSERTIVE。

只返回严格 JSON 对象，逐个候选给出决定。missing_predicates 只填写候选 span 内明确遗漏的最小谓词字面；KEEP/UNCERTAIN 时必须为 []。

source:
{source_text}

candidates:
{candidates_json}
"""


def collect_clause_critic_candidates(
    components: list[FactComponent | HypothesisComponent],
    *,
    min_text_length: int = 8,
) -> list[ClauseCriticCandidate]:
    """Select broad candidates cheaply; the LLM critic makes the semantic call."""

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
            if len(atom.text.strip()) < min_text_length:
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
    min_text_length: int = 8,
) -> Callable[[str, list[FactComponent | HypothesisComponent]], ClauseCriticResult]:
    """Create the optional semantic critic used after deterministic validation."""

    prompt = ChatPromptTemplate.from_template(CLAUSE_CRITIC_PROMPT)
    schema_client = llm_client.bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "noesis_clause_critic",
                "strict": True,
                "schema": ClauseCriticResult.model_json_schema(),
            },
        }
    )
    chain = prompt | schema_client

    def critique(
        source_text: str,
        components: list[FactComponent | HypothesisComponent],
    ) -> ClauseCriticResult:
        candidates = collect_clause_critic_candidates(
            components,
            min_text_length=min_text_length,
        )
        if not candidates:
            return ClauseCriticResult(decisions=[])

        response = chain.invoke(
            {
                "source_text": source_text,
                "candidates_json": json.dumps(
                    [candidate.model_dump() for candidate in candidates],
                    ensure_ascii=False,
                ),
            }
        )
        content = response.text if isinstance(response, BaseMessage) else response
        if not isinstance(content, str):
            raise TypeError("Noesis clause critic response content must be text")

        result = ClauseCriticResult.model_validate_json(content)
        allowed = {
            (candidate.component_index, candidate.atom_pos) for candidate in candidates
        }
        decisions = [
            decision
            for decision in result.decisions
            if (decision.component_index, decision.atom_pos) in allowed
        ]
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
        "上一次输出存在 Entity Atom 从句泄漏。请重新执行完整的谓词候选与断言边界判断；"
        "不要机械拆词，也不要把建议/指令改造成事实。"
    ]
    for issue in issues:
        predicates = (
            "、".join(issue.missing_predicates)
            if issue.missing_predicates
            else "未确定"
        )
        lines.append(
            f"- component[{issue.component_index}] atom pos={issue.atom_pos}: "
            f"遗漏/被包裹谓词候选={predicates}；{issue.reason}"
        )
    return "\n".join(lines)
