"""Production Stage 1 prompt, JSON protocol and bounded failure retries."""

import hashlib
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class SummaryLLM(BaseChatModel):
    response: str = '{"text":"慢查询日志已达41G，成为磁盘负担。"}'
    finish_reason: str = "stop"
    messages: list = Field(default_factory=list)
    parameters: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "summary-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.messages.append(messages)
        self.parameters.append(kwargs)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self.response,
                        response_metadata={"finish_reason": self.finish_reason},
                    )
                )
            ]
        )


def test_production_prompt_matches_approved_version():
    from hyperextract.noesis.summary import SUMMARY_PROMPT

    assert hashlib.sha256(SUMMARY_PROMPT.encode()).hexdigest() == (
        "299ece48ba7f7bf10a9c25434b0c3f9dc3e98de6aadf8a5caf812be1d4f64fff"
    )


def test_production_summarizer_uses_approved_prompt_and_high_thinking():
    from hyperextract.noesis import create_noesis_summarizer
    from hyperextract.noesis.summary import SUMMARY_PROMPT

    client = SummaryLLM()
    summarize = create_noesis_summarizer(llm_client=client)
    assert summarize("User: 原始对话。") == "慢查询日志已达41G，成为磁盘负担。"
    assert client.messages[0][0].content == SUMMARY_PROMPT
    assert json.loads(client.messages[0][1].content) == {"content": "User: 原始对话。"}
    assert client.parameters[0] == {
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "high",
        "max_tokens": 131072,
        "timeout": 900,
    }


@pytest.mark.parametrize(
    "raw", ["[]", "null", '{"text":1}', '{"text":"x","events":[]}']
)
def test_rejects_invalid_summary_output(raw):
    from hyperextract.noesis import create_noesis_summarizer

    with pytest.raises(ValueError):
        create_noesis_summarizer(llm_client=SummaryLLM(response=raw))("原始对话")


def test_rejects_truncated_even_when_json_parses():
    from hyperextract.noesis import create_noesis_summarizer

    with pytest.raises(ValueError):
        create_noesis_summarizer(llm_client=SummaryLLM(finish_reason="length"))(
            "原始对话"
        )


def test_summary_call_failure_retries_once():
    from hyperextract.noesis import summarize_noesis_text

    calls = []

    def summarize(text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return "整理后的事实"

    assert summarize_noesis_text("原文", summarize_once=summarize) == "整理后的事实"
    assert calls == ["原文", "原文"]


def test_invalid_summary_retries_then_raises_without_raw_fallback():
    from hyperextract.noesis import summarize_noesis_text

    calls = []

    def summarize(text):
        calls.append(text)
        return {"text": "不是字符串"}

    with pytest.raises(TypeError):
        summarize_noesis_text("原文", summarize_once=summarize)
    assert calls == ["原文", "原文"]


def test_empty_summary_is_success_not_retry():
    from hyperextract.noesis import summarize_noesis_text

    calls = []

    def summarize(text):
        calls.append(text)
        return "  "

    assert summarize_noesis_text("只有建议", summarize_once=summarize) == ""
    assert calls == ["只有建议"]


def test_he_json_mode_does_not_require_server_json_schema():
    from hyperextract.noesis import create_noesis_extractor
    from hyperextract.noesis.prompt import NOESIS_CANONICAL_PROMPT

    client = SummaryLLM(response="[]")
    assert create_noesis_extractor(llm_client=client, json_mode=True)("整理文本") == []
    assert client.parameters[0] == {"response_format": {"type": "json_object"}}
    assert client.messages[0][0].content == NOESIS_CANONICAL_PROMPT.format(
        source_text="整理文本", retry_feedback="无。"
    )
