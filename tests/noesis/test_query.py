"""Query-rewrite stage shares the summarizer's call protocol and retry contract."""

import json

import pytest

from tests.noesis.test_summary import SummaryLLM


def test_rewriter_uses_query_prompt_and_summary_protocol():
    from hyperextract.noesis import create_noesis_query_rewriter
    from hyperextract.noesis.query import QUERY_REWRITE_PROMPT

    client = SummaryLLM(response='{"text":"连接池被打满"}')
    rewrite = create_noesis_query_rewriter(llm_client=client)
    assert rewrite("为啥连接池被打满？") == "连接池被打满"
    assert client.messages[0][0].content == QUERY_REWRITE_PROMPT
    assert json.loads(client.messages[0][1].content) == {"content": "为啥连接池被打满？"}
    assert client.parameters[0]["reasoning_effort"] == "high"


def test_summarizer_default_prompt_unchanged():
    from hyperextract.noesis import create_noesis_summarizer
    from hyperextract.noesis.summary import SUMMARY_PROMPT

    client = SummaryLLM()
    create_noesis_summarizer(llm_client=client)("原文")
    assert client.messages[0][0].content == SUMMARY_PROMPT


def test_rewrite_retries_once_then_raises():
    from hyperextract.noesis import rewrite_noesis_query

    calls = []

    def rewrite_once(text):
        calls.append(text)
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        rewrite_noesis_query("你好", rewrite_once=rewrite_once)
    assert calls == ["你好", "你好"]


def test_empty_rewrite_is_success():
    from hyperextract.noesis import rewrite_noesis_query

    assert rewrite_noesis_query("你好", rewrite_once=lambda t: " ") == ""
