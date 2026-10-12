"""Query-atoms stage: prompt examples, call protocol, validation and retry contract."""

import json
import re

import pytest

from tests.noesis.test_summary import SummaryLLM


def _examples():
    from hyperextract.noesis.query_prompt import QUERY_ATOMS_PROMPT

    pairs = re.findall(r"输入：(\{.*?\})\n输出：(\{.*\})", QUERY_ATOMS_PROMPT)
    return [(json.loads(i)["query"], json.loads(o)) for i, o in pairs]


def test_prompt_has_examples():
    assert len(_examples()) == 8


@pytest.mark.parametrize("query,expected", _examples())
def test_every_prompt_example_passes_the_validator(query, expected):
    from hyperextract.noesis import extract_noesis_query_atoms

    atoms = extract_noesis_query_atoms(query, extract_once=lambda t: expected)
    assert atoms == expected["atoms"]


def test_extractor_wraps_query_as_json_and_sets_schema():
    from hyperextract.noesis import create_noesis_query_extractor
    from hyperextract.noesis.query_prompt import QUERY_ATOMS_PROMPT

    client = SummaryLLM(response='{"atoms":[]}')
    extract = create_noesis_query_extractor(llm_client=client)
    assert extract('为啥"连接池"满了') == {"atoms": []}
    assert client.messages[0][0].content == QUERY_ATOMS_PROMPT
    assert json.loads(client.messages[0][1].content) == {"query": '为啥"连接池"满了'}
    assert client.parameters[0]["response_format"]["type"] == "json_schema"
    assert client.parameters[0]["max_tokens"] == 32768


def test_json_mode_and_truncation():
    from hyperextract.noesis import create_noesis_query_extractor

    client = SummaryLLM(response='{"atoms":[]}')
    create_noesis_query_extractor(llm_client=client, json_mode=True)("x")
    assert client.parameters[0]["response_format"] == {"type": "json_object"}
    with pytest.raises(ValueError):
        create_noesis_query_extractor(llm_client=SummaryLLM(finish_reason="length"))("x")


def test_filters_atoms_not_in_input_or_type_role_mismatch():
    from hyperextract.noesis import extract_noesis_query_atoms

    raw = {
        "atoms": [
            {"text": "连接池", "type": "E", "role": "patient"},
            {"text": "凭空出现", "type": "E", "role": "modifier"},
            {"text": "满了", "type": "E", "role": "predicate"},
            {"text": "连接池", "type": "P", "role": "agent"},
        ]
    }
    assert extract_noesis_query_atoms("连接池满了", extract_once=lambda t: raw) == [
        {"text": "连接池", "type": "E", "role": "patient"}
    ]


def test_empty_atoms_is_a_valid_answer_without_retry():
    from hyperextract.noesis import extract_noesis_query_atoms

    calls = []
    assert extract_noesis_query_atoms("你好", extract_once=lambda t: calls.append(t) or {"atoms": []}) == []
    assert calls == ["你好"]


@pytest.mark.parametrize(
    "raw",
    [[], {"atoms": [{"text": "a", "type": "G", "role": "agent"}]}, {"atoms": [], "x": 1}, "text"],
)
def test_invalid_shape_retries_once_then_raises(raw):
    from hyperextract.noesis import extract_noesis_query_atoms

    calls = []
    with pytest.raises(Exception):
        extract_noesis_query_atoms("a", extract_once=lambda t: calls.append(t) or raw)
    assert len(calls) == 2


def test_all_atoms_invalid_counts_as_failure():
    from hyperextract.noesis import extract_noesis_query_atoms

    raw = {"atoms": [{"text": "凭空出现", "type": "E", "role": "modifier"}]}
    with pytest.raises(ValueError):
        extract_noesis_query_atoms("连接池满了", extract_once=lambda t: raw)


def test_call_failure_retries_once_then_succeeds():
    from hyperextract.noesis import extract_noesis_query_atoms

    calls = []

    def flaky(text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return {"atoms": [{"text": "连接池", "type": "E", "role": "patient"}]}

    assert extract_noesis_query_atoms("连接池满了", extract_once=flaky) == [
        {"text": "连接池", "type": "E", "role": "patient"}
    ]
    assert len(calls) == 2
