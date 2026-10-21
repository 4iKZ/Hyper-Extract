"""Lossless repair of LLM JSON: repair what is provably extra, raise everything else."""

import json
from pathlib import Path

import pytest

from hyperextract.noesis._llm_json import loads_llm_json

# Real Aixapex DeepSeek-V4-Flash query-atoms replies (2026-10-21): an opening stub, then the full object.
GATEWAY_REPLIES = json.loads(
    (Path(__file__).parent / "fixtures" / "gateway_restart_replies.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("raw", GATEWAY_REPLIES)
def test_real_gateway_restart_replies_recover_to_the_restarted_object(raw):
    value = loads_llm_json(raw)
    assert set(value) == {"atoms"} and value["atoms"]
    # Nothing inside the restarted object was edited: it is a verbatim suffix of the reply.
    assert any(_parses(raw[i:]) and json.loads(raw[i:]) == value for i in range(1, 65))


def _parses(text):
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"atoms": []}', {"atoms": []}),
        ("[1, 2]", [1, 2]),
        ('"text"', "text"),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('<think>reasoning</think>{"a": 1}', {"a": 1}),
        ('结果如下：{"a": 1} 完毕', {"a": 1}),
        ('{{"a": 1}', {"a": 1}),
        ('{"a": 1}}', {"a": 1}),
        ('[{"utt[{"utterance_type": "fact"}]', [{"utterance_type": "fact"}]),
        ('{"a": [1, 2,], "b": "x,]",}', {"a": [1, 2], "b": "x,]"}),
        ('{"a": "say \\"hi\\",", }', {"a": 'say "hi",'}),
        ('{"a": "line1\nline2"}', {"a": "line1\nline2"}),
        ('```json\n{"at{"atoms": []}\n```', {"atoms": []}),
    ],
)
def test_lossless_repairs(raw, expected):
    assert loads_llm_json(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '{"atoms": [{"text": "a"',  # truncated: never completed
        '{"x": 1{"atoms": []}',  # the dropped part is not a stub of what follows
        '{"atoms": [{"te"xt": "a"}]}',  # damage inside the object
        "",
        "plain text",
    ],
)
def test_unrecoverable_replies_raise_the_original_error(raw):
    with pytest.raises(json.JSONDecodeError) as caught:
        loads_llm_json(raw)
    with pytest.raises(json.JSONDecodeError) as strict:
        json.loads(raw)
    assert (caught.value.msg, caught.value.pos) == (strict.value.msg, strict.value.pos)
