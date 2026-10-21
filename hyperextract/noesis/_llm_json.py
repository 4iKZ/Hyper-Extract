"""Lossless parse of the JSON a Noesis LLM call returns (an object, or the extractor's root array).

Strict ``json.loads`` first. Only when that fails, try a fixed ladder of
transformations that delete characters which are provably extra (fences, a
think block, surrounding prose, a restarted opening stub, stray closing
brackets, trailing commas) or accept raw control characters inside strings.
Nothing is guessed or completed: truncation and damage inside the object still
raise the original error so the caller's single retry stays the fallback.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# The gateway sometimes emits the first few characters of the value, aborts, and
# restarts it in full: '{"at{"atoms": [...]}'. The stub is never longer than this.
_STUB_MAX_CHARS = 64
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_DECODER = json.JSONDecoder()


def _squash(text: str) -> str:
    return "".join(text.split())


def _strip_fence(text: str) -> str | None:
    match = _FENCE.match(text)
    return match.group(1) if match else None


def _strip_think(text: str) -> str | None:
    match = _THINK.match(text)
    return text[match.end() :] if match else None


def _outer_span(text: str) -> str | None:
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i >= 0]
    if not starts or not ends:
        return None
    start, end = min(starts), max(ends)
    if end <= start or text[start : end + 1] == text:
        return None
    return text[start : end + 1]


def _drop_restart_stub(text: str) -> str | None:
    """Drop an opening stub only when what follows begins with that same stub."""
    body = text.lstrip()
    for i in range(1, min(len(body), _STUB_MAX_CHARS + 1)):
        if body[i] not in "{[":
            continue
        rest = body[i:]
        if not _squash(rest).startswith(_squash(body[:i])):
            continue
        try:
            json.loads(rest, strict=False)
        except ValueError:
            continue
        return rest
    return None


def _drop_trailing_closers(text: str) -> str | None:
    """Keep the first complete value when only stray '}' / ']' follow it."""
    body = text.strip()
    try:
        _, end = _DECODER.raw_decode(body)
    except ValueError:
        return None
    tail = body[end:]
    return body[:end] if tail and not tail.strip("}] \t\r\n") else None


def _drop_trailing_commas(text: str) -> str | None:
    """Remove commas that sit, outside strings, right before a closing bracket."""
    out: list[str] = []
    in_string = escaped = changed = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        elif ch in "}]":
            j = len(out) - 1
            while j >= 0 and out[j] in " \t\r\n":
                j -= 1
            if j >= 0 and out[j] == ",":
                del out[j]
                changed = True
        out.append(ch)
    return "".join(out) if changed else None


# Applied in order, each on the latest candidate, so independent defects stack.
_STEPS = (
    ("fence", _strip_fence),
    ("think", _strip_think),
    ("prose", _outer_span),
    ("restart_stub", _drop_restart_stub),
    ("trailing_closers", _drop_trailing_closers),
    ("trailing_commas", _drop_trailing_commas),
)


def loads_llm_json(content: str) -> object:
    """``json.loads`` for an LLM reply; on failure repair only what is provably extra.

    A strict parse is returned untouched. A repaired result must be an object or array,
    otherwise the original ``JSONDecodeError`` is raised for the caller's retry.
    """
    try:
        return json.loads(content)
    except ValueError as error:
        original = error

    candidate, applied = content, []
    for name, step in _STEPS:
        changed = step(candidate)
        if changed is None:
            continue
        candidate = changed
        applied.append(name)
        try:
            value = json.loads(candidate, strict=False)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            logger.warning("Recovered LLM JSON via %s (dropped %d chars)", "+".join(applied), len(content) - len(candidate))
            return value
    try:
        value = json.loads(content, strict=False)  # raw control characters inside strings only
    except ValueError:
        pass
    else:
        if isinstance(value, (dict, list)):
            logger.warning("Recovered LLM JSON via control_chars")
            return value
    pos = original.pos
    logger.warning(
        "Unrecoverable LLM JSON (%s at %d): %r ... %r",
        original.msg,
        pos,
        content[max(0, pos - 80) : pos + 80],
        content[-80:],
    )
    raise original
