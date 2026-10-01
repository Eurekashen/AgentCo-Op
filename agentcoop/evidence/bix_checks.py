"""Answer-free checks over BixBench analyses (no reference answer, no hidden data).

An ``analysis`` artifact carries an agent's short answer plus its executed trace (code and
output) and code files. These checks look only at that material and the question:

- ``same_answer`` — do two analyses state the same result? (numeric tolerance from the stated
  precision, else normalized text, else an LLM equivalence call that sees only the two answers
  and the question);
- ``provenance`` — is a numeric answer actually produced by the executed code (it appears in an
  output), or was it typed in without support?
"""

from __future__ import annotations

import math
import re
from typing import Any

from agentcoop import llm

_NUM = re.compile(r"(?<![\w.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers(text: str | None) -> list[tuple[float, int]]:
    """Numbers in ``text`` with their stated decimal places (for a precision-aware match)."""
    out = []
    for m in _NUM.finditer(text or ""):
        s = m.group(0).replace(",", "").replace("−", "-")
        if not re.search(r"\d", s):
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        mant = s.lower().split("e")[0]
        places = len(mant.split(".")[1]) if "." in mant else 0
        if "e" in s.lower():
            places = max(places - int(s.lower().split("e")[1]), 0)
        out.append((v, places))
    return out


def _close(a: float, b: float, places: int) -> bool:
    """``a`` (stated with ``places`` decimals) is ``b`` rounded, or within 1% of it."""
    if math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-12):
        return True
    return abs(a - b) <= 0.5 * 10 ** (-places) + 1e-12


def number_of(answer: str | None) -> tuple[float, int] | None:
    """The answer's value when it is essentially one number (e.g. '0.42', '42%', '1.2e-5')."""
    if not answer:
        return None
    nums = numbers(answer)
    if len(nums) != 1:
        return None
    stripped = _NUM.sub("", answer.replace(",", ""))
    if len(re.sub(r"[\s%~≈<>=a-zA-Z().:]", "", stripped)) > 3 or len(answer) > 60:
        return None
    return nums[0]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def same_answer(question: str, a: str | None, b: str | None) -> bool | None:
    """True/False, or None when either answer is missing."""
    if not a or not b:
        return None
    na, nb = number_of(a), number_of(b)
    if na and nb:
        return _close(na[0], nb[0], min(na[1], nb[1])) or _close(nb[0], na[0], min(na[1], nb[1]))
    if _norm(a) == _norm(b):
        return True
    r = llm.chat([{"role": "user", "content": (
        f"Question: {question}\n\nAnswer A: {a}\nAnswer B: {b}\n\nDo A and B state the same "
        "final result for this question (ignore wording, formatting and rounding to the stated "
        "precision)? Reply with exactly one word: same or different.")}],
        reasoning_effort="low")
    return r.content.strip().lower().startswith("same")


def provenance(analysis: dict[str, Any]) -> str:
    """'supported' | 'unsupported' | 'unassessed' — does an executed output contain the answer?"""
    num = number_of(analysis.get("answer"))
    if num is None:
        return "unassessed"
    value, places = num
    pct = value / 100.0
    for step in analysis.get("trace", []):
        for v, _ in numbers(step.get("output", "")):
            if _close(value, v, places) or _close(pct, v, places + 2):
                return "supported"
    return "unsupported"
