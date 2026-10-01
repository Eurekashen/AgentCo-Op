"""BixBench open-answer grading (evaluation harness only: reads ``hidden/truth.json``).

Follows BixBench's own open-answer grader (``bixbench/graders.py``, ``OpenEndedGrader`` with
LLM matching): ``str_verifier`` = normalized exact match, else the LLM equivalence prompt;
``llm_verifier`` = the LLM prompt; ``range_verifier`` = the LLM range prompt. Two deviations:
BixBench's "partial match" (normalized prediction is a *substring* of the target) is dropped —
it accepts "2" for "0.0002"; and a range answer that parses as a single number is checked
numerically (the LLM is only asked when it does not parse). The grading model
is the pinned instrument (``llm.grader_model()``).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from agentcoop import llm

GRADING = """You are given a question, target answer and a predicted answer. Your task is to compare the target answer with the predicted and assess if the predicted answer is correct, incorrect or it refused to answer.
Question: {question}
Target Answer: {target}
Predicted Answer: {predicted}

Important: You must only output one from `correct`, `incorrect` or `refused` between <grade> tags.
Example Output: <grade> correct </grade>
"""

RANGE_GRADING = """You are given a question, target range using the format (lower,upper) and a predicted answer. Your task is to compare the target range with the predicted and assess if the predicted answer falls within the specified range. If it falls within the range, it is correct, otherwise it is incorrect. If the predicted answer cannot be compared to the target range, it is refused.
Question: {question}
Target Range: {target}
Predicted Answer: {predicted}

Important: You must only output one from `correct`, `incorrect` or `refused` between <grade> tags.
Example Output: <grade> correct </grade>
"""


MCQ = """A question was answered by an analysis of the data. Map the proposed answer onto the multiple-choice options.

Question: {question}

Proposed answer: {proposed}

Options:
{options}

Select the option that the proposed answer states (allowing for rounding and wording). If the proposed answer does not correspond to any option, or does not commit to an answer, select the "Insufficient information" option.
Reply with only the letter, in XML: <answer>X</answer>"""

REFUSAL = "Insufficient information to answer the question"


def load_truth(task_dir: Path) -> dict[str, Any]:
    return json.loads((Path(task_dir) / "hidden" / "truth.json").read_text())


def _llm_grade(template: str, question: str, target: str, predicted: str) -> bool:
    r = llm.chat([{"role": "user", "content": template.format(
        question=question, target=target, predicted=predicted)}],
        reasoning_effort="low", model=llm.grader_model())
    m = re.search(r"<grade>\s*(.*?)\s*</grade>", r.content, re.DOTALL)
    return bool(m) and m.group(1).strip().lower() == "correct"


def _number(text: str) -> float | None:
    s = text.strip().replace(",", "").replace("−", "-").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def grade(question: str, truth: dict[str, Any], predicted: str | None) -> bool:
    if not predicted or not str(predicted).strip():
        return False
    predicted, target, mode = str(predicted), str(truth["ideal"]), truth["eval_mode"]
    if mode == "str_verifier":
        clean_t = re.sub(r"[^a-zA-Z0-9]", "", target).lower()
        clean_p = re.sub(r"[^a-zA-Z0-9]", "", predicted).lower()
        if clean_p == clean_t:
            return True
        return _llm_grade(GRADING, question, target, predicted)
    if mode == "range_verifier":
        x = _number(predicted)
        if x is not None:
            lo, hi = ast.literal_eval(target)
            return float(lo) <= x <= float(hi)
        return _llm_grade(RANGE_GRADING, question, target, predicted)
    if mode == "llm_verifier":
        return _llm_grade(GRADING, question, target, predicted)
    raise ValueError(f"unknown eval_mode {mode!r}")


def grade_mcq(question: str, truth: dict[str, Any], predicted: str | None,
              qid: str = "") -> bool:
    """BixBench's MCQ-with-refusal protocol from an open answer: the pinned grader maps the
    proposed answer onto the shuffled options (ideal, the three distractors, refusal); correct
    iff it maps onto the ideal. Random choice scores 0.2. Only the answer is shown (not a
    notebook), so single agents and teams are mapped identically."""
    import random

    if not predicted or not str(predicted).strip():
        return False
    options = [str(truth["ideal"]), *map(str, truth["distractors"]), REFUSAL]
    random.Random(f"bixmcq:{qid or question}").shuffle(options)
    letters = "ABCDE"
    r = llm.chat([{"role": "user", "content": MCQ.format(
        question=question, proposed=predicted,
        options="\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options)))}],
        reasoning_effort="low", model=llm.grader_model())
    m = re.search(r"<answer>\s*([A-E])\s*</answer>", r.content)
    return bool(m) and options[letters.index(m.group(1))] == str(truth["ideal"])
