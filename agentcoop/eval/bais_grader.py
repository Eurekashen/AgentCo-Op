"""BAIS-DPTA grading (evaluation harness only: reads ``hidden/labels.json``).

Reproduces the benchmark's metric (BaisBench ``Task1.ipynb``, repo cfab1be): predicted label
names are first mapped by an LLM onto the organ's uHAF reference cell types (the benchmark's
prompt; ``gpt-4.1`` there, the pinned grading model here), then every expert-labelled cell
scores 1 if the prediction is its label or a descendant of it, 0.5 for its parent, 0.2 for its
grandparent (unless that is the root ``Cell``), else 0; cells labelled ``Unclassified`` are
skipped. The score is the mean over cells. uHAF 2.2.0 (the version bundled with ``uhaf``).
"""

from __future__ import annotations

import ast
import functools
import json
import re
from pathlib import Path
from typing import Any

from agentcoop import llm

MAP_PROMPT = (
    "Given the reference cell types: {reference}, "
    "please map the user input cell types: {user_input} "
    "to the reference cell types (find the most related one and map to it). Return the result "
    "as a dictionary. For example, if user input is 'naive t' and reference includes 'Naive T "
    "cell', the output should be something like {{'naive t': 'Naive T cell'}}. "
    "if user input is 'Intestinal epithelial stem cell' and reference includes 'Epithelial "
    "cell',the output should be something like {{'Intestinal epithelial stem cell': "
    "'Epithelial cell'}}. Note that the item in the dictionary should be strictly included in "
    "the reference cell types and the user input cell types,do not invent anything by yourself. "
    "If there is organ name (such as Intestinal) in the user input cell type, just ignore it."
    "If there is unknown cell type, labeled it as Unclassified."
    "Please output the result strictly as a valid Python dictionary without any additional "
    "explanation, formatting, or markdown code block."
)


@functools.lru_cache(maxsize=1)
def _uhaf():
    import uhaf

    return uhaf.build_uhaf(latest=True)


def tree(organ: str):
    return _uhaf().df_uhafs[organ.replace("_", " ")]


def load_labels(task_dir: Path) -> dict[str, Any]:
    return json.loads((Path(task_dir) / "hidden" / "labels.json").read_text())


def map_names(organ: str, names: list[str]) -> dict[str, str]:
    reference = list(tree(organ).index.values)
    r = llm.chat([{"role": "user", "content": MAP_PROMPT.format(
        reference=", ".join(reference), user_input=sorted(set(names)))}],
        reasoning_effort="low", model=llm.grader_model())
    text = re.sub(r"^```.*\n|\n```$", "", r.content.strip())
    try:
        mapping = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        mapping = {}
    return {str(k): str(v) for k, v in mapping.items() if str(v) in reference} \
        if isinstance(mapping, dict) else {}


def _descendants(t, node: str) -> set[str]:
    out, queue = set(), [node]
    while queue:
        cur = queue.pop()
        for child in t[t["father"] == cur]["child"].tolist():
            if child not in out:
                out.add(child)
                queue.append(child)
    return out


def score(organ: str, truth: list[str], predicted: list[str | None]) -> dict[str, Any]:
    """Mean hierarchical score over expert-labelled cells; per-label breakdown included."""
    if len(truth) != len(predicted):
        raise ValueError(f"{len(predicted)} predictions for {len(truth)} cells")
    t = tree(organ)
    names = sorted({p for p in predicted if p})
    mapping = map_names(organ, names) if names else {}
    mapped = [mapping.get(p, "unclassified") if p else "unclassified" for p in predicted]
    credit: dict[tuple[str, str], float] = {}
    total, n = 0.0, 0
    per_label: dict[str, list[float]] = {}
    for y, p in zip(truth, mapped):
        if y == "Unclassified":
            continue
        key = (y, p)
        if key not in credit:
            father = t.loc[y]["father"] if y in t.index else None
            grand = t.loc[father]["father"] if father in t.index else None
            if p == y or p in _descendants(t, y):
                credit[key] = 1.0
            elif p == father:
                credit[key] = 0.5
            elif father != "Cell" and grand is not None and p == grand:
                credit[key] = 0.2
            else:
                credit[key] = 0.0
        total += credit[key]
        n += 1
        per_label.setdefault(y, []).append(credit[key])
    return {"score": total / n if n else 0.0, "n_cells": n, "mapping": mapping,
            "per_label": {k: sum(v) / len(v) for k, v in per_label.items()}}
