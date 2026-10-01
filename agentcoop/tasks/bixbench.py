"""BixBench v1.5 (FutureHouse; 205 open-answer questions over 59 published-analysis capsules).

    python -m agentcoop.tasks.bixbench          # after downloading BixBench.jsonl + capsule zips

Each question becomes a task ``data/tasks/bixbench/<question_id>`` with the same public/hidden
split as our own suites: ``public/task.json`` holds only the question and the capsule's data
directory; the ideal answer, distractors, eval mode, the paper's result and hypothesis go to
``hidden/truth.json``. Capsules are unpacked once (``data/bench/bixbench/capsules/<uuid>``) the
way BixBench's harness does it — data files only. The reference notebook (it contains the
answers) is moved to ``data/bench/bixbench/reference/<uuid>`` and is never shown to an agent.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "bench" / "bixbench"
TASKS = ROOT / "data" / "tasks"
SUITE = "bixbench"
HIDDEN_FIELDS = ("ideal", "distractors", "eval_mode", "result", "answer", "hypothesis", "paper",
                 "categories")


def unpack(zip_path: Path) -> Path:
    uuid = zip_path.stem.removeprefix("CapsuleFolder-")
    data_dir, ref_dir = BENCH / "capsules" / uuid, BENCH / "reference" / uuid
    if data_dir.exists():
        return data_dir
    tmp = BENCH / "capsules" / f".{uuid}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    ref_dir.mkdir(parents=True, exist_ok=True)
    data_src = None
    for p in sorted(tmp.rglob("*")):
        if p.is_dir() and "Notebook" in p.name:
            shutil.move(str(p), ref_dir / p.name)
        elif p.is_dir() and "Data" in p.name and data_src is None:
            data_src = p
    if data_src is None:
        raise FileNotFoundError(f"{zip_path.name}: no Data folder")
    for nb in list(data_src.rglob("*.ipynb")):  # notebooks may carry answers
        shutil.move(str(nb), ref_dir / nb.name)
    shutil.move(str(data_src), data_dir)
    shutil.rmtree(tmp)
    return data_dir


def build(jsonl: Path = BENCH / "BixBench.jsonl", out: Path = TASKS) -> list[Path]:
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    paths = []
    for row in rows:
        task_dir = out / SUITE / row["question_id"]
        if task_dir.exists():
            raise FileExistsError(f"{task_dir} exists; build a new suite instead")
    for row in rows:
        data_dir = unpack(BENCH / "zips" / row["data_folder"])
        task_dir = out / SUITE / row["question_id"]
        (task_dir / "public").mkdir(parents=True)
        (task_dir / "hidden").mkdir()
        (task_dir / "public" / "task.json").write_text(json.dumps({
            "task_id": row["question_id"], "question": row["question"], "shape": "bixbench",
            "capsule": row["capsule_uuid"], "data": str(data_dir)}, indent=2))
        (task_dir / "hidden" / "truth.json").write_text(json.dumps(
            {k: row[k] for k in HIDDEN_FIELDS}, indent=2))
        paths.append(task_dir)
    return paths



def split_of(capsule: str) -> str:
    """Pre-registered split by capsule (2026-10-01, before any BixBench run): ~25% of capsules
    (14 capsules / 39 questions) are ``bixdev`` for designing and debugging; the remaining 45
    capsules / 166 questions are ``bixtest``, run once with the frozen method."""
    import hashlib

    return "bixdev" if hashlib.sha256(f"agentcoop-bix-split:{capsule}".encode()).digest()[0] < 64 \
        else "bixtest"


def question_ids(split: str) -> list[str]:
    out = []
    for d in sorted((TASKS / SUITE).iterdir()):
        info = json.loads((d / "public" / "task.json").read_text())
        if split in ("all", split_of(info["capsule"])):
            out.append(info["task_id"])
    return sorted(out, key=lambda q: (int(q.split("-")[1]), int(q.split("-q")[1])))


if __name__ == "__main__":
    built = build(*(Path(a) for a in sys.argv[1:2]))
    print(f"{len(built)} tasks in {TASKS / SUITE}")
