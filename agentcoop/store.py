"""Memoized expensive calls (first writer wins) and content-addressed directory blobs.

Site-agent proposals, repairs and coordinator decisions cost minutes and money, and several arms
must compare the *same* step-1 claims. :func:`memo` stores each result once per key:

* **One value per key.** An entry is published with an atomic hard link; a writer that loses
  the race returns the winner's value, so concurrent runs and arms converge on one result.
* **Nominal vs actual cost.** An entry carries what computing it cost (``nominal_usd``); a hit
  ledgers that amount as a $0-actual replay, so method comparisons do not depend on cache state.
* **Failures are not results.** An exception from ``compute`` propagates and nothing is stored;
  its spend stays in the ledger under the caller's tag, which the memo tag nests under.

Ported from the v3 node cache (``workflow.execute``, tag ``v3.4.1``). Directories (site views)
are stored as tar.gz under the archive's sha256, with member times and owners zeroed so the id
depends on content only.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
import threading
import time
from pathlib import Path
from typing import Callable

from agentcoop import llm

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "runs" / "store"


def _publish(path: Path, data: bytes) -> bytes:
    """Write ``data`` to ``path`` atomically; the first writer wins and everyone gets its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    except FileExistsError:
        data = path.read_bytes()
    finally:
        tmp.unlink()
    return data


def memo(key: dict, compute: Callable[[], dict], *, tag: str) -> dict:
    """``compute()`` once per ``key``; every later call (any thread, process or arm) replays it.

    ``key`` must name everything the result depends on, including the ``sample`` index (same
    index => exact replay, new index => honest re-draw). The result is returned in its stored
    JSON form, so a miss and a hit return equal dicts, plus ``nominal_usd`` = the ledgered
    nominal spend of ``compute`` + any ``nominal_usd_external`` it reports for usage outside the
    ledger (usage it must then not also ledger).
    """
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = STORE / "memo" / h[:2] / f"{h}.json"
    if path.exists():
        value = json.loads(path.read_text())["value"]
        llm.record_replay(llm.backbone(), value["nominal_usd"])
        return value
    with llm.tag(f"{tag}#{h[:8]}.{time.time_ns()}"):
        exec_tag = llm.current_tag()
        result = compute()
    nominal = llm.spent(exec_tag)["nominal_usd"] + result.get("nominal_usd_external", 0.0)
    entry = json.dumps({"key": key, "value": {**result, "nominal_usd": nominal}})
    return json.loads(_publish(path, entry.encode()))["value"]


def _blob_path(blob_id: str) -> Path:
    return STORE / "blobs" / blob_id[:2] / f"{blob_id}.tar.gz"


def _zeroed(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
    return info


def put_blob(src_dir: Path) -> str:
    """Store a directory as tar.gz and return the archive's sha256 (first writer wins)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for p in sorted(src_dir.rglob("*")):
            tar.add(p, arcname=p.relative_to(src_dir).as_posix(), recursive=False,
                    filter=_zeroed)
    data = buf.getvalue()
    blob_id = hashlib.sha256(data).hexdigest()
    _publish(_blob_path(blob_id), data)
    return blob_id


def get_blob(blob_id: str, dst_dir: Path) -> None:
    """Extract a stored directory into ``dst_dir`` (created). Archives hold agent-written files,
    so the ``data`` filter refuses absolute paths and links that leave ``dst_dir``."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(_blob_path(blob_id), mode="r:gz") as tar:
        tar.extractall(dst_dir, filter="data")
