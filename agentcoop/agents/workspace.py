"""Private, OS-sandboxed workspaces for site agents (spec §8).

A site agent works in its own directory under ``WS_ROOT`` (outside the repo) holding APFS clones
of its inputs, so its writes never reach the originals. Every process it starts runs under macOS
``sandbox-exec`` with a profile that denies the repo's ``data/``, ``runs/`` and ``.secrets/``, the
Codex login and every other workspace, and all network except loopback. Loopback stays open in
both directions: Finch reaches the model through the local proxy, and its Jupyter kernel binds
loopback ports (without ``network-bind`` the kernel cannot start and every Finch site would fail
for a reason that has nothing to do with science). Codex calls its own API, so it is confined by
Codex's permission profile with the same deny list instead (``agents/codex.py``).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
from pathlib import Path

from agentcoop.llm import ROOT

WS_ROOT = Path(os.environ.get("AGENTCOOP_WS_ROOT", Path.home() / "agentcoop_ws"))
CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
DENY: tuple[Path, ...] = (ROOT / "data", ROOT / "runs", ROOT / ".secrets", CODEX_HOME, WS_ROOT)
#: The site agents' analysis interpreter: a read-only conda env (scanpy, anndata, pydeseq2, R).
BIXENV = Path(os.environ.get("AGENTCOOP_BIXBENCH_ENV",
                             Path.home() / "miniconda3" / "envs" / "bixbench"))
KERNELS = ROOT / "external_agents" / "bixbench_env" / "jupyter"


def new_workspace(name: str, files: dict[str, Path]) -> Path:
    """``WS_ROOT/name``, emptied, with ``files`` (name in the workspace -> source) cloned in.

    ``cp -c`` makes an APFS clone (instant, no extra disk); a plain copy where cloning is not
    available. A stale workspace is removed first, so a run never sees an earlier run's files.
    """
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"workspace name must stay under WS_ROOT: {name!r}")
    ws = WS_ROOT / name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for rel, src in files.items():
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode:
            shutil.copy2(src, dst)
    return ws


def exec_env(ws: Path) -> dict[str, str]:
    """Environment of every site-agent process: no API key (Codex must use its local login) and
    no AGENTCOOP_* settings (agents never see the experiment's configuration); the analysis env
    first on PATH; HOME, temp dirs and package caches inside the workspace (scanpy needs writable
    numba/matplotlib caches; an agent's installs stay in its own workspace)."""
    env = {k: v for k, v in os.environ.items()
           if k != "OPENAI_API_KEY" and not k.startswith("AGENTCOOP_")}
    tmp = ws / ".tmp"
    env.update(PATH=f"{BIXENV / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}",
               HOME=str(ws / ".home"), TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp / "mpl"),
               NUMBA_CACHE_DIR=str(tmp / "numba"), JUPYTER_PATH=str(KERNELS),
               PYTHONUSERBASE=str(tmp / "pyuser"), R_LIBS_USER=str(tmp / "Rlib"))
    for d in (ws / ".home", tmp / "mpl", tmp / "numba", tmp / "Rlib",
              tmp / "pyuser" / "lib" / "python3.12" / "site-packages"):
        d.mkdir(parents=True, exist_ok=True)
    return env


def _q(path: Path) -> str:
    """``path`` resolved, as an SBPL string (Seatbelt matches real paths: /tmp is /private/tmp)."""
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def sandbox_profile(ws: Path) -> str:
    """Everything allowed except the silo paths and non-loopback network; ``ws`` is re-allowed
    last (the later rule wins), so it stays usable although it lies under WS_ROOT."""
    deny = " ".join(f"(subpath {_q(p)})" for p in DENY)
    return ("(version 1)(allow default)(deny network*)"
            '(allow network* (remote ip "localhost:*"))'
            '(allow network-bind network-inbound (local ip "localhost:*"))'
            f"(deny file-read* file-write* {deny})"
            f"(allow file-read* file-write* (subpath {_q(ws)}))")


def sandboxed(cmd: list[str], ws: Path) -> list[str]:
    return ["sandbox-exec", "-p", sandbox_profile(ws), *cmd]


def run_group(cmd: list[str], ws: Path, env: dict[str, str], timeout: float,
              stdin: str | None = None) -> tuple[int, str, str, bool]:
    """Run ``cmd`` in ``ws`` as a new process group -> (returncode, stdout, stderr, timed_out).

    On timeout the whole group is killed (the agent and everything it started) and what it
    printed so far is returned. A Jupyter kernel runs in its own session and exits by itself once
    its parent is gone; the bounded second wait covers it still holding the pipes.
    """
    proc = subprocess.Popen(cmd, cwd=ws, env=env, text=True, errors="replace",
                            stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.wait()
            out, err = "", ""
        return proc.returncode, out, err, True
