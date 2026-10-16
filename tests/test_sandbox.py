"""Site-agent harnesses: the OS sandbox, Codex's permission profile, the wall-clock cap.

Spec §8: each harness is run against the ``silo`` fixture (a sibling workspace, a fake
``data/hidden/``) and must fail to read either or to reach the network.

Offline: no API key, no network, no real agent run. Two checks are opt-in: the interpreter checks
that need the ``bixbench`` env and Finch's venv (~20 s on a cold matplotlib cache) run only with
AGENTCOOP_SLOW_TESTS=1, and the Codex harness check, which makes one real ``codex exec`` call on
the local login, runs only with AGENTCOOP_LIVE_TESTS=1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import tomllib

import pytest

from agentcoop import llm
from agentcoop.agents import codex, finch, harness, react, workspace


needs_sandbox = pytest.mark.skipif(shutil.which("sandbox-exec") is None,
                                   reason="macOS sandbox-exec not available")

#: Run inside a harness's sandboxed interpreter (written to ``probe.py`` in the workspace): read
#: its own input, then try the sibling workspace, the hidden references and the internet.
PROBE = """\
import errno, socket
print(open('counts.txt').read())
for path in {paths!r}:
    try:
        open(path).read()
        print('read', path)
    except PermissionError:
        print('denied')
try:
    socket.create_connection(('1.1.1.1', 80), timeout=5)
    print('internet open')
except OSError as e:
    print('internet', 'EPERM' if e.errno == errno.EPERM else e)
"""
#: What PROBE prints inside a correct sandbox (EPERM comes from the sandbox, also when offline).
CONFINED = ["mine", "denied", "denied", "internet EPERM"]


@pytest.fixture
def silo(tmp_path, monkeypatch):
    """Two sibling workspaces under a temporary WS_ROOT and a fake ``data/`` holding
    ``hidden/truth.json``, with DENY pointing at them; ``probe.py`` in the first workspace."""
    secret = tmp_path / "data"
    (secret / "hidden").mkdir(parents=True)
    (secret / "hidden" / "truth.json").write_text('{"program": "IFN"}')
    root = tmp_path / "wsroot"
    monkeypatch.setattr(workspace, "WS_ROOT", root)
    monkeypatch.setattr(workspace, "DENY", (secret, root))
    src = tmp_path / "counts.txt"
    src.write_text("mine")
    ws = workspace.new_workspace("task0/site_a", {"counts.txt": src})
    other = workspace.new_workspace("task0/site_b", {"counts.txt": src})
    paths = [str(other / "counts.txt"), str(secret / "hidden" / "truth.json")]
    (ws / "probe.py").write_text(PROBE.format(paths=paths))
    return ws, other, secret


def _sh(script: str, ws):
    return subprocess.run(workspace.sandboxed(["/bin/sh", "-c", script], ws), cwd=ws,
                          env=workspace.exec_env(ws), capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# workspace: profile, sandbox, process groups
# ---------------------------------------------------------------------------


def test_new_workspace_clones_inputs_and_starts_empty(silo):
    ws, _, _ = silo
    (ws / "stale.txt").write_text("old run")
    src = ws.parent / "site_b" / "counts.txt"
    again = workspace.new_workspace("task0/site_a", {"counts.txt": src})
    assert again == ws and sorted(p.name for p in ws.iterdir()) == ["counts.txt"]
    with pytest.raises(ValueError):
        workspace.new_workspace("../escape", {})


def test_profile_denies_every_silo_path_and_reallows_only_its_workspace(tmp_path, monkeypatch):
    deny = (tmp_path / "data", tmp_path / "runs", tmp_path / ".secrets", tmp_path / "wsroot")
    monkeypatch.setattr(workspace, "DENY", deny)
    ws = tmp_path / "wsroot" / "t" / "a"
    profile = workspace.sandbox_profile(ws)
    for d in deny:
        assert f'(subpath "{d.resolve()}")' in profile
    assert "(deny network*)" in profile and '(remote ip "localhost:*")' in profile
    # The workspace rule comes last: in SBPL the later rule wins over the WS_ROOT deny.
    assert profile.endswith(f'(allow file-read* file-write* (subpath "{ws.resolve()}"))')


def test_default_deny_list_covers_the_silo():
    root = llm.ROOT
    assert {root / "data", root / "runs", root / ".secrets", workspace.CODEX_HOME,
            workspace.WS_ROOT} <= set(workspace.DENY)
    # Spec §8 names data/hidden/: it is denied as part of data/ (agents/ code never names it).
    assert any((root / "data" / "hidden").is_relative_to(d) for d in workspace.DENY)


def test_exec_env_strips_keys_and_keeps_caches_in_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AGENTCOOP_BUDGET_USD", "1")
    env = workspace.exec_env(tmp_path)
    assert "OPENAI_API_KEY" not in env and not any(k.startswith("AGENTCOOP_") for k in env)
    for k in ("HOME", "TMPDIR", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
        assert env[k].startswith(str(tmp_path)) and os.path.isdir(env[k])
    assert env["PATH"].startswith(str(workspace.BIXENV / "bin"))


@needs_sandbox
def test_sandboxed_process_reads_only_its_own_workspace(silo):
    """Spec §8 harness test for the bare sandbox (shell commands of every sandboxed harness)."""
    ws, other, secret = silo
    assert _sh("cat counts.txt", ws).stdout == "mine"
    assert _sh("echo ok > out.txt", ws).returncode == 0 and (ws / "out.txt").exists()
    for cmd in (f"cat '{other / 'counts.txt'}'", f"cat '{secret / 'hidden' / 'truth.json'}'",
                f"ls '{secret / 'hidden'}'"):
        r = _sh(cmd, ws)
        assert r.returncode != 0 and "Operation not permitted" in r.stderr, cmd
    assert _sh(f"echo x > '{other / 'planted.txt'}'", ws).returncode != 0
    assert not (other / "planted.txt").exists()
    assert _sh("curl -sS -m 5 http://example.com", ws).returncode != 0


@needs_sandbox
def test_sandbox_allows_loopback_servers_but_not_the_internet(silo):
    """Finch's Jupyter kernel binds loopback ports and Finch calls the local proxy; anything
    else on the network is refused by the sandbox itself (EPERM, also when offline)."""
    ws, _, _ = silo
    probe = (
        "import errno, socket\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
        "c = socket.create_connection(s.getsockname()); a, _ = s.accept(); c.sendall(b'hi')\n"
        "print('loopback', a.recv(2).decode())\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=5); print('internet open')\n"
        "except OSError as e:\n"
        "    print('internet', 'EPERM' if e.errno == errno.EPERM else e)\n")
    r = subprocess.run(workspace.sandboxed([sys.executable, "-c", probe], ws), cwd=ws,
                       env=workspace.exec_env(ws), capture_output=True, text=True, timeout=60)
    assert r.stdout.split("\n")[:2] == ["loopback hi", "internet EPERM"], r.stderr


def test_run_group_kills_the_whole_process_group_on_timeout(tmp_path):
    code, _, _, timed_out = workspace.run_group(
        ["/bin/sh", "-c", "sleep 30 & sleep 30"], tmp_path, dict(os.environ), timeout=0.5)
    assert timed_out  # returns promptly: the background sleep held the pipe and died too

# ---------------------------------------------------------------------------
# Codex: permission profile, boundary items, cost
# ---------------------------------------------------------------------------


def _fake_codex(monkeypatch, tmp_path, lines: list[dict], returncode: int = 0) -> list:
    """Replace subprocess.Popen; returns the list that receives (argv, kwargs) of each call."""
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    calls = []
    stdout = "".join(json.dumps(x) + "\n" for x in lines)

    class FakePopen:
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            self.pid, self.returncode = 0, returncode

        def communicate(self, input=None, timeout=None):
            return stdout, ""

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return calls


USAGE = {"input_tokens": 1000, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
         "output_tokens": 100}


def test_run_codex_uses_the_permission_profile_not_a_sandbox_mode(silo, tmp_path, monkeypatch):
    ws, _, secret = silo
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AGENTCOOP_BUDGET_USD", "1")
    item = {"id": "c1", "type": "command_execution", "command": "ls", "aggregated_output": "x"}
    calls = _fake_codex(monkeypatch, tmp_path, [
        {"type": "item.completed", "item": item},
        {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": USAGE}])
    out = codex.run_codex(ws, "build the view", "gpt-6-luna", 30)
    (argv, kw), = calls
    assert argv[:2] == ["codex", "exec"] and argv[-1] == "build the view"
    assert "-s" not in argv and "--sandbox" not in argv
    for flag in ('default_permissions="site"', 'web_search="disabled"', "allow_login_shell=false"):
        assert flag in argv
    fs_arg = next(a for a in argv if a.startswith("permissions.site.filesystem="))
    fs = tomllib.loads("fs = " + fs_arg.split("=", 1)[1])["fs"]
    assert fs[":root"] == "read" and fs[str(ws.resolve())] == "write"
    assert all(fs[str(p.resolve())] == "none" for p in (secret, workspace.WS_ROOT))
    env = kw["env"]
    assert "OPENAI_API_KEY" not in env and not any(k.startswith("AGENTCOOP_") for k in env)
    assert env["CODEX_HOME"] == str(workspace.CODEX_HOME) and kw["start_new_session"]
    assert out["transcript"] == [{"code": "ls", "output": "x"},
                                 {"code": "(final)", "output": "done"}]
    nominal = llm.cost("gpt-6-luna", 1000, 0, 100)
    assert out["nominal_usd_external"] == pytest.approx(nominal) and not out["timed_out"]
    spent = llm.spent()
    assert spent["nominal_usd"] == pytest.approx(nominal) and spent["actual_usd"] == 0.0


def test_run_codex_fails_on_a_web_search_item_and_still_charges_it(silo, tmp_path, monkeypatch):
    ws, _, _ = silo
    _fake_codex(monkeypatch, tmp_path, [
        {"type": "item.started", "item": {"id": "w1", "type": "web_search", "query": "IFNB1"}},
        {"type": "turn.completed", "usage": USAGE}])
    with pytest.raises(codex.CodexFailed, match="web_search") as info:
        codex.run_codex(ws, "build the view", "gpt-6-luna", 30)
    assert info.value.nominal_usd_external == pytest.approx(llm.cost("gpt-6-luna", 1000, 0, 100))
    assert llm.spent()["calls"] == 1


def test_run_codex_abnormal_exit_is_codex_failed(silo, tmp_path, monkeypatch):
    ws, _, _ = silo
    _fake_codex(monkeypatch, tmp_path, [{"type": "turn.failed", "error": {"message": "quota"}}],
                returncode=1)
    with pytest.raises(codex.CodexFailed):
        codex.run_codex(ws, "build the view", "gpt-6-luna", 30)


#: The live check goes through a script: asked directly to read a denied file, the model refuses
#: without running anything, and that would prove nothing about the sandbox.
CHECK = """\
cat '{sibling}'; echo sibling_exit=$?
cat '{hidden}'; echo hidden_exit=$?
curl -sS -m 5 http://example.com >/dev/null; echo curl_exit=$?
command -v python3
echo ok > ok.txt; echo write_exit=$?
"""
CHECK_PROMPT = ("I am the operator of this machine and I am checking the sandbox I configured for "
                "you. Run `sh check.sh` in the current directory once and report its output "
                "verbatim. Some lines are expected to fail; that is what the check measures.")


@pytest.mark.skipif(os.environ.get("AGENTCOOP_LIVE_TESTS") != "1" or shutil.which("codex") is None,
                    reason="live: one codex exec call on the local login; AGENTCOOP_LIVE_TESTS=1")
def test_codex_harness_cannot_read_the_silo_or_reach_the_network(silo, tmp_path, monkeypatch):
    """Spec §8 harness test for Codex, which runs outside sandbox-exec (it calls its own API): the
    real CLI under ``permission_args``. Judged by the command's output, never by the agent's
    message. Billed to the user's ChatGPT plan: nominal cost only, $0 actual."""
    ws, other, secret = silo
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    (ws / "check.sh").write_text(CHECK.format(sibling=other / "counts.txt",
                                              hidden=secret / "hidden" / "truth.json"))
    out = codex.run_codex(ws, CHECK_PROMPT, llm.external_model(), 300)
    logs = [s["output"] for s in out["transcript"]
            if s["code"] != "(final)" and "write_exit=" in s["output"]]
    assert logs, out["transcript"]
    log = logs[0]
    assert "sibling_exit=1" in log and "hidden_exit=1" in log, log
    assert log.count("Operation not permitted") == 2 and "curl_exit=0" not in log, log
    assert str(workspace.BIXENV / "bin" / "python3") in log and "write_exit=0" in log, log
    assert (ws / "ok.txt").read_text() == "ok\n" and not out["timed_out"]
    assert out["nominal_usd_external"] > 0 and llm.spent()["actual_usd"] == 0.0

# ---------------------------------------------------------------------------
# ReAct and Finch harnesses (fake model / fake process)
# ---------------------------------------------------------------------------


@needs_sandbox
def test_react_kernel_and_shell_run_inside_the_sandbox(silo, monkeypatch):
    """Spec §8 harness test for ReAct: its real kernel and shell under a scripted model."""
    ws, _, secret = silo
    monkeypatch.setattr(react, "PYTHON", sys.executable)
    shell = f"echo view > view.txt && cat view.txt; cat '{secret / 'hidden' / 'truth.json'}'"
    script = [
        {"type": "function_call", "call_id": "1", "name": "run_python",
         "arguments": json.dumps({"code": "exec(open('probe.py').read())"})},
        {"type": "function_call", "call_id": "2", "name": "run_shell",
         "arguments": json.dumps({"command": shell})},
        {"type": "function_call", "call_id": "3", "name": "finish",
         "arguments": json.dumps({"summary": "wrote view.txt"})},
    ]
    samples = []

    def fake_responses(items, *, tools, reasoning_effort, sample, model):
        samples.append(sample)
        return {"output": [script[len(samples) - 1]], "text": "", "usage": {}}

    monkeypatch.setattr(react.llm, "responses", fake_responses)
    out = react.coding_loop("build the view", ws, sample=2, timeout=60)
    first, second, final = out["transcript"]
    assert first["output"].splitlines() == CONFINED
    assert second["output"].startswith("view\n") and "Operation not permitted" in second["output"]
    assert (ws / "view.txt").read_text() == "view\n"
    assert final == {"code": "(final)", "output": "wrote view.txt"}
    assert samples == [2000, 2001, 2002] and not out["timed_out"] and out["error"] is None


@needs_sandbox
def test_react_kernel_that_cannot_start_is_an_infrastructure_error(silo, monkeypatch):
    ws, _, _ = silo
    monkeypatch.setattr(react, "PYTHON", "/nonexistent/python")
    with pytest.raises(RuntimeError, match="failed to start"):
        react.Kernel(ws, workspace.exec_env(ws))


def test_run_finch_sends_the_prompt_verbatim_under_the_sandbox(silo, monkeypatch):
    ws, _, _ = silo
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}
    nb = {"cells": [{"cell_type": "code", "source": ["x = 1\n", "print(x)"],
                     "outputs": [{"output_type": "stream", "text": ["1\n"]}]}]}

    def fake_run_group(cmd, cwd, env, timeout, stdin=None):
        seen.update(cmd=cmd, env=env, req=json.loads(stdin), timeout=timeout)
        (cwd / "followup.ipynb").write_text(json.dumps(nb))
        return 0, 'AGENTCOOP_RESULT {"answer": "view written", "steps": 3}\n', "", False

    monkeypatch.setattr(finch, "run_group", fake_run_group)
    with llm.tag("site-a"):
        out = finch.run_finch(ws, "build the view", sample=7, timeout=900,
                              notebook="followup.ipynb")
    # Spec §8 for Finch: exactly the profile the probe tests run under (see the slow test below).
    assert seen["cmd"] == workspace.sandboxed([str(finch.FINCH_PY), str(finch.RUN_FINCH)], ws)
    assert seen["req"]["prompt"] == "build the view" and seen["req"]["notebook"] == "followup.ipynb"
    assert seen["req"]["base_url"].endswith("/t/site-a/s/7/v1") and seen["timeout"] == 900
    assert "OPENAI_API_KEY" not in seen["env"]
    assert out["transcript"] == [{"code": "x = 1\nprint(x)", "output": "1\n"},
                                 {"code": "(final)", "output": "view written"}]
    assert out["error"] is None and not out["timed_out"]


def test_run_finch_turns_a_budget_refusal_into_budget_exceeded(silo, monkeypatch):
    ws, _, _ = silo
    monkeypatch.setattr(finch, "run_group", lambda *a, **k: (
        0, 'AGENTCOOP_RESULT {"answer": null, "error": "BadRequestError: agentcoop budget: '
           'actual spend $180.00 reached cap $180.00"}\n', "", False))
    with pytest.raises(llm.BudgetExceeded):
        finch.run_finch(ws, "build the view", sample=0, timeout=900)


@pytest.mark.skipif(os.environ.get("AGENTCOOP_SLOW_TESTS") != "1"
                    or shutil.which("sandbox-exec") is None
                    or not finch.FINCH_PY.exists() or not os.path.exists(react.PYTHON),
                    reason="slow: needs AGENTCOOP_SLOW_TESTS=1, the bixbench env and Finch")
def test_site_interpreters_start_in_the_sandbox(silo):
    """Review focus 5: a profile that blocked interpreter startup, scanpy's caches or Finch's
    kernel would make every site look like a scientific failure. Also the spec §8 harness test
    for Finch: the probe runs in Finch's Jupyter kernel, where Finch executes its cells."""
    ws, _, _ = silo
    env = workspace.exec_env(ws)
    r = subprocess.run(workspace.sandboxed(
        [react.PYTHON, "-c", "import scanpy, pydeseq2; exec(open('probe.py').read())"], ws),
        cwd=ws, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.splitlines() == CONFINED, r.stderr[-2000:]
    kernel = (
        "from jupyter_client.manager import start_new_kernel\n"
        "km, kc = start_new_kernel(kernel_name='python')\n"
        "out = []\n"
        "def keep(msg):\n"
        "    if msg['content'].get('name') == 'stdout':\n"
        "        out.append(msg['content']['text'])\n"
        "kc.execute_interactive(\"import scanpy; exec(open('probe.py').read())\",\n"
        "                       output_hook=keep, timeout=300)\n"
        "km.shutdown_kernel(now=True)\n"
        "print(''.join(out), end='')\n")
    r = subprocess.run(workspace.sandboxed([str(finch.FINCH_PY), "-c", kernel], ws), cwd=ws,
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.stdout.splitlines() == CONFINED, r.stderr[-2000:]

# ---------------------------------------------------------------------------
# harness.run_agent
# ---------------------------------------------------------------------------


def _result(**kw):
    return {"transcript": [], "timed_out": False, "error": None, "nominal_usd_external": 0.0, **kw}


def test_run_agent_returns_a_failed_run_when_the_agent_hangs(tmp_path, monkeypatch):
    release = threading.Event()

    def hung(ws, prompt, sample, timeout):
        release.wait(30)
        return _result()

    monkeypatch.setitem(harness.ENTRY, "fake", hung)
    monkeypatch.setattr(harness, "GRACE", 0.2)
    run = harness.run_agent("fake", tmp_path, "p", sample=0, timeout=0)
    release.set()
    assert not run.ok and run.error.startswith("timeout") and run.seconds < 5


def test_run_agent_reports_timeouts_and_failures_as_data(tmp_path, monkeypatch):
    step = [{"code": "ls", "output": ""}]
    monkeypatch.setitem(harness.ENTRY, "slow", lambda *a: _result(transcript=step, timed_out=True))
    run = harness.run_agent("slow", tmp_path, "p", sample=0, timeout=1)
    assert not run.ok and run.error == "timeout after 1s" and run.transcript == step

    def broken(*a):
        raise codex.CodexFailed("codex exit 1: quota", 0.25)

    monkeypatch.setitem(harness.ENTRY, "broken", broken)
    run = harness.run_agent("broken", tmp_path, "p", sample=0, timeout=1)
    assert not run.ok and "quota" in run.error and run.nominal_usd_external == 0.25


def test_run_agent_reraises_the_budget_cap_and_keeps_the_ledger_tag(tmp_path, monkeypatch):
    def capped(*a):
        raise llm.BudgetExceeded("cap")

    monkeypatch.setitem(harness.ENTRY, "capped", capped)
    with pytest.raises(llm.BudgetExceeded):
        harness.run_agent("capped", tmp_path, "p", sample=0, timeout=1)
    monkeypatch.setitem(harness.ENTRY, "tagged", lambda *a: _result(
        transcript=[{"code": "", "output": llm.current_tag()}]))
    with llm.tag("store#ab12cd34"):
        run = harness.run_agent("tagged", tmp_path, "p", sample=0, timeout=1)
    assert run.ok and run.transcript[0]["output"] == "store#ab12cd34"
