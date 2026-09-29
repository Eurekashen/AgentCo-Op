"""Codex CLI as the study lead of a two-site study (shape-2 control).

The user's logged-in Codex runs hermetically (see ``codex_agent``; same ``model`` semantics) in a
workspace that contains no data — only a small
``ask_site.py`` client. The client talks to a loopback HTTP service hosted by this process, which
runs the requested site agent on that site's private data and returns its JSON report. Codex
never receives a data path; after the run, its executed commands are scanned for any attempt to
read site data, and a violation is recorded in the transcript.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents.codex_agent import CODEX_HOME
from agentcoop.agents.coder import coding_loop
from agentcoop.agents.integrator import SCHEMA
from agentcoop.agents.site import GENESETS, site_brief, site_file
from agentcoop.workflow import Artifact, RunContext, agent

CLIENT = '''"""Ask a site's analysis agent. Usage: python ask_site.py A "request text" > answer.json"""
import json, sys, urllib.request
site, request = sys.argv[1], sys.argv[2]
req = urllib.request.Request("http://127.0.0.1:{port}/ask", method="POST",
                             data=json.dumps({{"site": site, "request": request}}).encode(),
                             headers={{"Content-Type": "application/json"}})
print(urllib.request.urlopen(req, timeout=1800).read().decode())
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@agent("codex_orchestrator")
def codex_orchestrator(inputs: list[Artifact], config: dict[str, Any],
                       ctx: RunContext) -> list[Artifact]:
    tag = llm.current_tag()
    asks: list[dict[str, Any]] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            site = str(body.get("site", "A")).upper()[:1]
            if site not in ctx.task.info["sites"]:
                self.send_response(400)
                self.end_headers()
                return
            with lock:
                k = len(asks)
                asks.append({"site": site, "request": body.get("request", "")[:2000]})
            files = {f"site_{site}.h5ad": site_file(ctx, site),
                     "GO_Biological_Process_2023.txt": GENESETS / "GO_Biological_Process_2023.txt",
                     "Reactome_2022.txt": GENESETS / "Reactome_2022.txt"}
            token = llm._tag.set(tag)
            try:
                rep, _ = coding_loop(site_brief(site, str(body.get("request", ""))),
                                     sample=ctx.sample * 100 + k, files=files,
                                     model=llm.external_model())
            finally:
                llm._tag.reset(token)
            out = json.dumps(rep if isinstance(rep, dict) else {"error": "no report"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="codex_lead_") as tmp:
            ws = Path(tmp)
            (ws / "ask_site.py").write_text(CLIENT.format(port=port))
            subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
            home = ws / ".home"
            home.mkdir()
            env = dict(os.environ, CODEX_HOME=str(CODEX_HOME), HOME=str(home))
            brief = (f"You lead a two-site study.\n\n{ctx.task.question}\n\nYou cannot read any "
                     "site's data. To request an analysis from a site's analysis agent run "
                     "`python ask_site.py <A|B> \"<request>\"`; it prints that site's JSON report "
                     "(each call takes a few minutes). You decide what to ask and may ask "
                     "follow-ups. When done, write the final report to report.json in this "
                     "directory.\n\n" + SCHEMA)
            model = config.get("model") or llm.backbone()
            native = model == "native"
            if native:
                env["HOME"] = str(Path.home())
            cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
                   *([] if native else ["--ignore-user-config", "-c", "project_doc_max_bytes=0",
                                        "-m", model]),
                   "-c", "shell_environment_policy.inherit=all",
                   "-c", "sandbox_workspace_write.network_access=true",
                   "-s", "workspace-write", "--json", brief]
            try:
                proc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True,
                                      timeout=int(config.get("timeout_s", 5400)))
                stdout, timed_out = proc.stdout, False
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                timed_out = True
            events = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
            usage = {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
                     "output_tokens": 0}
            for e in events:
                if e.get("type") == "turn.completed":
                    for key in usage:
                        usage[key] += int(e.get("usage", {}).get(key, 0) or 0)
            from agentcoop.agents.codex_agent import _native_model

            llm.record_external(_native_model() if native else model, usage["input_tokens"],
                                usage["cached_input_tokens"], usage["cache_write_input_tokens"],
                                usage["output_tokens"], billed=False)
            commands = [str(e.get("item", {}).get("command", "")) for e in events
                        if e.get("item", {}).get("type") == "command_execution"]
            data_dir = str(ctx.task.data_path.parent)
            violations = [c for c in commands if ".h5ad" in c or data_dir in c]
            try:
                report = json.loads((ws / "report.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                report = {"claims": [], "summary": "(no valid report.json written)"}
    finally:
        server.shutdown()
    if isinstance(report, list):
        report = {"claims": report}  # an array of claims is still a report
    if not isinstance(report, dict):
        report = {"claims": [], "summary": "(report.json is not a JSON object)"}
    claims = report.get("claims") if isinstance(report.get("claims"), list) else []
    return [Artifact(kind="report", producer=ctx.node_id,
                     data={"claims": claims, "summary": str(report.get("summary", ""))},
                     derived_from=[a.id for a in inputs]),
            Artifact(kind="transcript", producer=ctx.node_id,
                     data={"usage": usage, "n_site_requests": len(asks), "asks": asks,
                           "n_commands": len(commands), "data_access_violations": violations,
                           "timed_out": timed_out})]
