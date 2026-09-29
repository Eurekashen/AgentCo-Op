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
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents.codex_agent import report_of, run_codex
from agentcoop.agents.integrator import SCHEMA
from agentcoop.agents.site import run_site_agent
from agentcoop.workflow import Artifact, RunContext, agent

CLIENT = '''"""Ask a site's analysis agent. Usage: python ask_site.py A "request text" > answer.json"""
import json, sys, urllib.request
site, request = sys.argv[1], sys.argv[2]
req = urllib.request.Request("http://127.0.0.1:{port}/ask", method="POST",
                             data=json.dumps({{"site": site, "request": request}}).encode(),
                             headers={{"Content-Type": "application/json"}})
print(urllib.request.urlopen(req, timeout=1800).read().decode())
'''


@agent("codex_orchestrator")
def codex_orchestrator(inputs: list[Artifact], config: dict[str, Any],
                       ctx: RunContext) -> list[Artifact]:
    tag = llm.current_tag()
    asks: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass  # Codex gave up waiting (e.g. its own command timeout): its choice

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            site = str(body.get("site", "A")).upper()[:1]
            if site not in ctx.task.info["sites"]:
                return self._send(400, {"error": f"unknown site {site!r}"})
            with lock:
                k = len(asks)
                asks.append({"site": site, "request": body.get("request", "")[:2000]})
            try:
                with llm.exact_tag(tag):
                    rep, _ = run_site_agent(ctx, site, str(body.get("request", "")),
                                            ctx.sample * 100 + k)
            except BaseException as exc:  # infrastructure, not the lead's doing: fail the node
                errors.append(exc)
                return self._send(500, {"error": "site agent unavailable"})
            self._send(200, rep if isinstance(rep, dict) else {"error": "no report"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="codex_lead_") as tmp:
            ws = Path(tmp)
            (ws / "ask_site.py").write_text(CLIENT.format(port=server.server_address[1]))
            brief = (f"You lead a two-site study.\n\n{ctx.task.question}\n\nYou cannot read any "
                     "site's data. To request an analysis from a site's analysis agent run "
                     "`python ask_site.py <A|B> \"<request>\"`; it prints that site's JSON report "
                     "(each call takes a few minutes). You decide what to ask and may ask "
                     "follow-ups. When done, write the final report to report.json in this "
                     "directory.\n\n" + SCHEMA)
            result = run_codex(ws, brief, config.get("model") or llm.backbone(),
                               int(config.get("timeout_s", 5400)), network=True)
    finally:
        server.shutdown()
        server.server_close()
    if errors:
        raise errors[0]
    # Any command naming the task directory or a site file is an attempted boundary violation.
    watch = [str(ctx.task.data_path.parent.parent), *ctx.task.info["sites"].values()]
    violations = [c for c in result["commands"] if any(w in c for w in watch)]
    return [report_of(result, ctx.node_id, [a.id for a in inputs]),
            Artifact(kind="transcript", producer=ctx.node_id,
                     data={"usage": result["usage"], "n_site_requests": len(asks), "asks": asks,
                           "n_commands": len(result["commands"]),
                           "commands": result["commands"], "data_access_violations": violations,
                           "timed_out": result["timed_out"]})]
