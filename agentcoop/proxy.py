"""Local OpenAI-compatible endpoint backed by ``agentcoop.llm``.

External harnesses (FutureHouse's Finch via litellm, GEPA, ...) talk to
``http://127.0.0.1:<port>/t/<tag>/s/<sample>/v1`` as if it were the OpenAI API. Every request
goes through the same cache, ledger, spending cap and model-capability layer as our own agents,
so all arms are run on one backbone and costed on one basis. The tag (URL-quoted) attributes
spend; the sample index makes repeated runs honest re-draws instead of cache replays.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

from agentcoop import llm


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # quiet
        pass

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # litellm/openai clients may list models
        if self.path.rstrip("/").endswith("/models"):
            self._reply(200, {"object": "list", "data": [{"id": llm.backbone(), "object": "model"}]})
        else:
            self._reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        parts = self.path.split("/")
        # /t/<tag>/s/<sample>/v1/chat/completions
        if len(parts) < 7 or parts[1] != "t" or parts[3] != "s" or not self.path.endswith(
                "/chat/completions"):
            self._reply(404, {"error": {"message": f"unsupported path {self.path}"}})
            return
        tag, sample = urllib.parse.unquote(parts[2]), int(parts[4])
        params = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if params.get("stream"):
            self._reply(400, {"error": {"message": "streaming is not supported by the proxy"}})
            return
        params["model"] = params["model"].split("/")[-1]
        try:
            with llm.exact_tag(tag):
                body = llm.chat_raw(params, sample=sample)
        except llm.BudgetExceeded as exc:
            # 400 (not 429/5xx) so clients do not retry into the cap.
            self._reply(400, {"error": {"message": f"agentcoop budget: {exc}",
                                        "type": "budget_exceeded"}})
            return
        except llm.LLMError as exc:
            text = str(exc)
            status = int(text[:3]) if text[:3].isdigit() else 502
            self._reply(status, {"error": {"message": text}})
            return
        self._reply(200, body)


@contextmanager
def serve() -> Iterator[str]:
    """Run the proxy on a free local port; yields its root URL (``http://127.0.0.1:<port>``)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


_shared: list[str] = []
_shared_lock = threading.Lock()


def shared() -> str:
    """A process-wide proxy, started on first use and kept for the life of the process."""
    with _shared_lock:
        if not _shared:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            _shared.append(f"http://127.0.0.1:{server.server_address[1]}")
        return _shared[0]


def base_url(root: str, tag: str, sample: int) -> str:
    return f"{root}/t/{urllib.parse.quote(tag, safe='')}/s/{sample}/v1"
