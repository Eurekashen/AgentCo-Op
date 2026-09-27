"""One door to the model: cache, cost ledger, budget cap.

Every LLM call in the project — our own agents, the judge, and the wrapped external agents
(GeneAgent reaches this through its openai-0.28 shim) — goes through :func:`chat`. That gives
three properties the experiments depend on:

* **Controlled resampling.** The cache key includes an explicit ``sample`` index. Re-running a
  workflow with the same sample index replays identical model outputs, so an edit's effect is
  not confounded with sampling noise; a *new* sample index is an honest re-draw, which is how the
  no-edit rerun control and equal-budget best-of-N are produced.
* **Nominal vs. actual cost.** A cache hit costs nothing to *us*, but a method that needed that
  call would have paid for it. Budget comparisons between methods use ``nominal_usd``; the
  spending cap uses ``actual_usd``.
* **Attribution.** Each call is tagged (``AGENTCOOP_TAG`` env / :func:`tag` context) so the
  ledger can say what every method, task and optimization step cost.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-6-luna"

#: USD per 1M tokens (short context): input, cached input, output.
#: Source: developers.openai.com/api/docs/pricing, checked 2026-09-27.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
}

_tag: contextvars.ContextVar[str] = contextvars.ContextVar("agentcoop_tag", default="")
_lock = threading.Lock()
_client = None


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    nominal_usd: float = 0.0
    actual_usd: float = 0.0
    cached: bool = False
    raw_message: dict[str, Any] = field(default_factory=dict)


def _db_path() -> Path:
    return Path(os.environ.get("AGENTCOOP_LLM_DB", ROOT / "runs" / "llm.sqlite"))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, "
        "input_tokens INT, cached_input_tokens INT, output_tokens INT, nominal_usd REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ledger (ts REAL, tag TEXT, model TEXT, cached INT, "
        "input_tokens INT, output_tokens INT, nominal_usd REAL, actual_usd REAL)"
    )
    return conn


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    path = ROOT / ".secrets" / "openai"
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError("no OpenAI key: set OPENAI_API_KEY or write .secrets/openai")


def _http():
    # Plain REST rather than the openai SDK: external agents run with an openai-0.28 shim on
    # sys.path, and this module must work inside that process too.
    global _client
    if _client is None:
        import httpx

        _client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=httpx.Timeout(240.0, connect=30.0),
        )
    return _client


def current_tag() -> str:
    return _tag.get() or os.environ.get("AGENTCOOP_TAG", "")


@contextlib.contextmanager
def tag(label: str) -> Iterator[None]:
    """Attribute every call inside the block to ``label`` (nested labels join with '/')."""
    parent = current_tag()
    token = _tag.set(f"{parent}/{label}" if parent else label)
    try:
        yield
    finally:
        _tag.reset(token)


def cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    p_in, p_cached, p_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    fresh = max(input_tokens - cached_input_tokens, 0)
    return (fresh * p_in + cached_input_tokens * p_cached + output_tokens * p_out) / 1e6


def spent(prefix: str = "") -> dict[str, float]:
    """Totals from the ledger, optionally restricted to tags starting with ``prefix``."""
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(nominal_usd),0), COALESCE(SUM(actual_usd),0), "
        "COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM ledger "
        "WHERE tag LIKE ?",
        (prefix + "%",),
    ).fetchone()
    conn.close()
    return {
        "calls": row[0],
        "nominal_usd": row[1],
        "actual_usd": row[2],
        "input_tokens": row[3],
        "output_tokens": row[4],
    }


def _budget_cap() -> Optional[float]:
    value = os.environ.get("AGENTCOOP_BUDGET_USD")
    return float(value) if value else None


def _key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    response_format: Optional[dict[str, Any]] = None,
    max_completion_tokens: Optional[int] = None,
    sample: int = 0,
) -> ChatResult:
    """Chat Completions call through the cache and ledger.

    gpt-6-luna only accepts function tools and ``temperature`` when ``reasoning_effort`` is
    ``"none"``; callers that pass tools get that setting unless they chose one explicitly.
    """
    if tools and reasoning_effort is None:
        reasoning_effort = "none"
    params: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        params["tools"] = tools
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed
    if response_format is not None:
        params["response_format"] = response_format
    if max_completion_tokens is not None:
        params["max_completion_tokens"] = max_completion_tokens

    key = _key({"params": params, "sample": sample})
    conn = _connect()
    try:
        hit = conn.execute(
            "SELECT response, input_tokens, cached_input_tokens, output_tokens, nominal_usd "
            "FROM cache WHERE key=?",
            (key,),
        ).fetchone()
        if hit is not None:
            message = json.loads(hit[0])
            conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), current_tag(), model, 1, hit[1], hit[3], hit[4], 0.0),
            )
            return _result(message, hit[1], hit[2], hit[3], hit[4], 0.0, cached=True)

        cap = _budget_cap()
        if cap is not None:
            total = conn.execute("SELECT COALESCE(SUM(actual_usd),0) FROM ledger").fetchone()[0]
            if total >= cap:
                raise BudgetExceeded(f"actual spend ${total:.2f} reached cap ${cap:.2f}")

        response = _call_with_retries(params)
        message = {k: v for k, v in response["choices"][0]["message"].items() if v is not None}
        usage = response.get("usage", {})
        n_in = usage.get("prompt_tokens", 0)
        n_out = usage.get("completion_tokens", 0)
        cached_in = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        usd = cost(model, n_in, cached_in, n_out)
        with _lock:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                (key, json.dumps(message), n_in, cached_in, n_out, usd),
            )
            conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), current_tag(), model, 0, n_in, n_out, usd, usd),
            )
        return _result(message, n_in, cached_in, n_out, usd, usd)
    finally:
        conn.close()


class LLMError(RuntimeError):
    pass


def _call_with_retries(params: dict[str, Any], attempts: int = 7,
                       path: str = "/chat/completions") -> dict[str, Any]:
    import httpx

    delay = 2.0
    last: Exception = LLMError("no attempt made")
    for _ in range(attempts):
        try:
            response = _http().post(path, json=params)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response.json()
            last = LLMError(f"{response.status_code}: {response.text[:500]}")
            if response.status_code != 429 and response.status_code < 500:
                raise last
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise last


def _result(
    message: dict[str, Any], n_in: int, n_cached: int, n_out: int, nominal: float,
    actual: float, cached: bool = False,
) -> ChatResult:
    calls = []
    for call in message.get("tool_calls", []) or []:
        fn = call.get("function", {})
        calls.append({"id": call.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments", "")})
    return ChatResult(
        content=message.get("content") or "",
        tool_calls=calls,
        input_tokens=n_in,
        cached_input_tokens=n_cached,
        output_tokens=n_out,
        nominal_usd=nominal,
        actual_usd=actual,
        cached=cached,
        raw_message=message,
    )


def record_external(model: str, input_tokens: int, cached_input_tokens: int,
                    cache_write_tokens: int, output_tokens: int) -> float:
    """Ledger entry for model usage that happened outside this client (e.g. the Codex CLI),
    priced with the same table so every method is costed on the same basis."""
    p_in, p_cached, p_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    p_write = 0.125 if model == "gpt-6-luna" else p_in * 1.25
    fresh = max(input_tokens - cached_input_tokens - cache_write_tokens, 0)
    usd = (fresh * p_in + cached_input_tokens * p_cached + cache_write_tokens * p_write
           + output_tokens * p_out) / 1e6
    conn = _connect()
    try:
        conn.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                     (time.time(), current_tag(), model, 0, input_tokens, output_tokens, usd, usd))
    finally:
        conn.close()
    return usd


def responses(
    input_items: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning_effort: str = "medium",
    instructions: Optional[str] = None,
    sample: int = 0,
) -> dict[str, Any]:
    """Responses API call (tools *with* reasoning), stateless so it can be cached and replayed.

    ``store=False`` plus encrypted reasoning content lets the caller pass the returned output
    items back as input on the next turn without server-side state. Returns
    ``{"output": [...items], "text": str, "usage": {...}, "nominal_usd", "actual_usd", "cached"}``.
    """
    params: dict[str, Any] = {
        "model": model, "input": input_items, "store": False,
        "reasoning": {"effort": reasoning_effort},
        "include": ["reasoning.encrypted_content"],
    }
    if tools:
        params["tools"] = tools
    if instructions:
        params["instructions"] = instructions
    key = _key({"responses": params, "sample": sample})
    conn = _connect()
    try:
        hit = conn.execute(
            "SELECT response, input_tokens, cached_input_tokens, output_tokens, nominal_usd "
            "FROM cache WHERE key=?", (key,)).fetchone()
        if hit is not None:
            body = json.loads(hit[0])
            conn.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                         (time.time(), current_tag(), model, 1, hit[1], hit[3], hit[4], 0.0))
            return body | {"nominal_usd": hit[4], "actual_usd": 0.0, "cached": True}
        cap = _budget_cap()
        if cap is not None:
            total = conn.execute("SELECT COALESCE(SUM(actual_usd),0) FROM ledger").fetchone()[0]
            if total >= cap:
                raise BudgetExceeded(f"actual spend ${total:.2f} reached cap ${cap:.2f}")
        raw = _call_with_retries(params, path="/responses")
        usage = raw.get("usage", {})
        n_in, n_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cached_in = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        usd = cost(model, n_in, cached_in, n_out)
        text = "".join(c.get("text", "") for item in raw.get("output", [])
                       if item.get("type") == "message" for c in item.get("content", [])
                       if c.get("type") == "output_text")
        body = {"output": raw.get("output", []), "text": text, "usage": usage}
        with _lock:
            conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                         (key, json.dumps(body), n_in, cached_in, n_out, usd))
            conn.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                         (time.time(), current_tag(), model, 0, n_in, n_out, usd, usd))
        return body | {"nominal_usd": usd, "actual_usd": usd, "cached": False}
    finally:
        conn.close()


def chat_json(
    messages: list[dict[str, Any]], *, reasoning_effort: Optional[str] = "low", sample: int = 0,
    retries: int = 2, **kwargs: Any,
) -> tuple[Any, ChatResult]:
    """A chat call whose content must parse as a JSON object. Re-asks (new sample) on bad JSON."""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        result = chat(
            messages,
            reasoning_effort=reasoning_effort,
            response_format={"type": "json_object"},
            sample=sample * 100 + attempt,
            **kwargs,
        )
        try:
            return json.loads(result.content), result
        except json.JSONDecodeError as exc:
            last = exc
    raise ValueError(f"model did not return JSON after {retries + 1} attempts: {last}")
