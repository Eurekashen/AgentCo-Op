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
#: The backbone every role uses unless told otherwise. Override with AGENTCOOP_MODEL; the method
#: must not depend on this choice (see ``backbone()`` / ``grader_model()``).
DEFAULT_MODEL = "gpt-6-luna"

#: USD per 1M tokens (short context): input, cached input, output.
#: Source: developers.openai.com/api/docs/pricing, checked 2026-09-27.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


def backbone() -> str:
    """Model for every agent role (integrator, site agents, coder, judges, optimizer)."""
    return os.environ.get("AGENTCOOP_MODEL", DEFAULT_MODEL)


def external_model() -> str:
    """Model behind the *external* agents (GeneAgent, site analysis agents). They are given
    components, not part of the method, so they do not follow the framework backbone: switching
    the framework's model must not silently change the world it collaborates with."""
    return os.environ.get("AGENTCOOP_EXTERNAL_MODEL", DEFAULT_MODEL)


def grader_model() -> str:
    """The evaluation instrument is pinned and never follows the backbone: changing the system
    under test must not change the ruler it is measured with."""
    return os.environ.get("AGENTCOOP_GRADER_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Capability layer: callers state *intent* (a reasoning level, tools, deterministic decoding);
# each model's accepted parameters are learned once by probing and cached, and the intent is
# mapped onto them. No caller hardcodes one model's quirks.
# ---------------------------------------------------------------------------

#: Profiles known without probing (verified by hand on 2026-09-27).
KNOWN_PROFILES: dict[str, dict[str, Any]] = {
    "gpt-6-luna": {"reasoning": True, "effort_none": True, "chat_tools_with_reasoning": False,
                   "temperature_with_none": True, "responses_reasoning": True},
}
_profiles: dict[str, dict[str, Any]] = {}


def _profiles_path() -> Path:
    return _db_path().parent / "model_profiles.json"


def _raw_ok(path: str, params: dict[str, Any]) -> tuple[bool, str]:
    """Does the model accept these parameters? Only a 400 means "no"; a rate limit, server or
    network error says nothing about capability and must not be stored as one."""
    r = _http().post(path, json=params)
    if r.status_code in (200, 400):
        return r.status_code == 200, r.text[:300]
    raise LLMError(f"capability probe inconclusive ({r.status_code}): {r.text[:200]}")


def profile(model: str) -> dict[str, Any]:
    """Capabilities of ``model`` (probed with a handful of tiny calls the first time)."""
    if model in _profiles:
        return _profiles[model]
    if model in KNOWN_PROFILES:
        _profiles[model] = KNOWN_PROFILES[model]
        return _profiles[model]
    path = _profiles_path()
    stored = json.loads(path.read_text()) if path.exists() else {}
    if model in stored:
        _profiles[model] = stored[model]
        return _profiles[model]
    hi = [{"role": "user", "content": "Reply with the single word: ok"}]
    tool = [{"type": "function", "function": {"name": "noop", "description": "noop",
                                              "parameters": {"type": "object", "properties": {}}}}]
    base = {"model": model, "messages": hi, "max_completion_tokens": 64}
    reasoning, _ = _raw_ok("/chat/completions", {**base, "reasoning_effort": "low",
                                                 "max_completion_tokens": 256})
    prof: dict[str, Any] = {"reasoning": reasoning}
    if reasoning:
        prof["effort_none"] = _raw_ok("/chat/completions", {**base, "reasoning_effort": "none"})[0]
        prof["chat_tools_with_reasoning"] = _raw_ok("/chat/completions", {
            **base, "tools": tool, "reasoning_effort": "low", "max_completion_tokens": 256})[0]
        prof["temperature_with_none"] = prof["effort_none"] and _raw_ok(
            "/chat/completions", {**base, "reasoning_effort": "none", "temperature": 0})[0]
        prof["responses_reasoning"] = _raw_ok("/responses", {
            "model": model, "input": "Reply with the single word: ok",
            "reasoning": {"effort": "low"}, "max_output_tokens": 256})[0]
    else:
        prof.update(effort_none=False, chat_tools_with_reasoning=False,
                    temperature_with_none=False, responses_reasoning=False)
    stored[model] = prof
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=1))
    _profiles[model] = prof
    return prof


def _adapt_chat(params: dict[str, Any]) -> dict[str, Any]:
    """Map a chat request's intent onto what its model accepts."""
    prof = profile(params["model"])
    p = dict(params)
    effort = p.pop("reasoning_effort", None)
    temperature = p.pop("temperature", None)
    if prof["reasoning"]:
        if p.get("tools") and not prof["chat_tools_with_reasoning"]:
            effort = "none" if prof["effort_none"] else None
        if effort == "none" and not prof["effort_none"]:
            effort = None  # the model has no "none" level: leave its default
        if effort is not None:
            p["reasoning_effort"] = effort
        if temperature is not None and effort == "none" and prof["temperature_with_none"]:
            p["temperature"] = temperature
    elif temperature is not None:
        p["temperature"] = temperature
    return p


def _adapt_responses(params: dict[str, Any]) -> dict[str, Any]:
    prof = profile(params["model"])
    p = dict(params)
    if not prof["reasoning"] or not prof.get("responses_reasoning", True):
        p.pop("reasoning", None)
        p.pop("include", None)
    return p

_tag: contextvars.ContextVar[str] = contextvars.ContextVar("agentcoop_tag", default="")
_client = None
_local = threading.local()


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    nominal_usd: float = 0.0
    cached: bool = False


def _db_path() -> Path:
    return Path(os.environ.get("AGENTCOOP_LLM_DB", ROOT / "runs" / "llm.sqlite"))


def _conn() -> sqlite3.Connection:
    """One connection per thread and database path (autocommit, WAL)."""
    path = _db_path()
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    if path not in conns:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=60, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, "
                     "input_tokens INT, cached_input_tokens INT, output_tokens INT, nominal_usd REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS ledger (ts REAL, tag TEXT, model TEXT, cached INT, "
                     "input_tokens INT, output_tokens INT, nominal_usd REAL, actual_usd REAL)")
        conns[path] = conn
    return conns[path]


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


@contextlib.contextmanager
def exact_tag(label: str) -> Iterator[None]:
    """Attribute calls to ``label`` exactly (for work done on another thread's behalf)."""
    token = _tag.set(label)
    try:
        yield
    finally:
        _tag.reset(token)


def cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int,
         cache_write_tokens: int = 0) -> float:
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}: add it to llm.PRICES before using it")
    p_in, p_cached, p_out = PRICES[model]
    fresh = max(input_tokens - cached_input_tokens - cache_write_tokens, 0)
    return (fresh * p_in + cached_input_tokens * p_cached + cache_write_tokens * p_in * 1.25
            + output_tokens * p_out) / 1e6


def spent(prefix: str = "", exclude: str = "") -> dict[str, float]:
    """Ledger totals for tags that start with ``prefix`` (exact, case-sensitive prefix) and do
    not contain ``exclude``."""
    row = _conn().execute(
        "SELECT COUNT(*), COALESCE(SUM(nominal_usd),0), COALESCE(SUM(actual_usd),0), "
        "COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM ledger "
        "WHERE substr(tag, 1, ?) = ? AND (? = '' OR instr(tag, ?) = 0)",
        (len(prefix), prefix, exclude, exclude)).fetchone()
    return {"calls": row[0], "nominal_usd": row[1], "actual_usd": row[2],
            "input_tokens": row[3], "output_tokens": row[4]}


def _ledger(model: str, cached: bool, n_in: int, n_out: int, nominal: float,
            actual: float) -> None:
    _conn().execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), current_tag(), model, int(cached), n_in, n_out, nominal, actual))


def _key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _cached_call(key: str, model: str, call) -> tuple[Any, int, int, int, float, bool]:
    """Serve ``key`` from the cache or run ``call()`` → (body, n_in, n_cached, n_out, usd).

    Returns (body, n_in, n_cached, n_out, nominal_usd, cached). The first writer wins: if another
    thread or process stored the same key while this call was in flight, the stored body is
    returned (and this call's spend is still ledgered), so every consumer of a key sees one value.
    """
    conn = _conn()
    select = ("SELECT response, input_tokens, cached_input_tokens, output_tokens, nominal_usd "
              "FROM cache WHERE key=?")
    hit = conn.execute(select, (key,)).fetchone()
    if hit is None:
        cap = _budget_cap()
        if cap is not None:
            total = conn.execute("SELECT COALESCE(SUM(actual_usd),0) FROM ledger").fetchone()[0]
            if total >= cap:
                raise BudgetExceeded(f"actual spend ${total:.2f} reached cap ${cap:.2f}")
        body, n_in, n_cached, n_out = call()
        usd = cost(model, n_in, n_cached, n_out)
        conn.execute("INSERT OR IGNORE INTO cache VALUES (?,?,?,?,?,?)",
                     (key, json.dumps(body), n_in, n_cached, n_out, usd))
        _ledger(model, False, n_in, n_out, usd, usd)
        stored = conn.execute(select, (key,)).fetchone()
        if stored[0] == json.dumps(body):
            return body, n_in, n_cached, n_out, usd, False
        hit = stored  # lost the race: use the value everyone else sees
    else:
        _ledger(model, True, hit[1], hit[3], hit[4], 0.0)
    return json.loads(hit[0]), hit[1], hit[2], hit[3], hit[4], True


def _budget_cap() -> Optional[float]:
    value = os.environ.get("AGENTCOOP_BUDGET_USD")
    return float(value) if value else None


def chat(
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    response_format: Optional[dict[str, Any]] = None,
    sample: int = 0,
) -> ChatResult:
    """Chat Completions call through the cache and ledger.

    ``reasoning_effort`` and ``temperature`` express intent; ``_adapt_chat`` maps them onto what
    the model accepts (e.g. gpt-6-luna takes function tools only with ``reasoning_effort="none"``;
    a non-reasoning model takes no reasoning parameter at all).
    """
    model = model or backbone()
    params: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        params["tools"] = tools
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        params["temperature"] = temperature
    if response_format is not None:
        params["response_format"] = response_format
    params = _adapt_chat(params)

    def call():
        response = _call_with_retries(params)
        message = {k: v for k, v in response["choices"][0]["message"].items() if v is not None}
        usage = response.get("usage", {})
        return (message, usage.get("prompt_tokens", 0),
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                usage.get("completion_tokens", 0))

    message, n_in, _, n_out, usd, cached = _cached_call(
        _key({"params": params, "sample": sample}), model, call)
    calls = [{"id": c.get("id"), "name": c.get("function", {}).get("name"),
              "arguments": c.get("function", {}).get("arguments", "")}
             for c in message.get("tool_calls", []) or []]
    return ChatResult(message.get("content") or "", calls, n_in, n_out, usd, cached)


def _call_with_retries(params: dict[str, Any], attempts: int = 7,
                       path: str = "/chat/completions") -> dict[str, Any]:
    import httpx

    delay = 2.0
    last: Exception = LLMError("no attempt made")
    for attempt in range(attempts):
        try:
            response = _http().post(path, json=params)
        except httpx.TransportError as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response.json()
            last = LLMError(f"{response.status_code}: {response.text[:500]}")
            if response.status_code != 429 and response.status_code < 500:
                raise last
        if attempt < attempts - 1:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last


def record_replay(model: str, nominal_usd: float) -> None:
    """Ledger a result served from a cache kept outside this client (nominal cost, $0 actual)."""
    _ledger(model, True, 0, 0, nominal_usd, 0.0)


def record_external(model: str, input_tokens: int, cached_input_tokens: int,
                    cache_write_tokens: int, output_tokens: int, *, billed: bool) -> float:
    """Ledger entry for model usage that happened outside this client (e.g. the Codex CLI),
    priced with the same table so every method is costed on the same basis. ``billed=False``
    (e.g. Codex on the user's ChatGPT login) records the API-equivalent cost as nominal only."""
    usd = cost(model, input_tokens, cached_input_tokens, output_tokens, cache_write_tokens)
    _ledger(model, False, input_tokens, output_tokens, usd, usd if billed else 0.0)
    return usd


def responses(
    input_items: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning_effort: str = "medium",
    instructions: Optional[str] = None,
    sample: int = 0,
) -> dict[str, Any]:
    """Responses API call (tools *with* reasoning), stateless so it can be cached and replayed.

    ``store=False`` plus encrypted reasoning content lets the caller pass the returned output
    items back as input on the next turn without server-side state. Returns
    ``{"output": [...items], "text": str, "usage": {...}}``.
    """
    model = model or backbone()
    params: dict[str, Any] = {
        "model": model, "input": input_items, "store": False,
        "reasoning": {"effort": reasoning_effort},
        "include": ["reasoning.encrypted_content"],
    }
    if tools:
        params["tools"] = tools
    if instructions:
        params["instructions"] = instructions
    params = _adapt_responses(params)

    def call():
        raw = _call_with_retries(params, path="/responses")
        usage = raw.get("usage", {})
        text = "".join(c.get("text", "") for item in raw.get("output", [])
                       if item.get("type") == "message" for c in item.get("content", [])
                       if c.get("type") == "output_text")
        return ({"output": raw.get("output", []), "text": text, "usage": usage},
                usage.get("input_tokens", 0),
                (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0,
                usage.get("output_tokens", 0))

    body, *_ = _cached_call(_key({"responses": params, "sample": sample}), model, call)
    return body


def chat_json(
    messages: list[dict[str, Any]], *, reasoning_effort: Optional[str] = "low", sample: int = 0,
    retries: int = 2, **kwargs: Any,
) -> tuple[Any, ChatResult]:
    """A chat call whose content must parse as a JSON object. Re-asks (new sample) on bad JSON."""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        result = chat(messages, reasoning_effort=reasoning_effort,
                      response_format={"type": "json_object"}, sample=sample * 100 + attempt,
                      **kwargs)
        try:
            return json.loads(result.content), result
        except json.JSONDecodeError as exc:
            last = exc
    raise ValueError(f"model did not return JSON after {retries + 1} attempts: {last}")
