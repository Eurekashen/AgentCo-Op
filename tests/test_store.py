"""The memo store: one value per key, nominal cost carried with it, failures never stored."""

from __future__ import annotations

import hashlib
import os
import threading

import pytest

from agentcoop import llm, store


@pytest.fixture
def model(monkeypatch):
    """A fake transport: every uncached chat call costs exactly $0.10 nominal on gpt-6-luna."""
    sent: list[dict] = []

    def fake(params, attempts=7, path="/chat/completions"):
        sent.append(params)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}

    monkeypatch.setattr(llm, "_call_with_retries", fake)
    return sent


def ask(text: str) -> str:
    return llm.chat([{"role": "user", "content": text}]).content


def test_a_miss_computes_once_and_stores_its_nominal_cost(model):
    out = store.memo({"op": "propose", "sample": 0}, lambda: {"text": ask("hi")}, tag="t")
    assert out == {"text": "ok", "nominal_usd": pytest.approx(0.10)}
    assert len(model) == 1
    assert llm.spent("t#")["nominal_usd"] == pytest.approx(0.10)
    assert len(list(store.STORE.rglob("*.json"))) == 1


def test_external_cost_reported_by_compute_is_added_to_the_nominal_cost(model):
    out = store.memo({"op": "codex"}, lambda: {"text": ask("hi"), "nominal_usd_external": 0.5},
                     tag="t")
    assert out["nominal_usd"] == pytest.approx(0.60) and out["nominal_usd_external"] == 0.5


def test_a_hit_returns_the_stored_dict_without_computing_and_ledgers_a_replay(model):
    key = {"op": "propose", "sample": 0}
    first = store.memo(key, lambda: {"text": ask("hi"), "pair": (1, 2)}, tag="t")

    def never() -> dict:
        raise AssertionError("compute ran on a hit")

    with llm.tag("replay"):
        second = store.memo(key, never, tag="t")
    assert second == first and first["pair"] == [1, 2]  # a miss returns the stored JSON form
    replay = llm.spent("replay")
    assert replay["calls"] == 1 and replay["actual_usd"] == 0.0
    assert replay["nominal_usd"] == pytest.approx(0.10)
    redraw = store.memo({**key, "sample": 1}, lambda: {"text": "new"}, tag="t")
    assert redraw["text"] == "new"  # a new sample index is a new key: an honest re-draw


def test_exceptions_propagate_are_not_cached_and_stay_charged_to_the_caller(model):
    def fails() -> dict:
        ask("hi")
        raise RuntimeError("agent timed out")

    with llm.tag("caller"), pytest.raises(RuntimeError, match="timed out"):
        store.memo({"op": "propose"}, fails, tag="t")
    assert llm.spent("caller")["nominal_usd"] == pytest.approx(0.10)
    assert not list(store.STORE.rglob("*.json"))
    assert store.memo({"op": "propose"}, lambda: {"ok": True}, tag="t") == \
        {"ok": True, "nominal_usd": 0.0}


def test_concurrent_writers_of_one_key_all_get_the_first_published_value():
    llm.spent()  # the ledger exists before the writers start, as in every real run
    barrier = threading.Barrier(2)
    results: dict[str, dict] = {}

    def worker(name: str) -> None:
        def compute() -> dict:
            barrier.wait(timeout=10)  # both are past the cache check before either publishes
            return {"value": name}

        results[name] = store.memo({"op": "race"}, compute, tag=name)

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["a"] == results["b"] and results["a"]["value"] in ("a", "b")
    assert store.memo({"op": "race"}, lambda: {"value": "c"}, tag="c") == results["a"]
    assert not list(store.STORE.rglob("*.tmp"))


def test_blobs_round_trip_a_directory_under_a_content_sha256_id(tmp_path):
    src = tmp_path / "view"
    (src / "sub").mkdir(parents=True)
    (src / "genes.tsv").write_text("feature\tsymbol\nENSG01\tISG15\n")
    (src / "sub" / "spec.json").write_text('{"unit": "donor"}')
    blob = store.put_blob(src)
    (stored,) = store.STORE.rglob(f"{blob}.tar.gz")
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == blob
    os.utime(src / "genes.tsv", (1, 1))
    assert store.put_blob(src) == blob  # same content, other mtime: same id, stored once
    dst = tmp_path / "out" / "view"
    store.get_blob(blob, dst)
    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*")) == \
        ["genes.tsv", "sub", "sub/spec.json"]
    assert (dst / "genes.tsv").read_text() == "feature\tsymbol\nENSG01\tISG15\n"
    assert (dst / "sub" / "spec.json").read_text() == '{"unit": "donor"}'
    (src / "genes.tsv").write_text("changed")
    assert store.put_blob(src) != blob
