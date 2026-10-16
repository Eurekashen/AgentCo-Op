"""The model client: capability layer, stop-sequence emulation, model pinning. Offline."""

from __future__ import annotations

import pytest

from agentcoop import llm


@pytest.fixture
def profiles(monkeypatch):
    monkeypatch.setattr(llm, "_profiles", {
        "reasoner-no-chat-tools": {"reasoning": True, "effort_none": True,
                                   "chat_tools_with_reasoning": False,
                                   "temperature_with_none": True, "responses_reasoning": True},
        "reasoner": {"reasoning": True, "effort_none": False, "chat_tools_with_reasoning": True,
                     "temperature_with_none": False, "responses_reasoning": True},
        "plain": {"reasoning": False, "effort_none": False, "chat_tools_with_reasoning": False,
                  "temperature_with_none": False, "responses_reasoning": False},
    })
    return llm


def test_tools_on_a_reasoner_without_chat_tool_support_fall_back_to_effort_none(profiles):
    p = profiles._adapt_chat({"model": "reasoner-no-chat-tools", "messages": [], "tools": [1],
                              "reasoning_effort": "none", "temperature": 0})
    assert p["reasoning_effort"] == "none" and p["temperature"] == 0


def test_a_non_reasoning_model_gets_no_reasoning_parameter(profiles):
    p = profiles._adapt_chat({"model": "plain", "messages": [], "reasoning_effort": "low"})
    assert "reasoning_effort" not in p and "temperature" not in p
    q = profiles._adapt_chat({"model": "plain", "messages": [], "tools": [1],
                              "reasoning_effort": "none", "temperature": 0})
    assert "reasoning_effort" not in q and q["temperature"] == 0
    r = profiles._adapt_responses({"model": "plain", "input": [], "reasoning": {"effort": "low"},
                                   "include": ["reasoning.encrypted_content"]})
    assert "reasoning" not in r and "include" not in r


def test_a_reasoner_keeps_its_effort_and_drops_unsupported_temperature(profiles):
    p = profiles._adapt_chat({"model": "reasoner", "messages": [], "tools": [1],
                              "reasoning_effort": "low", "temperature": 0})
    assert p["reasoning_effort"] == "low" and "temperature" not in p
    q = profiles._adapt_chat({"model": "reasoner", "messages": [], "reasoning_effort": "none"})
    assert "reasoning_effort" not in q  # no "none" level: the model default applies


def test_grader_model_is_pinned_independently_of_the_backbone(monkeypatch):
    monkeypatch.setenv("AGENTCOOP_MODEL", "some-other-model")
    assert llm.backbone() == "some-other-model" and llm.grader_model() == llm.DEFAULT_MODEL


def test_external_agents_do_not_follow_the_framework_backbone(monkeypatch):
    monkeypatch.setenv("AGENTCOOP_MODEL", "another-model")
    assert llm.external_model() == llm.DEFAULT_MODEL
    monkeypatch.setenv("AGENTCOOP_EXTERNAL_MODEL", "yet-another")
    assert llm.external_model() == "yet-another" and llm.backbone() == "another-model"


def test_stop_sequences_are_emulated_for_a_model_without_stop(monkeypatch):
    monkeypatch.setattr(llm, "_profiles", {"m": {"reasoning": False, "stop": False}})
    monkeypatch.setitem(llm.PRICES, "m", (0.0, 0.0, 0.0))
    sent = {}

    def fake(params, attempts=7, path="/chat/completions"):
        sent.update(params)
        return {"choices": [{"message": {"role": "assistant",
                                         "content": "Thought: x\nAction: y\nObservation: z"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(llm, "_call_with_retries", fake)
    body = llm.chat_raw({"model": "m", "messages": [], "stop": ["Action:"], "stream": False})
    assert "stop" not in sent and "stream" not in sent
    assert body["choices"][0]["message"]["content"] == "Thought: x\n"
