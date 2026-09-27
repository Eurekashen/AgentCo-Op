"""openai==0.28 surface for vendored external agents, routed through agentcoop.llm.

GeneAgent is used exactly as published; only its model transport is replaced. It was written
against Azure GPT-4o with ``temperature=0``; here every call goes to ``AGENTCOOP_EXTERNAL_MODEL``
(default: the backbone) with the intent "no extra reasoning, temperature 0". ``agentcoop.llm``
maps that intent onto what the model accepts (for gpt-6-luna: ``reasoning_effort="none"`` +
``temperature=0``; for a non-reasoning model: just ``temperature=0``).

Translation notes (kept faithful rather than "improved"):

* 0.28 ``functions=[...]`` become Chat Completions ``tools``; a returned tool call is presented
  back as ``message["function_call"]`` (first call only, as 0.28 had a single slot).
* GeneAgent appends ``{"role": "function", ...}`` results *without* appending the assistant
  message that requested them. The modern API rejects an orphan tool result, so each one is
  preceded by a synthesized assistant tool call carrying the arguments GeneAgent itself recorded
  in the result text ("called with params {...}").
"""

from __future__ import annotations

import ast
import json
import os
import re

from agentcoop import llm as _llm

api_type = "open_ai"
api_base = ""
api_version = ""
api_key = ""


class OpenAIObject(dict):
    """dict with attribute access, like 0.28's OpenAIObject."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


_PARAMS = re.compile(r"called with params (\{.*?\})(?:, and returns|, but returned)", re.S)


def _arguments_from(content: str) -> str:
    match = _PARAMS.search(content or "")
    if match:
        try:
            return json.dumps(ast.literal_eval(match.group(1)))
        except (ValueError, SyntaxError):
            pass
    return "{}"


def _translate(messages):
    out = []
    counter = 0
    for message in messages:
        message = dict(message)
        role = message.get("role")
        if role == "function":
            counter += 1
            call_id = f"call_{counter}"
            out.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": message.get("name", "fn"),
                                 "arguments": _arguments_from(message.get("content", ""))},
                }],
            })
            out.append({"role": "tool", "tool_call_id": call_id,
                        "content": str(message.get("content", ""))})
            continue
        message.pop("function_call", None)
        out.append({"role": role, "content": message.get("content") or ""})
    return out


class ChatCompletion:
    @staticmethod
    def create(engine=None, model=None, messages=None, functions=None, temperature=None, **_):
        tools = [{"type": "function", "function": f} for f in functions] if functions else None
        sample = int(os.environ.get("AGENTCOOP_SAMPLE", "0"))
        result = _llm.chat(
            _translate(messages or []),
            model=os.environ.get("AGENTCOOP_EXTERNAL_MODEL") or _llm.backbone(),
            tools=tools,
            reasoning_effort="none",
            temperature=0 if temperature is None else temperature,
            sample=sample,
        )
        message = OpenAIObject(role="assistant", content=result.content)
        if result.tool_calls:
            first = result.tool_calls[0]
            message["function_call"] = OpenAIObject(name=first["name"], arguments=first["arguments"])
        choice = OpenAIObject(message=message, finish_reason="stop", index=0)
        return OpenAIObject(choices=[choice], usage=OpenAIObject(
            prompt_tokens=result.input_tokens, completion_tokens=result.output_tokens))
