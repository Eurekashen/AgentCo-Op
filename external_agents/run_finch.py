"""Run FutureHouse's Finch (data-analysis-crow, the BixBench agent) unmodified on one question.

Executed with Finch's own interpreter (``external_agents/finch/.venv``, Python 3.12). Reads a
JSON request on stdin and prints one JSON line. The configuration is BixBench's own
``generate_trajectories.yaml`` (ReActAgent, temperature 1.0, hidden old env states, 40 steps,
open-answer prompt, no images); only the model endpoint is ours: an OpenAI-compatible proxy over
``agentcoop.llm`` so the run uses the shared backbone, cache, ledger and spending cap. One
local-mode fix (no Docker): the notebook path shown to the agent is the real one.

request: {"work_dir", "question" | "prompt", "base_url", "model", "max_steps", "language",
"notebook"}. A "prompt" is used verbatim instead of BixBench's question template (v4 site agents
get our task package); "notebook" names the notebook file (default notebook.ipynb).
"""

import asyncio
import json
import os
import sys
import traceback


async def main(req: dict) -> dict:
    os.environ["USE_DOCKER"] = "false"
    from aviary.core import EvalAnswerMode
    from fhda import prompts
    from fhda.data_analysis_env import DataAnalysisEnv
    from fhda.notebook_env import NBEnvironmentState

    # Local-mode harness fix: Finch always reports the notebook's *container* path
    # (/workspace/...), which only exists in its Docker mode; locally the agent then looks for
    # the data in a directory that does not exist. Report the real path instead.
    NBEnvironmentState.get_container_path = lambda self, path: path
    from fhda.utils import NBLanguage
    from ldp.agent import AgentConfig

    model = f"openai/{req['model']}"
    llm_model = {
        "name": model,
        "config": {"model_list": [{"model_name": model, "litellm_params": {
            "model": model, "api_base": req["base_url"], "api_key": "agentcoop-proxy",
            "temperature": 1.0, "num_retries": 5}}]},
    }
    agent = AgentConfig(agent_type="ReActAgent", agent_kwargs={
        "llm_model": llm_model, "hide_old_env_states": True}).construct_agent()
    language = NBLanguage[req.get("language", "python").upper()]
    prompt = req.get("prompt") or (
        prompts.OPEN_PROMPT_TEMPLATE + "\n" + prompts.AVOID_IMAGES).format(
        question=req["question"], language=language.value)
    env = DataAnalysisEnv(
        problem_id="q", problem=prompt, eval_mode=EvalAnswerMode.EXACT, answer=None,
        nb_path=os.path.join(req["work_dir"], req.get("notebook", "notebook.ipynb")),
        work_dir=req["work_dir"],
        language=language, system_prompt=prompts.CAPSULE_SYSTEM_PROMPT_OPEN,
        use_tmp_work_dir=False)
    obs, tools = await env.reset()
    state = await agent.init_state(tools)
    steps, done = 0, False
    try:
        for steps in range(1, req.get("max_steps", 40) + 1):
            action, state, _ = await agent.get_asv(state, obs)
            obs, _, done, trunc = await env.step(action.value)
            if done or trunc:
                break
    finally:
        if not done:
            await env.close()
    return {"answer": env.state.answer, "steps": steps, "done": bool(done)}


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    try:
        out = asyncio.run(main(request))
    except Exception as exc:  # reported, never raised: the caller decides what a failure means
        out = {"answer": None, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-3000:]}
    print("AGENTCOOP_RESULT " + json.dumps(out, default=str), flush=True)
