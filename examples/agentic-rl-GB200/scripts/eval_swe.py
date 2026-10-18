"""Held-out SWE-agent evaluation: resolve rate over N tasks x K samples, no training.

    python eval_swe.py --model CKPT --base-url http://127.0.0.1:18410 \
        --tasks swe_eval.jsonl [--n 24] [--samples 4] [--dump transcripts.jsonl]

Serve CKPT first with `vllm serve CKPT --served-model-name policy --port 18410`. Episodes run in
enroot_task sandboxes (FS_SANDBOX_* and, inside a container, FS_SANDBOX_BROKER* must be set), with
the training run's loop, tools, sampling (T 1.0, top_p 0.95), verify() and SweReward, so the score
is comparable to the rollout reward. Prints one JSON line (mean = resolve rate); --dump writes one
JSON line per episode with the decoded transcript, the verify() report and the verdict.
"""

import argparse
import asyncio
import json
import statistics
from dataclasses import replace
from pathlib import Path

from foundationscale.agentic_rl.cli import _HFChatTokenizer
from foundationscale.agentic_rl.engines.vllm import VLLMClient
from foundationscale.agentic_rl.envs.base import EnvSpec, get_env_backend
from foundationscale.agentic_rl.harness.base import EpisodeBudget, EpisodeTask, SamplingParams
from foundationscale.agentic_rl.harness.native_tool_loop import NativeToolLoop
from foundationscale.agentic_rl.rewards.swe import SweReward
from foundationscale.agentic_rl.tools import (
    HermesJsonToolCallParser,
    QwenXmlToolCallParser,
    default_tools,
)


async def evaluate(a: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    tok = _HFChatTokenizer(AutoTokenizer.from_pretrained(a.model))
    client = VLLMClient(base_url=a.base_url, timeout_s=1800.0, served_model_name="policy")
    sampling = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=0, min_p=0.0, presence_penalty=0.0,
        repetition_penalty=1.0, max_new_tokens=a.max_tokens,
    )  # fmt: skip
    loop = NativeToolLoop(
        tools=default_tools(),
        parser=QwenXmlToolCallParser() if a.parser == "qwen_xml" else HermesJsonToolCallParser(),
        sampling=sampling,
        tokenizer=tok,
    )
    reward = SweReward()
    backend = get_env_backend("enroot_task")
    spec = EnvSpec(backend="enroot_task", env={}, exec_timeout_s=60.0, episode_timeout_s=1500.0)
    budget = EpisodeBudget(
        step_limit=a.step_limit, max_response_tokens=a.total_tokens, max_observation_chars=4096
    )
    with Path(a.tasks).open() as fh:
        tasks = [json.loads(line) for line in fh if line.strip()][: a.n]
    sem = asyncio.Semaphore(a.concurrency)
    dump = Path(a.dump).open("w") if a.dump else None  # noqa: SIM115 -- closed after the gather

    async def one(t: dict, k: int) -> float | None:
        async with sem:
            task = EpisodeTask(
                uid=t["uid"], session_id=f"{t['uid']}#{k}", messages=tuple(t["messages"]),
                metadata=t.get("metadata", {}),
            )  # fmt: skip
            env = backend.create_for_task(spec, task, instance_id=task.session_id)
            await env.start()
            try:
                out = await loop.run_episode(task, client=client, env=env, budget=budget)
                report = await env.verify()
            finally:
                await env.close()
            out = replace(out, metadata={**out.metadata, **report})
            verdict = await reward.score(task, out)
            if dump is not None:
                text = tok.decode([i for turn in out.trajectory_turns for i in turn.token_ids])
                dump.write(json.dumps({
                    "session": task.session_id, "reward": verdict.value, "reason": verdict.reason,
                    "termination": str(out.termination), "report": dict(report), "transcript": text,
                }) + "\n")  # fmt: skip
                dump.flush()
            return verdict.value

    res = await asyncio.gather(*(one(t, k) for t in tasks for k in range(a.samples)))
    if dump is not None:
        dump.close()
    vals = [v for v in res if v is not None]
    return {
        "model": a.model, "tasks": len(tasks), "samples": a.samples, "measured": len(vals),
        "abstained": len(res) - len(vals), "mean": statistics.fmean(vals) if vals else None,
        "resolved": sum(v == 1.0 for v in vals), "parser": a.parser,
    }  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="checkpoint dir (for the tokenizer)")
    p.add_argument("--base-url", required=True, help="the vLLM server serving the checkpoint")
    p.add_argument("--tasks", required=True, help="held-out task JSONL")
    p.add_argument("--step-limit", type=int, default=20, help="max model turns")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8000, help="per model turn")
    p.add_argument("--total-tokens", type=int, default=16000, help="per episode")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--parser", choices=["qwen_xml", "hermes_json"], default="qwen_xml")
    p.add_argument("--dump", default=None, help="write per-episode transcripts here (JSONL)")
    print(json.dumps(asyncio.run(evaluate(p.parse_args()))))


if __name__ == "__main__":
    main()
