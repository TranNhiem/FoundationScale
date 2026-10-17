"""Held-out music evaluation: mean music reward over N tasks x K samples, no training.

    python eval_music.py --model CKPT --base-url http://127.0.0.1:18410 \
        --tasks music_eval.jsonl --abc2midi /path/to/abc2midi [--n 100] [--samples 4]

Serve CKPT first with `vllm serve CKPT --served-model-name policy --port 18410`. The episode
loop, tools, sampling (T 1.0, top_p 0.95) and reward are the training run's, so the score is
comparable to the rollout reward. Prints one JSON line: measured and abstained counts, mean,
the number of samples above zero, and the max.
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from foundationscale.agentic_rl.cli import _HFChatTokenizer
from foundationscale.agentic_rl.engines.vllm import VLLMClient
from foundationscale.agentic_rl.envs.base import EnvSpec, get_env_backend
from foundationscale.agentic_rl.harness.base import EpisodeBudget, EpisodeTask, SamplingParams
from foundationscale.agentic_rl.harness.native_tool_loop import NativeToolLoop
from foundationscale.agentic_rl.rewards.base import MusicRewardFn
from foundationscale.agentic_rl.rewards.music import MusicReward
from foundationscale.agentic_rl.tools import QwenXmlToolCallParser, default_tools


async def evaluate(a: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    tok = _HFChatTokenizer(AutoTokenizer.from_pretrained(a.model))
    client = VLLMClient(base_url=a.base_url, timeout_s=1800.0, served_model_name="policy")
    sampling = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=0, min_p=0.0, presence_penalty=0.0,
        repetition_penalty=1.0, max_new_tokens=a.max_tokens,
    )  # fmt: skip
    loop = NativeToolLoop(
        tools=default_tools(), parser=QwenXmlToolCallParser(), sampling=sampling, tokenizer=tok
    )
    reward = MusicRewardFn(MusicReward(abc2midi_bin=a.abc2midi, timeout_s=60.0), decode=tok.decode)
    backend = get_env_backend("local")
    spec = EnvSpec(backend="local", env={}, allow_unsandboxed=True, episode_timeout_s=3600.0)
    budget = EpisodeBudget(
        step_limit=2, max_response_tokens=a.max_tokens, max_observation_chars=2048
    )
    with Path(a.tasks).open() as fh:
        tasks = [json.loads(line) for line in fh if line.strip()][: a.n]
    sem = asyncio.Semaphore(a.concurrency)

    async def one(t: dict, k: int) -> float | None:
        async with sem:
            task = EpisodeTask(
                uid=t["uid"], session_id=f"{t['uid']}#{k}", messages=tuple(t["messages"]),
                metadata=t.get("metadata", {}),
            )  # fmt: skip
            env = backend.create(spec, instance_id=task.session_id)
            await env.start()
            try:
                out = await loop.run_episode(task, client=client, env=env, budget=budget)
            finally:
                await env.close()
            return (await reward.score(task, out)).value

    res = await asyncio.gather(*(one(t, k) for t in tasks for k in range(a.samples)))
    vals = [v for v in res if v is not None]
    return {
        "model": a.model, "tasks": len(tasks), "samples": a.samples, "measured": len(vals),
        "abstained": len(res) - len(vals), "mean": statistics.fmean(vals) if vals else None,
        "pass_gt0": sum(v > 0 for v in vals), "max": max(vals) if vals else None,
    }  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="checkpoint dir (for the tokenizer)")
    p.add_argument("--base-url", required=True, help="the vLLM server serving the checkpoint")
    p.add_argument("--tasks", required=True, help="held-out task JSONL")
    p.add_argument("--abc2midi", required=True, help="abc2midi binary")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--concurrency", type=int, default=64)
    print(json.dumps(asyncio.run(evaluate(p.parse_args()))))


if __name__ == "__main__":
    main()
