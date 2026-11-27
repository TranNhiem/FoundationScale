"""Oracle/no-op gate for a ``harbor_task`` jsonl, through the exact env + reward the trainer uses.

Per task: a no-op episode (start, verify) must score 0.0, and an oracle episode (start, copy the
task's ``solution/`` to /solution, ``bash /solution/solve.sh``, verify) must score 1.0, both via
``harbor_task`` + ``rewards/swe.py``. Anything else -- an abstention included -- is not accepted.
The solution is copied into the running container only here; it is never in a training image.

    python validate_harbor_tasks.py --tasks OUT/tasks.jsonl --tasks-root DIR --out OUT/gate.json [--workers 4]
        [--write-tasks OUT/tasks.gated.jsonl]

``--write-tasks`` keeps only accepted tasks and sets ``fail_to_pass`` to the discriminating reward leaves: those the
oracle passes and the no-op fails. Leaves the no-op already passes would be free reward under partial credit, so they
are dropped (listed in ``harbor_nop_leaves``). With no discriminating leaf the task keeps ``["harbor::all"]``.
Exit 0 when every task is accepted, 5 when any is rejected, 95 when none could be measured.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from foundationscale.agentic_rl.envs.base import EnvSpec, get_env_backend
from foundationscale.agentic_rl.harness.base import EpisodeTask
from foundationscale.agentic_rl.rewards.swe import SweReward


class _Outcome:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.metadata = metadata


async def episode(task: EpisodeTask, solution: Path | None, idx: int) -> dict:
    meta = task.metadata
    spec = EnvSpec(backend="harbor_task", env={}, exec_timeout_s=float(meta.get("agent_timeout_s", 1800)),
                   episode_timeout_s=float(meta.get("agent_timeout_s", 1800)) + 600)
    t0 = time.monotonic()
    out: dict = {}
    env = None
    try:
        env = get_env_backend("harbor_task").create_for_task(spec, task, instance_id=f"{task.uid}#{idx}")
        await env.start()
        if solution is not None:
            target = env._sandbox.host_path("/solution")
            shutil.copytree(solution, target, dirs_exist_ok=True)
            res = await env.execute("bash /solution/solve.sh", timeout_s=float(meta.get("agent_timeout_s", 1800)))
            out["solve_exit"] = res.return_code if hasattr(res, "return_code") else None
        verdict = await env.verify()
        score = await SweReward().score(task, _Outcome(verdict))
        out.update(reward=score.value, reason=score.reason, swe_status=verdict.get("swe_status"),
                   swe_error=verdict.get("swe_error"), tests=verdict.get("swe_tests"))
    except Exception as exc:  # noqa: BLE001 - an infra fault is recorded, never scored
        out.update(reward=None, reason=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        if env is not None:
            await env.close()
    out["wall_s"] = round(time.monotonic() - t0, 1)
    return out


def task_fingerprint(task_dir: Path) -> str:
    """sha256 over every file under the task dir (environment, tests, solution): what the image and verdict depend on."""
    h = hashlib.sha256()
    for f in sorted(p for p in task_dir.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(task_dir)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def reusable(uid: str, result: dict, tasks_root: Path) -> bool:
    """A stored verdict is reused only when the task bytes it was measured on are unchanged."""
    d = tasks_root / uid
    return d.is_dir() and result.get("fingerprint") == task_fingerprint(d)


async def gate(task: EpisodeTask, tasks_root: Path, sem: asyncio.Semaphore, log: Path | None = None) -> tuple[str, dict]:
    async with sem:
        nop = await episode(task, None, 0)
        oracle = await episode(task, tasks_root / task.uid / "solution", 1)
    if oracle.get("reward") is None or nop.get("reward") is None:
        status = "UNMEASURED"
    else:
        status = "PASS" if oracle["reward"] == 1.0 and nop["reward"] == 0.0 else "RED"
    print(task.uid, status, oracle.get("reward"), nop.get("reward"), flush=True)
    result = {"status": status, "oracle": oracle, "nop": nop, "fingerprint": task_fingerprint(tasks_root / task.uid)}
    if log is not None:  # one line per finished task: a crash or kill never loses finished work
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"uid": task.uid, **result}) + "\n")
    return task.uid, result


async def run(args: argparse.Namespace) -> dict:
    rows = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [EpisodeTask(uid=r["uid"], session_id=f"{r['uid']}#gate", messages=tuple(r["messages"]),
                         metadata=r["metadata"]) for r in rows]
    log = args.out.with_suffix(".progress.jsonl")
    done: dict = {}
    if args.resume and log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                uid = r.pop("uid")
                if reusable(uid, r, args.tasks_root):
                    done[uid] = r
        print(f"resuming: {len(done)} task(s) already gated on unchanged bytes", flush=True)
    sem = asyncio.Semaphore(args.workers)
    fresh = await asyncio.gather(*(gate(t, args.tasks_root, sem, log) for t in tasks if t.uid not in done))
    return {**done, **dict(fresh)}


def discriminating(result: dict) -> tuple[list[str], list[str]]:
    def tests(side: str) -> dict:
        return json.loads(result[side].get("tests") or "{}")
    oracle, nop = tests("oracle"), tests("nop")
    leaves = sorted(k for k, v in oracle.items() if k != "harbor::all" and v == "passed")
    return [k for k in leaves if nop.get(k) != "passed"], [k for k in leaves if nop.get(k) == "passed"]


def write_tasks(tasks_path: Path, results: dict, out: Path) -> int:
    rows = []
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        res = results.get(r["uid"])
        if not res or res["status"] != "PASS":
            continue
        keep, free = discriminating(res)
        r["metadata"] = {**r["metadata"], "fail_to_pass": json.dumps(keep or ["harbor::all"]),
                         "harbor_nop_leaves": json.dumps(free), "validated": "oracle+nop"}
        rows.append(r)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--tasks-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write-tasks", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="skip tasks already in <out>.progress.jsonl")
    ap.add_argument("--from-gate", type=Path, default=None, help="reuse an earlier gate.json instead of re-running")
    args = ap.parse_args(argv)
    if args.from_gate:
        results = json.loads(args.from_gate.read_text(encoding="utf-8"))
        for uid, r in results.items():  # a verdict measured on other bytes is not evidence for these
            if r.get("status") == "PASS" and not reusable(uid, r, args.tasks_root):
                results[uid] = {**r, "status": "UNMEASURED", "stale": "task bytes changed since gating"}
    else:
        results = asyncio.run(run(args))
        args.out.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
    if args.write_tasks:
        print("WROTE", write_tasks(args.tasks, results, args.write_tasks), flush=True)
    counts = {s: sum(r["status"] == s for r in results.values()) for s in ("PASS", "RED", "UNMEASURED")}
    print("GATE", json.dumps(counts), flush=True)
    if counts["PASS"] == len(results) and results:
        return 0
    return 5 if counts["PASS"] or counts["RED"] else 95


if __name__ == "__main__":
    sys.exit(main())
