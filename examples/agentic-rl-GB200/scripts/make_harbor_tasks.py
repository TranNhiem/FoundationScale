"""Convert Harbor-format task dirs (Skill2Env / Terminal-Bench layout) into a ``harbor_task`` jsonl.

Each task dir holds ``environment/Dockerfile``, ``tests/test.sh``, ``instruction.md`` and
``task.toml``. Per task this script builds one ``.sqsh`` whose image also carries ``tests/`` at
``/tests`` (the ``harbor_task`` backend seals that path away while the agent works) and writes one
row: the instruction as the user message, and metadata the ``harbor_task`` backend and
``rewards/swe.py`` read (``fail_to_pass=["harbor::all"]``: resolved iff Harbor would award full
reward). The solution is never baked into an image.

Tasks the backend cannot serve faithfully are skipped with a named reason, never converted:
an ENTRYPOINT (``harbor_task`` starts no services) or an agent phase that needs network.

    python make_harbor_tasks.py --tasks-root DIR --out OUT_DIR [--only a,b] [--limit N]
        [--workers 8]
Writes OUT_DIR/tasks.jsonl, OUT_DIR/skipped.json and images under OUT_DIR/images.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tomllib

from foundationscale.agentic_rl.envs.harbor_task import ALL_TEST
from foundationscale.agentic_rl.sandbox.build import build_image

SYSTEM = (
    "You are an engineer working in a Linux shell inside a container with no network access. "
    "Use the bash tool to inspect files, run programs and create or edit files. The task's "
    "checks are hidden from you; you may write and run your own. Finish the task described by "
    "the user, then call the submit tool with a one-line summary."
)
TEST_CMD = "bash /tests/test.sh"
HIDDEN_DIR = "_fs_hidden_tests"


def last_workdir(dockerfile: str) -> str:
    found = re.findall(r"^\s*WORKDIR\s+(\S+)", dockerfile, re.M | re.I)
    return found[-1] if found and found[-1].startswith("/") else "/"


def skip_reason(dockerfile: str, toml: dict) -> str | None:
    if re.search(r"^\s*ENTRYPOINT\b", dockerfile, re.M | re.I):
        return "entrypoint: harbor_task starts no services"
    if (toml.get("agent") or {}).get("network_mode", "no-network") != "no-network":
        return "agent phase needs network: harbor_task is always offline"
    return None


def build_context(task_dir: Path, work: Path) -> Path:
    """environment/ plus tests/ copied as HIDDEN_DIR, and a Dockerfile that
    installs it at /tests."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(task_dir / "environment", work, symlinks=True)
    shutil.copytree(task_dir / "tests", work / HIDDEN_DIR, symlinks=True)
    df = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")
    # The hidden tests sit in the build context, so an original broad `COPY . <dest>` also installs
    # an UNSEALED copy
    # (measured 2026-11-27: 114/811 gated Skill2Env images had
    # /workspace/_fs_hidden_tests/{oracle,verify,test.sh}
    # readable by the agent). After installing them at /tests, delete every other copy from the
    # final rootfs.
    scrub = (
        f"RUN find / -xdev -path /tests -prune -o -type d -name {HIDDEN_DIR} "
        f"-prune -exec rm -rf {{}} +"
    )
    (work / "Dockerfile").write_text(
        df.rstrip("\n") + f"\nCOPY {HIDDEN_DIR}/ /tests/\n{scrub}\n", encoding="utf-8"
    )
    return work


def row(task_dir: Path, image: Path, toml: dict, dockerfile: str) -> dict:
    verifier = toml.get("verifier") or {}
    agent = toml.get("agent") or {}
    return {
        "uid": task_dir.name,
        # One user turn carrying the instructions: the native tool loop renders cumulative prefixes
        # starting
        # at messages[:1], and Qwen3.5's chat template refuses a render with no user message in it.
        "messages": [
            {
                "role": "user",
                "content": SYSTEM
                + "\n\n"
                + (task_dir / "instruction.md").read_text(encoding="utf-8"),
            }
        ],
        "metadata": {
            "image": str(image.resolve()),
            "workdir": last_workdir(dockerfile),
            "test_cmd": TEST_CMD,
            "hidden_paths": json.dumps(["/tests"]),
            "test_timeout_s": str(float(verifier.get("timeout_sec", 600.0))),
            "agent_timeout_s": str(float(agent.get("timeout_sec", 1800.0))),
            "fail_to_pass": json.dumps([ALL_TEST]),
            "pass_to_pass": "[]",
            "source": "harbor",
            "harbor_task": str((toml.get("task") or {}).get("name", task_dir.name)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated task dir names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-timeout-s", type=float, default=3600.0)
    ap.add_argument("--workers", type=int, default=1, help="parallel image builds")
    args = ap.parse_args(argv)
    names = [n for n in args.only.split(",") if n] or sorted(
        p.name for p in args.tasks_root.iterdir() if (p / "task.toml").is_file()
    )
    if args.limit:
        names = names[: args.limit]
    env = os.environ
    roots = {
        k: env[v]
        for k, v in (
            ("data_root", "FS_SANDBOX_DATA_ROOT"),
            ("image_store", "FS_SANDBOX_IMAGE_STORE"),
            ("scratch_root", "FS_SANDBOX_SCRATCH_ROOT"),
        )
    }
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)

    def convert(name: str) -> tuple[str, dict | None, str | None]:
        task_dir = args.tasks_root / name
        try:
            toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
            dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")
        except (OSError, tomllib.TOMLDecodeError) as exc:
            return name, None, f"unreadable task: {type(exc).__name__}: {exc}"
        reason = skip_reason(dockerfile, toml)
        if reason:
            return name, None, reason
        image = images / f"{name}.sqsh"
        if not image.is_file():
            ctx = build_context(task_dir, args.out / "build" / name)
            try:
                build_image(
                    ctx / "Dockerfile",
                    ctx,
                    image,
                    timeout_s=args.build_timeout_s,
                    enroot_sysconf=env.get("FS_SANDBOX_ENROOT_SYSCONF"),
                    **roots,
                )
            except Exception as exc:  # noqa: BLE001 - one bad build skips that task, named
                return name, None, f"image build failed: {type(exc).__name__}: {exc}"[:500]
            finally:
                shutil.rmtree(ctx, ignore_errors=True)
        return name, row(task_dir, image, toml, dockerfile), None

    rows, skipped = [], {}
    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        for name, built, reason in pool.map(convert, names):
            if built is None:
                skipped[name] = reason
                print(f"skip {name}: {reason}"[:300], flush=True)
            else:
                rows.append(built)
                print(f"ok {name}", flush=True)
    (args.out / "tasks.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (args.out / "skipped.json").write_text(
        json.dumps(skipped, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"converted {len(rows)} skipped {len(skipped)}", flush=True)
    return 0 if rows else 5


if __name__ == "__main__":
    sys.exit(main())
