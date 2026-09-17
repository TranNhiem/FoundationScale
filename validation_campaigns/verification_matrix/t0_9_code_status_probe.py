"""T0-9: is `code.status is a real commit` actually RED, and which half of it?

This row has carried "RED (#346)" with no adjudicator and no recorded run. Three
rows carrying an asserted ABSENT were retracted by measurement this week, so an
asserted RED on a tier-0 row is not something to publish unexamined.

The control arm as declared is COMPOUND, and the two halves are independent:

  (a) a COPY must RECORD status=not_a_repository
  (b) and the run must SAY SO LOUDLY, not proceed silently

(a) is clearly implemented -- CaptureStatus.NOT_A_REPOSITORY exists, tests cover
it, and manifest.py's own comment states the design: "an unversioned working
directory yields an honest manifest, not a missing one". (b) has no consumer:
nothing outside manifest.py reads code.status at all. So the expected outcome is
RED on (b) only, which is a materially different statement from "RED" and worth
publishing as such. The probe is written so that it can also retract.

CONTROLS, because "it said not_a_repository" proves nothing on its own:
  capture_on_real_repo   a real checkout must NOT come back not_a_repository and
                         must carry a 40-hex commit. A capture that answered
                         not_a_repository for everything would otherwise score
                         as correct.
  copy_parent_not_repo   `git rev-parse` walks UPWARD. If the copy is written
                         inside some enclosing repository, the copy is not a
                         copy and the whole run measures the enclosing tree.
                         Recorded, not assumed.
  detector_selftest      the stream detector must fire on a synthetic line that
                         contains the token. A detector keyed on prose nothing
                         emits reports silence that is its own.
  dry_run_on_real_repo   --dry-run must exit 0 from the real checkout, so a
                         different exit from the copy is about the copy.

CPU only, no GPU, no download, no training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Anything a run could plausibly print to make the absence audible. Deliberately
# WIDE: a narrow pattern would report silence that is really a vocabulary miss,
# and the finding here is "nothing at all", which only means something if the
# net was wide enough to have caught something.
STREAM_TOKENS = (
    "not_a_repository",
    "not a repository",
    "not_captured",
    "no commit",
    "unversioned",
    "provenance",
    "code.status",
)

BASE = [
    "--model", "Qwen/Qwen2.5-1.5B",
    "--dataset", "fancyzhx/ag_news",
    "--nodes", "1",
    "--gpus-per-node", "1",
    "--profile-name", "local-single-node",
    "--max-steps", "1",
    "--dry-run",
]


def _hits(text: str) -> list[str]:
    low = text.lower()
    return [t for t in STREAM_TOKENS if t in low]


def _capture(root: Path, tree: Path) -> dict[str, Any]:
    """Call the shipped capture function, in a subprocess so `tree` sets sys.path.

    Run out-of-process on purpose: the two arms must import the SAME shipped
    module but ask about DIFFERENT roots, and doing that in one interpreter
    invites a cached module answering for the wrong tree.
    """
    code = (
        "import json,sys;"
        "sys.path.insert(0, sys.argv[1] + '/src');"
        "from foundationscale.provenance import capture_code_provenance as c;"
        "p = c(sys.argv[2]);"
        "print(json.dumps({'status': getattr(p.status,'value',str(p.status)),"
        "'commit': p.commit, 'dirty_files': p.dirty_files, 'root': p.root}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tree), str(root)],
        capture_output=True, text=True, timeout=300,
    )
    out = (proc.stdout or "").strip().splitlines()
    rec: dict[str, Any] = {"rc": proc.returncode, "raw": out[-1][:300] if out else ""}
    try:
        rec.update(json.loads(out[-1]))
    except Exception as exc:  # noqa: BLE001
        rec["parse_error"] = f"{type(exc).__name__}: {exc}"
        rec["stderr_tail"] = (proc.stderr or "").strip()[-300:]
    return rec


def _dry_run(tree: Path, out_dir: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "src")
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.run(
        [sys.executable, "-m", "foundationscale.train.cli",
         *BASE, "--output-dir", str(out_dir)],
        cwd=str(tree), capture_output=True, text=True, timeout=1800, env=env,
    )
    stream = (proc.stdout or "") + (proc.stderr or "")
    # The claim is about what the MANIFEST records, so read the shipped artifact
    # rather than trusting that the capture function and the artifact agree --
    # they are different code paths and the disagreement is itself scoreable.
    manifest_code: dict[str, Any] | None = None
    manifest_error = None
    man_path = out_dir / "run_manifest.json"
    try:
        manifest_code = json.loads(man_path.read_text()).get("code")
    except Exception as exc:  # noqa: BLE001
        manifest_error = f"{type(exc).__name__}: {exc}"
    return {
        "rc": proc.returncode,
        "stream_bytes": len(stream),
        "stream_lines": len(stream.splitlines()),
        "hits": _hits(stream),
        "tail": stream.strip().splitlines()[-1][:200] if stream.strip() else "",
        "manifest_path": str(man_path),
        "manifest_code": manifest_code,
        "manifest_error": manifest_error,
    }


def _inside_repo(path: Path) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(path), capture_output=True, text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="a real git checkout")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The copy is made with `git archive`, which emits the tracked tree and NO
    # .git at all -- closer to "somebody unpacked a tarball" than `cp -R` plus a
    # delete, and it cannot leave a stray .git fragment behind.
    copy = out / "copy_no_git"
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    tar = subprocess.run(
        f"git -C {repo} archive HEAD | tar -x -C {copy}",
        shell=True, capture_output=True, text=True,
    )
    if tar.returncode != 0:
        print(f"REFUSE: could not build the non-git copy: {tar.stderr[:200]}", file=sys.stderr)
        return 96

    payload: dict[str, Any] = {"controls": {}, "arms": {}}

    # The commit the manifest SHOULD carry, read independently of the framework.
    # Without it, a capture that returned one hard-coded sha would pass a 40-hex
    # check forever.
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    payload["controls"]["expected_head"] = {
        "rc": head.returncode,
        "commit": head.stdout.strip() or None,
    }
    payload["controls"]["capture_on_real_repo"] = _capture(repo, repo)
    payload["controls"]["copy_parent_not_repo"] = {
        "copy": str(copy),
        "inside_work_tree": _inside_repo(copy),
        "dot_git_present": (copy / ".git").exists(),
    }
    synthetic = "the run root is not_a_repository and no commit was captured"
    payload["controls"]["detector_selftest"] = {
        "line": synthetic,
        "hits": _hits(synthetic),
        "negative_line_hits": _hits("training step 1 loss 2.9"),
    }
    payload["controls"]["dry_run_on_real_repo"] = _dry_run(repo, out / "run_repo")

    payload["arms"]["capture_on_copy"] = _capture(copy, copy)
    payload["arms"]["dry_run_on_copy"] = _dry_run(copy, out / "run_copy")

    for section in ("controls", "arms"):
        for name, rec in payload[section].items():
            slim = {k: v for k, v in rec.items() if k not in ("raw", "tail")}
            print(f"[{section} {name}] {slim}", flush=True)

    dest = out / "t0_9_arms.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"payload: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
