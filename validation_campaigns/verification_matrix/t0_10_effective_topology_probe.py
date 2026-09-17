#!/usr/bin/env python3
"""T0-10 on REAL multi-node hardware: is the effective topology READ, or echoed?

The row claims the framework derives the topology it actually got from runtime
evidence rather than repeating the declaration back. Until now that claim was
backed only by unit tests that set ``WORLD_SIZE`` in ``os.environ`` by hand --
which proves the parser, not the plane. This probe runs the SHIPPED entry point
(``python -m foundationscale.train.cli``) on four real Slurm/torchrun launches
across sixteen GPUs and asks the only question that can separate the two:

    when the declaration and the launch DISAGREE, which one does it report?

An implementation that echoed the config would sail through the disagreement
arm -- it would "agree" with itself -- so that arm is the deciding one and the
two truthful arms exist to prove the disagreement arm is not simply a broken
launch. Written as four arms because a single green run is compatible with a
framework that reports a constant.

    truthful_16    declared 4 nodes x 4 GPUs, launched with sixteen ranks.
                   Must train.
    truthful_4     declared 1 node x 4 GPUs, launched with four ranks on one
                   node. Must train, and must report a DIFFERENT topology from
                   the arm above -- otherwise the reading is a constant.
    misdeclared    declared 4 nodes x 4 GPUs, launched with EIGHT ranks. Must
                   refuse before touching a GPU and must name the override.
                   This is the deciding arm.
    no_torchrun    declared 4 x 4, launched with no torchrun at all. Must say
                   the comparison was SKIPPED rather than quietly agreeing --
                   an absent reading must not read as a matching one.

Nothing here decides the verdict; it records. The verdict is
``t0_10_effective_topology_arms.py``, which is a pure function of this payload.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("FS_MN_ROOT", "")).expanduser()
OUT = Path(os.environ.get("FS_MN_OUT", "")).expanduser()
JOBID = os.environ.get("FS_MN_JOBID", "")
PY = os.environ.get("FS_MN_PY", "")
MODEL = os.environ.get("FS_T1_MODEL", "")
DATASET = os.environ.get("FS_T1_DATASET", "")
MASTER = os.environ.get("FS_MN_MASTER", "")
MAX_STEPS = os.environ.get("FS_MN_MAX_STEPS", "20")
PORT = os.environ.get("FS_MN_PORT", "29517")

for name, value in (
    ("FS_MN_ROOT", ROOT),
    ("FS_MN_OUT", OUT),
    ("FS_MN_JOBID", JOBID),
    ("FS_MN_PY", PY),
    ("FS_T1_MODEL", MODEL),
    ("FS_T1_DATASET", DATASET),
    ("FS_MN_MASTER", MASTER),
):
    if not str(value):
        # A missing input is an UNMEASURED condition, and it is cheaper to say
        # so here than to discover it in the middle of a sixteen-GPU launch.
        print(f"UNMEASURED: {name} is unset")
        raise SystemExit(95)

MARKER = re.compile(r"^\[fs:[a-z0-9:_-]+\]")
# The validator's per-finding lines are CONTINUATIONS of the validated marker --
# they do not start with "[fs:", so the marker extractor above cannot see them.
# That matters here more than anywhere: the deciding arm's whole content is two
# [block] lines naming which degree was overridden, and a payload that recorded
# only "2/5 finding(s) block" would force the adjudicator to score a COUNT.
FINDING_LINE = re.compile(r"^\[(block|warn|ok)\]\s")
TIMEOUT_S = int(os.environ.get("FS_MN_TIMEOUT", "2400"))


def _markers(text: str) -> list[str]:
    """Every FoundationScale marker line, in order, stripped of trailing space."""
    return [ln.rstrip() for ln in text.splitlines() if MARKER.match(ln.strip())]


def _finding_lines(text: str) -> list[str]:
    """Deduped [block]/[warn]/[ok] lines, first-seen order.

    Deduped because every rank prints them: sixteen copies of one finding is
    the same one finding, and a scorer that counted would read rank count as
    severity.
    """
    seen: dict[str, None] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if FINDING_LINE.match(line):
            seen.setdefault(line, None)
    return list(seen)


def _config_values(manifest: dict | None, prefix: str) -> dict:
    """The manifest's config entries under a prefix, as {key: value-string}.

    ``config`` stores EffectiveValue RECORDS, not scalars, so a scanner that
    reads config[key] gets a dict and a scanner that greps for the key name
    finds it under a namespaced spelling (``extra.blocking``, not
    ``blocking``). Both of those cost a wrong answer on the first attempt here,
    so the payload records the unwrapped values and the adjudicator never has
    to know the record shape.
    """
    config = (manifest or {}).get("config")
    if not isinstance(config, dict):
        return {}
    out = {}
    for key, rec in config.items():
        if key.startswith(prefix):
            out[key] = rec.get("value") if isinstance(rec, dict) else rec
    return out


def _marker_selftest() -> dict:
    """Can the extractor above fire at all, and does it stay quiet otherwise?

    Without this, an arm that produced no markers is indistinguishable from an
    extractor that cannot match one. The positive is a real marker line copied
    from the shipped format; the negative is an ordinary log line that contains
    a bracket, because the cheap version of this regex matches that too.
    """
    positive = "[fs:train:consistency]  WORLD_SIZE=16: declared-vs-effective compared"
    negative = "  0%|          | 0/20 [00:00<?, ?it/s] [not a marker]"
    finding_positive = "[block] topology.effective_overrides_dp: declared dp=16 but the runtime"
    finding_negative = "Traceback (most recent call last): [block] not a finding line"
    return {
        "positive_hits": _markers(positive),
        "negative_hits": _markers(negative),
        "finding_positive_hits": _finding_lines(finding_positive),
        "finding_negative_hits": _finding_lines(finding_negative),
    }


def _cli_args(nodes: int, gpn: int, dp: int, outdir: Path, extra: list[str]) -> list[str]:
    return [
        "-m",
        "foundationscale.train.cli",
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--output-dir",
        str(outdir),
        "--nodes",
        str(nodes),
        "--gpus-per-node",
        str(gpn),
        "--dp",
        str(dp),
        "--profile-name",
        "slurm-generic",
        "--max-steps",
        MAX_STEPS,
        "--per-device-batch-size",
        "1",
        "--logging-steps",
        "5",
        "--save-interval",
        str(MAX_STEPS),
        "--seed",
        "1234",
        *extra,
    ]


def _run(
    name: str,
    *,
    srun_nodes: int,
    nnodes: int,
    nproc: int,
    declared_nodes: int,
    declared_gpn: int,
    declared_dp: int,
    use_torchrun: bool,
    extra: list[str] | None = None,
) -> dict:
    """One arm: a real srun inside the held allocation, output captured whole."""
    outdir = OUT / name
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    args = _cli_args(declared_nodes, declared_gpn, declared_dp, outdir / "run", extra or [])
    if use_torchrun:
        # The rendezvous host is derived from the STEP's own nodelist, not from
        # the job's. Two of these arms take a SUBSET of the held allocation, and
        # a master pinned to the job's first node is a host that subset may not
        # contain -- the launch would then hang on a rendezvous nobody is
        # listening for, and a hang is indistinguishable from the refusal this
        # probe is trying to observe.
        inner = (
            'M=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1); '
            f"{PY} -m torch.distributed.run --nnodes={nnodes} "
            f"--node_rank=$SLURM_NODEID --nproc_per_node={nproc} "
            f'--master_addr="$M" --master_port={PORT} -m ' + " ".join(args[1:])
        )
    else:
        inner = f"{PY} " + " ".join(args)
    cmd = [
        "srun",
        "--overlap",
        f"--jobid={JOBID}",
        f"--nodes={srun_nodes}",
        "--ntasks-per-node=1",
        "--cpus-per-task=32",
        "--gres=gpu:4",
        "bash",
        "-c",
        f"export PYTHONPATH={ROOT}/src; export TOKENIZERS_PARALLELISM=false; {inner}",
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
        rc, out = proc.returncode, proc.stdout + proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc, timed_out = None, True
        out = (exc.stdout or b"").decode("utf-8", "replace") + (exc.stderr or b"").decode(
            "utf-8", "replace"
        )
    elapsed = round(time.time() - started, 1)
    (outdir / "console.log").write_text(out, encoding="utf-8")

    manifest_path = outdir / "run" / "run_manifest.json"
    manifest: dict | None = None
    manifest_error: str | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001 -- unreadable is a fact to record
            manifest_error = f"{type(exc).__name__}: {exc}"
    ckpt_files = (
        sorted(p.name for p in (outdir / "run").rglob("*") if p.is_file())
        if (outdir / "run").exists()
        else []
    )
    marks = _markers(out)
    return {
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "declared": {"nodes": declared_nodes, "gpus_per_node": declared_gpn, "dp": declared_dp},
        "launched": {"nnodes": nnodes, "nproc_per_node": nproc, "world": nnodes * nproc}
        if use_torchrun
        else {"nnodes": None, "nproc_per_node": None, "world": None},
        "use_torchrun": use_torchrun,
        "lines": len(out.splitlines()),
        "markers": marks,
        "finding_lines": _finding_lines(out),
        "tail": out.splitlines()[-60:],
        "manifest_topology": (manifest or {}).get("topology"),
        # The artifact half of the claim. `topology` above is the DECLARED
        # shape; these are the config records that say where each degree came
        # from and, when a run is stopped, which findings stopped it.
        "manifest_topology_config": _config_values(manifest, "topology."),
        "manifest_stage": _config_values(manifest, "stage").get("stage"),
        "manifest_blocking": _config_values(manifest, "extra.blocking").get("extra.blocking"),
        "manifest_keys": sorted(manifest) if manifest else None,
        "manifest_error": manifest_error,
        "artifact_names": ckpt_files[:60],
        "artifact_count": len(ckpt_files),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "row": "T0-10",
        "subject_root": str(ROOT),
        "jobid": JOBID,
        "master": MASTER,
        "model": MODEL,
        "dataset": DATASET,
        "max_steps": MAX_STEPS,
        "arms": {},
        "controls": {"marker_selftest": _marker_selftest()},
    }
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    payload["subject_head"] = head.stdout.strip() or None

    # Order matters only for the operator's patience: the deciding control is
    # cheap (it must refuse before a GPU is touched), so it runs FIRST. If the
    # plane cannot tell a bad declaration from a good one, the two long arms
    # are not worth their wall clock.
    payload["controls"]["misdeclared_16_launched_8"] = _run(
        "misdeclared_16_launched_8",
        srun_nodes=2,
        nnodes=2,
        nproc=4,
        declared_nodes=4,
        declared_gpn=4,
        declared_dp=16,
        use_torchrun=True,
    )
    payload["controls"]["no_torchrun"] = _run(
        "no_torchrun",
        srun_nodes=1,
        nnodes=1,
        nproc=1,
        declared_nodes=4,
        declared_gpn=4,
        declared_dp=16,
        use_torchrun=False,
        extra=["--dry-run"],
    )
    payload["arms"]["truthful_4"] = _run(
        "truthful_4",
        srun_nodes=1,
        nnodes=1,
        nproc=4,
        declared_nodes=1,
        declared_gpn=4,
        declared_dp=4,
        use_torchrun=True,
    )
    payload["arms"]["truthful_16"] = _run(
        "truthful_16",
        srun_nodes=4,
        nnodes=4,
        nproc=4,
        declared_nodes=4,
        declared_gpn=4,
        declared_dp=16,
        use_torchrun=True,
    )

    dest = OUT / "t0_10_arms.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"payload -> {dest}")
    for section in ("arms", "controls"):
        for name, arm in payload[section].items():
            if not isinstance(arm, dict) or "rc" not in arm:
                continue
            print(
                f"  {section[:4]:5s} {name:28s} rc={arm['rc']} "
                f"lines={arm['lines']} artifacts={arm['artifact_count']} "
                f"{arm['elapsed_s']}s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
