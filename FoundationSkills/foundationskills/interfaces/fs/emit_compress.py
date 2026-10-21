"""Emit the fs_launch_spec of one RTN weight-only quantize-dequantize (A4.8): the ``rtn_worker.py`` measurement of a
FoundationSkills compression skill, probed first and rendered here, never executed here.

``emit_compress`` shapes the worker's CLI (min-max asym groupwise round-to-nearest dequantize with perplexity measure)
and names every gap that would keep the measurement from running: the model directory and its text corpus, bits and
group-size legality, a results path that must not be overwritten, ``torch`` and ``transformers`` in the interpreter that
would run it, and the vendored worker itself. ``preflight.py`` is vendored verbatim from the estate skill and serves
only as the capacity probe of the same one-GPU ptq cell.

The spec runs through ``fskills launch --spec S --confirm <hash>`` like every launch: its plan hash is printed by
``fskills hash`` for the user to type back, never echoed in a refusal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKER = Path(__file__).resolve().parents[2] / "skills" / "compression" / "rtn_worker.py"
PREFLIGHT = WORKER.parent / "preflight.py"
# The estate env-hygiene gate (preflight env_hygiene): user site-packages never leak into a measurement.
WORKER_ENV = {"PYTHONNOUSERSITE": "1"}


def sha256_file(path: Path) -> str | None:
    """The SHA-256 of a file's bytes as hex, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _nearest_existing(path: Path) -> Path | None:
    ancestor = path
    while not ancestor.exists():
        if ancestor == ancestor.parent:  # nothing above exists on this walk
            return None
        ancestor = ancestor.parent
    return ancestor


def import_gaps(python: str) -> list[str]:
    """``torch`` and ``transformers`` missing from the interpreter that would run the worker."""
    if python == sys.executable:
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        return [f"not importable in {python}: {', '.join(missing)}"] if missing else []
    try:
        probe = subprocess.run([python, "-c", "import torch, transformers"], timeout=120,
                               capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"import probe of {python} failed: {exc}"]
    return [] if probe.returncode == 0 else [
        f"torch/transformers are not importable in {python} (import probe exit {probe.returncode})"]


def compress_probes(model: Path, text_file: Path, bits: int, group_size: int, results: Path,
                    python: str = sys.executable) -> list[str]:
    """Every measured gap that keeps the compress measurement from running here; [] means it can run."""
    gaps: list[str] = []
    if not model.is_dir() or not (model / "config.json").is_file():
        gaps.append(f"model {model} is not a model directory (no config.json)")
    elif not any(model.glob("*.safetensors")):
        gaps.append(f"model {model} has no weight shards (*.safetensors)")
    try:
        empty = not text_file.is_file() or text_file.stat().st_size == 0
    except OSError:
        empty = True
    if empty:
        gaps.append(f"text file {text_file} is not a non-empty file")
    if bits not in (4, 8):
        gaps.append(f"bits {bits} is not one of (4, 8)")
    if not 32 <= group_size <= 1024 or group_size & (group_size - 1):
        gaps.append(f"group_size {group_size} is not a power of two within [32, 1024]")
    if results.exists():
        gaps.append(f"results {results} exists (a measurement is never overwritten)")
    else:
        ancestor = _nearest_existing(results.parent)  # the parent exists or can be created
        if ancestor is None or not os.access(ancestor, os.W_OK):
            gaps.append(f"results parent {results.parent} cannot be created (nothing writable above it)")
    gaps.extend(import_gaps(python))
    if not WORKER.is_file():
        gaps.append(f"rtn worker {WORKER} is not found")
    return gaps


def emit_compress(model: str, text_file: str, bits: int, group_size: int, budget_ppl_ratio: float,
                  results: str, ctx: int = 2048, max_eval_tokens: int = 65536,
                  exempt: str = "embed,lm_head,vision,audio", python: str = sys.executable) -> dict[str, Any]:
    """The compress fs_launch_spec (schema fs_launch_spec, entry ``fskills-compress``)."""
    m, t, r = Path(model).expanduser(), Path(text_file).expanduser(), Path(results).expanduser()
    gaps = compress_probes(m, t, bits, group_size, r, python)
    argv = [python, str(WORKER), "--model", str(m), "--text-file", str(t), "--bits", str(bits),
            "--group-size", str(group_size), "--ctx", str(ctx), "--max-eval-tokens", str(max_eval_tokens),
            "--exempt", exempt, "--results", str(r)]
    out_dir = str(r.parent)  # the spec schema types cwd as a string
    return {
        "stage_name": "compress", "entry": "fskills-compress", "argv": argv, "env": dict(WORKER_ENV),
        "sbatch": None, "dry_run_argv": None, "expected_outputs": [str(r)], "executable": not gaps,
        "missing": "; ".join(gaps) or None, "cwd": out_dir, "output_dir": out_dir,
        "compress": {"model": str(m), "text_file": str(t), "text_sha256": sha256_file(t),
                     "model_config_sha256": sha256_file(m / "config.json"), "bits": bits, "group_size": group_size,
                     "ctx": ctx, "max_eval_tokens": max_eval_tokens, "exempt": exempt.split(","),
                     "budget_ppl_ratio": budget_ppl_ratio, "results": str(r),
                     "method": "rtn_minmax_asym_groupwise", "efficiency": "UNMEASURED"},
    }


def preflight_argv(model: str, artifact_root: str, bits: int, python: str = sys.executable) -> list[str]:
    """The capacity-probe argv of ``preflight.py`` for the same one-GPU ptq cell this measurement would fill."""
    m, a = str(Path(model).expanduser()), str(Path(artifact_root).expanduser())
    argv = [python, str(PREFLIGHT), "--model", m, "--artifact-root", a, "--memory-profile", "ptq",
            "--gpus-per-cell", "1", "--out", str(Path(a) / "cap.json")]
    return argv + (["--planned-formats", "rtnste_int4"] if bits == 4 else [])  # no weight-only int8 token
