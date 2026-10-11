"""Harness runners: the only place a benchmark is actually executed.

``SubprocessLmEvalRunner`` runs ``lm-eval run`` as a subprocess with an argv
list (never a shell string), one task per invocation so that one missing
dataset costs one benchmark rather than the whole suite. The environment is
forced offline and pointed at the eval cache. ``VlmHeldoutRunner`` scores VLM
held-out rows (image/video/text, video rows included) with FoundationScale's own
``examples/vlm-2xH200/scripts/eval_heldout_loss.py``, again as an argv list with
the inherited environment; exit 96 (the script refusing) is a REFUSED. Unit tests
inject any object with the same ``run(request) -> HarnessOutcome`` /
``-> HeldoutOutcome`` shape instead.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Protocol

PINNED_LM_EVAL = "0.4.12"
_CACHE_OVERRIDES = ("HF_DATASETS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")


def harness_version() -> str | None:
    try:
        return metadata.version("lm_eval")
    except metadata.PackageNotFoundError:
        return None


def offline_env(eval_cache: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Offline env rooted at the eval cache; per-cache overrides are dropped so HF_HOME is authoritative."""
    env = {k: v for k, v in dict(os.environ if base is None else base).items() if k not in _CACHE_OVERRIDES}
    env.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(eval_cache),
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


@dataclass(frozen=True)
class HarnessRequest:
    role: str  # "checkpoint" | "baseline"
    model_dir: str
    task: str
    output_path: str
    eval_cache: str
    adapter_dir: str | None = None
    tokenizer_dir: str | None = None
    num_fewshot: int | None = None
    seed: int = 1234
    dtype: str = "bfloat16"
    chat_template: bool = False
    fewshot_as_multiturn: bool = False
    system_instruction: str | None = None
    gen_kwargs: str | None = None
    limit: int | None = None
    batch_size: str | None = None
    device: str | None = None
    parallelize: bool = False
    trust_remote_code: bool = False
    include_path: str | None = None


@dataclass(frozen=True)
class HarnessOutcome:
    returncode: int
    results: dict[str, Any] | None
    argv: tuple[str, ...] = ()
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HarnessRunner(Protocol):
    name: str

    def run(self, request: HarnessRequest) -> HarnessOutcome: ...


def _model_args(req: HarnessRequest) -> str:
    for label, value in (("model", req.model_dir), ("adapter", req.adapter_dir), ("tokenizer", req.tokenizer_dir)):
        if value and ("," in value or "=" in value):
            raise ValueError(f"precondition failed: {label} path {value!r} contains ',' or '=' (breaks --model_args)")
    parts = [f"pretrained={req.model_dir}"]
    if req.adapter_dir:
        parts.append(f"peft={req.adapter_dir}")
    if req.tokenizer_dir:
        parts.append(f"tokenizer={req.tokenizer_dir}")
    parts.append(f"dtype={req.dtype}")
    if req.parallelize:
        parts.append("parallelize=True")
    if req.trust_remote_code:
        parts.append("trust_remote_code=True")
    return ",".join(parts)


def build_argv(req: HarnessRequest, executable: str = "lm-eval") -> list[str]:
    argv = [executable, "run", "--model", "hf", "--model_args", _model_args(req), "--tasks", req.task,
            "--seed", str(req.seed), "--output_path", req.output_path, "--log_samples"]
    if req.num_fewshot is not None:
        argv += ["--num_fewshot", str(req.num_fewshot)]
    if req.chat_template:
        argv.append("--apply_chat_template")
        if req.fewshot_as_multiturn:
            argv.append("--fewshot_as_multiturn")
    if req.system_instruction:
        argv += ["--system_instruction", req.system_instruction]
    if req.gen_kwargs:
        argv += ["--gen_kwargs", req.gen_kwargs]
    if req.limit is not None:
        argv += ["--limit", str(req.limit)]
    if req.batch_size:
        argv += ["--batch_size", req.batch_size]
    if req.device:
        argv += ["--device", req.device]
    if req.include_path:
        argv += ["--include_path", req.include_path]
    if req.trust_remote_code:
        argv.append("--trust_remote_code")
    return argv


def extract_metric(results: Mapping[str, Any] | None, task: str, metric: str) -> dict[str, Any] | None:
    """``{"score", "stderr", "n"}`` for ``task``/``metric``, or None when the harness did not report it."""
    per_task = ((results or {}).get("results") or {}).get(task)
    if not isinstance(per_task, dict):
        return None
    score = per_task.get(metric)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    name, _, filt = metric.partition(",")
    stderr = per_task.get(f"{name}_stderr,{filt}" if filt else f"{name}_stderr")
    stderr = float(stderr) if isinstance(stderr, (int, float)) and not isinstance(stderr, bool) else None
    counts = ((results or {}).get("n-samples") or {}).get(task) or {}
    n = counts.get("effective") if isinstance(counts, dict) else None
    return {"score": float(score), "stderr": stderr, "n": n}


def _newest_results(output_path: Path) -> Path | None:
    found = sorted(output_path.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


class SubprocessLmEvalRunner:
    name = "lm-eval"

    def __init__(self, executable: str | None = None, timeout_s: float | None = None) -> None:
        # The lm-eval next to this interpreter is the one EV-IN-004 version-checked; a bare
        # "lm-eval" resolves through PATH, which in a Slurm job may be a different install.
        sibling = Path(sys.executable).with_name("lm-eval")
        self.executable = executable or (str(sibling) if sibling.exists() else "lm-eval")
        self.timeout_s = timeout_s

    def run(self, request: HarnessRequest) -> HarnessOutcome:
        argv = build_argv(request, self.executable)
        out = Path(request.output_path)
        out.mkdir(parents=True, exist_ok=True)
        log = out / "harness.log"
        try:
            with log.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(argv, env=offline_env(request.eval_cache), stdout=fh,
                                      stderr=subprocess.STDOUT, text=True, timeout=self.timeout_s, check=False)
            code = proc.returncode
        except FileNotFoundError as exc:
            return HarnessOutcome(127, None, tuple(argv), f"harness executable not found: {exc}")
        except subprocess.TimeoutExpired:
            return HarnessOutcome(124, None, tuple(argv), f"harness timed out after {self.timeout_s}s; log {log}")
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        if code != 0:
            return HarnessOutcome(code, None, tuple(argv), "\n".join(tail) or f"exit {code}", {"log": str(log)})
        path = _newest_results(out)
        if path is None:
            return HarnessOutcome(code, None, tuple(argv), f"harness exited 0 but wrote no results_*.json under {out}",
                                  {"log": str(log)})
        return HarnessOutcome(code, json.loads(path.read_text(encoding="utf-8")), tuple(argv), None,
                              {"log": str(log), "results_path": str(path)})


# -- VLM held-out loss: FoundationScale's examples/vlm-2xH200/scripts/eval_heldout_loss.py --------

HELDOUT_DEFAULT_VIDEO_COLUMN = "video"


@dataclass(frozen=True)
class HeldoutRequest:
    """One held-out loss scoring run; every field mirrors an eval_heldout_loss.py flag.

    ``adapter`` marks a PEFT checkpoint (its base is the ``model``), ``video_frames`` is
    required as soon as the rows carry video, and an unset field means "not declared" --
    :func:`build_heldout_argv` refuses those instead of guessing.
    """
    model: str
    adapter: str | None = None
    data: str = ""
    max_length: int = 0
    video_frames: int | None = None
    image_column: str = "image"
    video_column: str = "video"
    limit: int | None = None
    out: str = ""
    script: str | None = None
    python: str | None = None


@dataclass(frozen=True)
class HeldoutOutcome:
    rc: int
    result: dict[str, Any] | None
    stderr_tail: str
    argv: tuple[str, ...] = ()


def default_heldout_script() -> Path | None:
    """Shipped eval_heldout_loss.py (two packages up from FoundationScale), or None without FoundationScale."""
    try:
        from importlib import import_module
        package = import_module("foundationscale")
    except Exception:  # FoundationScale is only needed on the held-out path
        return None
    module = getattr(package, "__file__", None)
    if not module:
        return None
    return Path(module).resolve().parents[2] / "examples" / "vlm-2xH200" / "scripts" / "eval_heldout_loss.py"


def build_heldout_argv(req: HeldoutRequest) -> list[str]:
    """eval_heldout_loss.py argv as a list -- never a shell string; flag order mirrors the script."""
    for label, value in (("model", req.model), ("data", req.data), ("out", req.out)):
        if not value:
            raise ValueError(f"missing input: held-out {label}")
    for label, value in (("model", req.model), ("adapter", req.adapter), ("data", req.data),
                         ("out", req.out), ("script", req.script)):
        if value and ("," in value or "=" in value):
            raise ValueError(f"precondition failed: {label} path {value!r} contains ',' or '=' (breaks the flag list)")
    if req.max_length < 1:
        raise ValueError(f"missing input: held-out max_length (got {req.max_length!r})")
    script = req.script or default_heldout_script()
    if script is None:
        raise ValueError("missing input: held-out script (pass heldout_script)")
    argv = [req.python or sys.executable, str(script), "--model", req.model]
    if req.adapter:
        argv += ["--adapter", req.adapter]
    argv += ["--data", req.data, "--max-length", str(req.max_length), "--image-column", req.image_column]
    if req.video_frames is not None:
        argv += ["--video-frames", str(req.video_frames)]
    argv += ["--video-column", req.video_column or HELDOUT_DEFAULT_VIDEO_COLUMN]
    if req.limit is not None:
        argv += ["--limit", str(req.limit)]
    argv += ["--out", req.out]
    return argv


def extract_heldout(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """rows_used/rows_dropped/nll/perplexity/tokens_scored of one role, or None when the result lacks the block."""
    doc = result or {}
    rows_used, rows_dropped, nll = doc.get("rows_used"), doc.get("rows_dropped"), doc.get("mean_nll_nats")
    if isinstance(rows_used, bool) or not isinstance(rows_used, int):
        return None
    if isinstance(rows_dropped, bool) or not isinstance(rows_dropped, int):
        return None
    if isinstance(nll, bool) or not isinstance(nll, (int, float)):
        return None

    def _opt(key: str) -> float | None:
        value = doc.get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    return {
        "rows_used": rows_used,
        "rows_dropped": rows_dropped,
        "nll": float(nll),
        "perplexity": _opt("perplexity"),
        "tokens_scored": _opt("tokens_scored"),
    }


class VlmHeldoutRunner:
    """Score one role (base model, or base + PEFT adapter) on the canonical held-out rows.

    The script inherits the environment and runs with ``cwd`` at the output
    directory; its result JSON at ``--out`` is the measurement. Exit 96 is the
    script refusing (video rows without ``--video-frames``, ...), which the skill
    maps to REFUSED. Tests inject any ``run(request) -> HeldoutOutcome`` object.
    """
    name = "fs-heldout"

    def __init__(self, timeout_s: float | None = None) -> None:
        self.timeout_s = timeout_s

    def run(self, request: HeldoutRequest) -> HeldoutOutcome:
        try:
            argv = build_heldout_argv(request)
        except ValueError as exc:
            return HeldoutOutcome(96, None, str(exc))
        out = Path(request.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(argv, cwd=str(out.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=self.timeout_s, check=False)
        except FileNotFoundError as exc:
            return HeldoutOutcome(127, None, f"held-out script or interpreter not found: {exc}", tuple(argv))
        except subprocess.TimeoutExpired:
            return HeldoutOutcome(124, None, f"held-out run timed out after {self.timeout_s}s", tuple(argv))
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        if proc.returncode != 0:
            return HeldoutOutcome(proc.returncode, None, tail or f"exit {proc.returncode}", tuple(argv))
        try:
            doc = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return HeldoutOutcome(3, None, f"held-out result {out} is not readable JSON ({exc})", tuple(argv))
        return HeldoutOutcome(0, doc if isinstance(doc, dict) else None, tail, tuple(argv))
