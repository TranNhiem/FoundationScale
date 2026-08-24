#!/usr/bin/env python3
"""Executable pre-flight blocklist for the FoundationScale-gated E4B launch.

Why this file exists
--------------------
The launch it gates is real: Gemma-4-E4B, GB200 tray (4 GPUs), LoRA and full
fine-tuning, under Megatron-Bridge. The design that this file operationalizes
(``docs/risk-review §4``, "What CLEAR must mean before the re-run") is ten
items of prose; prose cannot block a launch. Every item is implemented here
as a check that runs on the LOGIN NODE with no GPU and no torch import —
the only tensor facts required (names, shapes, dtypes) live in safetensors
headers, which are 8 length bytes plus JSON, readable with stdlib alone.

If a future check genuinely needs torch: import it INSIDE that check, and
treat ``ImportError`` as a BLOCK that names the check, never as a skip.
There is no such check today, and that absence is deliberate.

The clearance algebra (design item 4, the load-bearing one)
-----------------------------------------------------------
Reuse ``foundationscale.gates.core.Verdict`` and ``Coverage`` unchanged —
their members fit every state a pre-flight check can land in. What this file
does NOT reuse is the gate REPORT's tolerance for SKIP next to real passes.
Design item 4 states SKIP/VACUOUS/INAPPLICABLE ==> NOT-VERIFIED, so the
clearance predicate here is, verbatim:

    bool(results) and all(r.verdict is Verdict.PASS and r.coverage.checked > 0)

That is strictly stronger than ``Verdict.blocking``, on purpose: this tool's
only output that anyone acts on is the word CLEAR, and the framework's
founding incident is the word "pass" emitted over an empty examination.
Every non-PASS line renders with a (NOT-VERIFIED) tag so the design's
vocabulary survives contact with stdout.

Exit codes: 0 = CLEAR, 1 = BLOCKED, 2 = the tool itself could not run
(which is also not a clearance).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from foundationscale.gates.core import Coverage, Verdict
except ImportError:  # pragma: no cover - depends on invocation environment
    # Console-script installs ship foundationscale; a bare ``python
    # tools/preflight.py`` from a checkout may not have ``src`` on sys.path.
    # Bootstrap it rather than refuse: the tool must run from the login node
    # with zero installation ceremony. Path.resolve() plays abspath()'s role
    # here — a bare checkout invocation may hand us a relative __file__ — and
    # additionally follows a symlinked script back to the true checkout, which
    # is the tree whose src/ belongs on sys.path.
    _HERE = Path(__file__).resolve().parent
    _SRC = _HERE.parent / "src"
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))
        from foundationscale.gates.core import Coverage, Verdict
    else:  # foundationscale is a hard dependency; degrading to local enums
        # would silently fork Verdict's meaning, which requirement A forbids.
        raise

__all__ = ["main", "run_self_test", "REGISTRY"]

TOOL_VERSION = 1
EXIT_CLEAR = 0
EXIT_BLOCKED = 1
EXIT_TOOL_ERROR = 2

_CHUNK = 1 << 16

# Byte widths for safetensors dtype tags. Deliberately mirrors
# foundationscale.provenance.manifest._KNOWN_DTYPE_WIDTHS and
# gates.checkpoint_gates._DTYPE_BYTES: three small tables that must each
# stay honest beats an import edge that drags gate registration side
# effects onto a login node.
_SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

# Distinguishes "key absent" from "key present and set to None". A config that
# explicitly pins a value to null is making a statement; a config that omits the
# key is not, and collapsing the two would let an unset denominator read as a
# deliberate one.
_MISSING = object()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolError(RuntimeError):
    """The tool itself could not run (exit 2). Not a clearance, not a block: an abort."""


class ConfigError(RuntimeError):
    """The config file parsed but fails validation (exit 1, BLOCKED, keys named).

    Aggregates EVERY problem rather than raising on the first: an operator fixing
    one missing key per run discovers the last one after the launch window closed.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(f"config invalid: {len(problems)} problem(s)")


class ArtifactError(RuntimeError):
    """A declared artifact is missing or unreadable. Checks convert this to BLOCK."""


# ---------------------------------------------------------------------------
# Config schema + validation
# ---------------------------------------------------------------------------
# Validation is fail-closed in BOTH directions, matching the discipline of
# DeclaredCheckpoint.from_dict: missing keys are named, and unknown keys are
# named. A typo'd key a reader silently ignores is a denominator that never
# applied — the manifest's "num_expert (sic)" lesson, one tool down.


@dataclass(frozen=True)
class K:
    """Leaf spec: kind, whether the key is required, whether empty values are refused."""

    kind: str
    required: bool = True
    nonempty: bool = False
    default: Any = None


_SCHEMA: dict[str, Any] = {
    "run_name": K("str", nonempty=True),
    "world_size": K("int"),
    # -- item 1: frozen manifest ------------------------------------------------
    "frozen": {
        "model": {
            "files": K("list[str]", nonempty=True),
            "tensor_count": K("int"),
            "total_bytes": K("int"),
        },
        "corpus": {"files": K("list[dict]", nonempty=True)},
        "run_config": {"path": K("str", nonempty=True), "sha256": K("str", nonempty=True)},
    },
    # -- item 2: template audit ---------------------------------------------------
    "template": {
        "probe_command": K("list[str]", nonempty=True),
        "rows_per_file": K("int"),
        "files": K("list[str]", nonempty=True),
        "keep_cot_env": K("str", nonempty=True),
        "chat_template_path": K("str", nonempty=True),
        # md5 is RECORDED always (design item 2) but only COMPARED when pinned:
        # an unpinned md5 is a fact, not a claim, and doctrine 5 forbids
        # minting a check against a denominator nobody declared.
        "chat_template_md5": K("str", required=False, default=""),
    },
    # -- item 3: corpus wiring ----------------------------------------------------
    "corpus_wiring": {
        "env_var": K("str", nonempty=True),
        "recipe_files": K("list[str]", nonempty=True),
        "attestation_path": K("str", nonempty=True),
    },
    # -- item 5: conversion --------------------------------------------------------
    "conversion": {
        "hf_config_json": K("str", nonempty=True),
        "coverage_map_json": K("str", nonempty=True),
        "iter_metrics_jsonl": K("str", nonempty=True),
        "iter1_loss_band": K("list", nonempty=True),
        "expected_param_count": K("int"),
        "divisibility": K("list[dict]", nonempty=True),
        "tied_grounding": K("str", nonempty=True),
        "shared_kv_grounding": K("str|null"),
    },
    # -- item 6: LoRA probe ---------------------------------------------------------
    "lora": {
        "run_log": K("str", nonempty=True),
        "target_classes": K("list[str]", nonempty=True),
        "trainable_band": K("list", nonempty=True),
        "probe_metrics_jsonl": K("str", nonempty=True),
        "delta_audit_json": K("str", nonempty=True),
        "merged_dir": K("str", nonempty=True),
        "pinned_merged_total_bytes": K("int"),
        "expected_iters": K("int"),
    },
    # -- item 7: schedule ------------------------------------------------------------
    "schedule": {
        "train_iters": K("int"),
        "lr_decay_iters": K("int"),
        "save_interval": K("int"),
        "explicit_final_save": K("bool"),
        "smoke": K("bool"),
    },
    # -- item 8: evidence completeness ------------------------------------------------
    "evidence": {
        "log_glob": K("str", nonempty=True),
        "mem_regex": K("str", nonempty=True),
        "max_log_age_s": K("int"),
        # null job id is legitimate ONLY alongside a declared opt-out below —
        # the same declared-extension-point discipline as GateRegistry's
        # event_allow_empty: absence of Slurm must be stated, not silent.
        "slurm_job_id": K("str|null"),
        "allow_no_slurm": K("bool"),
        "slurm_absent_reason": K("str", required=False, default=""),
    },
    # -- item 9: dynamics -------------------------------------------------------------
    "dynamics": {
        "metrics_jsonl": K("str", nonempty=True),
        "bands": K("list[dict]", nonempty=True),
        "hard_floor": K("number", required=False, default=0.1),
    },
    # -- item 10: provenance ------------------------------------------------------------
    "provenance": {
        "checkpoint_dirs": K("list[str]", nonempty=True),
        "resume_guard_files": K("list[str]", nonempty=True),
        "min_walltime_s": K("number"),
        "job_window_utc": K("list", nonempty=True),
        "mtime_slack_s": K("int", required=False, default=5),
        "artifacts": K("list[str]", nonempty=True),
        "walltime_jsonl": K("str", nonempty=True),
    },
}

_KIND_CHECKS: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    # Underscored because it really is unexamined: "any" is the table's escape
    # hatch for a config slot whose kind is deliberately unconstrained, and the
    # parameter exists only to match the Callable[[Any], bool] shape every other
    # row obeys. Named `v`, it reads as a value someone forgot to check.
    "any": lambda _v: True,
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "str|null": lambda v: v is None or isinstance(v, str),
    "list[str]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "list[dict]": lambda v: isinstance(v, list) and all(isinstance(x, dict) for x in v),
}


def _walk_spec(
    spec: Mapping[str, Any], node: Any, path: str, problems: list[str], out: dict[str, Any]
) -> None:
    if not isinstance(node, dict):
        problems.append(f"{path or '<root>'}: must be an object, got {type(node).__name__}")
        return
    for key in sorted(set(node) - set(spec)):
        problems.append(
            f"{path + '.' if path else ''}{key}: unknown config key — a key this "
            f"tool cannot read is a pin it cannot enforce; fix the name or remove it"
        )
    for key, sub in spec.items():
        dotted = f"{path}.{key}" if path else key
        if key not in node:
            if isinstance(sub, K) and not sub.required:
                out[key] = sub.default
                continue
            problems.append(
                f"missing config key: {dotted!r} (fail closed: an absent pin is not a pass)"
            )
            continue
        value = node[key]
        if isinstance(sub, K):
            if not _KIND_CHECKS[sub.kind](value):
                problems.append(f"{dotted!r}: expected {sub.kind}, got {type(value).__name__}")
                continue
            if sub.nonempty and not value:
                problems.append(f"{dotted!r}: empty — a zero-length pin list is the vacuous case")
                continue
            if (
                sub.kind == "int"
                and dotted.endswith(
                    ("tensor_count", "expected_param_count", "expected_iters", "train_iters")
                )
                and value <= 0
            ):
                problems.append(f"{dotted!r}: must be a positive int, got {value!r}")
                continue
            out[key] = value
        else:
            child: dict[str, Any] = {}
            _walk_spec(sub, value, dotted, problems, child)
            out[key] = child


def _post_validate(cfg: Mapping[str, Any], problems: list[str]) -> None:
    """Composite shape rules the leaf walker cannot express. Same taxonomy, same exit."""

    fr = cfg.get("frozen", {})
    for i, entry in enumerate(fr.get("corpus", {}).get("files", []) or []):
        if not isinstance(entry, dict):
            continue  # leaf walker already named it
        base = f"frozen.corpus.files[{i}]"
        extra = sorted(set(entry) - {"path", "sha256", "lines"})
        if extra:
            problems.append(f"{base}: unknown keys {extra!r}")
        for name in ("path", "sha256", "lines"):
            if name not in entry:
                problems.append(f"missing config key: {base}.{name!r}")
        if "lines" in entry and (
            not isinstance(entry["lines"], int) or isinstance(entry["lines"], bool)
        ):
            problems.append(f"{base}.lines: expected int")

    conv = cfg.get("conversion", {})
    band = conv.get("iter1_loss_band")
    if isinstance(band, list) and band:
        if len(band) != 2 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in band
        ):
            problems.append("conversion.iter1_loss_band: expected [lo, hi] numbers")
        elif band[0] > band[1]:
            problems.append(
                f"conversion.iter1_loss_band: lo {band[0]} > hi {band[1]} — "
                f"an inverted band passes everything"
            )
    for i, assertion in enumerate(conv.get("divisibility", []) or []):
        if not isinstance(assertion, dict):
            continue
        base = f"conversion.divisibility[{i}]"
        if not isinstance(assertion.get("field"), str) or not assertion.get("field"):
            problems.append(f"{base}.field: required non-empty dotted key into the HF config")
        forms = [k for k in ("divisible_by", "equals") if k in assertion]
        if len(forms) != 1:
            problems.append(f"{base}: exactly one of 'divisible_by'/'equals' is required")
        elif forms[0] == "divisible_by" and (
            not isinstance(assertion["divisible_by"], int)
            or isinstance(assertion["divisible_by"], bool)
            or assertion["divisible_by"] <= 0
        ):
            problems.append(f"{base}.divisible_by: must be a positive int")

    lora = cfg.get("lora", {})
    tb = lora.get("trainable_band")
    if isinstance(tb, list) and tb:
        if len(tb) != 2 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in tb
        ):
            problems.append("lora.trainable_band: expected [lo, hi] numbers")
        elif tb[0] > tb[1]:
            problems.append("lora.trainable_band: lo > hi")

    dyn = cfg.get("dynamics", {})
    for i, b in enumerate(dyn.get("bands", []) or []):
        if not isinstance(b, dict):
            continue
        base = f"dynamics.bands[{i}]"
        for name in ("iteration", "lo", "hi"):
            if name not in b:
                problems.append(f"missing config key: {base}.{name!r}")
        if "iteration" in b and (
            not isinstance(b["iteration"], int)
            or isinstance(b["iteration"], bool)
            or b["iteration"] < 1
        ):
            problems.append(f"{base}.iteration: must be an int >= 1")
        if (
            "lo" in b
            and "hi" in b
            and isinstance(b["lo"], (int, float))
            and isinstance(b["hi"], (int, float))
            and b["lo"] > b["hi"]
        ):
            problems.append(f"{base}: lo {b['lo']} > hi {b['hi']}")

    ev = cfg.get("evidence", {})
    if (
        ev.get("slurm_job_id") is None
        and ev.get("allow_no_slurm") is True
        and not str(ev.get("slurm_absent_reason", "")).strip()
    ):
        problems.append(
            "evidence.slurm_absent_reason: required when allow_no_slurm is true — "
            "the declared opt-out must say WHY, or it is a silent hole"
        )

    prov = cfg.get("provenance", {})
    window = prov.get("job_window_utc")
    if isinstance(window, list) and window:
        parsed = [_parse_iso(w) if isinstance(w, str) else None for w in window]
        if len(window) != 2 or any(p is None for p in parsed):
            problems.append(
                "provenance.job_window_utc: expected [start_iso, end_iso] ISO-8601 strings"
            )
        elif parsed[0] >= parsed[1]:
            problems.append(
                "provenance.job_window_utc: start must precede end — "
                "an empty window contains every artifact or none"
            )


def _load_config(path: str) -> tuple[dict[str, Any], str]:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        # Unreadable config: the TOOL could not run. Exit 2, not 1 — a block
        # verdict over checks that never executed would be a claim without evidence.
        raise ToolError(f"config unreadable: {p}: {exc}") from exc
    cfg_sha = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolError(f"config at {p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(["<root>: config must be a JSON object"])
    problems: list[str] = []
    out: dict[str, Any] = {}
    _walk_spec(_SCHEMA, data, "", problems, out)
    _post_validate(out, problems)
    if not out.get("frozen", {}).get("model") or not out.get("frozen", {}).get("corpus"):
        # Structural restatement of requirement E's "a config that pins nothing":
        # with neither a model pin nor a corpus pin, nothing downstream has a
        # denominator and every check would abstain against fiction.
        problems.append(
            "config pins nothing: frozen.model and frozen.corpus are both absent — "
            "BLOCK, by doctrine 1"
        )
    if problems:
        raise ConfigError(problems)
    return out, cfg_sha


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    """Streamed sha256 and a wc -l line count (count of b'\\n'), one pass.

    wc -l counts newline characters, not "lines"; a trailing unterminated line
    is invisible to both wc and to this function — the contract states so rather
    than disagreeing with the reference tool it replaces.
    """
    h = hashlib.sha256()
    lines = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    return h.hexdigest(), lines


def _read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a safetensors header with stdlib only: 8 LE length bytes + JSON.

    Returns {tensor_name: {dtype, shape, numel}} with __metadata__ excluded.
    Any deviation raises ArtifactError -> the caller BLOCKS; a shard whose
    format we cannot price is not a shard we clear.

    Chaining contract, load-bearing for _check_frozen_manifest: an OS-level
    refusal (open/read) is re-raised with the originating OSError chained as
    __cause__; a bytes-level format defect (truncated length prefix, bad JSON,
    malformed shape, unpriced dtype) raises bare, with __cause__ None. That is
    the only reliable separator between "the environment refused the read"
    (ERROR, fail closed — the operator goes to the machine) and "the artifact
    is corrupt" (a FAIL the check exists to name — the operator goes to the
    checkpoint), and the two demand opposite responses.
    """
    try:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ArtifactError(f"{path}: shorter than a safetensors length prefix")
            (n,) = struct.unpack("<Q", raw)
            if n > 512 * 1024 * 1024:
                raise ArtifactError(
                    f"{path}: header claims {n} bytes — implausible, refusing to buffer it"
                )
            payload = fh.read(n)
            if len(payload) != n:
                raise ArtifactError(f"{path}: truncated header ({len(payload)}/{n} bytes)")
    except OSError as exc:
        raise ArtifactError(f"{path}: unreadable: {exc}") from exc
    try:
        meta = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # `from None`, not chained: this function's docstring chaining contract
        # reserves an OSError __cause__ for environmental refusals (frozen_manifest
        # classifies on exactly that), and the decode error's text already rides
        # inside the message, so discarding the exception object costs no evidence.
        raise ArtifactError(
            f"{path}: header is not JSON ({exc}); cannot count tensors — BLOCK, not guess"
        ) from None
    if not isinstance(meta, dict):
        raise ArtifactError(f"{path}: header JSON is not an object")
    out: dict[str, dict[str, Any]] = {}
    for name, entry in meta.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ArtifactError(f"{path}: tensor {name!r} entry is not an object")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise ArtifactError(f"{path}: tensor {name!r} has a malformed shape {shape!r}")
        numel = 1
        for d in shape:
            numel *= d
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ArtifactError(
                f"{path}: dtype {dtype!r} has no known byte width; extend "
                f"_SAFETENSORS_DTYPE_BYTES — arithmetic over an unpriced dtype is a false number"
            )
        out[str(name)] = {"dtype": str(dtype), "shape": shape, "numel": numel}
    return out


def _canonical_sample_sha256(path: Path) -> str:
    """Hash of the batch-0 sample as a human would decode it: first JSONL row, canonicalized."""

    try:
        with path.open("r", encoding="utf-8") as fh:
            line = fh.readline()
    except OSError as exc:
        raise ArtifactError(f"{path}: unreadable while decoding batch-0 sample: {exc}") from exc
    if not line.strip():
        raise ArtifactError(f"{path}: first line is empty — there is no batch-0 sample to read")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: first line does not decode as JSON: {exc}") from exc
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _parse_iso(text: str) -> _dt.datetime | None:
    try:
        dt = _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _manifest_hash_for(frozen: Mapping[str, Any], cfg_sha: str) -> tuple[str, dict[str, Any]]:
    """One canonical payload + hash, used by BOTH the check and the fixture world.

    A single source of truth is load-bearing: launch_provenance ties checkpoint
    provenance records to *this* hash, and the self-test world must compute the
    identical value or the fixture would prove nothing about the real equality.
    """
    payload = {
        "schema": 1,
        "config_sha256": cfg_sha,
        "model": {
            "files": list(frozen["model"]["files"]),
            "tensor_count": frozen["model"]["tensor_count"],
            "total_bytes": frozen["model"]["total_bytes"],
        },
        "corpus": [
            {"path": f["path"], "sha256": f["sha256"], "lines": f["lines"]}
            for f in frozen["corpus"]["files"]
        ],
        "run_config": dict(frozen["run_config"]),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload


# ---------------------------------------------------------------------------
# Check results + the clearance algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    title: str
    verdict: Verdict
    coverage: Coverage
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def blocking(self) -> bool:
        # For CLEAR purposes every non-PASS blocks (see clearance predicate), but
        # this property mirrors core semantics for report anatomy.
        return self.verdict is not Verdict.PASS

    def render(self) -> str:
        line = f"[{self.verdict.symbol:>7}] {self.check_id}: {self.coverage}"
        if self.detail:
            line += f" — {self.detail}"
        if self.verdict is not Verdict.PASS:
            # Design item 4's vocabulary, verbatim, on every non-clearing line:
            # a reader must never have to know that SKIP is 'non-blocking' in
            # gate-land to understand that this launch is not cleared.
            line += "  (NOT-VERIFIED)"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "title": self.title,
            "verdict": self.verdict.value,
            "checked": self.coverage.checked,
            "expected": self.coverage.expected,
            "unit": self.coverage.unit,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "duration_s": round(self.duration_s, 4),
        }


def _finalize(
    check_id: str,
    title: str,
    verdict: Verdict,
    coverage: Coverage,
    detail: str,
    evidence: Mapping[str, Any] | None,
) -> CheckResult:
    """The Gate.ok downgrade ladder, restated for checks.

    Same order, same semantics: a requested PASS over zero units becomes
    VACUOUS; short-of-denominator becomes UNDERCOVERED; numerator outrunning
    the denominator becomes OVERCOVERED. The check author cannot override any
    of the three, which — as in Gate.ok — is the point.
    """
    evidence = dict(evidence or {})
    if verdict is Verdict.PASS:
        if coverage.is_vacuous:
            return CheckResult(
                check_id,
                title,
                Verdict.VACUOUS,
                coverage,
                f"check examined 0 {coverage.unit} and therefore proves nothing"
                + (f" (claimed: {detail})" if detail else ""),
                evidence,
            )
        if coverage.is_short and not coverage.sampled:
            return CheckResult(
                check_id,
                title,
                Verdict.UNDERCOVERED,
                coverage,
                f"examined {coverage.checked} of {coverage.expected} {coverage.unit}"
                + (f" (claimed: {detail})" if detail else ""),
                evidence,
            )
        if coverage.is_over:
            return CheckResult(
                check_id,
                title,
                Verdict.OVERCOVERED,
                coverage,
                f"examined {coverage.checked} of {coverage.expected} {coverage.unit} — "
                f"the numerator outruns the denominator; one of them is wrong"
                + (f" (claimed: {detail})" if detail else ""),
                evidence,
            )
    return CheckResult(check_id, title, verdict, coverage, detail, evidence)


def _discipline(res: CheckResult) -> CheckResult:
    """PASS with an EMPTY evidence map is not a PASS — design item 4 made the
    evidence payload part of the record shape, and an absent one is the claim
    'we counted and hashed' detached from every count and hash. Downgrade to
    ERROR: it is an author defect in this file, not a property of the run."""
    if res.verdict is Verdict.PASS and not res.evidence:
        return CheckResult(
            res.check_id,
            res.title,
            Verdict.ERROR,
            res.coverage,
            "check returned PASS with an empty evidence map — the verdict schema "
            "requires counts/hashes/files attached to every clearance",
            {"author_defect": True},
            res.duration_s,
        )
    return res


def _is_clear(results: Sequence[CheckResult]) -> bool:
    # The single sentence this whole file exists to enforce.
    return bool(results) and all(
        r.verdict is Verdict.PASS and r.coverage.checked > 0 for r in results
    )


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------


@dataclass
class _Lane:
    """One MUST_FIRE control: apply() corrupts a fresh fixture world; the check
    must then NOT return PASS. description names the injected defect."""

    name: str
    description: str
    apply: Callable[[_World], None]


@dataclass
class _Check:
    id: str
    title: str
    section: str | None  # config section handed to fn; None for meta checks
    fn: Callable[[dict, dict, dict, dict, Sequence[_Check] | None], CheckResult]
    lanes: tuple[_Lane, ...] = ()


_REGISTRY_ORDER: list[_Check] = []
REGISTRY: dict[str, _Check] = {}


def _register(
    check_id: str,
    title: str,
    section: str | None,
    lanes: Sequence[_Lane] = (),
) -> Callable[[Callable[..., CheckResult]], Callable[..., CheckResult]]:
    def deco(fn: Callable[..., CheckResult]) -> Callable[..., CheckResult]:
        if check_id in REGISTRY:
            raise ValueError(f"duplicate preflight check id {check_id!r}")
        chk = _Check(check_id, title, section, fn, tuple(lanes))
        REGISTRY[check_id] = chk
        _REGISTRY_ORDER.append(chk)
        return fn

    return deco


def _execute(
    chk: _Check,
    cfg: dict,
    env: Mapping[str, str],
    shared: dict,
    registry: Sequence[_Check] | None = None,
) -> CheckResult:
    """One check, timed, fail-closed. Mirrors Gate.run's conversion discipline:
    an exception inside a check is ERROR and blocks; a check that returns
    anything but a CheckResult is ERROR and blocks."""
    t0 = time.perf_counter()
    try:
        section = cfg.get(chk.section, {}) if chk.section else {}
        res = chk.fn(cfg, section, env, shared, registry)
        if not isinstance(res, CheckResult):
            res = CheckResult(
                chk.id,
                chk.title,
                Verdict.ERROR,
                Coverage.none("units"),
                f"check returned {type(res).__name__}, expected CheckResult",
            )
    except Exception as exc:  # noqa: BLE001 — fail closed, deliberately broad
        res = CheckResult(
            chk.id,
            chk.title,
            Verdict.ERROR,
            Coverage.none("units"),
            f"{type(exc).__name__}: {exc}",
            {"traceback": traceback.format_exc(limit=12)},
        )
    res = _discipline(res)
    return CheckResult(
        res.check_id,
        res.title,
        res.verdict,
        res.coverage,
        res.detail,
        res.evidence,
        time.perf_counter() - t0,
    )


def _shared_or_error(chk: _Check, shared: dict, need: Sequence[str]) -> CheckResult | None:
    """Cross-check denominators (manifest hash, pinned corpus order) exist only
    if frozen_manifest ran and passed. Unwired shared state BLOCKS the consumer
    with the cause named — doctrine 4 applied to the tool's own plumbing."""
    missing = [k for k in need if k not in shared]
    if missing:
        return CheckResult(
            chk.id,
            chk.title,
            Verdict.ERROR,
            Coverage.none("units"),
            f"frozen_manifest did not establish {missing!r}; downstream checks "
            f"may not source denominators from anywhere else (design item 1)",
        )
    return None


# ---------------------------------------------------------------------------
# Item 1 — frozen manifest
# ---------------------------------------------------------------------------


def _check_frozen_manifest(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 1: the run's denominators, frozen and hashed into one manifest.

    Verifies, against PINS IN THE CONFIG (nowhere else — item 1's final
    sentence), that: every declared model file exists with the pinned st_size;
    the safetensors headers across all model files contain exactly the pinned
    tensor count; every declared corpus file's streamed sha256 and wc-l line
    count match; the run config's sha256 matches. Then computes THE manifest
    hash that the banner, launch_provenance and any later checkpoint bind to.

    INPUT CONTRACT: generic (design text only). Model files MUST be
    safetensors — anything else is a BLOCK naming the file, never a guess at
    its tensor count.
    """
    expected_files = len(s["model"]["files"]) + len(s["corpus"]["files"]) + 1
    evidence: dict[str, Any] = {}
    checked = 0

    # Three failure classes reach this loop and only two verdict vocabularies
    # were previously used, so one class was misfiled. They are kept distinct:
    #   * absent or unparseable shard — a defect of the ARTIFACT under
    #     clearance: a FAIL this check exists to NAME ("your checkpoint is
    #     missing a shard" sends the operator to the artifact);
    #   * an OS-level refusal (EACCES, EIO, a file vanishing mid-sweep) — a
    #     defect of the CHECK's ability to run: ERROR, fail closed ("the
    #     preflight cannot read this filesystem" sends them to the machine).
    #     The model-file-missing MUST_FIRE lane caught the conflation by
    #     rightly demanding the NAMED verdict, not merely any blocking one.
    # stat() orders the evidence: FileNotFoundError is absence; any other
    # OSError is environmental; once stat succeeds, an ArtifactError from
    # _read_safetensors_header WITHOUT a chained OSError cause is a bytes-level
    # defect, while one carrying an OSError __cause__ is an open/read refusal
    # that arrived (or a file that vanished) after stat succeeded — still
    # environmental, because the filesystem moved under the check mid-sweep.
    mismatches: list[str] = []
    tensor_total = 0
    param_total = 0
    model_files_detail = []
    model_unexamined: list[dict[str, Any]] = []
    for rel in s["model"]["files"]:
        p = Path(rel)
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            # Counted as a mismatch, never as examined: 'checked' stays honest
            # AND 'expected' does not move — a FAIL that read 6 of 7 declared
            # artifacts says "6/7", it does not re-target its denominator.
            mismatches.append(
                f"declared model file absent: {rel} — the shard the frozen "
                f"manifest pins is not on disk"
            )
            model_unexamined.append({"path": rel, "state": "absent"})
            continue
        except OSError as exc:
            return _finalize(
                "frozen_manifest",
                "Frozen manifest",
                Verdict.ERROR,
                Coverage(checked, "artifact files", expected=expected_files),
                f"model artifact unreadable (environmental): {rel}: {exc}",
                evidence,
            )
        try:
            header = _read_safetensors_header(p)
        except ArtifactError as exc:
            if isinstance(exc.__cause__, OSError):
                return _finalize(
                    "frozen_manifest",
                    "Frozen manifest",
                    Verdict.ERROR,
                    Coverage(checked, "artifact files", expected=expected_files),
                    f"model artifact unreadable (environmental): {rel}: {exc}",
                    evidence,
                )
            mismatches.append(f"model shard corrupt or unparseable as safetensors: {exc}")
            model_unexamined.append({"path": rel, "state": "corrupt", "problem": str(exc)})
            # st_size is knowable even when the header won't parse; recording
            # the bytes keeps sum(st_size) literally true, while the tensor
            # pricing below (correctly) never sees this shard.
            model_files_detail.append(
                {"path": rel, "bytes": size, "tensors": None, "state": "corrupt"}
            )
            continue
        tensor_total += len(header)
        param_total += sum(t["numel"] for t in header.values())
        model_files_detail.append({"path": rel, "bytes": size, "tensors": len(header)})
        checked += 1
    observed_bytes = sum(d["bytes"] for d in model_files_detail)
    evidence["model"] = {
        "files": model_files_detail,
        "tensors_observed": tensor_total,
        "tensors_pinned": s["model"]["tensor_count"],
        "bytes_observed": observed_bytes,
        "bytes_pinned": s["model"]["total_bytes"],
        "header_param_sum": param_total,
        # Files declared but never priced: absent/corrupt shards land here so
        # the FAIL evidence names exactly what was NOT examined — 'checked'
        # counts only fully parsed units, so this list is where the under-count
        # stays legible to both humans and JSON consumers.
        "unexamined": model_unexamined,
    }
    if tensor_total != s["model"]["tensor_count"]:
        mismatches.append(f"tensor count {tensor_total} != pinned {s['model']['tensor_count']}")
    if observed_bytes != s["model"]["total_bytes"]:
        mismatches.append(f"sum(st_size) {observed_bytes} != pinned {s['model']['total_bytes']}")

    corpus_detail = []
    for entry in s["corpus"]["files"]:
        p = Path(entry["path"])
        try:
            sha, lines = _sha256_and_lines(p)
        except OSError as exc:
            return _finalize(
                "frozen_manifest",
                "Frozen manifest",
                Verdict.ERROR,
                Coverage(checked, "artifact files", expected=expected_files),
                f"corpus file unreadable: {entry['path']}: {exc}",
                evidence,
            )
        corpus_detail.append(
            {
                "path": entry["path"],
                "sha256": sha,
                "lines": lines,
                "sha_matches_pin": sha == entry["sha256"],
                "lines_match_pin": lines == entry["lines"],
            }
        )
        if sha != entry["sha256"]:
            mismatches.append(
                f"corpus sha256 mismatch on {entry['path']} "
                f"(pinned {entry['sha256'][:12]}…, observed {sha[:12]}…)"
            )
        if lines != entry["lines"]:
            mismatches.append(
                f"corpus line count {lines} != pinned {entry['lines']} on {entry['path']}"
            )
        checked += 1
    evidence["corpus"] = {"files": corpus_detail, "count": len(corpus_detail)}

    rc_path = Path(s["run_config"]["path"])
    try:
        rc_sha = hashlib.sha256(rc_path.read_bytes()).hexdigest()
    except OSError as exc:
        return _finalize(
            "frozen_manifest",
            "Frozen manifest",
            Verdict.ERROR,
            Coverage(checked, "artifact files", expected=expected_files),
            f"run config unreadable: {rc_path}: {exc}",
            evidence,
        )
    checked += 1
    evidence["run_config"] = {
        "path": str(rc_path),
        "sha256": rc_sha,
        "matches_pin": rc_sha == s["run_config"]["sha256"],
    }
    if rc_sha != s["run_config"]["sha256"]:
        mismatches.append(
            "run config sha256 mismatch — the recipe being launched is not "
            "the recipe that was frozen"
        )

    manifest_sha, payload = _manifest_hash_for(s, shared["_config_sha256"])
    evidence["manifest_sha256"] = manifest_sha
    evidence["config_sha256"] = shared["_config_sha256"]
    # Publish the run's ONLY sanctioned denominator source.
    shared["manifest_sha256"] = manifest_sha
    shared["manifest_payload"] = payload
    shared["corpus_files"] = [f["path"] for f in s["corpus"]["files"]]

    cov = Coverage(checked, "artifact files", expected=expected_files)
    if mismatches:
        return _finalize(
            "frozen_manifest",
            "Frozen manifest",
            Verdict.FAIL,
            cov,
            "; ".join(mismatches[:4]) + (" …" if len(mismatches) > 4 else ""),
            evidence,
        )
    return _finalize(
        "frozen_manifest",
        "Frozen manifest",
        Verdict.PASS,
        cov,
        f"all pins verified; manifest_sha256={manifest_sha[:16]}…",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 2 — template audit (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_template_audit(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 2 (CPU): the chat template must keep CoT inside the masked span.

    INPUT CONTRACT (unverified against the FoxBrain repo — see 'What this fix
    does NOT close'):
      * template.probe_command: argv; the substrings '{file}' and '{rows}' are
        substituted. Ran once per declared file via subprocess. It must print
        EXACTLY rows_per_file JSON lines, each:
            {"row": int, "tokens_stock": int, "tokens_patched": int,
             "cot_span": [start, end], "masked_span": [start, end]}
        spans are token-index half-open ranges into that row's encoding.
      * template.files: the corpus JSONL files probed (8 rows each in the
        design; rows_per_file makes the count explicit).
      * env[template.keep_cot_env] (design: FOXBRAIN_GEMMA4_KEEP_COT): must be
        set and != "0".
      * template.chat_template_path: chat_template.jinja; md5 always recorded,
        compared when template.chat_template_md5 is pinned.

    Assertions per design: masked_span ⊇ cot_span for EVERY row examined;
    stock-vs-patched token-count diff computed for every row (both numbers
    required — a row missing a variant means the diff was never measured over
    it); row count exactly rows_per_file × len(files) (a probe that returns
    fewer rows than asked produced an under-denominator audit); KEEP_COT pin.
    """
    cov_unit = "rows"
    expected = int(s["rows_per_file"]) * len(s["files"])
    evidence: dict[str, Any] = {"files": [], "env_var": s["keep_cot_env"]}

    keep = env.get(s["keep_cot_env"])
    if keep is None:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.ERROR,
            Coverage.none(cov_unit),
            f"environment variable {s['keep_cot_env']} is not set — "
            f"the design asserts it ≠ 0; absent is not ≠ 0",
            evidence,
        )
    evidence["keep_cot_value"] = keep
    if keep == "0":
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.FAIL,
            Coverage.none(cov_unit),
            f"{s['keep_cot_env']}=0: CoT is being dropped from supervision",
            evidence,
        )

    tpl = Path(s["chat_template_path"])
    try:
        md5 = hashlib.md5(tpl.read_bytes()).hexdigest()
    except OSError as exc:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.ERROR,
            Coverage.none(cov_unit),
            f"chat template unreadable: {tpl}: {exc}",
            evidence,
        )
    evidence["chat_template"] = {
        "path": str(tpl),
        "md5": md5,
        "pinned_md5": s.get("chat_template_md5") or None,
    }
    if s.get("chat_template_md5") and md5 != s["chat_template_md5"]:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.FAIL,
            Coverage.none(cov_unit),
            f"chat_template.jinja md5 {md5} != pinned {s['chat_template_md5']} "
            f"— the template under test is not the template that was reviewed",
            evidence,
        )

    checked = 0
    containment_violations: list[str] = []
    rows_with_diff = 0
    for file_path in s["files"]:
        argv = [
            part.replace("{file}", str(file_path)).replace("{rows}", str(s["rows_per_file"]))
            for part in s["probe_command"]
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=300, env={**os.environ, **dict(env)}
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _finalize(
                "template_audit",
                "Template audit",
                Verdict.ERROR,
                Coverage(checked, cov_unit, expected=expected),
                f"template probe could not run ({exc}); the audit was not performed",
                evidence,
            )
        if proc.returncode != 0:
            return _finalize(
                "template_audit",
                "Template audit",
                Verdict.ERROR,
                Coverage(checked, cov_unit, expected=expected),
                f"template probe exited {proc.returncode}: {proc.stderr.strip()[:200]}",
                evidence,
            )
        rows = []
        for line_no, line in enumerate(proc.stdout.splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return _finalize(
                    "template_audit",
                    "Template audit",
                    Verdict.ERROR,
                    Coverage(checked, cov_unit, expected=expected),
                    f"probe row {line_no} of {file_path} is not JSON: {exc}",
                    evidence,
                )
        file_rec = {"path": file_path, "rows_returned": len(rows)}
        if len(rows) != int(s["rows_per_file"]):
            return _finalize(
                "template_audit",
                "Template audit",
                Verdict.FAIL,
                Coverage(checked, cov_unit, expected=expected),
                f"probe returned {len(rows)} rows for {file_path}; "
                f"{s['rows_per_file']} were asked for — a short audit is an under-covered claim",
                {**evidence, "files": evidence["files"] + [file_rec]},
            )
        for row in rows:
            try:
                m_lo, m_hi = row["masked_span"]
                c_lo, c_hi = row["cot_span"]
                t_stock = int(row["tokens_stock"])
                t_patched = int(row["tokens_patched"])
            except (KeyError, TypeError, ValueError) as exc:
                return _finalize(
                    "template_audit",
                    "Template audit",
                    Verdict.ERROR,
                    Coverage(checked, cov_unit, expected=expected),
                    f"probe row missing contract fields ({exc!r}) in {file_path}",
                    evidence,
                )
            checked += 1
            if t_stock != t_patched:
                rows_with_diff += 1
            if not (m_lo <= c_lo and c_hi <= m_hi):
                containment_violations.append(
                    f"{file_path} row {row.get('row', '?')}: cot_span "
                    f"[{c_lo},{c_hi}) escapes masked_span [{m_lo},{m_hi})"
                )
        evidence["files"].append(file_rec)

    evidence["rows_with_stock_vs_patched_diff"] = rows_with_diff
    evidence["rows_examined"] = checked
    cov = Coverage(checked, cov_unit, expected=expected)
    if containment_violations:
        return _finalize(
            "template_audit",
            "Template audit",
            Verdict.FAIL,
            cov,
            "; ".join(containment_violations[:3])
            + (
                f" (+{len(containment_violations) - 3} more)"
                if len(containment_violations) > 3
                else ""
            ),
            evidence,
        )
    return _finalize(
        "template_audit",
        "Template audit",
        Verdict.PASS,
        cov,
        f"CoT span inside masked span for {checked}/{expected} rows; "
        f"KEEP_COT={keep}; {rows_with_diff}/{checked} rows show a stock-vs-patched token diff",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 3 — corpus wiring (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_corpus_wiring(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 3: the corpus the launcher EXPORTS is the corpus the manifest PINNED.

    INPUT CONTRACT (unverified against the FoxBrain repo):
      * env[corpus_wiring.env_var] (design: FOXBRAIN_SFT_JSONLS): os.pathsep-
        separated list of JSONL paths IN TRAINING ORDER; the first entry is
        the batch-0 source. Empty entries refused.
      * corpus_wiring.recipe_files: source files of the recipe entrypoint; at
        least one must textually contain a call to ``_env_jsonls(`` — this is
        a grep, declared as such: it proves the responsible code path names
        the env reader, not that the call is live on every branch. A human
        must confirm that before first launch (see 'does NOT close').
      * corpus_wiring.attestation_path: JSON {"reader": str, "sample_sha256":
        hex, "note": str?}. The sample hash is MACHINE-VERIFIED: we decode the
        first row of the first resolved corpus file ourselves and require
        equality — an attestation over a different sample, or a file that
        changed since, FAILS. What no machine can verify is that 'reader' is
        a human who paid attention; the artifact pins WHO asserted it.
    """
    err = _shared_or_error(_Check("corpus_wiring", "", None, _stub_fn), shared, ["corpus_files"])
    if err:
        return err

    evidence: dict[str, Any] = {}
    raw = env.get(s["env_var"])
    if raw is None:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage.none("corpus files"),
            f"--export does not carry {s['env_var']}: the corpus list is not in "
            f"the launch environment",
            evidence,
        )
    resolved = [part for part in raw.split(os.pathsep) if part.strip()]
    if not resolved:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage.none("corpus files"),
            f"{s['env_var']} is set but empty — a zero-file corpus list is the vacuous case",
            evidence,
        )
    pinned = list(shared["corpus_files"])
    evidence["resolved_files"] = resolved
    evidence["pinned_files"] = pinned
    expected = len(pinned)
    if resolved != pinned:
        # Name the drift, not just its cardinality: 'resolved 5 != frozen 4'
        # makes an operator diff two path lists by eye. Naming the phantom
        # entry is the difference between a verdict and a hint, and the
        # repository's own review rule applies to failure strings too: every
        # claim carries its denominator AND names its offenders.
        extras = [r for r in resolved if r not in pinned]
        dropped = [f for f in pinned if f not in resolved]
        named = []
        if extras:
            named.append(f"resolved names files the manifest never pinned: {extras[:3]}")
        if dropped:
            named.append(f"pinned files missing from the resolved list: {dropped[:3]}")
        if not named:
            # Identical sets, different order: batch-0 is entry zero of the
            # resolved list, so ORDER is part of the corpus pin, not a detail.
            named.append(
                "the same files in a different order — order is pinned (batch-0 is entry 0)"
            )
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len([r for r in resolved if r in pinned]), "corpus files", expected=expected),
            f"resolved {s['env_var']} ({len(resolved)} files, in order) != frozen manifest corpus "
            f"({len(pinned)} files): "
            + "; ".join(named)
            + " — the banner would not match the manifest",
            evidence,
        )

    recipe_hits: dict[str, int] = {}
    for path in s["recipe_files"]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return _finalize(
                "corpus_wiring",
                "Corpus wiring",
                Verdict.ERROR,
                Coverage(len(resolved), "corpus files", expected=expected),
                f"recipe source unreadable: {path}: {exc}",
                evidence,
            )
        recipe_hits[path] = text.count("_env_jsonls(")
    evidence["recipe_files_examined"] = len(recipe_hits)
    evidence["recipe_hits"] = recipe_hits
    if not any(recipe_hits.values()):
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len(resolved), "corpus files", expected=expected),
            f"no call to _env_jsonls( found in {len(recipe_hits)} declared recipe files — "
            f"the recipe is not proven to read {s['env_var']}",
            evidence,
        )

    attest_path = Path(s["attestation_path"])
    try:
        attestation = json.loads(attest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage(len(resolved), "corpus files", expected=expected),
            f"batch-0 human attestation unreadable: {attest_path}: {exc}",
            evidence,
        )
    reader = str(attestation.get("reader", "")).strip()
    claimed = str(attestation.get("sample_sha256", ""))
    evidence["attestation"] = {
        "reader": reader,
        "claimed_sample_sha256": claimed,
        "path": str(attest_path),
    }
    if not reader:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len(resolved), "corpus files", expected=expected),
            "attestation names no reader — a claim of human review with no human named",
            evidence,
        )
    try:
        actual = _canonical_sample_sha256(Path(resolved[0]))
    except ArtifactError as exc:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage(len(resolved), "corpus files", expected=expected),
            str(exc),
            evidence,
        )
    evidence["attestation"]["recomputed_sample_sha256"] = actual
    if claimed != actual:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len(resolved), "corpus files", expected=expected),
            f"attestation sample hash {claimed[:12]}… != recomputed {actual[:12]}… over "
            f"{resolved[0]} — either the sample read was not batch-0 of this corpus, or the "
            f"file changed since; both void the human-review claim",
            evidence,
        )

    return _finalize(
        "corpus_wiring",
        "Corpus wiring",
        Verdict.PASS,
        Coverage(len(resolved), "corpus files", expected=expected),
        f"env export matches manifest {len(resolved)}/{expected}; _env_jsonls( wired in "
        f"{sum(1 for v in recipe_hits.values() if v)}/{len(recipe_hits)} recipe files; "
        f"batch-0 sample attested by {reader} and machine-confirmed",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 4 — verdict schema, operationalized as the launch-time red team
# ---------------------------------------------------------------------------


def _check_verdict_schema(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 4: 'corrupt-artifact red-team dry run must flip gates before any launch.'

    Not a data check: this re-runs EVERY peer check's MUST_FIRE lanes against
    fresh synthesized corrupt worlds, right now, on this login node, inside
    this clearance. A peer whose lane does not flip FAILS this check and the
    launch. A peer that declares zero lanes FAILS this check by name — shipping
    no positive control is disqualifying whether or not the check passes on
    the happy path (--self-test enforces the MUST_PASS half offline; this
    check enforces the MUST_FIRE half at every launch).

    'Flipped' means verdict.blocking AND verdict is not ERROR: an ERROR says
    the detector died on the corrupt input, which is a verifier exception,
    not a demonstrated firing — doctrine 4 binds detectors too.
    """
    peers = [
        c
        for c in (registry if registry is not None else _REGISTRY_ORDER)
        if c.id != "verdict_schema"
    ]
    total_lanes = sum(len(c.lanes) for c in peers)
    evidence: dict[str, Any] = {
        "peers_examined": len(peers),
        "lanes_total": total_lanes,
        "lane_results": [],
    }
    cov = Coverage(0, "red-team lanes", expected=total_lanes if total_lanes else None)
    if not peers:
        return _finalize(
            "verdict_schema",
            "Verdict schema / red team",
            Verdict.ERROR,
            cov,
            "no peer checks to red-team — a registry of one meta-check verifies nothing",
            evidence,
        )

    unflipped: list[str] = []
    laneless: list[str] = []
    checked = 0
    for peer in peers:
        if not peer.lanes:
            laneless.append(peer.id)
            continue
        for lane in peer.lanes:
            checked += 1
            outcome_verdict = "ERROR"
            note = ""
            try:
                res = _run_lane_against(peer, lane)
                outcome_verdict = res.verdict.value
            except Exception as exc:  # noqa: BLE001 — a broken red-team lane is unproven, not green
                note = f"lane harness raised {type(exc).__name__}: {exc}"
            flipped = outcome_verdict not in (
                Verdict.PASS.value,
                Verdict.SKIP.value,
                Verdict.ERROR.value,
            )
            evidence["lane_results"].append(
                {
                    "check": peer.id,
                    "lane": lane.name,
                    "defect": lane.description,
                    "verdict": outcome_verdict,
                    "flipped": flipped,
                    **({"note": note} if note else {}),
                }
            )
            if not flipped:
                unflipped.append(f"{peer.id}/{lane.name} (got {outcome_verdict})")

    evidence["peers_with_lanes"] = len(peers) - len(laneless)
    cov = Coverage(checked, "red-team lanes", expected=total_lanes)
    if laneless:
        return _finalize(
            "verdict_schema",
            "Verdict schema / red team",
            Verdict.FAIL,
            cov,
            f"checks shipping NO MUST_FIRE lane: {', '.join(laneless)} — a check that has "
            f"never been shown to fire is not evidence of anything",
            evidence,
        )
    if unflipped:
        return _finalize(
            "verdict_schema",
            "Verdict schema / red team",
            Verdict.FAIL,
            cov,
            f"{len(unflipped)}/{total_lanes} red-team lanes did NOT flip: "
            + "; ".join(unflipped[:5])
            + (" …" if len(unflipped) > 5 else ""),
            evidence,
        )
    return _finalize(
        "verdict_schema",
        "Verdict schema / red team",
        Verdict.PASS,
        cov,
        f"all {checked}/{total_lanes} corrupt-artifact lanes flipped their gates "
        f"across {len(peers)} peer checks",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 5 — conversion coverage (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_conversion_coverage(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 5: every one of the pinned tensors is converted or EXPLICITLY allow-listed.

    INPUT CONTRACT (unverified against the FoxBrain / Megatron-Bridge repo):
      * frozen.model.files (safetensors) supply the tensor NAMESPACE via
        headers; the count must re-equal the manifest pin here too (denomina-
        tors source from the manifest, but re-deriving them at the point of
        use is the proof they were not swapped).
      * conversion.coverage_map_json: {"tensors": [...]} produced by the
        Megatron-Bridge conversion probe, one entry per name:
          {"name": str, "bytes": int, "coverage": "converted"}             — or —
          {"name": str, "coverage": "allowlist", "rule": "tied"|"shared_kv"}
        'converted' names must exist in the headers with EQUAL stored bytes.
        'allowlist' names may be absent from headers (they share storage) but
        their rule must be GROUNDED: 'tied' requires the HF config key
        conversion.tied_grounding (e.g. tie_word_embeddings) == true;
        'shared_kv' requires conversion.shared_kv_grounding (the E4B key a
        human must supply; null ⇒ any shared_kv entry FAILS). Unknown rules
        FAIL: an allow-list entry nobody can ground is an unexamined tensor
        wearing permission.
      * conversion.hf_config_json: E4B config.json. conversion.divisibility is
        the design's "settles TP-divisibility, MoE-block pattern, EP=4
        divisibility": a list of {"field": dotted.key, "divisible_by": N} or
        {"field": ..., "equals": X} assertions, each evaluated here.
      * conversion.iter_metrics_jsonl: JSONL records; the iteration==1 record
        must carry "loss" within conversion.iter1_loss_band (design pins
        ≈[1.0, 3.0]); exactly one record must carry "param_count", and it must
        equal conversion.expected_param_count (the pinned ~8.0e9 arithmetic).
    """
    evidence: dict[str, Any] = {}

    # -- namespace from headers ------------------------------------------------
    headers: dict[str, dict[str, Any]] = {}
    for rel in cfg["frozen"]["model"]["files"]:
        try:
            for name, meta in _read_safetensors_header(Path(rel)).items():
                headers[name] = meta
        except ArtifactError as exc:
            return _finalize(
                "conversion_coverage",
                "Conversion coverage",
                Verdict.ERROR,
                Coverage.none("tensors"),
                str(exc),
                evidence,
            )
    expected = len(headers)
    pinned = cfg["frozen"]["model"]["tensor_count"]
    evidence["header_tensor_count"] = expected
    evidence["pinned_tensor_count"] = pinned
    if expected != pinned:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            Coverage(0, "tensors", expected=pinned),
            f"safetensors headers name {expected} tensors; the frozen manifest pins "
            f"{pinned} — the model under test is not the model that was frozen",
            evidence,
        )

    # -- HF config facts ---------------------------------------------------------
    hf_path = Path(s["hf_config_json"])
    try:
        hf = json.loads(hf_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"HF config unreadable: {hf_path}: {exc}",
            evidence,
        )
    if not isinstance(hf, dict):
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"{hf_path} is not a JSON object",
            evidence,
        )

    def dotted(key: str) -> Any:
        node: Any = hf
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return _MISSING
            node = node[part]
        return node

    _MISSING_LOCAL = _MISSING
    assertion_failures: list[str] = []
    assertion_count = 0
    for assertion in s["divisibility"]:
        assertion_count += 1
        value = dotted(assertion["field"])
        if value is _MISSING_LOCAL:
            assertion_failures.append(
                f"HF config has no key {assertion['field']!r} — assertion unevaluable"
            )
            continue
        if "divisible_by" in assertion:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value % assertion["divisible_by"] != 0
            ):
                assertion_failures.append(
                    f"{assertion['field']}={value!r} is not divisible by "
                    f"{assertion['divisible_by']}"
                )
        else:
            if value != assertion["equals"]:
                assertion_failures.append(
                    f"{assertion['field']}={value!r} != required {assertion['equals']!r}"
                )
    evidence["divisibility_assertions"] = {"count": assertion_count, "failures": assertion_failures}

    # -- coverage map -------------------------------------------------------------
    cm_path = Path(s["coverage_map_json"])
    try:
        coverage_map = json.loads(cm_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"coverage map unreadable: {cm_path}: {exc}",
            evidence,
        )
    entries = coverage_map.get("tensors") if isinstance(coverage_map, dict) else None
    if not isinstance(entries, list):
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"{cm_path} carries no 'tensors' list — the map covers nothing",
            evidence,
        )

    covered: set[str] = set()
    allowlist_count = 0
    uncovered: list[str] = []
    problems: list[str] = []
    by_name: dict[str, dict] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("name"), str):
            by_name[e["name"]] = e
    for name in by_name:
        if name not in headers and by_name[name].get("coverage") == "converted":
            problems.append(
                f"map converts {name!r}, which no model header declares — "
                f"numerator outruns denominator"
            )
    for name, meta in headers.items():
        e = by_name.get(name)
        if e is None:
            uncovered.append(name)
            continue
        if e.get("coverage") != "converted":
            uncovered.append(f"{name} (map entry is {e.get('coverage')!r}, not 'converted')")
            continue
        want = meta["numel"] * _SAFETENSORS_DTYPE_BYTES[meta["dtype"]]
        if e.get("bytes") != want:
            problems.append(f"{name}: map claims {e.get('bytes')} bytes; header implies {want}")
            continue
        covered.add(name)
    for name, e in by_name.items():
        if e.get("coverage") != "allowlist":
            continue
        rule = e.get("rule")
        allowlist_count += 1
        if rule == "tied":
            ground = dotted(s["tied_grounding"])
            if ground is not True:
                problems.append(
                    f"allow-listed {name!r} under rule 'tied', but HF config "
                    f"{s['tied_grounding']!r} is {ground!r} — the ground is absent or false"
                )
        elif rule == "shared_kv":
            key = s.get("shared_kv_grounding")
            ground = dotted(key) if key else _MISSING_LOCAL
            if ground is not True:
                problems.append(
                    f"allow-listed {name!r} under rule 'shared_kv', but grounding key "
                    f"{key!r} is {None if key is None else ground!r} — "
                    f"declare and confirm the E4B key"
                )
        else:
            problems.append(
                f"allow-listed {name!r} under unknown rule {rule!r} — no ground exists for it"
            )
    evidence["converted"] = len(covered)
    evidence["allowlisted"] = allowlist_count
    evidence["uncovered_count"] = len(uncovered)
    evidence["uncovered_sample"] = uncovered[:8]

    # -- iter-1 band + param echo ---------------------------------------------------
    try:
        metrics = [
            json.loads(line)
            for line in Path(s["iter_metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(len(covered), "tensors", expected=expected),
            f"iter metrics unreadable: {s['iter_metrics_jsonl']}: {exc}",
            evidence,
        )
    evidence["metrics_records"] = len(metrics)
    iter1 = [r for r in metrics if isinstance(r, dict) and r.get("iteration") == 1]
    if not iter1:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            Coverage(len(covered), "tensors", expected=expected),
            f"no iteration==1 record in {len(metrics)} metric rows — the loss band was "
            f"never evaluated, and absent evidence is not in-band evidence",
            evidence,
        )
    lo, hi = s["iter1_loss_band"]
    loss1 = iter1[0].get("loss")
    evidence["iter1_loss"] = loss1
    evidence["iter1_band"] = [lo, hi]
    if not isinstance(loss1, (int, float)) or isinstance(loss1, bool):
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(len(covered), "tensors", expected=expected),
            f"iter-1 loss is not numeric: {loss1!r}",
            evidence,
        )
    if len(iter1) > 1:
        problems.append(
            f"{len(iter1)} records claim iteration==1 — a duplicated band row contradicts itself"
        )
    echoes = [r["param_count"] for r in metrics if isinstance(r, dict) and "param_count" in r]
    evidence["param_echoes"] = echoes
    if not echoes:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            Coverage(len(covered), "tensors", expected=expected),
            f"no param_count echo in {len(metrics)} metric rows — the ~8.0e9 pin was "
            f"never compared against the trainer's own count",
            evidence,
        )
    if len(set(echoes)) != 1:
        problems.append(
            f"param_count echo is self-contradictory across rows: {sorted(set(echoes))!r}"
        )

    band_ok = lo <= loss1 <= hi
    param_ok = len(set(echoes)) == 1 and echoes[0] == s["expected_param_count"]
    cov = Coverage(len(covered), "tensors", expected=expected)
    reasons = []
    if uncovered:
        reasons.append(
            f"{len(uncovered)} of {expected} header tensors are neither converted nor allow-listed"
        )
    if problems:
        reasons.append("; ".join(problems[:3]) + (" …" if len(problems) > 3 else ""))
    if assertion_failures:
        reasons.append(
            f"{len(assertion_failures)}/{assertion_count} divisibility assertions failed"
        )
    if not band_ok:
        reasons.append(f"iter-1 loss {loss1} outside pinned band [{lo}, {hi}]")
    if not param_ok and not problems:
        reasons.append(f"param echo {echoes[0]} != pinned {s['expected_param_count']}")
    if reasons:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            cov,
            " | ".join(reasons),
            evidence,
        )
    return _finalize(
        "conversion_coverage",
        "Conversion coverage",
        Verdict.PASS,
        cov,
        f"{len(covered)}/{expected} tensors converted ({allowlist_count} grounded allow-list "
        f"entries); iter-1 loss {loss1} in [{lo}, {hi}]; param echo == {s['expected_param_count']}",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 6 — LoRA probe (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_lora_probe(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 6: the 20-iter LoRA probe actually attached, trained, and merged.

    INPUT CONTRACT (unverified against the FoxBrain repo):
      * lora.run_log: text log of the probe run. Must contain >0 lines
        matching 'Adding lora to', and >0 such lines naming EACH class in
        lora.target_classes (substring). Must contain a trainable-params line
        matching r'trainable[^\\n]*?params?[^\\n]*?([0-9][0-9,]*)' whose value
        sits inside lora.trainable_band.
      * lora.probe_metrics_jsonl: JSONL {"iteration": i, ...}; iterations must
        be exactly 1..lora.expected_iters complete; the max-iteration record
        must carry "lora_b_norm": {class: float} with every target class > 0.
      * lora.delta_audit_json: {class: {"delta_l2": >0, "tensors_checked":
        int>=1}} for EVERY target class — the merged Δ-audit, with its own
        per-class denominator.
      * lora.merged_dir: merged HF export. Parity = sum of st_size over every
        regular file beneath it MUST equal lora.pinned_merged_total_bytes.
        NEVER the dir's own index.json metadata_size/total_size: a self-index
        is the artifact attesting about itself, and the design says so in so
        many words.
    """
    evidence: dict[str, Any] = {}
    classes = list(s["target_classes"])
    expected = len(classes)

    try:
        log_text = Path(s["run_log"]).read_text(encoding="utf-8")
    except OSError as exc:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage.none("LoRA target classes"),
            f"run log unreadable: {s['run_log']}: {exc}",
            evidence,
        )
    lora_lines = [ln for ln in log_text.splitlines() if "Adding lora to" in ln]
    evidence["adding_lora_lines"] = len(lora_lines)
    if not lora_lines:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage.none("LoRA target classes"),
            "grep -c 'Adding lora to' == 0 — LoRA attached to nothing",
            evidence,
        )
    per_class_lines = {c: sum(1 for ln in lora_lines if c in ln) for c in classes}
    evidence["per_class_attach_lines"] = per_class_lines
    silent = [c for c, n in per_class_lines.items() if n == 0]
    if silent:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected - len(silent), "LoRA target classes", expected=expected),
            f"zero 'Adding lora to' lines name intended classes: {silent} — attached somewhere "
            f"else, or nowhere",
            evidence,
        )

    m = re.search(r"trainable[^\n]*?params?[^\n]*?([0-9][0-9,]*)", log_text)
    if not m:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            "no trainable-params line in the run log — the band was never evaluated",
            evidence,
        )
    trainable = int(m.group(1).replace(",", ""))
    lo, hi = s["trainable_band"]
    evidence["trainable_params"] = trainable
    evidence["trainable_band"] = [lo, hi]
    if not lo <= trainable <= hi:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"trainable params {trainable:,} outside band [{lo:,}, {hi:,}] — adapters are "
            f"not the size the layout implies",
            evidence,
        )

    try:
        probe_rows = [
            json.loads(ln)
            for ln in Path(s["probe_metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"probe metrics unreadable: {s['probe_metrics_jsonl']}: {exc}",
            evidence,
        )
    iters = sorted(
        {
            int(r["iteration"])
            for r in probe_rows
            if isinstance(r, dict) and isinstance(r.get("iteration"), int)
        }
    )
    evidence["probe_iterations_observed"] = iters
    if iters != list(range(1, int(s["expected_iters"]) + 1)):
        missing = sorted(set(range(1, int(s["expected_iters"]) + 1)) - set(iters))[:5]
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"probe ran iterations {iters[:3]}…{iters[-3:] if iters else ''} ({len(iters)} of "
            f"{s['expected_iters']}); missing e.g. {missing} — a {s['expected_iters']}-iter "
            f"claim over {len(iters)} iters is under-covered",
            evidence,
        )
    final_rec = max(
        (r for r in probe_rows if isinstance(r, dict) and isinstance(r.get("iteration"), int)),
        key=lambda r: r["iteration"],
    )
    bnorms = final_rec.get("lora_b_norm") if isinstance(final_rec, dict) else None
    if not isinstance(bnorms, dict):
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"final probe record (iter {final_rec.get('iteration')}) carries no lora_b_norm map "
            f"— B>0 was asserted, never measured",
            evidence,
        )
    evidence["lora_b_norm"] = bnorms
    b_bad = [
        c
        for c in classes
        if not isinstance(bnorms.get(c), (int, float))
        or isinstance(bnorms.get(c), bool)
        or bnorms.get(c) <= 0
    ]
    if b_bad:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected - len(b_bad), "LoRA target classes", expected=expected),
            f"lora_b_norm not > 0 for classes {b_bad} — zero-init survived the probe; the "
            f"B>0 assertion exists because exactly this once shipped silently",
            evidence,
        )

    try:
        delta = json.loads(Path(s["delta_audit_json"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"delta audit unreadable: {s['delta_audit_json']}: {exc}",
            evidence,
        )
    evidence["delta_audit"] = delta if isinstance(delta, dict) else {}
    d_bad = []
    for c in classes:
        rec = delta.get(c) if isinstance(delta, dict) else None
        if (
            not isinstance(rec, dict)
            or not isinstance(rec.get("delta_l2"), (int, float))
            or rec.get("delta_l2", 0) <= 0
        ):
            d_bad.append(f"{c} (delta_l2 missing or <= 0)")
        elif not isinstance(rec.get("tensors_checked"), int) or rec.get("tensors_checked", 0) < 1:
            d_bad.append(
                f"{c} (tensors_checked denominator missing or 0 — an unqualified 'Δ nonzero')"
            )
    if d_bad:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected - len(d_bad), "LoRA target classes", expected=expected),
            f"merged Δ-audit failed for: {', '.join(d_bad)}",
            evidence,
        )

    merged = Path(s["merged_dir"])
    if not merged.is_dir():
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"merged dir not present: {merged}",
            evidence,
        )
    observed = 0
    n_files = 0
    for dirpath, _dirnames, filenames in os.walk(merged):
        for fn in filenames:
            p = Path(dirpath) / fn
            # Deliberately stat ONLY: any model.safetensors.index.json in this
            # walk is summed as BYTES-ON-DISK, never parsed for its claimed
            # total_size. Design: parity is vs. the pinned 14.89 GiB — never
            # vs. self-index.
            observed += p.stat().st_size
            n_files += 1
    evidence["merged"] = {
        "dir": str(merged),
        "files": n_files,
        "observed_bytes": observed,
        "pinned_total_bytes": s["pinned_merged_total_bytes"],
        "comparison_source": "external pin (sum of st_size), never self-index",
    }
    if observed != int(s["pinned_merged_total_bytes"]):
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"merged HF bytes {observed:,} != pinned {int(s['pinned_merged_total_bytes']):,} "
            f"across {n_files} files",
            evidence,
        )

    return _finalize(
        "lora_probe",
        "LoRA probe",
        Verdict.PASS,
        Coverage(expected, "LoRA target classes", expected=expected),
        f"{expected}/{expected} classes attached, B>0, Δ>0; {len(iters)}/"
        f"{s['expected_iters']} iters; trainable {trainable:,} in band; merged "
        f"{observed:,} B == pin",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 7 — schedule
# ---------------------------------------------------------------------------


def _check_schedule(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 7: the banner's schedule invariants, arithmetic only. 2 invariants:
    (a) train_iters == lr_decay_iters; (b) train_iters % save_interval == 0,
    pardoned ONLY by an explicit_final_save declaration — the pardon is
    surfaced in evidence, never silently folded into a pass."""
    ti, ldi, si = int(s["train_iters"]), int(s["lr_decay_iters"]), int(s["save_interval"])
    evidence = {
        "train_iters": ti,
        "lr_decay_iters": ldi,
        "save_interval": si,
        "explicit_final_save": s["explicit_final_save"],
        "smoke": s["smoke"],
    }
    checked = 0
    bad = []
    checked += 1  # invariant (a)
    if ti != ldi:
        bad.append(
            f"train_iters {ti} != lr_decay_iters {ldi} — "
            f"the LR schedule ends before/after training does"
        )
    checked += 1  # invariant (b)
    if si <= 0:
        bad.append(f"save_interval {si} <= 0")
    elif ti % si != 0:
        if s["explicit_final_save"]:
            evidence["final_save_pardon"] = (
                f"{ti} % {si} == {ti % si}; pardoned by explicit_final_save=true — "
                f"declared, not silent"
            )
        else:
            bad.append(
                f"train_iters {ti} %% save_interval {si} == {ti % si} "
                f"and explicit_final_save is false — "
                f"the final state is never written"
            )
    if bad:
        return _finalize(
            "schedule_consistency",
            "Schedule banner",
            Verdict.FAIL,
            Coverage(checked - len(bad), "schedule invariants", expected=2),
            " | ".join(bad),
            evidence,
        )
    return _finalize(
        "schedule_consistency",
        "Schedule banner",
        Verdict.PASS,
        Coverage(checked, "schedule invariants", expected=2),
        "train_iters == lr_decay_iters; save cadence lands on the final step"
        + (" (via declared explicit final save)" if evidence.get("final_save_pardon") else "")
        + ("; SMOKE run — banner will carry the qualifier" if s["smoke"] else ""),
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 8 — evidence completeness
# ---------------------------------------------------------------------------


def _check_evidence_completeness(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 8: the evidence the other checks consumed is complete and LIVE.

    Contract: evidence.log_glob resolves to exactly world_size per-rank logs;
    every log parses evidence.mem_regex (one capture group, MiB, float) ≥1
    time; no log's mtime is older than evidence.max_log_age_s; if
    evidence.slurm_job_id is set, `bash -lc 'sacct …'` (item 8's wrapping
    discipline is satisfied BY CONSTRUCTION: the only Slurm query this file
    ever makes is that literal argv) must corroborate the job; a null job id
    requires the declared allow_no_slurm opt-out, recorded loudly.
    """
    world_size = int(cfg["world_size"])
    evidence: dict[str, Any] = {"world_size": world_size}
    # glob.glob, deliberately: log_glob is an operator-supplied PATTERN (an
    # absolute pattern in the real config), while Path.glob roots a relative
    # pattern at a base dir and refuses absolute ones — a rewrite would change
    # which rank logs this check enumerates. The sorted() order is itself part
    # of the denominator, so it stays exactly where it is.
    paths = sorted(glob.glob(s["log_glob"]))  # noqa: PTH207 — see comment above
    evidence["logs_found"] = len(paths)
    evidence["log_paths"] = paths
    if len(paths) != world_size:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.FAIL,
            Coverage(len(paths), "per-rank logs", expected=world_size),
            f"found {len(paths)} logs for world size {world_size} via {s['log_glob']} — "
            f"per-rank evidence is incomplete",
            evidence,
        )

    try:
        mem_re = re.compile(s["mem_regex"])
    except re.error as exc:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.ERROR,
            Coverage(0, "per-rank logs", expected=world_size),
            f"evidence.mem_regex does not compile: {exc}",
            evidence,
        )

    checked = 0
    max_mib = -1.0
    max_src = None
    unparsable = []
    stale = []
    now = time.time()
    for path in paths:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            unparsable.append(f"{path} (unreadable: {exc})")
            continue
        vals = [float(m.group(1)) for m in mem_re.finditer(text)]
        if not vals:
            unparsable.append(path)
            continue
        peak = max(vals)
        if peak > max_mib:
            max_mib, max_src = peak, path
        age = now - p.stat().st_mtime
        if age > int(s["max_log_age_s"]):
            stale.append(f"{path} (age {age:.0f}s > {s['max_log_age_s']}s)")
        checked += 1
    evidence["max_memory_mib"] = max_mib if max_src else None
    evidence["max_memory_source"] = max_src
    if unparsable:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.FAIL,
            Coverage(checked, "per-rank logs", expected=world_size),
            f"{len(unparsable)} logs carry no parsed memory line ({s['mem_regex']}): "
            + ", ".join(unparsable[:4]),
            evidence,
        )
    if stale:
        return _finalize(
            "evidence_completeness",
            "Evidence completeness",
            Verdict.FAIL,
            Coverage(checked, "per-rank logs", expected=world_size),
            f"{len(stale)} logs violate .out mtime liveness: " + ", ".join(stale[:4]),
            evidence,
        )

    job = s.get("slurm_job_id")
    if job:
        # Item 8, satisfied structurally: every Slurm query this tool performs
        # goes through this exact bash -lc argv. Timeout-bounded; any failure
        # means the job identity could not be corroborated — BLOCK, not shrug.
        cmd = ["bash", "-lc", f"sacct -j {shlex.quote(str(job))} --format=JobID,Elapsed,State -Pn"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _finalize(
                "evidence_completeness",
                "Evidence completeness",
                Verdict.ERROR,
                Coverage(checked, "per-rank logs", expected=world_size),
                f"slurm corroboration could not run ({exc}); job identity unverified",
                evidence,
            )
        rows = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        evidence["slurm"] = {
            "job_id": job,
            "argv": " ".join(cmd)[:80] + "…",
            "rc": proc.returncode,
            "rows": len(rows),
        }
        if proc.returncode != 0 or not rows:
            return _finalize(
                "evidence_completeness",
                "Evidence completeness",
                Verdict.FAIL,
                Coverage(checked, "per-rank logs", expected=world_size),
                f"sacct for job {job} returned rc={proc.returncode}, {len(rows)} rows — "
                f"the evidence cannot be tied to a scheduler job",
                evidence,
            )
    else:
        evidence["slurm"] = {
            "job_id": None,
            "opt_out": {
                "allow_no_slurm": s["allow_no_slurm"],
                "reason": s.get("slurm_absent_reason", ""),
            },
        }
        # Config validation already refused an undeclared absence.

    return _finalize(
        "evidence_completeness",
        "Evidence completeness",
        Verdict.PASS,
        Coverage(checked, "per-rank logs", expected=world_size),
        f"{checked}/{world_size} rank logs live and parsed; peak memory {max_mib:.1f} MiB "
        f"({max_src}); slurm {'corroborated job ' + str(job) if job else 'absent BY DECLARATION'}",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 9 — training dynamics
# ---------------------------------------------------------------------------


def _check_training_dynamics(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 9: loss bands at pinned iterations, the <0.1 hard floor, LR on
    every evidence row, and sec/iter measured ONLY past iteration 100.

    Contract: dynamics.metrics_jsonl rows {"iteration": int, "loss": float,
    "lr": float, "iter_time_s": float}. Every record scanned is an evidence
    row, so EVERY record must carry a numeric lr — a row without it
    disqualifies the sweep (that is what 'lr on every evidence row' means
    operationally). Bands failing to find their iteration FAIL by name.
    """
    evidence: dict[str, Any] = {"bands": s["bands"], "hard_floor": s["hard_floor"]}
    try:
        rows = [
            json.loads(ln)
            for ln in Path(s["metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.ERROR,
            Coverage.none("banded iterations"),
            f"metrics unreadable: {s['metrics_jsonl']}: {exc}",
            evidence,
        )
    evidence["records_scanned"] = len(rows)
    if not rows:
        # Coverage will render this VACUOUS via the ladder only for PASS; an
        # empty metrics file is asserted directly here so the detail names it.
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage.none("banded iterations"),
            "0 metric records — a dynamics claim over zero rows is doctrine 1 verbatim",
            evidence,
        )

    floor = float(s["hard_floor"])
    no_lr = []
    floor_hits = []
    by_iter: dict[int, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("iteration"), int):
            return _finalize(
                "training_dynamics",
                "Training dynamics",
                Verdict.ERROR,
                Coverage(0, "banded iterations", expected=len(s["bands"])),
                "a metric record lacks an integer 'iteration' — the sweep cannot key its evidence",
                evidence,
            )
        it = r["iteration"]
        by_iter[it] = r
        lr = r.get("lr")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            no_lr.append(it)
        loss = r.get("loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool) and loss < floor:
            floor_hits.append((it, loss))
    if no_lr:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(0, "banded iterations", expected=len(s["bands"])),
            f"{len(no_lr)} evidence rows carry no numeric lr (e.g. iterations {no_lr[:5]}) — "
            f"item 9: lr on EVERY evidence row or the row is not evidence",
            evidence,
        )
    if floor_hits:
        it, loss = floor_hits[0]
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(0, "banded iterations", expected=len(s["bands"])),
            f"HARD FLOOR: loss {loss} < {floor} at iteration {it}"
            + (f" (+{len(floor_hits) - 1} more)" if len(floor_hits) > 1 else ""),
            evidence,
        )

    checked = 0
    band_misses = []
    band_out = []
    lrs = []
    for band in s["bands"]:
        rec = by_iter.get(int(band["iteration"]))
        if rec is None:
            band_misses.append(band["iteration"])
            continue
        checked += 1
        loss = rec.get("loss")
        lr = rec["lr"]
        lrs.append(float(lr))
        if not isinstance(loss, (int, float)) or isinstance(loss, bool):
            band_out.append(f"iter {band['iteration']}: loss not numeric ({loss!r})")
            continue
        if not (float(band["lo"]) <= float(loss) <= float(band["hi"])):
            band_out.append(
                f"iter {band['iteration']}: loss {loss} outside [{band['lo']}, {band['hi']}]"
            )
    if band_misses:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.UNDERCOVERED,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            f"no metric record exists at banded iterations {band_misses} — bands that were "
            f"never evaluated cannot be 'within band'",
            evidence,
        )
    if band_out:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            " | ".join(band_out[:4]),
            evidence,
        )

    speed_rows = [r for r in by_iter.values() if r["iteration"] > 100]
    evidence["speed_window"] = {"iteration_gt": 100, "rows": len(speed_rows)}
    if not speed_rows:
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            f"0 records past iteration 100 (max seen: {max(by_iter)}) — item 9 forbids "
            f"reporting sec/iter from the warmup window, so none is reportable",
            evidence,
        )
    times = [r.get("iter_time_s") for r in speed_rows]
    if any(not isinstance(t, (int, float)) or isinstance(t, bool) for t in times):
        return _finalize(
            "training_dynamics",
            "Training dynamics",
            Verdict.FAIL,
            Coverage(checked, "banded iterations", expected=len(s["bands"])),
            "some post-100 records lack numeric iter_time_s — sec/iter would be partially measured",
            evidence,
        )
    mean_t = sum(float(t) for t in times) / len(times)
    evidence["sec_per_iter"] = {"mean": round(mean_t, 4), "samples": len(times)}
    evidence["lr_range_on_bands"] = [min(lrs), max(lrs)]

    return _finalize(
        "training_dynamics",
        "Training dynamics",
        Verdict.PASS,
        Coverage(checked, "banded iterations", expected=len(s["bands"])),
        f"{checked}/{len(s['bands'])} banded iterations in-band; 0 floor breaches over "
        f"{len(rows)} records; mean sec/iter {mean_t:.2f}s over {len(times)} post-100 rows; "
        f"lr present on all evidence rows",
        evidence,
    )


# ---------------------------------------------------------------------------
# Item 10 — launch provenance
# ---------------------------------------------------------------------------


def _check_launch_provenance(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 10: checkpoints carry THIS preflight's manifest hash; resume is
    statically shown to refuse mismatch; walltime has a floor; every declared
    artifact's mtime lies inside the declared job window."""
    err = _shared_or_error(
        _Check("launch_provenance", "", None, _stub_fn), shared, ["manifest_sha256"]
    )
    if err:
        return err
    manifest_sha = shared["manifest_sha256"]
    evidence: dict[str, Any] = {"manifest_sha256": manifest_sha}

    # (a) embedding: each probe checkpoint must name this exact manifest.
    ck_reports = []
    ck_bad = []
    for d in s["checkpoint_dirs"]:
        prov = Path(d) / "provenance.json"
        try:
            rec = json.loads(prov.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ck_bad.append(
                f"{d}: provenance.json unreadable ({exc}) — "
                f"nothing ties this checkpoint to any preflight"
            )
            continue
        got = rec.get("manifest_hash") if isinstance(rec, dict) else None
        ck_reports.append({"dir": d, "embedded_manifest_hash": got, "matches": got == manifest_sha})
        if got != manifest_sha:
            ck_bad.append(
                f"{d}: embedded hash {str(got)[:12]}… != this clearance's {manifest_sha[:12]}…"
            )
    evidence["checkpoints"] = ck_reports

    # (b) resume guard, shown statically. Declared as what it is: a text
    # sweep proving the resume path NAMES the hash AND has a raise/exit —
    # existence proof of the guard, not of its every branch (human confirms;
    # see 'does NOT close').
    guard_hits = {}
    guard_bad = []
    for path in s["resume_guard_files"]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            guard_bad.append(f"{path}: unreadable ({exc})")
            continue
        names_hash = "manifest_hash" in text
        blocks = ("raise" in text) or ("sys.exit" in text) or ("ManifestMismatch" in text)
        guard_hits[path] = {"names_manifest_hash": names_hash, "has_refusal": blocks}
        if not (names_hash and blocks):
            guard_bad.append(f"{path} (names_manifest_hash={names_hash}, has_refusal={blocks})")
    evidence["resume_guards"] = guard_hits

    # (c) walltime floor from cumulative elapsed_s.
    try:
        wt_rows = [
            json.loads(ln)
            for ln in Path(s["walltime_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.ERROR,
            Coverage(0, "provenance artifacts", expected=len(s["artifacts"])),
            f"walltime metrics unreadable: {s['walltime_jsonl']}: {exc}",
            evidence,
        )
    elapsed = [
        r.get("elapsed_s")
        for r in wt_rows
        if isinstance(r, dict) and isinstance(r.get("elapsed_s"), (int, float))
    ]
    wall = max(elapsed) if elapsed else None
    evidence["walltime_s"] = wall
    evidence["min_walltime_s"] = s["min_walltime_s"]
    if wall is None:
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(0, "provenance artifacts", expected=len(s["artifacts"])),
            f"0 elapsed_s records in {len(wt_rows)} walltime rows — no wallclock evidence exists",
            evidence,
        )
    if wall < float(s["min_walltime_s"]):
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(0, "provenance artifacts", expected=len(s["artifacts"])),
            f"walltime {wall}s < floor {s['min_walltime_s']}s — evidence produced 'too fast' "
            f"is evidence that was fabricated, replayed, or misattributed",
            evidence,
        )

    # (d) artifact mtimes ⊂ job window (with declared clock skew).
    start = _parse_iso(s["job_window_utc"][0])
    end = _parse_iso(s["job_window_utc"][1])
    assert start is not None and end is not None  # guaranteed by _post_validate
    slack = int(s["mtime_slack_s"])
    checked = 0
    outside = []
    for path in s["artifacts"]:
        p = Path(path)
        if not p.exists():
            outside.append(f"{path} (absent)")
            continue
        mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime, _dt.timezone.utc)
        checked += 1
        if not (
            start - _dt.timedelta(seconds=slack) <= mtime <= end + _dt.timedelta(seconds=slack)
        ):
            outside.append(f"{path} (mtime {mtime.isoformat(timespec='seconds')} outside window)")
    evidence["artifacts_examined"] = checked
    evidence["job_window_utc"] = s["job_window_utc"]
    if outside:
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(checked, "provenance artifacts", expected=len(s["artifacts"])),
            f"{len(outside)} artifacts outside the job window: " + "; ".join(outside[:4]),
            evidence,
        )
    if ck_bad or guard_bad:
        reasons = []
        if ck_bad:
            reasons.append("checkpoint/manifest tie failed: " + "; ".join(ck_bad[:2]))
        if guard_bad:
            reasons.append("resume guard not shown: " + "; ".join(guard_bad[:2]))
        return _finalize(
            "launch_provenance",
            "Launch provenance",
            Verdict.FAIL,
            Coverage(checked, "provenance artifacts", expected=len(s["artifacts"])),
            " | ".join(reasons),
            evidence,
        )

    return _finalize(
        "launch_provenance",
        "Launch provenance",
        Verdict.PASS,
        Coverage(checked, "provenance artifacts", expected=len(s["artifacts"])),
        f"{len(ck_reports)} checkpoints embed this manifest hash; "
        f"{len(guard_hits)} resume guards shown; walltime {wall:.0f}s ≥ floor; "
        f"{checked}/{len(s['artifacts'])} artifact mtimes inside the job window",
        evidence,
    )


def _stub_fn(*_a: Any, **_k: Any) -> CheckResult:  # never called; placeholder for _shared_or_error
    raise AssertionError("stub")


# ---------------------------------------------------------------------------
# Registrations (order matters: frozen_manifest MUST run before its consumers)
# ---------------------------------------------------------------------------


def _mk(check_id, title, section, fn, lanes):
    return _register(check_id, title, section, lanes=lanes)(fn)


_mk(
    "frozen_manifest",
    "Frozen manifest (design item 1)",
    "frozen",
    _check_frozen_manifest,
    lanes=[
        _Lane(
            "corpus-bytes-tampered",
            "a corpus file's bytes no longer match the pinned sha256",
            lambda w: w.append_bytes(w.corpus[0], b"# tampered\n"),
        ),
        _Lane(
            "model-file-missing", "a declared model shard is absent", lambda w: w.model[0].unlink()
        ),
    ],
)
_mk(
    "template_audit",
    "Template audit / CoT containment (item 2)",
    "template",
    _check_template_audit,
    lanes=[
        _Lane(
            "cot-escapes-mask",
            "probe reports cot_span outside masked_span on every row",
            lambda w: w.probe_sick(True),
        ),
        _Lane(
            "keep-cot-zero",
            "FOXBRAIN_GEMMA4_KEEP_COT=0 in the launch environment",
            lambda w: w.env.__setitem__("FOXBRAIN_GEMMA4_KEEP_COT", "0"),
        ),
    ],
)
_mk(
    "corpus_wiring",
    "Corpus wiring / banner-manifest equality (item 3)",
    "corpus_wiring",
    _check_corpus_wiring,
    lanes=[
        _Lane(
            "recipe-loses-env-call",
            "no declared recipe file names _env_jsonls(",
            lambda w: w.recipe.write_text(
                "# corpus loading was refactored; env reader removed\n", encoding="utf-8"
            ),
        ),
        _Lane(
            "resolved-env-drifts",
            "FOXBRAIN_SFT_JSONLS names a file the frozen manifest does not",
            lambda w: w.env.__setitem__(
                "FOXBRAIN_SFT_JSONLS",
                os.pathsep.join(str(p) for p in w.corpus + [w.root / "phantom.jsonl"]),
            ),
        ),
    ],
)
_mk(
    "verdict_schema",
    "Verdict schema / launch-time red team (item 4)",
    None,
    _check_verdict_schema,
    lanes=[
        # check 4's own positive control: a registry that contains a peer with NO
        # fire lane must fail it by name. Exercised by the self-test harness with
        # a doctored registry; declared here as a lane so the runtime red team
        # covers meta-failure too.
        _Lane(
            "peer-ships-no-fire-lane",
            "a peer check declares zero MUST_FIRE lanes",
            lambda w: setattr(w, "registry_override", _doctored_registry_no_lanes()),
        ),
    ],
)
_mk(
    "conversion_coverage",
    "Conversion coverage map (item 5)",
    "conversion",
    _check_conversion_coverage,
    lanes=[
        _Lane(
            "map-drops-tensor",
            "the coverage map silently omits one header tensor",
            lambda w: w.coverage_map_drop_one(),
        ),
        _Lane(
            "iter1-loss-out-of-band",
            "iter-1 loss lands outside the pinned band",
            lambda w: w.rewrite_conv_metrics(loss=9.9),
        ),
    ],
)
_mk(
    "lora_probe",
    "LoRA probe 20-iter (item 6)",
    "lora",
    _check_lora_probe,
    lanes=[
        _Lane(
            "intended-class-silent",
            "an intended target class has zero 'Adding lora to' lines",
            lambda w: w.lora_log_strip("kv_proj"),
        ),
        _Lane(
            "merged-bytes-mismatch",
            "merged HF export bytes no longer match the external pin",
            lambda w: w.append_bytes(w.merged[0], b"\x00"),
        ),
    ],
)
_mk(
    "schedule_consistency",
    "Schedule banner (item 7)",
    "schedule",
    _check_schedule,
    lanes=[
        _Lane(
            "lr-decay-mismatch",
            "lr_decay_iters != train_iters",
            lambda w: w.cfg["schedule"].__setitem__(
                "lr_decay_iters", w.cfg["schedule"]["train_iters"] + 7
            ),
        ),
    ],
)
_mk(
    "evidence_completeness",
    "Evidence completeness (item 8)",
    "evidence",
    _check_evidence_completeness,
    lanes=[
        _Lane(
            "rank-log-missing", "fewer per-rank logs than world size", lambda w: w.logs[0].unlink()
        ),
        _Lane(
            "log-stale",
            "a per-rank log violates .out mtime liveness",
            lambda w: os.utime(w.logs[1], (946684800, 946684800)),
        ),
    ],
)
_mk(
    "training_dynamics",
    "Training dynamics (item 9)",
    "dynamics",
    _check_training_dynamics,
    lanes=[
        _Lane(
            "loss-floor-breach",
            "a loss below the hard floor appears mid-run",
            lambda w: w.dynamics_patch(42, loss=0.05),
        ),
        _Lane(
            "lr-row-missing",
            "an evidence row carries no lr",
            lambda w: w.dynamics_patch(7, lr=None),
        ),
    ],
)
_mk(
    "launch_provenance",
    "Launch provenance (item 10)",
    "provenance",
    _check_launch_provenance,
    lanes=[
        _Lane(
            "checkpoint-hash-mismatch",
            "a checkpoint's embedded manifest hash names a different preflight",
            lambda w: (
                Path(w.cfg["provenance"]["checkpoint_dirs"][0]) / "provenance.json"
            ).write_text('{"manifest_hash": "' + "0" * 64 + '"}', encoding="utf-8"),
        ),
        _Lane(
            "artifact-outside-window",
            "a declared artifact's mtime lies outside the job window",
            lambda w: os.utime(w.merged[0], (946684800, 946684800)),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Fixture world for --self-test and for the launch-time red team (item 4)
# ---------------------------------------------------------------------------


@dataclass
class _World:
    """A fully materialized, KNOWN-HEALTHY miniature of every artifact class the
    ten checks read, under one temp dir. cfg is a complete, schema-valid config
    with pins MEASURED from the files actually written (pins are computed, not
    asserted — a fixture whose pins disagree with its files proves nothing).
    Lanes then corrupt one artifact class at a time."""

    root: Path
    cfg: dict[str, Any]
    env: dict[str, str]
    corpus: list[Path]
    model: list[Path]
    logs: list[Path]
    merged: list[Path]
    recipe: Path
    probe_path: Path
    registry_override: Sequence[_Check] | None = None

    # -- lane helpers ---------------------------------------------------------
    def write(self, rel: str, blob: bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        return p

    @staticmethod
    def append_bytes(p: Path, blob: bytes) -> None:
        with p.open("ab") as fh:
            fh.write(blob)

    def probe_sick(self, sick: bool) -> None:
        argv = self.cfg["template"]["probe_command"]
        if sick and "--sick" not in argv:
            argv.append("--sick")
        elif not sick and "--sick" in argv:
            argv.remove("--sick")

    def coverage_map_drop_one(self) -> None:
        data = json.loads(
            (self.root / "artifacts" / "coverage_map.json").read_text(encoding="utf-8")
        )
        # Drop exactly one CONVERTED entry: the resulting uncovered header
        # tensor is the classic silent-coverage gap the map exists to catch.
        data["tensors"] = [
            e
            for e in data["tensors"]
            if not (e.get("coverage") == "converted" and e.get("name") == "encoder.layers.0.w")
        ]
        (self.root / "artifacts" / "coverage_map.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def rewrite_conv_metrics(self, loss: float) -> None:
        rows = [{"iteration": 0, "param_count": 96}, {"iteration": 1, "loss": loss, "lr": 1e-4}]
        (self.root / "artifacts" / "conv_metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    def lora_log_strip(self, cls: str) -> None:
        log = self.root / "artifacts" / "lora_run.log"
        log.write_text(
            "\n".join(ln for ln in log.read_text(encoding="utf-8").splitlines() if cls not in ln)
            + "\n",
            encoding="utf-8",
        )

    def dynamics_patch(self, iteration: int, **fields: Any) -> None:
        path = self.root / "artifacts" / "dynamics.jsonl"
        rows = [
            json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        for r in rows:
            if r.get("iteration") == iteration:
                for k, v in fields.items():
                    if v is None:
                        r.pop(k, None)
                    else:
                        r[k] = v
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _safetensors_blob(tensors: Mapping[str, tuple[int, ...]]) -> bytes:
    header: dict[str, Any] = {}
    data = bytearray()
    for name, shape in tensors.items():
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [len(data), len(data) + nbytes],
        }
        data += b"\x00" * nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + bytes(data)


_PROBE_SOURCE = '''\
"""Fixture stand-in for the FoxBrain template probe (INPUT CONTRACT item 2).

Emits rows_per_file JSON rows for one file. With --sick, cot_span escapes
masked_span on every row: the MUST_FIRE articulation of the CoT-containment
defect the real audit exists to catch."""
import json
import sys


def main() -> int:
    path, rows = sys.argv[1], int(sys.argv[2])
    sick = "--sick" in sys.argv
    for i in range(rows):
        row = {
            "row": i,
            "file": path,
            "tokens_stock": 120 + i,
            "tokens_patched": 132 + i,   # stock-vs-patched diff present
            "cot_span": [4, 12] if sick else [40, 80],
            "masked_span": [32, 96],
        }
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _fresh_world() -> _WorldCtx:
    return _WorldCtx()


class _WorldCtx:
    def __enter__(self) -> _World:
        self._tmp = tempfile.TemporaryDirectory(prefix="preflight-selftest-")
        return _build_world(Path(self._tmp.name))

    def __exit__(self, *exc: Any) -> None:
        self._tmp.cleanup()


def _build_world(root: Path) -> _World:
    art = root / "artifacts"
    art.mkdir(parents=True)
    corpus = []
    corpus_pins = []
    for i in range(4):
        p = art / f"corpus-{i}.jsonl"
        rows = [
            {"idx": i * 100 + j, "text": f"trace {i}-{j}", "cot": f"reasoning {i}-{j}"}
            for j in range(3)
        ]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        sha, lines = _sha256_and_lines(p)
        corpus.append(p)
        corpus_pins.append({"path": str(p), "sha256": sha, "lines": lines})

    model = [
        _fresh := art / "shard-00001.safetensors",
        art / "shard-00002.safetensors",
    ]
    model[0].write_bytes(
        _safetensors_blob({"encoder.layers.0.w": (4, 8), "encoder.layers.1.w": (4, 8)})
    )
    model[1].write_bytes(_safetensors_blob({"encoder.layers.2.w": (8, 4)}))
    total_bytes = sum(p.stat().st_size for p in model)

    run_config = art / "run_config.json"
    run_config.write_text('{"recipe": "e4b-fixture"}\n', encoding="utf-8")

    chat = art / "chat_template.jinja"
    chat.write_text("{{ bos }}{% for m in messages %}...{% endfor %}\n", encoding="utf-8")

    probe_path = art / "fixture_template_probe.py"
    probe_path.write_text(_PROBE_SOURCE, encoding="utf-8")

    recipe = root / "recipe" / "train.py"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(
        "def main():\n    jsonls = _env_jsonls('FOXBRAIN_SFT_JSONLS')\n    return load(jsonls)\n",
        encoding="utf-8",
    )

    coverage_map = art / "coverage_map.json"
    coverage_map.write_text(
        json.dumps(
            {
                "tensors": [
                    {"name": "encoder.layers.0.w", "bytes": 128, "coverage": "converted"},
                    {"name": "encoder.layers.1.w", "bytes": 128, "coverage": "converted"},
                    {"name": "encoder.layers.2.w", "bytes": 128, "coverage": "converted"},
                    # lm_head shares storage with the embedding table: allow-listed, and
                    # the ground (tie_word_embeddings==true) is checked against the HF config.
                    {"name": "lm_head.weight", "coverage": "allowlist", "rule": "tied"},
                ]
            }
        ),
        encoding="utf-8",
    )

    hf_cfg = art / "hf_config.json"
    hf_cfg.write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_experts": 4,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )

    conv_metrics = art / "conv_metrics.jsonl"
    conv_metrics.write_text(
        json.dumps({"iteration": 0, "param_count": 96})
        + "\n"
        + json.dumps({"iteration": 1, "loss": 2.15, "lr": 1e-4})
        + "\n",
        encoding="utf-8",
    )

    lora_log = art / "lora_run.log"
    lora_log.write_text(
        "Adding lora to q_proj of layer 0\n"
        "Adding lora to kv_proj of layer 0\n"
        "trainable params: 1,234,567 || all params: 8,000,000,000 || trainable%: 0.0154\n",
        encoding="utf-8",
    )

    lora_metrics = art / "lora_probe.jsonl"
    lora_rows = [{"iteration": i} for i in range(1, 21)]
    lora_rows[-1]["lora_b_norm"] = {"q_proj": 0.53, "kv_proj": 0.41}
    lora_metrics.write_text("\n".join(json.dumps(r) for r in lora_rows) + "\n", encoding="utf-8")

    delta = art / "delta_audit.json"
    delta.write_text(
        json.dumps(
            {
                "q_proj": {"delta_l2": 1.7e-3, "tensors_checked": 4},
                "kv_proj": {"delta_l2": 9.2e-4, "tensors_checked": 4},
            }
        ),
        encoding="utf-8",
    )

    merged_dir = root / "merged"
    merged_dir.mkdir()
    merged = [merged_dir / "model-00001.safetensors", merged_dir / "model-00002.safetensors"]
    merged[0].write_bytes(b"\x00" * 1000)
    merged[1].write_bytes(b"\x00" * 500)
    # A self-index that LIES about the total — planted in every fixture world
    # so a regression to self-index parity turns red everywhere at once.
    (merged_dir / "model.safetensors.index.json").write_text(
        '{"metadata": {"total_size": 999999999}, "weight_map": {}}', encoding="utf-8"
    )

    # The external pin is MEASURED HERE, at world-build time, by walking the
    # same bytes-on-disk quantity that lora_probe independently walks at check
    # time. Legitimate here and forbidden inside the check, for mirror-image
    # reasons: _build_world is the operator's stand-in — an authority EXTERNAL
    # to the check under test, whose pins are measurements of what it wrote,
    # never the artifact's own claims — whereas a pin derived inside the check
    # from the same walk would be self-referential (pin == observed trivially,
    # both control halves green on a detector that verifies nothing: exactly
    # the hole the planted lying index exists to expose). The walk prices the
    # planted index like any other regular file, per the check's contract, so
    # this pin is deliberately NOT 1500 (= 1000 + 500 shards-only arithmetic,
    # which forgot the planted file's real bytes).
    pinned_merged_total_bytes = sum(p.stat().st_size for p in merged_dir.rglob("*") if p.is_file())

    now = time.time()
    dyn_rows = []
    for i in range(1, 111):
        dyn_rows.append(
            {
                "iteration": i,
                "loss": max(0.6, 3.0 * (0.97**i)),
                "lr": 1e-4 * (0.999**i),
                "iter_time_s": 2.5,
                "elapsed_s": i * 2.5,
            }
        )
    dyn_path = art / "dynamics.jsonl"
    dyn_path.write_text("\n".join(json.dumps(r) for r in dyn_rows) + "\n", encoding="utf-8")

    log_dir = root / "logs"
    log_dir.mkdir()
    logs = []
    for r in range(4):
        p = log_dir / f"job.rank{r}.out"
        p.write_text(f"step=110 rank={r} max_memory_mib={4096.5 - r * 10}\n", encoding="utf-8")
        logs.append(p)

    guard = root / "recipe" / "checkpointing.py"
    guard.write_text(
        "def resume(ckpt, manifest_hash):\n"
        "    if ckpt.manifest_hash != manifest_hash:\n"
        "        raise ManifestMismatchError('checkpoint names a different preflight')\n",
        encoding="utf-8",
    )

    cfg: dict[str, Any] = {
        "run_name": "fixture-e4b",
        "world_size": 4,
        "frozen": {
            "model": {
                "files": [str(p) for p in model],
                "tensor_count": 3,
                "total_bytes": total_bytes,
            },
            "corpus": {"files": corpus_pins},
            "run_config": {
                "path": str(run_config),
                "sha256": hashlib.sha256(run_config.read_bytes()).hexdigest(),
            },
        },
        "template": {
            "probe_command": [sys.executable, str(probe_path), "{file}", "{rows}"],
            "rows_per_file": 2,
            "files": [str(corpus[0]), str(corpus[1])],
            "keep_cot_env": "FOXBRAIN_GEMMA4_KEEP_COT",
            "chat_template_path": str(chat),
            "chat_template_md5": hashlib.md5(chat.read_bytes()).hexdigest(),
        },
        "corpus_wiring": {
            "env_var": "FOXBRAIN_SFT_JSONLS",
            "recipe_files": [str(recipe)],
            "attestation_path": str(art / "attestation.json"),
        },
        "conversion": {
            "hf_config_json": str(hf_cfg),
            "coverage_map_json": str(coverage_map),
            "iter_metrics_jsonl": str(conv_metrics),
            "iter1_loss_band": [1.0, 3.0],
            "expected_param_count": 96,
            "divisibility": [
                {"field": "num_attention_heads", "divisible_by": 4},
                {"field": "num_key_value_heads", "divisible_by": 4},
                {"field": "num_experts", "divisible_by": 4},
                {"field": "tie_word_embeddings", "equals": True},
            ],
            "tied_grounding": "tie_word_embeddings",
            "shared_kv_grounding": None,
        },
        "lora": {
            "run_log": str(lora_log),
            "target_classes": ["q_proj", "kv_proj"],
            "trainable_band": [1_000_000, 2_000_000],
            "probe_metrics_jsonl": str(lora_metrics),
            "delta_audit_json": str(delta),
            "merged_dir": str(merged_dir),
            "pinned_merged_total_bytes": pinned_merged_total_bytes,
            "expected_iters": 20,
        },
        "schedule": {
            "train_iters": 1000,
            "lr_decay_iters": 1000,
            "save_interval": 250,
            "explicit_final_save": False,
            "smoke": False,
        },
        "evidence": {
            "log_glob": str(log_dir / "*.out"),
            "mem_regex": r"max_memory_mib=([0-9.]+)",
            "max_log_age_s": 3600,
            "slurm_job_id": None,
            "allow_no_slurm": True,
            "slurm_absent_reason": "fixture host has no Slurm; declared, per the opt-out contract",
        },
        "dynamics": {
            "metrics_jsonl": str(dyn_path),
            "bands": [
                {"iteration": 1, "lo": 1.0, "hi": 3.5},
                {"iteration": 50, "lo": 0.5, "hi": 2.0},
                {"iteration": 100, "lo": 0.5, "hi": 1.2},
            ],
            "hard_floor": 0.1,
        },
        "provenance": {
            "checkpoint_dirs": [str(root / "ckpt-probe")],
            "resume_guard_files": [str(guard)],
            "min_walltime_s": 1,
            "job_window_utc": [
                _dt.datetime.fromtimestamp(now - 300, _dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                _dt.datetime.fromtimestamp(now + 86400, _dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            ],
            "mtime_slack_s": 5,
            "artifacts": [
                str(root / "ckpt-probe" / "provenance.json"),
                str(merged[0]),
                str(logs[0]),
            ],
            "walltime_jsonl": str(dyn_path),
        },
    }

    env = {
        "FOXBRAIN_SFT_JSONLS": os.pathsep.join(str(p) for p in corpus),
        "FOXBRAIN_GEMMA4_KEEP_COT": "1",
    }

    world = _World(
        root=root,
        cfg=cfg,
        env=env,
        corpus=corpus,
        model=model,
        logs=logs,
        merged=merged,
        recipe=recipe,
        probe_path=probe_path,
    )

    # Dependent artifacts, computed with the SAME helper the check uses —
    # pins measured, never asserted:
    cfg_path = root / "preflight.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    manifest_sha, _payload = _manifest_hash_for(cfg["frozen"], cfg_sha)
    ckpt = root / "ckpt-probe"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "provenance.json").write_text(
        json.dumps({"manifest_hash": manifest_sha}), encoding="utf-8"
    )
    (art / "attestation.json").write_text(
        json.dumps(
            {
                "reader": "fixture-human",
                "sample_sha256": _canonical_sample_sha256(corpus[0]),
                "note": "batch-0 sample read; this fixture stands in for the human act",
            }
        ),
        encoding="utf-8",
    )
    world._cfg_sha = cfg_sha  # type: ignore[attr-defined]
    return world


def _doctored_registry_no_lanes() -> list[_Check]:
    """A registry copy in which the FIRST peer ships no fire lane: check 4's own
    positive control. Only ever synthesized; never installed."""
    out = []
    for c in _REGISTRY_ORDER:
        if c.id == "verdict_schema":
            continue
        out.append(_Check(c.id, c.title, c.section, c.fn, ()))
        break
    return out


def _run_lane_against(peer: _Check, lane: _Lane) -> CheckResult:
    """Fresh healthy world -> inject the lane's defect -> run ONLY the target.
    The shared manifest state is established by a real frozen_manifest run on
    the healthy world FIRST: a lane whose healthy precondition fails is
    inconclusive-by-ERROR, never silently counted."""
    with _fresh_world() as world:
        shared: dict[str, Any] = {"_config_sha256": world._cfg_sha}  # type: ignore[attr-defined]
        baseline = _execute(REGISTRY["frozen_manifest"], world.cfg, world.env, shared)
        if baseline.verdict is not Verdict.PASS:
            return CheckResult(
                peer.id,
                peer.title,
                Verdict.ERROR,
                Coverage.none("units"),
                f"red-team precondition failed: frozen_manifest did not pass on the "
                f"healthy fixture world ({baseline.detail}) — the lane is inconclusive, "
                f"which is not a proven firing",
            )
        lane.apply(world)
        peers = world.registry_override if world.registry_override is not None else None
        return _execute(peer, world.cfg, world.env, shared, registry=peers)


# ---------------------------------------------------------------------------
# --self-test: prove every check can FAIL and can PASS, with denominators
# ---------------------------------------------------------------------------


def run_self_test(out: Callable[[str], None] = print) -> tuple[int, dict[str, Any]]:
    """Both halves, per check, on fresh synthesized worlds.

    MUST_PASS: healthy world -> the check must PASS (a check that blocks on
    everything satisfies every MUST_FIRE lane while verifying nothing — the
    exact hole verify_controls' MUST_PASS guard exists to close).
    MUST_FIRE: each declared lane -> the check must NOT return PASS; a lane
    that leaves the verdict PASS is a detector proven not to fire on the
    defect class it claims.

    Exitcode 0 iff every check proves both halves. The returned dict carries
    the denominators so callers (and tests) consume them structurally.
    """
    failures: list[str] = []
    checks_total = len(_REGISTRY_ORDER)
    fire_total = fire_proven = 0
    pass_total = pass_proven = 0
    per_check: list[dict[str, Any]] = []

    out(
        "preflight --self-test — proving each check can FAIL on a known defect "
        "and PASS on a healthy world"
    )
    for chk in _REGISTRY_ORDER:
        rec: dict[str, Any] = {
            "check": chk.id,
            "lanes": len(chk.lanes),
            "must_pass": None,
            "fire": [],
        }

        # ---- MUST_PASS half ------------------------------------------------
        pass_total += 1
        try:
            with _fresh_world() as world:
                shared: dict[str, Any] = {"_config_sha256": world._cfg_sha}  # type: ignore[attr-defined]
                baseline = _execute(REGISTRY["frozen_manifest"], world.cfg, world.env, shared)
                if chk.id == "frozen_manifest":
                    res = baseline
                elif baseline.verdict is not Verdict.PASS:
                    raise RuntimeError(f"healthy-world precondition failed: {baseline.detail}")
                else:
                    # The runtime check red-teams every peer; proving the
                    # mechanism on ONE real peer (with its real lanes) keeps
                    # the self-test sub-quadratic. Runtime runs all peers.
                    #
                    # The ternary carries the comment now because it is the only
                    # binding of `peers`: an earlier reading called it dead and
                    # proposed deleting it, but its `else None` arm is what binds
                    # `peers` for every non-verdict_schema check. The redundant
                    # half was the `if` block that recomputed the same list.
                    peers = [REGISTRY["frozen_manifest"]] if chk.id == "verdict_schema" else None
                    res = _execute(chk, world.cfg, world.env, shared, registry=peers)
            rec["must_pass"] = res.verdict.value
            if res.verdict is Verdict.PASS:
                pass_proven += 1
            else:
                failures.append(
                    f"{chk.id}: MUST_PASS half failed on a known-healthy world "
                    f"(got {res.verdict.value}: {res.detail})"
                )
        except Exception as exc:  # noqa: BLE001
            rec["must_pass"] = "HARNESS-ERROR"
            failures.append(f"{chk.id}: MUST_PASS harness raised {type(exc).__name__}: {exc}")

        # ---- MUST_FIRE half -------------------------------------------------
        if not chk.lanes:
            failures.append(
                f"{chk.id}: declares NO MUST_FIRE lane — a check that has never been "
                f"shown to fire may not certify a launch (design item 4)"
            )
        for lane in chk.lanes:
            fire_total += 1
            try:
                res = _run_lane_against(chk, lane)
                flipped = res.verdict is not Verdict.PASS and res.verdict is not Verdict.ERROR
                rec["fire"].append({"lane": lane.name, "verdict": res.verdict.value})
                if flipped:
                    fire_proven += 1
                else:
                    failures.append(
                        f"{chk.id}/{lane.name}: MUST_FIRE lane left verdict "
                        f"{res.verdict.value} ({res.detail[:120]}) — the defect "
                        f"'{lane.description}' did not flip the check"
                    )
            except Exception as exc:  # noqa: BLE001
                rec["fire"].append({"lane": lane.name, "verdict": "HARNESS-ERROR"})
                failures.append(
                    f"{chk.id}/{lane.name}: MUST_FIRE harness raised {type(exc).__name__}: {exc}"
                )
        per_check.append(rec)

    report = {
        "checks_total": checks_total,
        "must_fire_proven": fire_proven,
        "must_fire_total": fire_total,
        "must_pass_proven": pass_proven,
        "must_pass_total": pass_total,
        "failures": failures,
        "per_check": per_check,
    }
    out(f"checks: {checks_total}")
    out(f"MUST_FIRE proven: {fire_proven}/{fire_total} lanes")
    out(f"MUST_PASS proven: {pass_proven}/{pass_total} checks")
    if failures:
        out(f"\n{len(failures)} self-test failure(s):")
        for f in failures:
            out(f"  - {f}")
        out(
            "\nresult: FAILED — at least one check is not proven for this "
            "launch's artifact classes."
        )
        return 1, report
    out(
        "\nresult: OK — every check flipped on its deliberately corrupt "
        "artifacts and passed its healthy world."
    )
    return 0, report


# ---------------------------------------------------------------------------
# Report rendering + banner
# ---------------------------------------------------------------------------


def _render_report(
    cfg: dict[str, Any],
    results: Sequence[CheckResult],
    manifest_sha: str | None,
    out: Callable[[str], None],
) -> bool:
    n_pass = sum(1 for r in results if r.verdict is Verdict.PASS)
    clear = _is_clear(results)
    out(f"preflight @ {cfg.get('run_name', '?')}: {len(results)} checks run")
    for r in results:
        out("  " + r.render())
    # The denominator summary line: no reader tallies the column by hand.
    total_units = sum(r.coverage.checked for r in results)
    out(
        f"  — {n_pass}/{len(results)} checks PASS; {total_units} units examined across "
        f"{len(results)} checks"
    )
    if clear:
        smoke = bool(cfg.get("schedule", {}).get("smoke"))
        out("")
        out("=== FOUNDATIONSCALE PRE-FLIGHT BANNER ===")
        out(f"run: {cfg['run_name']}")
        out(f"manifest_sha256: {manifest_sha}")
        out(f"checks: {n_pass}/{len(results)} PASS")
        if smoke:
            # Design item 7, second sentence, made unrejectable: the ONLY
            # clearance this tool can emit for a smoke-tagged config carries
            # the qualifier inside the banner itself.
            out("clearance: CLEAR (SMOKE — this banner makes NO training-correctness claim)")
        else:
            out("clearance: CLEAR")
        out("# checkpoint writers: embed manifest_sha256 into every provenance record;")
        out("# resume must refuse any checkpoint naming a different hash (item 10).")
    else:
        blocking = [r for r in results if r.verdict is not Verdict.PASS]
        out(
            f"overall: BLOCKED — {len(blocking)} check(s) NOT-VERIFIED; a launch cleared "
            f"over them would be all([]) with extra steps"
        )
        if manifest_sha:
            out(f"(frozen manifest hash for reference: {manifest_sha})")
    return clear


def _write_json_record(
    path: str,
    cfg: dict,
    cfg_sha: str,
    results: Sequence[CheckResult],
    manifest_sha: str | None,
    clear: bool,
) -> None:
    record = {
        "tool": "foundationscale-preflight",
        "tool_version": TOOL_VERSION,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "run_name": cfg.get("run_name"),
        "config_sha256": cfg_sha,
        "overall": "CLEAR" if clear else "BLOCKED",
        "manifest_sha256": manifest_sha,
        "checks_run": len(results),
        "checks_passing": sum(1 for r in results if r.verdict is Verdict.PASS),
        "units_examined": sum(r.coverage.checked for r in results),
        "results": [r.to_dict() for r in results],
        "clearance_rule": "all checks PASS with checked > 0 and at least one check ran; "
        "SKIP/VACUOUS/INAPPLICABLE are NOT-VERIFIED and block (design item 4)",
    }
    try:
        Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write --json record to {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundationscale-preflight",
        description="Executable pre-flight blocklist for the E4B launch (design: risk-review §4).",
    )
    parser.add_argument(
        "--config", help="JSON config describing the run being gated (required unless --self-test)"
    )
    parser.add_argument("--json", dest="json_path", help="write the machine record to this path")
    parser.add_argument("--only", help="comma-separated check ids to run")
    parser.add_argument("--exclude", default="", help="comma-separated check ids to leave out")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove every check flips on synthesized corrupt artifacts and passes "
        "on a healthy world (design item 4's red-team drill, executable)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list registered checks (diagnostic; NOT a clearance)"
    )
    args = parser.parse_args(argv)

    # Zero-check refusal, registry layer (doctrine 1 aimed at the tool itself).
    if not REGISTRY:
        print("preflight: BLOCKED — 0 checks are registered; a sweep over nothing proves nothing")
        return EXIT_BLOCKED

    if args.list:
        for chk in _REGISTRY_ORDER:
            print(f"  {chk.id} — {chk.title} [{len(chk.lanes)} MUST_FIRE lane(s)]")
        print("(diagnostic listing only — this output clears nothing)")
        return EXIT_CLEAR

    if args.self_test:
        try:
            code, _report = run_self_test()
        except Exception as exc:  # noqa: BLE001
            print(
                f"preflight: self-test could not run: {type(exc).__name__}: {exc}", file=sys.stderr
            )
            return EXIT_TOOL_ERROR
        return code

    if not args.config:
        print(
            "preflight: --config is required (fail closed: with no config there are no pins, "
            "and a pinless sweep is the vacuous case)",
            file=sys.stderr,
        )
        return EXIT_TOOL_ERROR

    try:
        cfg, cfg_sha = _load_config(args.config)
    except ToolError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return EXIT_TOOL_ERROR
    except ConfigError as exc:
        print(
            f"preflight: BLOCKED — config names no launchable run "
            f"({exc.problems.__len__()} problem(s)):"
        )
        for p in exc.problems:
            print(f"  - {p}")
        print(
            "0 checks examined — the configuration layer refused before any artifact was touched."
        )
        return EXIT_BLOCKED

    # ---- selection, with the run_event discipline -----------------------------
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
    known = set(REGISTRY)
    unknown = ((wanted - known) if wanted else set()) | (excluded - known)
    if unknown:
        # A typo'd check id must read as a block, naming what was asked for and
        # what exists — silently dropping it would report "all selected checks
        # clear" over fewer checks than were asked for.
        print(f"preflight: BLOCKED — selection names unknown check id(s): {sorted(unknown)}")
        print(f"registered checks: {sorted(known)}")
        print("0 checks examined — selection refused.")
        return EXIT_BLOCKED
    selected = [
        c for c in _REGISTRY_ORDER if (wanted is None or c.id in wanted) and c.id not in excluded
    ]
    if not selected:
        print(
            f"preflight: BLOCKED — the selection ran 0 of {len(REGISTRY)} registered checks; "
            f"a sweep over nothing proves nothing. Broaden --only/--exclude."
        )
        return EXIT_BLOCKED

    shared: dict[str, Any] = {"_config_sha256": cfg_sha, "_run_name": cfg["run_name"]}
    results: list[CheckResult] = []
    try:
        for chk in selected:
            results.append(_execute(chk, cfg, dict(os.environ), shared))
    except Exception as exc:  # noqa: BLE001
        print(
            f"preflight: the tool itself failed mid-sweep: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=6)}",
            file=sys.stderr,
        )
        return EXIT_TOOL_ERROR

    manifest_sha = shared.get("manifest_sha256")
    clear = _render_report(cfg, results, manifest_sha, print)
    if args.json_path:
        try:
            _write_json_record(args.json_path, cfg, cfg_sha, results, manifest_sha, clear)
        except ToolError as exc:
            print(f"preflight: {exc}", file=sys.stderr)
            return EXIT_TOOL_ERROR
        print(f"machine record: {args.json_path}")
    return EXIT_CLEAR if clear else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
