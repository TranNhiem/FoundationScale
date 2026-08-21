"""Point the shipped checkpoint gates at a REAL on-disk checkpoint.

Why this tool exists
--------------------
Until this probe, every fact known about ``foundationscale.gates.checkpoint_gates``
came from synthetic fixtures (``gates/fixtures.py``) written by the same hand that
wrote the detectors. ``verify_controls`` proves each gate *can* fire on corruption its
author *imagined*. It has never said what the gates conclude — or abstain from — on a
real artifact: multi-GB, written by a real trainer, described imperfectly by an
upstream ``config.json``. A detector suite validated only by its author's fixtures
finds exactly the defects its author thought to fabricate. This probe is the first
time the gates meet content they were not designed around.

Two rules make the probe itself safe to believe:

1. **The denominator is independent or absent.** ``declared_*`` fields are derived
   from the HF ``config.json`` beside the weights — never from the checkpoint's own
   metadata, and never from the framework's run manifest (which is written by the
   same training stack that wrote the checkpoint, so it cannot adjudicate it).
   Denominators the config does not *state* are left ``None`` and said to be
   ``None``; the affected gate then abstains (SKIP) or blocks (VACUOUS), and the
   probe reports that. Fabricating a denominator so every gate "runs" would be this
   repository committing its own founding defect inside its own probe.

2. **The exit code is the deliverable.** ``0`` only when every gate reached a real
   verdict (PASS over non-vacuous coverage, or SKIP with a stated reason) and every
   requested control fired. Any blocking verdict — FAIL, VACUOUS, UNDERCOVERED,
   ERROR — means NOT CLEARED and exit ``1``. A VACUOUS gate on a real 16 GB
   checkpoint is not a footnote; it is the ``all([])`` result this repository exists
   to prevent, arriving from the framework itself, and the probe must be able to say
   that about its own employer. Exit ``3`` means *the probe could not measure*
   (unreadable checkpoint, missing/invalid config) — distinct from argparse's ``2``
   (used wrongly) and from ``1`` (measured, and the answer was not "clear").

The probe is read-only: nothing here writes to or modifies the checkpoint.

Usage:
    python tools/real_checkpoint_probe.py CKPT_DIR [--config CONFIG_JSON] \\
        [--json OUT] [--inject-alias N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from foundationscale.checkpoint.dcp import CheckpointFormatError
from foundationscale.checkpoint.dcp_meta import CheckpointMetadata, read_metadata

# Private names, imported deliberately. The --inject-alias control must target
# exactly the tensor population ExpertDistinctnessGate examines, and the inventory
# byte total must price dtypes by exactly the table the gates use — paraphrasing
# either here would satisfy a reader but violate the controls contract ("a control
# must exercise the real code path, not a paraphrase"). If these drift, this import
# failing loudly is the point.
from foundationscale.gates.checkpoint_gates import (
    _DTYPE_BYTES,
    _SHARD_SUFFIX_RE,
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    FirstSaveGate,
    SaveCompletenessGate,
    TensorMeta,
    _expert_weights,
)
from foundationscale.gates.core import Gate, GateResult, Verdict

EXIT_CLEAR = 0
EXIT_BLOCKED = 1
EXIT_UNMEASURED = 3  # 2 belongs to argparse; 3 keeps "could not measure" its own signal

_ALIAS_STORAGE_ID = "probe://injected-alias/many-names-one-storage"

# The only keys this probe will read, per its own rule: derive what the config
# states and nothing else. ``text_config`` scope is searched first (multimodal
# models nest the LM config there), then the top level. Everything absent or null
# stays absent.
_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts")
_MOE_LAYER_KEYS = ("num_moe_layers",)

# Enumerated, not pulled from REGISTRY: the sweep's population must be a returned
# fact printed below, not an inference from import side effects (importing
# gates.example or verify.parity would silently swell a registry-driven sweep, and
# this file's whole job is exact statements about what ran).
_CHECKPOINT_GATES: tuple[type[Gate], ...] = (
    ExpertDistinctnessGate,
    ExpertByteVolumeGate,
    SaveCompletenessGate,
    FirstSaveGate,
)


class ProbeUnmeasured(RuntimeError):
    """The probe could not measure. Distinct from 'measured, and the answer blocks'."""


# ---------------------------------------------------------------------------
# Measurement: checkpoint metadata + the independent declared block
# ---------------------------------------------------------------------------


def _measure_checkpoint(path: Path) -> CheckpointMetadata:
    try:
        return read_metadata(path)
    except (CheckpointFormatError, OSError) as exc:
        # read_metadata already refuses non-checkpoints and zero-key sources with
        # precise messages; the probe adds nothing by re-explaining, so relay.
        raise ProbeUnmeasured(f"checkpoint unreadable: {path}: {exc}") from exc


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ProbeUnmeasured(
            f"config file not found: {config_path} — the declared block has no "
            f"source. Pass --config, or accept that without an independent "
            f"denominator this probe has nothing to compare the checkpoint against"
        )
    try:
        raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProbeUnmeasured(
            f"config unreadable or not valid JSON: {config_path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(raw, dict):
        raise ProbeUnmeasured(f"config is not a JSON object: {config_path}")
    return raw


def _scoped_int(
    config: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
    notes: list[str],
) -> tuple[int | None, str | None]:
    """First non-null integer among ``keys``, ``text_config`` scope before top level.

    Returns ``(value, basis)``; ``(None, None)`` means the config states nothing
    this probe can responsibly use. Only JSON integers count as declared counts:
    ``int(7.5) == 7`` would mint a denominator the config never stated, and a
    printed "7" wearing a malformed ``7.5``'s clothes is this probe committing
    its own founding defect inside its own denominator derivation. Integral
    floats are accepted *with the coercion recorded in the notes*; every other
    wrong-typed or out-of-domain value is treated as absent and recorded —
    "silently coerced" and "silently dropped" are both ways a probe starts
    telling stories about its denominator.
    """
    scopes: list[tuple[str, dict[str, Any]]] = []
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        scopes.append(("text_config", text_cfg))
    scopes.append(("", config))
    for scope_name, scope in scopes:
        for key in keys:
            if key not in scope:
                continue
            dotted = f"{scope_name}.{key}" if scope_name else key
            raw = scope[key]
            if raw is None:
                notes.append(f"{label}: {dotted} is null — treated as absent (HF convention)")
                continue
            # bool before int: in Python isinstance(True, int) is True, and a
            # JSON true is not a count however eagerly it converts like one.
            if isinstance(raw, bool):
                notes.append(
                    f"{label}: {dotted} is a boolean ({raw!r}), not a count — treated as absent"
                )
                continue
            if isinstance(raw, float):
                if not raw.is_integer():
                    # inf/NaN also land here; int(inf) raising OverflowError is
                    # why nothing below this branch calls int() on a float.
                    notes.append(
                        f"{label}: {dotted} = {raw!r} is not an integral number — "
                        f"treated as absent rather than truncated into a declaration"
                    )
                    continue
                value = int(raw)
                notes.append(
                    f"{label}: {dotted} is the float {raw!r}, not a JSON integer — "
                    f"used as {value} and recorded as a coercion, because a silent "
                    f"one would read as a declaration"
                )
            elif isinstance(raw, int):
                value = raw
            else:
                notes.append(f"{label}: {dotted} is not an integer ({raw!r}) — treated as absent")
                continue
            if value < 0:
                notes.append(
                    f"{label}: {dotted} = {value} is negative — a negative count is "
                    f"not a denominator; treated as absent"
                )
                continue
            return value, f"{dotted} = {value}"
    return None, None


def derive_declared(config: dict[str, Any]) -> dict[str, Any]:
    """The whole declared block, with every field's provenance written next to it.

    Anything not literally stated by the config is None. ``declared_fqns`` is None
    *always*: an HF config states hyperparameters, not module FQNs, and guessing
    the FQN set from layer counts would manufacture the very denominator
    SaveCompletenessGate exists to demand.
    """
    notes: list[str] = []

    num_experts, experts_basis = _scoped_int(config, _EXPERT_COUNT_KEYS, "num_experts", notes)
    if num_experts is None:
        num_experts = 0
        experts_basis = (
            f"no {'/'.join(_EXPERT_COUNT_KEYS)} key (non-null) in text_config or at "
            f"top level — the config declares no routed-expert count; treated as "
            f"dense (num_experts=0). That is the config's statement, not an "
            f"inference from the checkpoint. If this model IS MoE under key names "
            f"this probe does not know, the dense classification is wrong and the "
            f"model_type/architectures line above is the evidence left to catch it."
        )

    num_moe_layers, layers_basis = _scoped_int(config, _MOE_LAYER_KEYS, "num_moe_layers", notes)
    if num_moe_layers is None:
        layers_basis = (
            "config states no num_moe_layers — left None. Unknown is not zero: "
            "gates will report coverage without a fabricated denominator rather "
            "than count every layer as MoE by architectural assumption."
        )

    return {
        "num_experts": num_experts,
        "num_moe_layers": num_moe_layers,
        "declared_fqns": None,
        "expected_expert_bytes": None,
        "basis": {
            "num_experts": experts_basis,
            "num_moe_layers": layers_basis,
            "declared_fqns": (
                "an HF config.json states hyperparameters, not module FQNs — the "
                "full declared tensor set is NOT derivable from it. Left None. "
                "SaveCompletenessGate will answer VACUOUS (blocking), which is the "
                "honest result here: 'what is there matches what is there' is not "
                "a check, and this probe will not be the place that pretends it is."
            ),
            "expected_expert_bytes": (
                "expert byte volume is not *stated* anywhere in config.json. It is "
                "computable from hidden dims in principle, but a computed "
                "denominator is inference wearing declaration's clothes — left "
                "None. Consequence stated once, plainly: on a real MoE checkpoint "
                "the byte-volume gate abstains (SKIP). That gap is reported, not "
                "papered over."
            ),
        },
        "notes": notes,
    }


def build_inventory(meta: CheckpointMetadata) -> dict[str, Any]:
    """One-line facts about the real artifact, counted the way the gates count.

    Real tensors vs ``_extra_state`` byte blobs are separated with the same rule the
    gates apply, because the 8,042-of-8,970 lesson is that keys are not tensors.
    """
    real = [
        (fqn, tm)
        for fqn, tm in meta.tensors.items()
        if not (tm.is_extra_state or "_extra_state" in fqn)
    ]
    dtype_hist: dict[str, int] = {}
    total_bytes = 0
    uncounted = 0
    with_sid = 0
    for _fqn, tm in real:
        dtype_hist[tm.dtype] = dtype_hist.get(tm.dtype, 0) + 1
        width = _DTYPE_BYTES.get(tm.dtype)
        if width is None:
            # TensorMeta.nbytes prices an unknown dtype at 4 B/elem without saying
            # so (its .get default). The byte gate inherits that approximation;
            # this inventory refuses to print it as fact.
            uncounted += 1
        else:
            total_bytes += math.prod(tm.shape) * width
        if tm.storage_id is not None:
            with_sid += 1
    return {
        "origin": meta.origin,
        "format": meta.format,
        "entries_total": len(meta.tensors),
        "real_tensors": len(real),
        "extra_state_blobs": len(meta.tensors) - len(real),
        "metadata_implied_bytes": total_bytes,
        "uncounted_unknown_dtype_tensors": uncounted,
        "dtypes": dict(sorted(dtype_hist.items())),
        "with_storage_id": with_sid,
        "without_storage_id": len(real) - with_sid,
    }


def build_context(
    meta: CheckpointMetadata, declared: dict[str, Any], config_path: Path
) -> CheckpointGateContext:
    """Assemble the gate context from the two independent halves.

    Per-tensor mapping mirrors ``CheckpointGateContext.from_path`` exactly, but the
    declared fields come from ``derive_declared`` — NOT from ``load_manifest``. The
    run manifest is produced by the same training stack that produced the
    checkpoint; a denominator that can be wrong in exactly the same way as the thing
    it adjudicates is not a denominator.
    """
    tensors = tuple(
        TensorMeta(
            fqn=fqn,
            shape=tuple(tm.shape),
            dtype=tm.dtype,
            storage_id=tm.storage_id,
            kind="extra_state" if (tm.is_extra_state or "_extra_state" in fqn) else "tensor",
        )
        for fqn, tm in meta.tensors.items()
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=declared["declared_fqns"],
        num_experts=declared["num_experts"],
        num_moe_layers=declared["num_moe_layers"],
        expected_expert_bytes=declared["expected_expert_bytes"],
        # One origin field, both provenances: what was measured vs what was declared.
        origin=f"{meta.origin} [metadata={meta.format}; declared={config_path}]",
    )


# ---------------------------------------------------------------------------
# MUST_FIRE control on real content: --inject-alias N
# ---------------------------------------------------------------------------


def run_alias_control(
    ctx: CheckpointGateContext, n: int, baseline: GateResult | None
) -> dict[str, Any]:
    """Collapse N real expert-shaped tensors onto one storage_id; demand a block.

    This is the incident's mechanism (many names, one byte span) reconstructed
    inside a context built from the real checkpoint — the fixture lesson, re-proven
    on tensors the framework's author never synthesized. The gate must BLOCK. A
    quiet gate here means the aliasing detector is dead on real metadata, and that
    is a finding about the framework, not the checkpoint.
    """
    experts = _expert_weights(ctx)  # the gate's own selection, verbatim
    if not experts:
        return {
            "status": "skipped",
            "reason": (
                f"the checkpoint holds zero expert-shaped tensors by the gate's "
                f"own selection (declared num_experts={ctx.num_experts}); there is "
                f"nothing to alias. On a dense artifact this control is "
                f"inapplicable — and an inapplicable control must say so, because "
                f"a control that quietly 'passes' over nothing is the all([]) "
                f"result one level up."
            ),
        }
    groups: dict[str, list[TensorMeta]] = {}
    for t in experts:
        m = _SHARD_SUFFIX_RE.match(t.fqn)
        if m:
            groups.setdefault(m.group("stem"), []).append(t)
    eligible = [members for members in groups.values() if len(members) >= 2]
    if not eligible:
        return {
            "status": "skipped",
            "reason": (
                f"{len(experts)} expert-shaped tensor(s) exist, but none sit in a "
                f"sharded group of >= 2 (layout reads as fused). "
                f"ExpertDistinctnessGate checks storage distinctness only within "
                f"sharded groups; there is no fused-layout aliasing check for this "
                f"control to exercise (see # NOTES). The control is inapplicable "
                f"here, stated."
            ),
        }

    members = max(eligible, key=len)
    targets = members[: min(n, len(members))]
    target_fqns = {t.fqn for t in targets}
    injected_ctx = dataclasses.replace(
        ctx,
        tensors=tuple(
            dataclasses.replace(t, storage_id=_ALIAS_STORAGE_ID) if t.fqn in target_fqns else t
            for t in ctx.tensors
        ),
    )
    result = ExpertDistinctnessGate().run(injected_ctx)
    fired = result.blocking
    confounded = baseline is not None and baseline.blocking
    return {
        "status": "fired" if fired else "not_fired",
        "requested": n,
        "aliased": len(targets),
        "verdict": result.verdict.value,
        "detail": result.detail,
        "aliased_fqns": sorted(target_fqns)[:8],
        # Base rates matter: if the unmodified artifact already blocks this gate,
        # the control shows the gate blocks, not that the injection caused it.
        "confounded": confounded,
        "confound_note": (
            "baseline (unmodified) run was already blocking; this run shows the "
            "gate blocks on the injected artifact, not that aliasing is why"
        )
        if confounded
        else "",
        "aliasing_leg_observed": ("storages" in result.detail) or ("alias" in result.detail),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _h(title: str) -> None:
    print(f"\n== {title} " + "=" * max(4, 68 - len(title)))


def _print_inventory(inv: dict[str, Any]) -> None:
    _h(f"inventory: {inv['origin']}")
    gib = inv["metadata_implied_bytes"] / float(1 << 30)
    pct = (100.0 * inv["with_storage_id"] / inv["real_tensors"]) if inv["real_tensors"] else 0.0
    print(f"  format                  : {inv['format']}")
    print(
        f"  named metadata entries  : {inv['entries_total']:,} total -> "
        f"{inv['real_tensors']:,} tensors + {inv['extra_state_blobs']:,} extra_state "
        f"byte blobs (blobs excluded from every tensor count below)"
    )
    caveat = (
        f"; {inv['uncounted_unknown_dtype_tensors']:,} tensor(s) EXCLUDED — dtype not "
        f"in the gates' byte table"
        if inv["uncounted_unknown_dtype_tensors"]
        else ""
    )
    print(
        f"  metadata-implied bytes  : {inv['metadata_implied_bytes']:,} "
        f"({gib:.2f} GiB) over known-dtype tensors{caveat}"
    )
    dtypes = ", ".join(f"{name} x{count:,}" for name, count in inv["dtypes"].items()) or "<none>"
    print(f"  dtypes                  : {dtypes}")
    print(
        f"  storage_id coverage     : {inv['with_storage_id']:,}/{inv['real_tensors']:,} "
        f"real tensors ({pct:.1f}%)"
    )
    if inv["without_storage_id"]:
        print(
            f"  WARNING: {inv['without_storage_id']:,} tensor(s) carry storage_id=None; "
            f"for those, storage aliasing is UNDETECTABLE from metadata — None reads "
            f"as 'unknown', never as 'distinct'"
        )


def _print_declared(declared: dict[str, Any], config_path: Path, config: dict[str, Any]) -> None:
    _h(f"declared block (independent source: {config_path})")
    bits = []
    if isinstance(config.get("model_type"), str):
        bits.append(f"model_type={config['model_type']!r}")
    arch = config.get("architectures")
    if isinstance(arch, list) and arch:
        bits.append(f"architectures={arch!r}")
    if bits:
        # Display only. A MoE model whose expert keys this probe does not know is
        # classified dense; this line is the evidence left in view to falsify that.
        print(
            "  identity                : "
            + ", ".join(bits)
            + "  (display only — never fed to a gate)"
        )
    basis = declared["basis"]
    print(f"  num_experts             : {declared['num_experts']}\n      <- {basis['num_experts']}")
    print(
        f"  num_moe_layers          : {declared['num_moe_layers']}"
        f"\n      <- {basis['num_moe_layers']}"
    )
    print(f"  declared_fqns           : None\n      <- {basis['declared_fqns']}")
    print(f"  expected_expert_bytes   : None\n      <- {basis['expected_expert_bytes']}")
    for note in declared["notes"]:
        print(f"  note: {note}")


def _print_control(control: dict[str, Any]) -> None:
    _h("MUST_FIRE control: --inject-alias")
    if control["status"] == "skipped":
        print(f"  SKIPPED — {control['reason']}")
        return
    print(
        f"  rewrote {control['aliased']} expert tensor(s) (requested {control['requested']}) "
        f"onto one storage_id:"
    )
    for fqn in control["aliased_fqns"]:
        print(f"      {fqn}")
    print(
        f"  re-ran checkpoint.expert_distinctness on the injected context: "
        f"{control['verdict']} — {control['detail']}"
    )
    if control["status"] == "fired":
        print(
            "  control FIRED: the gate blocked on aliased storage derived from this real artifact"
        )
        if control["confounded"]:
            print(f"  CONFOUNDED: {control['confound_note']}")
    else:
        print(
            "  control DID NOT FIRE: the detector accepted artificially aliased "
            "experts built from real metadata — checkpoint.expert_distinctness "
            "cannot be trusted on this artifact. Exit blocks regardless of the sweep."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="real_checkpoint_probe",
        description="Run the shipped checkpoint gates against a real on-disk "
        "checkpoint, with the declared block derived from an independent HF "
        "config.json. Exit 0 = cleared; 1 = blocking verdict reached; "
        "3 = the probe could not measure.",
    )
    p.add_argument("ckpt_dir", help="DCP directory, or safetensors file/directory (read-only)")
    p.add_argument(
        "--config",
        default=None,
        metavar="CONFIG_JSON",
        help="HF config.json supplying the declared block "
        "(default: CKPT_DIR/config.json, or beside a single-file checkpoint)",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        default=None,
        metavar="OUT",
        help="write the full machine-readable report to OUT",
    )
    p.add_argument(
        "--inject-alias",
        type=int,
        default=None,
        metavar="N",
        help="MUST_FIRE control: force N expert-shaped tensors to share one "
        "storage_id and assert checkpoint.expert_distinctness blocks (N >= 2)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.inject_alias is not None and args.inject_alias < 2:
        # A span cannot alias with itself; N=1 would control nothing while looking
        # like a control. Usage errors are argparse's exit code (2), on purpose.
        parser.error("--inject-alias N requires N >= 2")

    ckpt_path = Path(args.ckpt_dir)
    config_path = (
        Path(args.config)
        if args.config
        else (
            ckpt_path / "config.json" if ckpt_path.is_dir() else ckpt_path.with_name("config.json")
        )
    )

    try:
        meta = _measure_checkpoint(ckpt_path)
        config = _read_config(config_path)
        inventory = build_inventory(meta)
        declared = derive_declared(config)
        ctx = build_context(meta, declared, config_path)
    except ProbeUnmeasured as exc:
        print(f"probe could not measure: {exc}", file=sys.stderr)
        return EXIT_UNMEASURED
    except Exception:
        # A bug in the probe is also "could not measure" — never "clear".
        traceback.print_exc()
        print(
            "probe could not measure: unexpected failure above "
            "(a probe bug is not a checkpoint verdict)",
            file=sys.stderr,
        )
        return EXIT_UNMEASURED

    try:
        return _deliver(
            args=args,
            ckpt_path=ckpt_path,
            config_path=config_path,
            config=config,
            inventory=inventory,
            declared=declared,
            ctx=ctx,
        )
    except ProbeUnmeasured as exc:
        print(f"probe could not measure: {exc}", file=sys.stderr)
        return EXIT_UNMEASURED
    except Exception:
        # The fail-closed rule cannot stop at measurement. From here the probe is
        # rendering and ruling on real measurements; if it dies in this phase the
        # interpreter's own exit status is 1, and every pipeline that has read
        # the contract files exit 1 as "measured, and the answer blocks". A
        # probe bug must never wear a checkpoint verdict — it is exit 3, with
        # the traceback in view.
        traceback.print_exc()
        print(
            "probe could not measure: unexpected failure after measurement above "
            "(a probe bug is not a checkpoint verdict)",
            file=sys.stderr,
        )
        return EXIT_UNMEASURED


def _deliver(
    *,
    args: argparse.Namespace,
    ckpt_path: Path,
    config_path: Path,
    config: dict[str, Any],
    inventory: dict[str, Any],
    declared: dict[str, Any],
    ctx: CheckpointGateContext,
) -> int:
    """Render, rule, and report on a checkpoint that is already measured.

    Everything after measurement lives behind this one call so ``main`` can
    hold a single invariant around the entire phase: a failure inside is a
    probe failure (exit 3), never a checkpoint verdict (exit 1). The gate
    sweep, summary wording, blocking-reason list and report schema are exactly
    what they were when this body lived inline in ``main``.
    """
    _print_inventory(inventory)
    _print_declared(declared, config_path, config)

    results: list[GateResult] = []
    invariant_breaches: list[str] = []
    _h(f"gates ({len(_CHECKPOINT_GATES)} shipped checkpoint gates; population explicit)")
    for gate_cls in _CHECKPOINT_GATES:
        result = gate_cls().run(ctx)  # run(), not check(): ERROR on exception, timed
        results.append(result)
        print(f"  {result.render()}")
        # The framework states Gate.ok cannot return PASS over zero coverage.
        # "States" is doing work in that sentence, so verify it: the day this
        # fires, the framework has its own founding bug, and the probe reports it
        # instead of inheriting it.
        if result.verdict is Verdict.PASS and result.coverage.checked == 0:
            invariant_breaches.append(result.gate_id)

    control: dict[str, Any] | None = None
    if args.inject_alias is not None:
        baseline = next((r for r in results if r.gate_id == ExpertDistinctnessGate.id), None)
        control = run_alias_control(ctx, args.inject_alias, baseline)
        _print_control(control)

    blocking = [r for r in results if r.blocking]
    skipped = [r for r in results if r.verdict is Verdict.SKIP]
    passed = [r for r in results if r.verdict is Verdict.PASS]
    reasons = [f"{r.gate_id}={r.verdict.value}" for r in blocking]
    if control is not None and control["status"] == "not_fired":
        reasons.append("--inject-alias MUST_FIRE control stayed quiet on real content")
    elif control is not None and control["status"] == "skipped":
        # A requested MUST_FIRE control that could not run leaves its claim —
        # that the aliasing detector fires on real metadata — unverified on
        # this artifact. The module's exit-0 contract is "every requested
        # control fired", so a skipped control must weigh against CLEAR exactly
        # as a quiet one does; anything softer is the control "passing" over
        # zero exercised units. (As shipped, checkpoint.save_complete blocks
        # every run first, so this branch cannot change today's exit code. It
        # exists for the day an independent declared_fqns source makes CLEAR
        # reachable — which is precisely when it will matter.)
        reasons.append(
            "--inject-alias MUST_FIRE control was requested but reported 'skipped' — "
            "the claim it exists to test is unverified on this artifact"
        )
    reasons.extend(
        f"{gid}: PASS over 0 checked units — the framework's own coverage rule did not hold"
        for gid in invariant_breaches
    )
    exit_code = EXIT_BLOCKED if reasons else EXIT_CLEAR

    _h("summary")
    print(
        f"  gates run               : {len(results)} "
        f"(population enumerated by this file, not inferred from the registry)"
    )
    print(f"  PASS (non-vacuous)      : {len(passed)}")
    print(f"  SKIP (reason stated)    : {len(skipped)}")
    for r in skipped:
        print(f"      {r.gate_id}: {r.detail}")
    print(f"  blocking                : {len(blocking)}")
    for reason in reasons:
        print(f"      {reason}")
    if control is not None:
        print(
            "  --inject-alias control  : "
            + {"fired": "FIRED", "not_fired": "DID NOT FIRE", "skipped": "SKIPPED"}[
                control["status"]
            ]
        )
    if exit_code == EXIT_CLEAR:
        print(
            "PROBE VERDICT: CLEAR (exit 0) — every gate reached a real verdict or "
            "stated why it abstained; nothing passed over zero examined units."
        )
    else:
        print(f"PROBE VERDICT: NOT CLEARED (exit {EXIT_BLOCKED})")
        print(
            "  NOT CLEARED is not an accusation of corruption. It is the honest "
            "extent of what is verifiable about this artifact on independent "
            "evidence: gates above either found defects or could not run for want "
            "of a denominator the config does not state. Any consumer requiring "
            "'verified' is unserved until the blocking items close — a run "
            "manifest carrying declared_fqns would let checkpoint.save_complete "
            "actually check completeness instead of refusing to."
        )

    report = {
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "inventory": inventory,
        "declared": declared,
        "gates": [r.to_dict() for r in results],
        "control": control,
        "framework_invariant_breaches": invariant_breaches,
        "blocking_reasons": reasons,
        "exit_code": exit_code,
    }
    if args.json_out:
        out = Path(args.json_out)
        try:
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"could not write --json report {out}: {exc}", file=sys.stderr)
            return EXIT_UNMEASURED  # measurement happened, the deliverable did not
        print(f"\n  report written: {out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
