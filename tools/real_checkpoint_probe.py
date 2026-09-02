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
   verdict (PASS over non-vacuous coverage, or SKIP with a stated reason — and
   "stated" is re-verified at the summary, not inherited: a SKIP whose reason
   string is blank is printed as reason-MISSING and blocks CLEAR) and every
   requested control fired. Any blocking verdict — FAIL, VACUOUS, UNDERCOVERED,
   ERROR — means NOT CLEARED and exit ``1``. "Fired" is a CAUSAL claim,
   adjudicated in :func:`run_alias_control`: the unmodified artifact must NOT
   have blocked the gate, and the injected artifact must have returned FAIL.
   A block with verdict ERROR (a crash) or VACUOUS/UNDERCOVERED (a coverage
   tripwire), and any block the baseline already produced, is reported
   INCONCLUSIVE — shown, named, and weighed against CLEAR like any unmet
   control, because crediting it would certify a detector through a confound.
   A VACUOUS gate on a real 16 GB
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

# The CLI wrapper's own imports. The measurement helpers this file drives are
# re-exported from foundationscale.gates.probe below.
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from foundationscale.gates.checkpoint_gates import (
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    FirstSaveGate,
    SaveCompletenessGate,
)
from foundationscale.gates.core import Gate, GateResult, Verdict

# ---------------------------------------------------------------------------
# #219: the measurement helpers moved into the package.
#
# derive_declared, run_alias_control and the module-level names they close
# over used to be defined HERE, in a script -- and
# foundationscale.gates.adjudication imported them FROM here, behind a
# try/except ImportError ladder. tools/ is not distributed
# ([tool.setuptools.packages.find] where = ["src"]), so on a clean pip
# install the ladder always fell through and the library's decision plane
# refused every call: the headline API was unusable exactly where it was
# installed. The dependency is now inverted, per the same rule as
# T2_lib_script_boundary#0 for live_save_gate / adjudication: the library
# owns the logic, the script is a thin CLI. The helpers live in
# foundationscale.gates.probe and are re-exported here so every existing
# reference -- `from real_checkpoint_probe import derive_declared`, the
# callers below, the mutation anchors -- resolves unchanged. The CLI, its
# exit codes, and its output are untouched.
#
# CAUTION for test authors: re-export binds a NAME, not the defining module's
# globals. Monkeypatching `real_checkpoint_probe.derive_declared` does NOT
# change what library consumers read, because they read
# foundationscale.gates.probe's (or foundationscale.gates.adjudication's)
# globals. Patch those modules instead.
# ---------------------------------------------------------------------------
# The list is the library module's ENTIRE unconditional top-level surface (29
# names), not the subset this CLI happens to call. A wrapper that re-exports
# only what it uses drifts one name at a time: the first draft of this block
# omitted eleven, and the suite went red on exactly one of them
# (`_ENABLE_MOE_BLOCK_KEY`) -- which is what a partial surface looks like,
# silent until some caller reaches for the twelfth.
#
# That near-miss is why tests/test_reexport_surface.py was generalised from one
# boundary pair to two (#205: a fix that leaves the class open is incomplete).
# It derives this surface from foundationscale.gates.probe's own AST and fails
# if any of it is missing here, so the next name added to the library is caught
# on the commit that adds it rather than by the caller who needs it. Verified
# by removing those same eleven from this module in-process: the control goes
# red naming exactly them.
from foundationscale.gates.probe import (
    _ALIAS_STORAGE_ID,  # noqa: F401 -- re-export; see the boundary comment above
    _DTYPE_BYTES,  # noqa: F401 -- re-export; see the boundary comment above
    _ENABLE_MOE_BLOCK_KEY,  # noqa: F401 -- re-export; see the boundary comment above
    _EXPERT_COUNT_KEYS,  # noqa: F401 -- re-export; see the boundary comment above
    _MOE_LAYER_KEYS,  # noqa: F401 -- re-export; see the boundary comment above
    CheckpointFormatError,  # noqa: F401 -- re-export; see the boundary comment above
    CheckpointMetadata,  # noqa: F401 -- re-export; see the boundary comment above
    ProbeUnmeasured,
    TensorMeta,  # noqa: F401 -- re-export; see the boundary comment above
    _census_expert_family,
    _enable_moe_block_flag,  # noqa: F401 -- re-export; see the boundary comment above
    _expert_named,  # noqa: F401 -- re-export; see the boundary comment above
    _expert_weights,  # noqa: F401 -- re-export; see the boundary comment above
    _matches_expert_family,  # noqa: F401 -- re-export; see the boundary comment above
    _measure_checkpoint,
    _read_config,
    _scoped_int,  # noqa: F401 -- re-export; see the boundary comment above
    _split_expert_layouts,  # noqa: F401 -- re-export; see the boundary comment above
    build_context,
    build_inventory,
    derive_declared,
    read_metadata,  # noqa: F401 -- re-export; see the boundary comment above
    run_alias_control,
)

EXIT_CLEAR = 0
EXIT_BLOCKED = 1
EXIT_UNMEASURED = 3  # 2 belongs to argparse; 3 keeps "could not measure" its own signal

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
    # The two verdicts print side by side BEFORE any interpretation, because
    # "the gate blocked" and "the injection made the gate block" are different
    # claims with different denominators, and only the second credits the control.
    baseline_text = (
        control["baseline_verdict"] if control["baseline_verdict"] is not None else "<not recorded>"
    )
    print(f"  baseline verdict (unmodified artifact)  : {baseline_text}")
    print(f"  verdict on the alias-injected artifact  : {control['verdict']} — {control['detail']}")
    if control["status"] == "fired":
        print(
            "  control FIRED: clean baseline, FAIL on injection — the block is "
            "attributable to the injected aliasing, on evidence printed above."
        )
    elif control["status"] == "inconclusive":
        print(f"  control INCONCLUSIVE: {control['inconclusive_reason']}.")
        print(
            "  An inconclusive control has NOT fired: the run blocked, but crediting "
            "the detection would assert a cause that was never observed. The exit "
            "code below treats it accordingly."
        )
    else:
        print(
            "  control DID NOT FIRE: the detector returned a non-blocking verdict on "
            "artificially aliased experts built from real metadata — "
            "checkpoint.expert_distinctness cannot be trusted on this artifact. "
            "Exit blocks regardless of the sweep."
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
        # The census is measured BEFORE the declared block is derived: a dense
        # declaration is only minted when the config affirmatively says dense
        # AND the artifact holds zero expert-family tensors, so derive_declared
        # needs the artifact half measured up front.
        census, census_sample = _census_expert_family(meta)
        declared = derive_declared(
            config, expert_family_census=census, expert_family_sample=census_sample
        )
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
    sweep and summary wording grew from the body that lived inline in ``main``;
    the blocking-reason list and report schema have since gained two honesty
    rules the inline version lacked, both enforced below rather than assumed:
    a skip is "stated" only if its reason string is non-blank, and a MUST_FIRE
    control is credited only when its block is attributable (see
    :func:`run_alias_control`).
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
    # An abstention is only as good as its stated reason. Gate.skip() enforces
    # that at construction, but a hand-built GateResult — the exact move the
    # gate-contract docstring warns against — can carry Verdict.SKIP with a
    # blank detail, and Gate.run forwards it untouched. The probe prints
    # "SKIP (reason stated)" and counts SKIP toward the CLEAR wording "stated
    # why it abstained"; both are CLAIMS MADE HERE, so both are VERIFIED HERE,
    # at the point of claim, rather than inherited from the helper some gate
    # may have bypassed. A blank-reason SKIP is not a stated abstention: it
    # prints under its own honest header and weighs against CLEAR.
    unstated_skips = [r for r in skipped if not r.detail.strip()]
    stated_skips = [r for r in skipped if r.detail.strip()]
    reasons = [f"{r.gate_id}={r.verdict.value}" for r in blocking]
    if control is not None and control["status"] == "not_fired":
        reasons.append("--inject-alias MUST_FIRE control stayed quiet on real content")
    elif control is not None and control["status"] == "inconclusive":
        # The injected run blocked but never earned attribution to the
        # injection (run_alias_control names which leg failed). The exit-0
        # contract is "every requested control FIRED"; inconclusive is not
        # fired, and crediting it would repair a confound by minting a pass.
        reasons.append(
            "--inject-alias MUST_FIRE control was INCONCLUSIVE, not credited as fired — "
            + control["inconclusive_reason"]
        )
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
        f"{r.gate_id}: SKIP with no stated reason — an unexplained abstention is a "
        f"hole. Gate.skip rejects blank reasons, so this result bypassed the "
        f"helper, and the probe re-checks the claims it prints rather than "
        f"inheriting them"
        for r in unstated_skips
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
    print(f"  SKIP (reason stated)    : {len(stated_skips)}")
    for r in stated_skips:
        print(f"      {r.gate_id}: {r.detail}")
    if unstated_skips:
        # Its own header, because "reason stated" is a claim with a denominator
        # and these results are outside it. They already entered `reasons` above.
        print(f"  SKIP (reason MISSING)   : {len(unstated_skips)}")
        for r in unstated_skips:
            print(f"      {r.gate_id}: <no reason stated>")
    print(f"  blocking                : {len(blocking)}")
    for reason in reasons:
        print(f"      {reason}")
    if control is not None:
        print(
            "  --inject-alias control  : "
            + {
                "fired": "FIRED",
                "not_fired": "DID NOT FIRE",
                "inconclusive": "INCONCLUSIVE (blocked, but not attributable — not credited)",
                "skipped": "SKIPPED",
            }[control["status"]]
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
        # Downstream consumers must not have to re-derive WHICH gates were
        # counted as abstentions from per-gate verdicts and blank details —
        # the denominator of the "stated" claim travels explicitly.
        "unstated_skip_gates": [r.gate_id for r in unstated_skips],
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
