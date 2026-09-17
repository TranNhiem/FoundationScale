"""T1-4: save -> load bit-parity adjudication for the FoundationScale verification matrix.

Row claim
    reloaded weights are bit-identical (save -> load parity)
Run arm
    save then load, compare_keys EXACT (strict zero-tolerance policy, bitwise)
Control
    perturb exactly one tensor, parity MUST go RED

This instrument runs two arms against a real on-disk checkpoint directory of
safetensors:

    identity : copy the checkpoint, load and re-SAVE every shard, then compare the
               original against that round trip under a strict bit-exact tolerance
               policy (max_abs_diff=0, cosine>=1.0, rel_frob=0). Comparing the
               checkpoint against itself would ask only whether one file reads the
               same twice; the row claims ``save then load``, so the arm must write.
    perturbed: the SAME round trip, plus exactly one element of exactly one tensor
               mutated, compared against the original under the same policy. Both
               arms are materialised identically, so the single element is the only
               difference between them and the control is attributable.

WHY THE CONTROL PERTURBATION IS EXACTLY 1 ULP
    Unit-in-the-last-place is the smallest nonzero perturbation a dtype can
    represent; for float dtypes a +1 ULP step changes exactly the last bit of one
    element's binary encoding, and for integer/bool dtypes flipping one element is
    the analogous single-bit change. The control exists to prove the comparator is
    *exact*, not merely *tight*. Any perturbation larger than 1 ULP cannot separate
    the two instruments: a comparator with tolerance, say, max_abs_diff=1e-6 would
    also fail a 1e-5 perturbation and therefore look "bit-exact" on this row while
    silently acquitting genuine bit-drift below its floor. Only the minimum
    representable delta forces the issue — if the comparator certifies equality
    between two tensors that differ in exactly one bit pattern, its claim of
    bit-identity is falsified at the limit of representability. 1 ULP is also
    unambiguous to apply (nextafter toward +inf) and immune to
    magnitude/scale-dependent arguments. Therefore: identity arm EXACT AND the
    1-ULP control RED is the only combination proving a save->load round trip is a
    bitwise identity under a detector that can see the smallest possible lie.

VERDICT ORDER (load-bearing)
    96  arms compared different key sets (not the same instrument twice)
    96  identity arm compared zero elements (report.is_vacuous; all([]) shape)
    95  perturbation could not be applied at all (absence, not failure)
    5   identity arm NOT ok (save->load changed bits)
    5   perturbed arm IS ok (control did not fire; detector blind)
    0   otherwise

EXIT CONTRACT
    GREEN=0  RED=5  UNMEASURED=95  REFUSE=96.  Never 1 or 2.
    Unmet preconditions are 95/96, never 5. UNMEASURED is not PASS.
    Required args with no defensible default (--checkpoint, --work-dir or
    --out-dir) REFUSE(96) naming them, rather than letting argparse mint 2.
    An escaping exception is classified 95 or 96, with the traceback; the process
    never exits 1 or 2.

REPORTING SEAM
    A completed measurement writes one payload for each declared arm through
    run_row.write_arm_payload. The destination filename is derived inside that
    runner helper, never chosen here. A boundary abort writes the same two arm
    files with UNMEASURED or CANNOT_MEASURE status, because an absent file is
    indistinguishable from an arm that never attempted to report.

    The wrapper's status is the source arm's own PASS or FAIL, not the row's
    differential verdict. In particular, the perturbed arm reporting FAIL is the
    expected control shape and can underpin a GREEN row verdict while remaining
    truthful about what that arm observed.

    torch / safetensors are imported lazily inside the functions that need them,
    so --self-test runs on a bare laptop with no GPU, model, network or durable
    filesystem. Its reporting controls use only a TemporaryDirectory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, NoReturn

# The row directory is a sibling import root, exactly as `python3 t1_4_...py`
# gives it. The boundary classifier lives there; four other rows already share
# it, and this row used to carry a local copy that adjudicated every escape RED.
# The runner-side payload writer must use the identical direct-file bootstrap:
# a package-relative import would work only when this row is launched a certain
# way, which silently makes reporting depend on operator choice.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_row import write_arm_payload  # noqa: E402
from t1_interpreter_floor import classify_boundary_exception  # noqa: E402

EXIT_GREEN = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

_RC_NAMES = {
    EXIT_GREEN: "GREEN",
    EXIT_RED: "RED",
    EXIT_UNMEASURED: "UNMEASURED",
    EXIT_REFUSE: "REFUSE",
}

ROW_ID = "T1-4"


# ---------------------------------------------------------------------------
# CLI plumbing: argparse must never exit 2
# ---------------------------------------------------------------------------


class _RefusingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose errors REFUSE (96) instead of argparse's default 2."""

    def error(self, message: str) -> NoReturn:
        sys.stderr.write(f"REFUSE({EXIT_REFUSE}): {message}\n")
        raise SystemExit(EXIT_REFUSE)


def _build_parser() -> _RefusingArgumentParser:
    parser = _RefusingArgumentParser(
        prog="t1_4_save_load_parity",
        description=(
            "T1-4: adjudicate save->load bit-parity of a safetensors checkpoint. "
            "The identity arm re-saves every shard and compares that round trip "
            "against the original (compare_keys EXACT); the control arm does the "
            "same round trip with one element stepped +1 ULP and must go RED."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help=(
            "real checkpoint directory of safetensors to round-trip "
            "(required; no defensible default)"
        ),
    )
    parser.add_argument(
        "--work-dir",
        metavar="PATH",
        default=None,
        help="directory where the perturbed copy is written (required; no defensible default)",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        default=None,
        help=("directory receiving one runner payload per arm (required; no defensible default)"),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic controls against the pure verdict function; no other args needed",
    )
    return parser


# ---------------------------------------------------------------------------
# The measurement (real arms; lazy heavy imports)
# ---------------------------------------------------------------------------


def _strict_policy() -> Any:
    """Bit-exact compare_keys policy: EXACT or nothing."""
    from foundationscale.verify.parity import TolerancePolicy, Tolerances

    strict = Tolerances(
        name="t1-4-compare-keys-exact",
        max_abs_diff=0.0,
        min_cosine=1.0,
        max_rel_frob=0.0,
    )
    return TolerancePolicy(default=strict)


def _resave_all(target_dir: Path) -> int:
    """Load and re-save EVERY safetensors file in place: the actual save->load round trip.

    Comparing a checkpoint against itself would be very nearly a tautology -- it asks
    whether reading one file twice returns the same bytes, not whether writing what was
    read reproduces it. The row's claim is ``save then load``, so the identity arm has to
    write. This is also what keeps the two arms honest as a differential: BOTH arms are
    materialised by this function, so the only thing that distinguishes them is the single
    perturbed element. Without it the perturbed arm would be the only one carrying re-save
    artifacts, and a dtype or metadata change on write would be scored as the control
    firing when the control had in fact not been tested.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    count = 0
    for file_path in sorted(target_dir.rglob("*.safetensors")):
        tensors = load_file(str(file_path))
        with safe_open(str(file_path), framework="pt") as handle:
            metadata = handle.metadata()
        save_file(tensors, str(file_path), metadata=metadata)
        count += 1
    return count


def _apply_ulp_perturbation(copy_dir: Path) -> dict[str, Any]:
    """Perturb exactly ONE element of ONE tensor in the copied checkpoint.

    Deterministic: first ``*.safetensors`` file (sorted, recursive), first tensor
    key (sorted) that qualifies. Float dtypes are preferred: one finite element is
    stepped +1 ULP via ``torch.nextafter`` toward +inf (a single last-bit flip of
    its encoding). If no float tensor exists, an integer tensor gets one element
    XORed with 1; a bool tensor gets one element inverted. If no tensor with at
    least one element exists at all, nothing is written and the returned dict
    reports ``applied=False`` — an unmeasured control, never a failure finding.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    info: dict[str, Any] = {
        "applied": False,
        "key": None,
        "file": None,
        "element_index": None,
        "before": None,
        "after": None,
        "reason": "",
    }
    files = sorted(copy_dir.rglob("*.safetensors"))
    if not files:
        info["reason"] = f"no .safetensors files under copied checkpoint {copy_dir}"
        return info

    def _is_int_tensor(t: Any) -> bool:
        return not t.is_floating_point() and not t.is_complex() and t.dtype != torch.bool

    def _mutate(t: Any) -> tuple[int, Any, Any] | None:
        """Return (flat_index, before_scalar_tensor, after_scalar_tensor) or None."""
        flat = t.reshape(-1)
        if t.is_floating_point():
            finite_idx = torch.isfinite(flat).nonzero(as_tuple=False).reshape(-1)
            if finite_idx.numel() == 0:
                return None
            idx = int(finite_idx[0].item())
            before = flat[idx].clone()
            after = torch.nextafter(before, torch.full_like(before, float("inf")))
        elif _is_int_tensor(t):
            idx = 0
            before = flat[idx].clone()
            after = before ^ 1
        else:  # torch.bool
            idx = 0
            before = flat[idx].clone()
            after = torch.logical_not(before)
        return idx, before, after

    predicates = (
        lambda t: t.is_floating_point(),
        _is_int_tensor,
        lambda t: t.dtype == torch.bool,
    )
    for predicate in predicates:
        for file_path in files:
            tensors = load_file(str(file_path))
            for key in sorted(tensors):
                tensor = tensors[key]
                if tensor.numel() == 0 or not predicate(tensor):
                    continue
                mutated = _mutate(tensor)
                if mutated is None:
                    continue
                idx, before, after = mutated
                replacement = tensor.clone()
                replacement.reshape(-1)[idx] = after
                tensors[key] = replacement
                with safe_open(str(file_path), framework="pt") as handle:
                    metadata = handle.metadata()
                save_file(tensors, str(file_path), metadata=metadata)
                info.update(
                    applied=True,
                    key=key,
                    file=str(file_path.relative_to(copy_dir)),
                    element_index=idx,
                    before=before.item(),
                    after=after.item(),
                    reason="",
                )
                return info

    info["reason"] = (
        "no float, integer or bool tensor with at least one element found in "
        f"copied checkpoint {copy_dir} (no float tensor / empty checkpoint)"
    )
    return info


def _arm_payload(report: Any) -> dict[str, Any]:
    """Project a ParityReport into the JSON payload fields the verdict consumes."""
    return {
        "ok": bool(report.ok),
        "vacuous": bool(report.is_vacuous),
        "compared_keys": len(report.compared),
        "compared_elements": int(report.compared_elements),
        "keys": sorted(entry.key for entry in report.keys),
        "only_in_left": list(report.only_in_left),
        "only_in_right": list(report.only_in_right),
        "findings": [
            {"key": f.key, "status": f.status.value, "note": f.note} for f in report.findings
        ],
        "render": report.render(),
    }


def run_measurement(checkpoint: Path, work_dir: Path) -> dict[str, Any]:
    """Execute both arms and return the raw payload (pre-verdict)."""
    from foundationscale.verify.parity import compare_sources

    policy = _strict_policy()

    def _materialise(name: str) -> Path:
        out = work_dir / name
        if out.exists():
            shutil.rmtree(out)
        # Create the parent with a real syscall immediately before writing into it, rather
        # than trusting that the caller's earlier mkdir is still true. On a shared filesystem
        # the work dir can be removed by another host between the two, and copytree decides
        # whether to create the parent from a CACHED os.path.exists -- so it skips the
        # creation and then fails ENOENT on the child, which reads as a measurement failure
        # when it is only a stale dentry.
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint, out)
        _resave_all(out)
        return out

    # Arm 1: save -> load, nothing else touched. Compared against the ORIGINAL bytes, so a
    # write path that silently changes a dtype, drops metadata or reorders a shared tensor
    # is a RED here rather than something the control has to absorb.
    roundtrip_dir = _materialise("t1-4-roundtrip")
    identity_report = compare_sources(checkpoint, roundtrip_dir, policy=policy)

    # Arm 2: the SAME round trip plus exactly one perturbed element. The two arms are now
    # byte-identical in construction apart from that element, so a difference between their
    # verdicts is attributable to the perturbation and to nothing else.
    copy_dir = _materialise("t1-4-perturbed-copy")
    perturb_info = _apply_ulp_perturbation(copy_dir)
    perturbed_report = compare_sources(checkpoint, copy_dir, policy=policy)

    perturbed_arm = _arm_payload(perturbed_report)
    perturbed_arm.update(
        perturbation_applied=bool(perturb_info["applied"]),
        perturbed_key=perturb_info["key"],
        perturbed_file=perturb_info["file"],
        element_index=perturb_info["element_index"],
        before=perturb_info["before"],
        after=perturb_info["after"],
        perturbation_note=perturb_info["reason"],
    )

    return {
        "row": ROW_ID,
        "claim": "reloaded weights are bit-identical (save->load parity)",
        "checkpoint": str(checkpoint),
        "work_dir": str(work_dir),
        "roundtrip_copy": str(roundtrip_dir),
        "perturbed_copy": str(copy_dir),
        "policy": policy.describe(),
        "arms": {
            "identity": _arm_payload(identity_report),
            "perturbed": perturbed_arm,
        },
    }


# ---------------------------------------------------------------------------
# The verdict: PURE (data in -> (rc, payload) out). No torch, no filesystem.
# ---------------------------------------------------------------------------


def verdict(data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Adjudicate the row from payload fields only.

    Decision order is load-bearing; see the module docstring. Returns a NEW
    payload dict carrying a ``verdict`` record; the input is not mutated.
    """
    arms = data.get("arms") or {}
    identity = arms.get("identity") or {}
    perturbed = arms.get("perturbed") or {}

    identity_keys = set(identity.get("keys") or ())
    perturbed_keys = set(perturbed.get("keys") or ())

    if identity_keys != perturbed_keys:
        only_identity = sorted(identity_keys - perturbed_keys)[:5]
        only_perturbed = sorted(perturbed_keys - identity_keys)[:5]
        rc = EXIT_REFUSE
        reason = (
            "the two arms did not compare the same key set — this is not the same "
            f"instrument run twice (identity-only keys: {only_identity}, "
            f"perturbed-only keys: {only_perturbed}); refusing to adjudicate"
        )
    elif int(identity.get("compared_elements") or 0) == 0:
        rc = EXIT_REFUSE
        reason = (
            "identity arm compared ZERO elements (report.is_vacuous): an empty "
            "comparison is the all([]) shape, and it can never mint a pass; "
            "refusing to adjudicate a vacuous instrument"
        )
    elif not perturbed.get("perturbation_applied"):
        note = perturbed.get("perturbation_note") or "no reason recorded"
        rc = EXIT_UNMEASURED
        reason = (
            "the control perturbation could not be applied at all "
            f"({note}); without a live control the instrument's sensitivity is "
            "unmeasured — absence of evidence, not a failure finding"
        )
    elif not identity.get("ok"):
        rc = EXIT_RED
        reason = (
            "identity arm is NOT ok: a save->load round trip changed bits, so the "
            "claim 'reloaded weights are bit-identical' is falsified"
        )
    elif perturbed.get("ok"):
        rc = EXIT_RED
        reason = (
            "perturbed arm is ok although exactly one element was moved by the "
            "smallest representable amount (+1 ULP / one flipped integer-bool "
            "element): the control did not fire, so the detector cannot see a "
            "single-bit change and the whole row is instrumented by a blind gate"
        )
    else:
        rc = EXIT_GREEN
        reason = (
            "identity arm is bit-exact (compare_keys EXACT) and the +1 ULP "
            f"control fired (key={perturbed.get('perturbed_key')!r}, "
            f"element_index={perturbed.get('element_index')}, "
            f"before={perturbed.get('before')!r}, after={perturbed.get('after')!r})"
        )

    result = dict(data)
    result["verdict"] = {
        "row": ROW_ID,
        "rc": rc,
        "name": _RC_NAMES[rc],
        "reason": reason,
        "adjudicated_from_payload_fields_only": True,
    }
    return rc, result


# ---------------------------------------------------------------------------
# Runner reporting: one payload for each declared arm, independent of verdict
# ---------------------------------------------------------------------------


def _reported_arm_status_reason(arm: str, report: dict[str, Any]) -> tuple[str, str]:
    """Project the source arm's own outcome before the row's differential verdict.

    A comparator disagreement is FAIL at the source-arm seam even when it is the
    perturbed control and therefore expected by the row. RED is an adjudication
    over both arms, not a synonym for a single arm's observation; keeping these
    levels separate preserves the evidence without changing the measurement.
    """
    if not report:
        return (
            "UNMEASURED",
            f"the {arm} source arm report is absent; no arm outcome was observed",
        )
    if arm == "perturbed" and not report.get("perturbation_applied"):
        note = report.get("perturbation_note") or "no reason recorded"
        return (
            "UNMEASURED",
            "the control perturbation could not be applied "
            f"({note}), so this arm was never a live control",
        )
    if bool(report.get("ok")):
        status, outcome = "PASS", "compared bit-exact"
    else:
        status, outcome = "FAIL", "reported a difference"
    return (
        status,
        f"the {arm} source arm {outcome} under the strict bit-parity policy",
    )


def _write_arm_payloads(out_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    """Write the two source arms; the row verdict must never suppress a file."""
    arms = payload.get("arms") or {}
    written: dict[str, Path] = {}
    for arm in ("identity", "perturbed"):
        raw_report = arms.get(arm) if isinstance(arms, dict) else None
        report = dict(raw_report) if isinstance(raw_report, dict) else {}
        status, reason = _reported_arm_status_reason(arm, report)
        # An absent source report is supplied as an explicit empty dict rather than
        # None. Present-and-empty therefore means "there was nothing reportable",
        # while None would mean only "this row does not collect telemetry". Only
        # the first preserves that this arm failed to leave evidence.
        telemetry = _json_safe(report)
        # These arms execute in this Python process, so launcher_exit_code is left
        # absent rather than inventing a zero for a child process that never existed.
        written[arm] = write_arm_payload(
            out_dir,
            ROW_ID,
            arm,
            status=status,
            reason=reason,
            telemetry=telemetry,
        )
    return written


def _write_boundary_payloads(out_dir: Path, code: int, reason: str, error_trace: str) -> None:
    """Leave both arm files after a boundary abort, with classification as status."""
    status = "UNMEASURED" if code == EXIT_UNMEASURED else "CANNOT_MEASURE"
    for arm in ("identity", "perturbed"):
        write_arm_payload(
            out_dir,
            ROW_ID,
            arm,
            status=status,
            reason=(f"{reason}; no adjudicated source report reached the runner for the {arm} arm"),
            telemetry={"exception": _json_safe(error_trace)},
        )


# ---------------------------------------------------------------------------
# Self-test: synthetic verdict and reporting controls. No torch or model files.
# ---------------------------------------------------------------------------


def _synthetic_payload(
    *,
    identity_ok: bool = True,
    perturbed_ok: bool = False,
    identity_elements: int = 8,
    perturbed_elements: int = 8,
    identity_keys: tuple[str, ...] = ("layer.0.weight", "layer.1.weight"),
    perturbed_keys: tuple[str, ...] | None = None,
    perturbation_applied: bool = True,
) -> dict[str, Any]:
    if perturbed_keys is None:
        perturbed_keys = identity_keys
    return {
        "row": ROW_ID,
        "synthetic": True,
        "arms": {
            "identity": {
                "ok": identity_ok,
                "vacuous": identity_elements == 0,
                "compared_keys": len(identity_keys),
                "compared_elements": identity_elements,
                "keys": sorted(identity_keys),
                "findings": (
                    []
                    if identity_ok
                    else [
                        {
                            "key": sorted(identity_keys)[0] if identity_keys else "?",
                            "status": "differ",
                            "note": "synthetic bit drift",
                        }
                    ]
                ),
            },
            "perturbed": {
                "ok": perturbed_ok,
                "vacuous": False,
                "compared_keys": len(perturbed_keys),
                "compared_elements": perturbed_elements,
                "keys": sorted(perturbed_keys),
                "perturbation_applied": perturbation_applied,
                "perturbed_key": (
                    sorted(perturbed_keys)[0] if perturbation_applied and perturbed_keys else None
                ),
                "element_index": 0 if perturbation_applied else None,
                "before": 1.0 if perturbation_applied else None,
                "after": 1.0000001192092896 if perturbation_applied else None,
                "perturbation_note": (
                    "" if perturbation_applied else "synthetic: no float tensor, empty checkpoint"
                ),
                "findings": (
                    []
                    if perturbed_ok
                    else [
                        {
                            "key": sorted(perturbed_keys)[0] if perturbed_keys else "?",
                            "status": "differ",
                            "note": "synthetic 1-ULP finding",
                        }
                    ]
                ),
            },
        },
    }


def _controls() -> list[tuple[str, str, dict[str, Any], int]]:
    return [
        (
            "C1-green",
            "identity bit-exact and the 1-ULP control fires",
            _synthetic_payload(),
            EXIT_GREEN,
        ),
        (
            "C2-red-identity",
            "identity arm NOT ok: save->load round trip changed bits (RED must fire)",
            _synthetic_payload(identity_ok=False),
            EXIT_RED,
        ),
        (
            "C3-red-blind",
            "perturbed arm ok: the 1-ULP control did not fire, detector blind (RED must fire)",
            _synthetic_payload(perturbed_ok=True),
            EXIT_RED,
        ),
        (
            "C4-refuse-keys",
            "arms compared different key sets (not the same instrument twice) (REFUSE must fire)",
            _synthetic_payload(perturbed_keys=("layer.0.weight", "layer.2.weight")),
            EXIT_REFUSE,
        ),
        (
            "C5-refuse-vacuous",
            "identity arm compared zero elements / is_vacuous (REFUSE must fire)",
            _synthetic_payload(identity_elements=0),
            EXIT_REFUSE,
        ),
        (
            "C6-unmeasured-perturb",
            "perturbation could not be applied at all (UNMEASURED must fire)",
            _synthetic_payload(perturbation_applied=False),
            EXIT_UNMEASURED,
        ),
        (
            "C7-order-keys-over-red",
            "key-set mismatch AND failing identity still REFUSE: first order branch wins",
            _synthetic_payload(
                identity_ok=False,
                perturbed_keys=("layer.0.weight", "layer.2.weight"),
            ),
            EXIT_REFUSE,
        ),
        (
            "C8-order-vacuous-over-unmeasured",
            "vacuous identity AND unapplied perturbation still REFUSE: order is load-bearing",
            _synthetic_payload(identity_elements=0, perturbation_applied=False),
            EXIT_REFUSE,
        ),
    ]


def _write_and_load_report(payload: dict[str, Any]) -> tuple[int, dict[str, dict[str, Any]]]:
    """Write through the public seam, then prove what actually landed on disk."""
    with tempfile.TemporaryDirectory(prefix="t1-4-report-") as temporary:
        out_dir = Path(temporary)
        _write_arm_payloads(out_dir, payload)
        paths = sorted(out_dir.glob("*.json"))
        loaded: dict[str, dict[str, Any]] = {}
        for path in paths:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            loaded[str(envelope.get("arm"))] = envelope
    return len(paths), loaded


def _control_every_arm_writes_a_file() -> bool:
    file_count, reported = _write_and_load_report(_synthetic_payload())
    return (
        file_count == 2
        and set(reported) == {"identity", "perturbed"}
        and all(envelope.get("row") == ROW_ID for envelope in reported.values())
    )


def _control_failing_arm_still_writes_a_file() -> bool:
    file_count, reported = _write_and_load_report(_synthetic_payload(identity_ok=False))
    return file_count == 2 and set(reported) == {"identity", "perturbed"}


def _control_missing_out_dir_refuses() -> bool:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        rc = main(["--checkpoint", "nowhere", "--work-dir", "nowhere"])
    return rc == EXIT_REFUSE and "--out-dir" in stderr.getvalue()


def _control_real_status_reaches_disk() -> bool:
    # MUST FIRE: a hard-coded PASS or UNMEASURED would make this control fail, as
    # would writing a synthetic status without the source payload that minted it.
    file_count, reported = _write_and_load_report(_synthetic_payload(identity_ok=False))
    identity = reported.get("identity")
    telemetry = identity.get("telemetry") if isinstance(identity, dict) else None
    return (
        file_count == 2
        and isinstance(identity, dict)
        and identity.get("status") == "FAIL"
        and isinstance(telemetry, dict)
        and telemetry.get("ok") is False
    )


def run_self_test() -> int:
    controls = _controls()
    verdict_passed = 0
    for control_id, desc, payload, want in controls:
        got, adjudicated = verdict(payload)
        verdict_rc = adjudicated["verdict"]["rc"]
        ok = got == want and verdict_rc == want
        if ok:
            verdict_passed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {control_id} {desc} rc={got} want={want}")

    reporting_controls = [
        (
            "R1-every-arm-writes",
            "both declared arms leave exactly one runner-readable payload",
            _control_every_arm_writes_a_file,
        ),
        (
            "R2-failing-arm-writes",
            "a nonzero-evidence arm is still represented by a payload file",
            _control_failing_arm_still_writes_a_file,
        ),
        (
            "R3-out-dir-required",
            "--out-dir absent without --self-test REFUSES and names the flag",
            _control_missing_out_dir_refuses,
        ),
        (
            "R4-real-status-must-fire",
            "the FAIL source status, not a reporting default, reaches disk",
            _control_real_status_reaches_disk,
        ),
    ]
    reporting_passed = 0
    for control_id, desc, control in reporting_controls:
        try:
            ok = control()
        except Exception:  # noqa: BLE001 - a failed self-test must not escape
            traceback.print_exc()
            ok = False
        if ok:
            reporting_passed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {control_id} {desc}")

    total = len(controls) + len(reporting_controls)
    passed = verdict_passed + reporting_passed
    print(f"  {ROW_ID} self-test: {passed}/{total} controls PASS")
    return EXIT_GREEN if passed == total else EXIT_RED


# ---------------------------------------------------------------------------
# Output hygiene and main
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Recursively convert to strict-JSON-safe values (no NaN/Infinity literals)."""
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    missing = [
        flag
        for flag, value in (
            ("--checkpoint", args.checkpoint),
            ("--work-dir", args.work_dir),
            ("--out-dir", args.out_dir),
        )
        if value is None
    ]
    if missing:
        sys.stderr.write(
            f"REFUSE({EXIT_REFUSE}): required argument(s) with no defensible default "
            f"missing: {', '.join(missing)} (or pass --self-test)\n"
        )
        return EXIT_REFUSE

    checkpoint = Path(args.checkpoint)
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)

    if not checkpoint.is_dir():
        sys.stderr.write(
            f"UNMEASURED({EXIT_UNMEASURED}): checkpoint precondition unmet — "
            f"{checkpoint} is not an existing directory; absence, not failure\n"
        )
        return EXIT_UNMEASURED

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(
            f"UNMEASURED({EXIT_UNMEASURED}): work-dir precondition unmet — cannot "
            f"create {work_dir} for the perturbed copy: {exc}; absence, not failure\n"
        )
        return EXIT_UNMEASURED

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(
            f"UNMEASURED({EXIT_UNMEASURED}): --out-dir precondition unmet — cannot "
            f"create {out_dir} for runner payloads: {exc}; absence, not failure\n"
        )
        return EXIT_UNMEASURED

    try:
        payload = run_measurement(checkpoint, work_dir)
        rc, adjudicated = verdict(payload)
        _write_arm_payloads(out_dir, payload)
    except Exception as exc:  # noqa: BLE001 - classified, never adjudicated (#417)
        # An escape means the measurement did not happen, so there is nothing to
        # refute. classify_boundary_exception sorts environment faults (import,
        # link, ABI, CUDA-init) to 95 and harness faults to 96. This used to
        # return 5, which turned a missing libcudart into a refutation.
        traceback.print_exc()
        error_trace = traceback.format_exc()
        code, reason = classify_boundary_exception(exc)
        name = "UNMEASURED" if code == EXIT_UNMEASURED else "CANNOT_MEASURE"
        emergency = {
            "row": ROW_ID,
            "error": error_trace,
            "verdict": {
                "row": ROW_ID,
                "rc": code,
                "name": name,
                "reason": (
                    f"unexpected {type(exc).__name__} escaped the measurement: {exc}; "
                    f"classified {name} ({code}): {reason}"
                ),
            },
        }
        try:
            _write_boundary_payloads(
                out_dir,
                code,
                str(emergency["verdict"]["reason"]),
                error_trace,
            )
        except Exception as write_exc:  # noqa: BLE001 - never mask the original
            sys.stderr.write(
                f"{ROW_ID}: runner reporting also failed while recording a "
                f"boundary abort: {type(write_exc).__name__}: {write_exc}\n"
            )
        _emit(emergency)
        sys.stderr.write(f"{ROW_ID} verdict: {name} rc={code} (escaping exception)\n")
        return code

    _emit(adjudicated)
    sys.stderr.write(
        f"{ROW_ID} verdict: {adjudicated['verdict']['name']} rc={rc} — "
        f"{adjudicated['verdict']['reason']}\n"
    )
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - never 1, never 2, never 5
        traceback.print_exc()
        code, _reason = classify_boundary_exception(exc)
        sys.exit(code)
