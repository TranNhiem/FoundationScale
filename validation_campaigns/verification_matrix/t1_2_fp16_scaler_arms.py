"""T1-2: fp16 grad-scaler overflow-skip adjudication for the FoundationScale verification matrix.

Row claim
    fp16 grad scaler skips an overflowing step
Run arm
    fp16: the SAME scaler object, handed a finite gradient, MUST step the optimizer
Control
    the SAME scaler object, handed an inf gradient, MUST skip the step, reduce its
    scale, and leave the parameter finite

WHAT THIS INSTRUMENT DOES NOT PROVE
    It does not observe a skip during a real FoundationScale forward/backward pass;
    no FS training step is executed anywhere in this file. FoundationScale itself
    never constructs a GradScaler -- ``grep -rn "GradScaler|grad_scaler"
    src/foundationscale`` returns NOTHING. Per train/loop.py:2976 FS only declares
    precision by setting ``kwargs["fp16"] = True`` on a transformers
    TrainingArguments, and transformers + accelerate own the scaler from there. The
    row's claim is therefore a claim about the COMPOSED system. What this instrument
    proves is exactly that: the GradScaler which FS's fp16 declaration installs in
    the Trainer is (a) enabled, and (b) behaves correctly when handed an overflow --
    it skips the step, reduces its scale, keeps the parameter finite -- while
    stepping normally on a finite gradient. It does not prove FS orders its own
    calls around that scaler correctly, because FS makes no such calls.

ARM D (declaration pin)
    ARM W hand-writes ``fp16=True`` into TrainingArguments, which reproduces
    FoundationScale's LITERAL but not its CODE PATH: if FS ever stopped mapping
    precision="fp16" onto ``kwargs["fp16"] = True``, ARM W would stay GREEN
    while measuring only transformers. The mapping is unreachable as a function
    -- it is inline inside train/loop.py's ``_train(cfg)``, a ~700-line
    function that loads models and datasets -- so arm D pins it STRUCTURALLY.
    It locates the INSTALLED package with importlib.util.find_spec (never a
    hardcoded relative path: the pin must bind the module that would actually
    run, not a file that happens to sit near it on disk), parses the origin
    with ast, and walks the WHOLE ``_train`` subtree -- the mapping is an
    ``elif``, so in the AST it lives in an outer If's ``orelse``, and walking
    only the direct body would report absent on code that is present, a
    manufactured false RED. Verdict reads the pin BEFORE all wiring evidence:
    source not locatable -> 95 (the framework is ABSENT here, not failed);
    located but the mapping gone -> 5 (the claim is false at the declaration
    level, however well transformers' own scaler behaves).

ARM W (wiring)
    Build the SAME declaration FS builds for precision="fp16"
    (transformers.TrainingArguments(..., fp16=True, report_to=[], ...)), attach a
    transformers.Trainer over a 2-parameter toy nn.Module and a 4-row dummy dataset,
    call create_optimizer_and_scheduler / _wrap_model / accelerator.prepare until
    accelerate materialises its scaler, then RESOLVE the live object by searching, in
    order: trainer.accelerator.scaler, trainer.accelerator.scalar (typo-guard: these
    three names are searched, none invented), trainer.scaler.

ARMS R and C
    ONE scaler object, two arms through the IDENTICAL code path
    (scale -> backward -> unscale_ -> step -> update). The only difference anywhere
    is the injected gradient constant: finite (0.5) in R, inf in C. Because both
    arms are materialised by the same function, a divergence in outcome is
    attributable to the overflow and to nothing else. p is snapshotted as a float
    BEFORE each arm and compared with exact ==, because "did the optimizer step
    happen" is a discrete question, not a numeric one -- a tolerance would smear a
    shrunk-update or a partial write into an answer the experiment cannot give.
    lr=1.0 and grad 0.5 make the run arm's step exactly 1.0 -> 0.5, both exactly
    representable in fp32, so == is sound rather than lucky.

VERDICT ORDER (load-bearing; refuse before judging)
    95  FoundationScale's loop source is not locatable in this environment (arm D)
    5   _train no longer maps precision="fp16" onto kwargs["fp16"] = True (arm D)
    96  no scaler object resolved at all (rule 1)
    96  the two arms did not start from the same parameter value (rule 2)
    95  cuda unavailable or CPU run device AND the scaler cannot be enabled (rule 3)
    5   the resolved scaler is disabled under fp16=True (rule 4)
    96  scaler enabled but no arm data -- guard: an empty comparison never mints a pass
    5   the control arm did NOT skip (rule 5; control dead AND behaviour wrong)
    5   the run arm DID skip (rule 6; a skip-everything scaler is useless)
    5   p became NaN in either arm (rule 7)
    0   otherwise (rule 8)
    Rules 5 and 6 are what make this a real control pair: the detector must fire on
    the overflow AND stay silent without it.

EXIT CONTRACT
    GREEN=0  RED=5  UNMEASURED=95  REFUSE=96.  Never 1 or 2.
    Unmet preconditions are 95/96, never 5.  UNMEASURED is not PASS.
    An escaping exception is adjudicated RED(5) with the traceback; the process
    never exits 1.
    torch / transformers / accelerate are imported lazily inside the functions that
    need them so --self-test runs on a bare laptop with no GPU, network or files.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, NoReturn

# The row directory is a sibling import root, exactly as `python3 t1_2_...py`
# gives it. The boundary classifier lives there because four other rows already
# share it; this row used to carry its own -- see _ESCAPE_REASON below.
# run_row owns the per-arm write seam. Routing every arm file through
# write_arm_payload keeps the filename DERIVED in one place -- a caller-typed
# name is exactly how a measurement gets silently dropped -- and its
# keyword-only signature makes the #493 class of key-typo bug ("launcher_exit"
# for "launcher_exit_code", "loss_series" for "loss_curve", silently dropped by
# an `if key in payload` lift) unrepresentable, because Python itself raises
# TypeError on a wrong keyword.
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

ROW_ID = "T1-2"
CLAIM_TEXT = "fp16 grad scaler skips an overflowing step"

# Chosen so the run arm's step (lr=1.0, grad 0.5) moves p by exactly 0.5 in fp32:
# 1.0 -> 0.5 are both exactly representable, which is what licenses exact ==.
P_INIT = 1.0
RUN_GRAD = 0.5

# #417: an escaping exception is CLASSIFIED, not adjudicated. This row used to
# read "an instrument that cannot account for itself is adjudicated RED", which
# means a missing libcudart published a refutation of a claim the arms never
# reached. classify_boundary_exception sorts environment faults (import, link,
# ABI, CUDA-init) to 95 and harness faults to 96; neither is 5, because a crash
# refutes nothing.
_ESCAPE_REASON_PREFIX = "escaping exception classified at the main() boundary"


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
        prog="t1_2_fp16_grad_scaler_skip",
        description=(
            "T1-2: adjudicate 'fp16 grad scaler skips an overflowing step' for the "
            "composed FS -> transformers -> accelerate system. ARM W reproduces "
            "FS's fp16 declaration and resolves accelerate's scaler; ARM R hands "
            "that scaler a finite gradient (MUST step); ARM C hands it an inf "
            "gradient (MUST skip, scale reduced, parameter finite)."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic controls against the pure verdict function; no torch needed",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="also write the full JSON payload (including the verdict) to PATH",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help=(
            "write ONE payload file per arm (D, W, R, C) under DIR via "
            "run_row.write_arm_payload; required unless --self-test is given. "
            "Enforced in main, not argparse, so a missing flag REFUSES 96 "
            "naming the flag instead of argparse's exit 2"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "device for the wiring arm and measurement arms; 'auto' picks cuda "
            "when available, else cpu (default: auto)"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# ARM D: the declaration pin -- FS's fp16 mapping, located structurally, no import
# ---------------------------------------------------------------------------


def _declaration_pin() -> dict[str, Any]:
    """Pin FS's precision="fp16" -> kwargs["fp16"]=True mapping WITHOUT importing it.

    Locates the INSTALLED foundationscale.train.loop via importlib.util.find_spec
    (never a hardcoded relative path: the pin must bind the module that would
    actually run), parses its source with ast, and walks the WHOLE ``_train``
    subtree for an If testing ``cfg.precision == "fp16"`` whose body assigns
    ``kwargs["fp16"] = True``. Never raises: any failure to locate or read the
    source returns ``source_found: False`` with the error string.
    """

    def _absent(error: str, source_path: str | None = None) -> dict[str, Any]:
        return {
            "source_found": False,
            "source_path": source_path,
            "declares_fp16": False,
            "error": error,
        }

    def _is_precision_fp16_test(test: ast.expr) -> bool:
        # The test must be EXACTLY `cfg.precision == "fp16"`: a Compare whose
        # left is Attribute(attr="precision", value=Name(id="cfg")), with one
        # Eq operator and one comparator, the constant string "fp16".
        if not isinstance(test, ast.Compare):
            return False
        left = test.left
        if not (isinstance(left, ast.Attribute) and left.attr == "precision"):
            return False
        if not (isinstance(left.value, ast.Name) and left.value.id == "cfg"):
            return False
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
        if len(test.comparators) != 1:
            return False
        comparator = test.comparators[0]
        return isinstance(comparator, ast.Constant) and comparator.value == "fp16"

    def _is_fp16_true_assign(stmt: ast.stmt) -> bool:
        # The body must contain EXACTLY `kwargs["fp16"] = True`: an Assign with
        # a single Subscript target (Name "kwargs" subscripted by the constant
        # "fp16") whose value is the constant True. `is True`, not == True: the
        # integer 1 compares equal to True and must NOT satisfy the pin.
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            return False
        target = stmt.targets[0]
        if not isinstance(target, ast.Subscript):
            return False
        if not (isinstance(target.value, ast.Name) and target.value.id == "kwargs"):
            return False
        if not (isinstance(target.slice, ast.Constant) and target.slice.value == "fp16"):
            return False
        return isinstance(stmt.value, ast.Constant) and stmt.value.value is True

    try:
        spec = importlib.util.find_spec("foundationscale.train.loop")
    except Exception as exc:  # noqa: BLE001 - a raising find_spec means ABSENT
        return _absent(f"importlib.util.find_spec raised: {exc!r}")
    if spec is None:
        return _absent("find_spec returned None: foundationscale.train.loop not importable")
    if spec.origin is None:
        return _absent("find_spec gave a spec with origin None (e.g. a namespace package)")
    source_path = str(spec.origin)
    try:
        source = Path(source_path).read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - an unreadable origin is ABSENT
        return _absent(f"could not read {source_path}: {exc!r}", source_path)
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        # An unparseable module cannot be imported either, so the framework's
        # behaviour here is ABSENT (95 via source_found False), not RED.
        return _absent(f"{source_path} does not parse: {exc}", source_path)

    declares_fp16 = False
    for node in ast.walk(tree):
        if declares_fp16:
            break
        if isinstance(node, ast.FunctionDef) and node.name == "_train":
            # The mapping is an `elif`: in the AST it lives in the ORELSE of an
            # outer If. ast.walk over the WHOLE _train subtree is therefore
            # load-bearing -- checking only _train's direct body would report
            # absent on code that is present and manufacture a false RED.
            for inner in ast.walk(node):
                if declares_fp16:
                    break
                if not isinstance(inner, ast.If):
                    continue
                if not _is_precision_fp16_test(inner.test):
                    continue
                if any(_is_fp16_true_assign(stmt) for stmt in inner.body):
                    declares_fp16 = True
    return {
        "source_found": True,
        "source_path": source_path,
        "declares_fp16": declares_fp16,
        "error": None,
    }


# ---------------------------------------------------------------------------
# ARM W: rebuild FS's fp16 declaration and RESOLVE the scaler it installs
# ---------------------------------------------------------------------------


def _resolve_scaler(trainer: Any) -> tuple[Any | None, str | None]:
    """Search the documented attribute paths, in order; invent no others."""
    accelerator = getattr(trainer, "accelerator", None)
    candidates: list[tuple[str, Any | None]] = [
        ("trainer.accelerator.scaler", getattr(accelerator, "scaler", None)),
        ("trainer.accelerator.scalar", getattr(accelerator, "scalar", None)),
        ("trainer.scaler", getattr(trainer, "scaler", None)),
    ]
    for attribute, obj in candidates:
        if obj is not None:
            return obj, attribute
    return None, None


def _wire_foundation_scale_fp16(device_used: str) -> tuple[dict[str, Any], Any | None]:
    """Reproduce FoundationScale's fp16 declaration and resolve the scaler it buys.

    FS never constructs a GradScaler; per train/loop.py:2976 it declares precision
    by setting ``kwargs["fp16"] = True`` on a transformers TrainingArguments, after
    which transformers + accelerate own the scaler. So the wiring arm builds the
    same declaration on a throwaway 2-parameter module with a 4-row dataset, drives
    the Trainer through the points at which accelerate materialises an fp16
    GradScaler, and resolves the live object via _resolve_scaler.

    Any exception raised while constructing the composed system is caught INTO the
    payload (wiring.error) rather than propagated: a system that cannot even be
    built is a CANNOT-MEASURE (verdict rule 1 resolves no scaler), not a measured
    false and not an escaping exception (which the exit contract maps to RED).
    """
    import torch
    import transformers

    wiring: dict[str, Any] = {
        "attempted": True,
        "error": None,
        "declaration": {
            "fp16": True,
            "report_to": [],
            "use_cpu": device_used == "cpu",
        },
        "scaler_resolved": False,
        "scaler_attribute": None,
        "scaler_type": None,
        "scaler_enabled": None,
        "scaler_init_scale": None,
    }
    scaler: Any | None = None
    output_dir = Path(tempfile.mkdtemp(prefix="t1-2-wiring-"))
    try:

        class _ToyModule(torch.nn.Module):
            """The smallest module carrying TWO parameters, as the row requires."""

            def __init__(self) -> None:
                super().__init__()
                self.alpha = torch.nn.Parameter(torch.full((1,), 0.7))
                self.beta = torch.nn.Parameter(torch.full((1,), -0.3))

            def forward(self) -> Any:
                # Never executed: no training step is run here; the module exists so
                # the Trainer has parameters to place on a device and to prepare.
                return self.alpha + self.beta

        dataset = [{"x": float(i)} for i in range(4)]
        training_args = transformers.TrainingArguments(
            output_dir=str(output_dir),
            fp16=True,  # the exact declaration train/loop.py:2976 makes for precision="fp16"
            report_to=[],
            use_cpu=device_used == "cpu",
            per_device_train_batch_size=4,
            num_train_epochs=1,
            logging_steps=1,
            disable_tqdm=True,
        )
        trainer = transformers.Trainer(
            model=_ToyModule(),
            args=training_args,
            train_dataset=dataset,
        )
        # Mirror Trainer.train()'s own order: optimizer/scheduler first, then model
        # wrapping, then accelerate prepare -- prepare is where accelerate creates
        # its fp16 GradScaler.
        trainer.create_optimizer_and_scheduler(num_training_steps=len(dataset))
        wrap = getattr(trainer, "_wrap_model", None)
        if wrap is not None:
            try:
                wrap(trainer.model, training=True)
            except TypeError:
                # Signature drift across transformers versions on the training kwarg;
                # the positional call is the lowest common denominator.
                wrap(trainer.model)
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None and getattr(accelerator, "scaler", None) is None:
            accelerator.prepare(trainer.model, trainer.optimizer, trainer.lr_scheduler)
        scaler, attribute = _resolve_scaler(trainer)
        wiring["scaler_resolved"] = scaler is not None
        wiring["scaler_attribute"] = attribute
        if scaler is not None:
            wiring["scaler_type"] = f"{type(scaler).__module__}.{type(scaler).__name__}"
            wiring["scaler_enabled"] = bool(scaler.is_enabled())
            try:
                wiring["scaler_init_scale"] = float(scaler.get_scale())
            except Exception:  # noqa: BLE001 - informational only
                wiring["scaler_init_scale"] = None
    except Exception:  # noqa: BLE001 - converted to wiring evidence, see docstring
        wiring["error"] = traceback.format_exc()
        scaler = None
        wiring["scaler_resolved"] = False
        wiring["scaler_enabled"] = None
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    return wiring, scaler


# ---------------------------------------------------------------------------
# ARMS R and C: ONE code path through ONE scaler, differing only in the gradient
# ---------------------------------------------------------------------------


def _scaler_arm(scaler: Any, *, grad_const: float, device: str) -> dict[str, Any]:
    """One arm of the experiment, run through the shared, live scaler object.

    Both arms execute EXACTLY this sequence; the only difference anywhere in the
    instrument is ``grad_const`` (finite 0.5 vs inf). That is what makes a
    divergence between the arms attributable to the injected overflow and to
    nothing else: one instrument run twice, not two instruments.

    ``p`` is fresh per arm and initialised to P_INIT so the verdict can verify from
    the payload that both arms started from the same value rather than merely
    trusting the claim. The snapshot comparison is exact == by design: with lr=1.0
    and grad 0.5 the stepping arm moves 1.0 -> 0.5 (exact in fp32), so the discrete
    question "did the optimizer step" needs no tolerance, and a tolerance would
    only open the door to mis-attributing partial or shrunken writes.
    """
    import torch

    p = torch.nn.Parameter(torch.full((1,), P_INIT, dtype=torch.float32, device=device))
    optimizer = torch.optim.SGD([p], lr=1.0)
    p_start = float(p.detach().cpu().item())
    scale_before = float(scaler.get_scale())
    # The forward is degenerate on purpose: multiplying p by the injected constant
    # makes autograd write S * grad_const into p.grad once scaler.scale has
    # multiplied the loss by the current scale S. For the control, grad_const is
    # inf and p starts nonzero, so the loss is inf and p.grad comes out exactly inf.
    loss = p * grad_const
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    grad_after_backward = float(p.grad.detach().cpu().item())
    # Correct scaler order: scale, backward, unscale_ (detects non-finite grads and
    # divides by S), step (a no-op when unscale_ flagged non-finite state), update
    # (halves the scale on overflow). Any reordering defeats the detector itself.
    scaler.unscale_(optimizer)
    grad_after_unscale = float(p.grad.detach().cpu().item())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    p_end = float(p.detach().cpu().item())
    return {
        "grad_const_injected": grad_const,
        "p_start": p_start,
        "p_end": p_end,
        "moved": bool(p_end != p_start),
        "became_nan": bool(math.isnan(p_end)),
        "grad_after_backward": grad_after_backward,
        "grad_after_unscale": grad_after_unscale,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "scale_reduced": bool(scale_after < scale_before),
    }


def run_measurement(device_used: str, device_requested: str) -> dict[str, Any]:
    """Run arm D (the declaration pin), then arm W, then arms R/C; return the payload."""
    import torch
    import transformers

    try:
        import importlib.metadata as importlib_metadata

        accelerate_version = importlib_metadata.version("accelerate")
    except Exception:  # noqa: BLE001 - absence of metadata is informational only
        accelerate_version = None

    # Arm D runs BEFORE the wiring arm: it pins the FS declaration whose code path
    # arm W merely presupposes, and the verdict reads this pin before any wiring.
    declaration = _declaration_pin()

    wiring, scaler = _wire_foundation_scale_fp16(device_used)

    payload: dict[str, Any] = {
        "row": ROW_ID,
        "claim": CLAIM_TEXT,
        "mode": "measurement",
        "note": (
            "FoundationScale never constructs a GradScaler (grep over "
            "src/foundationscale returns nothing); train/loop.py:2976 declares fp16 "
            "on a transformers TrainingArguments and transformers + accelerate own "
            "the scaler. This instrument therefore measures the COMPOSED system: "
            "it does not observe a real FS training step."
        ),
        "device_requested": device_requested,
        "device_used": device_used,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "accelerate_version": accelerate_version,
        "wiring": wiring,
        "declaration": declaration,
        "arms": None,
        "arms_skipped_reason": None,
    }

    if scaler is None:
        payload["arms_skipped_reason"] = "no scaler object resolved; arms never ran"
        return payload
    if wiring["scaler_enabled"] is not True:
        payload["arms_skipped_reason"] = (
            "the resolved scaler is disabled; a disabled amp GradScaler raises on "
            "unscale_ rather than measuring, so the arms were not attempted -- "
            "enablement is adjudicated from the wiring alone"
        )
        return payload

    # Run arm FIRST: both arms share ONE scaler object and the control's update()
    # will halve its scale on overflow. Running the finite arm first guarantees it
    # steps at exactly the scale the declaration installed; the control then
    # inherits that same object and must skip. The pair shares everything except
    # the injected gradient constant.
    run_arm = _scaler_arm(scaler, grad_const=RUN_GRAD, device=device_used)
    control_arm = _scaler_arm(scaler, grad_const=float("inf"), device=device_used)
    payload["arms"] = {
        "shared_scaler_object": True,
        "arm_order": ["run", "control"],
        "p_init": P_INIT,
        "lr": 1.0,
        "run": run_arm,
        "control": control_arm,
    }
    return payload


# ---------------------------------------------------------------------------
# The verdict: PURE (data in -> (rc, payload) out). No torch, no files.
# ---------------------------------------------------------------------------


def verdict(data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Adjudicate the row from payload fields only.

    Decision order is load-bearing; see the module docstring. Returns a NEW
    payload dict carrying a ``verdict`` record; the input is not mutated.
    """
    wiring = data.get("wiring") or {}
    declaration = data.get("declaration") or {}
    arms = data.get("arms")
    cuda_available = bool(data.get("cuda_available"))
    device_used = str(data.get("device_used") or "cpu")
    resolved = bool(wiring.get("scaler_resolved"))
    enabled = wiring.get("scaler_enabled")
    attribute = wiring.get("scaler_attribute") or "<unresolved>"

    run = (arms or {}).get("run") or {}
    control = (arms or {}).get("control") or {}

    if declaration.get("source_found") is False:
        # Arm D absence: outranks even rule 1 -- when the framework whose fp16
        # declaration this row adjudicates is not locatable here, every later
        # question (scaler, arms) is unaskable; report the ABSENCE, not 96.
        rc = EXIT_UNMEASURED
        declaration_error = declaration.get("error") or "no importable origin"
        reason = (
            "FoundationScale's loop module (foundationscale.train.loop) is not "
            f"locatable in this environment ({declaration_error}); the framework "
            "whose fp16 declaration this row is about cannot run here, so its "
            "behaviour is ABSENT -- UNMEASURED, not a failure of the row and "
            "not a fault of the wiring arm"
        )
    elif declaration.get("source_found") is True and declaration.get("declares_fp16") is False:
        # Arm D falsification: also outranks rule 1 and both behavioural REDs --
        # the mapping the claim depends on is gone upstream, so the report must
        # name the upstream cause, not downstream wiring or control evidence.
        rc = EXIT_RED
        source_path = declaration.get("source_path") or "foundationscale/train/loop.py"
        reason = (
            "FoundationScale no longer maps precision='fp16' onto the fp16 "
            f"TrainingArguments flag: {source_path} contains no If testing "
            "cfg.precision == 'fp16' whose body assigns kwargs['fp16'] = True "
            "inside _train; nothing downstream can skip an overflowing step, so "
            "the row's claim is false at the declaration level -- regardless of "
            "how well transformers' own scaler behaves on this machine"
        )
    elif not resolved:
        # Rule 1: outranks every later branch, including 95 -- the ORDER pin.
        error = wiring.get("error")
        last_line = error.strip().splitlines()[-1] if error else ""
        hint = f" (wiring raised: {last_line})" if last_line else ""
        rc = EXIT_REFUSE
        reason = (
            "no scaler object could be resolved at any documented attribute path "
            "(trainer.accelerator.scaler, trainer.accelerator.scalar, "
            f"trainer.scaler){hint}; the instrument could not find the thing it "
            "is supposed to test, so this is CANNOT-MEASURE, not a failure of the row"
        )
    elif arms is not None and run.get("p_start") != control.get("p_start"):
        # Rule 2: outranks disabled-scaler RED and both behavioural REDs.
        rc = EXIT_REFUSE
        reason = (
            "the two arms did not start from the same parameter value "
            f"(run p_start={run.get('p_start')!r}, control "
            f"p_start={control.get('p_start')!r}); two different starting states "
            "would make any divergence un-attributable to the overflow -- this is "
            "not one instrument run twice, so the verdict refuses"
        )
    elif enabled is not True and (device_used == "cpu" or not cuda_available):
        # Rule 3: an fp16 CUDA amp scaler physically cannot be enabled on a CPU or
        # CUDA-less machine; absence of the capability is absence, not failure.
        rc = EXIT_UNMEASURED
        reason = (
            f"the scaler cannot be enabled on this device (device_used={device_used!r}, "
            f"cuda_available={cuda_available}); an amp scaler that cannot be enabled "
            "is an ABSENCE of the capability under test on this machine, which is "
            "UNMEASURED -- not PASS and not RED"
        )
    elif enabled is not True:
        # Rule 4: CUDA exists and fp16 was declared, yet the scaler is disabled.
        rc = EXIT_RED
        reason = (
            "declaring fp16=True produced a DISABLED GradScaler "
            f"(is_enabled() False at {attribute}) on a CUDA-capable device; nothing "
            "in the composed system can skip an overflowing step, so the row's "
            "claim is false at the wiring level"
        )
    elif arms is None or not run or not control:
        rc = EXIT_REFUSE
        reason = (
            "the scaler resolved and reported enabled, but the payload carries no "
            "arm results; an empty comparison is the all([]) shape and can never "
            "mint a pass -- UNMEASURED-by-construction is refused, not passed"
        )
    elif control.get("moved") is True:
        # Rule 5: the control did not fire; the row is unproven AND the behaviour
        # is wrong, hence RED rather than UNMEASURED.
        rc = EXIT_RED
        reason = (
            "the control arm did NOT skip: with an inf gradient the optimizer "
            f"still STEPPED (p {control.get('p_start')!r} -> "
            f"{control.get('p_end')!r}); the scaler does not reject overflow, "
            "so the claim 'fp16 grad scaler skips an overflowing step' is falsified"
        )
    elif run.get("moved") is not True:
        # Rule 6: keeps a skip-everything scaler from trivially passing the control.
        rc = EXIT_RED
        reason = (
            "the run arm DID skip: with a finite gradient the optimizer did NOT "
            f"step (p stayed {run.get('p_start')!r}); a scaler that rejects "
            "everything trivially 'passes' the overflow control while being "
            "useless, so the pair fails"
        )
    elif bool(run.get("became_nan")) or bool(control.get("became_nan")):
        # Rule 7
        rc = EXIT_RED
        reason = (
            "the parameter became NaN "
            f"(run={run.get('became_nan')}, control={control.get('became_nan')}); "
            "overflow reached the parameter value, which is precisely the failure "
            "an fp16 scaler exists to prevent"
        )
    else:
        rc = EXIT_GREEN
        reason = (
            f"the scaler FoundationScale's fp16 declaration installs (at {attribute}) "
            "is enabled and correct on the handoff: finite gradient stepped "
            f"(p {run.get('p_start')!r} -> {run.get('p_end')!r}), inf gradient "
            f"skipped with the parameter untouched (p {control.get('p_start')!r} "
            f"-> {control.get('p_end')!r}) and the scale reduced "
            f"({control.get('scale_before')!r} -> {control.get('scale_after')!r}); "
            "no NaN -- the row's claim is measured true for the composed system"
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
# Self-test: synthetic controls driving the pure verdict. No torch, no GPU.
# ---------------------------------------------------------------------------


def _synthetic_payload(
    *,
    scaler_resolved: bool = True,
    scaler_enabled: bool | None = True,
    cuda_available: bool = True,
    device_used: str = "cuda",
    arms_present: bool = True,
    run_p_start: float = P_INIT,
    control_p_start: float = P_INIT,
    run_p_end: float | None = None,
    control_p_end: float | None = None,
    wiring_error: str | None = None,
    declaration_found: bool = True,
    declares_fp16: bool = True,
    declaration_error: str | None = None,
) -> dict[str, Any]:
    if run_p_end is None:
        run_p_end = P_INIT - RUN_GRAD  # a stepping run arm
    if control_p_end is None:
        control_p_end = control_p_start  # a skipping control arm
    arms: dict[str, Any] | None
    if arms_present:
        arms = {
            "shared_scaler_object": True,
            "arm_order": ["run", "control"],
            "p_init": P_INIT,
            "lr": 1.0,
            "run": {
                "grad_const_injected": RUN_GRAD,
                "p_start": run_p_start,
                "p_end": run_p_end,
                "moved": bool(run_p_end != run_p_start),
                "became_nan": bool(math.isnan(run_p_end)),
                "scale_before": 65536.0,
                "scale_after": 65536.0,
                "scale_reduced": False,
            },
            "control": {
                "grad_const_injected": float("inf"),
                "p_start": control_p_start,
                "p_end": control_p_end,
                "moved": bool(control_p_end != control_p_start),
                "became_nan": bool(math.isnan(control_p_end)),
                "scale_before": 65536.0,
                "scale_after": 32768.0,
                "scale_reduced": True,
            },
        }
    else:
        arms = None
    return {
        "row": ROW_ID,
        "claim": CLAIM_TEXT,
        "synthetic": True,
        "device_requested": "auto",
        "device_used": device_used,
        "cuda_available": cuda_available,
        "wiring": {
            "attempted": True,
            "error": wiring_error,
            "declaration": {"fp16": True, "report_to": [], "use_cpu": device_used == "cpu"},
            "scaler_resolved": scaler_resolved,
            "scaler_attribute": "trainer.accelerator.scaler" if scaler_resolved else None,
            "scaler_type": "torch.amp.grad_scaler.GradScaler" if scaler_resolved else None,
            "scaler_enabled": scaler_enabled,
            "scaler_init_scale": 65536.0 if scaler_enabled else None,
        },
        "declaration": {
            "source_found": declaration_found,
            "source_path": "/synthetic/fs/train/loop.py" if declaration_found else None,
            "declares_fp16": bool(declares_fp16 and declaration_found),
            "error": declaration_error,
        },
        "arms": arms,
        "arms_skipped_reason": None if arms_present else "synthetic: arms absent",
    }


def _controls() -> list[tuple[str, str, dict[str, Any], int]]:
    return [
        (
            "C01-green",
            "finite arm steps, overflow arm skips with scale reduced, no NaN",
            _synthetic_payload(),
            EXIT_GREEN,
        ),
        (
            "C02-refuse-no-scaler",
            "no scaler object resolved at any path: CANNOT-MEASURE, not a failure",
            _synthetic_payload(
                scaler_resolved=False,
                scaler_enabled=None,
                arms_present=False,
                wiring_error="RuntimeError: synthetic wiring blowup",
            ),
            EXIT_REFUSE,
        ),
        (
            "C03-refuse-start-mismatch",
            "arms started from different p values: two instruments, refuse",
            _synthetic_payload(control_p_start=2.0),
            EXIT_REFUSE,
        ),
        (
            "C04-unmeasured-cannot-enable",
            "cuda absent / cpu device AND scaler cannot be enabled: absence, not failure",
            _synthetic_payload(
                scaler_enabled=False,
                cuda_available=False,
                device_used="cpu",
                arms_present=False,
            ),
            EXIT_UNMEASURED,
        ),
        (
            "C05-red-disabled-on-cuda",
            "fp16=True produced a DISABLED scaler on a CUDA device: claim false at wiring",
            _synthetic_payload(scaler_enabled=False, arms_present=False),
            EXIT_RED,
        ),
        (
            "C06-red-control-stepped",
            "control arm did NOT skip: p moved on an inf gradient (detector dead + wrong)",
            _synthetic_payload(control_p_end=0.75),
            EXIT_RED,
        ),
        (
            "C07-red-run-skipped",
            "run arm DID skip on a finite gradient: a skip-everything scaler is useless",
            _synthetic_payload(run_p_end=P_INIT),
            EXIT_RED,
        ),
        (
            "C08-red-nan-in-run-arm",
            "p became NaN in the run arm (rule 7): overflow leaked into the parameter",
            _synthetic_payload(run_p_end=float("nan")),
            EXIT_RED,
        ),
        (
            "C09-refuse-empty-arms",
            "scaler enabled but arms absent: an empty comparison can never mint a pass",
            _synthetic_payload(arms_present=False),
            EXIT_REFUSE,
        ),
        (
            "C10-order-no-scaler-over-control-moved",
            "payload is BOTH 'no scaler' AND 'control did not skip': rule 1 wins -> 96, not 5",
            _synthetic_payload(
                scaler_resolved=False,
                scaler_enabled=None,
                control_p_end=0.75,
                wiring_error="RuntimeError: nothing to prepare",
            ),
            EXIT_REFUSE,
        ),
        (
            "C11-order-mismatch-over-disabled",
            "start mismatch AND disabled scaler on cuda: rule 2 wins -> 96, not 5",
            _synthetic_payload(scaler_enabled=False, control_p_start=2.0),
            EXIT_REFUSE,
        ),
        (
            "C12-order-unmeasured-over-control-moved",
            "cannot-enable-on-device AND control moved: rule 3 wins -> 95, not 5",
            _synthetic_payload(
                scaler_enabled=False,
                cuda_available=False,
                device_used="cpu",
                control_p_end=0.75,
            ),
            EXIT_UNMEASURED,
        ),
        (
            "C13-order-no-scaler-over-unmeasured",
            "no scaler AND cannot-enable-on-device: rule 1 wins -> 96, not 95",
            _synthetic_payload(
                scaler_resolved=False,
                scaler_enabled=None,
                cuda_available=False,
                device_used="cpu",
                arms_present=False,
            ),
            EXIT_REFUSE,
        ),
        (
            "C14-unmeasured-declaration-source-missing",
            "MUST-FIRE: source_found False -> 95; the framework itself is ABSENT, not failed",
            _synthetic_payload(
                declaration_found=False,
                declaration_error="importlib.util.find_spec returned None",
            ),
            EXIT_UNMEASURED,
        ),
        (
            "C15-red-declaration-mapping-gone",
            "MUST-FIRE: declares_fp16 False -> 5, and the reason names FoundationScale",
            _synthetic_payload(declares_fp16=False),
            EXIT_RED,
        ),
        (
            "C16-order-declaration-absent-over-no-scaler",
            "source_found False AND no scaler resolved: the more fundamental absence wins -> 95",
            _synthetic_payload(
                declaration_found=False,
                declaration_error="importlib.util.find_spec returned None",
                scaler_resolved=False,
                scaler_enabled=None,
                arms_present=False,
                wiring_error="RuntimeError: nothing to prepare",
            ),
            EXIT_UNMEASURED,
        ),
        (
            "C17-order-declaration-red-over-control-red",
            "declares_fp16 False AND the control did not skip -> 5 for the DECLARATION reason",
            _synthetic_payload(declares_fp16=False, control_p_end=0.75),
            EXIT_RED,
        ),
    ]


def _out_dir_controls() -> list[tuple[str, str, bool]]:
    """Filesystem controls for the per-arm reporting seam. No torch, no GPU.

    These drive _write_arm_payloads against synthetic payloads inside a
    TemporaryDirectory and read the files BACK from disk: a write seam that is
    never re-read proves nothing. The read-back is also what makes C20
    MUST-FIRE -- if the seam stamped one constant status on every arm file, the
    'real status' control could not pass, because it demands two DIFFERENT
    statuses from one payload.
    """
    controls: list[tuple[str, str, bool]] = []

    try:
        with tempfile.TemporaryDirectory() as out_dir:
            paths = _write_arm_payloads(Path(out_dir), _synthetic_payload())
            ok = set(paths) == set(_ARM_IDS) and all(path.is_file() for path in paths.values())
    except Exception:  # noqa: BLE001 - a raising control is a FAILING control
        ok = False
    controls.append(
        (
            "C18-outdir-every-arm-writes",
            "an --out-dir run writes exactly one file per arm (D, W, R, C)",
            ok,
        )
    )

    try:
        with tempfile.TemporaryDirectory() as out_dir:
            paths = _write_arm_payloads(
                Path(out_dir),
                _synthetic_payload(
                    scaler_resolved=False,
                    scaler_enabled=None,
                    arms_present=False,
                    wiring_error="RuntimeError: synthetic wiring blowup",
                ),
            )
            statuses = {
                arm: json.loads(path.read_text(encoding="utf-8"))["status"]
                for arm, path in paths.items()
            }
            ok = (
                set(paths) == set(_ARM_IDS)
                and all(path.is_file() for path in paths.values())
                and statuses["W"] == "REFUSE"
                and statuses["R"] == "UNMEASURED"
                and statuses["C"] == "UNMEASURED"
            )
    except Exception:  # noqa: BLE001 - a raising control is a FAILING control
        ok = False
    controls.append(
        (
            "C19-outdir-failed-arm-still-writes",
            "a FAILED arm (wiring blew up before any scaler resolved) still writes its "
            "file for every arm, with an honest CANNOT-MEASURE/UNMEASURED status "
            "rather than vanishing into an absent file",
            ok,
        )
    )

    try:
        with tempfile.TemporaryDirectory() as out_dir:
            paths = _write_arm_payloads(
                Path(out_dir),
                _synthetic_payload(control_p_end=0.75),  # control STEPPED: arm C is RED
            )
            statuses = {
                arm: json.loads(path.read_text(encoding="utf-8"))["status"]
                for arm, path in paths.items()
            }
            ok = statuses["C"] == "RED" and statuses["R"] == "GREEN"
    except Exception:  # noqa: BLE001 - a raising control is a FAILING control
        ok = False
    controls.append(
        (
            "C20-outdir-real-status-must-fire",
            "MUST-FIRE: the on-disk status is each arm's REAL status -- the stepped "
            "control arm lands RED while the stepping run arm lands GREEN from the "
            "SAME payload; any default stamped uniformly on all four files fails "
            "this control",
            ok,
        )
    )

    try:
        rc = main([])
        ok = rc == EXIT_REFUSE
    except Exception:  # noqa: BLE001 - a raising control is a FAILING control
        ok = False
    controls.append(
        (
            "C21-outdir-missing-refuses-96",
            "--out-dir absent without --self-test refuses 96 (naming the flag on "
            "stderr), never argparse's exit 2 and never a RED adjudication",
            ok,
        )
    )

    return controls


def run_self_test() -> int:
    controls = _controls()
    passed = 0
    for control_id, desc, payload_for_control, want in controls:
        got, adjudicated = verdict(payload_for_control)
        verdict_rc = adjudicated["verdict"]["rc"]
        ok = got == want and verdict_rc == want
        if ok:
            passed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {control_id} {desc} rc={got} want={want}")
    out_dir_controls = _out_dir_controls()
    for control_id, desc, ok in out_dir_controls:
        if ok:
            passed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {control_id} {desc}")
    total = len(controls) + len(out_dir_controls)
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


def _write_out(out_path: str | None, payload: dict[str, Any]) -> None:
    """Mirror the payload to --out. A write failure downgrades nothing: it is
    reported on stderr and the adjudication stands."""
    if out_path is None:
        return
    try:
        text = json.dumps(_json_safe(payload), indent=2, sort_keys=True)
        Path(out_path).write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        sys.stderr.write(f"warning: could not write --out {out_path}: {exc}\n")


# ---------------------------------------------------------------------------
# Per-arm reporting seam: ONE file per arm (D, W, R, C) via run_row
# ---------------------------------------------------------------------------

_ARM_IDS = ("D", "W", "R", "C")


def _derive_arm_entries(
    payload: dict[str, Any],
) -> list[tuple[str, str, str, dict[str, Any], list[str] | None]]:
    """Derive one (arm, status, reason, telemetry, excerpts) entry per arm.

    Statuses are per-ARM, not the row verdict copied four times: a runner that
    read the row rc from every arm file could never see WHICH arm carried the
    evidence, and self-test control C20 pins exactly this -- from one payload
    whose control arm stepped, arm C must land on disk RED while arm R lands
    GREEN. An arm whose evidence is absent is still WRITTEN, with an EMPTY
    telemetry dict rather than None: write_arm_payload omits None optionals and
    omission means "this row does not collect this", while present-and-empty
    means "there was nothing to collect" (the arms were correctly skipped, e.g.
    because no scaler resolved). Only the second is a measurement, and for a
    skipped arm it is the true one.
    """
    declaration = payload.get("declaration") or {}
    wiring = payload.get("wiring") or {}
    arms = payload.get("arms") or {}
    run = arms.get("run") or {}
    control = arms.get("control") or {}
    cuda_available = bool(payload.get("cuda_available"))
    device_used = str(payload.get("device_used") or "cpu")
    arms_skipped_reason = payload.get("arms_skipped_reason")

    # ARM D mirrors the verdict's two declaration branches: source not
    # locatable is ABSENT (95), the mapping gone is RED (5), else the pin held.
    if declaration.get("source_found") is not True:
        d_status = _RC_NAMES[EXIT_UNMEASURED]
        d_reason = (
            "arm D: FoundationScale's loop source was not locatable "
            f"({declaration.get('error') or 'no importable origin'}); the "
            "declaration this row adjudicates is ABSENT in this environment"
        )
    elif declaration.get("declares_fp16") is not True:
        d_status = _RC_NAMES[EXIT_RED]
        d_reason = (
            "arm D: no If testing cfg.precision == 'fp16' whose body assigns "
            "kwargs['fp16'] = True anywhere in the _train subtree; the mapping "
            "the row's claim depends on is gone at the declaration level"
        )
    else:
        d_status = _RC_NAMES[EXIT_GREEN]
        d_reason = (
            "arm D: the precision='fp16' -> kwargs['fp16'] = True mapping is "
            "present in the installed train/loop.py _train subtree"
        )

    # ARM W mirrors verdict rules 1, 3 and 4, in that order.
    wiring_error = wiring.get("error")
    if wiring_error or not wiring.get("scaler_resolved"):
        w_status = _RC_NAMES[EXIT_REFUSE]
        w_reason = (
            "arm W: no scaler object resolved at any documented attribute "
            "path; CANNOT-MEASURE, not a failure of the row"
        )
    elif wiring.get("scaler_enabled") is True:
        w_status = _RC_NAMES[EXIT_GREEN]
        w_reason = (
            "arm W: accelerate's scaler resolved at "
            f"{wiring.get('scaler_attribute') or '<unresolved>'} and reports "
            "enabled under the fp16 declaration"
        )
    elif device_used == "cpu" or not cuda_available:
        w_status = _RC_NAMES[EXIT_UNMEASURED]
        w_reason = (
            "arm W: the resolved scaler cannot be enabled on this device; an "
            "absent capability is UNMEASURED, not a failure"
        )
    else:
        w_status = _RC_NAMES[EXIT_RED]
        w_reason = (
            "arm W: fp16=True produced a DISABLED GradScaler on a CUDA-capable "
            "device; the claim is false at the wiring level"
        )
    w_excerpts = wiring_error.strip().splitlines()[-5:] if isinstance(wiring_error, str) else None

    # ARMS R and C: absent arms are REPORTED, not dropped -- the file exists so
    # the runner can tell "correctly skipped" apart from "never ran".
    if not run:
        r_status = _RC_NAMES[EXIT_UNMEASURED]
        r_reason = f"arm R: never ran -- {arms_skipped_reason or 'no run arm in the payload'}"
    elif run.get("moved") is True and run.get("became_nan") is not True:
        r_status = _RC_NAMES[EXIT_GREEN]
        r_reason = (
            "arm R: the finite-gradient arm stepped "
            f"(p {run.get('p_start')!r} -> {run.get('p_end')!r})"
        )
    else:
        r_status = _RC_NAMES[EXIT_RED]
        r_reason = (
            "arm R: the finite-gradient arm did NOT step "
            f"(p {run.get('p_start')!r} -> {run.get('p_end')!r}, "
            f"became_nan={run.get('became_nan')})"
        )
    if not control:
        c_status = _RC_NAMES[EXIT_UNMEASURED]
        c_reason = f"arm C: never ran -- {arms_skipped_reason or 'no control arm in the payload'}"
    elif (
        control.get("moved") is not True
        and control.get("scale_reduced") is True
        and control.get("became_nan") is not True
    ):
        c_status = _RC_NAMES[EXIT_GREEN]
        c_reason = (
            "arm C: the overflow arm skipped with the scale reduced "
            f"({control.get('scale_before')!r} -> {control.get('scale_after')!r}) "
            "and the parameter untouched"
        )
    else:
        c_status = _RC_NAMES[EXIT_RED]
        c_reason = (
            "arm C: the overflow arm did NOT skip cleanly "
            f"(moved={control.get('moved')}, "
            f"scale_reduced={control.get('scale_reduced')}, "
            f"became_nan={control.get('became_nan')})"
        )

    return [
        ("D", d_status, d_reason, dict(declaration), None),
        ("W", w_status, w_reason, dict(wiring), w_excerpts),
        ("R", r_status, r_reason, dict(run), None),
        ("C", c_status, c_reason, dict(control), None),
    ]


def _write_arm_payloads(out_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    """Write ONE payload file per arm (D, W, R, C), including arms that FAILED.

    Every entry goes through run_row.write_arm_payload: the filename is DERIVED
    there (a caller-typed name is how a measurement gets silently dropped) and
    the keyword-only signature is exactly ARM_SCALAR_KEYS, so the #493 class of
    bug -- adjudicators writing "launcher_exit" for "launcher_exit_code" or
    "loss_series" for "loss_curve", silently dropped by an ``if key in payload``
    lift -- is unrepresentable on this path: a wrong keyword is a TypeError
    from Python itself. These arms are not launched subprocesses, so
    launcher_exit_code stays None and is OMITTED ("this row does not collect
    this"); an arm that never ran still writes its file, because an ABSENT
    file cannot be told apart from an arm that never ran, and
    write_arm_payload's non-empty-status check mirrors the #475 read-side
    guard onto the write side so a missing status is loud here, not at the
    runner. Returns the arm -> path mapping for the self-test controls.
    """
    # The runner hands a path that may not exist yet; the arm files ARE the
    # evidence, so materialising the directory is part of writing them.
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for arm, status, reason, telemetry, excerpts in _derive_arm_entries(payload):
        written[arm] = write_arm_payload(
            out_dir,
            ROW_ID,
            arm,
            status=status,
            reason=reason,
            excerpts=excerpts,
            telemetry=telemetry,
        )
    return written


def _print_verdict_line(name: str, rc: int, reason: str) -> None:
    sys.stdout.write(f"{ROW_ID} verdict: {name} rc={rc} - {reason}\n")


def _conclude_early(
    rc: int,
    reason: str,
    out_path: str | None = None,
    out_dir: str | None = None,
) -> int:
    """Emit a minimal payload + verdict line for pre-measurement UNMEASURED exits.

    The per-arm files are written too when out_dir is known: a runner that
    finds NO file for an arm cannot tell "precondition absent" apart from "the
    row never ran", and the second reading silently rescues a harness bug. All
    four arms therefore report the same status with the shared reason -- the
    #475 guard can only guard a file that exists.
    """
    adjudicated: dict[str, Any] = {
        "row": ROW_ID,
        "claim": CLAIM_TEXT,
        "mode": "precondition-check",
        "verdict": {
            "row": ROW_ID,
            "rc": rc,
            "name": _RC_NAMES[rc],
            "reason": reason,
            "adjudicated_from_payload_fields_only": True,
        },
    }
    if out_dir is not None:
        precondition_out_dir = Path(out_dir)
        precondition_out_dir.mkdir(parents=True, exist_ok=True)
        for arm in _ARM_IDS:
            write_arm_payload(
                precondition_out_dir,
                ROW_ID,
                arm,
                status=_RC_NAMES[rc],
                reason=reason,
            )
    _write_out(out_path, adjudicated)
    _emit(adjudicated)
    _print_verdict_line(_RC_NAMES[rc], rc, reason)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.out_dir is None:
        # The runner consumes one file per arm from --out-dir; without the flag
        # there is nowhere for the per-arm payloads to land, so refuse BEFORE
        # importing torch or measuring anything. Checked here rather than via
        # argparse's required= so the refusal names the flag in OUR words and
        # exits 96 -- never argparse's exit 2, and never a RED adjudication of
        # a row whose arms never ran.
        sys.stderr.write(
            f"REFUSE({EXIT_REFUSE}): --out-dir is required unless --self-test "
            "is given; the runner reads one payload file per arm (D, W, R, C) "
            "from that directory\n"
        )
        return EXIT_REFUSE

    try:
        import torch  # noqa: F401
    except ImportError:
        return _conclude_early(
            EXIT_UNMEASURED,
            "torch is not importable on this machine; the scaler under test lives "
            "inside torch/transformers/accelerate, so a precondition for the "
            "measurement (module 'torch') is ABSENT -- UNMEASURED, not a failure",
            args.out,
            args.out_dir,
        )
    try:
        import transformers  # noqa: F401
    except ImportError:
        return _conclude_early(
            EXIT_UNMEASURED,
            "transformers is not importable on this machine; FoundationScale hands "
            "its fp16 declaration to transformers.TrainingArguments, so the "
            "composed system (module 'transformers') is ABSENT -- UNMEASURED, "
            "not a failure",
            args.out,
            args.out_dir,
        )

    device_requested = str(args.device)
    cuda_available = bool(torch.cuda.is_available())
    if device_requested == "cuda" and (not cuda_available):
        return _conclude_early(
            EXIT_UNMEASURED,
            "--device cuda was requested but torch.cuda.is_available() is False; "
            "the requested device is ABSENT on this machine -- UNMEASURED, not a "
            "failure",
            args.out,
            args.out_dir,
        )
    device_used = "cuda" if (device_requested in ("auto", "cuda") and cuda_available) else "cpu"

    try:
        payload = run_measurement(device_used, device_requested)
        rc, adjudicated = verdict(payload)
    except Exception as exc:  # noqa: BLE001 - classified, never adjudicated (#417)
        traceback.print_exc()
        rc, reason = classify_boundary_exception(exc)
        name = "UNMEASURED" if rc == EXIT_UNMEASURED else "CANNOT_MEASURE"
        adjudicated = {
            "row": ROW_ID,
            "claim": CLAIM_TEXT,
            "error": traceback.format_exc(),
            "verdict": {
                "row": ROW_ID,
                "rc": rc,
                "name": name,
                "reason": (
                    f"{_ESCAPE_REASON_PREFIX}: unexpected {type(exc).__name__} "
                    f"escaped the run body: {exc}; classified {name} ({rc}): {reason}"
                ),
                "adjudicated_from_payload_fields_only": False,
            },
        }

    _write_out(args.out, adjudicated)
    try:
        _write_arm_payloads(Path(args.out_dir), adjudicated)
    except OSError as exc:
        # A write failure downgrades nothing, exactly as with --out: the
        # adjudication above already stands, and a runner that finds a missing
        # arm file treats it as dropped evidence, not as GREEN.
        sys.stderr.write(f"warning: could not write per-arm payloads under {args.out_dir}: {exc}\n")
    _emit(adjudicated)
    _print_verdict_line(adjudicated["verdict"]["name"], rc, adjudicated["verdict"]["reason"])
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - last-resort guard: never 1, never 2, never 5
        traceback.print_exc()
        code, _reason = classify_boundary_exception(exc)
        sys.exit(code)
