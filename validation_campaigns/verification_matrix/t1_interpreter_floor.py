"""Capability floor, boundary classifier, and interleaved-timing reduction for T1.

This module does three things for the four GPU rows, and each exists because the
rows were publishing verdicts the instrumentation could not support.

1. INTERPRETER + CAPABILITY FLOOR (#412) -- an unmet toolchain is REFUSE, not RED.

   This began as an *interpreter* floor. The repository declares
   ``requires-python = ">=3.10"`` and formats to ``py310`` and uses 3.10+ syntax
   (``zip(..., strict=True)``), and nothing enforced that at runtime: measured on
   a 3.9.6 interpreter, the syntax fault landed inside the run body and the
   main()-boundary handler faithfully adjudicated it against the claim --

       T1-9 VERDICT RED: unexpected TypeError escaped the run body:
       zip() takes no keyword arguments

   The claim was innocent; the toolchain was at fault. ``python_floor_reason``
   and ``floor_controls`` (kept with their original names and signatures -- four
   rows import them by name) are that floor, EXTENDED here from interpreter to
   capability: ``import torch`` succeeding (caught BROADLY -- a wheel present
   with dead native libs raises OSError, and ABI skew raises ValueError, not
   ImportError), ``torch.cuda.is_available()``, ``device_count() >= 1`` (one,
   not arms-count: the arms run SEQUENTIALLY through a blocking subprocess on a
   one-device tray, so an arms-count floor would be a permanent false 95), and a
   guarded smoke op -- a tiny alloc, one kernel, and ``.item()`` -- because
   ``is_available()`` is a metadata query on several builds and answers True on
   trays whose first real context creation faults. A metadata query does not
   measure a capability. Rows also declare extra required module names; the
   floor import-probes each. ``runtime_version_record`` collects provenance, and
   its docstring carries the caveat: call it INSIDE the arm subprocess, because
   the floor runs in the parent and a shadowed PYTHONPATH would make a
   parent-side record name versions the child never used.

2. BOUNDARY CLASSIFIER (#417) -- a crash refutes nothing.

   All four rows closed main() with ``except Exception -> RED``. RED means the
   claim is REFUTED; an unhandled exception means the arms did not run. This is
   not hypothetical: an NFS ENOLCK was once adjudicated RED against a
   scientific claim. ``classify_boundary_exception`` maps (import, link, ABI,
   CUDA-init) faults to 95 naming the exception CLASS and the capability, maps
   anything else out of the run body to 96 (harness fault), and cannot return
   RED. Unlike a timing threshold, an exception->state mapping really is
   estate-independent, so sharing it is correct; a shared timing threshold is
   not, which is why there is none below.

3. INTERLEAVED-TIMING HELPERS (#413) -- sign agreement per round, not a shared
   spread threshold. WHY, on measured grounds:

   Five unmodified passes of t1_11 showed throughput across the two arms
   correlating at r = 0.743: the dominant noise is a SHARED per-pass drift
   (filesystem-cache and clock state wandering between passes), not independent
   per-arm jitter. That correlation is the entire justification for the design
   here: ``interleaved_round_plan`` rotates the arm order every round (A B, B A,
   A B for two arms) so the arms stay adjacent within a round while arm order is
   not welded to identity, and a PAIRED per-round delta cancels the drift
   delta ranged 8.5 pp where each raw arm ranged 16%), and
   ``reduce_paired_rounds`` is the verdict over those paired deltas.

   The rejected alternative thresholded the widest WITHIN-arm spread against the
   tightest cross-pair separation under one shared constant. That rule is
   unsound twice over, both times with the wrong sign. A range GROWS with N: the
   headroom between widest spread and tightest separation measured 2.59x at n=2
   and only 1.36x at n=5 -- sampling more than HALVED the margin, so a
   range-based abstention gets stricter the harder you measure. And one shared
   constant conflates INSTRUMENT resolution (a hardware property) with REQUIRED
   resolution, which every row already declares in-tree (t1_11's
   _MOVE_TOLERANCE = 0.05; t1_12's STEP_TIME_IDENTICAL_RTOL = 0.05). A threshold
   calibrated on t1_11's large signal would make t1_12 -- whose whole
   discriminating band is a few percent -- abstain forever. Sign agreement per
   round against the row's OWN tolerance has the correct sign: every additional
   round is another conjunct that must independently clear, so the rule
   STRENGTHENS with N. It also cannot see a steady bias (a clock quota
   suppressing both arms equally); no variance rule can, so per-round raw
   samples, deltas, sd, and SEM go into the returned record -- a reader must be
   able to recompute the verdict from it -- and clock/power state is the rows'
   to record alongside as a named unmeasured confound.

This module stays deliberately stdlib-only and importable by interpreters BELOW
the floor it enforces: a guard that cannot be imported by the interpreter it is
meant to reject is not a guard. Annotations are safe (``from __future__ import
annotations`` leaves them unevaluated strings); absent here, as before, are
``zip(strict=)``, runtime ``X | Y``, ``match``, and ``tomllib``.
"""

from __future__ import annotations

import importlib
import math
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import SimpleNamespace

# The floor the repository already declares. Stated here as the single runtime source;
# the packaging metadata is not readable from a standalone campaign script.
MIN_PYTHON = (3, 10)

# The four-state exit contract, re-exported so the classifier has something to
# RETURN and consumers have one source. Absence of an instrument is never a
# refutation of a claim, and nothing below ever returns GREEN or RED from an
# environment fault. (The rows keep their own module-level constants; these are
# the classifier's return vocabulary.)
EXIT_GREEN = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_CANNOT_MEASURE = 96

# Measured default: three interleaved rounds were enough for the paired deltas
# to expose the cache-cold pass' sign reversal with room to spare.
DEFAULT_ROUNDS = 3


def python_floor_reason(
    version: tuple[int, ...] | None = None,
    executable: str | None = None,
) -> str | None:
    """Return None if the interpreter meets the floor, else a reason naming both versions.

    ``version`` and ``executable`` are injectable ONLY so a self-test can exercise the
    refusing branch on a green interpreter. Without them the below-floor arm would be
    unreachable on any machine that can run the suite, and an assertion that never fires
    is not a control -- it is decoration that reads as coverage.
    """
    have_tuple = tuple(sys.version_info[:3]) if version is None else tuple(version)
    where = sys.executable if executable is None else executable
    if have_tuple[:2] >= MIN_PYTHON:
        return None
    want = ".".join(str(part) for part in MIN_PYTHON)
    have = ".".join(str(part) for part in have_tuple)
    return (
        f"interpreter is Python {have}, below the declared floor of {want} "
        f"(pyproject requires-python). The adjudicator uses {want}+ syntax, so the "
        f"claim cannot be measured here -- this is an unmet precondition, NOT a failed "
        f"claim. Interpreter: {where}"
    )


# ---------------------------------------------------------------- classroom of
# capability floor (#412)


def _default_cuda_smoke(torch_module: object) -> None:
    """One tiny allocation, one kernel launch, one host sync, on the default device.

    ``torch.cuda.is_available()`` is a METADATA query on several builds and answers
    True on trays whose first real context creation faults. Only a kernel that the
    host then blocks on -- ``.item()`` forces the sync -- measures the capability.
    """
    tensor = torch_module.zeros(2, device="cuda")
    (tensor + 1.0).sum().item()


def capability_floor_reasons(
    required_modules: Iterable[str] = (),
    *,
    import_module: Callable[[str], object] | None = None,
    smoke_op: Callable[[object], None] | None = None,
) -> list[str]:
    """Return the named capability-floor misses for a GPU row, [] when the tray is capable.

    Never raises and never returns RED: every entry here is an UNMEASURED (95)
    reason -- the environment the arms need is absent or broken, which says
    nothing about the claim. ``import_module``/``smoke_op`` are injectable so the
    self-test can exercise every refusing branch on a machine where the real tray
    is either healthy or absent; without the seam the failure arms would be
    unreachable decoration. ``torch`` is always probed (all four GPU rows ride on
    it); ``required_modules`` is the row's own extra list (kept model-agnostic --
    module names only, no model or dataset names), deduped against torch.
    """
    importer = importlib.import_module if import_module is None else import_module
    smoke = _default_cuda_smoke if smoke_op is None else smoke_op
    reasons: list[str] = []

    py_reason = python_floor_reason()
    if py_reason is not None:
        reasons.append(f"capability floor (interpreter): {py_reason}")

    imported: dict[str, object] = {}
    for name in dict.fromkeys(("torch", *required_modules)):
        try:
            imported[name] = importer(name)
        except Exception as exc:  # noqa: BLE001 -- ImportError only is NOT enough:
            # a wheel present with dead native libs raises OSError, and ABI skew
            # arrives as ValueError from the importing package. Broad is the point.
            reasons.append(
                f"capability floor: `import {name}` raised {type(exc).__name__}: {exc} "
                "-- the module is absent or its native libraries are dead "
                "(link/ABI faults surface here); an unmet capability, not a failed claim"
            )

    torch_module = imported.get("torch")
    if torch_module is None:
        # Every probe below rides on torch; the import reason already named the miss.
        return reasons

    try:
        available = bool(torch_module.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 -- even the metadata probe can fault
        reasons.append(
            "capability floor: torch.cuda.is_available() itself raised "
            f"{type(exc).__name__}: {exc} -- the device layer never came up"
        )
        return reasons
    if not available:
        reasons.append(
            "capability floor: torch.cuda.is_available() reported False -- torch's own "
            "metadata says no usable device exists on this tray"
        )
        return reasons

    try:
        count = int(torch_module.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        reasons.append(
            "capability floor: torch.cuda.device_count() raised "
            f"{type(exc).__name__}: {exc} -- metadata is up but enumeration faults"
        )
        return reasons
    if count < 1:
        reasons.append(
            f"capability floor: torch.cuda.device_count() == {count}; the floor is "
            ">= 1 device, NOT >= arms -- the arms run SEQUENTIALLY through a blocking "
            "subprocess on a one-device tray, so an arms-count floor would be a "
            "permanent false 95"
        )
        return reasons

    try:
        smoke(torch_module)
    except Exception as exc:  # noqa: BLE001
        reasons.append(
            "capability floor: CUDA smoke op (tiny alloc + one kernel + .item()) raised "
            f"{type(exc).__name__}: {exc} -- is_available() was only metadata; the "
            "first real context creation faults here"
        )
    return reasons


def runtime_version_record(
    module_names: Iterable[str] = (),
    *,
    import_module: Callable[[str], object] | None = None,
) -> dict[str, str]:
    """Return a never-raising name->version provenance record for the artifact.

    CAVEAT that decides WHERE the row must call this: the capability floor runs
    in the PARENT, but the arms run in a CHILD whose PYTHONPATH may shadow the
    parent's build. A record taken in the parent can name versions the run never
    used. Call this from INSIDE the arm subprocess (the trainer can emit it),
    or record both sides and compare -- do not record only the parent.
    """
    importer = importlib.import_module if import_module is None else import_module
    record: dict[str, str] = {"python": sys.version.split()[0]}
    for name in dict.fromkeys(("torch", *module_names)):
        try:
            module = importer(name)
        except Exception as exc:  # noqa: BLE001 -- provenance must never raise
            record[name] = f"UNIMPORTABLE ({type(exc).__name__})"
            continue
        try:
            version = getattr(module, "__version__", None)
        except Exception as exc:  # noqa: BLE001 -- modules with mischievous __getattr__
            record[name] = f"UNREADABLE ({type(exc).__name__})"
            continue
        record[name] = version if isinstance(version, str) else "no __version__ attribute"
        if name == "torch":
            try:
                cuda_version = getattr(getattr(module, "version", None), "cuda", None)
            except Exception:  # noqa: BLE001
                cuda_version = None
            record["torch.cuda"] = cuda_version if isinstance(cuda_version, str) else "UNREPORTED"
    return record


# ---------------------------------------------------------------- classroom of
# boundary classifier (#417)


# Substring tables, matched case-insensitively against "ClassName: message". The
# mapping is estate-independent -- a dead libcudart or an ABI-skewed numpy is the
# same fault on any tray -- which is exactly why it can be shared while a timing
# threshold cannot. Order matters: link then ABI then CUDA-first-context.
_LINK_HINTS = (
    "shared object",  # libcudart.so.<v>: cannot open shared object file
    "error while loading shared libraries",
    "undefined symbol",
    "dll load failed",
    # macOS dyld -- bound to the linker context, NOT the bare phrase "image not
    # found": a data loader reports a missing input PNG with that phrase (#420
    # vs #410's image pathway), and that collision would launder a real defect
    # into an environment abstention.
    "dyld: library not loaded",
    "libcudart",
    "libcudnn",
    "libcublas",
    "libcusparse",
    "libnccl",
    "libtorch",
)
_ABI_HINTS = (
    "dtype size changed",  # numpy.dtype size changed, may indicate binary incompatibility
    "binary incompatibility",
    "compiled against api",
)
_CUDA_HINTS = (
    "no cuda-capable",
    "cuda driver",
    "cuda error",
    "cuda runtime",
    "cuda initialization",
    "nvidia driver",
)


def classify_boundary_exception(exc: BaseException) -> tuple[int, str]:
    """Map an exception that escaped a row's run body to (exit_code, reason).

    Replaces ``except Exception -> RED`` at main()'s boundary. RED means THE
    CLAIM IS REFUTED; a crash means the arms did not run, and an ENOLCK-class
    fault was already adjudicated RED against a scientific claim once. So this
    function is structurally incapable of returning RED (or GREEN): capability
    absent/broken (import, native-library link, binary ABI, CUDA first context)
    -> 95 naming the exception CLASS and the capability; anything else -> 96,
    a fault inside the harness itself. RED remains the run body's own word for
    "the arms ran and the claim failed".
    """
    cls = type(exc).__name__
    if isinstance(exc, ImportError):  # ModuleNotFoundError subclasses this
        module = getattr(exc, "name", None)
        capability = f"module import {module!r}" if module else "module import"
        return EXIT_UNMEASURED, (
            f"capability absent/broken ({capability}): {cls}: {exc} -- the arms' "
            "environment never came up, so nothing was measured; 95, NOT a refutation"
        )
    haystack = f"{cls}: {exc}".lower()
    for hints, capability in (
        (_LINK_HINTS, "native shared-library link"),
        (_ABI_HINTS, "binary ABI compatibility"),
        (_CUDA_HINTS, "CUDA runtime initialisation"),
    ):
        if any(hint in haystack for hint in hints):
            return EXIT_UNMEASURED, (
                f"capability absent/broken ({capability}): {cls}: {exc} -- the arms' "
                "environment never came up, so nothing was measured; 95, NOT a "
                "refutation"
            )
    return EXIT_CANNOT_MEASURE, (
        f"the adjudicator harness itself faulted ({cls}: {exc}) -- the arms did not "
        "complete, and an incomplete run refutes nothing; 96, NOT RED"
    )


# ---------------------------------------------------------------- classroom of
# interleaved timing (#413)


def interleaved_round_plan(
    arm_names: Sequence[str],
    rounds: int = DEFAULT_ROUNDS,
) -> list[tuple[int, str]]:
    """Emit [(round_index, arm_name), ...], rotating the arm order every round.

    Round i starts at arm i % N (two arms: A B, B A, A B), reproducibly from the
    round index alone. Adjacency still matters inside a round: measured across five
    unmodified passes, the two arms'
    throughput correlated at r = 0.743, so most of the noise is a shared per-pass
    drift that a paired per-round delta cancels -- all-A-then-all-B cannot. The
    ONE global warmup the budget allows is the row's to run before consuming this
    plan; it is deliberately not emitted here (the expensive cold state is shared
    filesystem cache, not per-process, so per-arm/per-order warmups buy nothing).
    Two rounds is the floor, not a style choice: with one round there is no sign
    agreement to check, which is precisely the bug interleaving removes.
    """
    names = list(arm_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"interleaving needs at least two DISTINCT arm names, got {names!r}")
    if rounds < 2:
        raise ValueError(
            f"rounds={rounds}: fewer than two rounds cannot observe sign agreement "
            "between the paired deltas -- a one-round GREEN is one sample"
        )
    plan: list[tuple[int, str]] = []
    for round_index in range(rounds):
        offset = round_index % len(names)
        for name in names[offset:] + names[:offset]:
            plan.append((round_index, name))
    return plan


def reduce_paired_rounds(
    baseline_arm: str,
    treatment_arm: str,
    samples: Mapping[str, Sequence[float]],
    tolerance: float,
    claimed_direction: str = "lower",
) -> dict[str, object]:
    """Reduce per-round paired samples to a verdict plus a fully recomputable record.

    ``claimed_direction`` is the row's claim about the metric: "lower" asserts
    treatment < baseline (e.g. t1_11 throughput and peak memory under
    checkpointing), "higher" asserts treatment > baseline. ``tolerance`` is the
    row's OWN already-declared move tolerance as a fraction (relative to
    |baseline|) -- deliberately NOT a module-level constant, because the required
    resolution is per-claim and each row declares it in-tree (t1_11 0.05, t1_12
    0.05/identity).

    The per-round movement is (baseline - treatment)/|baseline| for "lower",
    negated for "higher", so a positive movement means the claim held that round.
    Verdict: GREEN iff EVERY round clears +tolerance; RED iff every round lies
    beyond -tolerance (the movement consistently happens the WRONG way -- every
    round agrees the claimed movement did not); 95 if rounds disagree in
    direction (this is what catches the cache-cold pass: -20.1% against
    +21.7..+30.2%) or if any round straddles ±tolerance. Mismatched or
    too-short sample lists raise ValueError (a harness bug, the row's 96);
    non-finite samples or a zero baseline return 95 (a measurement fact).

    Everything a reader needs to recompute the verdict -- raw per-round samples,
    deltas, movements, clears, straddles, sign agreement, sd, SEM -- is in the
    returned record.
    """
    if claimed_direction not in ("lower", "higher"):
        raise ValueError(
            f"claimed_direction must be 'lower' or 'higher', got {claimed_direction!r}"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")
    try:
        base = [float(v) for v in samples[baseline_arm]]
        treat = [float(v) for v in samples[treatment_arm]]
    except KeyError as exc:
        raise ValueError(f"samples is missing arm {exc.args[0]!r}") from None
    if len(base) != len(treat) or len(base) < 2:
        raise ValueError(
            "per-arm sample lists must be the same length and at least two rounds, "
            f"got {len(base)} and {len(treat)}"
        )

    record: dict[str, object] = {
        "rule": "per-round paired-delta sign agreement against the row's own declared tolerance",
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "claimed_direction": claimed_direction,
        "tolerance": tolerance,
        "rounds": len(base),
        "per_arm_samples": {arm: list(values) for arm, values in samples.items()},
        "round_deltas_baseline_to_treatment": None,
        "round_movements_claim_direction": None,
        "round_clears_tolerance": None,
        "round_straddles_tolerance": None,
        "movement_signs_agree": None,
        "sd": None,
        "sem": None,
    }

    for i, (bv, tv) in enumerate(zip(base, treat, strict=True)):
        if bv == 0.0 or not (math.isfinite(bv) and math.isfinite(tv)):
            record.update(
                exit_code=EXIT_UNMEASURED,
                state="UNMEASURED",
                reason=(
                    f"round {i} carries baseline={bv!r}, treatment={tv!r}: the paired "
                    "delta divides by the baseline and is undefined here, so no round "
                    "set built on this pass can adjudicate the claim (95, never RED)"
                ),
            )
            return record

    deltas = [(tv - bv) / abs(bv) for bv, tv in zip(base, treat, strict=True)]
    movements = [(-delta if claimed_direction == "lower" else delta) for delta in deltas]
    clears = [m > tolerance for m in movements]
    straddles = [abs(m) <= tolerance for m in movements]
    signs = {(1 if m > 0 else -1 if m < 0 else 0) for m in movements}
    sd = statistics.stdev(movements)
    sem = sd / math.sqrt(len(movements))
    record.update(
        round_deltas_baseline_to_treatment=deltas,
        round_movements_claim_direction=movements,
        round_clears_tolerance=clears,
        round_straddles_tolerance=straddles,
        movement_signs_agree=len(signs) <= 1,
        sd=sd,
        sem=sem,
    )

    shown = ", ".join(f"{m:+.2%}" for m in movements)
    if all(clears):
        record.update(
            exit_code=EXIT_GREEN,
            state="GREEN",
            reason=(
                f"all {len(movements)} interleaved rounds move in the claimed direction "
                f"beyond the row's declared tolerance (weakest round {min(movements):+.2%} "
                f"vs +{tolerance:.2%}; per round {shown})"
            ),
        )
    elif any(straddles):
        rounds_straddling = [i for i, s in enumerate(straddles) if s]
        record.update(
            exit_code=EXIT_UNMEASURED,
            state="UNMEASURED",
            reason=(
                f"round(s) {rounds_straddling} straddle ±{tolerance:.2%} (per round "
                f"{shown}): the instrument/row-tolerance band cannot resolve those "
                "rounds, so the row abstains rather than guess (SEMs and raw samples "
                "are in the record)"
            ),
        )
    elif all(m < -tolerance for m in movements):
        record.update(
            exit_code=EXIT_RED,
            state="RED",
            reason=(
                f"every interleaved round moves BEYOND the tolerance in the opposite "
                f"direction (strongest round {max(movements):+.2%} vs -{tolerance:.2%}; "
                f"per round {shown}): every round agrees the claimed movement did not "
                "happen -- the only configuration in which timing may refute"
            ),
        )
    else:
        record.update(
            exit_code=EXIT_UNMEASURED,
            state="UNMEASURED",
            reason=(
                "interleaved rounds disagree in direction (per round "
                f"{shown}) -- at least one round on each side of ±{tolerance:.2%}; "
                "this is the signature of a shared per-pass drift (cache/clock state), "
                "not of the claim either way"
            ),
        )
    return record


# ---------------------------------------------------------------- classroom of
# controls (extended in place; F1/F2 verbatim from the interpreter-floor era)


def floor_controls(row: str, record: Callable[[str, bool, str], None]) -> None:
    """Run the floor controls for ``row``, reporting through the caller's ``record``.

    ``record(label, passed, detail)`` matches the callback every T1 self-test already
    builds, so the controls land in the module's own count rather than a second tally.
    """
    below = python_floor_reason(version=(3, 9, 6), executable="/usr/bin/python3")
    record(
        f"{row} F1 a below-floor interpreter yields a reason naming both versions",
        below is not None and "3.9.6" in below and "3.10" in below,
        repr(below),
    )
    at_floor = python_floor_reason(version=(3, 10, 0), executable="/usr/bin/python3")
    record(
        f"{row} F2 an at-floor interpreter yields no reason (the guard is not blanket)",
        at_floor is None,
        repr(at_floor),
    )

    # -- capability floor (#412): every refusing branch exercised via injection,
    #    because an unreachable failure arm is decoration.
    def _dead_link_importer(_name: str) -> object:
        raise OSError("libcudart.so.12: cannot open shared object file: No such file or directory")

    dead = capability_floor_reasons((), import_module=_dead_link_importer)
    record(
        f"{row} F3 a dead CUDA shared-library link yields a named 95 reason "
        "(module + exception class)",
        len(dead) == 1 and "torch" in dead[0] and "OSError" in dead[0],
        repr(dead),
    )

    torch_like = SimpleNamespace(
        __version__="0.0.selftest",
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    module_table = {"torch": torch_like, "packaging": SimpleNamespace(__version__="0.0.selftest")}

    def _fake_importer(name: str) -> object:
        try:
            return module_table[name]
        except KeyError:
            raise ImportError(f"No module named {name!r}") from None

    smoked: list[bool] = []
    capable = capability_floor_reasons(
        ("packaging",), import_module=_fake_importer, smoke_op=lambda _t: smoked.append(True)
    )
    record(
        f"{row} F4 a fully capable injected tray yields no reasons and the smoke op actually ran",
        capable == [] and smoked == [True],
        repr((capable, smoked)),
    )

    module_table["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    smoked.clear()
    blind = capability_floor_reasons(
        (), import_module=_fake_importer, smoke_op=lambda _t: smoked.append(True)
    )
    record(
        f"{row} F5 an is_available()-False tray yields a named reason and "
        "short-circuits before the smoke op",
        len(blind) == 1 and "is_available" in blind[0] and smoked == [],
        repr(blind),
    )

    module_table["torch"] = torch_like

    def _failing_smoke(_t: object) -> None:
        raise RuntimeError("CUDA error: initialization error")

    smoke_dead = capability_floor_reasons((), import_module=_fake_importer, smoke_op=_failing_smoke)
    record(
        f"{row} F6 a smoke-op fault on a metadata-green tray yields a named 95 reason",
        len(smoke_dead) == 1 and "smoke" in smoke_dead[0] and "RuntimeError" in smoke_dead[0],
        repr(smoke_dead),
    )

    # -- boundary classifier (#417): capability faults -> 95, harness faults -> 96,
    #    and structurally never RED.
    cases = (
        (ImportError("No module named 'torch'"), EXIT_UNMEASURED),
        (OSError("libcudart.so.12: cannot open shared object file"), EXIT_UNMEASURED),
        (
            ValueError(
                "numpy.dtype size changed, may indicate binary incompatibility. "
                "Expected 96 from C header, got 88 from PyObject"
            ),
            EXIT_UNMEASURED,
        ),
        (RuntimeError("CUDA error: no CUDA-capable device is detected"), EXIT_UNMEASURED),
        (KeyError("harness bookkeeping"), EXIT_CANNOT_MEASURE),
        (ZeroDivisionError("harness arithmetic"), EXIT_CANNOT_MEASURE),
    )
    verdicts = [classify_boundary_exception(exc)[0] for exc, _ in cases]
    record(
        f"{row} F7 the boundary classifier maps capability/ABI/CUDA faults to 95 "
        "and harness faults to 96 -- never RED",
        [got for got, (_, want) in zip(verdicts, cases, strict=True) if got != want] == []
        and EXIT_RED not in verdicts
        and EXIT_GREEN not in verdicts,
        repr(verdicts),
    )

    # -- interleaved timing (#413): plan shape, plan floor, and the reducer on the
    #    measured vectors, including the cache-cold fabrication killer.
    plan = interleaved_round_plan(("off", "on"), 3)
    plan_ok = plan == [
        (0, "off"),
        (0, "on"),
        (1, "on"),
        (1, "off"),
        (2, "off"),
        (2, "on"),
    ]
    try:
        interleaved_round_plan(("off", "on"), 1)
    except ValueError:
        refused_single_round = True
    else:
        refused_single_round = False
    record(
        f"{row} F8 the planner rotates the paired order A B, B A, A B and refuses one round",
        plan_ok and refused_single_round,
        repr((plan, refused_single_round)),
    )

    two_arm_firsts = [
        next(name for rnd, name in plan if rnd == round_index) for round_index in range(3)
    ]
    three_arm = interleaved_round_plan(("a", "b", "c"), 3)
    three_arm_firsts = [
        next(name for rnd, name in three_arm if rnd == round_index) for round_index in range(3)
    ]
    three_arm_round1 = [name for rnd, name in three_arm if rnd == 1]
    record(
        f"{row} F11 rotation puts each arm first across three rounds and round 1 "
        "starts at arm i % N for three arms",
        two_arm_firsts == ["off", "on", "off"]
        and three_arm_firsts == ["a", "b", "c"]
        and three_arm_round1 == ["b", "c", "a"],
        repr((plan, three_arm)),
    )

    # Measured t1_11 passes: the warm triple is a unanimous per-round movement
    # toward slower-with-checkpointing; the cold pass paired with a warm pass is
    # exactly the sign disagreement a single run published as RED.
    warm = reduce_paired_rounds(
        "off",
        "on",
        {"off": [2.154, 2.236, 2.099], "on": [1.504, 1.649, 1.544]},
        0.05,
        "lower",
    )
    cold_then_warm = reduce_paired_rounds(
        "off",
        "on",
        {"off": [1.288, 2.217], "on": [1.547, 1.478]},
        0.05,
        "lower",
    )
    record(
        f"{row} F9 the measured warm triple reduces to GREEN; the cache-cold "
        "round against a warm round reduces to 95, not RED",
        warm["exit_code"] == EXIT_GREEN and cold_then_warm["exit_code"] == EXIT_UNMEASURED,
        repr((warm["state"], cold_then_warm["state"])),
    )

    refuted = reduce_paired_rounds(
        "off",
        "on",
        {"off": [2.0, 2.0, 2.0], "on": [2.6, 2.7, 2.5]},
        0.05,
        "lower",
    )
    straddle = reduce_paired_rounds(
        "off",
        "on",
        {"off": [2.0, 2.0], "on": [1.5, 1.94]},
        0.05,
        "lower",
    )
    record(
        f"{row} F10 unanimous opposite-direction rounds reduce to RED; a "
        "tolerance-straddling round forces 95",
        refuted["exit_code"] == EXIT_RED
        and straddle["exit_code"] == EXIT_UNMEASURED
        and refuted["movement_signs_agree"] is True,
        repr((refuted["state"], straddle["state"])),
    )

    # -- #420: the macOS dyld linker hint must still fire on the real linker
    #    message (MUST_FIRE half) and must NOT launder a data-loader "image not
    #    found" -- #410's incoming pixel-carrying runs -- into an environment
    #    abstention (MUST_NOT_FIRE half). Either half alone is satisfiable by a
    #    broken matcher, so both are asserted separately in the same control.
    dyld_code, dyld_reason = classify_boundary_exception(
        OSError(
            "dyld: Library not loaded: /usr/local/lib/libhost.dylib "
            "Referenced from: /opt/venv/torch/lib/libtorch.dylib "
            "Reason: image not found"
        )
    )
    record(
        f"{row} F12 MUST_FIRE: the real dyld message is still a linker environment fault (95)",
        dyld_code == EXIT_UNMEASURED and "native shared-library link" in dyld_reason,
        repr((dyld_code, dyld_reason)),
    )
    loader_code, loader_reason = classify_boundary_exception(
        FileNotFoundError("image not found: /data/shards/000123.png")
    )
    record(
        f"{row} F13 MUST_NOT_FIRE: a data-loader missing-input image is NOT a linker fault; "
        "it reaches the adjudicator as a real fault (96), never laundered to 95",
        loader_code == EXIT_CANNOT_MEASURE and "native shared-library link" not in loader_reason,
        repr((loader_code, loader_reason)),
    )


# ---------------------------------------------------------------------------
# #432: the GPU count a row DECLARES versus the count it can actually REACH
# ---------------------------------------------------------------------------
#
# Measured on this estate: a tray allocated ``gres/gpu=4`` -- Slurm's own AllocTRES
# and TresPerNode both say 4, and the driver enumerates 4 under
# /proc/driver/nvidia/gpus -- exposes exactly 2 to the process. ``nvidia-smi -L``
# lists 2, ``torch.cuda.device_count()`` returns 2, and forcing all four indices
# into CUDA_VISIBLE_DEVICES changes neither number. Two hypotheses were tested and
# both are REFUTED: the cpuset spans both sockets, so it is not socket affinity
# narrowing a socket-affine GRES; and the override above rules out the scheduler
# merely composing a two-entry visibility list. The cause is unattributed.
#
# The cause does not have to be known for the harness to stop lying. Every T1 row
# passes ``--gpus-per-node`` straight through to the trainer, so a row on that tray
# composes a 4-GPU launch, measures whatever two GPUs do, and publishes the number
# 4. That is #124's class -- the measured count and the actual launch decoupled --
# and it is a 2x denominator error on every multi-GPU claim. Recording BOTH numbers
# and refusing when they disagree is estate-independent: it needs no theory of why
# this tray is short, and it fires identically on any machine where the allocation
# and the reachable set diverge.


def reachable_gpu_count() -> int | None:
    """Return the number of GPUs this PROCESS can reach, or None if unknowable.

    None is not zero. A machine with no torch, or a torch built without CUDA, has
    not told us that there are no GPUs -- it has told us nothing, and collapsing
    that into 0 would manufacture a disagreement out of an absent instrument.

    torch is imported inside the function deliberately. This module is stdlib-only
    so that an interpreter BELOW the floor can still import it and be rejected by
    ``python_floor_reason``; a module-scope torch import would make the floor guard
    unimportable on exactly the machines it exists to refuse, which is #354's class.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.device_count())
    except Exception:
        return None


def gpu_reachability_reason(
    declared: int,
    reachable: int | None = None,
    probe: Callable[[], int | None] | None = None,
) -> str | None:
    """Return None if ``declared`` GPUs are reachable, else a reason naming both.

    ``reachable`` and ``probe`` are injectable ONLY so the refusing branches can be
    exercised on a healthy machine -- the same reason ``python_floor_reason`` takes
    a version. A branch that cannot fire on the machine running the suite is not a
    control.

    The rule is ``reachable < declared``, not ``reachable != declared``. A node that
    exposes MORE GPUs than the row asked for is not a problem: the launch uses the
    declared number and the claim is about that number. Only a shortfall silently
    shrinks the thing being measured while the row keeps publishing the declared
    count.

    A single-process row (``declared <= 1``) makes no multi-GPU claim, so an
    unknowable reachable count is not an unmet precondition for it. Above 1 it is:
    "I am about to publish a 4-GPU measurement and I cannot confirm four reachable
    GPUs" is UNMEASURED, and reporting that as a pass is how #432 stayed invisible.

    MEASURED EVIDENCE (#432). On a tray allocated ``gres/gpu=4``, ``nvidia-smi -L``
    listed 2 devices and ``torch.cuda.device_count()`` returned 2. Overriding
    ``CUDA_VISIBLE_DEVICES`` changed nothing, so the shortfall is not a composition
    artifact of that variable; a socket-affinity explanation was also tested and
    REFUTED. The cause is unattributed, and the cure deliberately does not depend on
    knowing it -- it records both quantities and refuses when they disagree.

    The scheduler is NOT an oracle for this. ``sacct`` reports AllocTRES, which is
    what the allocation GRANTED; it cannot see what the process can OPEN, and on the
    tray above it said 4 while the process saw 2. Any guard that asks the scheduler
    is measuring the wrong side of the gap, which is why the probe here runs inside
    the process that will do the measuring.
    """
    if reachable is None:
        reachable = probe() if probe is not None else reachable_gpu_count()
    if declared <= 1:
        return None
    if reachable is None:
        return (
            f"row declares {declared} GPUs and the reachable count could not be measured "
            "(no torch, or a torch that cannot see CUDA). A multi-GPU claim "
            "published without confirming the GPUs are reachable is UNMEASURED, "
            "not a pass. This is an unmet precondition, NOT a failed claim."
        )
    if reachable < declared:
        return (
            f"row declares {declared} GPUs and only {reachable} are reachable from this process. "
            "The allocation and the devices a process can open are different "
            f"quantities, so the claim would be measured on {reachable} and published as "
            f"{declared}. This is an unmet precondition, NOT a failed claim."
        )
    return None


def gpu_reachability_controls(row: str, record: Callable[[str, bool, str], None]) -> None:
    """Run the five #432 controls for ``row``, reporting through the caller's ``record``.

    Two-halved by construction: a guard that refused everything would pass G1 and
    G4 while making every row unrunnable, and a guard that refused nothing would
    pass G2, G3 and G5 while leaving #432 exactly as invisible as it was.

    G1 asserts distinctive phrases, not the bare digits "4" and "2". The finding id
    "#432" contains both of those digits, so a digit-substring assertion would be
    satisfied by a message that named neither count -- the self-hit class, where a
    detector matches its own text and reads as coverage.
    """
    short = gpu_reachability_reason(4, reachable=2)
    record(
        f"{row} G1 a 4-declared / 2-reachable tray refuses, naming both counts",
        short is not None and "declares 4 GPUs" in short and "only 2 are reachable" in short,
        repr(short),
    )
    exact = gpu_reachability_reason(4, reachable=4)
    record(
        f"{row} G2 a fully-reachable allocation does not refuse (not blanket)",
        exact is None,
        repr(exact),
    )
    single = gpu_reachability_reason(1, probe=lambda: None)
    record(
        f"{row} G3 a single-process row is not refused for an unknowable count",
        single is None,
        repr(single),
    )
    unknown = gpu_reachability_reason(4, probe=lambda: None)
    record(
        f"{row} G4 a multi-GPU row with an unmeasurable count refuses, not passes",
        unknown is not None and "UNMEASURED" in unknown,
        repr(unknown),
    )
    over = gpu_reachability_reason(2, reachable=4)
    record(
        f"{row} G5 a node exposing MORE GPUs than declared does not refuse",
        over is None,
        repr(over),
    )
