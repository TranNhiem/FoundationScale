#!/usr/bin/env python3
"""#439 control: every T1 GPU row must declare a CONSTRUCTIBLE topology.

MEASURED on a 2-GPU GB200 tray (pass p414): all four rows emit ``--nodes`` and
``--gpus-per-node`` and none of them emits ``--dp``. ``--dp`` defaults to 1 and
``Topology.__post_init__`` requires ``dp*tp*pp*ep*cp == nodes*gpus_per_node``, so
at any width above 1 the trainer refused before loading a model::

    [fs:train:refuse] topology is not constructible (nothing touched):
      dp(1) x tp(1) x pp(1) x ep(1) x cp(1) = 1
      but nodes(1) x gpus_per_node(2) = 2

torchrun then flattened that declared 96 to launcher rc 1 (#171), so the receipt
said only "torchrun launcher exited 1" -- a crash and a refusal wearing one face.

WHY NO EXISTING CONTROL CAUGHT IT: the defect is invisible at width 1, because
1x1x1x1x1 == 1x1. Every local run, every CI leg and every row --self-test runs at
width 1, which is the one width at which the bug cannot fire. So this control
drives the rows at **width 2** -- that is the whole point of it, and a future
edit that lowers WIDTH to 1 disables it silently. Control C1 of ``--self-test``
exists to make that edit loud instead.

It does not re-implement the rule. It calls each row's REAL argv composer and
hands the emitted degrees to the trainer's REAL Topology, so the control and the
trainer cannot drift apart: if Topology's invariant changes (e.g. when a
Megatron-Core backend lands and EP stops being an independent multiplicative
factor -- Megatron derives ``dp = world // (tp*pp*cp)`` and puts EP in a separate
generator), this control changes with it for free.

Exit: 0 all rows constructible, 5 a row declares an impossible topology,
96 a precondition is unmet (the trainer's Topology is unreadable, a composer
could not be driven) -- never a scientific verdict on an environment failure
(#417).
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any

CLEAR = 0
RED = 5
REFUSE = 96

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
VM_REL = Path("validation_campaigns") / "verification_matrix"
VM = REPO_ROOT / VM_REL
SRC = REPO_ROOT / "src"

# The rows are scripts, not a package: they import each other by bare module name
# (``from t1_interpreter_floor import ...``), so their own directory has to be on
# the path before any of them can be imported.
if str(VM) not in sys.path:
    sys.path.insert(0, str(VM))

# src/ goes FIRST, deliberately. This gate compares four rows of THIS TREE against
# the Topology of THIS TREE; resolving foundationscale from site-packages instead
# would make the verdict depend on what happens to be installed (#83/#229), and
# would also make the gate unrunnable under `python3 -S`, which is how every other
# gate in this directory is invoked by launchers/test_checks_gates.sh.
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Width 2, deliberately. See the module docstring: width 1 cannot see this class
# of defect at all, so a control that runs at width 1 is a control that passes
# by construction.
NODES = 1
WIDTH = 2

DEGREES = ("dp", "tp", "pp", "ep", "cp")

ROWS = (
    "t1_9_optimizer_arms",
    "t1_10_accumulation_equivalence",
    "t1_11_grad_checkpointing",
    "t1_12_attention_impl",
)


class _Refusal(Exception):
    """A precondition is unmet, so there is no verdict to give (96)."""


def _load_topology() -> type:
    """Return the trainer's REAL Topology class, or refuse.

    An absent or unimportable package is a precondition, not a scientific result
    about the four rows -- #417 is exactly the habit of converting one into the
    other.
    """
    try:
        from foundationscale.topology import Topology
    except Exception as exc:
        raise _Refusal(f"cannot import the trainer's Topology: {exc}") from exc
    return Topology


def _synthesize(cls: type) -> Any:
    """Build a minimal instance of a row's ArmSpec from its dataclass fields.

    The rows' spec types differ and are private. Filling them by field TYPE
    rather than by name keeps this control from having to track four separate
    constructors -- and a spec whose values are placeholders is fine here,
    because the only fields this control reads back are the topology ones, which
    come from ``args``, not from the spec.
    """
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING or (
            field.default_factory is not dataclasses.MISSING
        ):
            continue
        ann = str(field.type)
        if "int" in ann:
            kwargs[field.name] = 1
        elif "Path" in ann:
            kwargs[field.name] = Path(tempfile.gettempdir())
        elif "float" in ann:
            kwargs[field.name] = 1.0
        elif "bool" in ann:
            kwargs[field.name] = False
        elif "dict" in ann:
            kwargs[field.name] = {}
        else:
            kwargs[field.name] = "probe"
    return cls(**kwargs)


def _as_mapping(emitted: Any) -> dict[str, str]:
    """Normalise a composer's output -- some rows emit a dict, some a flat argv."""
    if isinstance(emitted, dict):
        return {str(k): str(v) for k, v in emitted.items()}
    out: dict[str, str] = {}
    tokens = list(emitted)
    for i, tok in enumerate(tokens):
        if isinstance(tok, str) and tok.startswith("--"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            out[tok] = "" if str(nxt).startswith("--") else str(nxt)
    return out


def _drive(mod_name: str, tmp: Path) -> dict[str, str]:
    """Return the flags one arm of this row would actually send the trainer."""
    mod = importlib.import_module(mod_name)
    argv = [
        "--out-dir",
        str(tmp),
        "--profile-name",
        "local-single-node",
        "--nodes",
        str(NODES),
        "--gpus-per-node",
        str(WIDTH),
    ]
    args = mod._build_parser().parse_args(argv)

    if mod_name.startswith("t1_9"):
        return _as_mapping(mod._arm_flags(args, tmp, "adamw"))
    if mod_name.startswith("t1_10"):
        spec = _synthesize(mod.ArmSpec)
        return _as_mapping(mod._trainer_argv(spec, args, tmp))
    if mod_name.startswith("t1_11"):
        return _as_mapping(mod._trainer_argv("true", args, tmp, 29500, 1))
    if mod_name.startswith("t1_12"):
        return _as_mapping(mod._arm_specs(args)[0].flags)
    raise KeyError(mod_name)


def _judge(topology_cls: type, row: str, flags: dict[str, str]) -> tuple[bool, str]:
    """Hand one arm's declared degrees to the trainer's real Topology.

    Returns (constructible, one-line report). The self-test drives this same
    function with synthetic flag mappings, so a control that says "the judge
    fires" is a statement about the code main() runs, not about a copy of it.
    """
    degrees = {d: int(flags.get(f"--{d}", 1)) for d in DEGREES}
    nodes = int(flags.get("--nodes", NODES))
    gpus = int(flags.get("--gpus-per-node", WIDTH))
    declared = ",".join(f"{d}={degrees[d]}" for d in DEGREES)
    where = f"{row}: {declared} nodes={nodes} gpus={gpus}"
    try:
        topology_cls(nodes=nodes, gpus_per_node=gpus, **degrees)
    except ValueError as exc:
        return False, f"{where} -> {str(exc).splitlines()[0]}"
    return True, f"{where} -> constructible"


def measure() -> int:
    try:
        topology_cls = _load_topology()
    except _Refusal as exc:
        print(f"REFUSE 96: {exc}")
        return REFUSE

    bad = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for row in ROWS:
            try:
                flags = _drive(row, tmp)
            except Exception as exc:
                print(f"REFUSE 96: {row}: could not drive its composer: {exc!r}")
                return REFUSE

            ok, line = _judge(topology_cls, row, flags)
            if not ok:
                bad += 1
                print(f"[BAD] {line}")
                continue
            print(f"[ok ] {line}")

    total = len(ROWS)
    print(
        f"\n#439 topology declaration at width {WIDTH}: "
        f"{total - bad}/{total} rows declare a constructible topology"
    )
    if bad:
        print(
            "A row that cannot construct its own topology refuses before loading a "
            "model, and torchrun reports that refusal as launcher rc 1 (#171)."
        )
    return RED if bad else CLEAR


def self_test() -> int:
    """Prove this control's own machinery, including that its judge can FIRE.

    A control whose judge cannot return False is a control that passes by
    construction, which is the defect it was written to catch one level up.
    C4 is therefore a MUST_FIRE leg living inside the self-test.
    """
    try:
        topology_cls = _load_topology()
    except _Refusal as exc:
        # 96, not a failing control: an unreadable Topology means the oracle is
        # absent, so nothing was measured (#417).
        print(f"REFUSE 96: {exc}")
        return REFUSE

    controls: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        controls.append((name, passed, detail))

    record(
        "C1 the control drives the rows at a width where the defect can fire",
        WIDTH > 1,
        f"WIDTH={WIDTH}; at width 1 the invariant is 1x1x1x1x1 == 1x1 and always holds",
    )

    # C2/C3: the oracle is the trainer's, and it really discriminates. If these
    # two ever agree, Topology has stopped enforcing the invariant and this whole
    # gate is measuring nothing.
    try:
        topology_cls(nodes=1, gpus_per_node=2, dp=1, tp=1, pp=1, ep=1, cp=1)
        c2 = False
        c2_detail = "Topology ACCEPTED dp=1 at width 2 -- the invariant is gone"
    except ValueError as exc:
        c2 = True
        c2_detail = str(exc).splitlines()[0][:80]
    record("C2 the trainer's Topology REFUSES dp=1 at width 2", c2, c2_detail)

    try:
        topology_cls(nodes=1, gpus_per_node=2, dp=2, tp=1, pp=1, ep=1, cp=1)
        c3, c3_detail = True, "dp=2 at width 2 constructs"
    except ValueError as exc:
        c3, c3_detail = False, f"Topology REFUSED a valid declaration: {exc}"
    record("C3 the trainer's Topology ADMITS dp=2 at width 2", c3, c3_detail)

    # C4/C5: the same _judge() main() calls, driven with synthetic flags. C4 is
    # the pre-#439 shape of every one of the four rows, verbatim.
    pre_fix = {"--nodes": "1", "--gpus-per-node": "2"}
    ok4, line4 = _judge(topology_cls, "synthetic-pre-439", pre_fix)
    record(
        "C4 MUST_FIRE: the judge REJECTS the pre-#439 shape (no --dp at width 2)",
        not ok4,
        line4[:96],
    )

    post_fix = {"--nodes": "1", "--gpus-per-node": "2", "--dp": "2"}
    ok5, line5 = _judge(topology_cls, "synthetic-post-439", post_fix)
    record(
        "C5 the judge ACCEPTS the post-#439 shape (--dp = nodes*gpus_per_node)",
        ok5,
        line5[:96],
    )

    # C6-C8: the normaliser. Two rows emit a dict and two emit a flat argv, so a
    # normaliser that silently mishandles either shape would make this gate read
    # every degree as its default of 1 -- i.e. GREEN on nothing.
    got6 = _as_mapping({"--dp": 2, "--nodes": 1})
    record(
        "C6 _as_mapping passes a dict through as strings",
        got6 == {"--dp": "2", "--nodes": "1"},
        f"{got6}",
    )

    got7 = _as_mapping(["--dp", "2", "--nodes", "1"])
    record(
        "C7 _as_mapping pairs a flat argv",
        got7 == {"--dp": "2", "--nodes": "1"},
        f"{got7}",
    )

    got8 = _as_mapping(["--grad-checkpointing", "--dp", "2"])
    record(
        "C8 _as_mapping does not read the NEXT flag as a value",
        got8 == {"--grad-checkpointing": "", "--dp": "2"},
        f"{got8}",
    )

    # C9: provenance. The whole design claim is that this gate cannot drift from
    # the trainer, and that is only true if the Topology it loaded is the one in
    # this tree rather than whatever is installed.
    mod_file = Path(getattr(sys.modules["foundationscale.topology"], "__file__", ""))
    record(
        "C9 Topology was read from THIS TREE, not an installed distribution",
        SRC in mod_file.resolve().parents,
        f"{mod_file}",
    )

    # C10-C13: every row is importable and its composer is drivable at width 2.
    # These are preconditions of the measurement, so a break here has to read as
    # a broken control rather than as a RED on the rows.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for index, row in enumerate(ROWS, start=10):
            try:
                flags = _drive(row, tmp)
                detail = f"{len(flags)} flags emitted"
                passed = "--nodes" in flags and "--gpus-per-node" in flags
                if not passed:
                    detail = f"composer emitted neither --nodes nor --gpus-per-node: {flags}"
            except Exception as exc:
                passed, detail = False, f"{exc!r}"
            record(f"C{index} {row} is drivable at width {WIDTH}", passed, detail)

    width = max(len(name) for name, _, _ in controls)
    failures = [name for name, passed, _ in controls if not passed]
    for name, passed, detail in controls:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")

    behaved = len(controls) - len(failures)
    if failures:
        print(
            f"\nSELF-TEST DENOMINATOR: {behaved} of {len(controls)} controls behaved; "
            f"{len(failures)} did not: {', '.join(failures)}"
        )
        return RED
    print(
        f"\nSELF-TEST DENOMINATOR: {behaved} of {len(controls)} controls behaved; "
        "1x MUST_FIRE (C4) rejected the pre-#439 shape, so the judge is not "
        "passing by construction"
    )
    return CLEAR


def main(argv: list[str] | None = None) -> int:
    # Hand-parsed rather than argparse, deliberately: argparse exits 2 on a usage
    # error and 0 on --help, and both sit outside the 0/5/95/96 contract (#387).
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--self-test"]:
        return self_test()
    if args:
        print(
            f"REFUSE 96: unrecognised arguments {args}; usage: {Path(__file__).name} [--self-test]"
        )
        return REFUSE
    return measure()


if __name__ == "__main__":
    sys.exit(main())
