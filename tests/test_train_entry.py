"""Controls for the training entry. NO skips: torch and transformers are NOT
installed in CI, so this suite tests everything testable without them and the
two MUST_FIRE controls prove the failure paths are reachable and observed.

Ordering invariant under test: validation BLOCKS before any torch import.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundationscale.gates.core import Coverage, Gate, GateRegistry, Lifecycle, Verdict
from foundationscale.topology import ClusterProfile
from foundationscale.train.cli import main as cli_main
from foundationscale.train.loop import (
    EXIT_PASS,
    EXIT_RED,
    EXIT_REFUSE,
    FoundationScaleSaveGate,
    TrainConfig,
    train,
)

PROFILE_DATA: dict = {
    "name": "reference-single-node",
    "scheduler": "slurm",
    "partitions": ["batch"],
    # A REGEX, not a Slurm nodelist expansion: "compute-[01-08]" is a character
    # class containing the range 1-0, which re.compile refuses. ClusterProfile
    # caught that correctly -- the fixture was the defect, not the guard.
    "node_pattern": r"compute-0[1-8]",
    "gpus_per_node": 1,
    "nccl_socket_ifname": "eth0",
    "ib_hca_pattern": "mlx5_*",
    "mnnvl_available": False,
    "container_runtime": "none",
    "container_image": "",
    "filesystem_roots": ["/tmp"],
    "max_nodes": 8,
}


@pytest.fixture(autouse=True)
def _clean_launch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No leaked torchrun/scheduler evidence may steer the prologue."""
    for var in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_JOB_ID",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def profile_path(tmp_path: Path) -> Path:
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(PROFILE_DATA), encoding="utf-8")
    return path


def _cfg(tmp_path: Path, profile_file: Path, **overrides) -> TrainConfig:
    kwargs = {
        "model": "tiny/model",
        "dataset": "tiny/dataset",
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": 1,
        "dp": 1,
        "profile_path": profile_file,
        "dry_run": True,
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


class _ExplodingModule:
    """Raises on ANY attribute access -- proof that train() never reached torch."""

    def __init__(self) -> None:
        self.touched = False

    def __getattr__(self, name: str):
        self.touched = True
        raise AssertionError(
            f"attribute {name!r} accessed on a poisoned module: validation "
            "must complete BEFORE torch/transformers are touched"
        )


# ---------------------------------------------------------------------------
# MUST_PASS: TrainConfig construction and required-field validation
# ---------------------------------------------------------------------------


def test_config_construction_and_frozen(tmp_path: Path, profile_path: Path) -> None:
    cfg = _cfg(tmp_path, profile_path)
    assert cfg.seed == 42
    assert cfg.learning_rate == 5e-5
    assert cfg.dp == cfg.tp == cfg.pp == cfg.ep == cfg.cp == 1
    assert cfg.dry_run is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dp = 2  # type: ignore[misc]


def test_config_missing_required_field_raises(profile_path: Path) -> None:
    # `model` has no default: absence is a construction error, not agreement.
    with pytest.raises(TypeError):
        TrainConfig(  # type: ignore[call-arg]
            dataset="tiny/dataset",
            output_dir=Path("out"),
            nodes=1,
            gpus_per_node=1,
            profile_path=profile_path,
        )


def test_config_machine_facts_have_no_defaults(profile_path: Path) -> None:
    with pytest.raises(TypeError):  # nodes missing
        TrainConfig(  # type: ignore[call-arg]
            model="m",
            dataset="d",
            output_dir=Path("out"),
            gpus_per_node=1,
            profile_path=profile_path,
        )
    with pytest.raises(TypeError):  # gpus_per_node missing
        TrainConfig(  # type: ignore[call-arg]
            model="m",
            dataset="d",
            output_dir=Path("out"),
            nodes=1,
            profile_path=profile_path,
        )


def test_config_exactly_one_profile(tmp_path: Path, profile_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _cfg(tmp_path, profile_path, profile_path=None)  # none provided
    with pytest.raises(ValueError, match="exactly one"):
        _cfg(
            tmp_path,
            profile_path,
            profile_name="whatever",  # both provided
        )
    ok = _cfg(
        tmp_path,
        profile_path,
        profile_path=None,
        profile=ClusterProfile.from_dict(PROFILE_DATA),
    )
    assert ok.profile is not None


def test_config_rejects_nonpositive_degree(tmp_path: Path, profile_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _cfg(tmp_path, profile_path, nodes=0)


# ---------------------------------------------------------------------------
# MUST_PASS: the full --dry-run prologue, with zero torch involvement.
# ---------------------------------------------------------------------------


def test_dry_run_prologue_passes_without_torch(
    tmp_path: Path,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    poison = _ExplodingModule()
    monkeypatch.setitem(sys.modules, "torch", poison)
    cfg = _cfg(tmp_path, profile_path)
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc == EXIT_PASS, out
    assert poison.touched is False  # no torch attribute was ever reached
    # Every declared prologue step emitted its marker, in order.
    steps = [
        "fs:train:start",
        "fs:train:topology",
        "fs:train:profile",
        "fs:train:consistency",
        "fs:train:validated",
        "fs:train:manifest",
        "fs:train:done",
    ]
    positions = [out.index(f"[{s}]") for s in steps]
    assert positions == sorted(positions), out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# MUST_FIRE: a topology violating its ClusterProfile returns the blocking
# code BEFORE any torch import is attempted (observed red, ordering proven).
# ---------------------------------------------------------------------------


def test_profile_violation_blocks_before_any_torch(
    tmp_path: Path,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    poison_torch = _ExplodingModule()
    poison_transformers = _ExplodingModule()
    monkeypatch.setitem(sys.modules, "torch", poison_torch)
    monkeypatch.setitem(sys.modules, "transformers", poison_transformers)
    cfg = _cfg(
        tmp_path,
        profile_path,
        # Internally consistent (dp == nodes*gpus_per_node) but violates the
        # profile TWICE: 8 gpus/node vs 1 in the profile, 4 nodes vs max 8 ok
        # but gpus_per_node mismatch is a hardware fact. gpn/nodes chosen so
        # at least one hardware finding must block.
        nodes=4,
        gpus_per_node=8,
        dp=32,
        dry_run=True,
    )
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc == EXIT_RED == 5, out
    assert poison_torch.touched is False
    assert poison_transformers.touched is False
    assert "[fs:train:blocked]" in out
    # Blocking marker precedes BOTH the done marker's absence and any deps line.
    assert "[fs:train:deps]" not in out
    assert "[fs:train:done]" not in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# MUST_FIRE: torch absent -> 96 REFUSE naming the extra, never a traceback.
# ---------------------------------------------------------------------------


def test_missing_torch_refuses_naming_extra(
    tmp_path: Path,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # None in sys.modules makes `import torch` raise ImportError, exactly as
    # on a login node with nothing installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "datasets", None)
    cfg = _cfg(tmp_path, profile_path, dry_run=False)
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc == EXIT_REFUSE == 96, out
    assert "foundationscale[train]" in out  # the extra is NAMED
    assert "[fs:train:refuse]" in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# Save-gate callback controls. A gate that fires and lets training continue
# is a check that cannot fail -- the MUST_FIRE control proves it stops the run.
# ---------------------------------------------------------------------------


class _MustFireGate(Gate):
    id = "test.save_gate.must_fire"
    description = "deliberately broken checkpoint: always blocks"
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    context_type = None

    def check(self, ctx):
        return self.fail(
            "deliberately broken checkpoint (control)",
            Coverage(checked=1, unit="tensor", expected=1),
        )

    def coerce_context(self, ctx):
        return ctx

    def controls(self):
        return []


class _MustPassGate(Gate):
    id = "test.save_gate.must_pass"
    description = "known-good checkpoint: always verifies"
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    context_type = None

    def check(self, ctx):
        return self.ok(
            "known-good checkpoint (control)",
            Coverage(checked=4, unit="tensor", expected=4),
        )

    def coerce_context(self, ctx):
        return ctx

    def controls(self):
        return []


def _fake_hf(tmp_path: Path):
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=7)
    control = SimpleNamespace(should_training_stop=False)
    return args, state, control


def test_callback_must_fire_stops_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = GateRegistry()
    registry.register(_MustFireGate())
    cb = FoundationScaleSaveGate(registry=registry, context_builder=lambda p: object())
    args, state, control = _fake_hf(tmp_path)
    cb.on_save(args, state, control)
    out = capsys.readouterr().out
    assert control.should_training_stop is True  # the run STOPS
    assert cb.blocked is True
    # Denominators: exactly one sweep, one gate, one examined unit of one expected.
    assert len(cb.reports) == 1
    report = cb.reports[0]
    assert len(report.results) == 1
    assert report.results[0].coverage.checked == 1
    assert report.results[0].coverage.expected == 1
    assert report.results[0].coverage.is_vacuous is False
    assert cb.records[0]["verdicts"] is not None
    assert "1/1 gates" in out
    assert "should_training_stop=True" in out


def test_callback_must_pass_leaves_training_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = GateRegistry()
    registry.register(_MustPassGate())
    cb = FoundationScaleSaveGate(registry=registry, context_builder=lambda p: object())
    args, state, control = _fake_hf(tmp_path)
    cb.on_save(args, state, control)
    out = capsys.readouterr().out
    assert control.should_training_stop is False
    assert cb.blocked is False
    assert len(cb.reports) == 1
    cov = cb.reports[0].results[0].coverage
    assert (cov.checked, cov.expected) == (4, 4)  # denominators honoured
    assert cov.is_short is False
    assert "1/1 gates" in out


def test_callback_unmeasured_context_never_blocks_nor_passes_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(_path):
        raise RuntimeError("no readable checkpoint metadata")

    registry = GateRegistry()
    registry.register(_MustPassGate())
    cb = FoundationScaleSaveGate(registry=registry, context_builder=_boom)
    args, state, control = _fake_hf(tmp_path)
    cb.on_save(args, state, control)
    out = capsys.readouterr().out
    assert control.should_training_stop is False  # UNMEASURED is not RED...
    assert cb.reports == []  # ...and it is never collapsed into PASS either
    assert "UNMEASURED" in out
    assert "0/0 gates" in out  # zero denominator reported, not hidden


# ---------------------------------------------------------------------------
# CLI controls: --version exists (previously ZERO) and --dry-run is GPU-free.
# ---------------------------------------------------------------------------


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--version"])
    assert excinfo.value.code == 0
    assert "foundationscale-train" in capsys.readouterr().out


def test_cli_dry_run(
    tmp_path: Path,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    poison = _ExplodingModule()
    monkeypatch.setitem(sys.modules, "torch", poison)
    rc = cli_main(
        [
            "--model",
            "tiny/model",
            "--dataset",
            "tiny/dataset",
            "--output-dir",
            str(tmp_path / "out"),
            "--nodes",
            "1",
            "--gpus-per-node",
            "1",
            "--profile-path",
            str(profile_path),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == EXIT_PASS, out
    assert poison.touched is False
    assert "[fs:train:done]" in out


# ---------------------------------------------------------------------------
# The partition scan: both halves of its control. Pointing this detector at the
# cluster-profile JSON (which declares `partitions` as a FIELD, never as an
# sbatch line) made it find zero declarations and block EVERY run -- correctly,
# but for a reason unrelated to the run. These two legs pin the repair: a real
# launcher corpus is scanned, and its absence is DECLARED, never assumed clean.
# ---------------------------------------------------------------------------


def test_partition_scan_unmeasured_when_no_corpus_supplied(
    tmp_path: Path,
    profile_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST_PASS: no corpus -> UNMEASURED is stated, and the run still passes."""
    rc = train(_cfg(tmp_path, profile_path))
    out = capsys.readouterr().out
    assert rc == EXIT_PASS, out
    assert "[fs:train:partition]" in out
    assert "UNMEASURED" in out
    # The absent measurement must not be laundered into a clean one.
    assert "partition_consistent" not in out


def test_partition_scan_reads_a_real_launcher_corpus(
    tmp_path: Path,
    profile_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST_PASS: a corpus with ONE spelling scans clean over a real denominator."""
    corpus = tmp_path / "launchers"
    corpus.mkdir()
    (corpus / "a.sh").write_text("#SBATCH --partition=batch\n", encoding="utf-8")
    (corpus / "b.sbatch").write_text("srun -p batch hostname\n", encoding="utf-8")
    rc = train(_cfg(tmp_path, profile_path, launch_corpus=corpus))
    out = capsys.readouterr().out
    assert rc == EXIT_PASS, out
    assert "scanning 2 launcher file(s)" in out  # denominator is stated
    assert "UNMEASURED" not in out


def test_partition_scan_blocks_on_spelling_variants(
    tmp_path: Path,
    profile_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST_FIRE: two spellings of one partition is an observed red, before any GPU."""
    # The two spellings are deliberately NEUTRAL -- gpu-a100 / gpu_a100, the pair
    # topology.partition_consistency names as its own positive control. They were
    # the real estate's partition and one of its variants until #248, which put a
    # cluster-internal identifier into a PUBLIC repository as fixture data. It went
    # unnoticed because neither pre-push scanner carries that vocabulary: one holds
    # product literals and credential patterns, the other holds estate PATH shapes,
    # and a bare partition token is in neither denominator. The detector normalises
    # separators, so any two spellings exercise the same branch -- realism buys this
    # test nothing and costs a redaction leak. Do not "restore" the real names.
    poison = _ExplodingModule()
    corpus = tmp_path / "launchers"
    corpus.mkdir()
    (corpus / "a.sh").write_text("#SBATCH --partition=gpu-a100\n", encoding="utf-8")
    (corpus / "b.sh").write_text("#SBATCH --partition=gpu_a100\n", encoding="utf-8")
    cfg = _cfg(tmp_path, profile_path, launch_corpus=corpus, dry_run=False)
    import sys as _sys

    _saved = _sys.modules.get("torch")
    _sys.modules["torch"] = poison  # type: ignore[assignment]
    try:
        rc = train(cfg)
    finally:
        if _saved is None:
            _sys.modules.pop("torch", None)
        else:
            _sys.modules["torch"] = _saved
    out = capsys.readouterr().out
    assert rc == EXIT_RED, out
    assert "scanning 2 launcher file(s)" in out
    assert poison.touched is False  # blocked BEFORE torch


# ---------------------------------------------------------------------------
# #250: the save gate dispatches on declared context type, and an abstention is
# declared rather than fatal.
#
# The loop used to call GateRegistry.run, which broadcasts one context to every
# gate registered for the event. A gate declaring a different context family got
# a CheckpointGateContext and died inside check() as a raw AttributeError one
# frame down, which the sweep scores as a blocking ERROR and the callback turns
# into should_training_stop. Two real gates were in that state
# (checkpoint.weight_parity, objective.hparam_drift), so whether a run trained at
# all depended on whether anything in the process had imported their modules --
# registration is an import side effect. These controls pin the dispatch, not the
# two gates: a third gate declaring a third context would otherwise reintroduce
# the class silently.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ForeignContext:
    """A context no checkpoint builder produces."""

    token: str


class _ForeignContextGate(Gate):
    """Declares a context the save path does not build, and does NOT override
    coerce_context -- the base refuses by returning None. This is deliberately the
    exact shape of checkpoint.weight_parity, so the control fails the same way the
    real gate did rather than a way invented for the test."""

    id = "test.save_gate.foreign_context"
    description = "declares a context the save path does not build"
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    context_type = _ForeignContext

    def check(self, ctx):
        # Reached only with a _ForeignContext. fail() rather than ok() so that the
        # positive control below is unambiguous: if dispatch ever hands this gate
        # the checkpoint context again, the run stops and the suite says so.
        return self.fail(
            f"foreign context observed: {ctx.token}",
            Coverage(checked=1, unit="token", expected=1),
        )

    def controls(self):
        return []


def test_foreign_context_gate_abstains_and_training_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate whose declared context the caller does not build must SKIP.

    Pre-#250 this was a blocking ERROR and the run stopped here. The assertions
    below are the four separable claims: the run continues, the abstention is
    SKIP and not PASS, the gate stays inside the printed denominator, and the
    reason names the missing type so the SKIP is attributable.
    """
    registry = GateRegistry()
    registry.register(_ForeignContextGate())
    cb = FoundationScaleSaveGate(registry=registry, context_builder=lambda p: object())
    args, state, control = _fake_hf(tmp_path)
    cb.on_save(args, state, control)
    out = capsys.readouterr().out

    assert control.should_training_stop is False  # the run CONTINUES
    assert cb.blocked is False
    assert len(cb.reports) == 1
    results = cb.reports[0].results
    # Still counted: an abstention that vanished from the denominator would make
    # a 0/0 sweep read as complete, which is doctrine 1 with extra steps.
    assert len(results) == 1
    assert results[0].gate_id == "test.save_gate.foreign_context"
    assert results[0].verdict is Verdict.SKIP
    assert "_ForeignContext" in str(results[0].detail)
    assert "1/1 gates" in out
    # The pre-fix signature was a traceback from inside check(); its absence is
    # what distinguishes "dispatched away" from "crashed quietly".
    assert "Traceback" not in out


def test_foreign_context_gate_fires_when_its_context_is_supplied(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control for the test above: the gate is not merely inert.

    A SKIP proves nothing on its own -- a gate that could never fire would also
    produce one. Handing this gate the context it declares must reach check() and
    stop the run, so the abstention above is attributable to dispatch rather than
    to a dead gate.
    """
    registry = GateRegistry()
    registry.register(_ForeignContextGate())
    cb = FoundationScaleSaveGate(
        registry=registry,
        context_builder=lambda p: _ForeignContext(token="wired"),
    )
    args, state, control = _fake_hf(tmp_path)
    cb.on_save(args, state, control)
    out = capsys.readouterr().out

    assert control.should_training_stop is True  # the run STOPS
    assert cb.blocked is True
    results = cb.reports[0].results
    assert len(results) == 1
    assert results[0].verdict is Verdict.FAIL
    assert "foreign context observed: wired" in str(results[0].detail)
    assert "should_training_stop=True" in out


def test_legacy_and_typed_gates_coexist_in_one_sweep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mixed registry is the migration state, and it is where #250 bit.

    A legacy gate (context_type is None) must still receive the bare context
    unchanged, while a gate declaring a foreign type abstains -- in the SAME
    sweep, with both inside the denominator. Asserting them separately would not
    catch a dispatch that handles each shape correctly only when it is alone.
    """
    registry = GateRegistry()
    registry.register(_MustFireGate())  # legacy: reads whatever it is handed
    registry.register(_ForeignContextGate())  # typed: declares a foreign context
    cb = FoundationScaleSaveGate(registry=registry, context_builder=lambda p: object())
    args, state, control = _fake_hf(tmp_path)
    cb.on_save(args, state, control)
    out = capsys.readouterr().out

    verdicts = {r.gate_id: r.verdict for r in cb.reports[0].results}
    assert verdicts == {
        "test.save_gate.must_fire": Verdict.FAIL,
        "test.save_gate.foreign_context": Verdict.SKIP,
    }
    # The legacy gate's FAIL is what stops the run; the typed gate's abstention
    # neither adds to nor subtracts from that verdict.
    assert control.should_training_stop is True
    assert "2/2 gates" in out
    assert "Traceback" not in out
