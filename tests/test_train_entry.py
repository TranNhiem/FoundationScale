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

from foundationscale.gates.core import Coverage, Gate, GateRegistry, Lifecycle
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
    poison = _ExplodingModule()
    corpus = tmp_path / "launchers"
    corpus.mkdir()
    (corpus / "a.sh").write_text("#SBATCH --partition=hhai\n", encoding="utf-8")
    (corpus / "b.sh").write_text("#SBATCH --partition=hh_ai\n", encoding="utf-8")
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
