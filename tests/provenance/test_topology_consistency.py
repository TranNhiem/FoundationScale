"""Controls for #221/#222: a recorded topology must be arithmetically possible.

#221. ``provenance.manifest.Topology`` was a bare dataclass with no validation,
and it is the one wired to ``RunManifest.topology``. Measured before the fix: a
record stating nodes=2, gpus_per_node=8 (16 GPUs) with tp=4, pp=2, cp=2, dp=2
(product 32) constructed, round-tripped through ``to_dict``/``from_dict``, and
fingerprinted intact. A fingerprint whose stated purpose is to distinguish runs
by topology could not distinguish a possible layout from an impossible one.

#222. The obvious repair -- delegate to ``foundationscale.topology.Topology``,
which does validate on construction -- is wrong, and the legs below pin why.
The two classes use the SAME field names for DIFFERENT quantities:

  * here, ``data_parallel`` is Megatron's ``data_parallel_size``; world size is
    ``tp x pp x cp x dp`` and expert parallelism is carved OUT of the DP group;
  * there, ``dp`` is the count of complete model replicas, and construction
    requires ``dp x tp x pp x ep x cp == total``.

They agree exactly at ``ep == 1``. ``test_must_pass_a_real_moe_layout_is_consistent``
is the divergence made executable: an 8-GPU DP=8/EP=8 run is legitimate in this
vocabulary and would be rejected outright by that class's gate (8x8 = 64 != 8).
The bridge would have blocked every MoE run. The reason nobody has been bitten
is that the validating class has no production consumer at all.

MUST_PASS: valid dense and valid MoE layouts read CONSISTENT; the emitter emits
them; a historical impossible record still LOADS (auditing an incident must not
require the record to be well-formed).
MUST_FIRE: an impossible world size and an indivisible expert split each read
INCONSISTENT; the emitter REFUSES both with rc pinned and the arithmetic named.
UNMEASURED: a degree the launcher never recorded abstains -- and the abstention
must not read as a pass, which is the leg that would go quiet if ``consistency``
ever started substituting a 1 for ``None``.

Precedence is RED > UNMEASURED > CONSISTENT, pinned by its own leg: a record
with both an undecidable leg and a failing one is INCONSISTENT, never excused
into an abstention.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from foundationscale.provenance import Topology, TopologyConsistency

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EMITTER_PATH = ROOT / "tools" / "emit_run_manifest.py"


def _load_emitter():
    """Import the emitter by path; tools/ is deliberately not a package.

    A failed load raises here rather than skipping: an emitter that cannot be
    imported makes every leg below UNMEASURED, and an unmeasured control that
    reports green is the exact shape this suite exists to refuse.
    """
    spec = importlib.util.spec_from_file_location("_erm_topology", EMITTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EMITTER_PATH}")  # FAIL CLOSED
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


erm = _load_emitter()


def _dense(**over: int | None) -> Topology:
    """A layout that is possible: 16 GPUs, tp x pp x cp x dp = 4x2x1x2 = 16."""
    base: dict[str, int | None] = dict(
        nodes=2,
        gpus_per_node=8,
        tensor_parallel=4,
        pipeline_parallel=2,
        context_parallel=1,
        data_parallel=2,
        expert_parallel=1,
    )
    base.update(over)
    return Topology(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# MUST_PASS
# --------------------------------------------------------------------------


def test_must_pass_a_valid_dense_layout_is_consistent() -> None:
    verdict = _dense().consistency()
    assert verdict.verdict == "CONSISTENT", verdict.detail
    assert verdict.blocking is False
    # The detail must carry the arithmetic, not just the word. A verdict that
    # states no numbers cannot be audited against the record it describes.
    assert "16" in verdict.detail, verdict.detail


def test_must_pass_a_real_moe_layout_is_consistent() -> None:
    """#222 made executable: DP=8/EP=8 on one node is legitimate HERE.

    ``foundationscale.topology.Topology`` requires dp x tp x pp x ep x cp ==
    total, which is 1x1x1x8x8 = 64 against 8 GPUs -- it would reject this. The
    two vocabularies are not interchangeable, so this class validates in its
    own, and this leg is the standing proof that the bridge R1 recommended
    would have blocked every expert-parallel run.
    """
    verdict = Topology(
        nodes=1,
        gpus_per_node=8,
        tensor_parallel=1,
        pipeline_parallel=1,
        context_parallel=1,
        data_parallel=8,
        expert_parallel=8,
    ).consistency()
    assert verdict.verdict == "CONSISTENT", verdict.detail
    assert verdict.blocking is False


def test_must_pass_an_impossible_historical_record_still_loads() -> None:
    """Validation lives at the mint site, never in the loader.

    A record already on disk must keep opening -- an impossible one most of
    all, because reading it is how the incident that produced it gets
    diagnosed. This leg goes red the moment somebody "helpfully" moves the
    check into ``__post_init__`` or ``from_dict``.
    """
    impossible = _dense(context_parallel=2)  # 4x2x2x2 = 32 against 16 GPUs
    restored = Topology.from_dict(impossible.to_dict())
    assert restored == impossible
    assert restored.consistency().verdict == "INCONSISTENT"


# --------------------------------------------------------------------------
# MUST_FIRE
# --------------------------------------------------------------------------


def test_must_fire_an_impossible_world_size_is_inconsistent() -> None:
    """The measured #221 record: 16 GPUs, degrees multiplying to 32."""
    verdict = _dense(context_parallel=2).consistency()
    assert verdict.verdict == "INCONSISTENT", verdict.detail
    assert verdict.blocking is True
    assert "32" in verdict.detail and "16" in verdict.detail, verdict.detail


def test_must_fire_an_indivisible_expert_split_is_inconsistent() -> None:
    """Expert parallelism partitions the DP group, so ep must divide dp."""
    verdict = _dense(expert_parallel=3).consistency()  # dp=2, 2 % 3 = 2
    assert verdict.verdict == "INCONSISTENT", verdict.detail
    assert verdict.blocking is True
    assert "expert split" in verdict.detail, verdict.detail


def test_must_fire_a_decidable_failure_outranks_an_undecidable_leg() -> None:
    """Precedence RED > UNMEASURED: one blind leg must not launder the other.

    Without the precedence rule an impossible expert split would be reported
    as UNMEASURED -- non-blocking -- merely because a DIFFERENT degree was
    never recorded. That is the vacuous-truth failure in miniature.
    """
    verdict = _dense(context_parallel=None, expert_parallel=3).consistency()
    assert verdict.verdict == "INCONSISTENT", verdict.detail
    assert verdict.blocking is True


# --------------------------------------------------------------------------
# UNMEASURED
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("blind", "named"),
    [("context_parallel", "world size"), ("expert_parallel", "expert split")],
)
def test_unmeasured_a_degree_never_recorded_abstains_and_is_not_a_pass(
    blind: str, named: str
) -> None:
    """``None`` means "never recorded" and must not be read as a 1.

    Both halves matter. The verdict must be UNMEASURED (not CONSISTENT, which
    would be a fabricated pass) and it must be non-blocking (not INCONSISTENT,
    which would be a fabricated failure). The state has to exist in its own
    right or the two-valued collapse comes back.
    """
    verdict = _dense(**{blind: None}).consistency()  # type: ignore[arg-type]
    assert verdict.verdict == "UNMEASURED", verdict.detail
    assert verdict.blocking is False
    assert named in verdict.detail, verdict.detail
    assert blind in verdict.detail, verdict.detail


def test_the_three_verdicts_are_all_reachable_from_this_class() -> None:
    """#200's rule applied here: a declared state must be REACHABLE.

    Three states are documented on ``TopologyConsistency``. If one of them
    could never be produced, the docstring would be describing a contract the
    code does not have.
    """
    reached = {
        _dense().consistency().verdict,
        _dense(context_parallel=2).consistency().verdict,
        _dense(expert_parallel=None).consistency().verdict,
    }
    assert reached == {"CONSISTENT", "INCONSISTENT", "UNMEASURED"}, reached


def test_only_inconsistent_blocks() -> None:
    assert TopologyConsistency("INCONSISTENT", "x").blocking is True
    assert TopologyConsistency("UNMEASURED", "x").blocking is False
    assert TopologyConsistency("CONSISTENT", "x").blocking is False


# --------------------------------------------------------------------------
# The emitter: the check is worthless if the one mint site never calls it
# --------------------------------------------------------------------------


def _emit_argv(tmp_path: Path, run_id: str, **degrees: str) -> list[str]:
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    argv = [
        "--lora",
        "--out-dir",
        str(out_dir),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--run-id",
        run_id,
    ]
    for flag, value in degrees.items():
        argv += [f"--{flag.replace('_', '-')}", value]
    return argv


def test_must_fire_the_emitter_refuses_an_impossible_topology(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """rc pinned to EXIT_REFUSED, and the refusal must name the arithmetic.

    This is the leg that proves the check is WIRED. Every dataclass leg above
    would stay green if ``_emit`` never called ``consistency()``.
    """
    rc = erm.main(
        _emit_argv(
            tmp_path,
            "topology-impossible",
            nodes="2",
            gpus_per_node="8",
            tp="4",
            pp="2",
            cp="2",
            dp="2",
        )
    )
    assert rc == erm.EXIT_REFUSED, f"impossible topology must refuse, got rc={rc}"
    err = capsys.readouterr().err
    assert "REFUSED" in err and "topology" in err, err
    assert "32" in err and "16" in err, f"the refusal must show its arithmetic:\n{err}"


def test_must_fire_the_emitter_refuses_before_allocating_an_attempt(
    tmp_path: Path,
) -> None:
    """A refused emission must leave no attempt behind.

    Refusing after ``allocate_attempt`` would burn attempt numbers on records
    that were never written, so a later audit of "attempt 3" would find a hole
    with no explanation. Denominator stated: the store subdirectory must not
    exist at all, having never been reached.
    """
    argv = _emit_argv(
        tmp_path,
        "topology-refused-early",
        nodes="1",
        gpus_per_node="8",
        tp="2",
        pp="1",
        cp="1",
        dp="8",  # 2x1x1x8 = 16 against 8
    )
    assert erm.main(argv) == erm.EXIT_REFUSED
    store_dir = tmp_path / "out" / erm._STORE_SUBDIR
    assert not store_dir.exists(), f"an attempt was allocated for a refused emission: {store_dir}"


def test_must_pass_the_emitter_emits_a_consistent_topology_and_states_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the control: a possible layout must still get through.

    A refusal gate that refuses everything is not a gate. The receipt line is
    asserted too -- printed on CONSISTENT as well, so a reader can tell
    "checked and fine" from "never checked".
    """
    rc = erm.main(
        _emit_argv(
            tmp_path,
            "topology-possible",
            nodes="2",
            gpus_per_node="8",
            tp="4",
            pp="2",
            cp="1",
            dp="2",
            ep="1",
        )
    )
    out = capsys.readouterr().out
    assert rc == erm.EXIT_OK, f"a possible topology must emit, got rc={rc}\n{out}"
    assert "topology" in out and "CONSISTENT" in out, out


def test_the_emitter_states_unmeasured_rather_than_reporting_a_silent_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--ep`` is optional, so an ordinary run abstains on the expert leg.

    It emits -- withholding provenance because one degree is unknown loses the
    whole record to protect one field of it -- but the receipt must SAY so.
    The failure this pins is the tempting one: defaulting the absent ep to 1
    and printing CONSISTENT over a quantity nothing measured.
    """
    rc = erm.main(
        _emit_argv(
            tmp_path,
            "topology-abstains",
            nodes="1",
            gpus_per_node="4",
            tp="2",
            pp="1",
            cp="2",
            dp="1",
        )
    )
    out = capsys.readouterr().out
    assert rc == erm.EXIT_OK, out
    assert "UNMEASURED" in out, f"the abstention must be stated:\n{out}"
    assert "CONSISTENT" not in out, f"an abstention read as a pass:\n{out}"
