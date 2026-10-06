"""M4 proposer tests: the CMA-ES adapter, package versions and the close-time re-check (C2/C4/C6)."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from foundationskills.skills.auto_research import proposers
from foundationskills.skills.auto_research.proposers import (
    CatalogProposer,
    _card_in_axes,
    get_proposer,
    package_version,
    reverify,
    select,
)
from foundationskills.skills.auto_research.proposers_optuna import OptunaProposer

METRIC = "val_accuracy"
FPR = "sha256:" + "a1" * 32
SPEC: dict[str, Any] = {
    "id": "c1",
    "objective": {"metric": METRIC, "direction": "max", "benchmarks": []},
    "eval_policy": {"fingerprint": FPR, "metrics": [METRIC]},
    "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [1, 2, 3]},
    "axes": [
        {"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3},
        {"key": "optim.warmup_ratio", "type": "float", "min": 0.0, "max": 0.1},
        {"key": "lora.rank", "type": "int", "min": 4, "max": 64},
        {"key": "train.method", "type": "categorical", "values": ["full", "lora"]},
    ],
}
CURRENT: dict[str, Any] = {"optim.lr": 1e-3, "optim.warmup_ratio": 0.02, "lora.rank": 24, "train.method": "lora"}
SYMPTOMS: list[str] = ["loss diverging"]
BASELINES: list[dict[str, Any]] = [
    {"trial": "baseline", "role": "baseline", "seed": i, "status": "ok", "limited": False,
     "metrics": {METRIC: {"value": 0.5, "se": 0.001}}}
    for i in (1, 2, 3)
]
DRAW: dict[str, Any] = {"optim.lr": 2e-4, "optim.warmup_ratio": 0.05, "lora.rank": 8, "train.method": "lora"}
TAMPERED: dict[str, Any] = {"idea": "tampered", "delta": dict(CURRENT), "score": None, "reason": "tampered"}
STUB_PACKAGE = "stub-pkg-1"


def trial_row(trial: str, value: float | None, *, delta: dict[str, Any] | None = None) -> dict[str, Any]:
    """One trial_result row shaped like a ledger payload (test_proposers.py)."""
    row: dict[str, Any] = {
        "trial": trial, "role": "candidate", "seed": 1, "status": "ok", "limited": False,
        "eval_policy_fingerprint": FPR, "metrics": {METRIC: {"value": value, "se": 0.0}},
    }
    if delta is not None:
        row["delta"] = delta
    return row


def model_ledger(n: int = 2) -> list[dict[str, Any]]:
    """n measured non-baseline trials on top of the calibrated baseline repeats."""
    return [*BASELINES, *(trial_row(f"t{i}", 0.2 + 0.01 * i, delta={"optim.lr": (i + 1) * 1e-4}) for i in range(n))]


def spec_with(name: str, *, min_rows: int = 10) -> dict[str, Any]:
    """SPEC carrying a proposer block."""
    return {**SPEC, "proposer": {"name": name, "min_rows": min_rows}}


def record(name: str, cards: Any, *, package: str | None = "builtin", k: int = 2,
           results_count: int = 0, launches_count: int = 0, status: str | None = "byte_identical") -> dict[str, Any]:
    """One recorded ``proposal`` payload (C5): the provenance block plus the replay record."""
    payload: dict[str, Any] = {
        "proposer": {"name": name, "version": "1"}, "package_version": package, "seed": 0, "k": k, "cards": cards,
        "replay_inputs": {"current": CURRENT, "symptoms": SYMPTOMS,
                          "results_count": results_count, "launches_count": launches_count},
    }
    if status is not None:
        payload["replay_status"] = status
    return payload


def live_cards(who: str, *, k: int = 2, results: Any = (), launches: Any = ()) -> list[dict[str, Any]]:
    """The cards a record of ``who`` holds: the raw catalog output or only the in-axes model cards (C6)."""
    model = get_proposer(who, 0, REGISTRY)
    raw, _drops = model.propose(SPEC, list(results or ()), list(launches or ()), CURRENT, SYMPTOMS, k=k)
    return list(raw) if who == "catalog" else [card for card in raw if _card_in_axes(SPEC, card)]


# ---- fakes: a fake optuna module and a deterministic model stub -------------


class FakeTPESampler:
    """Stands for ``optuna.samplers.TPESampler(seed=...)``."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed


class FakeCmaEsSampler:
    """Stands for ``optuna.samplers.CmaEsSampler(seed=...)``."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed


class FakeDist:
    """Stands for any optuna distribution (only its key matters here)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class FakeStudy:
    """Serves the one canned in-axes draw and records its asks."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    def add_trial(self, trial: Any) -> None:
        return None

    def ask(self, dists: dict[str, Any]) -> SimpleNamespace:
        self.asked.append(dict(dists))
        return SimpleNamespace(params={key: DRAW[key] for key in dists})


def fake_optuna(version: str = "3.6.0-fake") -> SimpleNamespace:
    """A fake optuna module: seeded samplers and one recorded ``create_study`` call per study."""
    calls: list[dict[str, Any]] = []

    def create_study(*, direction: str = "maximize", sampler: Any = None) -> FakeStudy:
        calls.append({"direction": direction, "sampler": sampler})
        return FakeStudy()

    def create_trial(*, params: Any, distributions: Any, value: Any) -> SimpleNamespace:
        return SimpleNamespace(params=dict(params or {}), distributions=dict(distributions or {}), value=float(value))

    return SimpleNamespace(
        __version__=version,
        distributions=SimpleNamespace(FloatDistribution=FakeDist, IntDistribution=FakeDist,
                                      CategoricalDistribution=FakeDist),
        samplers=SimpleNamespace(TPESampler=FakeTPESampler, CmaEsSampler=FakeCmaEsSampler),
        trial=SimpleNamespace(create_trial=create_trial),
        create_study=create_study,
        logging=SimpleNamespace(WARNING=30, set_verbosity=lambda level: None),
        study_calls=calls,
    )


class StubProposer:
    """Deterministic model stub: k full in-axes assignments plus one out-of-axes draw."""

    name = "stub"
    version = "3"
    package_version = STUB_PACKAGE

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any, *, k: int) -> tuple[list[dict[str, Any]], list[str]]:
        """k in-axes cards keyed on k/seed only (byte-reproducible over any ledger prefix)."""
        cards = [
            {"idea": f"stub:{i}", "delta": {**CURRENT, "optim.lr": 1e-5 * (i + 1)}, "score": None,
             "reason": f"stub seed={self.seed}"}
            for i in range(max(0, k))
        ]
        cards.append({"idea": "stub:out", "delta": {**CURRENT, "optim.warmup_ratio": 9.0}, "score": None,
                      "reason": f"stub seed={self.seed}"})
        return cards, []


class BrokenProposer(StubProposer):
    """A rebuild that raises: a broken extra is unmeasured:model_error, never a crash."""

    package_version = "broken-1"

    def propose(self, spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any, *, k: int) -> tuple[list[dict[str, Any]], list[str]]:
        raise RuntimeError("broken model")


REGISTRY: dict[str, Any] = {"catalog": CatalogProposer, "stub": StubProposer}


def missing_extra(seed: int) -> Any:
    """A lazy factory for an extra that is not installed (get_proposer counts it once)."""
    raise ImportError("no such extra")


JUNK: list[Any] = [
    object(),
    {"replay_inputs": {}, "replay_status": "byte_identical"},
    {"replay_inputs": {"results_count": "two", "launches_count": 0}, "replay_status": "byte_identical"},
    {"replay_inputs": {"results_count": 0, "launches_count": -3}, "replay_status": "byte_identical"},
    {**record("catalog", [{"idea": "nan", "delta": {"optim.lr": float("nan")}}])},
    {**record("catalog", []), "proposer": "catalog"},
    {**record("catalog", []), "proposer": {"name": 7}},
]


class TestPackageVersion:
    """C4: ``package_version`` and the select() provenance key feeding the C5 payload."""

    def test_helper_reads_the_recorded_attribute(self) -> None:
        assert package_version(CatalogProposer()) == "builtin"
        assert package_version(StubProposer()) == STUB_PACKAGE
        assert package_version(SimpleNamespace()) is None

    def test_select_records_builtin_for_catalog_and_for_a_fallback(self) -> None:
        results = model_ledger(2)
        _, _, direct = select(spec_with("catalog"), results, [], CURRENT, SYMPTOMS)
        assert direct["package_version"] == "builtin" and direct["fallback"] is None
        _, _, fallback = select(spec_with("optuna", min_rows=10), results, [], CURRENT, SYMPTOMS)
        assert fallback["package_version"] == "builtin" and fallback["fallback"] == "below_min_rows:2/10"

    def test_select_records_the_returned_model_version(self) -> None:
        spec = spec_with("stub", min_rows=2)
        cards, drops, provenance = select(spec, model_ledger(2), [], CURRENT, SYMPTOMS, k=2, registry=REGISTRY)
        assert provenance["package_version"] == STUB_PACKAGE and provenance["fallback"] is None
        assert provenance["proposer"] == {"name": "stub", "version": "3"}
        assert [card["idea"] for card in cards] == ["stub:0", "stub:1"]
        assert drops == ["model_out_of_axes:stub:out"]


class TestReverify:
    """Every C6 branch in order; reverify answers and never raises."""

    @pytest.mark.parametrize(
        "junk", [None, [], "proposal", {}, {"replay_inputs": None}, {"replay_inputs": "junk"}],
        ids=["none", "list", "str", "empty", "null_inputs", "str_inputs"],
    )
    def test_c6_1_legacy_proposal_is_unmeasured(self, junk: Any) -> None:
        """1. Not a recorded M4 proposal (an M3 one included): unmeasured:legacy_proposal."""
        assert reverify(SPEC, junk, model_ledger(2), []) == ("unmeasured", "legacy_proposal")

    @pytest.mark.parametrize("status", ["unmeasured", "stale", None], ids=["unmeasured", "other", "missing"])
    def test_c6_2_recorded_unmeasured_is_reported(self, status: str | None) -> None:
        """2. The record held no measured replay: unmeasured:recorded_unmeasured."""
        proposal = record("catalog", live_cards("catalog"), status=status)
        assert reverify(SPEC, proposal, model_ledger(2), []) == ("unmeasured", "recorded_unmeasured")

    @pytest.mark.parametrize(
        ("who", "counts"), [("catalog", (5, 0)), ("catalog", (0, 5)), ("stub", (5, 0)), ("stub", (0, 5))],
    )
    def test_c6_3_ledger_prefix_missing(self, who: str, counts: tuple[int, int]) -> None:
        """3. The append-only ledger lost the recorded prefix: unmeasured:ledger_prefix_missing."""
        proposal = record(who, [], results_count=counts[0], launches_count=counts[1])
        assert reverify(SPEC, proposal, [], [], registry=REGISTRY) == ("unmeasured", "ledger_prefix_missing")

    @pytest.mark.parametrize("name", ["gone", "broken_extra"])
    def test_c6_4_extra_missing(self, name: str) -> None:
        """4./5. An unknown name or a raising factory rebuilds to None: unmeasured:extra_missing:<name>."""
        registry = {"catalog": CatalogProposer, "broken_extra": missing_extra}
        assert reverify(SPEC, record(name, []), [], [], registry=registry) == ("unmeasured", f"extra_missing:{name}")

    @pytest.mark.parametrize("who", ["catalog", "stub"])
    def test_c6_5_version_changed(self, who: str) -> None:
        """5. A rebuilt package version that differs: unmeasured:version_changed:<name>."""
        proposal = record(who, live_cards(who), package="previous")
        assert reverify(SPEC, proposal, [], [], registry=REGISTRY) == ("unmeasured", f"version_changed:{who}")

    @pytest.mark.parametrize("who", ["catalog", "stub"])
    def test_c6_6_model_error(self, who: str) -> None:
        """6. A rebuild that raises is unmeasured:model_error, never a crash."""
        proposal = record(who, live_cards(who), package="broken-1")
        assert reverify(SPEC, proposal, [], [], registry={who: BrokenProposer}) == ("unmeasured", "model_error")

    @pytest.mark.parametrize("who", ["catalog", "stub"])
    def test_c6_7_drifted_on_tampered_cards(self, who: str) -> None:
        """6. Cards no replayed proposer emits are drift: ('drifted', None)."""
        package = "builtin" if who == "catalog" else STUB_PACKAGE
        assert reverify(SPEC, record(who, [TAMPERED], package=package), model_ledger(2), [], registry=REGISTRY) == ("drifted", None)

    @pytest.mark.parametrize("who", ["catalog", "stub"])
    def test_c6_8_byte_identical(self, who: str) -> None:
        """6. The rebuilt cards match the record exactly: ('byte_identical', None) over the recorded prefix."""
        package = "builtin" if who == "catalog" else STUB_PACKAGE
        results, launches = model_ledger(2), [{"trial": "t0", "launch_spec": {"trial": "t0"}}]
        cards = live_cards(who, k=2, results=results, launches=launches)
        proposal = record(who, cards, package=package, k=2,
                          results_count=len(results), launches_count=len(launches))
        late_results = [*results, trial_row("late", 0.9, delta={"optim.lr": 5e-4})]
        late_launches = [*launches, {"trial": "late"}]
        assert reverify(SPEC, proposal, late_results, late_launches, registry=REGISTRY) == ("byte_identical", None)

    def test_c6_9_fresh_model_cards_are_filtered_before_the_comparison(self) -> None:
        """A model record holds only the live cards: a fresh out-of-axes draw is filtered, never drift."""
        spec, results = spec_with("stub", min_rows=2), model_ledger(2)
        cards, drops, _ = select(spec, results, [], CURRENT, SYMPTOMS, k=2, registry=REGISTRY)
        proposal = record("stub", cards, package=STUB_PACKAGE, k=2, results_count=len(results))
        assert drops == ["model_out_of_axes:stub:out"]
        assert reverify(spec, proposal, results, [], registry=REGISTRY) == ("byte_identical", None)

    @pytest.mark.parametrize(
        "junk", JUNK,
        ids=["object", "empty_inputs", "str_count", "negative_count", "nan_cards", "str_proposer", "int_name"],
    )
    def test_c6_10_junk_proposals_never_raise(self, junk: Any) -> None:
        """A junk record is a declined claim: unmeasured with a stable reason, never a crash."""
        status, reason = reverify(SPEC, junk, model_ledger(2), [], registry=REGISTRY)
        assert status == "unmeasured" and isinstance(reason, str)


class TestOptunaCma:
    """C2: the ``cmaes`` adapter - sampler wiring, names, package version and the refused value."""

    def test_cmaes_builds_a_seeded_cma_es_sampler(self) -> None:
        """sampler='cmaes' -> CmaEsSampler(seed=seed); the cards keep the exact M3 format."""
        fake = fake_optuna()
        cards, drops = OptunaProposer(seed=5, optuna_module=fake, sampler="cmaes").propose(
            SPEC, model_ledger(2), [], CURRENT, SYMPTOMS, k=1)
        started = fake.study_calls[0]
        assert type(started["sampler"]) is FakeCmaEsSampler and started["sampler"].seed == 5
        assert [card["idea"] for card in cards] == ["optuna:0"] and cards[0]["delta"] == DRAW
        assert cards[0]["reason"] == "cmaes seed=5 told 2 row(s)" and drops == []

    def test_tpe_stays_byte_identical(self) -> None:
        """The default sampler is still the seeded TPESampler with the unchanged M3 card format."""
        fake = fake_optuna()
        cards, _drops = OptunaProposer(seed=2, optuna_module=fake).propose(SPEC, model_ledger(1), [], CURRENT, SYMPTOMS, k=1)
        assert type(fake.study_calls[0]["sampler"]) is FakeTPESampler and fake.study_calls[0]["sampler"].seed == 2
        assert cards == [{"idea": "optuna:0", "delta": DRAW, "score": None, "reason": "tpe seed=2 told 1 row(s)"}]

    def test_names_and_package_versions(self) -> None:
        fake = fake_optuna()
        tpe = OptunaProposer(seed=0, optuna_module=fake)
        cma = OptunaProposer(seed=0, optuna_module=fake, sampler="cmaes")
        assert (tpe.name, cma.name) == ("optuna", "optuna-cma")
        assert (tpe.version, cma.version) == ("3.6.0-fake", "3.6.0-fake")
        assert (package_version(tpe), package_version(cma)) == ("3.6.0-fake", "3.6.0-fake")
        assert OptunaProposer(seed=0, optuna_module=SimpleNamespace()).package_version is None

    def test_invalid_sampler_is_a_construction_error(self) -> None:
        """Any other sampler value is a ValueError at construction (never a silent TPE)."""
        with pytest.raises(ValueError):
            OptunaProposer(seed=0, optuna_module=fake_optuna(), sampler="grid")

    def test_optuna_cma_is_a_known_lazy_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'optuna-cma' is registered lazily: a missing optuna is a missing extra again."""
        monkeypatch.setitem(sys.modules, "optuna", None)
        assert "optuna-cma" in proposers.KNOWN_PROPOSERS
        assert get_proposer("optuna-cma") is None
        assert reverify(SPEC, record("optuna-cma", [], package="3.6.0"), [], [], registry=proposers.REGISTRY) == (
            "unmeasured", "extra_missing:optuna-cma")


def test_an_unbuildable_axis_is_a_named_drop_not_a_silent_skip():
    """A distribution the optuna module cannot build is counted (model_axis_skipped:<key>), never silently lost."""
    fake = fake_optuna()

    def no_log_kwarg(*args: Any) -> FakeDist:  # an optuna whose FloatDistribution rejects log=
        return FakeDist(*args)

    fake.distributions.FloatDistribution = no_log_kwarg
    _cards, drops = OptunaProposer(seed=0, optuna_module=fake).propose(SPEC, model_ledger(1), [], CURRENT, SYMPTOMS, k=1)
    assert "model_axis_skipped:optim.lr" in drops


def test_a_seedless_record_is_legacy_never_replayed_with_a_default_seed() -> None:
    """A missing/non-int seed is unmeasured (legacy_proposal), never a false drift from a defaulted seed 0."""
    clean = record("catalog", live_cards("catalog"))
    assert reverify(SPEC, clean, [], [], registry=REGISTRY) == ("byte_identical", None)
    without = {key: value for key, value in clean.items() if key != "seed"}
    for bad in (without, {**clean, "seed": "0"}, {**clean, "seed": True}):
        assert reverify(SPEC, bad, [], [], registry=REGISTRY) == ("unmeasured", "legacy_proposal")


def test_uncomparable_fresh_cards_are_a_model_error_not_a_legacy_record() -> None:
    """Fresh cards that cannot be canonicalised are model_error (C6 step 6), not legacy_proposal."""

    class OpaqueProposer(StubProposer):
        def propose(self, *args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], list[str]]:
            return [{"idea": "x", "delta": {}, "score": object(), "reason": "r"}], []

    proposal = record("catalog", [], package=STUB_PACKAGE)
    assert reverify(SPEC, proposal, [], [], registry={"catalog": OpaqueProposer}) == ("unmeasured", "model_error")


def test_a_broken_construction_is_a_model_error_not_a_missing_extra() -> None:
    """A factory raising anything but ImportError is model_error: the extra is present, it is broken."""

    def broken_construction(seed: int) -> Any:
        raise ValueError("cma dependency broken")

    proposal = record("stub", [], package=STUB_PACKAGE)
    assert reverify(SPEC, proposal, [], [], registry={"stub": broken_construction}) == ("unmeasured", "model_error")


@pytest.mark.parametrize("k", [True, 2.0, -1, "2", None])
def test_a_junk_k_is_legacy_never_handed_to_the_proposer(k: Any) -> None:
    """select() records a sanitised int k; any other k is a legacy/junk record, not a replay."""
    proposal = {**record("catalog", live_cards("catalog")), "k": k}
    assert reverify(SPEC, proposal, [], [], registry=REGISTRY) == ("unmeasured", "legacy_proposal")


def test_reverify_never_lets_the_proposer_mutate_the_record() -> None:
    """The recorded replay inputs are deep-copied: a mutating proposer cannot alter the ledger record."""

    class MutatingProposer(StubProposer):
        def propose(self, spec: Any, results: Any, launches: Any, current: Any, symptoms: Any, *, k: int) -> Any:
            symptoms.append("derived")
            current["optim.lr"] = -1.0
            return super().propose(spec, results, launches, current, symptoms, k=k)

    proposal = record("stub", [], package=STUB_PACKAGE)
    before = repr(proposal["replay_inputs"])
    reverify(SPEC, proposal, [], [], registry={"stub": MutatingProposer})
    assert repr(proposal["replay_inputs"]) == before
