""""M3 proposer tests: catalog regression oracle, model rows, gating, fallbacks and replay."""
from __future__ import annotations

import copy
import sys

import pytest

from foundationskills.skills.auto_research.ledger import canonical
from foundationskills.skills.auto_research.propose import evaluate as catalog_evaluate
from foundationskills.skills.auto_research.propose import propose as catalog_propose
from foundationskills.skills.auto_research.proposers import (
    DEFAULT_K,
    DEFAULT_MIN_ROWS,
    CatalogProposer,
    _card_in_axes,
    eligible,
    get_proposer,
    model_rows,
    proposer_config,
    replay,
    rows_digest,
    select,
    trial_params,
)

METRIC = "val_accuracy"
FPR = "sha256:" + "a1" * 32
SPEC = {
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
CURRENT = {"optim.lr": 1e-3, "optim.warmup_ratio": 0.02, "lora.rank": 24, "train.method": "lora"}
SYMPTOMS = ["loss diverging"]
BASELINES = [
    {"trial": "baseline", "role": "baseline", "seed": i, "status": "ok", "limited": False,
     "metrics": {METRIC: {"value": 0.5, "se": 0.001}}}
    for i in (1, 2, 3)
]


def trial_row(trial, value, *, seed=1, role="candidate", status="ok", limited=False, delta=None):
    row = {
        "trial": trial,
        "role": role,
        "seed": seed,
        "status": status,
        "limited": limited,
        "eval_policy_fingerprint": FPR,
        "metrics": {METRIC: {"value": value, "se": 0.0}},
    }
    if delta is not None:
        row["delta"] = delta
    return row


def model_ledger(n=10):
    """n measured non-baseline model rows (t0..t{n-1}) on top of the calibrated baseline repeats."""
    rows = list(BASELINES)
    for i in range(n):
        rows.append(trial_row(f"t{i}", 0.2 + 0.01 * i, delta={"optim.lr": (i + 1) * 1e-4}))
    return rows


def catalog_drops(spec, results, launches, current, symptoms):
    """The catalog's own pruned ideas as select() names them: catalog_<reason>:<idea>."""
    return [f"catalog_{c['reason']}:{c['idea']}" for c in catalog_evaluate(spec, results, current, symptoms, launches)[1]]


def spec_with(name, *, min_rows=DEFAULT_MIN_ROWS):
    spec = copy.deepcopy(SPEC)
    spec["proposer"] = {"name": name, "min_rows": min_rows}
    return spec


class StubGood:
    """Deterministic in-axes model stub."""

    name = "stub-good"
    version = "7"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec, results, launches, current, symptoms, *, k):
        cards = [
            {"idea": "stub-idea", "delta": {"optim.lr": 5e-5}, "score": None, "reason": "stub"},
            {"idea": "stub-two", "delta": {"lora.rank": 16}, "score": 1.25, "reason": "stub"},
        ]
        return cards[:k], ["stub-drop"]


class StubOutOfAxes:
    name = "stub-oob"
    version = "1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec, results, launches, current, symptoms, *, k):
        return [{"idea": "bad-idea", "delta": {"parallel.tp": 2}, "score": None, "reason": "stub"}], []


class StubBoom:
    name = "stub-boom"
    version = "1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec, results, launches, current, symptoms, *, k):
        raise RuntimeError("model exploded")


class StubWobbly:
    """Non-deterministic stub: every propose() call renders different card bytes."""

    name = "stub-wobbly"
    version = "1"
    calls = 0

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def propose(self, spec, results, launches, current, symptoms, *, k):
        StubWobbly.calls += 1
        return [
            {"idea": "wobbly", "delta": {"optim.lr": 5e-5}, "score": float(StubWobbly.calls), "reason": "stub"}
        ], []


def registry_for(stub_factory, name):
    return {"catalog": CatalogProposer, name: stub_factory}


class TestProposerConfig:
    def test_defaults_and_overrides(self):
        assert proposer_config({}) == {"name": "catalog", "min_rows": 10, "require_model": False, "seed": 0}
        assert proposer_config({"proposer": {}}) == {"name": "catalog", "min_rows": 10, "require_model": False, "seed": 0}
        assert proposer_config({"proposer": {"name": "optuna", "min_rows": 4, "require_model": True, "seed": 7}}) == {
            "name": "optuna", "min_rows": 4, "require_model": True, "seed": 7,
        }


class TestCatalogOracle:
    def test_catalog_selector_is_byte_identical_to_propose(self):
        launches = [{"trial": "t0", "launch_spec": {"trial": "t0", "delta": {"lora.rank": 16}, "gpu_hours_est": 4}}]
        results = list(BASELINES)
        cards, drops, provenance = select(SPEC, results, launches, CURRENT, SYMPTOMS)
        expected = catalog_propose(SPEC, results, CURRENT, SYMPTOMS, k=3, launches=launches)
        assert canonical(cards) == canonical(expected)
        assert len(cards) >= 1 and cards[0]["idea"] != "baseline"
        assert drops == catalog_drops(SPEC, results, launches, CURRENT, SYMPTOMS)
        assert provenance["proposer"] == {"name": "catalog", "version": "1"}
        assert provenance["requested"] == "catalog" and provenance["fallback"] is None
        assert provenance["seed"] == 0 and provenance["k"] == DEFAULT_K
        assert provenance["rows_digest"] == rows_digest(SPEC, results, launches, CURRENT, SYMPTOMS)

    def test_catalog_proposer_byte_identity_stable_twice(self):
        one, drops_one = CatalogProposer(seed=7).propose(SPEC, BASELINES, [], CURRENT, SYMPTOMS, k=3)
        two, drops_two = CatalogProposer().propose(SPEC, BASELINES, [], CURRENT, SYMPTOMS, k=3)
        expected = catalog_propose(SPEC, BASELINES, CURRENT, SYMPTOMS, k=3, launches=[])
        assert canonical(one) == canonical(expected) == canonical(two)
        assert drops_one == drops_two == catalog_drops(SPEC, BASELINES, [], CURRENT, SYMPTOMS)
        assert rows_digest(SPEC, BASELINES, [], CURRENT, SYMPTOMS) == rows_digest(SPEC, BASELINES, [], CURRENT, SYMPTOMS)


class TestRowsDigest:
    def test_stable_across_key_order_and_sensitive_to_bytes(self):
        base = rows_digest(SPEC, BASELINES, [], CURRENT, SYMPTOMS)
        assert base.startswith("sha256:") and len(base) == 7 + 64
        reordered = {"train.method": "lora", "lora.rank": 24, "optim.warmup_ratio": 0.02, "optim.lr": 1e-3}
        flipped_row = {
            "metrics": {METRIC: {"se": 0.001, "value": 0.5}},
            "limited": False,
            "status": "ok",
            "seed": 1,
            "role": "baseline",
            "trial": "baseline",
        }
        assert rows_digest(SPEC, [flipped_row, *BASELINES[1:]], [], reordered, ["loss diverging"]) == base
        changed_rows = [dict(BASELINES[0], metrics={METRIC: {"value": 0.6, "se": 0.001}}), *BASELINES[1:]]
        assert rows_digest(SPEC, changed_rows, [], CURRENT, SYMPTOMS) != base
        tight = copy.deepcopy(SPEC)
        tight["axes"][0]["min"] = 1e-5
        assert rows_digest(tight, BASELINES, [], CURRENT, SYMPTOMS) != base


class TestTrialParams:
    def test_result_delta_wins_launch_falls_back_and_junk_drops(self):
        results = [
            "junk-row",
            {"trial": "t1", "delta": {"optim.warmup_ratio": 0.05}},   # earlier dict delta, superseded
            {"trial": "t1", "delta": {"optim.lr": 5e-4}},              # the latest dict delta wins
            {"trial": "t2"},
            {"trial": "t2", "delta": "not-a-dict"},                    # ignored: no result dict delta
        ]
        launches = [
            5,
            {"trial": "t1", "launch_spec": {"trial": "t1", "delta": {"lora.rank": 56}}},   # loses to the result
            {"trial": "t3", "delta": {"lora.rank": 4}},                                    # earlier payload
            {"trial": "t3", "delta": {"lora.rank": 12}},                                   # latest payload wins
            {"trial": "t2", "delta": "junk", "launch_spec": {"delta": {
                "optim.lr": 2e-4, "train.seq_len": 4096, "optim.warmup_ratio": 99.0,
            }}},
            {"trial": "t4", "launch_spec": {"trial": "t4", "delta": {"lora.rank": 32}}},
            {"trial": "t5", "delta": 0, "launch_spec": {"delta": [], "trial_spec": {"delta": {"train.method": "lora"}}}},
            {"launch_spec": {"trial": "t6", "delta": {
                "optim.lr": 8e-4, "optim.warmup_ratio": 5.0, "train.seq_len": 4096, "optim.spurious": 3,
            }}},
            {"trial": "t7", "launch_spec": "junk"},                 # no usable delta: no entry
        ]
        assert trial_params(SPEC, results, launches) == {
            "t1": {"optim.lr": 5e-4},
            "t2": {"optim.lr": 2e-4},
            "t3": {"lora.rank": 12},
            "t4": {"lora.rank": 32},
            "t5": {"train.method": "lora"},
            "t6": {"optim.lr": 8e-4},
        }


class TestModelRows:
    def test_measured_means_only_real_trials(self):
        results = [
            *BASELINES,
            trial_row("b1", 0.4, role="baseline", delta={"optim.lr": 1e-4}),       # role baseline: never a row
            trial_row("baseline", 0.9, role="candidate", delta={"optim.lr": 2e-4}),  # baseline trial name: excluded
            trial_row("t1", 0.4, seed=1, delta={"optim.lr": 2e-4}),
            trial_row("t1", 0.6, seed=2, role="confirm", delta={"optim.lr": 2e-4}),  # mean over 2 measured seeds
            trial_row("t1", 0.9, seed=3, status="crash", delta={"optim.lr": 2e-4}),
            trial_row("t1", 0.9, seed=4, limited=True, delta={"optim.lr": 2e-4}),
            trial_row("t1", None, seed=5, delta={"optim.lr": 2e-4}),                # unmeasured: never 0
            trial_row("a2", 0.3, delta={"lora.rank": 8}),
            trial_row("t-naked", 0.3),                                              # measured, no params anywhere
            trial_row("z-naked", 0.4),                                              # ditto: drops sorted/deduped
        ]
        rows, drops = model_rows(SPEC, results, [])
        assert [r["trial"] for r in rows] == ["a2", "t1"]
        assert rows[0] == {"trial": "a2", "params": {"lora.rank": 8}, "value": 0.3}
        assert rows[1]["params"] == {"optim.lr": 2e-4}
        assert rows[1]["value"] == pytest.approx(0.5)
        assert drops == ["model_row_no_params:t-naked", "model_row_no_params:z-naked"]


class TestEligible:
    def test_truth_table(self):
        rows9 = [{"trial": f"t{i}"} for i in range(9)]
        rows10 = rows9 + [{"trial": "t9"}]
        good = registry_for(StubGood, "optuna")
        assert eligible("catalog", SPEC, []) == (True, None)
        spec_optuna = spec_with("optuna")
        assert eligible("optuna", spec_optuna, rows9, good) == (False, "below_min_rows:9/10")
        missing = registry_for(_raise_import_error, "optuna")
        assert eligible("optuna", spec_optuna, rows10, missing) == (False, "extra_missing:optuna")
        assert eligible("optuna", spec_optuna, [], missing) == (False, "below_min_rows:0/10")
        no_axes = spec_with("optuna")
        no_axes.pop("axes")
        assert eligible("optuna", no_axes, rows10, good) == (False, "no_axes")
        assert eligible("optuna", spec_optuna, rows10, good) == (True, None)
        assert eligible("vizier", spec_optuna, []) == (False, "unknown_proposer:vizier")
        assert eligible("bogus", spec_optuna, rows10, registry_for(StubGood, "optuna")) == (False, "unknown_proposer:bogus")
        strict = spec_with("optuna", min_rows=2)
        assert eligible("optuna", strict, rows9, good) == (True, None)   # min_rows is honoured


def _raise_import_error(seed: int):
    raise ImportError("optuna is an optional extra")


class TestGetProposer:
    def test_registry_gap_and_missing_extra(self):
        got = get_proposer("catalog", seed=5)
        assert isinstance(got, CatalogProposer) and got.seed == 5
        assert get_proposer("bogus") is None
        assert get_proposer("optuna", registry=registry_for(_raise_import_error, "optuna")) is None
        assert isinstance(get_proposer("optuna", registry=registry_for(StubGood, "optuna")), StubGood)


class TestCardInAxes:
    def test_bounds_check(self):
        assert _card_in_axes(SPEC, {"idea": "x", "delta": {"optim.lr": 1e-5, "lora.rank": 8}}) is True
        assert _card_in_axes(SPEC, {"idea": "x", "delta": {}}) is False
        assert _card_in_axes(SPEC, {"idea": "x", "delta": {"parallel.tp": 2}}) is False
        assert _card_in_axes(SPEC, {"idea": "x", "delta": {"optim.lr": 5.0}}) is False
        assert _card_in_axes(SPEC, {"idea": "x", "delta": "junk"}) is False
        assert _card_in_axes(SPEC, "not-a-card") is False


class TestSelectFallbacks:
    def test_out_of_axes_card_only_falls_back(self):
        results = model_ledger()
        spec = spec_with("stub-oob")
        cards, drops, provenance = select(
            spec, results, [], CURRENT, SYMPTOMS, registry=registry_for(StubOutOfAxes, "stub-oob")
        )
        assert "model_out_of_axes:bad-idea" in drops
        assert "proposer_fallback_catalog:no_model_cards" in drops
        assert canonical(cards) == canonical(catalog_propose(spec, results, CURRENT, SYMPTOMS, k=3, launches=[]))
        assert provenance["fallback"] == "no_model_cards"
        assert provenance["proposer"] == {"name": "catalog", "version": "1"}
        assert provenance["rows_digest"] == rows_digest(spec, results, [], CURRENT, SYMPTOMS)

    def test_model_error_falls_back(self):
        results = model_ledger()
        spec = spec_with("stub-boom")
        cards, drops, provenance = select(spec, results, [], CURRENT, SYMPTOMS, registry=registry_for(StubBoom, "stub-boom"))
        assert "proposer_fallback_catalog:model_error" in drops
        assert canonical(cards) == canonical(catalog_propose(spec, results, CURRENT, SYMPTOMS, k=3, launches=[]))
        assert provenance["fallback"] == "model_error"

    def test_non_deterministic_stub_is_unverified(self):
        StubWobbly.calls = 0
        results = model_ledger()
        spec = spec_with("stub-wobbly")
        cards, drops, provenance = select(
            spec, results, [], CURRENT, SYMPTOMS, registry=registry_for(StubWobbly, "stub-wobbly")
        )
        assert StubWobbly.calls >= 2
        assert "proposer_fallback_catalog:unverified" in drops
        assert canonical(cards) == canonical(catalog_propose(spec, results, CURRENT, SYMPTOMS, k=3, launches=[]))
        assert provenance["fallback"] == "unverified"

    def test_below_min_rows_falls_back(self):
        results = model_ledger(3)
        spec = spec_with("stub-good")
        cards, drops, provenance = select(spec, results, [], CURRENT, SYMPTOMS, registry=registry_for(StubGood, "stub-good"))
        assert "proposer_fallback_catalog:below_min_rows:3/10" in drops
        assert canonical(cards) == canonical(catalog_propose(spec, results, CURRENT, SYMPTOMS, k=3, launches=[]))
        assert provenance["fallback"] == "below_min_rows:3/10"
        assert provenance["proposer"] == {"name": "catalog", "version": "1"}

    def test_good_stub_speaks_and_score_none_stays_none(self):
        results = model_ledger() + [trial_row("t-naked", 0.3)]    # measured but unrecoverable params -> named row drop
        spec = spec_with("stub-good")
        cards, drops, provenance = select(spec, results, [], CURRENT, SYMPTOMS, registry=registry_for(StubGood, "stub-good"))
        assert [c["idea"] for c in cards] == ["stub-idea", "stub-two"]
        assert cards[0]["score"] is None                       # never faked as 0
        assert cards[1]["score"] == 1.25
        assert provenance["proposer"] == {"name": "stub-good", "version": "7"}
        assert provenance["requested"] == "stub-good" and provenance["fallback"] is None
        assert provenance["seed"] == 0 and provenance["k"] == DEFAULT_K
        assert "stub-drop" in drops and "model_row_no_params:t-naked" in drops
        assert replay(spec, results, [], CURRENT, SYMPTOMS, provenance, cards, registry_for(StubGood, "stub-good")) is True
        tampered = [dict(cards[0], score=0.0), cards[1]]
        assert replay(spec, results, [], CURRENT, SYMPTOMS, provenance, tampered, registry_for(StubGood, "stub-good")) is False


class TestReplay:
    def test_catalog_replay_true_with_exactly_the_recorded_cards(self):
        cards, _, provenance = select(SPEC, BASELINES, [], CURRENT, SYMPTOMS)
        assert replay(SPEC, BASELINES, [], CURRENT, SYMPTOMS, provenance, cards) is True

    def test_mutated_card_fails(self):
        cards, _, provenance = select(SPEC, BASELINES, [], CURRENT, SYMPTOMS)
        mutated = [dict(cards[0], idea="tampered"), *cards[1:]]
        assert replay(SPEC, BASELINES, [], CURRENT, SYMPTOMS, provenance, mutated) is False
        short = cards[:-1]
        assert replay(SPEC, BASELINES, [], CURRENT, SYMPTOMS, provenance, short) is False

    def test_changed_results_fail_the_digest_pin(self):
        cards, _, provenance = select(SPEC, BASELINES, [], CURRENT, SYMPTOMS)
        changed = [dict(BASELINES[0], metrics={METRIC: {"value": 0.9, "se": 0.001}}), *BASELINES[1:]]
        assert replay(SPEC, changed, [], CURRENT, SYMPTOMS, provenance, cards) is False
        assert replay(SPEC, BASELINES, [], CURRENT, ["other hints"], provenance, cards) is False


class TestOptionalExtra:
    def test_optuna_extra_is_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "optuna", None)      # `import optuna` fails in the bench env
        assert get_proposer("optuna") is None
        rows = [{"trial": f"t{i}", "params": {"optim.lr": 1e-4}, "value": 0.1} for i in range(DEFAULT_MIN_ROWS)]
        assert eligible("optuna", SPEC, rows) == (False, "extra_missing:optuna")


class TestUnmeasurableDigest:
    def test_nan_hint_gives_none_digest_not_a_crash(self):
        from foundationskills.skills.auto_research.proposers import select

        cards, drops, provenance = select(copy.deepcopy(SPEC), list(BASELINES), [], {"optim.lr": float("nan")}, [])
        assert provenance["rows_digest"] is None
        assert drops == catalog_drops(SPEC, list(BASELINES), [], {"optim.lr": float("nan")}, [])


class TestReviewRegressions:
    """Regressions for the M3 review findings (each fails on the pre-fix code)."""

    def test_empty_later_result_delta_does_not_shadow_an_earlier_one(self):
        results = [trial_row("t1", 0.3, delta={"optim.lr": 1e-4}), trial_row("t1", 0.31, seed=2, delta={})]
        assert trial_params(SPEC, results, []) == {"t1": {"optim.lr": 1e-4}}

    def test_result_delta_without_axis_params_falls_back_to_the_launch(self):
        results = [trial_row("t1", 0.3, delta={"not.an.axis": 1})]
        launches = [{"trial": "t1", "launch_spec": {"delta": {"lora.rank": 8}}}]
        assert trial_params(SPEC, results, launches) == {"t1": {"lora.rank": 8}}

    def test_payload_level_trial_spec_delta_is_recovered(self):
        launches = [{"trial": "t1", "trial_spec": {"delta": {"optim.lr": 2e-4}}}]
        assert trial_params(SPEC, [], launches) == {"t1": {"optim.lr": 2e-4}}

    def test_digest_pins_the_seeds_block(self):
        other = copy.deepcopy(SPEC)
        other["seeds"]["baseline_repeats"] = 5
        assert rows_digest(SPEC, BASELINES, [], CURRENT, SYMPTOMS) != rows_digest(other, BASELINES, [], CURRENT, SYMPTOMS)

    def test_broken_extra_is_model_error_not_a_crash(self):
        def broken(seed):
            raise RuntimeError("extra is installed but broken")

        rows = [{"trial": f"t{i}", "params": {"optim.lr": 1e-4}, "value": 0.1} for i in range(DEFAULT_MIN_ROWS)]
        assert eligible("optuna", spec_with("optuna"), rows, {"catalog": CatalogProposer, "optuna": broken}) == (
            False, "model_error")
