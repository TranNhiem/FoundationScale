"""Proposer tests: baseline card, multiplicative deltas, pruned ideas and deterministic ordering."""
from __future__ import annotations

from foundationskills.skills.auto_research.propose import evaluate, load_catalog, propose, resolve_delta

SPEC = {
    "id": "c1",
    "objective": {"metric": "val_accuracy", "direction": "max", "benchmarks": []},
    "eval_policy": {"fingerprint": "sha256:" + "a1" * 32, "metrics": ["val_accuracy"]},
    "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": [1, 2, 3]},
    "axes": [
        {"key": "optim.lr", "type": "log_float", "min": 1e-6, "max": 1e-3},
        {"key": "optim.warmup_ratio", "type": "float", "min": 0.0, "max": 0.1},
        {"key": "lora.rank", "type": "int", "min": 4, "max": 64},
        {"key": "train.method", "type": "categorical", "values": ["full", "lora"]},
    ],
}
CURRENT = {"optim.lr": 1e-3, "optim.warmup_ratio": 0.02, "lora.rank": 24, "train.method": "lora"}
BASELINES = [
    {"trial": "baseline", "role": "baseline", "seed": i, "status": "ok", "limited": False,
     "metrics": {"val_accuracy": {"value": 0.5, "se": 0.001}}} for i in (1, 2, 3)
]


def reasons(cards) -> dict:
    return {card["idea"]: card["reason"] for card in cards}


class TestCatalog:
    def test_catalog_shape(self):
        catalog = load_catalog()
        assert 8 <= len(catalog) <= 12
        for idea in catalog:
            assert {"id", "domain", "applies_when", "delta", "prior_gain", "complexity", "watch"} <= set(idea)
            assert idea["prior_gain"] in {"s", "m", "l"} and idea["complexity"] in {1, 2, 3}
            assert isinstance(idea["applies_when"], list) and isinstance(idea["watch"], list)


class TestResolveDelta:
    def test_multiplicative_and_int_rounding(self):
        delta, drop = resolve_delta({"optim.lr": "x0.5"}, CURRENT, SPEC)
        assert drop is None and delta == {"optim.lr": 0.0005}
        delta, drop = resolve_delta({"lora.rank": "x2"}, CURRENT, SPEC)
        assert drop is None and delta == {"lora.rank": 48} and isinstance(delta["lora.rank"], int)
        delta, drop = resolve_delta({"lora.rank": "x0.5"}, CURRENT, SPEC)
        assert delta == {"lora.rank": 12}

    def test_literal_categorical(self):
        delta, drop = resolve_delta({"train.method": "full"}, CURRENT, SPEC)
        assert drop is None and delta == {"train.method": "full"}

    def test_drop_reasons(self):
        assert resolve_delta({"optim.lr": "x2"}, CURRENT, SPEC)[1] == "out_of_axes"
        assert resolve_delta({"optim.spurious": "x2"}, CURRENT, SPEC)[1] == "not_in_axes"
        assert resolve_delta({"train.seq_len": "x2"}, CURRENT, SPEC)[1] == "not_in_axes"
        assert resolve_delta({"optim.lr": "x2"}, {}, SPEC)[1] == "missing_current"


class TestPropose:
    def test_baseline_card_until_repeats_exist(self):
        cards = propose(SPEC, BASELINES[:1], CURRENT, ["loss diverging"])
        assert len(cards) == 1 and cards[0]["idea"] == "baseline" and cards[0]["delta"] == {}
        assert "1/3" in cards[0]["reason"]

    def test_scores_and_order(self):
        cards = propose(SPEC, BASELINES, CURRENT, ["loss diverging"], k=5)  # the full live list, in order
        assert [c["idea"] for c in cards] == ["lr_down", "train_full", "warmup_up", "lora_rank_up", "train_lora"]
        assert all(cards[i]["score"] >= cards[i + 1]["score"] for i in range(len(cards) - 1))
        assert cards[0]["score"] == 3.9 and "matched 1 symptom(s)" in cards[0]["reason"]

    def test_top_k(self):
        assert len(propose(SPEC, BASELINES, CURRENT, ["loss diverging"], k=2)) == 2

    def test_out_of_range_and_off_axes_are_dropped(self):
        kept, dropped = evaluate(SPEC, BASELINES, CURRENT, [])
        dropped_reasons = reasons(dropped)
        assert dropped_reasons["lr_up"] == "out_of_axes"
        assert dropped_reasons["seq_len_up"] == "not_in_axes"
        assert "lr_up" not in [c["idea"] for c in kept]

    def test_duplicate_delta_is_pruned(self):
        replay = [*BASELINES, {"trial": "t1", "role": "candidate", "seed": 1, "status": "ok", "limited": False,
                               "delta": {"optim.lr": 5e-4}, "metrics": {"val_accuracy": {"value": 0.5, "se": 0.001}}}]
        kept, dropped = evaluate(SPEC, replay, CURRENT, [])
        assert reasons(dropped)["lr_down"] == "duplicate_delta"
        assert "lr_down" not in [c["idea"] for c in kept]

    def test_symptom_match_raises_the_score(self):
        cold, _ = evaluate(SPEC, BASELINES, CURRENT, [])
        warm, _ = evaluate(SPEC, BASELINES, CURRENT, ["unstable loss"])
        cold_score = next(c["score"] for c in cold if c["idea"] == "lr_down")
        warm_score = next(c["score"] for c in warm if c["idea"] == "lr_down")
        assert warm_score == cold_score + 2.0
