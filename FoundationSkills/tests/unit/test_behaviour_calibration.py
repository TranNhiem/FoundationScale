"""C3.2/C3.3: auto-calibration on Tier A items and the judge-only gate, at the real 50 / 0.80 floor."""

from foundationskills.agent.behaviour_eval import _gate_judge_only, judge_calibration


def _call(det, judge_met, measured=True):
    return {"measured": measured, "det": list(det), "judge_met": list(judge_met), "error": None}


def test_calibrated_when_enough_decidable_items_agree():
    calls = [_call([True, None, True], [True, False, True]) for _ in range(25)]  # 50 decidable pairs, all agree
    cal = judge_calibration(calls)
    assert cal["items"] == 50 and cal["agreement"] == 1.0 and cal["calibrated"]


def test_not_calibrated_below_the_item_floor_or_agreement():
    assert not judge_calibration([_call([True], [True]) for _ in range(49)])["calibrated"]
    half = [_call([True, False], [True, True]) for _ in range(30)]  # 60 items, 50% agreement
    assert not judge_calibration(half)["calibrated"]


def test_judge_only_calls_become_unmeasured_without_calibration():
    judge_only, mixed = _call([None, None], [True, True]), _call([True, None], [True, True])
    _gate_judge_only([judge_only, mixed], judge_calibration([judge_only, mixed]))
    assert judge_only["measured"] is False and "not calibrated" in judge_only["error"]
    assert mixed["measured"] is True  # a case with a decidable item still counts


def test_judge_only_calls_count_under_a_calibrated_judge():
    calls = [_call([True], [True]) for _ in range(50)] + [_call([None], [True])]
    _gate_judge_only(calls, judge_calibration(calls))
    assert calls[-1]["measured"] is True


def test_transferred_calibration_requires_same_judge_model_and_a_pass():
    from types import SimpleNamespace

    from foundationskills.agent.behaviour_eval import transferred_calibration

    good = {"judge_model": "mimo-v2.6-pro", "cases": "oracle", "created": "t",
            "judge_calibration": {"calibrated": True, "items": 96, "agreement": 0.9}}
    judge = SimpleNamespace(model="mimo-v2.6-pro")
    assert transferred_calibration(good, judge)["source"].startswith("transferred:oracle")
    assert transferred_calibration({**good, "judge_model": "other"}, judge) is None
    assert transferred_calibration({**good, "judge_calibration": {"calibrated": False}}, judge) is None


def test_pooled_calibration_reaches_the_floor_across_small_reports():
    from types import SimpleNamespace

    from foundationskills.agent.behaviour_eval import pooled_calibration

    judge = SimpleNamespace(model="mimo-v2.6-pro")
    small = [{"judge_model": "mimo-v2.6-pro", "judge_calibration": {"items": 28, "agreement": 0.9}} for _ in range(2)]
    pooled = pooled_calibration(small, judge)
    assert pooled["calibrated"] and pooled["items"] == 56 and abs(pooled["agreement"] - 0.9) < 1e-9
    assert pooled_calibration(small[:1], judge) is None  # 28 items: below the floor
    assert pooled_calibration([{**small[0], "judge_model": "other"}] * 3, judge) is None
