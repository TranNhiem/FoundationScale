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
