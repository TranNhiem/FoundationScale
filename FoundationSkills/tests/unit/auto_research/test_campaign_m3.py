"""AR-IN-008 M3 proposer config checks and the ``proposal`` ledger op."""
from __future__ import annotations

from foundationskills.skills.auto_research.campaign import check_spec
from foundationskills.skills.auto_research.ledger import Ledger

try:  # reuse the M2 spec builder when that test module is importable
    from tests.unit.auto_research.test_campaign_m2 import _spec
except Exception:  # flat collection: copy the minimal valid spec dict

    def _spec(**overrides):
        spec = {
            "objective": {"metric": "val/loss", "direction": "min"},
            "eval_policy": {"metrics": ["val/loss"], "fingerprint": "sha256:" + "a" * 64},
            "base": {"model": "m-core", "fingerprint": "sha256:" + "b" * 64},
            "budget": {"gpu_hours_total": 120.0, "max_runs": 6, "per_run_timeout_h": 24.0, "reserve_frac": 0.3},
            "cluster": {"time": "10-00:00:00", "exclude": ["r01dgx02"], "max_nodes": 2, "gpus_per_node": 8},
            "seeds": {
                "baseline_repeats": 3,
                "confirm_repeats": 3,
                "seed_list": [101, 102, 103],
                "screening_repeats": 1,
            },
            "axes": [],
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(spec.get(key), dict):
                spec[key] = {**spec[key], **value}
            else:
                spec[key] = value
        return spec


def _hits(problems, needle):
    return [msg for rule, msg in problems if rule == "AR-IN-008" and needle in msg]


def test_valid_spec_carries_no_ar_in_008():
    assert check_spec(_spec()) == []
    assert not any(rule == "AR-IN-008" for rule, _ in check_spec(_spec()))


def test_good_proposer_blocks_are_clean():
    assert check_spec(_spec(proposer={"name": "catalog"})) == []
    assert check_spec(_spec(proposer={"name": "optuna", "min_rows": 10, "require_model": True, "seed": 3})) == []
    assert check_spec(_spec(proposer={})) == []
    assert check_spec(_spec(proposer=None)) == []


def test_proposer_name_must_come_from_producer_names():
    problems = check_spec(_spec(proposer={"name": "vizier"}))
    msgs = _hits(problems, "proposer.name")
    assert len(msgs) == 1 and "vizier" in msgs[0] and "catalog" in msgs[0]
    assert [rule for rule, _ in problems] == ["AR-IN-008"]


def test_proposer_min_rows_must_be_an_int_ge_one():
    for bad in (0, True, 1.5):
        problems = check_spec(_spec(proposer={"min_rows": bad}))
        assert _hits(problems, "proposer.min_rows"), bad
        assert [rule for rule, _ in problems] == ["AR-IN-008"], bad


def test_proposer_require_model_must_be_a_bool():
    for bad in ("true", 1):
        problems = check_spec(_spec(proposer={"require_model": bad}))
        assert _hits(problems, "proposer.require_model"), bad
        assert [rule for rule, _ in problems] == ["AR-IN-008"], bad


def test_proposer_seed_must_be_an_int_any_sign():
    for bad in (True, "1"):
        problems = check_spec(_spec(proposer={"seed": bad}))
        assert _hits(problems, "proposer.seed"), bad
        assert [rule for rule, _ in problems] == ["AR-IN-008"], bad
    assert check_spec(_spec(proposer={"seed": -7})) == []  # negative seeds are real ints


def test_proposer_block_must_be_a_mapping_and_stop():
    problems = check_spec(_spec(proposer="optuna"))
    assert len(problems) == 1  # the shape complaint owns this input and stops
    rule, msg = problems[0]
    assert rule == "AR-IN-008" and "proposer must be a mapping" in msg and "str" in msg


def test_unknown_proposer_keys_are_rejected():
    problems = check_spec(_spec(proposer={"name": "catalog", "explore": 2}))
    msgs = _hits(problems, "proposer has unknown key(s)")
    assert len(msgs) == 1 and "explore" in msgs[0]
    assert check_spec(_spec(proposer={"name": "catalog", "min_rows": 2, "require_model": False, "seed": -3})) == []


def test_bad_name_and_min_rows_combo_is_exactly_two_findings():
    problems = check_spec(_spec(proposer={"name": "vizier", "min_rows": 0}))
    assert len(problems) == 2
    assert [rule for rule, _ in problems] == ["AR-IN-008", "AR-IN-008"]
    assert _hits(problems, "proposer.name") and _hits(problems, "proposer.min_rows")


def test_findings_keep_the_rule_message_tuple_shape():
    problems = check_spec(_spec(proposer={"min_rows": 0}))
    assert all(isinstance(rule, str) and isinstance(msg, str) and rule == "AR-IN-008" for rule, msg in problems)


def test_ledger_proposal_round_trip(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.append("proposal", "c1", "t", {"k": 3})
    assert ledger.proposals("c1") == [{"k": 3}]
    assert ledger.proposals("other") == []
    assert ledger.verify() == []
