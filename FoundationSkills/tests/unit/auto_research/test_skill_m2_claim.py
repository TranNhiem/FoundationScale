"""M2 claim wiring: the explicit ``claim`` action and the claim chain re-derived at ``close``.

Every test stages an append-only ledger with the pure M1 helpers (``test_skill_m1``) and runs the skill
through its injected connectors (``_Fakes``): no Slurm call, no socket and no fork may escape a unit
test. Claims are EXPLICIT (amendment A6): the ``claim`` action appends exactly one ``claim`` op, and
``close`` only re-derives the chain and reports ``champion``/``claims``/``unclaimed_gains``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from foundationskills.core.status import Status
from foundationskills.skills.auto_research.campaign import campaign_hash
from foundationskills.skills.auto_research.claims import claim_body, claim_id
from foundationskills.skills.auto_research.ledger import Ledger
from foundationskills.skills.auto_research.skill import _campaign, _job_payload, _result, _spec, _val

from .test_skill_m1 import _Fakes, _approved, _ctx, _findings, _request, _stage

SEEDS = (101, 102, 103)
SEED_PLAN = {"baseline_repeats": 3, "screening_repeats": 1, "confirm_repeats": 3, "seed_list": list(SEEDS)}
BASE_VALUE = (0.5, 0.502, 0.501)  # the calibrated baseline repeats (noise floor source)
ONE_VALUE = (0.7, 0.702, 0.701)   # t1 beats the baseline: a claimable gain
TWO_VALUE = (0.9, 0.902, 0.901)   # t2 beats the claimed champion: a claimable gain over t1
SAME_VALUE = (0.7, 0.701, 0.699)  # dead heat against the claimed champion (beats the baseline)
FLAT_VALUE = (0.5, 0.501, 0.502)  # dead heat against the baseline: nothing to claim


def _claim_spec():
    """Campaign spec with the pinned seed plan (confirm seeds 101..103 pair with the baseline rows)."""
    return _spec(seeds=dict(SEED_PLAN))


def _rows(trial, role, values, seeds=SEEDS):
    return [_result(trial, role, seed, _val(value)) for seed, value in zip(seeds, values)]


def _runs(campaign, trial, seeds):
    """One ledgered run per recorded seed (M2 provenance: seed + derived phase)."""
    events = []
    for seed in seeds:
        job = _job_payload(str(700 + int(seed)), trial, kind="eval_only")
        job.update({"seed": seed, "phase": "confirm", "gpu_hours_est": 1.0})
        events.append(("job_submitted", campaign, trial, dict(job)))
    return events


def _body_of(entry):
    inner = entry.get("body") if isinstance(entry.get("body"), dict) else entry
    return dict(inner or {})


def _trial_of(entry):
    return str(entry.get("trial") or _body_of(entry).get("trial") or "")


def _cid_of(entry):
    return str(entry.get("claim_id") or _body_of(entry).get("claim_id") or "")


def _claim_payload(spec, trial, seeds=SEEDS, prev="baseline"):
    """A hand-built ``claim`` ledger payload (one byte change would move its id)."""
    body = claim_body(
        trial, "baseline", prev, list(seeds),
        {"verdict": "accepted_gain", "mean_delta": 0.4, "tau": 0.01, "n_pairs": len(seeds)},
        str(dict(spec.get("eval_policy") or {}).get("fingerprint") or ""),
    )
    return {**body, "claim_id": claim_id(body)}


def _stage_campaign(tmp_path, spec, trials=None, claims=()):
    """campaign_approved + baseline rows + per-trial confirm rows (and runs) + explicit claim ops."""
    campaign = _campaign(spec)
    events = [("trial_result", campaign, "baseline", dict(row)) for row in _rows("baseline", "baseline", BASE_VALUE)]
    for trial, values in dict(trials or {}).items():
        rows = _rows(trial, "confirm", values)
        events.extend(_runs(campaign, trial, [row.get("seed") for row in rows]))
        events.extend(("trial_result", campaign, trial, dict(row)) for row in rows)
    events.extend(("claim", campaign, _trial_of(claim), dict(claim)) for claim in claims)
    _stage(tmp_path, _approved(spec, extra=events))
    return campaign


def _claim_request(tmp_path, spec, trial):
    request = _request(tmp_path, action="claim", campaign_spec=spec, campaign_confirm=campaign_hash(spec))
    request["trial"] = trial
    return request


def _close_request(tmp_path, spec):
    request = _request(tmp_path, action="close", campaign_spec=spec, campaign_confirm=campaign_hash(spec))
    request["stop_reason"] = "budget exhausted"
    return request


def _claim_ops(tmp_path):
    return [entry for entry in Ledger(tmp_path / "ledger").entries() if entry.get("op") == "claim"]


def _drops(result):
    """The close/report drops wherever they are carried (report key or the budget roll-up)."""
    payload = dict(result.payload or {})
    budgets = dict(payload.get("budgets") or {})
    return [str(drop) for drop in [*list(payload.get("drops") or []), *list(budgets.get("drops") or [])]]


def _evidence(result):
    parts = [str(result.findings)]
    for finding in result.findings:
        for attr in ("evidence", "data", "payload", "detail", "details", "context"):
            value = getattr(finding, attr, None)
            if isinstance(value, dict):
                parts.append(str(value))
    return "".join(parts)


def _fill(tmp_path, value):
    if isinstance(value, str):
        return value.replace("{tmp}", str(tmp_path))
    if isinstance(value, dict):
        return {_fill(tmp_path, key): _fill(tmp_path, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(tmp_path, item) for item in value]
    return value


def _materialize_fixture(fixture, tmp_path):
    """Write the fixture's ``files`` and fill ``{tmp}``/``{confirm}`` (the MUST_FIRE harness shape)."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for rel, data in dict(fixture.get("files") or {}).items():
        path = tmp_path / str(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else str(data).encode())
    request = fixture.get("request")
    if not isinstance(request, dict):
        request = {
            key: value for key, value in fixture.items()
            if key not in ("files", "rules", "rule", "refusal", "request")
        }
    request = _fill(tmp_path, dict(request))
    if str(request.get("campaign_confirm") or "") == "{confirm}":
        request["campaign_confirm"] = campaign_hash(request["campaign_spec"])
    return request


class TestClaimAction:
    """AR-RS-007: one explicit ``claim`` op per earned gain over a complete, measured confirm set."""

    def test_claim_appends_exactly_one_claim_op_at_the_root(self, tmp_path):
        spec = _claim_spec()
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE})

        result = _Fakes().skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path))

        assert result.status is Status.PASS
        ops = _claim_ops(tmp_path)
        assert len(ops) == 1
        assert result.payload["prev"] == "baseline"
        assert result.payload["claim_id"].startswith("sha256:")
        ledger = Ledger(tmp_path / "ledger")
        assert ledger.payload(ops[0]).get("claim_id") == result.payload["claim_id"]
        assert ledger.verify() == []

    def test_second_claim_for_the_same_trial_is_refused(self, tmp_path):
        spec = _claim_spec()
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE})
        skill = _Fakes().skill
        assert skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path)).status is Status.PASS

        result = skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-RS-007", "claim_exists:t1") in _findings(result)
        assert len(_claim_ops(tmp_path)) == 1

    def test_incomplete_confirm_set_is_refused_with_its_named_drops(self, tmp_path):
        spec = _claim_spec()
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE[:2]})  # only seeds 101 and 102 ran

        result = _Fakes().skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-RS-007", "claim_set_incomplete:t1") in _findings(result)
        assert "seed_missing:t1:103" in _evidence(result)
        assert _claim_ops(tmp_path) == []

    def test_claim_without_an_accepted_gain_is_refused(self, tmp_path):
        spec = _claim_spec()
        _stage_campaign(tmp_path, spec, {"t1": FLAT_VALUE})

        result = _Fakes().skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-RS-007", "claim_no_gain:t1") in _findings(result)
        assert _claim_ops(tmp_path) == []

    def test_second_trial_chains_on_the_first_claim(self, tmp_path):
        spec = _claim_spec()
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE, "t2": TWO_VALUE})

        head = _Fakes().skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path))
        assert head.status is Status.PASS

        second = _Fakes().skill.execute(_claim_request(tmp_path, spec, "t2"), _ctx(tmp_path))

        assert second.status is Status.PASS
        assert second.payload["prev"] == head.payload["claim_id"]
        assert second.payload["claim_id"] != head.payload["claim_id"]
        assert len(_claim_ops(tmp_path)) == 2

    def test_claim_is_decided_against_the_champion_rows(self, tmp_path):
        """t2 beats the baseline but is a dead heat against the claimed champion's rows: nothing to claim."""
        spec = _claim_spec()
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE, "t2": SAME_VALUE})
        skill = _Fakes().skill
        assert skill.execute(_claim_request(tmp_path, spec, "t1"), _ctx(tmp_path)).status is Status.PASS

        result = skill.execute(_claim_request(tmp_path, spec, "t2"), _ctx(tmp_path))

        assert result.status is Status.REFUSED
        assert ("AR-RS-007", "claim_no_gain:t2") in _findings(result)
        assert len(_claim_ops(tmp_path)) == 1


class TestCloseClaimChain:
    """A6: close re-derives the chain and reports it (close never writes claims)."""

    def test_close_reports_champion_claims_and_unclaimed_gains(self, tmp_path):
        spec = _claim_spec()
        claim = _claim_payload(spec, "t1")
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE, "t2": TWO_VALUE}, claims=[claim])

        result = _Fakes().skill.execute(_close_request(tmp_path, spec), _ctx(tmp_path))

        payload = result.payload
        assert _trial_of(payload["champion"]) == "t1"
        assert [_cid_of(entry) for entry in payload["claims"]] == [claim["claim_id"]]
        assert payload["unclaimed_gains"] == ["t2"]
        assert len(_claim_ops(tmp_path)) == 1  # close wrote no claim

    def test_close_keeps_the_champion_decided_against_what_it_was_earned_on(self, tmp_path):
        spec = _claim_spec()
        claim = _claim_payload(spec, "t1")
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE}, claims=[claim])

        payload = _Fakes().skill.execute(_close_request(tmp_path, spec), _ctx(tmp_path)).payload

        assert payload["decisions"]["t1"]["verdict"] == "accepted_gain"  # not t1 against its own rows
        assert payload["best"] == "t1" and payload["unclaimed_gains"] == []

    def test_close_downgrades_a_claim_with_a_lost_seed(self, tmp_path):
        spec = _claim_spec()
        claim = _claim_payload(spec, "t1")  # claims seeds 101/102 and the never-run 103
        _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE[:2]}, claims=[claim])

        result = _Fakes().skill.execute(_close_request(tmp_path, spec), _ctx(tmp_path))

        assert result.status is Status.UNMEASURED
        assert "claim_downgraded:%s" % claim["claim_id"] in _drops(result)
        assert any(rule_id == "AR-HO-004" for rule_id, _ in _findings(result))
        assert len(_claim_ops(tmp_path)) == 1


@pytest.mark.parametrize("bad", [{"seeds": 3}, {"stats": ["x"]}, {"claim_id": None}, {"seeds": []}])
def test_a_malformed_stored_claim_is_a_named_drop_not_a_crash_or_an_accepted_link(bad, tmp_path):
    spec = _claim_spec()
    claim = {**_claim_payload(spec, "t1"), **bad}
    _stage_campaign(tmp_path, spec, {"t1": ONE_VALUE}, claims=[claim])

    result = _Fakes().skill.execute(_close_request(tmp_path, spec), _ctx(tmp_path))

    assert result.status is Status.UNMEASURED
    assert "claim_malformed:0" in _drops(result)
    assert result.payload["claims"] == [] and result.payload["champion"]["claim_id"] == "baseline"


@pytest.mark.parametrize(
    "name, rule, message",
    [
        ("AR-RS-007", "AR-RS-007", "claim_set_incomplete:t1"),
        ("AR-IN-007", "AR-IN-007", ""),
    ],
)
def test_must_fire_fixture_fires(name, rule, message, tmp_path):
    fixture = _Fakes().skill.must_fire_fixtures()[name]
    request = _materialize_fixture(fixture, tmp_path / name)

    result = _Fakes().skill.execute(request, _ctx(tmp_path / name))

    assert result.status is Status.REFUSED
    assert any(rule_id == rule for rule_id, _ in _findings(result))
    if message:
        assert (rule, message) in _findings(result)
