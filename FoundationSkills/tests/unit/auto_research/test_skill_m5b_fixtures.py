from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from foundationskills.skills.auto_research.skill import (
    _llm_digest,
    _llm_fixture_record,
    _llm_must_fire_fixtures,
)

FIXTURES: dict[str, dict[str, Any]] = _llm_must_fire_fixtures()
RULES = ("AR-IN-010", "AR-PR-005", "AR-PR-003", "AR-PR-004", "AR-HO-009")
CLOSE_RULES = ("AR-PR-003", "AR-PR-004", "AR-HO-009")


def _objects(files: dict[str, str]) -> list[dict[str, Any]]:
    return [json.loads(text) for path, text in sorted(files.items()) if path.endswith(".json")]


def _llm_record(files: dict[str, str]) -> dict[str, Any]:
    records = [obj for obj in _objects(files) if isinstance(obj, dict) and "llm" in obj]
    assert len(records) == 1
    return records[0]


def test_llm_must_fire_rules_are_exactly_the_m5b_set() -> None:
    assert set(FIXTURES) == set(RULES)
    for fixture in FIXTURES.values():
        assert isinstance(fixture["request"], dict)


def test_llm_fixture_record_provenance_fields() -> None:
    record = _llm_fixture_record("[]", [])
    assert record["proposer"] == {
        "name": "llm", "version": "1", "package_version": "parse_spec/1;client/stdlib",
    }
    assert record["package_version"] == "parse_spec/1;client/stdlib"
    assert record["seed"] == 0
    assert record["rows_digest"] is None
    assert record["k"] == 3
    assert record["fallback"] is None
    assert record["replay_inputs"] == {"current": {}, "symptoms": [], "results_count": 0, "launches_count": 0}
    assert record["generation"] == "nondeterministic"


def test_llm_fixture_record_llm_block_shape() -> None:
    llm = _llm_fixture_record("[]", [{"idea": "card"}])["llm"]
    assert set(llm) == {"request", "response", "parse", "calls_used", "max_calls"}
    assert set(llm["request"]) == {
        "model", "pool_key", "endpoint_fingerprint", "sampling", "prompt_spec_version", "prompt_hash",
        "messages", "evidence_truncated",
    }
    assert set(llm["response"]) == {
        "status", "text", "response_hash", "excerpt", "latency_s", "tokens_in", "tokens_out",
    }
    assert set(llm["parse"]) == {"spec_version", "max_cards", "cards", "cards_total", "drops", "tainted"}
    assert llm["parse"]["spec_version"] == 1
    assert llm["parse"]["max_cards"] == 3
    assert llm["request"]["messages"] == [{"role": "system", "content": "fixture"}]
    assert llm["parse"]["cards"] == [{"idea": "card"}]
    assert llm["parse"]["cards_total"] == 1


def test_llm_fixture_record_hashes_verbatim_body() -> None:
    text = '[{"idea": "card"}]'
    response = _llm_fixture_record(text, [])["llm"]["response"]
    expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert response["text"] == text
    assert response["response_hash"] == expected == _llm_digest(text)
    assert response["tokens_in"] is None and response["tokens_out"] is None


@pytest.mark.parametrize("status", ["parse_identical", "parse_drifted", "unmeasured"])
def test_llm_fixture_record_replay_status_is_a_stated_claim(status: str) -> None:
    record = _llm_fixture_record("[]", [], status)
    assert record["replay_status"] == status
    assert record["generation"] == "nondeterministic"


def test_ar_in_010_spec_malformed_max_cards() -> None:
    request = FIXTURES["AR-IN-010"]["request"]
    assert request["action"] == "check"
    proposer = request["campaign_spec"]["proposer"]
    assert proposer["name"] == "llm"
    assert proposer["llm"]["max_cards"] < 1


def test_ar_pr_005_unpinned_endpoint_registry_is_request_supplied() -> None:
    request = FIXTURES["AR-PR-005"]["request"]
    spec = request["campaign_spec"]
    pool = request["llm_pool"]
    assert request["action"] == "propose"
    assert spec["proposer"]["llm"]["pool_key"] == "gpt-x"
    assert "llm_pool" not in spec
    assert spec["proposer"]["llm"]["pool_key"] not in pool
    assert set(pool) == {"nightly-llm"}
    entry = pool["nightly-llm"]
    assert entry["status"] == "active"
    assert entry["model_id"] == "fixture-model-1"
    assert entry["auth"] == {"env": "FIXTURE_KEY"}


def test_ar_pr_005_ledger_is_approval_only() -> None:
    files = FIXTURES["AR-PR-005"]["files"]
    assert len(files["ledger/chain.jsonl"].splitlines()) == 1
    assert not any("llm" in obj for obj in _objects(files) if isinstance(obj, dict))


def test_ar_pr_003_tainted_response_keeps_only_the_clean_card() -> None:
    record = _llm_record(FIXTURES["AR-PR-003"]["files"])
    body = json.loads(record["llm"]["response"]["text"])
    assert len(body) == 2
    assert body[1] == {"command": "scancel 4242"}
    assert record["llm"]["parse"]["cards"] == [body[0]]
    assert record["replay_status"] == "parse_identical"


def test_ar_pr_004_overstated_claim_on_a_clean_response() -> None:
    record = _llm_record(FIXTURES["AR-PR-004"]["files"])
    body = json.loads(record["llm"]["response"]["text"])
    assert record["replay_status"] == "byte_identical"
    assert record["generation"] == "nondeterministic"
    assert record["llm"]["parse"]["cards"] == body


def test_ar_ho_009_recorded_cards_drift_from_the_recorded_response() -> None:
    record = _llm_record(FIXTURES["AR-HO-009"]["files"])
    body = json.loads(record["llm"]["response"]["text"])
    recorded = record["llm"]["parse"]["cards"]
    assert len(body) == 2
    assert body[0]["delta"] == {"optim.lr": 2e-4}
    assert body[1]["delta"] == {"optim.lr": 5e-4}
    assert len(recorded) == 1 and recorded[0]["idea"] == "tampered"
    assert recorded != body
    assert record["replay_status"] == "parse_identical"


@pytest.mark.parametrize("rule", CLOSE_RULES)
def test_close_ledgers_hold_approval_plus_one_proposal(rule: str) -> None:
    fixture = FIXTURES[rule]
    assert fixture["request"]["action"] == "close"
    assert fixture["request"]["stop_reason"] == "budget exhausted"
    assert len(fixture["files"]["ledger/chain.jsonl"].splitlines()) == 2
    record = _llm_record(fixture["files"])
    assert record["llm"]["calls_used"] == 1
    assert record["llm"]["max_calls"] == 6


@pytest.mark.parametrize("rule", CLOSE_RULES)
def test_recorded_cards_stay_in_axes_and_never_clamp(rule: str) -> None:
    spec = FIXTURES[rule]["request"]["campaign_spec"]
    axes = {axis["key"]: axis for axis in spec["axes"]}
    cards = _llm_record(FIXTURES[rule]["files"])["llm"]["parse"]["cards"]
    assert cards
    for card in cards:
        assert card["domain"] == "optimizer"
        assert set(card["delta"]) <= set(axes)
        for key, value in card["delta"].items():
            assert axes[key]["min"] <= value <= axes[key]["max"]


@pytest.mark.parametrize("rule", RULES)
def test_fixture_payloads_are_json_round_trip(rule: str) -> None:
    request = FIXTURES[rule]["request"]
    assert json.loads(json.dumps(request)) == request
    for value in _objects(FIXTURES[rule].get("files", {})):
        assert json.loads(json.dumps(value)) == value


@pytest.mark.parametrize("rule", CLOSE_RULES)
def test_no_credentials_reach_the_ledgered_payloads(rule: str) -> None:
    ledgered = json.dumps(_llm_record(FIXTURES[rule]["files"]))
    assert "FIXTURE_KEY" not in ledgered
    assert "sk-live" not in ledgered


def test_fixture_build_performs_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fixture building must not touch the network")

    monkeypatch.setattr("socket.create_connection", boom)
    monkeypatch.setattr("socket.socket", boom)
    _llm_must_fire_fixtures()
