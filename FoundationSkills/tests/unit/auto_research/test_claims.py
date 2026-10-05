"""Test-only: claim bodies, chain linking, downgrades and the champion rule (pure, CPU-only)."""
from __future__ import annotations

from typing import Any

from foundationskills.skills.auto_research import claims
from foundationskills.skills.auto_research.ledger import canonical, sha256_hex

SEEDS = [101, 102, 103]
SPEC = {"eval_policy": {"fingerprint": "fp1"}}


def _decision(verdict: str = "accepted_gain", **extra: Any) -> dict[str, Any]:
    decision: dict[str, Any] = {"verdict": verdict, "mean_delta": 0.5, "tau": 0.1, "n_pairs": 3}
    decision.update(extra)
    return decision


def _status(complete: bool = True, paired: list[int] | None = None) -> dict[str, Any]:
    paired = list(SEEDS if paired is None else paired)
    return {
        "requested": paired,
        "measured": paired,
        "paired": paired,
        "complete": complete,
        "drops": [],
    }


def _result(
    trial: str, seed: int, role: str = "candidate", status: str = "ok", limited: bool = False
) -> dict[str, Any]:
    return {
        "trial": trial,
        "role": role,
        "seed": seed,
        "status": status,
        "limited": limited,
        "metrics": {"reward": {"value": 1.0, "se": 0.1}},
    }


def _claim(trial: str, reference: str, prev: str | None) -> dict[str, Any]:
    body = claims.claim_body(trial, reference, prev, SEEDS, _decision(), "fp1")
    return {**body, "claim_id": claims.claim_id(body)}


def _evidence() -> dict[tuple[str, int], dict[str, Any]]:
    evidence: dict[tuple[str, int], dict[str, Any]] = {}
    for trial, role in (("baseline", "baseline"), ("t1", "confirm"), ("t2", "confirm")):
        for seed in SEEDS:
            evidence[(trial, seed)] = _result(trial, seed, role=role)
    return evidence


def test_claim_body_shape_and_missing_stats_are_none() -> None:
    body = claims.claim_body(
        "t1", "baseline", "baseline", SEEDS, {"verdict": "accepted_gain", "mean_delta": 0.2}, "fp1"
    )
    assert set(body) == {
        "trial",
        "phase",
        "reference",
        "prev",
        "seeds",
        "stats",
        "eval_policy_fingerprint",
    }
    assert "claim_id" not in body
    assert body["phase"] == "confirm"
    assert body["prev"] == "baseline"
    assert body["seeds"] == SEEDS
    assert body["stats"] == {"verdict": "accepted_gain", "mean_delta": 0.2, "tau": None, "n_pairs": None}
    assert body["eval_policy_fingerprint"] == "fp1"


def test_claim_id_stable_and_byte_sensitive() -> None:
    body = claims.claim_body("t1", "baseline", "baseline", SEEDS, _decision(), "fp1")
    cid = claims.claim_id(body)
    assert cid == claims.claim_id(body)
    assert cid == "sha256:" + sha256_hex(canonical(body))
    probes = [
        {**body, "trial": "t1 "},
        {**body, "phase": "screening"},
        {**body, "reference": "t0"},
        {**body, "prev": "x"},
        {**body, "seeds": [101, 103, 102]},
        {**body, "stats": {**body["stats"], "tau": 0.2}},
        {**body, "stats": {**body["stats"], "n_pairs": 4}},
        {**body, "eval_policy_fingerprint": "fp2"},
    ]
    for probe in probes:
        assert claims.claim_id(probe) != cid


def test_build_claim_gate() -> None:
    assert claims.build_claim(SPEC, "t1", {"verdict": "no_gain"}, _status(), "baseline") is None
    assert claims.build_claim(SPEC, "t1", {"verdict": "unmeasured"}, _status(), "baseline") is None
    assert claims.build_claim(SPEC, "t1", {"verdict": "rejected_regress"}, _status(), "baseline") is None
    assert claims.build_claim(SPEC, "t1", _decision(), _status(complete=False), "baseline") is None
    assert claims.build_claim(SPEC, "t1", _decision(), None, "baseline") is None
    assert claims.build_claim(SPEC, "t1", None, _status(), "baseline") is None
    claim = claims.build_claim(SPEC, "t1", _decision(reference="t0"), _status(), "baseline")
    assert claim is not None
    assert claim["claim_id"] == "sha256:" + sha256_hex(canonical({k: v for k, v in claim.items() if k != "claim_id"}))
    assert claim["prev"] == "baseline"
    assert claim["reference"] == "t0"
    assert claim["seeds"] == SEEDS
    assert claim["eval_policy_fingerprint"] == "fp1"
    assert claim["stats"] == {
        "verdict": "accepted_gain",
        "mean_delta": 0.5,
        "tau": 0.1,
        "n_pairs": 3,
    }
    rooted = claims.build_claim(SPEC, "t1", _decision(), _status(), None)
    assert rooted is not None and rooted["prev"] == "baseline"


def test_claim_entries_order_and_serviced_links() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    c2 = _claim("t2", "t1", c1["claim_id"])
    ordered = claims.claim_entries([c2, c1])
    assert [entry["claim_id"] for entry in ordered] == [c1["claim_id"], c2["claim_id"]]
    assert [entry["prev"] for entry in ordered] == ["baseline", c1["claim_id"]]
    bare1 = claims.claim_body("t1", "baseline", None, SEEDS, _decision(), "fp1")
    bare2 = claims.claim_body("t2", "t1", None, SEEDS, _decision(), "fp1")
    linked = claims.claim_entries([bare1, bare2])
    assert linked[0]["claim_id"] == claims.claim_id({**bare1, "prev": "baseline"})
    assert linked[0]["prev"] == "baseline"
    assert linked[1]["claim_id"] == claims.claim_id({**bare2, "prev": linked[0]["claim_id"]})
    assert linked[1]["prev"] == linked[0]["claim_id"]
    assert [entry["claim_id"] for entry in linked] == [linked[0]["claim_id"], linked[1]["claim_id"]]


def test_derive_chain_links_and_accepts() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    c2 = _claim("t2", "t1", c1["claim_id"])
    chain, drops = claims.derive_chain([c2, c1], _evidence())
    assert [row["claim_id"] for row in chain] == [c1["claim_id"], c2["claim_id"]]
    assert [row["prev"] for row in chain] == ["baseline", c1["claim_id"]]
    assert [row["status"] for row in chain] == ["accepted", "accepted"]
    assert [row["seeds"] for row in chain] == [SEEDS, SEEDS]
    assert chain[1]["reference"] == "t1"
    assert chain[0]["stats"]["verdict"] == "accepted_gain"
    assert chain[1]["stats"]["n_pairs"] == 3
    assert drops == []


def test_derive_chain_downgrades_lost_seed_claim_and_champion_falls_back() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    c2 = _claim("t2", "t1", c1["claim_id"])
    missing = _evidence()
    missing.pop(("t2", 102))
    chain, drops = claims.derive_chain([c1, c2], missing)
    assert [row["status"] for row in chain] == ["accepted", "downgraded"]
    assert drops == [f"claim_downgraded:{c2['claim_id']}"]
    assert claims.champion(chain) == {"trial": "t1", "claim_id": c1["claim_id"]}


def test_derive_chain_downgrades_unmeasured_reference_rows() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    crashed = _evidence()
    crashed[("baseline", 103)] = _result("baseline", 103, role="baseline", status="crash")
    chain, drops = claims.derive_chain([c1], crashed)
    assert chain[0]["status"] == "downgraded"
    assert drops == [f"claim_downgraded:{c1['claim_id']}"]
    limited = _evidence()
    limited[("baseline", 103)] = _result("baseline", 103, role="baseline", limited=True)
    chain2, drops2 = claims.derive_chain([c1], limited)
    assert chain2[0]["status"] == "downgraded"
    assert drops2 == [f"claim_downgraded:{c1['claim_id']}"]


def test_derive_chain_downgrades_both_sides_of_a_lost_seed() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    c2 = _claim("t2", "t1", c1["claim_id"])
    evidence = _evidence()
    evidence.pop(("t1", 103))
    chain, drops = claims.derive_chain([c1, c2], evidence)
    assert [row["status"] for row in chain] == ["downgraded", "downgraded"]
    assert drops == [f"claim_downgraded:{c1['claim_id']}", f"claim_downgraded:{c2['claim_id']}"]
    assert claims.champion(chain) == {"trial": None, "claim_id": "baseline"}


def test_derive_chain_downgrade_does_not_cascade() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    c2 = _claim("t2", "t1", c1["claim_id"])
    evidence = _evidence()
    evidence.pop(("baseline", 103))  # only c1's reference side is lost
    chain, drops = claims.derive_chain([c1, c2], evidence)
    assert chain[0]["status"] == "downgraded"
    assert chain[1]["status"] == "accepted"  # keeps its own status
    assert drops == [f"claim_downgraded:{c1['claim_id']}"]
    assert claims.champion(chain) == {"trial": "t2", "claim_id": c2["claim_id"]}


def test_champion_walks_back_over_downgraded_tails() -> None:
    assert claims.champion([]) == {"trial": None, "claim_id": "baseline"}
    assert claims.champion(None) == {"trial": None, "claim_id": "baseline"}
    only_down = [{"claim_id": "c1", "trial": "t1", "status": "downgraded"}]
    assert claims.champion(only_down) == {"trial": None, "claim_id": "baseline"}
    chain = [
        {"claim_id": "c1", "trial": "t1", "status": "accepted"},
        {"claim_id": "c2", "trial": "t2", "status": "downgraded"},
    ]
    assert claims.champion(chain) == {"trial": "t1", "claim_id": "c1"}
    chain.append({"claim_id": "c3", "trial": "t3", "status": "accepted"})
    assert claims.champion(chain) == {"trial": "t3", "claim_id": "c3"}


def test_reference_rows_champion_else_baseline() -> None:
    rows = [_result("baseline", seed, role="baseline") for seed in SEEDS]
    rows += [_result("t1", seed) for seed in SEEDS]
    rows += [_result("t2", seed) for seed in SEEDS]
    assert claims.reference_rows([], rows) == [row for row in rows if row["role"] == "baseline"]
    chain = [
        {"claim_id": "c1", "trial": "t1", "status": "accepted"},
        {"claim_id": "c2", "trial": "t2", "status": "downgraded"},
    ]
    assert claims.reference_rows(chain, rows) == [row for row in rows if row["trial"] == "t1"]
    chain[1]["status"] = "accepted"
    assert claims.reference_rows(chain, rows) == [row for row in rows if row["trial"] == "t2"]
    downed = [
        {"claim_id": "c1", "trial": "t1", "status": "downgraded"},
        {"claim_id": "c2", "trial": "t2", "status": "downgraded"},
    ]
    assert claims.reference_rows(downed, rows) == [row for row in rows if row["role"] == "baseline"]
