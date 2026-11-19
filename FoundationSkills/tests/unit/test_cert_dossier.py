"""tests/unit/test_cert_dossier.py -- MODULE 2: certification dossier verdicts (fs.cert_dossier.v1).

Verdict order covered: REFUSED (broken manifest) -> UNMEASURED (only infra runs) -> RED
(beacon miss naming the fixture) -> UNMEASURED (one harness config) -> PASS (two configs),
plus infra exclusion from pass_rate, ValueError on unknown validity, robust_band numbers.
"""
from __future__ import annotations

import pytest

from foundationskills.cert.dossier import (SuiteResult, beacon_ok, build_dossier,
                                           robust_band, valid_results)
from foundationskills.core.harness import (UNMEASURED, HarnessManifest, SkillPin,
                                           manifest_hash, manifest_payload, validate_manifest)

SHA64 = "ab" * 32  # 64 lowercase hex chars (skill_md_sha256)
GOOD = ("r1", "r2")
GRID = {"alpha": 0.8, "beta": 0.7}
GRID1 = {"alpha": 0.8}


def result(fid="fx.a", expected=GOOD, fired=GOOD, verdict="PASS", validity="valid"):
    return SuiteResult(fixture_id=fid, rule_ids_expected=expected, rule_ids_fired=fired,
                       verdict=verdict, validity=validity)


def manifest():  # valid harness -> payload ("unmeasured" markers are gaps, not errors)
    harness = HarnessManifest(
        skills=(SkillPin("planner", "1.0.0", SHA64),),
        routing_registry_version=UNMEASURED,
        foundationskills_version="0.3.0",
        source_commit="deadbeef",
        pool_registry_sha256=UNMEASURED,
        pool_entries=("ns-1", "ns-2"),
        substrate_image="fs/substrate:1",
        seed=0,
        reset_semantics="cold",
        workspace_snapshot="snap-42")
    return manifest_payload(harness)


def test_valid_manifest_is_clean_even_with_unmeasured_markers():
    assert validate_manifest(manifest()) == []


def test_1_refused_on_broken_manifest_even_if_results_are_perfect():
    broken = dict(manifest())
    broken["seed"] = "0"  # wrong type -> manifest problems
    dossier = build_dossier(broken, [result(), result("fx.b")], GRID)
    assert dossier["verdict"] == "REFUSED"
    assert dossier["manifest_problems"] and validate_manifest(broken)
    assert dossier["beacon_ok"] is True  # perfect results never rescue a broken manifest
    assert dossier["pass_rate"] == pytest.approx(1.0)


def test_2_unmeasured_with_only_infra_results():
    dossier = build_dossier(manifest(),
                            [result(validity="infra"), result("fx.b", validity="infra")], GRID)
    assert (dossier["verdict"], dossier["n_valid"], dossier["n_infra"]) == ("UNMEASURED", 0, 2)
    assert any("no valid results" in r for r in dossier["reasons"])


def test_3_red_naming_the_fixture_that_did_not_fire():
    miss = result("fx.miss", expected=("r9",), fired=("r1",), verdict="RED")
    dossier = build_dossier(manifest(), [result(), miss], GRID)
    assert dossier["verdict"] == "RED"
    assert "fx.miss" in " ".join(dossier["reasons"])
    assert dossier["beacon_ok"] is False and dossier["pass_rate"] == pytest.approx(0.5)


def test_4_unmeasured_with_one_config():
    dossier = build_dossier(manifest(), [result(), result("fx.b")], GRID1)
    assert dossier["verdict"] == "UNMEASURED" and dossier["robust_band"] is None
    assert any("fewer than 2 harness configs" in r for r in dossier["reasons"])


def test_5_pass_with_two_configs():
    dossier = build_dossier(manifest(), [result(), result("fx.b")], GRID)
    assert dossier["verdict"] == "PASS"
    assert dossier["manifest_hash"] == manifest_hash(manifest())
    assert dossier["manifest_problems"] == [] and dossier["pass_rate"] == pytest.approx(1.0)
    assert dossier["robust_band"] == {"configs": 2, "min": 0.7, "max": 0.8,
                                      "spread": pytest.approx(0.1)}


def test_infra_runs_are_dropped_before_scoring_and_out_of_pass_rate():
    infra = result("fx.infra", expected=("r7",), fired=(), verdict="RED", validity="infra")
    dossier = build_dossier(manifest(), [result(), result("fx.b"), infra], GRID)
    assert (dossier["n_results"], dossier["n_valid"], dossier["n_infra"]) == (3, 2, 1)
    assert dossier["beacon_ok"] is True  # the infra beacon miss must not red the run
    assert dossier["pass_rate"] == pytest.approx(1.0)  # 2/2 valid PASS, not 2/3
    assert dossier["verdict"] == "PASS"


def test_valid_results_and_beacon_ok():
    keep = valid_results([result(), result("fx.i", validity="infra")])
    assert [r.fixture_id for r in keep] == ["fx.a"]
    assert beacon_ok([]) is False
    assert beacon_ok([result("fx.i", expected=("r1",), fired=(), validity="infra")]) is False
    assert beacon_ok([result(), result("fx.b", expected=("r1",), fired=("r1", "r2"))]) is True


def test_unknown_validity_raises_value_error():
    with pytest.raises(ValueError):
        valid_results([result(validity="flaky")])


def test_robust_band_numbers():
    assert robust_band({}) is None and robust_band(GRID1) is None
    assert robust_band(GRID) == {"configs": 2, "min": 0.7, "max": 0.8, "spread": pytest.approx(0.1)}
    assert robust_band({"a": 0.9, "b": 0.5, "c": 0.7}) == {"configs": 3, "min": 0.5, "max": 0.9,
                                                           "spread": pytest.approx(0.4)}
    assert robust_band({"a": 0.75, "b": 0.75})["spread"] == pytest.approx(0.0)


def test_garbage_manifest_is_refused_without_raising():
    d = build_dossier(None, [], None)
    assert d["verdict"] == "REFUSED" and d["manifest_hash"] is None
