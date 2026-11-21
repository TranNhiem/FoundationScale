"""Unit tests for MODULES A and B: frozen surface and admission gate."""
import pytest

from foundationskills.agent.frozen_surface import assert_editable, fingerprint
from foundationskills.agent.admit import admit, admission_hash, admission_payload

FENCE = '`' * 3
NAME = "self-improve-gates"

OLD = (
    "---\n"
    "name: self-improve-gates\n"
    "description: Inspect the gates for the agent.\n"
    "license: Apache-2.0\n"
    "metadata:\n"
    "  invocation: fskills self-improve\n"
    "  when_to_use: any session\n"
    "---\n"
    "\n"
    "This body sentence explains the flow.\n"
    "\n"
    "<!-- FS-USAGE:BEGIN -->\n"
    "Run: fskills self-improve hash\n"
    "<!-- FS-USAGE:END -->\n"
    "\n"
    "Rule CP-IN-004 applies here.\n"
    "\n"
    + FENCE + "python\nprint(1)\n" + FENCE + "\n" +
    "\n"
    "On failure use exit 96.\n"
    "\n"
    "Docs: https://example.org/gates\n"
)

PROSE = OLD.replace("This body sentence explains the flow.",
                    "This body sentence explains the admission flow clearly.")


def ev(gain=0.1, measured=True, drop=None, inc_drop=None, **cand):
    candidate = {"behaviour_case_pass": 0.8 + gain, "oracle_item_pass": 0.9, "routing_recall": 0.7}
    incumbent = {"behaviour_case_pass": 0.8, "oracle_item_pass": 0.9, "routing_recall": 0.7}
    candidate.update(cand)
    if drop is not None:
        del candidate[drop]
    if inc_drop is not None:
        del incumbent[inc_drop]
    return {"measured": measured, "target": "behaviour_case_pass",
            "incumbent": incumbent, "candidate": candidate}


def hash_for(new, evidence):
    return admission_hash(admission_payload(NAME, OLD, new, evidence))


def test_prose_only_edits_are_editable():
    desc = OLD.replace("description: Inspect the gates for the agent.",
                       "description: Inspect the admission gates of the agent.")
    assert assert_editable(OLD, desc) == []
    assert assert_editable(OLD, PROSE) == []
    assert fingerprint(desc) == fingerprint(OLD) == fingerprint(PROSE)


def test_each_frozen_part_change_is_detected():
    cases = [
        ("name: self-improve-gates", "name: renamed-skill", "name"),
        ("Run: fskills self-improve hash", "Run: fskills self-improve run", "usage"),
        ("print(1)", "print(2)", "code block"),
        ("Rule CP-IN-004 applies here.", "No rule applies here.", "rule id"),
        ("On failure use exit 96.", "On failure fail loudly.", "exit"),
        ("https://example.org/gates", "https://example.org/rules", "url"),
    ]
    for old_s, new_s, fragment in cases:
        edited = OLD.replace(old_s, new_s)
        problems = assert_editable(OLD, edited)
        assert problems and any(fragment in p for p in problems), problems
        assert fingerprint(edited) != fingerprint(OLD)


def test_admit_frozen_change_is_refused():
    new = OLD.replace("name: self-improve-gates", "name: totally-different")
    evidence = ev()
    res = admit(NAME, OLD, new, evidence, hash_for(new, evidence))
    assert res["verdict"] == "REFUSED"
    assert any("name" in r for r in res["reasons"])


def test_wrong_or_missing_confirm_refused_hash_hidden():
    evidence = ev()
    expected = hash_for(PROSE, evidence)
    for confirm in (None, "", "deadbeefdeadbeef"):
        res = admit(NAME, OLD, PROSE, evidence, confirm)
        assert res["verdict"] == "REFUSED"
        joined = "|".join(res["reasons"])
        assert "confirmation required" in joined
        assert expected not in joined


def test_unmeasured_evidence_is_flagged():
    for evidence in (ev(measured=False), ev(drop="routing_recall"), ev(inc_drop="oracle_item_pass")):
        res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
        assert res["verdict"] == "UNMEASURED"


def test_no_gain_is_red():
    for evidence in (ev(gain=0.0), ev(gain=-0.05)):
        res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
        assert res["verdict"] == "RED"
        assert res["reasons"]


def test_protected_metric_down_is_red_naming_it():
    for metric, value in (("routing_recall", 0.6), ("oracle_item_pass", 0.85)):
        evidence = ev(**{metric: value})
        res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
        assert res["verdict"] == "RED"
        assert any(metric in r for r in res["reasons"])


def test_valid_candidate_passes_with_gain():
    evidence = ev(gain=0.1)
    res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
    assert res["verdict"] == "PASS"
    assert res["gain"] == pytest.approx(0.1)
    assert res["reasons"] == []


def test_admission_hash_changes_with_new_text():
    other = PROSE.replace("This body sentence explains the admission flow clearly.",
                          "A total rewrite of this body sentence now.")
    evidence = ev()
    assert len(hash_for(PROSE, evidence)) == 16
    assert hash_for(PROSE, evidence) == hash_for(PROSE, evidence)
    assert hash_for(PROSE, evidence) != hash_for(other, evidence)
    assert hash_for(other, ev(measured=True, gain=0.1)) != hash_for(other, ev(measured=True, gain=0.0))
