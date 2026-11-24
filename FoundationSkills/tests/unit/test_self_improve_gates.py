"""Unit tests for MODULES A and B: frozen surface, beacon and paired admission gates, and evidence building."""
import pytest

from foundationskills.agent.frozen_surface import assert_editable, fingerprint
from foundationskills.agent.admit import admit, admission_hash, admission_payload, paired_signflip_p
from foundationskills.agent.self_improve_evidence import beacon_fired, build_evidence, paired_cases

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


def ev(gain=0.1, measured=True, drop=None, inc_drop=None, fired=1, pairs=None, **cand):
    candidate = {"behaviour_case_pass": 0.8 + gain, "oracle_item_pass": 0.9, "routing_recall": 0.7}
    incumbent = {"behaviour_case_pass": 0.8, "oracle_item_pass": 0.9, "routing_recall": 0.7}
    candidate.update(cand)
    if drop is not None:
        del candidate[drop]
    if inc_drop is not None:
        del incumbent[inc_drop]
    if pairs is None:
        pairs = [[0.0, 1.0] for _ in range(8)]
    return {"measured": measured, "target": "behaviour_case_pass",
            "incumbent": incumbent, "candidate": candidate,
            "beacon": {"condition": "atlas-activation-beacon", "fired": fired},
            "paired": {"target": "behaviour_case_pass", "pairs": pairs}}


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


def test_missing_beacon_is_unmeasured():
    evidence = ev()
    del evidence["beacon"]
    res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
    assert res["verdict"] == "UNMEASURED"
    assert any("candidate declares no activation beacon" in r for r in res["reasons"])


def test_beacon_fired_zero_is_red_naming_inert():
    evidence = ev(fired=0)
    res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
    assert res["verdict"] == "RED"
    assert any("inert" in r for r in res["reasons"])


def test_fewer_than_three_pairs_is_unmeasured():
    evidence = ev(pairs=[[0.0, 1.0], [0.0, 1.0]])
    res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
    assert res["verdict"] == "UNMEASURED"
    assert any("paired evidence missing" in r for r in res["reasons"])


def test_noisy_pairs_are_red_naming_not_significant():
    pairs = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    evidence = ev(pairs=pairs)
    res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
    assert res["verdict"] == "RED"
    assert any("not significant" in r for r in res["reasons"])


def test_paired_signflip_p_exact_values():
    mean, p = paired_signflip_p([[0, 1]] * 5)
    assert mean == 1.0
    assert p == 1 / 32

    mean, p = paired_signflip_p([[0, 0]] * 4)
    assert mean == 0.0
    assert p == 1.0


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


def test_valid_candidate_passes_with_gain_and_paired_significance():
    evidence = ev(gain=0.1)
    res = admit(NAME, OLD, PROSE, evidence, hash_for(PROSE, evidence))
    assert res["verdict"] == "PASS"
    assert res["gain"] == pytest.approx(0.1)
    assert res["paired_p"] < 0.05
    assert res["n_pairs"] == 8
    assert res["reasons"] == []


def _call(case_id, rep, met, graded=2, measured=True, arm="with_skill", package=NAME, answer=""):
    return {"arm": arm, "package": package, "case_id": case_id, "rep": rep, "items_met": met,
            "items_graded": graded, "measured": measured, "answer": answer}


def test_paired_cases_aligns_shared_measured_case_reps_incumbent_first():
    incumbent = {"calls": [_call("shared", 0, 1), _call("shared", 1, 2), _call("incumbent-only", 0, 2),
                           _call("unmeasured", 0, 0), _call("shared", 0, 2, arm="without_skill"),
                           _call("shared", 0, 2, package="another-package")]}
    candidate = {"calls": [_call("candidate-only", 0, 2), _call("unmeasured", 0, 2, measured=False),
                           _call("shared", 1, 2), _call("shared", 0, 2)]}
    assert paired_cases(incumbent, candidate, NAME) == [[0.0, 1.0], [1.0, 1.0]]


def test_beacon_fired_is_case_insensitive_and_arm_and_package_filtered():
    report = {"calls": [_call("a", 0, 2, answer="prefix AtLaS-BeAcOn suffix"), _call("b", 0, 2, answer="atlas-beacon"),
                        _call("c", 0, 2, answer="no activation marker"),
                        _call("d", 0, 2, arm="without_skill", answer="ATLAS-BEACON"),
                        _call("e", 0, 2, package="another-package", answer="ATLAS-BEACON"),
                        _call("f", 0, 2, measured=False, answer="ATLAS-BEACON")]}
    assert beacon_fired(report, "atlas-beacon", NAME) == 2


def test_build_evidence_omits_beacon_when_beacon_is_none():
    reports = {"routing_behaviour": {}, "oracle": {}, "routing": {}}
    evidence = build_evidence(NAME, reports, reports, beacon=None)
    assert "beacon" not in evidence


def test_admission_hash_changes_with_new_text():
    other = PROSE.replace("This body sentence explains the admission flow clearly.",
                          "A total rewrite of this body sentence now.")
    evidence = ev()
    assert len(hash_for(PROSE, evidence)) == 16
    assert hash_for(PROSE, evidence) == hash_for(PROSE, evidence)
    assert hash_for(PROSE, evidence) != hash_for(other, evidence)
    assert hash_for(other, ev(measured=True, gain=0.1)) != hash_for(other, ev(measured=True, gain=0.0))


def test_added_unsafe_span_is_red_and_localised():
    from foundationskills.agent.admit import unsafe_spans

    # prose edit (passes the frozen-surface check) smuggling a bidi override (VG-08, Trojan Source) into an instruction sentence
    bad = PROSE.replace("explains the admission flow clearly.", "explains the admission \u202eflow clearly.")
    assert bad != PROSE
    spans = unsafe_spans(OLD, bad)
    assert spans and all(s.startswith("unsafe span added (") for s in spans)
    evidence = ev(measured=True, gain=0.1)
    res = admit(NAME, OLD, bad, evidence, hash_for(bad, evidence))
    assert res["verdict"] == "RED" and res["reasons"] == spans


def test_preexisting_text_is_not_rescanned_only_added_lines():
    from foundationskills.agent.admit import added_text, unsafe_spans

    assert added_text("a\nb\n", "a\nb\nc\n") == "c"
    assert unsafe_spans(OLD, PROSE) == []
