"""C9: evidence is built from measured reports and is unmeasured when the candidate text was not measured."""

from foundationskills.agent.self_improve_evidence import build_evidence, trigger_recall


def _beh(case_pass, item_pass, override=None, verdict="pass"):
    rep = {"verdict": verdict, "results": {"with_skill": {"case_pass_rate": case_pass, "item_pass_rate": item_pass}},
           "calls": [{}] * 10}
    if override:
        rep["skill_md_override"] = {override: "ab" * 32}
    return rep


def _routing(hits, total, pkg="p"):
    calls = [{"arm": "full", "expected": pkg, "measured": True, "correct": i < hits} for i in range(total)]
    return {"verdict": "pass", "gated_arm": "full", "calls": calls}


def test_trigger_recall_counts_only_the_package_in_the_gated_arm():
    assert trigger_recall(_routing(9, 10), "p") == 0.9 and trigger_recall(_routing(9, 10), "q") is None


def test_evidence_is_measured_only_for_candidate_reports_over_the_candidate_text():
    inc = {"routing_behaviour": _beh(0.8, 0.9), "oracle": _beh(0.7, 0.9), "routing": _routing(9, 10)}
    cand = {"routing_behaviour": _beh(0.9, 0.9, "p"), "oracle": _beh(0.7, 0.92, "p"), "routing": _routing(9, 10)}
    ev = build_evidence("p", inc, cand)
    assert ev["measured"] and ev["candidate"]["behaviour_case_pass"] == 0.9 and ev["incumbent"]["routing_recall"] == 0.9
    assert ev["gain_per_100_calls"] == round(100 * 0.1 / 30, 4)
    stale = {**cand, "oracle": _beh(0.7, 0.92)}  # the oracle run did not measure the candidate text
    assert build_evidence("p", inc, stale)["measured"] is False
    unmeasured = {**cand, "routing_behaviour": _beh(0.9, 0.9, "p", verdict="unmeasured")}
    assert build_evidence("p", inc, unmeasured)["measured"] is False
