"""Unit tests for the M5a multi-objective acceptor: accept.decide_multi and accept.frontier (A.5/A.6).

Baseline rows vary per seed so noise floors calibrate:
  val_accuracy = 0.500/0.501/0.502 (se 0.001) -> noise_floor = 1.4826 * MAD = 0.0014826
  toks_per_sec = 100/101/102       (se 1.0)   -> noise_floor = 1.4826
  mem_gib      = 4.0   (se 0.0)              (guard metric; guardrail_directions = min)

Under the base confirm (k=2.0, noise_floor_rel=0.005, 3 seed pairs) the tau per objective is
  tau_val = max(0.0014826, 0.005*0.501, 2*sqrt(2e-6)) = 0.002828427...   (se_band dominates)
  tau_tps = max(1.4826,   0.005*101.0, 2*sqrt(2))    = 2.8284271...     (se_band dominates)
so per-seed candidate deltas of +-0.01 (val) / +-10 (tps) are unambiguous wins / losses and 0.0 is
an unambiguous tie.  The boundary fixture isolates a controlled spec where tau == 0.5 exactly
(noise_floor_rel * |mean_ref| with rel=0.5, constant ref 1.0, se 0) so delta == +-tau is asserted
at the strict boundary as a tie."""
from __future__ import annotations

import copy
from typing import Any

import pytest

from foundationskills.skills.auto_research.accept import decide, decide_multi, frontier

FP = "fp_m5a_test"
SEEDS = (101, 102, 103)
VAL, TPS, MEM = "val_accuracy", "toks_per_sec", "mem_gib"
S_VAL = {101: (0.500, 0.001), 102: (0.501, 0.001), 103: (0.502, 0.001)}
S_TPS = {101: (100.0, 1.0), 102: (101.0, 1.0), 103: (102.0, 1.0)}
S_MEM = {101: (4.0, 0.0), 102: (4.0, 0.0), 103: (4.0, 0.0)}
BOTH = list(sorted([VAL, TPS]))  # ["toks_per_sec", "val_accuracy"] mirrors code's sorted(...)


def _row(trial: str, role: str, seed: int, metrics: dict[str, tuple[float, float]],
         simpler: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial": trial, "role": role, "seed": seed, "status": "ok", "limited": False,
        "steps": 200, "eval_policy_fingerprint": FP,
        "metrics": {m: {"value": float(v), "se": float(s)} for m, (v, s) in metrics.items()},
    }
    if simpler:
        row["simpler"] = True
    return row


def _base() -> list[dict[str, Any]]:
    return [_row("base", "baseline", s, {VAL: S_VAL[s], TPS: S_TPS[s], MEM: S_MEM[s]}) for s in SEEDS]


def _cand(*, val_dp: float = 0.0, tps_dp: float = 0.0, mem_dp: float = 0.0,
          tps_omit: tuple[int, ...] = (), tps_all_drop: bool = False,
          simpler: bool = False) -> list[dict[str, Any]]:
    rows = []
    for s in SEEDS:
        metrics: dict[str, tuple[float, float]] = {}
        v, se = S_VAL[s]; metrics[VAL] = (v + val_dp, se)
        if not tps_all_drop and s not in tps_omit:
            v, se = S_TPS[s]; metrics[TPS] = (v + tps_dp, se)
        v, se = S_MEM[s]; metrics[MEM] = (v + mem_dp, se)
        rows.append(_row("cand", "candidate", s, metrics, simpler=simpler))
    return rows


def _spec(**over: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "eval_policy": {"fingerprint": FP, "metrics": [VAL, TPS, MEM]},
        "confirm": {"k": 2.0, "noise_floor_rel": 0.005, "guardrails": [MEM],
                    "guardrail_directions": {MEM: "min"}, "guard_abs_epsilon": 0.01},
        "seeds": {"baseline_repeats": 3, "confirm_repeats": 3, "seed_list": list(SEEDS)},
        "objective": {"metric": VAL, "direction": "max", "benchmarks": ["mmlu"]},
        "objectives": [{"metric": VAL, "direction": "max"}, {"metric": TPS, "direction": "max"}],
    }
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(spec.get(key), dict):
            spec[key] = {**spec[key], **val}
        else:
            spec[key] = val
    return spec


def _bnd_base() -> list[dict[str, Any]]:
    """Constant ref 1.0, se 0 -> tau == rel * |mean_ref| == 0.5 exactly (rel 0.5)."""
    return [_row("base", "baseline", s, {VAL: (1.0, 0.0), TPS: (1.0, 0.0), MEM: (4.0, 0.0)})
            for s in SEEDS]


def _bnd_cand(val_dp: float, tps_dp: float) -> list[dict[str, Any]]:
    return [_row("cand", "candidate", s,
                 {VAL: (1.0 + val_dp, 0.0), TPS: (1.0 + tps_dp, 0.0), MEM: (4.0, 0.0)})
            for s in SEEDS]


def _fx(cid: str, **kw: Any) -> dict[str, Any]:
    kw["id"] = cid
    return kw


_FIXTURES = [
    _fx("two_wins", spec=_spec(), baseline=_base(), candidate=_cand(val_dp=0.01, tps_dp=10.0),
        verdict="accepted_gain", val_cls="win", tps_cls="win", wins=BOTH, losses=[], ties=[]),
    _fx("win_and_tie", spec=_spec(), baseline=_base(), candidate=_cand(val_dp=0.01),
        verdict="accepted_gain", val_cls="win", tps_cls="tie", wins=[VAL], losses=[], ties=[TPS]),
    _fx("win_and_loss_tradeoff", spec=_spec(), baseline=_base(),
        candidate=_cand(val_dp=0.01, tps_dp=-10.0),
        verdict="tradeoff", val_cls="win", tps_cls="loss", wins=[VAL], losses=[TPS], ties=[]),
    _fx("loss_and_tie", spec=_spec(), baseline=_base(), candidate=_cand(tps_dp=-10.0),
        verdict="rejected_regress", val_cls="tie", tps_cls="loss", wins=[], losses=[TPS], ties=[VAL]),
    _fx("all_tie_simpler", spec=_spec(), baseline=_base(), candidate=_cand(simpler=True),
        verdict="accepted_flat", val_cls="tie", tps_cls="tie", wins=[], losses=[], ties=BOTH),
    _fx("all_tie", spec=_spec(), baseline=_base(), candidate=_cand(),
        verdict="no_gain", val_cls="tie", tps_cls="tie", wins=[], losses=[], ties=BOTH),
    _fx("loss_only", spec=_spec(), baseline=_base(), candidate=_cand(val_dp=-0.01, tps_dp=-10.0),
        verdict="rejected_regress", val_cls="loss", tps_cls="loss", wins=[], losses=BOTH, ties=[]),
    _fx("partial_secondary", spec=_spec(), baseline=_base(),
        candidate=_cand(val_dp=0.01, tps_dp=10.0, tps_omit=(103,)),
        verdict="unmeasured", val_cls="unmeasured", tps_cls="unmeasured",
        wins=[], losses=[], ties=[]),
    _fx("guardrail_breach_2_wins", spec=_spec(), baseline=_base(),
        candidate=_cand(val_dp=0.01, tps_dp=10.0, mem_dp=1.0),
        verdict="rejected_regress", val_cls="win", tps_cls="win", wins=BOTH, losses=[], ties=[]),
    _fx("boundary_tau_tie", spec=_spec(confirm={"noise_floor_rel": 0.5, "guardrails": []}),
        baseline=_bnd_base(), candidate=_bnd_cand(0.5, -0.5),
        verdict="no_gain", val_cls="tie", tps_cls="tie", wins=[], losses=[], ties=BOTH),
    _fx("j_below_confirm_repeats", spec=_spec(seeds={"confirm_repeats": 5}),
        baseline=_base(), candidate=_cand(val_dp=0.01, tps_dp=10.0),
        verdict="unmeasured", val_cls="unmeasured", tps_cls="unmeasured",
        wins=[], losses=[], ties=[]),
    _fx("secondary_absent", spec=_spec(), baseline=_base(),
        candidate=_cand(val_dp=0.01, tps_all_drop=True),
        verdict="unmeasured", val_cls="unmeasured", tps_cls="unmeasured",
        wins=[], losses=[], ties=[]),
]


@pytest.mark.parametrize("case", _FIXTURES, ids=[c["id"] for c in _FIXTURES])
def test_decide_multi_fixtures(case: dict[str, Any]) -> None:
    r = decide_multi(case["spec"], case["baseline"], case["candidate"])
    assert r["verdict"] == case["verdict"]
    for metric, want_cls in ((VAL, case["val_cls"]), (TPS, case["tps_cls"])):
        po = r["per_objective"][metric]
        assert po["class"] == want_cls
        if want_cls == "unmeasured":
            assert po["mean_delta"] is None and po["tau"] is None  # never 0
        else:
            assert po["mean_delta"] is not None and po["tau"] is not None
    assert r["wins"] == case["wins"]
    assert r["losses"] == case["losses"]
    assert r["ties"] == case["ties"]
    assert r["mean_delta"] == r["per_objective"][VAL]["mean_delta"]
    assert r["tau"] == r["per_objective"][VAL]["tau"]
    assert r["n_pairs"] == r["per_objective"][VAL]["n_pairs"]
    assert r["reasons"] and all(isinstance(s, str) and s for s in r["reasons"])


def test_boundary_eq_tau_is_tie() -> None:
    """delta_j == +tau_j and delta_j == -tau_j are strictly ties (A.6 boundary)."""
    spec = _spec(confirm={"noise_floor_rel": 0.5, "guardrails": []})
    r = decide_multi(spec, _bnd_base(), _bnd_cand(0.5, -0.5))
    val, tps = r["per_objective"][VAL], r["per_objective"][TPS]
    assert (val["mean_delta"], val["tau"], val["class"]) == (0.5, 0.5, "tie")
    assert (tps["mean_delta"], tps["tau"], tps["class"]) == (-0.5, 0.5, "tie")
    assert r["verdict"] == "no_gain"


def test_common_seed_set_is_intersection_across_objectives() -> None:
    """Per-metric seed coverage cannot let a full metric sneak a win through a shrunken J."""
    spec = _spec(seeds={"confirm_repeats": 3})
    cand = _cand(val_dp=0.01, tps_dp=10.0, tps_omit=(103,))
    r = decide_multi(spec, _base(), cand)
    assert r["verdict"] == "unmeasured"
    assert r["n_pairs"] == 2  # J is the intersection across all objective metrics
    assert any(s.startswith("objective_metric_absent:" + TPS + ":cand:103") for s in r["reasons"])
    assert any(s.startswith("n_pairs_below_confirm:2/") for s in r["reasons"])
    # even the fully-measured primary objective cannot declare a win on the shrunken J
    assert r["per_objective"][VAL]["class"] == "unmeasured"
    assert r["per_objective"][VAL]["mean_delta"] is None


def test_unmeasured_objective_delta_is_never_zero() -> None:
    """A missing secondary metric surfaces as mean_delta None, never 0 (A.6)."""
    r = decide_multi(_spec(), _base(), _cand(val_dp=0.01, tps_all_drop=True))
    assert r["per_objective"][TPS]["mean_delta"] is None
    assert r["per_objective"][TPS]["class"] == "unmeasured"
    assert r["per_objective"][VAL]["mean_delta"] is None
    assert r["per_objective"][VAL]["class"] == "unmeasured"
    assert any(s.startswith("objective_metric_absent:" + TPS + ":cand:") for s in r["reasons"])
    assert any(s.startswith("n_pairs_below_confirm:0/") for s in r["reasons"])


def test_decide_unchanged_on_single_objective_spec() -> None:
    """accept.decide is untouched: a spec without ``objectives`` yields M4-identical verdicts."""
    spec = _spec()
    spec.pop("objectives")  # M4 shape; decide() must not depend on it
    baseline = _base()
    for cand, expected in (
        (_cand(val_dp=0.01), "accepted_gain"),
        (_cand(), "no_gain"),
        (_cand(simpler=True), "accepted_flat"),
        (_cand(val_dp=-0.02), "rejected_regress"),
    ):
        out = decide(spec, baseline, cand, VAL)
        assert out["verdict"] == expected
        assert out["n_pairs"] == 3 and out["mean_delta"] is not None and out["tau"] is not None


def _dec(val: float | None, tps: float | None, wins: list[str], losses: list[str],
         ties: list[str], verdict: str) -> dict[str, Any]:
    tau = 0.01
    def cls(v: float | None, m: str) -> str:
        if v is None:
            return "unmeasured"
        return "win" if m in wins else ("loss" if m in losses else "tie")
    return {"verdict": verdict, "wins": sorted(wins), "losses": sorted(losses),
            "ties": sorted(ties), "mean_delta": val, "tau": tau, "n_pairs": 3,
            "per_objective": {
                VAL: {"mean_delta": val, "tau": tau, "class": cls(val, VAL), "n_pairs": 3},
                TPS: {"mean_delta": tps, "tau": tau, "class": cls(tps, TPS), "n_pairs": 3},
            }}


def test_frontier_presentation_only_ordering_and_drops() -> None:
    """tau-weak Pareto frontier is report-only; dominated / unmeasured drops are named."""
    decisions: dict[str, dict[str, Any]] = {
        "A": _dec(0.5, -0.5, wins=[VAL], losses=[TPS], ties=[], verdict="tradeoff"),
        "B": _dec(-0.5, 0.5, wins=[TPS], losses=[VAL], ties=[], verdict="tradeoff"),
        "C": _dec(0.2, 0.2, wins=[VAL, TPS], losses=[], ties=[], verdict="accepted_gain"),
        "D": _dec(0.1, 0.1, wins=[VAL, TPS], losses=[], ties=[], verdict="accepted_gain"),
        "U": _dec(None, None, wins=[], losses=[], ties=[], verdict="unmeasured"),
    }
    snapshot = copy.deepcopy(decisions)
    entries, drops = frontier(decisions)
    assert decisions == snapshot  # frontier is pure - inputs are not mutated
    assert [e["trial"] for e in entries] == ["C", "A", "B"]  # order: (-len(wins), len(losses), trial)
    assert "frontier_excluded:D:dominated" in drops
    assert "frontier_excluded:U:unmeasured" in drops
    assert entries[0]["class"] == "accepted_gain" and entries[0]["wins"] == sorted([VAL, TPS])
    assert entries[1]["losses"] == [TPS] and entries[2]["losses"] == [VAL]


def test_decide_multi_does_not_mutate_inputs() -> None:
    spec = _spec()
    baseline = _base()
    candidate = _cand(val_dp=0.01, tps_dp=-10.0)
    before = (copy.deepcopy(spec), copy.deepcopy(baseline), copy.deepcopy(candidate))
    decide_multi(spec, baseline, candidate)
    assert (spec, baseline, candidate) == before
