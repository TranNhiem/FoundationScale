"""M6 enforcement: rl declarations (AR-IN-013), unmeasured rows in acceptance, AR-RS-009 / AR-HO-010."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.skills.auto_research import accept, campaign
from foundationskills.skills.auto_research import skill as skill_mod


FINGERPRINT = "sha256:" + "ab" * 32


# --- shared builders -------------------------------------------------------------------


def _row(
    trial: str, seed: int, value: float, *, role: str = "confirm", status: str = "ok", **extra: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial": trial,
        "role": role,
        "seed": seed,
        "status": status,
        "limited": False,
        "eval_policy_fingerprint": FINGERPRINT,
        "metrics": {"m1": {"value": value, "se": 0.1}},
    }
    row.update(extra)
    return row


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": "m6-campaign",
        "stage": "sft",
        "objective": {"metric": "m1", "direction": "max"},
        "eval_policy": {"fingerprint": FINGERPRINT, "metrics": ["m1"]},
        "base": {"model": "model-7b", "fingerprint": "sha256:" + "cd" * 32},
        "confirm": {
            "k": 2.0,
            "noise_floor_rel": 0.005,
            "guard_abs_epsilon": 0.01,
            "guardrails": [],
            "guardrail_directions": {},
        },
        "seeds": {"baseline_repeats": 2, "confirm_repeats": 1},
        "budget": {"gpu_hours_total": 8, "max_runs": 4, "per_run_timeout_h": 1.0, "reserve_frac": 0.3},
        "cluster": {"exclude": []},
        "axes": [],
    }
    spec.update(overrides)
    return spec


def _rl_spec(**overrides: Any) -> dict[str, Any]:
    spec = {
        "stage": "rl",
        "rl_min_measured_fraction": 0.5,
        "confirm": {
            "guardrails": ["rl_measured_fraction", "m1"],
            "guardrail_directions": {"rl_measured_fraction": "max"},
        },
        "eval_policy": {"metrics": ["m1", "rl_measured_fraction"]},
    }
    spec.update(overrides)
    return spec


def _digest(path: Any) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path: Any, *, drop_manifest: bool) -> dict[str, Any]:
    """A bound evidence payload; the rl manifest can be made to disappear before close."""
    report = tmp_path / "eval_report.json"
    report.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    payload: dict[str, Any] = {
        "run_manifests": [],
        "eval_report": {
            "path": str(report),
            "sha256": _digest(report),
            "verdict": "PASS",
            "checkpoint": str(tmp_path / "final"),
        },
        "launch_logs": [],
    }
    if drop_manifest:
        manifest = tmp_path / "fskills_rl_manifest.json"
        manifest.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        payload["run_manifests"].append(
            {"path": str(manifest), "sha256": _digest(manifest), "kind": "rl", "status": "ok"}
        )
        manifest.unlink()
    return payload


def _texts(obj: Any) -> set[str]:
    """Every string reachable from a result: report fields, drops and finding objects alike."""
    found: set[str] = set()
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, str):
            found.add(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for value in node:
                walk(value)
        else:
            state = getattr(node, "__dict__", None)
            if isinstance(state, dict):
                walk(state)

    walk(obj)
    return found


def _field(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is not None:
        return value
    for source in (obj, getattr(obj, "__dict__", {}), getattr(obj, "payload", {})):
        if isinstance(source, dict) and name in source:
            return source[name]
    return None


# --- AR-IN-013: rl-stage declarations ---------------------------------------------------


@pytest.mark.parametrize("floor", [0, 0.0, -0.25, 1.5, True, "0.5", float("nan")])
def test_check_rl_spec_rejects_an_invalid_floor(floor: Any) -> None:
    problems = campaign.check_rl_spec(_rl_spec(rl_min_measured_fraction=floor))
    assert problems == [("AR-IN-013", f"rl_floor_invalid:{floor!r}")]


@pytest.mark.parametrize("floor", [0.01, 0.5, 1.0])
def test_check_rl_spec_accepts_a_valid_floor(floor: Any) -> None:
    assert campaign.check_rl_spec(_rl_spec(rl_min_measured_fraction=floor)) == []


def test_check_rl_spec_requires_a_declared_floor() -> None:
    problems = campaign.check_rl_spec(_rl_spec(rl_min_measured_fraction=None))
    assert [rule for rule, _ in problems] == ["AR-IN-013"]
    assert problems[0][1].startswith("rl_floor_undeclared")


@pytest.mark.parametrize(
    "confirm",
    [
        {"guardrails": ["m1"], "guardrail_directions": {"rl_measured_fraction": "max"}},
        {"guardrails": ["rl_measured_fraction"], "guardrail_directions": {"rl_measured_fraction": "min"}},
        {"guardrails": ["rl_measured_fraction"], "guardrail_directions": {}},
    ],
)
def test_check_rl_spec_requires_the_guardrail(confirm: dict[str, Any]) -> None:
    problems = campaign.check_rl_spec(_rl_spec(confirm=confirm))
    assert len(problems) == 1
    assert problems[0][0] == "AR-IN-013"
    assert problems[0][1].startswith("rl_guardrail_undeclared")


def test_check_rl_spec_requires_the_guardrail_metric() -> None:
    problems = campaign.check_rl_spec(_rl_spec(eval_policy={"metrics": ["m1"]}))
    assert [msg.split(":", 1)[0] for _, msg in problems] == ["rl_guardrail_metric_undeclared"]


def test_check_rl_spec_counts_an_rl_axis_as_the_rl_stage() -> None:
    spec = _rl_spec(stage=None, axes=[{"key": "rl.learning_rate"}], rl_min_measured_fraction=None)
    assert any(msg.startswith("rl_floor_undeclared") for _, msg in campaign.check_rl_spec(spec))


def test_check_rl_spec_is_quiet_without_the_rl_stage() -> None:
    spec: dict[str, Any] = {"stage": "sft", "confirm": {}, "eval_policy": {"metrics": ["m1"]}}
    assert campaign.check_rl_spec(spec) == []
    spec["rl_min_measured_fraction"] = 2.0  # a declared floor is validated even off the rl stage
    assert [msg.split(":", 1)[0] for _, msg in campaign.check_rl_spec(spec)] == ["rl_floor_invalid"]


def test_check_spec_carries_the_rl_findings() -> None:
    ok = _spec()
    ok.update(_rl_spec())
    ok["eval_policy"] = {"fingerprint": FINGERPRINT, "metrics": ["m1", "rl_measured_fraction"]}
    assert not [msg for rule, msg in campaign.check_spec(ok) if rule == "AR-IN-013"]
    undeclared = dict(ok, rl_min_measured_fraction=None)
    assert any(
        rule == "AR-IN-013" and msg.startswith("rl_floor_undeclared")
        for rule, msg in campaign.check_spec(undeclared)
    )


# --- acceptance: unmeasured rows dominate -----------------------------------------------


def _baseline() -> list[dict[str, Any]]:
    return [_row("base", 0, 1.0, role="baseline"), _row("base", 1, 1.2, role="baseline")]


def test_decide_measured_rows_stay_measurable() -> None:
    out = accept.decide(_spec(), _baseline(), [_row("c1", 0, 1.0), _row("c1", 1, 1.2)], "m1")
    assert out["verdict"] == "no_gain"
    assert not any(str(r).startswith("manifest_unmeasured:") for r in out["reasons"])


def test_decide_unmeasured_row_dominates() -> None:
    out = accept.decide(
        _spec(), _baseline(), [_row("c1", 0, 1.0), _row("c1", 1, 1.2, status="unmeasured")], "m1"
    )
    assert out["verdict"] == "unmeasured"
    assert out["reasons"] == ["manifest_unmeasured:c1:1"]
    assert not any(str(r).startswith("crash_excluded:") for r in out["reasons"])


def test_decide_multi_unmeasured_row_dominates() -> None:
    spec = _spec(objectives=[{"metric": "m1", "direction": "max"}])
    out = accept.decide_multi(
        spec, _baseline(), [_row("c1", 0, 1.0), _row("c1", 1, 1.2, status="unmeasured")]
    )
    assert out["verdict"] == "unmeasured"
    assert any("manifest_unmeasured:c1:1" in text for text in _texts(out))


# --- skill: claim refusal (AR-RS-009) and close (D3, AR-HO-010) -------------------------


def _skill_cls() -> Any:
    """The auto-research skill class (its module-local name is not part of the contract)."""
    for value in vars(skill_mod).values():
        if inspect.isclass(value) and "_close" in vars(value) and "_check_claim_request" in vars(value):
            return value
    raise AssertionError("auto-research skill class not found")


def _review() -> Any:
    """A skill instance whose measurement surface is quiet: nothing here runs on hardware."""

    class _Review(_skill_cls()):  # type: ignore[misc]
        def __init__(self) -> None:
            for args in ((), ("auto_research",)):
                try:
                    super().__init__(*args)  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - base wiring shape is version-defined
                    continue
                return

        def _measure(self, job: Any) -> Any:
            return None

        def _latest_envelope(self, ledger: Any, campaign: str) -> Any:
            return None

    return _Review()


class _Ledger:
    """Result rows only: every other ledger surface is empty here."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [dict(row) for row in rows]

    def results(self, campaign: str = "") -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def verify(self) -> list[str]:
        return []

    def launches(self, campaign: str = "") -> list[Any]:
        return []

    def entries(self) -> list[dict[str, Any]]:
        return []

    def head(self) -> dict[str, Any]:
        return {"count": 0, "head_hash": "0" * 64}

    def append(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def __getattr__(self, name: str) -> Any:  # unused surface -> empty listings
        return lambda *args, **kwargs: []


def _close(rows: list[dict[str, Any]], tmp_path: Any) -> Any:
    request = {"action": "close", "campaign": "m6-campaign", "stop_reason": "budget_exhausted"}
    return _review()._close(
        request, SkillContext(workdir=tmp_path), _Ledger(rows), _spec(), "m6-campaign", "sha256:" + "bb" * 32
    )


def test_claim_refused_when_the_confirm_set_is_asserted() -> None:
    rows = [
        _row("c1", 0, 1.05, evidence_class="asserted"),
        _row("base", 0, 1.0, role="baseline"),
    ]
    findings = _review()._check_claim_request({"trial": "c1"}, _spec(evidence_required=True), _Ledger(rows))
    texts = _texts(findings)
    assert "AR-RS-009" in texts
    assert "claim_set_asserted:c1" in texts


def test_legacy_campaign_claims_keep_m5_behaviour() -> None:
    rows = [
        _row("c1", 0, 1.05, evidence_class="asserted"),
        _row("base", 0, 1.0, role="baseline"),
    ]
    texts = _texts(_review()._check_claim_request({"trial": "c1"}, _spec(), _Ledger(rows)))
    assert "AR-RS-009" not in texts


def test_claim_refused_when_the_confirm_set_is_unmeasured() -> None:
    rows = [
        _row("c1", 0, 1.05, status="unmeasured", evidence_class="bound"),
        _row("base", 0, 1.0, role="baseline", evidence_class="bound"),
    ]
    findings = _review()._check_claim_request({"trial": "c1"}, _spec(), _Ledger(rows))
    texts = _texts(findings)
    assert "AR-RS-009" in texts
    assert "claim_set_unmeasured:c1" in texts


def test_claim_ignores_crash_rows_and_accepts_bound_rows(tmp_path: Any) -> None:
    rows = [
        _row("c1", 0, 1.05, evidence_class="bound", evidence=_evidence(tmp_path, drop_manifest=False)),
        _row("c1", 1, 1.05, evidence_class="bound", evidence=_evidence(tmp_path, drop_manifest=False)),
        _row("c1", 2, 0.0, status="crash"),
        _row("base", 0, 1.0, role="baseline", evidence_class="bound", evidence=_evidence(tmp_path, drop_manifest=False)),
    ]
    texts = _texts(_review()._check_claim_request({"trial": "c1"}, _spec(), _Ledger(rows)))
    assert all("AR-RS-009" not in text for text in texts)


def test_close_drops_the_named_evidence_that_disappeared(tmp_path: Any) -> None:
    saved = _evidence(tmp_path, drop_manifest=True)
    live = _evidence(tmp_path, drop_manifest=False)
    rows = [
        _row("base", 0, 1.0, role="baseline", evidence_class="bound", evidence=dict(live)),
        _row("base", 1, 1.2, role="baseline", evidence_class="bound", evidence=dict(live)),
        _row("c1", 0, 1.1, evidence_class="bound", evidence=dict(saved)),
    ]
    recorded = str(saved["run_manifests"][0]["sha256"]).split(":", 1)[1][:12]
    texts = _texts(_close(rows, tmp_path))
    assert any(text.startswith("evidence_file_missing:") and recorded in text for text in texts)
    assert not any(text.startswith("evidence_file_changed:") for text in texts)


def test_close_names_the_legacy_rows_with_ar_ho_010(tmp_path: Any) -> None:
    rows = [
        _row("base", 0, 1.0, role="baseline"),
        _row("base", 1, 1.2, role="baseline"),
        _row("c1", 0, 1.1, evidence_class="asserted"),
    ]
    texts = _texts(_close(rows, tmp_path))
    assert "AR-HO-010" in texts
    assert any("base:0" in text and "c1:0" in text for text in texts)


def test_close_status_survives_the_legacy_warning_alone(tmp_path: Any) -> None:
    live = _evidence(tmp_path, drop_manifest=False)
    rows = [_row("base", 0, 1.0, role="baseline"), _row("base", 1, 1.2, role="baseline"), _row("c1", 0, 1.1)]
    legacy = _close(rows, tmp_path)
    bound = _close([dict(row, evidence_class="bound", evidence=dict(live)) for row in rows], tmp_path)
    assert any("AR-HO-010" in text for text in _texts(legacy))
    assert not any("AR-HO-010" in text for text in _texts(bound))
    assert _field(legacy, "outcome") == _field(bound, "outcome")
    assert _field(legacy, "status") == _field(bound, "status")


def test_close_degrades_asserted_rows_where_evidence_is_required(tmp_path):
    """Review fix: an evidence-required (rl-stage) close never accepts on asserted rows; each is named as a drop."""
    import json
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_ar_skill_tests", Path(__file__).with_name("test_auto_research_skill.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    _materialize, _ctx = helpers._materialize, lambda path: SkillContext(workdir=path)
    from foundationskills.skills.auto_research.skill import AutoResearchSkill

    skill = AutoResearchSkill()
    request = _materialize(skill.must_fire_fixtures()["AR-HO-010"], tmp_path)
    result = skill.execute(request, _ctx(tmp_path))
    text = json.dumps(result.payload, default=str)
    assert "asserted_unbound:" in text
    assert result.payload.get("outcome") != "improved"
    assert "AR-HO-010" in [f.rule_id for f in result.findings]
