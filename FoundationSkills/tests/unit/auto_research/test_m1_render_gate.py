"""Regression tests for the auto_research M1 render gate (G1 + G2).

G1 (defence in depth): run(action="submit") re-runs the IDENTICAL _check_submit_request gate that
check_inputs runs before anything is emitted or submitted - any finding returns REFUSED (refusal = the
first finding message, findings attached) with emit_trial, submit_trial, launch_fn and the runner
untouched, and an absent/empty envelope token never derives an expected trial token (no
trial_launch_token("", ...)).

G2 (gate what actually runs): the executed argv/sbatch is rendered from the opaque train_request /
eval_request, so (a) emit_trial forces train_request.nodes / .gpus_per_node to the trial_spec values
(naming each "overrode train_request.<key>") and (b) check_inputs scans json.dumps(train_request /
eval_request) and run() scans the rendered argv + sbatch with the SAME AR-LN-004 forbidden-command
patterns and AR-LN-005 quarantined-node names check_launch uses - any hit refuses with that rule id,
nothing submitted, nothing appended to the ledger.

The construction mirrors test_skill_m1.py / test_m1_review_fixes.py (_Rec-style recorded fakes against a
real AutoResearchSkill): nothing here touches the network or a subprocess.
"""
from __future__ import annotations

import foundationskills.skills.auto_research.skill as skill_module
from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.interfaces.fs.emit_trial import emit_trial as render_trial
from foundationskills.skills.auto_research.campaign import campaign_hash, scan_command_text, scan_rendered
from foundationskills.skills.auto_research.envelope import envelope_token, trial_launch_token
from foundationskills.skills.auto_research.ledger import ledger_files
from foundationskills.skills.auto_research.skill import AutoResearchSkill, _campaign, _spec

CLOCK_S = 1750000000.0
READY_PROBE = {"state": "ready", "reason": "fabric_ready", "ts": CLOCK_S}
FORBIDDEN_COMMAND = "AR-LN-004"
QUARANTINED_NODE = "AR-LN-005"


class _Rec:
    """Injected connector: records (args, kwargs) and returns result (never forks or connects)."""

    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    @property
    def count(self) -> int:
        return len(self.calls)


class _Wire:
    """Every connector of one skill as recorded fakes (emit_trial/submit_trial/trial_launch_token patched
    at the skill module symbol): a refusal must leave each of them untouched."""

    def __init__(self, monkeypatch, fact=None):
        self.runner = _Rec()
        self.launch_fn = _Rec()
        self.fabric_probe = _Rec(READY_PROBE)
        self.measure = _Rec(None)
        self.emit = _Rec(fact if fact is not None else _fact(_trial()))
        self.submit = _Rec({"job_id": "42424"})
        self.tokens: list[tuple[str, dict]] = []
        monkeypatch.setattr(skill_module, "emit_trial", self.emit)
        monkeypatch.setattr(skill_module, "submit_trial", self.submit)
        derived = trial_launch_token

        def _guard(token, trial_spec):
            assert token, "an absent/empty envelope token must never derive an expected trial token"
            self.tokens.append((token, trial_spec))
            return derived(token, trial_spec)

        monkeypatch.setattr(skill_module, "trial_launch_token", _guard)

    def skill(self) -> AutoResearchSkill:
        return AutoResearchSkill(
            runner=self.runner,
            launch_fn=self.launch_fn,
            fabric_probe=self.fabric_probe,
            measure=self.measure,
            clock=lambda: CLOCK_S,
        )

    def assert_nothing_submitted(self):
        """submit_trial/launch_fn/the runner must stay untouched on every refusal (G1 + G2)."""
        assert self.submit.count == 0
        assert self.launch_fn.count == 0
        assert self.runner.count == 0


def _ctx(path):
    return SkillContext(workdir=path)


def _stage(tmp_path, events):
    """Materialise ledger_files archive bytes under tmp_path (same harness as test_skill_m1)."""
    for rel, text in ledger_files(events).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path / "ledger")


def _approved(spec):
    campaign = _campaign(spec)
    return [("campaign_approved", campaign, "-", {"spec_hash": campaign_hash(spec), "approver": "reviewer"})]


def _env(spec):
    env = {"budget": {"max_runs": 6, "gpu_hours_total": 24.0}, "scope": "one campaign"}
    return {
        "envelope_token": envelope_token(campaign_hash(spec), env),
        "spec_hash": campaign_hash(spec),
        "envelope": env,
        "budget": dict(env["budget"]),
        "approver": "reviewer",
    }


def _trial(**overrides):
    trial = {
        "trial": "t-sub", "role": "candidate", "kind": "train", "delta": {"optim.lr": 5e-4},
        "seed": 101, "nodes": 1, "gpus_per_node": 8, "partition": "rally", "time": "10-00:00:00",
        "commands": ["uv run train.py --config campaign.yaml"], "gpu_hours_est": 4.0,
        "train_request": {"config": "campaign.yaml"},
    }
    trial.update(overrides)
    return trial


def _request(tmp_path, spec, **overrides):
    request = {
        "campaign_spec": spec,
        "campaign_confirm": campaign_hash(spec),
        "approver": "reviewer",
        "ledger_dir": str(tmp_path / "ledger"),
    }
    request.update(overrides)
    return request


def _submit_request(tmp_path, spec, envelope, trial, **overrides):
    return _request(
        tmp_path, spec, action="submit", trial_spec=trial,
        launch_token=trial_launch_token(envelope["envelope_token"], trial), **overrides,
    )


def _fact(trial, argv=("uv", "run", "train.py"), sbatch="--time=10-00:00:00\n"):
    """One emitted trial fact exactly as emit_trial returns it (the render is what gets gated)."""
    spec = {"argv": list(argv), "sbatch": sbatch, "notes": [], "drops": [], "executable": True, "missing": []}
    return {
        "fs_launch_spec": spec, "trial_spec": trial, "confirm": "sha256:" + "c3" * 32,
        "notes": [], "drops": [], "executable": True, "missing": [],
    }


def _ledger_lines(tmp_path):
    chain = tmp_path / "ledger" / "chain.jsonl"
    return chain.read_text(encoding="utf-8").splitlines() if chain.exists() else []


def test_g1_run_without_envelope_refuses_before_any_connector(tmp_path, monkeypatch):
    """G1: run() re-runs the full submit gate - no envelope -> REFUSED with every fake untouched."""
    spec = _spec()
    _stage(tmp_path, _approved(spec))
    wire = _Wire(monkeypatch)
    request = _request(
        tmp_path, spec, action="submit", trial_spec=_trial(),
        launch_token="fs-ar-trial-v1-not-derived",  # with no envelope no derived token can match
    )
    before = _ledger_lines(tmp_path)

    result = wire.skill().run(request, _ctx(tmp_path))

    assert result.status == Status.REFUSED
    assert result.refusal
    assert "AR-LN-006" in [f.rule_id for f in result.findings]
    wire.assert_nothing_submitted()
    assert wire.emit.count == 0  # G1: emit_trial/submit_trial/launch_fn/runner never run on a gate refusal
    assert wire.tokens == []  # and no trial_launch_token("", ...) was ever derived
    assert _ledger_lines(tmp_path) == before


def test_g2_forbidden_command_refused_at_check_inputs_and_run(tmp_path, monkeypatch):
    """G2: a forbidden command inside the opaque train_request refuses at input time AND at run()."""
    spec = _spec()
    env = _env(spec)
    _stage(tmp_path, [*_approved(spec), ("launch_envelope", _campaign(spec), "-", env)])
    trial = _trial(train_request={"config": "campaign.yaml", "command": ["pkill", "-u", "victim"]})
    wire = _Wire(monkeypatch, fact=_fact(trial, argv=("pkill", "-u", "victim")))
    request = _submit_request(tmp_path, spec, env, trial)
    skill = wire.skill()

    assert FORBIDDEN_COMMAND in [f.rule_id for f in skill.check_inputs(request, _ctx(tmp_path))]

    before = _ledger_lines(tmp_path)
    result = skill.run(request, _ctx(tmp_path))

    assert result.status == Status.REFUSED
    assert FORBIDDEN_COMMAND in [f.rule_id for f in result.findings]
    wire.assert_nothing_submitted()
    assert wire.emit.count == 0  # the G1 gate refuses before emit_trial ever renders the bad argv
    assert _ledger_lines(tmp_path) == before


def test_g2_rendered_argv_forbidden_command_refuses_before_submit(tmp_path, monkeypatch):
    """G2: the RENDERED argv is what runs - a forbidden command there is refused before submit_trial."""
    spec = _spec()
    env = _env(spec)
    _stage(tmp_path, [*_approved(spec), ("launch_envelope", _campaign(spec), "-", env)])
    trial = _trial()
    wire = _Wire(monkeypatch, fact=_fact(trial, argv=("pkill", "-u", "victim")))
    request = _submit_request(tmp_path, spec, env, trial)
    before = _ledger_lines(tmp_path)

    result = wire.skill().run(request, _ctx(tmp_path))

    assert result.status == Status.REFUSED
    assert FORBIDDEN_COMMAND in [f.rule_id for f in result.findings]
    wire.assert_nothing_submitted()
    assert wire.emit.count == 1  # the render scan sits AFTER emit_trial and BEFORE submit_trial
    assert _ledger_lines(tmp_path) == before


def test_g2_rendered_sbatch_quarantined_node_refused_as_ar_ln_005(tmp_path, monkeypatch):
    """G2: a quarantined node named in the rendered sbatch is refused as AR-LN-005, nothing submitted."""
    spec = _spec()
    env = _env(spec)
    _stage(tmp_path, [*_approved(spec), ("launch_envelope", _campaign(spec), "-", env)])
    trial = _trial()
    sbatch = "#!/bin/bash\n#SBATCH --time=10-00:00:00\n#SBATCH --nodelist=r01dgx02\n"
    wire = _Wire(monkeypatch, fact=_fact(trial, sbatch=sbatch))
    request = _submit_request(tmp_path, spec, env, trial)
    before = _ledger_lines(tmp_path)

    result = wire.skill().run(request, _ctx(tmp_path))

    assert result.status == Status.REFUSED
    assert result.findings[0].rule_id == QUARANTINED_NODE  # rendered node gate -> AR-LN-005
    assert "AR-LN-005" in result.refusal
    wire.assert_nothing_submitted()
    assert _ledger_lines(tmp_path) == before


def test_g2_train_request_nodes_forced_to_trial_spec_with_note():
    """G2(a): the rendered train shape is never taken from the opaque train_request (named overrides)."""
    captured: dict = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"argv": ["uv", "run", "train.py"], "sbatch": "--time=10-00:00:00\n#SBATCH --partition=rally\n",
                "notes": [], "drops": [], "executable": True, "missing": []}

    trial = _trial(kind="train", nodes=1, gpus_per_node=8,
                   train_request={"config": "campaign.yaml", "nodes": 2, "gpus_per_node": 1})
    fact = render_trial({"trial_spec": trial, "campaign_spec": _spec()}, emit_train_fn=fake_train)

    assert fact["executable"] is True
    assert captured["nodes"] == 1 and captured["gpus_per_node"] == 8  # forced to the trial_spec values
    assert fact["notes"] == ["overrode train_request.nodes", "overrode train_request.gpus_per_node"]
    assert "overrode train_request.nodes" in fact["fs_launch_spec"]["notes"]

    # no carried value -> forced silently, nothing to name
    captured.clear()
    fact = render_trial({"trial_spec": _trial(kind="train"), "campaign_spec": _spec()}, emit_train_fn=fake_train)
    assert fact["notes"] == []
    assert captured["nodes"] == 1 and captured["gpus_per_node"] == 8


def test_g2_scan_helpers_map_forbidden_commands_and_nodes_to_their_rules():
    """The shared render scan: forbidden commands -> AR-LN-004, quarantined nodes -> AR-LN-005."""
    assert scan_command_text('{"command": ["pkill", "-u", "victim"]}')[0][0] == FORBIDDEN_COMMAND
    assert scan_command_text('{"command": ["eval", "run", "gsm8k"]}') == []
    safe = scan_rendered({"argv": ["uv", "run", "train.py"]},
                         "#SBATCH --time=10-00:00:00\n#SBATCH --exclude=r01dgx02\n")
    assert safe == []  # an exclude option NAMES the node to skip - it never refuses the render
    hits = scan_rendered({"argv": ["pkill", "-u", "victim"]},
                         "--time=10-00:00:00\n#SBATCH --nodelist=r01dgx02\n")
    assert [rule for rule, _ in hits] == [FORBIDDEN_COMMAND, QUARANTINED_NODE]
