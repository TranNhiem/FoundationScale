
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.artifacts import Artifact, ArtifactRef, read_artifact, verify_ref, write_artifact
from foundationskills.core.contract import (
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    Scope,
    Severity,
    SkillContext,
    SkillResult,
    RuleSpec,
)
from foundationskills.core.orchestrator import ConfirmationRequired, Orchestrator, Step, plan_hash, require_confirmation
from foundationskills.core.provenance import canonical_json, make_provenance, sha256_json
from foundationskills.core.registry import SkillRegistry
from foundationskills.core.schema import SchemaError, assert_valid, validate
from foundationskills.core.status import EXIT_CODES, Status, worst


def ctx(tmp_path: Path) -> SkillContext:
    return SkillContext(workdir=tmp_path)


def scope() -> Scope:
    return Scope(model_types=("llm",), families="any", stages=("sft",), algorithms=(), methods=("full",), hardware=())


class ToySkill(BaseSkill):
    name = "toy"
    version = "0.0.1"
    description = "toy"
    scope = scope()
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ("toy_out",)
    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["mode"],
        "properties": {"mode": {"type": "string"}, "bad": {"type": "boolean"}},
        "additionalProperties": True,
    }
    output_schema: dict[str, Any] = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    rules = (
        RuleSpec("TOY-IN-001", "bad input is forbidden", Severity.BLOCK, "input"),
        RuleSpec("TOY-OUT-001", "bad handoff is forbidden", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(entries=("foundationscale-train",), emits=("toy_out",), apis=("foundationscale.train.cli",))

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        return [self.finding("TOY-IN-001", "bad=true is refused")] if request.get("bad") else []

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        mode = request.get("mode")
        if mode == "boom":
            raise RuntimeError("explode")
        if mode == "bad_output":
            return SkillResult(Status.PASS, {"ok": "yes"})
        if mode == "bad_handoff":
            return SkillResult(Status.PASS, {"ok": True, "handoff_bad": True})
        if mode == "undeclared":
            return SkillResult(Status.PASS, {"ok": True}, findings=(Finding("NOPE-001", Severity.WARN, "not declared"),))
        if mode == "artifact":
            ref = ArtifactRef("dataset", "d1", str(ctx.workdir / "dataset.d1.json"), "0" * 64)
            return SkillResult(Status.PASS, {"ok": True}, artifacts=(ref,))
        return SkillResult(Status.PASS, {"ok": True})

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        if result.payload.get("handoff_bad"):
            return [self.finding("TOY-OUT-001", "handoff payload is not transferable")]
        return []

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        if isinstance(failure, BaseException):
            return Diagnosis("run exception", ("bug in toy run",), ("read CORE-EXC",), ("fix and retry",))
        return Diagnosis("bad result", (), (), ())

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "TOY-IN-001": {"phase": "input", "request": {"mode": "ok", "bad": True}},
            "TOY-OUT-001": {"phase": "handoff", "result_payload": {"ok": True, "handoff_bad": True}},
        }


def test_status_mapping_and_worst() -> None:
    assert EXIT_CODES == frozenset({0, 5, 95, 96})
    assert Status.PASS.exit_code == 0
    assert Status.RED.exit_code == 5
    assert Status.UNMEASURED.exit_code == 95
    assert Status.REFUSED.exit_code == 96
    assert Status.from_exit_code(96) is Status.REFUSED
    with pytest.raises(ValueError):
        Status.from_exit_code(1)
    assert worst() is Status.UNMEASURED
    assert worst(Status.PASS, Status.UNMEASURED) is Status.UNMEASURED
    assert worst(Status.PASS, Status.RED) is Status.RED
    assert worst(Status.UNMEASURED, Status.REFUSED, Status.RED) is Status.REFUSED


def test_schema_each_keyword_and_bool_not_integer() -> None:
    schema = {
        "type": "object",
        "required": ["i", "s", "a"],
        "properties": {
            "i": {"type": "integer", "minimum": 2, "maximum": 5, "exclusiveMinimum": 1},
            "s": {"type": "string", "minLength": 2, "pattern": "^a+$"},
            "a": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "number"}},
            "e": {"enum": ["x", "y"]},
            "c": {"const": 7},
            "n": {"type": "null"},
        },
        "additionalProperties": False,
    }
    good = {"i": 3, "s": "aaa", "a": [1.5], "e": "x", "c": 7, "n": None}
    assert validate(good, schema) == []
    bad = dict(good, i=True)
    assert any("$.i" in e and "integer" in e for e in validate(bad, schema))
    assert validate(dict(good, s="b", extra=1), schema)
    assert validate(dict(good, a=[]), schema)
    assert validate(dict(good, e="z"), schema)
    assert validate(dict(good, c=8), schema)


def test_schema_ref_oneof_anyof_and_assert_valid() -> None:
    schema = {
        "type": "object",
        "properties": {
            "node": {"$ref": "#/$defs/node"},
            "choice": {"oneOf": [{"type": "integer"}, {"type": "number"}]},
            "any": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "$defs": {"node": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
    }
    assert validate({"node": {"name": "x"}, "choice": 1.5, "any": "s"}, schema) == []
    errors = validate({"node": {}, "choice": 1, "any": 1.5}, schema)  # 1 matches BOTH branches
    assert any("$.node.name" in e for e in errors)
    assert any("$.choice" in e and "oneOf" in e for e in errors)
    assert any("$.any" in e and "anyOf" in e for e in errors)
    with pytest.raises(SchemaError):
        assert_valid({"node": {}, "choice": True, "any": 1.5}, schema, "fixture")


def test_artifact_atomic_roundtrip_and_tamper(tmp_path: Path) -> None:
    payload = {
        "format": "sft",
        "shards": [{"path": "a.jsonl", "sha256": "1" * 64, "records": 2}],
        "schema": {"columns": ["text"]},
        "num_records": 2,
        "num_tokens": 10,
        "tokenizer": "tok",
        "chat_template_family": "chatml",
        "fs_columns": {"text_column": "text", "image_column": None, "gold_key": None},
    }
    prov = make_provenance("toy", "0.0.1", {"request": sha256_json({"x": 1})})
    artifact = Artifact("dataset", "d1", payload, prov)
    assert artifact.content_hash() == sha256_json({"type": "dataset", "id": "d1", "payload": payload, "schema_version": 1})
    ref = write_artifact(artifact, tmp_path)
    assert Path(ref.path).name == "dataset.d1.json"
    assert verify_ref(ref)
    loaded = read_artifact(ref.path, expect_type="dataset")
    assert loaded.payload["num_records"] == 2
    with open(ref.path, "a", encoding="utf-8") as handle:
        handle.write("tamper")
    assert not verify_ref(ref)


def test_artifact_schema_error_and_unmeasured_type(tmp_path: Path) -> None:
    bad = Artifact("dataset", "bad", {"format": "nope"}, make_provenance("toy", "0"))
    with pytest.raises(SchemaError):
        write_artifact(bad, tmp_path)
    unknown = Artifact("future_type", "u", {}, make_provenance("toy", "0"))
    assert unknown.validate() == ["future_type: no schema registered (unmeasured)"]


def test_skill_result_invariants() -> None:
    with pytest.raises(ValueError):
        SkillResult(Status.REFUSED, {})
    with pytest.raises(ValueError):
        SkillResult(Status.PASS, {}, refusal="no")
    with pytest.raises(ValueError):
        SkillResult(Status.PASS, {}, findings=(Finding("A-1", Severity.BLOCK, "x"),))
    ok = SkillResult(Status.RED, {}, findings=(Finding("A-1", Severity.BLOCK, "x"),))
    assert ok.exit_code == 5 and ok.to_dict()["status"] == "RED"


def test_execute_all_core_paths(tmp_path: Path) -> None:
    toy = ToySkill()
    assert set(toy.must_fire_fixtures()) == {r.rule_id for r in toy.rules}
    refused_schema = toy.execute({"bad": False}, ctx(tmp_path))
    assert refused_schema.status is Status.REFUSED and "invalid request" in (refused_schema.refusal or "")
    refused_input = toy.execute({"mode": "ok", "bad": True}, ctx(tmp_path))
    assert refused_input.status is Status.REFUSED and "TOY-IN-001" in (refused_input.refusal or "")
    red_exc = toy.execute({"mode": "boom"}, ctx(tmp_path))
    assert red_exc.status is Status.RED
    assert any(f.rule_id == "CORE-EXC" for f in red_exc.findings)
    assert red_exc.diagnosis and red_exc.diagnosis.symptom == "run exception"
    red_out = toy.execute({"mode": "bad_output"}, ctx(tmp_path))
    assert red_out.status is Status.RED and any(f.rule_id == "CORE-OUT-SCHEMA" for f in red_out.findings)
    red_handoff = toy.execute({"mode": "bad_handoff"}, ctx(tmp_path))
    assert red_handoff.status is Status.RED and any(f.rule_id == "TOY-OUT-001" for f in red_handoff.findings)
    red_unknown = toy.execute({"mode": "undeclared"}, ctx(tmp_path))
    assert red_unknown.status is Status.RED and any(f.rule_id == "CORE-UNDECLARED-RULE" for f in red_unknown.findings)
    passed = toy.execute({"mode": "ok"}, ctx(tmp_path))
    assert passed.status is Status.PASS and passed.exit_code == 0


def test_registry_duplicate_and_plan_chain(tmp_path: Path) -> None:
    registry = SkillRegistry()
    a, b, c = ToySkill(), ToySkill(), ToySkill()
    a.name, a.produces = "a", ("goal_spec",)
    b.name, b.consumes, b.produces = "b", ("goal_spec",), ("training_plan",)
    c.name, c.consumes, c.produces = "c", ("training_plan",), ("fs_launch_spec",)
    registry.register(a)
    with pytest.raises(ValueError):
        registry.register(a)
    registry.register(b)
    registry.register(c)
    assert registry.names() == ["a", "b", "c"]
    assert registry.producers_of("training_plan") == ["b"]
    assert registry.consumers_of("goal_spec") == ["b"]
    assert registry.plan_chain({"training_plan"}, "training_plan") == []
    assert registry.plan_chain(set(), "fs_launch_spec") == ["a", "b", "c"]
    with pytest.raises(LookupError):
        registry.plan_chain({"zzz"}, "nope")
    with pytest.raises(KeyError) as missing:
        registry.get("nope")
    assert "a, b, c" in str(missing.value)


class ConsumerSkill(ToySkill):
    name = "consumer"
    consumes = ("dataset",)
    produces = ()

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        return SkillResult(Status.PASS, {"ok": True, "input_types": [i["type"] for i in request.get("inputs", [])]})


class UnmeasuredSkill(ToySkill):
    name = "unmeasured"
    produces = ()

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        return SkillResult(Status.UNMEASURED, {"ok": True})


class RedSkill(ToySkill):
    name = "red"
    produces = ()

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        return SkillResult(Status.RED, {"ok": False})


def test_orchestrator_injection_stop_and_journal(tmp_path: Path) -> None:
    registry = SkillRegistry()
    producer, consumer, later = ToySkill(), ConsumerSkill(), ToySkill()
    producer.name = "producer"
    producer.default_mode = "artifact"  # instance attr is harmless; request controls mode
    later.name = "later"
    for skill in (producer, consumer, later):
        registry.register(skill)
    orch = Orchestrator(registry, ctx(tmp_path))
    steps = [
        Step("producer", {"mode": "artifact"}),
        Step("consumer", {"mode": "ok"}),
        Step("later", {"mode": "ok"}),
    ]
    results = orch.run(steps)
    assert [r.status for r in results] == [Status.PASS, Status.PASS, Status.PASS]
    assert results[1].payload["input_types"] == ["dataset"]
    lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["skill"] == "producer" and first["status"] == "PASS" and first["exit_code"] == 0

    registry2 = SkillRegistry()
    red, after = RedSkill(), ToySkill()
    after.name = "after"
    registry2.register(red)
    registry2.register(after)
    stopped = Orchestrator(registry2, ctx(tmp_path / "w2")).run([Step("red", {"mode": "ok"}), Step("after", {"mode": "ok"})])
    assert [r.status for r in stopped] == [Status.RED]


def test_orchestrator_allow_unmeasured_and_final_status(tmp_path: Path) -> None:
    registry = SkillRegistry()
    unmeasured, after = UnmeasuredSkill(), ToySkill()
    after.name = "after"
    registry.register(unmeasured)
    registry.register(after)
    orch = Orchestrator(registry, ctx(tmp_path))
    results = orch.run([Step("unmeasured", {"mode": "ok"}, allow_unmeasured=True), Step("after", {"mode": "ok"})])
    assert [r.status for r in results] == [Status.UNMEASURED, Status.PASS]
    assert Orchestrator.final_status(results) is Status.UNMEASURED
    blocked = orch.run([Step("unmeasured", {"mode": "ok"}), Step("after", {"mode": "ok"})])
    assert [r.status for r in blocked] == [Status.UNMEASURED]


def test_confirmation_gate() -> None:
    plan = {"steps": [{"skill": "x"}]}
    token = plan_hash(plan)
    with pytest.raises(ConfirmationRequired) as exc:
        require_confirmation(plan, None)
    assert token in str(exc.value) and "never auto-fill" in str(exc.value)
    with pytest.raises(ConfirmationRequired):
        require_confirmation(plan, "deadbeefdeadbeef")
    require_confirmation(plan, token)


def test_provenance_and_canonical_json_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FS_COMMIT", raising=False)
    prov = make_provenance("s", "1", {"in": "0" * 64})
    assert prov.fs_commit is None and prov.skill == "s"
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'


def test_execute_keeps_input_warnings_and_propagates_interrupts(tmp_path) -> None:
    """Input WARN findings must reach the result; KeyboardInterrupt must not become RED."""
    from foundationskills.core.contract import BaseSkill, FSInterface, RuleSpec, Scope, Severity, SkillContext, SkillResult
    from foundationskills.core.status import Status

    class Warny(BaseSkill):
        name = "warny"; version = "0.0.1"; description = "t"
        scope = Scope(("llm",), "any", ("sft",), (), ("full",), ())
        consumes = (); produces = ()
        input_schema = {"type": "object"}; output_schema = {"type": "object"}
        rules = (RuleSpec("TW-WARN-001", "warn", Severity.WARN, "input"),)
        fs_interface = FSInterface((), (), ())
        def check_inputs(self, request, ctx): return [self.finding("TW-WARN-001", "heads up")]
        def run(self, request, ctx):
            if request.get("interrupt"): raise KeyboardInterrupt
            return SkillResult(Status.PASS, {})
        def check_handoff(self, result, ctx): return []
        def diagnose(self, failure): raise RuntimeError("broken diagnose")
        def must_fire_fixtures(self): return {"TW-WARN-001": {"phase": "input", "request": {}}}

    ctx = SkillContext(workdir=tmp_path)
    res = Warny().execute({}, ctx)
    assert res.status is Status.PASS and [f.rule_id for f in res.findings] == ["TW-WARN-001"]
    with pytest.raises(KeyboardInterrupt):
        Warny().execute({"interrupt": True}, ctx)
