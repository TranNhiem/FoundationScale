
"""End-to-end DataEngineSkill tests, must-fire fixtures, and SKILL.md coverage."""
from __future__ import annotations

from importlib import resources

import pytest

import foundationskills.skills.data_engine.ops  # noqa: F401 - register real ops
from foundationskills.core.artifacts import read_artifact
from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills import register_builtin_skills
from foundationskills.skills.data_engine.phase2 import PHASE2
from foundationskills.skills.data_engine.skill import DataEngineSkill

CORPUS_LINES = "".join(f'{{"id": "d{i}", "text": "fixture document number {i} with some body text"}}\n' for i in range(6))


def _ctx(tmp_path):
    return SkillContext(workdir=tmp_path)


def _materialize(fixture: dict, tmp_path):
    for rel, content in fixture.get("files", {}).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def substitute(value):
        if isinstance(value, str):
            return value.replace("{tmp}", str(tmp_path))
        if isinstance(value, list):
            return [substitute(v) for v in value]
        if isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        return value

    return substitute(fixture["request"])


class TestSkillEndToEnd:
    def test_execute_writes_readable_artifacts(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(CORPUS_LINES, encoding="utf-8")
        skill = DataEngineSkill()
        result = skill.execute(
            {
                "target_format": "cpt",
                "sources": [{"uri": str(corpus), "kind": "jsonl"}],
                "requirements": {"require_decontam": False, "require_dedup": True},
            },
            _ctx(tmp_path),
        )
        assert result.status in {Status.PASS, Status.UNMEASURED}
        assert not result.findings or all(f.rule_id != "DE-HO-003" for f in result.findings)
        artifact_types = set()
        for ref in result.artifacts:
            artifact = read_artifact(ref.path)  # validates payload + type
            artifact_types.add(artifact.type)
        assert {"data_pipeline_spec", "dataset", "readiness_report"} <= artifact_types
        dataset = next(read_artifact(r.path).payload for r in result.artifacts if r.type == "dataset")
        assert dataset["num_records"] > 0
        readiness = next(read_artifact(r.path).payload for r in result.artifacts if r.type == "readiness_report")
        status_by_verdict = {"PASS": Status.PASS, "UNMEASURED": Status.UNMEASURED, "RED": Status.RED}
        assert result.status == status_by_verdict[readiness["verdict"]]

    def test_explicit_phase2_pipeline_refuses_through_execute(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(CORPUS_LINES, encoding="utf-8")
        skill = DataEngineSkill()
        result = skill.execute(
            {
                "target_format": "pretrain",
                "sources": [{"uri": str(corpus), "kind": "jsonl"}],
                "pipeline": {
                    "target_format": "pretrain",
                    "seed": 0,
                    "tokenizer": None,
                    "rationale": [],
                    "ops": [
                        {"op": "ingest", "config": {"sources": [{"uri": str(corpus), "kind": "jsonl"}]}},
                        {"op": "synthesize", "config": {}},
                    ],
                },
            },
            _ctx(tmp_path),
        )
        assert result.status is Status.REFUSED
        assert result.refusal and "phase-2: synthesize" in result.refusal


class TestMustFireFixtures:
    def test_fixtures_cover_every_declared_rule(self):
        skill = DataEngineSkill()
        fixtures = skill.must_fire_fixtures()
        assert set(fixtures) == {r.rule_id for r in skill.rules}

    @pytest.mark.parametrize("rule_id", [r.rule_id for r in DataEngineSkill.rules])
    def test_each_fixture_fires_its_rule(self, tmp_path, rule_id):
        skill = DataEngineSkill()
        fixture = skill.must_fire_fixtures()[rule_id]
        request = _materialize(fixture, tmp_path)
        result = skill.execute(request, _ctx(tmp_path))
        fired = [f.rule_id for f in result.findings]
        assert rule_id in fired, f"{rule_id} did not fire (fired: {fired}, status: {result.status})"

    def test_phase2_fixture_names_capability(self, tmp_path):
        skill = DataEngineSkill()
        request = _materialize(skill.must_fire_fixtures()["DE-IN-004"], tmp_path)
        result = skill.execute(request, _ctx(tmp_path))
        assert result.status is Status.REFUSED
        assert "phase-2: semantic_dedup" in result.refusal


class TestSkillRegistration:
    def test_register_builtin_skills_idempotent(self):
        from foundationskills.core.registry import SkillRegistry

        registry = SkillRegistry()
        registered = register_builtin_skills(registry)
        assert "data_engine" in registered
        again = register_builtin_skills(registry)
        assert "data_engine" not in again  # skipped, still present
        assert "data_engine" in registry.names()

    def test_phase2_names_match(self):
        assert set(PHASE2) == {"semantic_dedup", "synthesize", "toolcall_format", "video_ingest"}


class TestSkillDoc:
    def test_skill_md_lists_every_rule_id(self):
        skill = DataEngineSkill()
        doc = (
            resources.files("foundationskills.skills.data_engine")
            .joinpath("SKILL.md")
            .read_text(encoding="utf-8")
        )
        declared = [r.rule_id for r in skill.rules] + [f"DE-RDY-{i:03d}" for i in range(1, 11)]
        for rule_id in declared:
            assert rule_id in doc, f"{rule_id} missing from SKILL.md"

    def test_skill_md_headings(self):
        doc = (
            resources.files("foundationskills.skills.data_engine")
            .joinpath("SKILL.md")
            .read_text(encoding="utf-8")
        )
        for heading in (
            "## Purpose",
            "## When to use",
            "## Inputs",
            "## Outputs",
            "## Scope",
            "## Validation rules",
            "## Failure handling",
            "## FS interface",
            "## Worked examples",
            "## Phase 2",
        ):
            assert heading in doc
