"""Unit tests for foundationskills.core.harness (C2 increments 1-2)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from foundationskills.core.harness import (
    SCHEMA_ID, UNMEASURED, HarnessManifest, REQUIRED, SkillPin, manifest_hash,
    manifest_payload, skill_pin, unmeasured_fields, validate_manifest,
)
HEX64 = "b1" * 32
OTHER64 = "c2" * 32
RESET = "fresh workspace; nothing carried across runs"
SNAPSHOT = "empty-workspace@v0"
PAYLOAD = {
    "schema": SCHEMA_ID,
    "skills": [{"name": "beta", "version": "1.0.0", "skill_md_sha256": HEX64},
               {"name": "alpha", "version": "2.0.0", "skill_md_sha256": None}],
    "routing_registry_version": UNMEASURED,
    "host_version": {"foundationskills": "0.1.0", "source_commit": "not_a_repository"},
    "model_pool": {"registry_sha256": UNMEASURED, "entries": ["z-model", "a-model"]},
    "substrate_image": "fs-substrate:main.sqsh", "seed": 7,
    "reset_semantics": RESET, "workspace_snapshot": SNAPSHOT,
}

def make_payload(**over):
    payload = json.loads(json.dumps(PAYLOAD))
    payload.update(over)
    return payload

def make_manifest(**over):
    fields = dict(
        skills=(SkillPin("beta", "1.0.0", HEX64), SkillPin("alpha", "2.0.0", None)), routing_registry_version=UNMEASURED,
        foundationskills_version="0.1.0", source_commit="not_a_repository", pool_registry_sha256=UNMEASURED,
        pool_entries=("z-model", "a-model"), substrate_image="fs-substrate:main.sqsh", seed=7,
        reset_semantics=RESET, workspace_snapshot=SNAPSHOT,
    )
    return HarnessManifest(**{**fields, **over})

def load_schema() -> dict:
    try:  # preferred: importlib.resources over the shipped package
        from importlib.resources import files
        blob = (files("foundationskills") / "schemas" / "harness_manifest.json").read_text()
    except Exception:  # fallback: a path relative to the package
        spec = importlib.util.find_spec("foundationskills")
        blob = (Path(spec.origin).resolve().parent / "schemas" / "harness_manifest.json").read_text()
    return json.loads(blob)

def only_problem(raw, needle):
    problems = validate_manifest(raw)
    assert len(problems) == 1, problems
    assert needle in problems[0], problems[0]
    return problems[0]

def test_schema_file_closes_objects_and_requires_manifest_keys():
    schema = load_schema()
    assert schema["properties"]["schema"]["const"] == SCHEMA_ID
    assert set(schema["required"]) == set(REQUIRED) | {"schema"}
    assert schema["additionalProperties"] is False

def test_manifest_payload_is_deterministic_and_sorted():
    one, two = manifest_payload(make_manifest()), manifest_payload(make_manifest())
    assert one == two
    assert [s["name"] for s in one["skills"]] == ["alpha", "beta"]
    assert one["model_pool"]["entries"] == ["a-model", "z-model"]
    assert "interpreter_sha256" not in one
    assert manifest_payload(make_manifest(interpreter_sha256=HEX64))["interpreter_sha256"] == HEX64

def test_manifest_hash_is_sixteen_hex_and_agrees_with_payload():
    manifest = make_manifest()
    assert re.fullmatch(r"[0-9a-f]{16}", manifest_hash(manifest))
    assert re.fullmatch(r"[0-9a-f]{16}", manifest_hash(manifest_payload(manifest)))
    assert manifest_hash(manifest) == manifest_hash(manifest_payload(manifest))

def test_manifest_hash_changes_when_any_field_changes():
    base = manifest_hash(make_manifest())
    assert manifest_hash(make_manifest(seed=8)) != base
    assert manifest_hash(make_manifest(substrate_image="fs-substrate:alt.sqsh")) != base
    resha = make_manifest(skills=(SkillPin("beta", "1.0.0", OTHER64), SkillPin("alpha", "2.0.0", None)))
    assert manifest_hash(resha) != base

def test_validate_manifest_accepts_a_good_payload():
    assert validate_manifest(make_payload()) == []

@pytest.mark.parametrize("key", tuple(REQUIRED) + ("schema",))
def test_validate_manifest_reports_each_missing_key(key):
    raw = make_payload()
    del raw[key]
    only_problem(raw, key)

DEFECTS = [
    ("unknown_key", {"bogus_extra": 1}, "bogus_extra"),
    ("bad_schema_id", {"schema": "fs.other.v9"}, "schema"),
    ("bad_sha", {"skills": [{"name": "beta", "version": "1", "skill_md_sha256": "abc"}]}, "skill_md_sha256"),
    ("skill_without_name", {"skills": [{"version": "1", "skill_md_sha256": None}]}, "name"),
    ("bool_seed", {"seed": True}, "seed"),
    ("empty_skills", {"skills": []}, "skills"),
    ("empty_reset_semantics", {"reset_semantics": ""}, "reset_semantics"),
    ("empty_workspace_snapshot", {"workspace_snapshot": ""}, "workspace_snapshot"),
]

@pytest.mark.parametrize("label,over,needle", DEFECTS, ids=[c[0] for c in DEFECTS])
def test_validate_manifest_reports_one_problem_per_defect(label, over, needle):
    only_problem(make_payload(**over), needle)

@pytest.mark.parametrize("garbage", [None, [], "junk", 3.5, object()])
def test_validate_manifest_never_raises_on_garbage(garbage):
    problems = validate_manifest(garbage)
    assert isinstance(problems, list) and problems

def test_unmeasured_fields_lists_both_honest_gaps():
    raw = make_payload()
    assert sorted(unmeasured_fields(raw)) == ["model_pool.registry_sha256", "routing_registry_version"]
    assert unmeasured_fields(make_payload(routing_registry_version="a" * 64)) == ["model_pool.registry_sha256"]

def test_skill_pin_hashes_markdown_and_reports_unmeasured_paths(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_bytes(b"# alpha\n")
    pin = skill_pin("alpha", "2.0.0", skill_md)
    assert isinstance(pin, SkillPin) and (pin.name, pin.version) == ("alpha", "2.0.0")
    assert pin.skill_md_sha256 == hashlib.sha256(b"# alpha\n").hexdigest()
    assert skill_pin("ghost", "0.0.1", None).skill_md_sha256 is None
    assert skill_pin("ghost", "0.0.1", tmp_path / "missing.md").skill_md_sha256 is None

def test_payload_validates_against_the_published_schema():
    from foundationskills.core.schema import SchemaError, assert_valid

    assert_valid(manifest_payload(make_manifest()), load_schema(), "harness manifest")  # no skip: FS's own validator
    with pytest.raises(SchemaError):
        assert_valid({**manifest_payload(make_manifest()), "extra": 1}, load_schema(), "harness manifest")
