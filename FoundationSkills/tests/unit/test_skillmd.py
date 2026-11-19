"""Unit tests for foundationskills.agent.skillmd (C1 increment 1)."""
from __future__ import annotations

import hashlib

import pytest

from foundationskills.agent import skillmd as sm
from foundationskills.cli import build_parser
from foundationskills.core.contract import Severity
from foundationskills.core.status import Status
from foundationskills.skills.compression.skill import CompressionSkill

SEVERITIES = {sev.value for sev in Severity}


class _Iface:
    entries = ("python api only",)


class _FakeSkill:
    """Minimal skill object: Python API only, no CLI entry."""

    name = "fake-skill"
    fs_interface = _Iface()
    produces = ()
    rules = ()


def _rule_fields(skill) -> dict:
    """(rule_id -> severity) from a skill's RuleSpecs, dict- or attr-shaped."""
    found = {}
    for attr in ("rules", "rule_specs", "specs"):
        for spec in getattr(skill, attr, ()) or ():
            if isinstance(spec, dict):
                rid, sev = spec.get("rule_id"), spec.get("severity")
            else:
                rid, sev = getattr(spec, "rule_id", None), getattr(spec, "severity", None)
            if rid:
                found[rid] = sev
    return found


@pytest.fixture(scope="module")
def skill():
    return CompressionSkill()


@pytest.fixture(scope="module")
def parser():
    return build_parser()


@pytest.fixture(scope="module")
def model(skill, parser):
    return sm.usage_model(skill, parser)


def test_cli_path_and_subparser(skill, parser):
    assert sm.cli_path(skill) == ["compress"]
    assert sm.subparser_for(parser, ["compress"]) is not None
    assert sm.subparser_for(parser, ["nope"]) is None


def test_iter_commands_verbs_and_plan_flags(parser):
    cmds = sm.iter_commands(parser, ["compress"])
    assert [c["verb"] for c in cmds] == [
        "fskills compress check",
        "fskills compress plan",
        "fskills compress verify",
    ]
    plan = next(c for c in cmds if c["verb"].endswith(" plan"))
    names = [f["long"] for f in plan["flags"]]
    assert names == sorted(names) and "-h" not in names and "--help" not in names
    flags = {f["long"]: f for f in plan["flags"]}
    assert flags["--text-file"]["required"] is True
    assert flags["--budget-ppl-ratio"]["required"] is False
    assert set(flags["--text-file"]) == {"long", "required", "choices", "default"}


def test_usage_model_shape_and_rule_severities(skill, model):
    assert model["name"] == skill.name
    assert model["artifacts"] == sorted(skill.produces)
    assert model["exit_codes"] == sm.EXIT_CODES
    rules = {r["rule_id"]: r for r in model["rules"]}
    assert list(rules) == sorted(rules)
    sev_of = _rule_fields(skill)
    for rid in ("CP-IN-004", "CP-HO-005"):
        rule = rules[rid]
        assert rule["severity"] == sev_of[rid]
        assert rule["severity"] in SEVERITIES
        assert rule["phase"] and rule["description"]


def test_render_block_deterministic_and_wrapped(model):
    first, second = sm.render_block(model), sm.render_block(model)
    assert first == second
    assert first.startswith(sm.BEGIN) and first.endswith(sm.END + "\n")
    assert "## Usage (generated)" in first and "### Rules" in first
    assert "### Artifacts" in first and "### Exit codes" in first
    assert "PASS 0 · RED 5 · UNMEASURED 95 · REFUSED 96" in first
    assert "(required)" in first
    assert sm.block_sha256(first) == hashlib.sha256(first.encode("utf-8")).hexdigest()


def test_exit_codes_match_status():
    assert set(sm.EXIT_CODES) == {s.name for s in Status}
    for status in Status:
        assert sm.EXIT_CODES[status.name] == status.exit_code


def test_extract_replace_roundtrip(model):
    block = sm.render_block(model)
    doc = "# Compression skill\n\n## Validation rules\n\n- budgets hold\n"
    out = sm.replace_block(doc, block)
    assert sm.extract_block(out) == block
    assert out.index(block) < out.index("## Validation rules")
    other = f"{sm.BEGIN}\nreplacement body\n{sm.END}\n"
    out2 = sm.replace_block(out, other)
    assert sm.extract_block(out2) == other
    assert out2.count(sm.BEGIN) == 1 and out2.count(sm.END) == 1
    tail = sm.replace_block("plain notes\n", block)
    assert sm.extract_block(tail) == block and tail.index(block) > tail.index("plain notes")


@pytest.mark.parametrize(
    "text",
    [
        f"{sm.BEGIN}\nopen only\n",
        f"{sm.END}\nclose only\n",
        f"{sm.BEGIN}\n{sm.END}\n{sm.END}\n",
        f"{sm.BEGIN}\n{sm.BEGIN}\n{sm.END}\n",
    ],
)
def test_extract_unbalanced_markers_raise(text):
    with pytest.raises(ValueError):
        sm.extract_block(text)


def test_check_missing_stale_fresh(skill, parser, model):
    doc = "# Compression skill\n\n## Validation rules\n\n- budgets hold\n"
    assert sm.check(skill, parser, doc) == ["missing generated usage block"]
    fresh = sm.replace_block(doc, sm.render_block(model))
    assert sm.check(skill, parser, fresh) == []
    block = sm.extract_block(fresh)
    flag_line = next(ln for ln in block.splitlines() if "--text-file" in ln)
    edited = fresh.replace(flag_line, flag_line + " # edited", 1)
    assert sm.check(skill, parser, edited) == ["generated usage block is stale"]
    assert sm.check(skill, parser, sm.replace_block(edited, block)) == []


def test_skill_without_cli_entry(parser):
    fake = _FakeSkill()
    assert sm.cli_path(fake) is None
    model = sm.usage_model(fake, parser)
    assert model["commands"] == []
    assert model["name"] == "fake-skill"
