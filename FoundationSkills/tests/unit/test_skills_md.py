"""A3.2: root SKILLS.md is generated from the registry and the using-foundationskills skill, never hand-written."""

from __future__ import annotations

from foundationskills.agent import export_skills
from foundationskills.core.registry import SkillRegistry
from foundationskills.skills import register_builtin_skills


def _render() -> str:
    _, plugin_root = export_skills._roots()
    return export_skills.render_skills_md(plugin_root).decode("utf-8")


def test_every_registered_skill_has_a_row_with_its_invocation() -> None:
    text = _render()
    registry = SkillRegistry()
    register_builtin_skills(registry)
    for name in registry.names():
        assert f"| `{name}` | {registry.get(name).invocation} |" in text, name


def test_it_carries_the_meta_skill_sections_and_stops_before_the_contract() -> None:
    text = _render()
    for heading in ("## Workflow", "## Non-negotiable rules", "## Skill types", "## Adding a skill"):
        assert heading in text, heading
    assert "## Validation rules" not in text  # the meta skill's own contract stays in its SKILL.md
    assert "Edit that SKILL.md, not this file" in text


def test_the_committed_file_is_the_generated_one() -> None:
    _, plugin_root = export_skills._roots()
    assert (plugin_root / "SKILLS.md").read_bytes() == export_skills.render_skills_md(plugin_root)
