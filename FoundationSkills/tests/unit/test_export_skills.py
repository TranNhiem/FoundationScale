"""Unit tests for foundationskills.agent.export_skills (no network, in a tmp repo copy).

Doctrine under test: the generated export tree is byte-identical to the skill
packages and spec-compliant, ``write_tree``/``check_tree`` round-trip and only
ever remove ``fskills-*`` export directories, and the CLI prints one JSON line
per call with exit 0 (written / in sync), 5 (drift) or 96 (refused).
"""
from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import pytest

import foundationskills
from foundationskills.agent import export_skills, routing_eval
from foundationskills.agent.export_skills import (
    ExportRefused,
    check_tree,
    expected_tree,
    spec_problems,
    write_tree,
)
from foundationskills.cli import main

SKILL_DIRNAME = "fskills-training"
DESCRIPTION = "Probe, launch and evaluate FoundationScale training runs."


def _skill_md(front: dict[str, object]) -> str:
    """A SKILL.md whose frontmatter is *front* (JSON scalars are valid YAML)."""
    lines = ["---"]
    lines.extend(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in front.items())
    lines += ["---", "", "The body.", ""]
    return "\n".join(lines)


def _valid_front() -> dict[str, object]:
    return {"name": SKILL_DIRNAME, "description": DESCRIPTION}


def _problems_of(front: dict[str, object], dirname: str = SKILL_DIRNAME) -> list[str]:
    problems = spec_problems(_skill_md(front), dirname)
    assert problems, f"{front!r} must not be spec-compliant"
    return problems


def _plugin_root() -> Path:
    return Path(foundationskills.__file__).resolve().parent.parent


def test_spec_problems_accepts_a_valid_frontmatter() -> None:
    front = {
        "name": SKILL_DIRNAME,
        "description": DESCRIPTION,
        "allowed-tools": "Read Grep",
        "metadata": {"when_to_use": "training runs"},
    }
    assert spec_problems(_skill_md(front), SKILL_DIRNAME) == []


def test_spec_problems_detects_a_disallowed_top_level_key() -> None:
    assert any("category" in p for p in _problems_of({**_valid_front(), "category": "training"}))


def test_spec_problems_detects_a_name_that_is_not_a_slug() -> None:
    problems = spec_problems(
        _skill_md({"name": "fskills_training", "description": DESCRIPTION}), "fskills_training"
    )
    assert problems and any("^[a-z0-9]" in p for p in problems)


def test_spec_problems_detects_name_and_directory_mismatch() -> None:
    assert any("directory name" in p for p in _problems_of({"name": "fskills-other", "description": DESCRIPTION}))


def test_spec_problems_detects_an_overlong_name() -> None:
    name = "a" * 65
    assert any("64" in p for p in _problems_of({"name": name, "description": DESCRIPTION}, dirname=name))


def test_spec_problems_detects_an_empty_description() -> None:
    assert any("`description`" in p for p in _problems_of({"name": SKILL_DIRNAME, "description": ""}))


def test_spec_problems_detects_an_overlong_description() -> None:
    assert any("`description`" in p for p in _problems_of({"name": SKILL_DIRNAME, "description": "x" * 1025}))


def test_spec_problems_detects_missing_required_keys() -> None:
    problems = _problems_of({"license": "MIT"})
    assert any("`name`" in p for p in problems) and any("`description`" in p for p in problems)


def test_spec_problems_detects_compatibility_over_500_chars() -> None:
    assert any("compatibility" in p for p in _problems_of({**_valid_front(), "compatibility": "x" * 501}))


def test_spec_problems_detects_non_string_metadata() -> None:
    assert any("metadata" in p for p in _problems_of({**_valid_front(), "metadata": {"accuracy": 0.9}}))


def test_spec_problems_detects_a_missing_frontmatter_block() -> None:
    problems = spec_problems("Just a body with no frontmatter.\n", SKILL_DIRNAME)
    assert problems and any("frontmatter" in p for p in problems)


def test_expected_tree_copies_one_skill_md_per_package_byte_identically() -> None:
    plugin_root = _plugin_root()
    tree = expected_tree(plugin_root)
    skills_prefix = f"{plugin_root.name}/skills/"
    bundle_names = {source for source, _destination in export_skills._bundle_dirs()}
    index = routing_eval.load_index()
    assert index
    assert len([rel for rel in tree if rel.endswith("/SKILL.md")]) == len(index)
    for entry in index:
        name = str(entry["name"])
        source = importlib.resources.files(str(entry["package"])).joinpath("SKILL.md").read_bytes()
        assert tree[f"{skills_prefix}{name}/SKILL.md"] == source
        assert spec_problems(source.decode("utf-8"), name) == []
        for rel in (rel for rel in tree if rel.startswith(f"{skills_prefix}{name}/")):
            tail = rel[len(f"{skills_prefix}{name}/"):]
            assert tail == "SKILL.md" or tail.split("/", 1)[0] in bundle_names
            assert "__pycache__" not in tail and not tail.endswith(".pyc") and not tail.endswith("skill.py")


def test_expected_tree_manifests_parse_and_list_every_skill() -> None:
    plugin_root = _plugin_root()
    tree = expected_tree(plugin_root)
    plugin = json.loads(tree[f"{plugin_root.name}/.claude-plugin/plugin.json"].decode("utf-8"))
    marketplace = json.loads(tree[".claude-plugin/marketplace.json"].decode("utf-8"))
    names = sorted(str(entry["name"]) for entry in routing_eval.load_index())
    assert plugin["name"] == "foundationskills"
    assert plugin["skills"] == "./skills"
    assert marketplace["name"] == "foundationscale"
    assert marketplace["plugins"][0]["source"] == f"./{plugin_root.name}"
    assert marketplace["plugins"][0]["version"] == plugin["version"]
    assert marketplace["plugins"][0]["skills"] == [f"./skills/{name}" for name in names]


def test_expected_tree_refuses_a_non_compliant_skill_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "fskills_refused"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "SKILL.md").write_text("---\nname: Bad_Name\ndescription: \n---\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        export_skills, "_index", lambda: [{"package": package.name, "name": "fskills-refused"}]
    )
    with pytest.raises(ExportRefused) as excinfo:
        expected_tree(_plugin_root())
    message = str(excinfo.value)
    assert "`name`" in message and "description" in message
    assert message.count("fskills-refused:") >= 2  # every problem is listed


def test_write_tree_and_check_tree_round_trip_in_a_tmp_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    plugin_root = repo_root / "FoundationSkills"
    keep = plugin_root / "skills" / "keepme"
    keep.mkdir(parents=True)
    (keep / "keep.md").write_text("mine\n", encoding="utf-8")

    written = write_tree(repo_root, plugin_root)
    assert written and (plugin_root / ".claude-plugin" / "plugin.json").is_file()
    assert not list(plugin_root.rglob("*.tmp"))  # nothing but the exports lands on disk
    assert check_tree(repo_root, plugin_root) == []

    skill_md = sorted(plugin_root.joinpath("skills").glob("fskills-*/SKILL.md"))[0]
    original = skill_md.read_bytes()
    skill_md.write_bytes(original + b"\n# drift\n")
    assert any("differs:" in problem for problem in check_tree(repo_root, plugin_root))

    skill_md.unlink()
    assert any("missing:" in problem for problem in check_tree(repo_root, plugin_root))
    write_tree(repo_root, plugin_root)
    assert check_tree(repo_root, plugin_root) == []

    stray = plugin_root / "skills" / "fskills-x"
    stray.mkdir()
    (stray / "notes.md").write_text("stray\n", encoding="utf-8")
    stale = plugin_root / "skills" / "fskills-stale"
    stale.mkdir()
    (stale / "SKILL.md").write_text("stale\n", encoding="utf-8")
    assert any("extra:" in problem for problem in check_tree(repo_root, plugin_root))

    done = write_tree(repo_root, plugin_root)
    assert not stray.exists() and not stale.exists()
    assert "FoundationSkills/skills/fskills-x" in done
    assert keep.is_dir() and (keep / "keep.md").read_text(encoding="utf-8") == "mine\n"
    assert check_tree(repo_root, plugin_root) == []


def test_cli_export_skills_writes_then_confirms_with_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    plugin_root = repo_root / "FoundationSkills"
    (plugin_root / "skills").mkdir(parents=True)
    monkeypatch.setattr(export_skills, "_roots", lambda: (repo_root, plugin_root))

    capsys.readouterr()
    assert main(["export-skills", "--check"]) == 5
    drift = json.loads(capsys.readouterr().out)
    assert drift["status"] == "RED" and drift["problems"]

    assert main(["export-skills"]) == 0
    export_line = json.loads(capsys.readouterr().out)
    assert export_line["status"] == "PASS" and export_line["written"]

    assert main(["export-skills", "--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "PASS"}


def test_cli_export_skills_refusal_is_96(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom() -> tuple[Path, Path]:
        raise ExportRefused("export-skills needs a FoundationSkills source checkout")

    monkeypatch.setattr(export_skills, "_roots", _boom)
    capsys.readouterr()
    assert main(["export-skills"]) == 96
    assert main(["export-skills", "--check"]) == 96
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2
    assert all(line["status"] == "REFUSED" for line in lines)
    assert lines[0]["reason"] == "export-skills needs a FoundationSkills source checkout"
