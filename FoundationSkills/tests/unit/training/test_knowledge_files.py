
"""Knowledge-base integrity tests for the training skills (Task C).

Validates every YAML knowledge file against its JSON-schema (using ONLY
foundationskills.core.schema + PyYAML) and cross-checks the knowledge against
the installed FoundationScale registries:

  * `fs_family` values (when not null) exist in
    ``foundationscale.families.registry`` and ``model_types`` match exactly;
  * every rl/preference algorithm card names an algorithm actually registered
    in FS (``foundationscale.rl.registry.available_algorithm_names``);
  * every recipe ``fs.args`` key normalises ('_' -> '-') to a real
    ``foundationscale-train`` CLI flag; RL stages (entry ``fskills-rl``) must
    keep ``fs.args`` EMPTY -- the RLTrainConfig fields live in the stage
    ``hparams``, not in the train-CLI flag vocabulary.

Also: recipe ids equal file stems and are unique; recipe families/hardware/
algorithms resolve to files in the knowledge base. No network, no GPU.
"""
from __future__ import annotations

from importlib import resources
from typing import Any

import pytest
import yaml

from foundationskills.core.schema import load_schema, validate

_REQUIRED_RECIPE_IDS = (
    "llama3-8b-manufacturing-cpt",
    "llama3-8b-manufacturing-sft-lora",
    "llama3-8b-manufacturing-sft-full",
    "llama3-8b-reasoning-rl",
)


def _knowledge_dir() -> Any:
    return resources.files("foundationskills").joinpath("skills", "training", "knowledge")


def _yaml_files(subdir: str) -> list[Any]:
    root = _knowledge_dir().joinpath(subdir)
    entries = sorted(
        (e for e in root.iterdir() if e.is_file() and e.name.endswith(".yaml")),
        key=lambda e: e.name,
    )
    assert entries, f"expected at least one YAML under knowledge/{subdir}"
    return entries


def _load(entry: Any) -> dict[str, Any]:
    data = yaml.safe_load(entry.read_text("utf-8"))
    assert isinstance(data, dict), f"{entry.name}: YAML root must be a mapping"
    return data


def _load_stemmed(subdir: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for entry in _yaml_files(subdir):
        stem = entry.name[: -len(".yaml")]
        out.append((stem, _load(entry)))
    return out


def _assert_valid(data: Any, schema_name: str, label: str) -> None:
    schema = load_schema(schema_name)
    errors = validate(data, schema)
    assert not errors, f"{label} failed {schema_name} validation: {errors}"


def _train_cli_flags() -> set[str]:
    """Flag names (no dashes) accepted by the installed FS train CLI."""
    from foundationscale.train import cli  # type: ignore

    parser = cli.build_parser()
    flags: set[str] = set()
    for action in getattr(parser, "_actions", []):
        for opt in getattr(action, "option_strings", []):
            if opt.startswith("--"):
                flags.add(opt[2:])
    assert flags, "no --flags found in foundationscale.train.cli parser"
    return flags


def _fs_family_specs() -> dict[str, tuple[str, ...]]:
    from foundationscale.families import registry as fam_reg  # type: ignore

    out: dict[str, tuple[str, ...]] = {}
    for spec in fam_reg.REGISTRY:
        out[str(spec.name)] = tuple(str(m) for m in spec.model_types)
    assert out, "FS family registry is empty"
    return out


def _fs_rl_algorithm_names() -> set[str]:
    from foundationscale.rl.registry import available_algorithm_names  # type: ignore

    names = {str(n) for n in available_algorithm_names()}
    assert names, "FS RL registry is empty"
    return names


def test_family_files_validate_and_match_fs_registry() -> None:
    specs = _fs_family_specs()
    for stem, data in _load_stemmed("families"):
        _assert_valid(data, "knowledge/family", f"families/{stem}.yaml")
        assert data["name"] == stem, f"families/{stem}.yaml: name {data['name']!r} != file stem"
        fs_family = data["fs_family"]
        if fs_family is None:
            quirk_text = " ".join(data["quirks"])
            assert "--adapter-target" in quirk_text, (
                f"families/{stem}.yaml: fs_family is null, so a quirk MUST mention "
                "passing --adapter-target explicitly"
            )
            continue
        assert fs_family in specs, (
            f"families/{stem}.yaml: fs_family {fs_family!r} not in FS registry {sorted(specs)}"
        )
        assert tuple(sorted(data["model_types"])) == tuple(sorted(specs[fs_family])), (
            f"families/{stem}.yaml: model_types {sorted(data['model_types'])} != "
            f"FS registry model_types of {fs_family!r} {sorted(specs[fs_family])}"
        )


def test_hardware_files_validate() -> None:
    for stem, data in _load_stemmed("hardware"):
        _assert_valid(data, "knowledge/hardware", f"hardware/{stem}.yaml")
        assert data["id"] == stem, f"hardware/{stem}.yaml: id {data['id']!r} != file stem"


def test_algorithm_cards_validate_and_match_fs_registry() -> None:
    rl_names = _fs_rl_algorithm_names()
    card_names: set[str] = set()
    for stem, card in _load_stemmed("algorithms"):
        _assert_valid(card, "knowledge/algorithm_card", f"algorithms/{stem}.yaml")
        assert card["name"] == stem, f"algorithms/{stem}.yaml: name {card['name']!r} != file stem"
        card_names.add(card["name"])
        if card["stage"] in ("rl", "preference"):
            assert card["name"] in rl_names, (
                f"algorithms/{stem}.yaml: {card['name']!r} not registered in "
                f"foundationscale.rl.registry {sorted(rl_names)}"
            )
            assert card["fs_entry"] == "fskills-rl"
        else:
            assert card["fs_entry"] == "foundationscale-train"
    assert "causal_lm" in card_names and "sft" in card_names, (
        "supervised algorithm cards causal_lm and sft are required"
    )


def test_stage_rules_validate() -> None:
    for entry in _yaml_files("."):
        if entry.name != "stage_rules.yaml":
            continue
        data = _load(entry)
        _assert_valid(data, "knowledge/stage_rules", "stage_rules.yaml")
        ids = [r["id"] for r in data["rules"]]
        assert len(ids) == len(set(ids)), "duplicate stage-rule ids"
        assert 10 <= len(ids) <= 14, f"expected 10-14 stage rules, got {len(ids)}"
        return
    pytest.fail("stage_rules.yaml not found under knowledge/")


def test_recipes_validate_and_cross_reference() -> None:
    flags = _train_cli_flags()
    family_stems = {stem for stem, _ in _load_stemmed("families")}
    hardware_stems = {stem for stem, _ in _load_stemmed("hardware")}
    card_names = {card["name"] for _, card in _load_stemmed("algorithms")}

    seen_ids: set[str] = set()
    for stem, recipe in _load_stemmed("recipes"):
        _assert_valid(recipe, "knowledge/recipe", f"recipes/{stem}.yaml")
        rid = recipe["id"]
        assert rid == stem, f"recipes/{stem}.yaml: id {rid!r} != file stem"
        assert rid not in seen_ids, f"duplicate recipe id {rid!r}"
        seen_ids.add(rid)

        assert recipe["index"]["family"] in family_stems, (
            f"recipe {rid}: unknown family {recipe['index']['family']!r}; "
            f"known: {sorted(family_stems)}"
        )
        for hw in recipe["index"]["hardware"]:
            assert hw in hardware_stems, (
                f"recipe {rid}: unknown hardware id {hw!r}; known: {sorted(hardware_stems)}"
            )

        for stage in recipe["stages"]:
            assert stage["algorithm"] in card_names, (
                f"recipe {rid} stage {stage['name']}: no algorithm card for "
                f"{stage['algorithm']!r}; known: {sorted(card_names)}"
            )
            entry = stage["fs"]["entry"]
            if entry == "fskills-rl":
                assert stage["fs"]["args"] == {}, (
                    f"recipe {rid} stage {stage['name']}: fskills-rl consumes an "
                    "RLTrainConfig (built from stage hparams), so fs.args must be "
                    "empty -- got " + ", ".join(stage["fs"]["args"])
                )
                continue
            for key in stage["fs"]["args"]:
                norm = key.replace("_", "-")
                assert norm in flags, (
                    f"recipe {rid} stage {stage['name']}: fs.args key {key!r} "
                    f"(normalised {norm!r}) is not a foundationscale-train flag"
                )

    for required in _REQUIRED_RECIPE_IDS:
        assert required in seen_ids, f"required recipe {required!r} missing"

    # The reference-scenario RL recipe must be dr_grpo through fskills-rl.
    rl_recipe = next(r for _, r in _load_stemmed("recipes") if r["id"] == "llama3-8b-reasoning-rl")
    rl_stages = [s for s in rl_recipe["stages"] if s["stage"] == "rl"]
    assert rl_stages, "llama3-8b-reasoning-rl has no rl stage"
    assert any(
        s["algorithm"] == "dr_grpo" and s["fs"]["entry"] == "fskills-rl" for s in rl_stages
    ), "llama3-8b-reasoning-rl must run dr_grpo through the fskills-rl entry"
