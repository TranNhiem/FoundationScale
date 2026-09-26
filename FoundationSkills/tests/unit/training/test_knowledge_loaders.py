
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from foundationskills.skills.training import knowledge as K


def _family_doc(name: str = "testfam") -> dict:
    return {
        "name": name,
        "display_name": "Test Family",
        "fs_family": name,
        "model_types": ["testmt", "testmt_text"],
        "modalities": ["text"],
        "variants": [
            {
                "id": "tiny",
                "hf_id": "test/tiny",
                "local_path": None,
                "size_b": 0.5,
                "active_b": None,
                "arch": "dense",
                "hidden": 768,
                "layers": 12,
                "heads": 12,
                "kv_heads": 4,
                "head_dim": 64,
                "vocab": 32000,
                "context_length": 4096,
                "num_experts": None,
                "experts_per_token": None,
                "tied_embeddings": True,
                "instruct": False,
            },
            {
                "id": "small",
                "hf_id": "test/small",
                "local_path": "/models/testfam-small",
                "size_b": 2.0,
                "active_b": None,
                "arch": "dense",
                "hidden": 2048,
                "layers": 24,
                "heads": 16,
                "kv_heads": 4,
                "head_dim": 128,
                "vocab": 32000,
                "context_length": 8192,
                "num_experts": None,
                "experts_per_token": None,
                "tied_embeddings": False,
                "instruct": True,
            },
        ],
        "tokenizer": {"vocab_size": 32000, "bos": "<bos>", "eos": "<eos>", "pad": None},
        "chat_template": {
            "source": "tokenizer",
            "family_tag": name,
            "assistant_marker": "<start_of_turn>model",
            "generation_prompt": "<start_of_turn>model\n",
            "notes": "test fixture",
        },
        "special_tokens": {"bos": "<bos>", "eos": "<eos>"},
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "quirks": [],
        "provenance": {"status": "literature", "evidence": ["test fixture"], "validated_on": None},
    }


def _hardware_doc(hw_id: str = "testgpu") -> dict:
    return {
        "id": hw_id,
        "gpu_name": "TEST GPU",
        "mem_gb": 80,
        "gpus_per_node": 8,
        "bf16_dense_tflops": 989,
        "peak_provenance": "datasheet",
        "interconnect": "nvlink",
        "mfu": {
            "dense": {"value": 0.4, "provenance": "literature", "evidence": "test fixture"},
            "moe": {"value": 0.3, "provenance": "literature", "evidence": "test fixture"},
        },
        "scheduler": "slurm",
        "cluster_rules": ["--time=10-00:00:00"],
        "notes": [],
    }


def _algo_doc() -> dict:
    return {
        "name": "sft",
        "stage": "sft",
        "family": "supervised",
        "data_format": "sft",
        "requires": [],
        "key_hparams": {
            "learning_rate": {"default": 2.0e-4, "range": [1.0e-5, 1.0e-3], "note": "fixture"}
        },
        "failure_modes": [{"symptom": "loss spike", "cause": "lr too high", "recovery": "lower lr"}],
        "fs_entry": "foundationscale-train",
        "references": ["fixture"],
    }


def _recipe_doc() -> dict:
    return {
        "id": "test-sft-tiny",
        "version": "1.0.0",
        "title": "Test SFT tiny",
        "index": {
            "family": "testfam",
            "size_b": [0.4, 0.6],
            "arch": "dense",
            "stage": "sft",
            "goal": "general_chat",
            "domain": "any",
            "method": "lora",
            "hardware": ["testgpu"],
        },
        "data": {
            "format": "sft",
            "min_tokens": None,
            "recommended_tokens": None,
            "min_examples": 100,
            "recommended_examples": 1000,
            "mixture": {"test": 1.0},
        },
        "stages": [
            {
                "name": "sft",
                "stage": "sft",
                "algorithm": "sft",
                "hparams": {"learning_rate": 2.0e-4},
                "fs": {"entry": "foundationscale-train", "args": {"objective": "sft"}},
            }
        ],
        "hardware": {"min_gpus": 1, "gpu_mem_gb": 80, "est_gpu_hours": None},
        "evaluation": {"benchmarks": ["fixture-bench"], "success": "fixture criteria"},
        "risks": [],
        "provenance": {"status": "literature", "evidence": ["fixture"], "validated_on": None},
    }


def make_root(base: Path) -> Path:
    root = base / "knowledge"
    (root / "families").mkdir(parents=True)
    (root / "hardware").mkdir(parents=True)
    (root / "algorithms").mkdir(parents=True)
    (root / "recipes").mkdir(parents=True)
    (root / "families" / "testfam.yaml").write_text(yaml.safe_dump(_family_doc()), encoding="utf-8")
    (root / "hardware" / "testgpu.yaml").write_text(yaml.safe_dump(_hardware_doc()), encoding="utf-8")
    (root / "algorithms" / "sft.yaml").write_text(yaml.safe_dump(_algo_doc()), encoding="utf-8")
    (root / "recipes" / "test-sft-tiny.yaml").write_text(yaml.safe_dump(_recipe_doc()), encoding="utf-8")
    (root / "stage_rules.yaml").write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "id": "SR-TEST-001",
                        "when": {"goal_in": ["reasoning"], "has_verifiable_answers": True},
                        "action": "add_stage",
                        "stage": "rl",
                        "algorithm_hint": "dr_grpo",
                        "because": "test fixture",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


def test_loaders_roundtrip(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    families = K.load_families(root)
    assert set(families) == {"testfam"}
    fam = families["testfam"]
    assert fam.fs_family == "testfam"
    assert len(fam.variants) == 2
    assert fam.variants[0].total_params == 500_000_000
    assert fam.variants[0].raw["id"] == "tiny"

    hardware = K.load_hardware(root)
    assert hardware["testgpu"].mem_gb == 80
    assert hardware["testgpu"].mfu["dense"]["value"] == 0.4

    algos = K.load_algorithm_cards(root)
    assert algos["sft"].fs_entry == "foundationscale-train"

    recipes = K.load_recipes(root)
    assert [r.id for r in recipes] == ["test-sft-tiny"]
    assert recipes[0].index["method"] == "lora"

    rules = K.load_stage_rules(root)
    assert rules[0]["id"] == "SR-TEST-001"
    assert rules[0]["when"]["has_verifiable_answers"] is True


def test_loaders_cached_by_root(tmp_path: Path) -> None:
    root_a = make_root(tmp_path / "a")
    root_b = make_root(tmp_path / "b")
    assert K.load_families(root_a) is K.load_families(root_a)
    assert K.load_families(root_a) is not K.load_families(root_b)


def test_missing_directory_raises_naming_it(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(K.KnowledgeError) as excinfo:
        K.load_families(root)
    assert "families" in str(excinfo.value)


def test_invalid_family_raises_naming_file(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    broken = _family_doc(name="brokenfam")
    del broken["variants"][0]["size_b"]  # variant missing required key
    (root / "families" / "broken.yaml").write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(K.KnowledgeError) as excinfo:
        K.load_families(root)
    assert "broken.yaml" in str(excinfo.value)


def test_invalid_yaml_raises_naming_file(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    bad = root / "hardware" / "bad.yaml"
    bad.write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(K.KnowledgeError) as excinfo:
        K.load_hardware(root)
    assert "bad.yaml" in str(excinfo.value)


def test_find_variant_by_id_hf_id_and_local_path(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert K.find_variant("tiny", root)[1].hidden == 768  # type: ignore[index]
    assert K.find_variant("test/small", root)[1].id == "small"  # type: ignore[index]
    assert K.find_variant("/models/testfam-small", root)[0].name == "testfam"  # type: ignore[index]
    assert K.find_variant("does-not-exist", root) is None


def test_find_variant_via_config_json(tmp_path: Path) -> None:
    root = make_root(tmp_path / "k")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "testmt", "hidden_size": 2048, "num_hidden_layers": 24}),
        encoding="utf-8",
    )
    found = K.find_variant(str(model_dir), root)
    assert found is not None
    assert found[0].name == "testfam"
    assert found[1].id == "small"


def test_find_variant_config_json_uses_text_config(tmp_path: Path) -> None:
    root = make_root(tmp_path / "k")
    model_dir = tmp_path / "vlm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "testmt",
                "text_config": {"model_type": "testmt_text", "hidden_size": 768, "num_hidden_layers": 12},
            }
        ),
        encoding="utf-8",
    )
    found = K.find_variant(str(model_dir), root)
    assert found is not None
    assert found[1].id == "tiny"


def test_find_variant_config_json_unknown_model_type(tmp_path: Path) -> None:
    root = make_root(tmp_path / "k")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "nope"}), encoding="utf-8")
    assert K.find_variant(str(model_dir), root) is None


def test_family_for_model_type(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert K.family_for_model_type("testmt_text", root).name == "testfam"  # type: ignore[union-attr]
    assert K.family_for_model_type("llama", root) is None


def test_real_package_data_loads() -> None:
    families = K.load_families()
    hardware = K.load_hardware()
    algos = K.load_algorithm_cards()
    recipes = K.load_recipes()
    rules = K.load_stage_rules()
    assert families and hardware and algos
    assert isinstance(recipes, list)
    assert rules and all("id" in r and "action" in r for r in rules)
