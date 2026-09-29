from __future__ import annotations

import json

import pytest

from foundationskills.skills.evaluation.baseline import BaselineCache, fingerprint, model_identity

FIELDS = {
    "harness_version": "0.4.12",
    "backend": "hf",
    "task": "mmlu",
    "metric": "acc,none",
    "num_fewshot": 5,
    "seed": 1234,
    "gen_kwargs": None,
    "chat_template": False,
    "dtype": "bfloat16",
    "limit": None,
    "base": {"path": "/m", "config_sha256": "c", "files": []},
}


def test_fingerprint_is_stable():
    assert fingerprint(dict(FIELDS)) == fingerprint(dict(reversed(list(FIELDS.items()))))


@pytest.mark.parametrize("key,value", [
    ("harness_version", "0.4.13"), ("backend", "vllm"), ("task", "hellaswag"), ("metric", "acc_norm,none"),
    ("num_fewshot", 0), ("seed", 1), ("gen_kwargs", "temperature=0"), ("chat_template", True),
    ("dtype", "float16"), ("limit", 8), ("base", {"path": "/m", "config_sha256": "d", "files": []}),
])
def test_fingerprint_changes_with_every_hashed_field(key, value):
    assert fingerprint({**FIELDS, key: value}) != fingerprint(FIELDS)


def test_model_identity_tracks_config_and_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "README.md").write_text("ignored")
    ident = model_identity(tmp_path)
    assert ident["files"] == [["config.json", 2], ["model.safetensors", 10]]
    (tmp_path / "config.json").write_text('{"a": 1}')
    assert model_identity(tmp_path)["config_sha256"] != ident["config_sha256"]


def test_cache_miss_then_hit(tmp_path):
    cache = BaselineCache(tmp_path / "bl")
    fp = fingerprint(FIELDS)
    assert cache.get(FIELDS["base"], fp) is None
    path = cache.put(FIELDS["base"], fp, {"score": 0.5, "stderr": 0.01})
    assert path.is_file() and path.parent.parent == tmp_path / "bl"
    assert cache.get(FIELDS["base"], fp)["score"] == 0.5


def test_record_with_mismatched_fingerprint_is_rejected(tmp_path):
    cache = BaselineCache(tmp_path)
    fp = fingerprint(FIELDS)
    path = cache.put(FIELDS["base"], fp, {"score": 0.5})
    record = json.loads(path.read_text())
    record["fingerprint"] = "0" * 64  # tampered / written by a different config
    path.write_text(json.dumps(record))
    assert cache.get(FIELDS["base"], fp) is None


def test_corrupt_record_is_a_miss(tmp_path):
    cache = BaselineCache(tmp_path)
    fp = fingerprint(FIELDS)
    path = cache.path(FIELDS["base"], fp)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert cache.get(FIELDS["base"], fp) is None
