from __future__ import annotations

import json

import pytest

from foundationskills.skills.evaluation.resolve import ResolveError, resolve, ships_chat_template


def _model(path, chat=False):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    if chat:
        (path / "chat_template.jinja").write_text("{{ messages }}")
    return path


def _adapter(path, base):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    return path


def _manifest(path, model, output_dir, adapter="None"):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: {"key": k, "value": v} for k, v in (("model", str(model)), ("output_dir", str(output_dir)), ("adapter", adapter))}
    path.write_text(json.dumps({"schema_version": 1, "config": cfg}))
    return path


def test_explicit_full_checkpoint_and_base(tmp_path):
    r = resolve(str(_model(tmp_path / "ck")), base=str(_model(tmp_path / "base")))
    assert r.adapter is False and r.base == tmp_path / "base" and r.base_source == "argument"


def test_adapter_base_from_adapter_config_when_local(tmp_path):
    base = _model(tmp_path / "base")
    r = resolve(str(_adapter(tmp_path / "ad", base)))
    assert r.adapter is True and r.base == base and r.base_source == "adapter_config"


def test_adapter_with_non_local_base_is_refused_and_named(tmp_path):
    ad = _adapter(tmp_path / "ad", "Qwen/Qwen3-8B")
    with pytest.raises(ResolveError) as exc:
        resolve(str(ad))
    assert exc.value.rule_id == "EV-IN-002"
    assert "base_model_name_or_path='Qwen/Qwen3-8B' is not a local path" in str(exc.value)


def test_full_checkpoint_without_base_is_refused(tmp_path):
    with pytest.raises(ResolveError, match="missing input: base model for checkpoint") as exc:
        resolve(str(_model(tmp_path / "ck")))
    assert exc.value.rule_id == "EV-IN-002"


def test_named_but_absent_base_is_not_skipped_over(tmp_path):
    base = _model(tmp_path / "real")
    ad = _adapter(tmp_path / "ad", base)
    with pytest.raises(ResolveError, match="from argument") as exc:
        resolve(str(ad), base=str(tmp_path / "typo"))
    assert exc.value.rule_id == "EV-IN-002"


def test_missing_checkpoint_dir(tmp_path):
    with pytest.raises(ResolveError, match="missing input: checkpoint directory") as exc:
        resolve(str(tmp_path / "nope"), base=str(tmp_path))
    assert exc.value.rule_id == "EV-IN-001"


def test_run_manifest_names_final_and_base(tmp_path):
    base = _model(tmp_path / "base")
    out = tmp_path / "run" / "sft"
    _adapter(out / "final", "not/local")
    r = resolve(run_manifest=str(_manifest(out / "run_manifest.json", base, out, "lora")))
    assert r.checkpoint == out / "final" and r.adapter is True
    assert r.base == base and r.base_source == "run_manifest"


def test_run_manifest_without_final_is_refused_not_guessed(tmp_path):
    base = _model(tmp_path / "base")
    out = tmp_path / "run"
    _model(out / "checkpoint-30")
    m = _manifest(out / "run_manifest.json", base, out)
    with pytest.raises(ResolveError, match="missing input: checkpoint directory .*final") as exc:
        resolve(run_manifest=str(m))
    assert exc.value.rule_id == "EV-IN-001"


def test_missing_run_manifest(tmp_path):
    with pytest.raises(ResolveError, match="missing input: run manifest"):
        resolve(run_manifest=str(tmp_path / "run_manifest.json"))


def test_chat_template_detection(tmp_path):
    assert ships_chat_template(_model(tmp_path / "a", chat=True)) is True
    plain = _model(tmp_path / "b")
    assert ships_chat_template(plain) is False
    (plain / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    assert ships_chat_template(plain) is True
