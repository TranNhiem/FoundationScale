"""A quantized checkpoint never inherits an unquantized serving cell's evidence (A4.8 v2)."""

import json

from foundationskills.interfaces.fs.emit_serve import model_quantization
from foundationskills.interfaces.fs.serving_envelope import lookup


def test_unquantized_cell_still_matches():
    assert lookup("gemma4_unified", "full", "text") is not None


def test_quantized_checkpoint_has_no_cell_until_measured():
    assert lookup("gemma4_unified", "full", "text", quantization="compressed-tensors:pack-quantized") is None


def test_model_quantization_reads_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4", "quantization_config": {
        "quant_method": "compressed-tensors", "format": "pack-quantized"}}), encoding="utf-8")
    assert model_quantization(tmp_path) == "compressed-tensors:pack-quantized"


def test_model_quantization_none_for_plain_or_missing_config(tmp_path):
    assert model_quantization(tmp_path) is None
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    assert model_quantization(tmp_path) is None


def test_measured_gptq_gemma4_cell_is_separate_from_bf16():
    q = lookup("gemma4", "full", "text", quantization="compressed-tensors:pack-quantized")
    assert q is not None and "GPTQ" in q["evidence"]
    assert lookup("gemma4", "full", "text")["evidence"] != q["evidence"]
