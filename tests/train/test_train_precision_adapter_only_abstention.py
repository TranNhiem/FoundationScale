"""#424: precision agreement must ABSTAIN over an adapter-only checkpoint.

The fixtures are REAL safetensors files written with the stdlib only -- an
8-byte little-endian header length, the JSON header, then a payload of exactly
the advertised length. No torch, no peft, no safetensors imports anywhere in
this file.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

# The module under test: the one housing on_save and the gates above.
from foundationscale.train import loop as sg

_DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}


def _write_shard(path: Path, tensors: Mapping[str, str]) -> Path:
    """Write one real safetensors shard: 8-byte LE length + JSON header + payload."""
    header: dict[str, dict[str, object]] = {}
    payload = bytearray()
    for name, dtype in tensors.items():
        size = 4 * _DTYPE_BYTES[dtype]  # shape [4]
        start = len(payload)
        header[name] = {"dtype": dtype, "shape": [4], "data_offsets": [start, start + size]}
        payload.extend(b"\x00" * size)
    raw = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(payload)
    return path


def _lora_only_tensors(layers: int = 4) -> dict[str, str]:
    """An adapter-only save the way peft writes it, with #424's trap intact:

    every name STARTS WITH the "base_model." wrapper prefix (which a "base"
    substring detector would misread as base weights), every name carries the
    .lora_ adapter marker, and every tensor is F32 -- peft's correct behaviour
    of holding adapter params in fp32 over a bf16 base -- which is exactly the
    all-F32 histogram _PRECISION_ACCEPTED_DTYPES accepts under BOTH bf16 and
    fp32.
    """
    tensors: dict[str, str] = {}
    for layer in range(layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            stem = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            tensors[f"{stem}.lora_A.weight"] = "F32"
            tensors[f"{stem}.lora_B.weight"] = "F32"
    return tensors


def _agreement(ckpt_dir: Path, declared: str) -> sg.PrecisionAgreement:
    """The on_save path: ONE header read feeds the histogram AND adapter_only."""
    entries = sg._safetensors_entries(ckpt_dir)
    histogram = sg._histogram_from_entries(entries)
    return sg.check_precision_agreement(
        declared, histogram, adapter_only=sg._is_adapter_only(entries)
    )


def test_adapter_only_artifact_abstains_under_bf16(tmp_path: Path) -> None:
    _write_shard(tmp_path / "model.safetensors", _lora_only_tensors())
    agreement = _agreement(tmp_path, "bf16")
    assert agreement.status == "abstain"
    # The guard rail this slipped past pre-#424: F32 is ACCEPTED for bf16, so
    # this exact histogram used to report "pass" -- a control that cannot fail.
    # The abstention must therefore carry the counted evidence out loud.
    assert agreement.observed == {"F32": 32}
    assert "32 of 32" in agreement.message
    assert "outside the adapter namespace" in agreement.message


def test_adapter_only_artifact_abstains_under_fp32(tmp_path: Path) -> None:
    _write_shard(tmp_path / "model.safetensors", _lora_only_tensors())
    agreement = _agreement(tmp_path, "fp32")
    assert agreement.status == "abstain"
    assert agreement.observed == {"F32": 32}
    assert "32 of 32" in agreement.message


def test_one_base_tensor_defeats_the_abstention(tmp_path: Path) -> None:
    """MUST-FIRE control: the SAME artifact with ONE non-adapter tensor must not abstain."""
    tensors = _lora_only_tensors()
    tensors["model.layers.0.self_attn.q_proj.weight"] = "F32"  # one BASE weight
    _write_shard(tmp_path / "model.safetensors", tensors)
    assert sg._is_adapter_only(sg._safetensors_entries(tmp_path)) is False
    agreement = _agreement(tmp_path, "bf16")
    # The histogram is still uniformly F32 and the tensor count moved by only
    # one; the ONLY thing that changed is that a base-model name is present. A
    # pass here proves the rule keys on the presence of base tensors -- not on
    # tensor count and not on dtype uniformity. Without this control, the two
    # abstentions above could pass vacuously.
    assert agreement.status == "pass"


def test_full_artifact_pass_outcome_unchanged(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.mlp.up_proj.weight": "BF16",
            "model.layers.0.mlp.down_proj.weight": "BF16",
        },
    )
    agreement = _agreement(tmp_path, "bf16")
    assert agreement.status == "pass"


def test_full_artifact_red_outcome_unchanged(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.mlp.up_proj.weight": "BF16",
            "model.layers.0.mlp.down_proj.weight": "BF16",
        },
    )
    agreement = _agreement(tmp_path, "fp32")
    assert agreement.status == "red"
    assert "'BF16'" in agreement.message


def test_dtype_histogram_is_none_never_empty_on_zero_tensors(tmp_path: Path) -> None:
    assert sg._dtype_histogram(tmp_path) is None


def test_dtype_histogram_still_raises_on_a_malformed_shard(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"\x01\x02\x03")
    with pytest.raises(ValueError):
        sg._dtype_histogram(tmp_path)
