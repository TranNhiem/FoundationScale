"""#424 control, written from the MEASURED artifact rather than from the patch.

The generated suite beside this one exercises the new code against fixtures
built to match it. That is necessary and not sufficient: a test written from an
implementation agrees with the implementation by construction. So this file is
written from the other direction -- from tensor names a real 2-step LoRA run
actually produced on GB200 (224 of 224 adapter-namespaced, zero base-model
tensors, every name under the ``base_model.`` wrapper prefix).

The names are the whole point. peft's wrapper prefix is literally
``base_model.``, so a detector keyed on the word "base" inverts: it reads an
adapter-only checkpoint as entirely base weights and answers the base-weight
dtype question with tensors that are not base weights. The fixtures below keep
that prefix for exactly that reason.

Every helper is imported by NAME. An earlier draft probed the module with
getattr and fell back across candidate names; that version could not fail when
a helper was renamed, which is the property a control exists to have.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from foundationscale.train.loop import (
    _histogram_from_entries,
    _is_adapter_only,
    _safetensors_entries,
    check_precision_agreement,
)

# Verbatim from the measured run.
ADAPTER_NAMES = (
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
)
BASE_NAME = "base_model.model.model.layers.0.self_attn.k_proj.weight"


def _write_shard(directory: Path, names: tuple[str, ...], dtype: str) -> None:
    """One real safetensors shard: 8-byte LE header length, JSON header, payload."""
    header: dict[str, object] = {}
    offset = 0
    for name in names:
        header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header).encode()
    (directory / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * offset
    )


def _read(directory: Path, names: tuple[str, ...], dtype: str = "F32"):
    _write_shard(directory, names, dtype)
    return _safetensors_entries(directory)


@pytest.mark.parametrize("declared", ["bf16", "fp32"])
def test_measured_adapter_only_artifact_abstains_under_both_declarations(
    tmp_path: Path, declared: str
) -> None:
    """The defect: an all-F32 adapter histogram satisfies BOTH tolerance sets.

    bf16 accepts ("BF16", "F32") because autocast masters stay fp32, and fp32
    accepts ("F32",). So the pre-#424 gate reported "pass" under either
    declaration while measuring tensors that cannot answer the question.
    """
    entries = _read(tmp_path, ADAPTER_NAMES)
    assert _is_adapter_only(entries) is True
    result = check_precision_agreement(
        declared, _histogram_from_entries(entries), adapter_only=True
    )
    assert result.status == "abstain", result


def test_one_base_tensor_stops_the_abstention(tmp_path: Path) -> None:
    """The must-fire control that makes the abstention tests non-vacuous.

    Without it, a detector that abstained on EVERYTHING would satisfy them.
    """
    entries = _read(tmp_path, (*ADAPTER_NAMES, BASE_NAME))
    assert _is_adapter_only(entries) is False
    result = check_precision_agreement("fp32", _histogram_from_entries(entries), adapter_only=False)
    assert result.status == "pass", result


def test_full_artifact_still_reds_on_a_real_mismatch(tmp_path: Path) -> None:
    entries = _read(tmp_path, (BASE_NAME,), dtype="BF16")
    assert _is_adapter_only(entries) is False
    result = check_precision_agreement("fp32", _histogram_from_entries(entries), adapter_only=False)
    assert result.status == "red", result


def test_full_artifact_still_passes_when_it_agrees(tmp_path: Path) -> None:
    entries = _read(tmp_path, (BASE_NAME,), dtype="BF16")
    result = check_precision_agreement("bf16", _histogram_from_entries(entries), adapter_only=False)
    assert result.status == "pass", result


def test_an_empty_artifact_is_not_adapter_only(tmp_path: Path) -> None:
    """``all([])`` is True. Nothing is not adapter-only; it is nothing."""
    assert _is_adapter_only([]) is False
    assert _safetensors_entries(tmp_path) == []


def test_adapter_only_is_required_and_keyword_only() -> None:
    """A defaulted argument is how this fix would become an orphan.

    Requiring it is what forced every pre-existing call site to state which
    kind of artifact it is comparing, instead of inheriting an answer.
    """
    with pytest.raises(TypeError):
        check_precision_agreement("bf16", {"F32": 3})  # type: ignore[call-arg]
