"""Coverage of emit_run_manifest's refusal/translation edges.

Why this file exists
--------------------
Per the suite-wide map (0c46636), 66 of this tool's lines execute in no test:
the argparse type guards, the independence guard's same/nested refusals, the
JSON reader's three failure arms, the effective/env entry translators, the
save-dir counter, the bare-null control's boundary arms, and much of
``derive_declared_full_ft``'s two-source refusal matrix. These are exactly
the lines that decide whether a BAD launch is refused readably -- the reason
the tool exists -- so this file drives them and asserts the refusal kind,
not just the raise.

The module is loaded by file path (the same idiom the neighboring
provenance/tooling tests use) because ``tools`` is not an installed package.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

# Load the tool by path, registered in sys.modules first: @dataclass resolves
# string annotations through sys.modules[cls.__module__] at decoration time.
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_spec = importlib.util.spec_from_file_location(
    "fs_emit_run_manifest", _REPO / "tools" / "emit_run_manifest.py"
)
emitter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = emitter
_spec.loader.exec_module(emitter)

EmitRefused = emitter.EmitRefused
EmitUnmeasured = emitter.EmitUnmeasured


# --- argparse type guard ------------------------------------------------------


def test_positive_int_refuses_non_integers() -> None:
    """A count flag that cannot parse is usage (exit-2 shape), never a default."""
    with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
        emitter._positive_int("two")


def test_positive_int_refuses_zero_and_negative() -> None:
    """0 observed saves is a named state in the control, not a valid CLI count."""
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        emitter._positive_int("0")


# --- ensure_declaration_is_independent ----------------------------------------


def test_same_directory_as_base_and_judged_is_refused(tmp_path: Path) -> None:
    """'What is there matches what is there' is not a check."""
    with pytest.raises(EmitRefused, match="same location"):
        emitter.ensure_declaration_is_independent(tmp_path, tmp_path)


def test_nesting_in_either_direction_is_refused(tmp_path: Path) -> None:
    outer = tmp_path / "judge"
    inner = outer / "base"
    inner.mkdir(parents=True)
    with pytest.raises(EmitRefused, match="nested with the directory"):
        emitter.ensure_declaration_is_independent(inner, outer)
    with pytest.raises(EmitRefused, match="nested with the directory"):
        emitter.ensure_declaration_is_independent(outer, inner)


def test_distinct_dirs_pass_and_do_not_touch_disk(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    emitter.ensure_declaration_is_independent(a, b)  # must not raise


# --- _read_json_mapping: the three failure arms are all UNMEASURED ------------


def test_missing_config_is_unmeasured_not_dense(tmp_path: Path) -> None:
    """An absent config cannot launder into a classification."""
    with pytest.raises(EmitUnmeasured, match="not found"):
        emitter._read_json_mapping(tmp_path / "config.json", "HF config")


def test_undecodable_config_is_unmeasured(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_bytes(b"\xff\xfe not json")
    with pytest.raises(EmitUnmeasured, match="not valid JSON"):
        emitter._read_json_mapping(target, "HF config")


def test_non_object_config_is_unmeasured(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text("[1, 2]")
    with pytest.raises(EmitUnmeasured, match="not a JSON object"):
        emitter._read_json_mapping(target, "HF config")


# --- entry translators: bad input is refused with the offending text ---------


class _PickyResolver:
    """The resolver double: one invalid key, everything else accepted."""

    def record_effective(self, key: str, value: Any, source: str) -> None:
        if key == "bad key!!":
            raise ValueError("key must not contain spaces")


def test_effective_pair_without_equals_sign_is_a_launcher_bug() -> None:
    with pytest.raises(EmitRefused, match="expects KEY=VALUE"):
        emitter._record_effective_pairs(_PickyResolver(), ["learning_rate"])


def test_effective_pair_with_invalid_key_names_the_entry() -> None:
    with pytest.raises(EmitRefused, match="invalid effective-config entry"):
        emitter._record_effective_pairs(_PickyResolver(), ["bad key!!=1"])


def test_record_watched_env_invalid_name_is_refused() -> None:
    with pytest.raises(EmitRefused, match="invalid --watch-env name"):
        emitter._record_watched_env(_PickyResolver(), ["bad key!!"])


def test_record_stated_entries_invalid_is_refused() -> None:
    with pytest.raises(EmitRefused, match="invalid in-manifest record entry"):
        emitter._record_stated_entries(_PickyResolver(), [("bad key!!", "v")], "emitter")


def test_record_watched_env_records_unset_as_a_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset watched var is recorded as <unset>, not dropped invisibly."""
    got: list[tuple[str, str, str]] = []

    class _Capture:
        def record_effective(self, key: str, value: Any, source: str) -> None:
            got.append((key, value, source))

    monkeypatch.delenv("FS_TEST_NEVER_SET_THIS", raising=False)
    emitter._record_watched_env(_Capture(), ["FS_TEST_NEVER_SET_THIS"])
    assert got == [("FS_TEST_NEVER_SET_THIS", "<unset>", "env:FS_TEST_NEVER_SET_THIS")]


# --- _count_save_dirs ----------------------------------------------------------


def test_count_save_dirs_counts_iter_dirs_and_ignores_other_entries(
    tmp_path: Path,
) -> None:
    (tmp_path / "iter_0000010").mkdir()
    (tmp_path / "iter_0000020").mkdir()
    (tmp_path / "iter_notes.txt").write_text("x")  # a FILE is not a save dir
    (tmp_path / "other").mkdir()
    assert emitter._count_save_dirs(tmp_path) == 2
    assert emitter._count_save_dirs(tmp_path / "empty") == 0


# --- check_saved_run_declaration: the #79 control's edges ---------------------


def test_negative_save_count_is_a_caller_bug() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        emitter.check_saved_run_declaration("{}", saves_observed=-1)


def test_zero_saves_is_named_not_exercised_never_pass() -> None:
    state = emitter.check_saved_run_declaration("{}", saves_observed=0)
    assert state.startswith("NOT-EXERCISED")


def test_saves_with_no_declared_key_at_all_fails_closed() -> None:
    """A record shape this control does not recognize must NOT default anywhere."""
    with pytest.raises(emitter.BareNullDeclarationError, match="NO 'declared' key"):
        emitter.check_saved_run_declaration('{"config": {}}', saves_observed=2)


def test_saves_with_a_declared_block_return_declared_state() -> None:
    state = emitter.check_saved_run_declaration(
        '{"declared": {"num_experts": 0}}', saves_observed=2
    )
    assert state.startswith("DECLARED:")


# --- derive_declared_full_ft: the two-source refusal matrix -------------------


def _write_shard(path: Path, entries: dict[str, tuple[str, tuple[int, ...]]]) -> Path:
    """Minimal valid safetensors shard (zero payload), matching the checksum-free reader."""
    header: dict[str, object] = {}
    cursor = 0
    widths = {"F32": 4, "BF16": 2, "I64": 8}
    for key, (dtype, shape) in entries.items():
        numel = 1
        for d in shape:
            numel *= int(d)
        nbytes = numel * widths[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * cursor)
    return path


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "hf_config.json"
    cfg.write_text(json.dumps(payload))
    return cfg


def test_derive_full_ft_dense_happy_path_records_positively(tmp_path: Path) -> None:
    """enable_moe_block=false + a clean dense base census => num_experts=0 recorded."""
    base = tmp_path / "base"
    base.mkdir()
    _write_shard(base / "model.safetensors", {"w": ("BF16", (2, 2))})
    judged = tmp_path / "out"
    judged.mkdir()
    cfg = _write_config(
        tmp_path / "conf",
        {"enable_moe_block": False, "num_hidden_layers": 2},
    )
    declared, info = emitter.derive_declared_full_ft(base, judged, cfg)
    # The declaration is written POSITIVELY: a measured zero, not a blank.
    assert declared.num_experts == 0
    assert declared.num_moe_layers is None
    assert "two independent sources agree" in declared.moe_layer_basis
    assert declared.declared_fqns == ("w",)
    assert info["expert_family_census"] == 0
    assert info["declared_fqn_count"] == 1


def test_derive_full_ft_refuses_an_unreadable_base_checkpoint(tmp_path: Path) -> None:
    """The denominator is the base artifact; an unreadable one is unmeasured."""
    base = tmp_path / "base"
    base.mkdir()
    # A "checkpoint" that is none: no metadata, no shards.
    judged = tmp_path / "out"
    judged.mkdir()
    cfg = _write_config(tmp_path / "conf", {"enable_moe_block": False, "num_hidden_layers": 2})
    with pytest.raises(EmitUnmeasured, match="unreadable as a checkpoint"):
        emitter.derive_declared_full_ft(base, judged, cfg)


def test_derive_full_ft_refuses_an_all_bookkeeping_base(tmp_path: Path) -> None:
    """A base listing only extra_state rows censors to zero REAL tensors: refuse."""
    base = tmp_path / "base"
    base.mkdir()
    _write_shard(base / "model.safetensors", {"opt_extra_state": ("BF16", (4,))})
    judged = tmp_path / "out"
    judged.mkdir()
    cfg = _write_config(tmp_path / "conf", {"enable_moe_block": False, "num_hidden_layers": 2})
    with pytest.raises(EmitRefused, match="0 declarable tensors"):
        emitter.derive_declared_full_ft(base, judged, cfg)


def test_derive_full_ft_flag_true_without_any_count_is_refused(tmp_path: Path) -> None:
    """MoE affirmed but no routed count key: naming the Gemma-4-26B defect class."""
    base = tmp_path / "base"
    base.mkdir()
    from foundationscale.checkpoint.dcp_meta import read_metadata  # noqa: F401 -- import probe

    _write_shard(
        base / "model.safetensors",
        {"layers.0.mlp.experts.0.w1": ("BF16", (2, 2))},
    )
    judged = tmp_path / "out"
    judged.mkdir()
    cfg = _write_config(tmp_path / "comp", {"enable_moe_block": True, "num_hidden_layers": 2})
    with pytest.raises(EmitRefused, match="no routed-expert count"):
        emitter.derive_declared_full_ft(base, judged, cfg)


def test_derive_full_ft_flag_false_with_live_experts_is_a_contradiction(tmp_path: Path) -> None:
    """Config says dense, artifact holds experts: refuse, name the first one."""
    base = tmp_path / "base"
    base.mkdir()
    _write_shard(
        base / "model.safetensors",
        {
            "layers.0.mlp.experts.0.w1": ("BF16", (2, 2)),
            "layers.0.mlp.experts.1.w1": ("BF16", (2, 2)),
        },
    )
    judged = tmp_path / "out"
    judged.mkdir()
    cfg = _write_config(tmp_path / "conf", {"enable_moe_block": False, "num_hidden_layers": 2})
    with pytest.raises(EmitRefused, match="contradict"):
        emitter.derive_declared_full_ft(base, judged, cfg)


def test_independence_guard_refuses_when_a_side_cannot_be_stat_ed(tmp_path: Path) -> None:
    """A missing base makes identity unverifiable -- 'could not verify they differ' refuses."""
    sib = tmp_path / "other"
    sib.mkdir()
    with pytest.raises(EmitRefused, match="cannot establish independence"):
        emitter.ensure_declaration_is_independent(tmp_path / "never", sib)


def test_derive_full_ft_refuses_a_config_that_declares_invalidly(tmp_path: Path) -> None:
    """A non-positive routed count is a refusal: fabricate less, not narrower."""
    base = tmp_path / "base"
    base.mkdir()
    _write_shard(base / "model.safetensors", {"w": ("BF16", (2, 2))})
    judged = tmp_path / "out"
    judged.mkdir()
    cfg = _write_config(
        tmp_path / "bad",
        {"enable_moe_block": False, "num_hidden_layers": 2, "num_experts": -2},
    )
    with pytest.raises(EmitRefused, match="declares invalidly"):
        emitter.derive_declared_full_ft(base, judged, cfg)
