from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from foundationscale import EXIT_PASS, EXIT_RED, EXIT_UNMEASURED
from foundationscale.train.speech_adjudication import (
    adjudicate_speech,
    digests_from_named_tensors,
    digests_from_safetensors_dir,
    dtype_mismatches,
    tensor_digest,
)


def _bf16_tensor(seed: int = 0, size: tuple[int, ...] = (3, 4)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*size, generator=g, dtype=torch.float32).to(torch.bfloat16)


def _fp32_tensor(seed: int = 0, size: tuple[int, ...] = (3, 4)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*size, generator=g, dtype=torch.float32)


def _healthy_manifest() -> dict:
    return {
        "rows_expected": 4,
        "rows_checked": 4,
        "rows_refused": 0,
        "refused": {},
        "placeholder_rows_verified": 4,
        "placeholder_rows_unmeasured": 0,
    }


def test_digest_matches_across_memory_and_safetensors(tmp_path: Path) -> None:
    """MUST_PASS — the movement gate rests on in-memory and on-disk bytes hashing alike."""
    t = _bf16_tensor(seed=7)
    save_file({"w": t}, str(tmp_path / "model.safetensors"))
    mem = tensor_digest(t)
    saved = digests_from_safetensors_dir(tmp_path, ["w"])["w"]
    assert mem == saved


def test_digest_changes_after_tiny_inplace_edit() -> None:
    """MUST_FIRE — a one-bit edit anywhere must flip the digest."""
    t = _bf16_tensor(seed=8)
    before = tensor_digest(t)
    edited = t.clone()
    edited.view(torch.uint8)[0] = torch.tensor(1, dtype=torch.uint8)
    after = tensor_digest(edited)
    assert before != after
    assert tensor_digest(_bf16_tensor(seed=8)) == before


def test_dtype_is_in_the_digest_and_mismatch_is_listed() -> None:
    """MUST_FIRE — same values at fp32 vs bf16 must read as a dtype change, not movement."""
    v = _fp32_tensor(seed=3)
    b = v.to(torch.bfloat16)
    d_fp32 = tensor_digest(v)
    d_bf16 = tensor_digest(b)
    assert d_fp32 != d_bf16
    assert d_fp32.split(":", 1)[0] == "float32"
    assert d_bf16.split(":", 1)[0] == "bfloat16"

    base = {"w": d_fp32, "stable": "float32:aa"}
    saved = {"w": d_bf16, "stable": "float32:aa"}
    assert dtype_mismatches(base, saved) == ["w"]


def test_prefix_whole_segment_filter() -> None:
    """MUST_FIRE — "audio.tower" must not claim its sibling "audio.tower_proj"."""
    named = [
        ("audio.tower", _bf16_tensor(seed=1)),
        ("audio.tower.w", _bf16_tensor(seed=2)),
        ("audio.tower_proj.w", _bf16_tensor(seed=3)),
        ("vision.w", _bf16_tensor(seed=4)),
    ]
    out = digests_from_named_tensors(named, ["audio.tower"])
    assert set(out) == {"audio.tower", "audio.tower.w"}
    assert "audio.tower_proj.w" not in out
    assert "vision.w" not in out


def test_duplicate_key_across_shards_raises(tmp_path: Path) -> None:
    """MUST_FIRE — one key in two shards is ambiguous, worse than absent, so it must raise."""
    t = _bf16_tensor(seed=5)
    save_file({"dup": t}, str(tmp_path / "a.safetensors"))
    save_file({"dup": t}, str(tmp_path / "b.safetensors"))
    with pytest.raises(ValueError):
        digests_from_safetensors_dir(tmp_path, ["dup"])


def _digest_pair(*, moved: bool, seed: int = 11) -> tuple[dict[str, str], dict[str, str]]:
    tower = _bf16_tensor(seed=seed)
    vision = _bf16_tensor(seed=seed + 1)
    saved_tower = tower.clone()
    if moved:
        saved_tower.view(torch.uint8)[0] = torch.tensor(9, dtype=torch.uint8)
    base = {
        "audio.tower.w": tensor_digest(tower),
        "vision.proj.w": tensor_digest(vision),
    }
    saved = {
        "audio.tower.w": tensor_digest(saved_tower),
        "vision.proj.w": tensor_digest(vision),
    }
    return base, saved


def test_audio_moved_vision_same_passes() -> None:
    """MUST_PASS — exercised audio tower moved and untouched vision is the healthy state."""
    base, saved = _digest_pair(moved=True)
    res = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests=base,
        saved_digests=saved,
        towers=[("audio.tower", True), ("vision", False)],
        adapter=None,
    )
    assert res.exit_code == EXIT_PASS
    assert not res.blocking
    assert not res.unmeasured


def test_audio_unchanged_while_exercised_is_red() -> None:
    """MUST_FIRE — declared exercised with zero movement is the defect this gate exists to catch."""
    base, saved = _digest_pair(moved=False)
    res = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests=base,
        saved_digests=saved,
        towers=[("audio.tower", True)],
        adapter=None,
    )
    assert res.exit_code == EXIT_RED
    assert res.blocking


def test_vision_moved_while_dormant_is_red() -> None:
    """MUST_FIRE — the dormant negative control must fire on any movement."""
    base, saved = _digest_pair(moved=True)
    swapped_saved = {
        "audio.tower.w": base["audio.tower.w"],
        "vision.proj.w": saved["audio.tower.w"],
    }
    vision_base = {"vision.proj.w": base["audio.tower.w"]}
    floor = {"vision.proj.w": swapped_saved["vision.proj.w"]}
    res = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests=vision_base,
        saved_digests=floor,
        towers=[("vision", False)],
        adapter=None,
    )
    assert res.exit_code == EXIT_RED
    assert res.blocking


def test_adapter_judges_exercised_towers_and_marks_the_dormant_control_not_applicable() -> None:
    """MUST_PASS/MUST_FIRE — an adapter run still proves the audio tower moved.

    peft freezes every non-target parameter and the adapter save carries only what
    trained, so the dormant control is NOT_APPLICABLE (a note, no exit effect) while
    the exercised tower is judged as in a full fine-tune.
    """
    moved = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests={"audio.tower.w": "float32:aa"},
        saved_digests={"audio.tower.w": "float32:bb"},
        towers=[("vision.proj", False), ("audio.tower", True)],
        adapter="lora",
    )
    assert moved.exit_code == EXIT_PASS, moved.render_lines()
    assert [r.gate_id for r in moved.results].count("speech.tower_movement") == 1
    assert len(moved.notes) == 1 and "NOT_APPLICABLE" in moved.notes[0]
    assert moved.unmeasured == ()
    assert any(line.startswith("[n/a] speech:") for line in moved.render_lines())

    frozen = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests={"audio.tower.w": "float32:aa"},
        saved_digests={"audio.tower.w": "float32:aa"},
        towers=[("vision.proj", False), ("audio.tower", True)],
        adapter="lora",
    )
    assert frozen.exit_code == EXIT_RED, "an adapter run that froze the audio tower must fail"


def test_dtype_mismatch_is_unmeasured_not_movement() -> None:
    """MUST_FIRE — a cross-ddtype pair can only prove movement training never produced."""
    v = _fp32_tensor(seed=21)
    base = {"audio.tower.w": tensor_digest(v)}
    saved = {"audio.tower.w": tensor_digest(v.to(torch.bfloat16))}
    res = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests=base,
        saved_digests=saved,
        towers=[("audio.tower", True)],
        adapter=None,
    )
    assert res.exit_code == EXIT_UNMEASURED
    assert "dtype" in res.unmeasured[0].lower()
    assert all(r.gate_id != "speech.tower_movement" for r in res.results)


def test_zero_rows_is_vacuous_red() -> None:
    """MUST_FIRE — zero loaded rows is VACUOUS and blocks even with healthy digests."""
    manifest = {
        "rows_expected": 4,
        "rows_checked": 0,
        "rows_refused": 0,
        "refused": {},
        "placeholder_rows_verified": 0,
        "placeholder_rows_unmeasured": 0,
    }
    res = adjudicate_speech(
        coverage_manifest=manifest,
        base_digests={"audio.tower.w": "float32:aa"},
        saved_digests={"audio.tower.w": "float32:ab"},
        towers=[("audio.tower", True)],
        adapter=None,
    )
    assert res.exit_code == EXIT_RED
    verdicts = {r.verdict.value for r in res.results}
    assert "VACUOUS" in verdicts


def test_render_lines_and_manifest_shapes() -> None:
    """MUST_PASS — serialization shapes must be json-ready and row-per-gate."""
    res = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests=None,
        saved_digests=None,
        towers=[("audio.tower", True)],
        adapter=None,
    )
    manifest = res.as_manifest()
    assert set(manifest) == {"gates", "unmeasured", "notes"}
    assert len(manifest["gates"]) == 2
    assert isinstance(manifest["unmeasured"], list)
    for g in manifest["gates"]:
        assert set(g) == {"gate_id", "verdict", "checked", "expected", "unit", "detail"}
        assert isinstance(g["verdict"], str)

    # One line per gate, then one per abstention (an unprinted abstention is a gap).
    lines = res.render_lines()
    gate_lines, rest = lines[: len(manifest["gates"])], lines[len(manifest["gates"]) :]
    for line, row in zip(gate_lines, manifest["gates"], strict=True):
        assert row["gate_id"] in line
        assert f"{row['checked']}/{row['expected']}" in line
    assert rest == [f"[UNMEASURED] speech: {reason}" for reason in manifest["unmeasured"]]


def test_fold_red_outranks_pass_and_unmeasured_moves_only_pass() -> None:
    """MUST_FIRE/MUST_PASS — the speech verdict folds into the run's in one direction."""
    from foundationscale import EXIT_PASS, EXIT_RED, EXIT_UNMEASURED
    from foundationscale.train.speech_adjudication import fold_speech_verdict

    assert fold_speech_verdict(EXIT_PASS, "PASS", EXIT_RED)[0] == EXIT_RED
    assert fold_speech_verdict(EXIT_UNMEASURED, "u", EXIT_RED)[0] == EXIT_RED
    assert fold_speech_verdict(EXIT_PASS, "PASS", EXIT_UNMEASURED)[0] == EXIT_UNMEASURED
    # An abstention never softens a RED measured elsewhere, and PASS leaves all alone.
    assert fold_speech_verdict(EXIT_RED, "red", EXIT_UNMEASURED) == (EXIT_RED, "red")
    assert fold_speech_verdict(EXIT_RED, "red", EXIT_PASS) == (EXIT_RED, "red")
    assert fold_speech_verdict(EXIT_PASS, "PASS", EXIT_PASS) == (EXIT_PASS, "PASS")


def test_capture_judges_every_modality_tower() -> None:
    """MUST_PASS — audio towers exercised, the rest the dormant control."""
    from types import SimpleNamespace

    from foundationscale.train.speech_adjudication import capture_speech_inputs

    family = SimpleNamespace(towers=(("m.vision", "image"), ("m.audio", "audio"), ("mtp", None)))
    calls: list[int] = []

    def items() -> list[tuple[str, object]]:
        calls.append(1)
        return [("m.audio.w", _bf16_tensor(seed=1)), ("m.vision.w", _bf16_tensor(seed=2))]

    towers, base = capture_speech_inputs(family, items)
    assert towers == [("m.vision", False), ("m.audio", True)]
    assert base is not None and set(base) == {"m.audio.w", "m.vision.w"}
    assert capture_speech_inputs(None, items) == ([], None)


def test_peft_names_canonicalise_to_the_base_names() -> None:
    """MUST_PASS — measured peft 0.18.1 spellings map back to the loaded names."""
    from foundationscale.train.speech_adjudication import (
        canonical_param_name,
        digests_from_named_tensors,
    )

    assert canonical_param_name("base_model.model.m.audio.w") == "m.audio.w"
    assert canonical_param_name("base_model.model.m.audio.original_module.w") == "m.audio.w"
    assert canonical_param_name("base_model.model.m.audio.modules_to_save.default.w") is None
    assert canonical_param_name("m.audio.w") == "m.audio.w"
    wrapped = [
        ("base_model.model.m.audio.original_module.w", _bf16_tensor(seed=1)),
        ("base_model.model.m.audio.modules_to_save.default.w", _bf16_tensor(seed=1)),
    ]
    assert set(digests_from_named_tensors(wrapped, ["m.audio"])) == {"m.audio.w"}


def test_final_adjudication_reads_the_save_and_abstains_without_safetensors(
    tmp_path: Path,
) -> None:
    """MUST_PASS on a real save; UNMEASURED (95) when the layout is not safetensors."""
    import torch
    from safetensors.torch import save_file

    from foundationscale import EXIT_PASS, EXIT_UNMEASURED
    from foundationscale.train.speech_adjudication import (
        digests_from_named_tensors,
        run_final_speech_adjudication,
    )

    base_t = {"m.audio.w": _bf16_tensor(seed=1), "m.vision.w": _bf16_tensor(seed=2)}
    base = digests_from_named_tensors(base_t.items(), ["m.audio", "m.vision"])
    trained = {k: v.clone() for k, v in base_t.items()}
    trained["m.audio.w"] = trained["m.audio.w"] + torch.ones_like(trained["m.audio.w"])
    save_file(trained, str(tmp_path / "model.safetensors"))
    coverage = {
        "rows_expected": 2,
        "rows_checked": 2,
        "rows_refused": 0,
        "refused": {},
        "placeholder_rows_verified": 2,
        "placeholder_rows_unmeasured": 0,
    }
    towers = [("m.vision", False), ("m.audio", True)]
    kwargs = dict(coverage_manifest=coverage, base_digests=base, towers=towers, adapter=None)
    ok = run_final_speech_adjudication(final_dir=tmp_path, has_safetensors=True, **kwargs)
    assert ok.exit_code == EXIT_PASS, ok.render_lines()
    dcp = run_final_speech_adjudication(final_dir=tmp_path, has_safetensors=False, **kwargs)
    assert dcp.exit_code == EXIT_UNMEASURED


def test_reading_a_missing_save_directory_raises_rather_than_reading_nothing(
    tmp_path: Path,
) -> None:
    """MUST_FIRE — a lost checkpoint must not read as "nothing to compare"."""
    import pytest

    from foundationscale.train.speech_adjudication import digests_from_safetensors_dir

    with pytest.raises(FileNotFoundError):
        digests_from_safetensors_dir(tmp_path / "absent", ["m.audio"])


def test_saved_modules_to_save_copies_are_skipped_and_peft_names_canonicalised(
    tmp_path: Path,
) -> None:
    """MUST_PASS — an adapter save is read under the base names, each parameter once."""
    from safetensors.torch import save_file

    from foundationscale.train.speech_adjudication import digests_from_safetensors_dir

    save_file(
        {
            "base_model.model.m.audio.w": _bf16_tensor(seed=1),
            "base_model.model.m.audio.modules_to_save.default.w": _bf16_tensor(seed=2),
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    assert set(digests_from_safetensors_dir(tmp_path, ["m.audio"])) == {"m.audio.w"}


def test_capture_with_no_modality_towers_takes_no_digests() -> None:
    """MUST_PASS — a family with only out-of-scope towers has nothing to judge."""
    from types import SimpleNamespace

    from foundationscale.train.speech_adjudication import capture_speech_inputs

    family = SimpleNamespace(towers=(("mtp", None),))
    assert capture_speech_inputs(family, lambda: []) == ([], None)


def test_dtype_tag_keeps_a_non_torch_dtype_name_verbatim() -> None:
    """MUST_PASS — only the "torch." prefix is dropped from the tag."""
    from types import SimpleNamespace

    from foundationscale.train.speech_adjudication import _dtype_name

    assert _dtype_name(SimpleNamespace(dtype="float8_e4m3")) == "float8_e4m3"
    assert _dtype_name(_bf16_tensor(seed=0)) == "bfloat16"
