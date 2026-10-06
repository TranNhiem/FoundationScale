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


def test_adapter_skips_movement_and_is_unmeasured_only_when_gates_pass() -> None:
    """MUST_FIRE — adapter run refuses the measurement but coverage must still adjudicate first."""
    res = adjudicate_speech(
        coverage_manifest=_healthy_manifest(),
        base_digests={"audio.tower.w": "float32:aa"},
        saved_digests={"audio.tower.w": "float32:bb"},
        towers=[("audio.tower", True)],
        adapter="lora",
    )
    assert res.exit_code == EXIT_UNMEASURED
    assert len(res.unmeasured) == 1
    assert "NOT_ESTABLISHED" in res.unmeasured[0]
    assert all(r.gate_id != "speech.tower_movement" for r in res.results)

    bad_manifest = _healthy_manifest()
    bad_manifest.update(rows_checked=0, placeholder_rows_verified=0, placeholder_rows_unmeasured=0)
    bad = adjudicate_speech(
        coverage_manifest=bad_manifest,
        base_digests={"audio.tower.w": "float32:aa"},
        saved_digests={"audio.tower.w": "float32:bb"},
        towers=[("audio.tower", True)],
        adapter="lora",
    )
    assert bad.exit_code == EXIT_RED


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
    assert set(manifest) == {"gates", "unmeasured"}
    assert len(manifest["gates"]) == 2
    assert isinstance(manifest["unmeasured"], list)
    for g in manifest["gates"]:
        assert set(g) == {"gate_id", "verdict", "checked", "expected", "unit", "detail"}
        assert isinstance(g["verdict"], str)

    lines = res.render_lines()
    assert len(lines) == len(manifest["gates"])
    for line, row in zip(lines, manifest["gates"], strict=True):
        assert row["gate_id"] in line
        assert f"{row['checked']}/{row['expected']}" in line


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


def test_capture_judges_every_modality_tower_and_skips_digests_for_adapters() -> None:
    """MUST_PASS — audio towers exercised, the rest the dormant control; adapters get none."""
    from types import SimpleNamespace

    from foundationscale.train.speech_adjudication import capture_speech_inputs

    family = SimpleNamespace(towers=(("m.vision", "image"), ("m.audio", "audio"), ("mtp", None)))
    calls: list[int] = []

    def items() -> list[tuple[str, object]]:
        calls.append(1)
        return [("m.audio.w", _bf16_tensor(seed=1)), ("m.vision.w", _bf16_tensor(seed=2))]

    towers, base = capture_speech_inputs(family, items, None)
    assert towers == [("m.vision", False), ("m.audio", True)]
    assert base is not None and set(base) == {"m.audio.w", "m.vision.w"}
    towers, base = capture_speech_inputs(family, items, "lora")
    assert base is None and len(calls) == 1, "an adapter run must not walk the state dict"
    assert capture_speech_inputs(None, items, None) == ([], None)


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
