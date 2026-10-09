from __future__ import annotations

import json
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from foundationscale import EXIT_PASS, EXIT_RED, EXIT_UNMEASURED
from foundationscale.train.speech_adjudication import (
    _coverage_counts_vector,
    adjudicate_speech,
    digests_from_named_tensors,
    digests_from_safetensors_dir,
    dtype_mismatches,
    finish_speech_run,
    reduce_coverage_across_ranks,
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


def test_adapter_run_judges_the_full_train_modules_not_the_whole_tower() -> None:
    """MUST_PASS: under an adapter, movement is measured on what modules_to_save saves whole."""
    from types import SimpleNamespace

    from foundationscale.train.speech_adjudication import capture_speech_inputs

    family = SimpleNamespace(towers=(("encoder", "audio"), ("vision", "image")))
    items = lambda: [  # noqa: E731
        ("encoder.subsampling.w", _bf16_tensor(seed=1)),
        ("encoder.layers.0.q.w", _bf16_tensor(seed=2)),
        ("vision.w", _bf16_tensor(seed=3)),
    ]
    towers, base = capture_speech_inputs(family, items, ["encoder.subsampling"])
    assert towers == [("encoder.subsampling", True), ("vision", False)]
    assert base is not None and set(base) == {"encoder.subsampling.w", "vision.w"}


def test_prepare_and_finish_speech_run(tmp_path: Path) -> None:
    """MUST_PASS: prepare freezes BN and digests; finish adjudicates, folds and renders."""
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file

    from foundationscale import EXIT_PASS
    from foundationscale.train.speech_adjudication import finish_speech_run, prepare_speech_run

    model = torch.nn.Module()
    model.enc = torch.nn.Sequential(torch.nn.Conv1d(2, 2, 1), torch.nn.BatchNorm1d(2))
    family = SimpleNamespace(towers=(("enc", "audio"),))
    towers, base, line = prepare_speech_run(model, family, None)
    assert towers == [("enc", True)] and base is not None and line is not None
    assert prepare_speech_run(torch.nn.Linear(1, 1), family, None)[2] is None
    with torch.no_grad():
        model.enc[0].weight.add_(1.0)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "m.safetensors")
    )
    coverage = {
        "rows_expected": 1,
        "rows_checked": 1,
        "rows_refused": 0,
        "refused": {},
        "placeholder_rows_verified": 0,
        "placeholder_rows_unmeasured": 0,
    }
    rc, done, lines, manifest = finish_speech_run(
        rc=EXIT_PASS,
        done="PASS",
        final_dir=tmp_path,
        has_safetensors=True,
        coverage_manifest=coverage,
        base_digests=base,
        towers=towers,
        adapter=None,
        placeholder_applicable=False,
    )
    assert rc == EXIT_PASS and lines[0].startswith("speech plane:") and '"gates"' in manifest


def _rank_manifest(
    expected: int = 1272,
    checked: int | None = None,
    refused: Mapping[str, int] | None = None,
    *,
    seconds: float = 51.5,
    verified: int | None = None,
    unmeasured: int = 0,
) -> dict[str, Any]:
    """One rank's collator manifest -- ``AudioCoverage.as_manifest``'s keys, per rank.

    The defaults are the measured ones (GB200, 2026-10-09): 1,200 rows trained
    plus 72 prefetched is the 1-GPU count ``speech.audio_row_coverage`` printed
    as "1272/1272 audio rows" while the 2-rank run trained 2,400 rows.
    """
    buckets = dict(refused or {})
    checked = expected if checked is None else checked
    return {
        "rows_expected": expected,
        "rows_checked": checked,
        "rows_refused": sum(buckets.values()),
        "seconds_total": seconds,
        "sampling_rate": 16000,
        "refused": buckets,
        "placeholder_rows_verified": checked if verified is None else verified,
        "placeholder_rows_unmeasured": unmeasured,
        "verdict": "COVERED",
    }


def _wire_sum_with(remote: Mapping[str, Any]) -> Callable[[list[float]], list[float]]:
    """A two-rank ``all_reduce(SUM)`` stand-in: the OTHER rank's packed row added in.

    Built from :func:`speech_adjudication._coverage_counts_vector` so the test
    speaks the wire layout the reduction speaks -- a mirrored literal here would
    prove nothing the day the two layouts drift apart.
    """
    other = _coverage_counts_vector(remote)
    return lambda row: [a + b for a, b in zip(row, other, strict=True)]


def test_reduce_coverage_across_ranks_single_process_is_the_census_copied() -> None:
    """MUST_PASS — no peers means no sums: same counts, ranks == 1, and a real copy."""
    local = _rank_manifest()
    reduced = reduce_coverage_across_ranks(local)
    assert reduced == {**local, "ranks": 1}
    assert (reduced["rows_expected"], reduced["rows_checked"]) == (1272, 1272)
    assert reduced["rows_refused"] == 0 and reduced["refused"] == {}
    assert reduced["placeholder_rows_verified"] == 1272
    assert reduced["seconds_total"] == local["seconds_total"]
    assert reduced["ranks"] == 1
    reduced["rows_checked"] = 0
    reduced["refused"]["unreadable"] = 5
    assert local["rows_checked"] == 1272
    assert local["refused"] == {}


def test_reduce_coverage_across_ranks_sums_the_two_rank_manifests() -> None:
    """MUST_FIRE — 1272/1272 + 1272/1270 is 2544/2542 with the 2 refusals ACCOUNTED."""
    from foundationscale.train.audio import AudioCoverage

    zero = _rank_manifest()
    short = _rank_manifest(checked=1270, refused={"unreadable": 2})
    reduced = reduce_coverage_across_ranks(short, all_reduce_sum=_wire_sum_with(zero), world_size=2)
    assert (reduced["rows_expected"], reduced["rows_checked"]) == (2544, 2542)
    assert reduced["rows_refused"] == 2
    assert reduced["refused"] == {"unreadable": 2}
    assert (reduced["placeholder_rows_verified"], reduced["placeholder_rows_unmeasured"]) == (
        2542,
        0,
    )
    assert reduced["sampling_rate"] == 16000
    assert reduced["ranks"] == 2
    # checked < expected with the refusals ACCOUNTED toward the denominator is
    # COVERED rather than UNDERCOVERED -- and the verdict is audio.py's own.
    from_audio = AudioCoverage(
        rows_expected=2544, rows_checked=2542, refused={"unreadable": 2}
    ).verdict()
    assert from_audio == "COVERED"
    assert reduced["verdict"] == from_audio


def test_reduce_coverage_across_ranks_sums_a_reason_only_one_rank_saw() -> None:
    """MUST_FIRE — reason slots are keyed by AUDIO_LOAD_REASONS, so single-rank reasons add up."""
    caps = _rank_manifest(checked=1270, refused={"too_long": 2})
    unreadable = _rank_manifest(checked=1270, refused={"unreadable": 2})
    reduced = reduce_coverage_across_ranks(
        caps, all_reduce_sum=_wire_sum_with(unreadable), world_size=2
    )
    assert reduced["refused"] == {"too_long": 2, "unreadable": 2}
    assert reduced["rows_refused"] == 4
    # 1270 checked on each rank: the sum is 2540, with 2 + 2 refused toward 2544 expected.
    assert (reduced["rows_expected"], reduced["rows_checked"]) == (2544, 2540)


def test_reduce_coverage_across_ranks_2544_is_the_measured_defect_regression() -> None:
    """MUST_FIRE (regression, GB200 2026-10-09) — two ranks of 1272/1272 are 2544/2544.

    The measured defect reported "1272/1272 audio rows" over these two manifests
    -- the 1-GPU count -- because each manifest was adjudicated inside its own
    process. The SUM is the coverage claim; 1272 is the defect.
    """
    local = _rank_manifest()
    reduced = reduce_coverage_across_ranks(
        local, all_reduce_sum=lambda row: [2.0 * x for x in row], world_size=2
    )
    assert (reduced["rows_expected"], reduced["rows_checked"]) == (2544, 2544)
    assert reduced["placeholder_rows_verified"] == 2544
    assert reduced["seconds_total"] == 2.0 * local["seconds_total"]
    assert reduced["ranks"] == 2
    assert (local["rows_expected"], local["rows_checked"]) == (1272, 1272)


def test_reduce_coverage_across_ranks_refuses_a_manifest_missing_a_required_count() -> None:
    """MUST_FIRE — every census key is required: an absent count is refused, never zero-filled."""
    for key in (
        "rows_expected",
        "rows_checked",
        "rows_refused",
        "refused",
        "placeholder_rows_verified",
        "placeholder_rows_unmeasured",
    ):
        manifest = _rank_manifest()
        del manifest[key]
        with pytest.raises(ValueError):
            reduce_coverage_across_ranks(manifest)


def test_reduce_coverage_across_ranks_keeps_absent_seconds_absent() -> None:
    """MUST_PASS — a census without seconds sums to no seconds: reductions invent no duration."""
    local = _rank_manifest()
    del local["seconds_total"]
    doubled = reduce_coverage_across_ranks(
        local, all_reduce_sum=lambda row: [2.0 * x for x in row], world_size=2
    )
    assert "seconds_total" not in doubled
    assert doubled["rows_expected"] == 2544
    assert "seconds_total" not in reduce_coverage_across_ranks(local)


def _free_port() -> int:
    """A port the OS says is free right now (mirrors tests/rl/test_megatron_lane.py)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _reduce_coverage_worker(rank: int, port: int) -> None:
    """One rank of the real two-process reduction over gloo.

    Each rank holds a DIFFERENT manifest and both must come out with the same
    summed one -- the two results are gathered and compared for identity
    ("identical", not "equal on the counts the test happened to name").
    """
    from datetime import timedelta  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=60),
    )
    try:
        local = _rank_manifest(
            checked=1272 if rank == 0 else 1270,
            refused={} if rank == 0 else {"unreadable": 2},
        )
        reduced = reduce_coverage_across_ranks(local)
        assert (reduced["rows_expected"], reduced["rows_checked"]) == (2544, 2542), reduced
        assert reduced["rows_refused"] == 2 and reduced["refused"] == {"unreadable": 2}, reduced
        assert reduced["ranks"] == 2 and reduced["verdict"] == "COVERED", reduced
        gathered: list[dict[str, Any]] = [{}, {}]
        dist.all_gather_object(gathered, reduced)
        assert gathered[0] == gathered[1] == reduced, gathered
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_reduce_coverage_across_ranks_really_sums_over_two_gloo_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUST_FIRE — a real 2-process group: BOTH ranks adjudicate 2544/2542, never 1272/1272."""
    pytest.importorskip("torch")
    import torch.multiprocessing as mp  # noqa: PLC0415

    # Another test in the suite can leave GLOO_SOCKET_IFNAME (e.g. "lo") in os.environ
    # through the fabric declaration; macOS has no "lo" and gloo dies with "Unable to find
    # address for: lo". Cleared for the spawn only, so gloo picks the loopback itself.
    for name in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
        monkeypatch.delenv(name, raising=False)
    mp.spawn(_reduce_coverage_worker, args=(_free_port(),), nprocs=2, join=True)


def test_finish_speech_run_holds_no_collective_and_adjudicates_the_census_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUST_PASS -- finish_speech_run runs on the WRITING rank only, so it must not reduce.

    A collective there deadlocked on GB200 (rank 0 spinning, rank 1 already gone).
    The spy raises if the reduction is reached; the gates must read the census they
    were handed, which train() already summed over the ranks.
    """
    import foundationscale.train.speech_adjudication as speech_adjudication

    summed = _rank_manifest(expected=8, checked=8, seconds=5.0)
    w = _bf16_tensor(seed=41)
    base = {"enc.w": tensor_digest(w)}
    moved = w.clone()
    moved.view(torch.uint8)[0] = torch.tensor(3, dtype=torch.uint8)
    save_file({"enc.w": moved}, str(tmp_path / "m.safetensors"))

    def _no_collective_here(manifest: Mapping[str, Any]) -> dict[str, Any]:
        raise AssertionError("finish_speech_run reached a collective")

    monkeypatch.setattr(speech_adjudication, "reduce_coverage_across_ranks", _no_collective_here)

    rc, done, lines, manifest_json = finish_speech_run(
        rc=EXIT_PASS,
        done="PASS",
        final_dir=tmp_path,
        has_safetensors=True,
        coverage_manifest=summed,
        base_digests=base,
        towers=[("enc", True)],
        adapter=None,
        placeholder_applicable=True,
    )
    assert rc == EXIT_PASS
    gates = json.loads(manifest_json)["gates"]
    row_gate = next(g for g in gates if g["gate_id"] == "speech.audio_row_coverage")
    assert (row_gate["expected"], row_gate["checked"]) == (8, 8), row_gate
    assert "8/8" in "\n".join(lines)


def test_finish_speech_run_refuses_a_missing_census(tmp_path: Path) -> None:
    """MUST_FIRE -- no census means train() skipped the rank sum; refuse, never read one rank."""
    with pytest.raises(ValueError, match="coverage_after_train"):
        finish_speech_run(
            rc=EXIT_PASS,
            done="PASS",
            final_dir=tmp_path,
            has_safetensors=False,
            coverage_manifest=None,
            base_digests={},
            towers=[],
            adapter=None,
            placeholder_applicable=False,
        )


def test_coverage_after_train_sums_over_ranks_only_when_audio_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUST_FIRE -- the census train() hands on is the RANKS' sum (8/8), never one rank's 4/4."""
    import foundationscale.train.speech_adjudication as speech_adjudication

    class _Coverage:
        def as_manifest(self) -> dict[str, Any]:
            return _rank_manifest(expected=4, checked=4, seconds=2.5)

    class _Collator:
        coverage = _Coverage()

    def _two_ranks(manifest: Mapping[str, Any]) -> dict[str, Any]:
        return reduce_coverage_across_ranks(
            manifest, all_reduce_sum=lambda row: [2.0 * x for x in row], world_size=2
        )

    monkeypatch.setattr(speech_adjudication, "reduce_coverage_across_ranks", _two_ranks)
    assert speech_adjudication.coverage_after_train(object(), audio_declared=False) is None
    summed = speech_adjudication.coverage_after_train(_Collator(), audio_declared=True)
    assert summed is not None
    assert (summed["rows_expected"], summed["rows_checked"], summed["ranks"]) == (8, 8, 2)


def test_reduce_coverage_across_ranks_refuses_an_unknown_refusal_reason() -> None:
    """MUST_FIRE -- a reason with no bucket would lose its rows from the summed census."""
    with pytest.raises(ValueError, match="not one of"):
        reduce_coverage_across_ranks(_rank_manifest(checked=1270, refused={"cosmic_ray": 2}))


def test_reduce_coverage_across_ranks_passes_through_when_torch_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUST_PASS -- no torch in sys.modules means no process group: the census is this process's."""
    import sys  # noqa: PLC0415

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    reduced = reduce_coverage_across_ranks(_rank_manifest())
    assert (reduced["rows_checked"], reduced["ranks"]) == (1272, 1)


class _FakeRow:
    def __init__(self, values: list[float], device: object) -> None:
        self.values, self.device = list(values), device

    def to(self, _where: str) -> _FakeRow:
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


def _fake_torch(*, world_size: int, backend: str) -> tuple[Any, list[object]]:
    """A torch stand-in in sys.modules: the default path's branches without real processes."""
    import types  # noqa: PLC0415

    devices: list[object] = []
    dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: world_size,
        get_backend=lambda: backend,
        ReduceOp=types.SimpleNamespace(SUM="sum"),
    )

    def _all_reduce(row: _FakeRow, op: object) -> None:
        assert op == "sum"
        row.values = [world_size * value for value in row.values]

    dist.all_reduce = _all_reduce

    def _tensor(values: list[float], dtype: object, device: object) -> _FakeRow:
        devices.append(device)
        return _FakeRow(values, device)

    fake = types.SimpleNamespace(
        distributed=dist,
        float64="float64",
        device=lambda kind, index=None: (kind, index),
        cuda=types.SimpleNamespace(current_device=lambda: 3),
        tensor=_tensor,
    )
    return fake, devices


@pytest.mark.parametrize(
    ("backend", "device"), [("gloo", ("cpu", None)), ("Backend.NCCL", ("cuda", 3))]
)
def test_reduce_coverage_across_ranks_default_path_sums_on_the_backends_device(
    monkeypatch: pytest.MonkeyPatch, backend: str, device: tuple[str, int | None]
) -> None:
    """MUST_FIRE -- the default path all-reduces one row: CPU for gloo, the CUDA device for NCCL."""
    import sys  # noqa: PLC0415

    fake, devices = _fake_torch(world_size=2, backend=backend)
    monkeypatch.setitem(sys.modules, "torch", fake)
    reduced = reduce_coverage_across_ranks(_rank_manifest())
    assert (reduced["rows_expected"], reduced["rows_checked"], reduced["ranks"]) == (2544, 2544, 2)
    assert devices == [device]


def test_reduce_coverage_across_ranks_one_rank_group_is_a_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUST_PASS -- an initialized group of one rank has no peers to sum with."""
    import sys  # noqa: PLC0415

    fake, devices = _fake_torch(world_size=1, backend="gloo")
    monkeypatch.setitem(sys.modules, "torch", fake)
    reduced = reduce_coverage_across_ranks(_rank_manifest())
    assert (reduced["rows_checked"], reduced["ranks"], devices) == (1272, 1, [])


def test_reduce_coverage_across_ranks_refuses_an_injected_sum_without_world_size() -> None:
    """MUST_FIRE -- a sum that does not say over how many ranks is not a coverage claim."""
    with pytest.raises(ValueError, match="without world_size"):
        reduce_coverage_across_ranks(_rank_manifest(), all_reduce_sum=lambda row: row)


def test_reduce_coverage_across_ranks_refuses_a_row_that_changed_width() -> None:
    """MUST_FIRE -- a reduction returning a different width would pair rows across ranks wrongly."""
    with pytest.raises(ValueError, match="values for a packed row"):
        reduce_coverage_across_ranks(
            _rank_manifest(), all_reduce_sum=lambda row: row[:-1], world_size=2
        )
