# SPDX-License-Identifier: Apache-2.0
"""GPU-only proof that the offline preference trainer's AUDIO declaration
(``PreferenceTrainConfig.audio_column``/``audio_language``) reaches the
Qwen3-ASR model, not just the transcription request's tensors.

Two claims, each needing real hardware, a real checkpoint and real clips,
hence living here rather than in ``tests/`` (see the CPU twin,
``tests/rl/test_preference_trainer_audio.py``, for what is claimed without a
GPU):

  1. An audio-declared DPO run on a real Qwen3-ASR checkpoint (LoRA r16)
     reproduces the SAME step-0 invariant the text-only and image-declared
     smoke tests prove: policy and reference are the same checkpoint with the
     adapter disabled, so every margin is zero and the loss is exactly
     ``softplus(0) = ln(2)`` (within 1e-3 -- the policy read IS a real forward,
     so this is a measured closeness, not an identity).
  2. THE CONTROL: two independent 2-step DPO runs on the SAME first 8 pairs,
     same model, same seed (so LoRA's random init reproduces identically --
     ``PreferenceTrainer.run()`` seeds ``torch`` before loading), once with
     the REAL clips the pairs declare and once with every clip replaced by an
     all-zero (silent) wav of EXACTLY the same length in tmp_path. If the
     model's forward ignored the audio tower's output, the two losses would
     be identical at every step; after one gradient step shaped by what the
     tower saw, they must differ.

Skipped (this campaign tier's convention, NOT ``tests/``'s "no skips" rule)
unless CUDA is available and the two env names below are set::

    FS_QWEN3_ASR_DIR=/models/qwen3-asr-hf      # a Qwen3-ASR-hf directory
    FS_ASR_PAIRS_JSONL=/data/asr/pairs.jsonl   # pairs in the SP-2 contract
                                               # format (sentinel prompt, one
                                               # audio carrier field, chosen /
                                               # rejected -- or completion /
                                               # label for KTO files)

Run::

    FS_QWEN3_ASR_DIR=... FS_ASR_PAIRS_JSONL=... \
        python -m pytest -s \
        validation_campaigns/speech_pref/test_preference_trainer_audio_dpo.py

``audio_language`` is the one knob the pair format does not carry; it is read
from the SFT plane's own ``FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE`` (or
``FS_ASR_AUDIO_LANGUAGE``) and, if neither is set, falls back to ``en`` WITH A
QUIET-DAY-PROOF ``print`` -- the trainer itself never defaults a language, and
a campaign default nobody saw must not read later as an operator declaration.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

ASR_DIR = os.environ.get("FS_QWEN3_ASR_DIR")
PAIRS_JSONL = os.environ.get("FS_ASR_PAIRS_JSONL")
LANGUAGE_ENVS = ("FS_ASR_LANGUAGE", "FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE")

_FAMILY = "qwen3-asr"
_PAIRS_USED = 8

# Fields a pair record may carry other than its audio carrier. The pairs file
# is operator-supplied and names its own carrier column, so the campaign
# derives the column from the first record instead of hardcoding one and
# silently reading a text cell as a waveform path.
_RESERVED_PAIR_FIELDS = {"prompt", "chosen", "rejected", "completion", "label"}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _asr_dir() -> Path | None:
    if not ASR_DIR:
        return None
    candidate = Path(ASR_DIR)
    return candidate if candidate.is_dir() else None


def _pairs_file() -> Path | None:
    if not PAIRS_JSONL:
        return None
    candidate = Path(PAIRS_JSONL)
    return candidate if candidate.is_file() else None


def _available() -> bool:
    return _cuda_available() and _asr_dir() is not None and _pairs_file() is not None


requires_gpu_model = pytest.mark.skipif(
    not _available(),
    reason=(
        "needs CUDA and FS_QWEN3_ASR_DIR (a Qwen3-ASR-hf dir) plus "
        "FS_ASR_PAIRS_JSONL (pairs in the SP-2 contract format)"
    ),
)


def _audio_column() -> str:
    """The pair record's audio column: build_pairs.py writes it as ``audio`` (the SP-2 contract)."""
    pairs = _pairs_file()
    assert pairs is not None
    first = json.loads(pairs.read_text(encoding="utf-8").splitlines()[0])
    if not isinstance(first.get("audio"), str) or not first["audio"].strip():
        raise AssertionError(f"{pairs}: the first pair record has no non-empty 'audio' path")
    return "audio"

def _audio_language() -> str:
    for name in LANGUAGE_ENVS:
        value = os.environ.get(name)
        if value:
            print(f"[speech_pref] using {name}={value!r} as audio_language")
            return value
    print("[speech_pref] no language env was set; falling back to 'en'")
    return "en"


def _first_pairs(count: int) -> list[dict[str, Any]]:
    pairs = _pairs_file()
    assert pairs is not None
    rows = [
        json.loads(line) for line in pairs.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) < count:
        raise AssertionError(
            f"{pairs}: {len(rows)} pair records but the control needs the first "
            f"{count}; supply at least {count} pairs"
        )
    return rows[:count]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _silent_copy(real_path: str, dst_dir: Path, index: int) -> str:
    """An all-zero wav with EXACTLY the real clip's length in ``dst_dir``.

    Same sample count and same rate: the only difference between the two
    conditions is the waveform's CONTENT. Never written over the operator's
    corpus -- the campaign must leave the measured inputs it was given alone.
    """
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(real_path, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=-1)
    dst = dst_dir / f"silent-{index}.wav"
    sf.write(str(dst), np.zeros(int(samples.shape[0]), dtype=np.float32), rate)
    return str(dst)


def _materialise_pairs(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    silent_dir: Path | None,
    audio_column: str,
) -> Path:
    """The rows as one JSONL, optionally re-pointed at silent twins."""
    written: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row = dict(row)
        if silent_dir is not None:
            row[audio_column] = _silent_copy(str(row[audio_column]), silent_dir, index)
        written.append(row)
    return _write_jsonl(out_path, written)


def _run_steps(dataset: Path, *, model_dir: str, max_steps: int) -> tuple[list[float], float]:
    """Run ``max_steps`` LoRA DPO steps over ``dataset``; (per-step losses, peak GB)."""
    import torch

    from foundationscale.rl.preference_trainer import (
        PreferenceTrainConfig,
        PreferenceTrainer,
    )

    torch.cuda.reset_peak_memory_stats()
    cfg = PreferenceTrainConfig(
        model=model_dir,
        dataset=str(dataset),
        algorithm="dpo",
        pairs_per_step=_PAIRS_USED,
        max_steps=max_steps,
        max_length=4096,
        logprob_micro_batch=2,
        learning_rate=1e-4,
        device="cuda",
        gradient_checkpointing=True,
        adapter="lora",
        adapter_rank=16,
        adapter_alpha=32.0,
        seed=0,
        audio_column=_audio_column(),
        audio_language=_audio_language(),
    )
    trainer = PreferenceTrainer(cfg)
    reports = trainer.run()
    assert len(reports) == max_steps, f"expected {max_steps} measured step(s), got {len(reports)}"
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return [entry["loss"] for entry in trainer.history], peak_gb


@requires_gpu_model
def test_audio_declared_dpo_step0_loss_is_ln2(tmp_path: Path) -> None:
    """Step 0 on a policy==reference step is exactly softplus(0) = ln(2): the
    LoRA B matrix initialises to zero and the reference IS the adapter-disabled
    policy, so the audio content -- whatever it is -- cancels out of every
    margin. A loss away from ln(2) here means the reference-disable-adapter
    path is not the one scoring an audio-declared run."""
    import torch

    pairs = _pairs_file()
    assert pairs is not None
    rows = _first_pairs(_PAIRS_USED)
    dataset = _materialise_pairs(
        rows,
        out_path=tmp_path / "pairs_real.jsonl",
        silent_dir=None,
        audio_column=_audio_column(),
    )
    try:
        losses, peak_gb = _run_steps(dataset, model_dir=str(_asr_dir()), max_steps=1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n[speech_pref_dpo] family={_FAMILY!r} audio-declared DPO step-0 "
        f"loss={losses[0]!r} peak_gb={peak_gb:.1f}"
    )
    assert abs(losses[0] - math.log(2)) < 1e-3, (
        f"audio-declared DPO step-0 loss {losses[0]!r} is not ln(2) within 1e-3 "
        f"on a policy==reference step -- the reference-disable-adapter path "
        f"may not be reached for an audio-declared run"
    )


@requires_gpu_model
def test_silent_audio_changes_the_loss_real_vs_silence_control(tmp_path: Path) -> None:
    """The control: the SAME 8 pairs, SAME model, SAME seed -- real clips vs
    silence of the same length must give a DIFFERENT loss after two steps.

    NOT step 0: at step 0 policy == reference EXACTLY (LoRA B is zero and the
    reference scores with the adapter disabled), so DPO's loss collapses to
    ln(2) whatever the tower saw -- exactly the invariant above, which would
    make this control vacuously "pass" (identical loss 0) on a CORRECT trainer
    too. Two steps and the SECOND loss compare a policy already moved by one
    gradient step shaped by the audio it was shown; real vs silence then
    diverge the policy differently and the comparison is meaningful. The
    step-0 pair is still asserted EQUAL: it is the proof that the two
    conditions hold everything except the waveform's content fixed.
    """
    import torch

    model_dir = str(_asr_dir())
    audio_column = _audio_column()
    rows = _first_pairs(_PAIRS_USED)
    real_dataset = _materialise_pairs(
        rows,
        out_path=tmp_path / "pairs_real.jsonl",
        silent_dir=None,
        audio_column=audio_column,
    )
    (tmp_path / "silent_clips").mkdir(exist_ok=True)
    silent_dataset = _materialise_pairs(
        rows,
        out_path=tmp_path / "pairs_silent.jsonl",
        silent_dir=tmp_path / "silent_clips",
        audio_column=audio_column,
    )
    assert silent_dataset != real_dataset

    try:
        losses_real, peak_real = _run_steps(real_dataset, model_dir=model_dir, max_steps=2)
        gc.collect()
        torch.cuda.empty_cache()
        losses_silent, peak_silent = _run_steps(silent_dataset, model_dir=model_dir, max_steps=2)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n[speech_pref_control] family={_FAMILY!r} "
        f"losses_real={losses_real!r} (peak {peak_real:.1f} GB) "
        f"losses_silent={losses_silent!r} (peak {peak_silent:.1f} GB)"
    )
    assert losses_real[0] == pytest.approx(losses_silent[0], abs=1e-6), (
        f"step-0 losses differ ({losses_real[0]!r} vs {losses_silent[0]!r}) "
        "before any gradient step was taken -- policy and reference should be "
        "identical regardless of clip content at step 0, so a mismatch here "
        "means something OTHER than the audio is not held fixed between the "
        "two conditions (dataset order, LoRA init, or the pairs' own text)"
    )
    assert losses_real[1] != losses_silent[1], (
        f"{_FAMILY}: step-1 loss (after one gradient step) is IDENTICAL for "
        f"real vs silent clips of the same length ({losses_real[1]!r} == "
        f"{losses_silent[1]!r}) from the same seed and the same pairs -- the "
        "model's forward pass does not appear to depend on the declared audio "
        "at all"
    )
