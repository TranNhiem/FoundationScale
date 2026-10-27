"""Decoding options of speech_repro/eval_qwen3_asr.py, without a model or a GPU.

Greedy stays the default and its generate() call is unchanged; --num-beams > 1 switches to beam
search; the threaded audio prefetch yields exactly the batches and waveforms the serial loader
yields, in manifest order.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import time
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[2]
    / "validation_campaigns"
    / "speech_repro"
    / "eval_qwen3_asr.py"
)
_SPEC = importlib.util.spec_from_file_location("eval_qwen3_asr", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def _args(**kw):
    base = {"max_new_tokens": 64, "num_beams": 1, "length_penalty": 1.0, "language": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_greedy_generate_kwargs_are_the_published_setting():
    assert ev.generate_kwargs(_args()) == {"max_new_tokens": 64, "do_sample": False}


def test_beam_generate_kwargs():
    kw = ev.generate_kwargs(_args(num_beams=4, length_penalty=0.8))
    assert kw == {
        "max_new_tokens": 64,
        "do_sample": False,
        "num_beams": 4,
        "length_penalty": 0.8,
        "early_stopping": True,
    }


@pytest.mark.parametrize("beams", [0, -2])
def test_nonpositive_beams_refused(beams):
    with pytest.raises(ValueError, match="num-beams"):
        ev.generate_kwargs(_args(num_beams=beams))


def _slow_load(path):
    time.sleep(random.random() * 0.002)  # finish out of order under threads
    return f"wave:{path}"


@pytest.mark.parametrize("batch_size", [1, 3, 7, 50])
def test_prefetch_yields_the_serial_batches_in_order(batch_size):
    rows = [{"id": i, "audio": f"a{i}.flac"} for i in range(23)]
    serial = [
        ([r["id"] for r in b], w)
        for b, w in ev.iter_audio_batches(rows, batch_size, _slow_load, workers=0)
    ]
    threaded = [
        ([r["id"] for r in b], w)
        for b, w in ev.iter_audio_batches(rows, batch_size, _slow_load, workers=6)
    ]
    assert serial == threaded
    assert [i for ids, _ in serial for i in ids] == list(range(23))
    assert all(w == [f"wave:a{i}.flac" for i in ids] for ids, w in serial)


def test_prefetch_on_empty_manifest():
    assert list(ev.iter_audio_batches([], 4, _slow_load, workers=3)) == []


class _Inputs(dict):
    def to(self, *_):
        return self


class _Tok:
    eos_token_id = 9


class _Proc:
    tokenizer = _Tok()

    def apply_transcription_request(self, audio, language):
        return _Inputs(input_ids=torch.zeros((len(audio), 2), dtype=torch.long))

    def decode(self, row, skip_special_tokens, return_format):
        return f" {'-'.join(str(int(t)) for t in row)} "


class _Model:
    device, dtype = "cpu", torch.float32

    def __init__(self):
        self.calls = []
        self.generation_config = argparse.Namespace(eos_token_id=[9])

    def generate(self, **kw):
        self.calls.append({k: v for k, v in kw.items() if k != "input_ids"})
        n = kw["input_ids"].shape[0]
        # row 0 stops (eos 9), row 1 spends the whole budget -> truncated
        return torch.tensor([[0, 0, 5, 9], [0, 0, 5, 6]][:n])


def test_step_generate_passes_beam_settings_and_counts_truncation():
    model = _Model()
    hyps, truncated = ev.step_generate(
        model, _Proc(), ["w0", "w1"], _args(num_beams=4, length_penalty=0.8)
    )
    assert model.calls == [
        {
            "max_new_tokens": 64,
            "do_sample": False,
            "num_beams": 4,
            "length_penalty": 0.8,
            "early_stopping": True,
        }
    ]
    assert hyps == ["5-9", "5-6"]
    assert truncated == 1
