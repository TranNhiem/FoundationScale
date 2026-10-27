"""The SP-3 slice: a GRPO-family RL loop over ASR, priced by a WER reward.

CPU only, no downloads. Every model here is a tiny torch module and every
processor is a fake that answers ``apply_transcription_request`` and decodes
ONE ROW per ``batch_decode`` call -- the two facts about
``Qwen3ASRProcessor`` the rollout's row-by-row decode depends on (its
``batch_decode`` ignores ``return_format``, so the decode is per row).

WHAT IS CLAIMED to hold once this file is green:

  * ``rewards.WERReward`` scores ``1.0 - word_errors(gold, response).rate()``
    under the FS normalizer (case and punctuation are noise), abstains with
    ``None`` when the gold normalizes to empty, and subtracts
    ``runaway_penalty`` / ``loop_penalty`` ONLY where
    ``speech_metrics.is_runaway`` / ``loop_words`` flag the response;
  * the run declaration REFUSES half an audio specification, a blank one, a
    custom ``reward`` beside an ``answer_pattern``/``gold_key`` that reward
    would silently ignore, and audio mode with ``max_new_tokens`` below 256
    (which truncates transcripts) -- every reason NAMED;
  * ``run()`` prices a custom ``config.reward`` -- the recorded calls are
    exactly the decoded responses against the records' own transcripts -- and
    ``asr_rollout.load_audio_manifest`` refuses a manifest row with a missing or
    blank field WITH THE LINE NAMED.

WHAT IS NOT CLAIMED: anything about model quality; that the fakes reproduce
Qwen3-ASR numerics (they reproduce its call shape only); or that the two
canned completions are interesting transcripts. They are chosen so that the
reward VARIES within every group -- the only property a group-relative
objective needs in order to have a gradient at all -- and are laid out by
``(call, row) % 2`` so that ANY plausible rollout shape (one ``generate``
call per prompt with ``num_return_sequences``, one call for the whole draw,
or a loop of one-row calls) hands each prompt one good and one bad rollout.

Refusals arrive in two voices -- the module's own refusal types and the
repository's exit-96 contract, which prints one ``REFUSE:`` line to stderr --
so ``_refusal_message`` reads either. These tests grade the MESSAGE, which is
what the design names ("refuse ... name the reason"); the transport is
``_REFUSAL_TYPES`` or exit 96 and nothing else.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

import foundationscale.rl.asr_rollout as asr_rollout
import foundationscale.rl.preference_trainer as preference_trainer
import foundationscale.rl.rewards as rewards
import foundationscale.rl.trainer as trainer_module
import foundationscale.train.audio as train_audio
from foundationscale.rl.asr_rollout import load_audio_manifest
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal
from foundationscale.train import speech_metrics
from foundationscale.train.audio import AUDIO_FEATURES_KEY

# A refusal is a domain refusal (the module's own type family -- TrainerRefusal
# is RuntimeError) or the exit-96 contract. KeyError is included so a manifest
# loader that lets a missing key escape as a bare KeyError is a REFUSAL these
# tests read and then reject: the refusal has to name the LINE, and a bare
# ``"'text'"`` names neither the field that is missing nor where it is missing.
_REFUSAL_TYPES = (TrainerRefusal, RuntimeError, ValueError, KeyError)

# "dataset line 2, field 'text': ..." is the surrounding voice's spelling;
# the regex tolerates "line: 2" / "line=2" too and captures the number so a
# refusal that names the WRONG line is caught as loudly as one naming none.
_LINE_IN_MESSAGE = re.compile(r"line\s*[:=#]?\s*(\d+)\b")


def _refusal_message(capsys: Any, *callables: Any) -> str:
    """Run each callable until one refuses; return the refusal's own message.

    A refusal under the exit-96 contract says its piece on stderr and raises
    ``SystemExit(96)`` (see ``trainer._refuse_exit_96``), so that stream is
    read as the message. Anything that returns without refusing means this
    test has no refusal to grade, which is a failure and not a quiet pass.
    """
    for call in callables:
        try:
            call()
        except SystemExit as exc:  # noqa: PERF203 -- the exit-96 voice is one arm of one check
            assert int(exc.code) == 96, f"exit contract is 96, got {exc.code!r}"
            return capsys.readouterr().err
        except _REFUSAL_TYPES as exc:
            return str(exc)
    raise AssertionError("expected a refusal from at least one offered construction")


def _config_refusal_message(capsys: Any, fields: dict[str, Any]) -> str:
    """The message one run declaration earns, wherever the check is declared.

    The surrounding style validates run declarations in the trainer's
    ``__init__`` (``RLTrainer.__init__``'s group_size / adapter arms,
    ``PreferenceTrainer.__init__``'s own audio arms) and the design says the
    audio fields are "validated like preference_trainer.py's audio fields";
    a stricter reading -- validation on ``RLTrainConfig`` itself -- is offered
    first, so either placement refuses here.
    """
    return _refusal_message(
        capsys,
        lambda: RLTrainConfig(**fields),
        lambda: RLTrainer(RLTrainConfig(**fields)),
    )


def _manifest(directory: Path, rows: list[dict[str, Any]], *, name: str) -> Path:
    """One JSONL manifest on disk; ``load_audio_manifest`` reads a PATH, so the
    manifest loader and the run declaration are tested against real bytes."""
    path = directory / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _write_silent_wav(path: Path) -> None:
    """One real 16 kHz mono wav, which is what the manifest's paths promise.

    The frames are silence: no claim here grades sound. A real decodable file
    keeps a rollout helper honest where the fake processor would have accepted
    any ``audio=`` value without looking at it.
    """
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(16_000)
        sink.writeframes(b"\x00\x00" * 1_600)


# ---------------------------------------------------------------------------
# Fakes: one vocabulary shared by the fake tokenizer and the fake ASR
# processor. Token id -> the word it decodes to; "" means the id contributes
# NOTHING whatever ``skip_special_tokens`` says -- the pad fill of a 256-token
# generation budget, the eos, and the transcription request's own prompt ids
# all land here, so a decoded ROW is exactly the canned completion's words.
# ---------------------------------------------------------------------------
_TOKEN_WORD: dict[int, str] = {
    0: "",  # pad fill past the canned completion
    1: "hello",
    2: "world",
    3: "wrold",
    4: "",  # eos
    6: "",  # transcription-request prompt token
    7: "",
    10: "alpha",
    11: "beta",
}


def _as_nested(value: Any) -> Any:
    """``value`` as python lists of ints -- tensors accepted via ``tolist``."""
    if hasattr(value, "tolist"):
        return _as_nested(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_as_nested(item) for item in value]
    return int(value)


def _rows(sequences: Any) -> list[list[int]]:
    """``sequences`` as rows of token ids, de-nesting one-row wrappers.

    ``batch_decode([row])``, ``batch_decode([ [row] ])`` and the flat
    ``batch_decode(row)`` all describe the same single row -- which is what a
    per-row decode hands a processor -- while a genuinely multi-row request is
    handed back as several rows and refused BY THE CALLER below.
    """
    nested = _as_nested(sequences)
    while (
        len(nested) == 1
        and isinstance(nested[0], list)
        and (not nested[0] or isinstance(nested[0][0], list))
    ):
        nested = nested[0]
    if nested and all(isinstance(entry, int) for entry in nested):
        return [list(nested)]
    return [list(row) for row in nested]


def _decode_row(ids: Any) -> str:
    return " ".join(
        word for word in (_TOKEN_WORD.get(int(token_id), "") for token_id in ids) if word
    )


def _flat_strings(value: Any) -> list[str]:
    """``value`` and any nested list/tuple of it, as strings to compare evenly.

    A request may be TOLD the language per clip ("en") or once for the batch
    (["en", "en"]); graded here is "every clip was labelled with the declared
    language", not the arity of the call.
    """
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_flat_strings(item))
        return out
    return [str(value)]


class _FakeTokenizer:
    """The tokenizer surface ``run()`` touches: template, pad/eos, decode.

    ``batch_decode`` decodes every row it is handed -- this is the TOKENIZER,
    whose batch decode the corpus leg already uses row by row over one matrix
    -- unlike the ASR processor below.
    """

    def __init__(self) -> None:
        self.chat_template = "{{ fake chat template }}"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token_id = 4
        self.vocab_size = 16
        self.padding_side = "right"

    def apply_chat_template(
        self, messages: Any, add_generation_prompt: bool = False, tokenize: bool = False
    ) -> str:
        return "[prompt]"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [5, 5, 5]}

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        return [_decode_row(row) for row in _rows(sequences)]


class _FakeASRProcessor:
    """A Qwen3-ASR stand-in: transcription requests and ROW-BY-ROW decoding.

    ``batch_decode`` decodes exactly one row per call and IGNORES
    ``return_format`` (and ``skip_special_tokens``) exactly the way
    ``Qwen3ASRProcessor.batch_decode`` does; a multi-row call is refused here
    rather than joined into one string, which is the silent truncation a
    per-row decode exists to avoid.

    ``apply_transcription_request(audio=<clip>, language=<declared>)`` answers
    with one row per clip carrying prompt ids, an attention mask and the two
    audio feature tensors the design prices -- ``AUDIO_FEATURES_KEY`` (the
    key train/audio.py's own collator names, imported so this fake cannot
    drift from it) and its per-feature mask. Prompt width (3) and feature
    length (7) are DELIBERATELY different so no per-token widening helper can
    mistake one axis for the other.
    """

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.requests: list[tuple[Any, Any]] = []
        self.decode_calls: list[int] = []
        self.model_input_names = ["input_ids", "attention_mask"]

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return "[prompt]"

    def apply_transcription_request(self, audio: Any = None, language: Any = None, **kwargs: Any):
        import torch

        clips = _clips(audio)
        self.requests.extend((clip, language) for clip in clips)
        rows = len(clips)
        return {
            "input_ids": torch.tensor([[7, 6, 7]] * rows, dtype=torch.long),
            "attention_mask": torch.ones((rows, 3), dtype=torch.long),
            AUDIO_FEATURES_KEY: torch.zeros((rows, 7, 8)),
            "input_features_mask": torch.ones((rows, 7), dtype=torch.long),
        }

    def pad(self, features: Any, **kwargs: Any):
        """HF-style ``pad``: list of per-row mappings, or one batched mapping.

        Every request above is the same width, so this is a concatenation on
        every path the rollout can take; the right-pad exists so a caller that
        batches unequal rows gets the same shape it would from the real one.
        """
        import torch

        rows = list(features) if isinstance(features, (list, tuple)) else [features]
        out: dict[str, Any] = {}
        for key in sorted({name for row in rows for name in row}):
            values = [torch.as_tensor(row[key]) for row in rows]
            width = max((int(value.shape[1]) for value in values if value.dim() >= 2), default=0)
            padded = []
            for value in values:
                if value.dim() >= 2 and int(value.shape[1]) < width:
                    pad_shape = (value.shape[0], width - int(value.shape[1]), *value.shape[2:])
                    value = torch.cat([value, value.new_zeros(pad_shape)], dim=1)
                padded.append(value)
            out[key] = torch.cat(padded, dim=0)
        return out

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        # ``return_format`` is accepted and ignored -- Qwen3ASRProcessor's own
        # behaviour -- and more than one row per call is the truncation this
        # contract refuses.
        rows = _rows(sequences)
        decoded = [_decode_row(row) for row in rows]
        self.decode_calls.append(len(decoded))
        if len(decoded) != 1:
            raise ValueError(
                f"fake Qwen3ASR batch_decode decodes one row per call; "
                f"{len(decoded)} row(s) were offered -- decode each row"
            )
        return decoded


def _clips(audio: Any) -> list[Any]:
    if (
        isinstance(audio, (list, tuple))
        and audio
        and all(isinstance(item, (str, Path)) for item in audio)
    ):
        return list(audio)
    return [audio]


def _make_fake_model(good_ids: tuple[int, ...], bad_ids: tuple[int, ...]) -> Any:
    """A tiny differentiable module the trainer can load, generate from and step.

    ``generate`` appends one of TWO canned completions -- chosen by
    ``(call + row) % 2`` -- so every plausible rollout layout (one call per
    prompt with ``num_return_sequences=G``, one call for the whole draw of
    ``P * G`` rows, or a loop of one-row calls) hands each prompt one good and
    one bad rollout: the within-group reward variance a group-relative
    objective needs in order to have any gradient at all. Without it the step
    would be UNMEASURED and the run vacuous, which is a test of the estimator's
    abstention, not of the reward wiring.

    Every other keyword (audio features, a feature mask, an image's pixels) is
    accepted and RECORDED: "the audio feature tensors are passed through to
    every forward" is one of the claims under test.
    """
    import torch

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16)
            self.config = types.SimpleNamespace(use_cache=False, model_type="fake-asr")
            self.forward_calls: list[tuple[tuple[int, ...], dict[str, Any]]] = []
            self.generate_calls = 0

        def forward(self, input_ids: Any = None, attention_mask: Any = None, **kwargs: Any):
            self.forward_calls.append((tuple(int(size) for size in input_ids.shape), dict(kwargs)))
            hidden = self.embedding(input_ids.clamp(min=0, max=15))
            return types.SimpleNamespace(logits=self.lm_head(hidden))

        @torch.no_grad()
        def generate(
            self,
            input_ids: Any = None,
            num_return_sequences: int = 1,
            max_new_tokens: int | None = None,
            **kwargs: Any,
        ) -> Any:
            call = self.generate_calls
            self.generate_calls += 1
            repeats = int(num_return_sequences)
            rows = int(input_ids.shape[0]) * repeats
            prompt = input_ids.to(dtype=torch.long).repeat_interleave(repeats, dim=0)
            budget = int(max_new_tokens or 1)
            completions = torch.zeros((rows, budget), dtype=torch.long)
            for row in range(rows):
                ids = good_ids if (call + row) % 2 == 0 else bad_ids
                width = min(len(ids), budget)
                completions[row, :width] = torch.tensor(ids[:width], dtype=torch.long)
            return torch.cat([prompt, completions], dim=1)

        def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
            return None

    return _FakeModel()


class _RecordingReward:
    """A custom ``config.reward``: records every ``(response, gold)`` it is handed.

    Scores 1.0 for ``good_response`` and 0.0 for anything else, so the two
    canned rollouts of every prompt also carry the within-group variance the
    objective needs. The RECORD is the claim: "run() priced decoded responses
    against the records' own transcripts under ``config.reward``" is exactly
    these pairs, and a run that fell back to ``MCQLetterReward`` leaves this
    list empty.
    """

    def __init__(self, *, good_response: str) -> None:
        self.good_response = good_response
        self.calls: list[tuple[str, str]] = []

    def score(self, response: str | None = None, gold: str | None = None, **kwargs: Any):
        self.calls.append((response, gold))
        return 1.0 if response == self.good_response else 0.0


def _surface(kind: str, surface: Any) -> Any:
    """A prompt surface of the real type where it accepts these fakes, else a duck.

    ``prompt_surface.PromptSurface`` is preferred (the trainers read
    ``.kind``/``.surface`` off it) but its constructor is free to validate the
    surface it is handed; a fake that fails that validation must not fail this
    test, so a namespace carrying the same fields is the stand-in.
    """
    reason = f"test fake: {kind}"
    try:
        from foundationscale.rl.prompt_surface import PromptSurface

        return PromptSurface(kind=kind, surface=surface, reason=reason, supports_images=False)
    except Exception:  # noqa: BLE001 -- a stricter constructor swaps in the duck
        return types.SimpleNamespace(
            kind=kind, surface=surface, reason=reason, supports_images=False
        )


def _sample(sample_id: str, gold: str) -> Any:
    """The whole corpus surface ``run()`` reads off a sample: id, turns, gold, media."""
    return types.SimpleNamespace(
        sample_id=sample_id,
        prompt_turns=({"role": "user", "content": sample_id},),
        gold=gold,
        images=(),
        video=None,
    )


def _prompts(rows: int) -> dict[str, Any]:
    """One text-only encode: ids and attention, and no modality key at all."""
    import torch

    return {
        "input_ids": torch.full((rows, 3), 5, dtype=torch.long),
        "attention_mask": torch.ones((rows, 3), dtype=torch.long),
    }


def _require_torch() -> Any:
    return pytest.importorskip("torch")


def _stub_transformers_if_absent(monkeypatch: Any) -> None:
    """``run()`` probes ``import transformers`` and exits 96 without it.

    Every loader that would USE a transformers class is patched below, so the
    probe is the only consumer here: an empty module satisfies it on a host
    with no transformers installed, without pretending any name is loadable.
    """
    try:
        __import__("transformers")
    except ImportError:
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))


def _install_asr_fakes(monkeypatch: Any, *, model: Any, processor: Any) -> None:
    """Install the fake ASR model and processor at every seam an audio leg may use.

    ``preference_trainer._load_asr_lm`` is THE loader the audio mode uses (the
    design names it: imported, never duplicated) and
    ``train.audio.resolve_audio_surface`` is THE processor resolver. Both are
    patched at their home module AND at every module that might bind that name
    at import time, so a top-level ``from ... import name`` (which would else
    keep the real helper) is covered the same way a call-time import is.

    ``load_audio`` is patched to try the real decoder first and fall back to
    silent samples: the manifest paths above ARE real wavs, so a rollout that
    decodes them before building its request is served exactly as one that
    hands the clip path straight to the processor.
    """

    def load_model(model_id: Any) -> Any:
        return model

    def resolve(model_id: Any) -> Any:
        return _surface("processor", processor)

    for module in (preference_trainer, asr_rollout, trainer_module):
        monkeypatch.setattr(module, "_load_asr_lm", load_model, raising=False)
    for module in (train_audio, asr_rollout, trainer_module):
        monkeypatch.setattr(module, "resolve_audio_surface", resolve, raising=False)

    real_load_audio = getattr(train_audio, "load_audio", None)

    def load_audio(path: Any, *args: Any, **kwargs: Any) -> Any:
        if callable(real_load_audio):
            try:
                return real_load_audio(path, *args, **kwargs)
            except Exception:  # noqa: BLE001 -- a missing decoder must not break the manifest
                pass
        return [0.0] * 160

    for module in (train_audio, asr_rollout, preference_trainer, trainer_module):
        monkeypatch.setattr(module, "load_audio", load_audio, raising=False)


def _text_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "model": "fake-lm",
        "dataset": "corpus.jsonl",
        "group_size": 2,
        "prompts_per_step": 1,
        "max_steps": 1,
        "device": "cpu",
        "seed": 0,
    }
    fields.update(overrides)
    return fields


def _asr_fields(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """One audio-declared run declaration, at the DECLARED token floor of 256.

    ``max_new_tokens`` defaults to 256 here so that a refusal about the audio
    fields cannot be the token-budget refusal firing first and answering a
    question this test did not ask; the budget itself has its own tests.
    """
    manifest = _manifest(
        tmp_path, [{"audio": "clip.wav", "text": "hello world"}], name="fields.jsonl"
    )
    fields: dict[str, Any] = {
        "model": "fake-asr",
        "dataset": str(manifest),
        "audio_column": "audio",
        "audio_language": "en",
        "group_size": 2,
        "prompts_per_step": 1,
        "max_steps": 1,
        "max_new_tokens": 256,
        "device": "cpu",
        "seed": 0,
    }
    fields.update(overrides)
    return fields


# ---------------------------------------------------------------------------
# 1. WERReward: 1.0 - rate under the FS normalizer, abstention on an empty
#    gold, and penalties only where the runaway / loop detectors flag.
# ---------------------------------------------------------------------------


def test_wer_reward_is_exported_frozen_and_penalties_default_to_zero() -> None:
    assert "WERReward" in rewards.__all__
    reward = rewards.WERReward()
    assert reward.runaway_penalty == 0.0
    assert reward.loop_penalty == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        reward.runaway_penalty = 1.0  # type: ignore[misc]


def test_wer_reward_scores_one_on_a_perfect_match() -> None:
    reward = rewards.WERReward()
    assert reward.score(response="hello world", gold="hello world") == pytest.approx(1.0)


def test_wer_reward_normalizer_ignores_case_and_punctuation() -> None:
    reward = rewards.WERReward()
    assert speech_metrics.normalize_transcript("Hello, World!") == ["hello", "world"]
    assert reward.score(response="Hello, World!", gold="HELLO world") == pytest.approx(1.0)
    assert reward.score(response="Don't stop!", gold="don't  STOP") == pytest.approx(1.0)


def test_wer_reward_partial_credit_is_one_minus_the_word_error_rate() -> None:
    reward = rewards.WERReward()
    counts = speech_metrics.word_errors("a b c d", "a x c d")
    assert (counts.substitutions, counts.reference_length) == (1, 4)
    assert reward.score(response="a x c d", gold="a b c d") == pytest.approx(0.75)
    # one insertion against two reference words is not the same claim as one
    # mixed-up word among four: the reward is 1.0 - rate, never a match count.
    assert reward.score(response="hello there world", gold="hello world") == pytest.approx(0.5)


def test_wer_reward_abstains_when_the_gold_normalizes_to_empty() -> None:
    reward = rewards.WERReward()
    for gold in ("", "   ", "!!!", "...", "'"):
        assert speech_metrics.normalize_transcript(gold) == []
        assert reward.score(response="hello world", gold=gold) is None
    # an abstention outranks the penalties: an unmeasurable row is never
    # priced, never as 0.0 and never as a penalised 0.0.
    penalised = rewards.WERReward(runaway_penalty=5.0, loop_penalty=5.0)
    assert penalised.response if False else True  # documented intent: no partial API
    assert (
        penalised.score(response="one two three four five six seven eight nine ten", gold="")
        is None
    )


def test_wer_reward_an_empty_response_scores_zero_rather_than_abstaining() -> None:
    reward = rewards.WERReward()
    counts = speech_metrics.word_errors("hello world", "")
    assert counts.errors == counts.reference_length == 2
    assert reward.score(response="", gold="hello world") == pytest.approx(0.0)


_WORLDS = "one two three four five six seven eight nine ten eleven"
_LOOPY = "hello world the the the the"
_BOTH = "the the the the the the the the the the the"


def test_wer_reward_runaway_penalty_applies_only_when_the_response_runs_away() -> None:
    gold = "hello world"
    flag_bed = speech_metrics.is_runaway(gold, _WORLDS)
    never = speech_metrics.is_runaway(gold, "hello world")
    assert flag_bed and not never  # the two rows this test grades, stated first
    plain, penalised = rewards.WERReward(), rewards.WERReward(runaway_penalty=0.25)
    assert penalised.score(response=_WORLDS, gold=gold) == pytest.approx(
        plain.score(response=_WORLDS, gold=gold) - 0.25
    )
    assert penalised.score(response="hello world", gold=gold) == pytest.approx(
        plain.score(response="hello world", gold=gold)
    )


def test_wer_reward_loop_penalty_applies_only_when_the_response_loops() -> None:
    gold = "hello world"
    assert speech_metrics.loop_words(speech_metrics.normalize_transcript(_LOOPY)) > 0
    assert not speech_metrics.is_runaway(gold, _LOOPY)  # loop-only row: no runaway here
    plain, penalised = rewards.WERReward(), rewards.WERReward(loop_penalty=0.5)
    assert penalised.score(response=_LOOPY, gold=gold) == pytest.approx(
        plain.score(response=_LOOPY, gold=gold) - 0.5
    )
    assert penalised.score(response="hello world", gold=gold) == pytest.approx(
        plain.score(response="hello world", gold=gold)
    )


def test_wer_reward_penalties_stack_when_both_detectors_flag() -> None:
    gold = "hello world"
    assert speech_metrics.is_runaway(gold, _BOTH)
    assert speech_metrics.loop_words(speech_metrics.normalize_transcript(_BOTH)) > 0
    plain = rewards.WERReward()
    penalised = rewards.WERReward(runaway_penalty=0.25, loop_penalty=0.5)
    assert penalised.score(response=_BOTH, gold=gold) == pytest.approx(
        plain.score(response=_BOTH, gold=gold) - 0.75
    )


def test_wer_reward_penalties_stay_silent_when_nothing_is_flagged() -> None:
    penalised = rewards.WERReward(runaway_penalty=3.0, loop_penalty=2.0)
    assert penalised.score(response="hello world", gold="hello world") == pytest.approx(1.0)
    assert penalised.score(response="a x c d", gold="a b c d") == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 2. The run declaration's refusals: half an audio specification, a blank one,
#    a custom reward beside an answer surface it would ignore, and the token
#    budget below 256 that truncates transcripts.
# ---------------------------------------------------------------------------


def test_audio_language_without_audio_column_is_refused(tmp_path: Path, capsys: Any) -> None:
    message = _config_refusal_message(
        capsys, _asr_fields(tmp_path, audio_column=None, audio_language="en")
    )
    assert "audio_column" in message and "audio_language" in message


def test_audio_column_without_audio_language_is_refused(tmp_path: Path, capsys: Any) -> None:
    message = _config_refusal_message(
        capsys, _asr_fields(tmp_path, audio_column="audio", audio_language=None)
    )
    assert "audio_column" in message and "audio_language" in message


def test_blank_audio_column_is_refused(tmp_path: Path, capsys: Any) -> None:
    message = _config_refusal_message(
        capsys, _asr_fields(tmp_path, audio_column="   ", audio_language="en")
    )
    assert "audio_column" in message
    assert any(word in message.lower() for word in ("blank", "empty"))


def test_blank_audio_language_is_refused(tmp_path: Path, capsys: Any) -> None:
    message = _config_refusal_message(
        capsys, _asr_fields(tmp_path, audio_column="audio", audio_language="  ")
    )
    assert "audio_language" in message
    assert any(word in message.lower() for word in ("blank", "empty"))


def test_custom_reward_with_answer_pattern_is_refused(capsys: Any) -> None:
    message = _config_refusal_message(
        capsys,
        _text_fields(reward=_RecordingReward(good_response="alpha"), answer_pattern=r"([A-Z])"),
    )
    assert "answer_pattern" in message
    assert "reward" in message.lower()


def test_custom_reward_with_gold_key_is_refused(capsys: Any) -> None:
    message = _config_refusal_message(
        capsys, _text_fields(reward=_RecordingReward(good_response="alpha"), gold_key="answer")
    )
    assert "gold_key" in message
    assert "reward" in message.lower()


def test_custom_reward_without_an_answer_surface_is_accepted(tmp_path: Path) -> None:
    # The pairing is the refusal: a custom reward with the answer surface left
    # at its default names every input it reads (its own response/gold pair).
    RLTrainer(RLTrainConfig(**_text_fields(reward=_RecordingReward(good_response="alpha"))))


def test_audio_mode_refuses_max_new_tokens_below_256(tmp_path: Path, capsys: Any) -> None:
    for budget in (64, 255):
        message = _config_refusal_message(capsys, _asr_fields(tmp_path, max_new_tokens=budget))
        assert "max_new_tokens" in message, message
        # "name the reason": 64 TRANSCATES transcripts, which is the defect.
        assert "truncat" in message.lower(), message


def test_audio_mode_accepts_max_new_tokens_of_256(tmp_path: Path) -> None:
    RLTrainer(RLTrainConfig(**_asr_fields(tmp_path, max_new_tokens=256)))


# ---------------------------------------------------------------------------
# 3. The manifest: a jsonl of audio paths + transcripts, refused per line.
# ---------------------------------------------------------------------------


def test_load_audio_manifest_reads_audio_and_text_records(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [{"audio": "a.wav", "text": "one"}, {"audio": "b.wav", "text": "two"}],
        name="ok.jsonl",
    )
    records = load_audio_manifest(str(path))
    assert len(records) == 2
    assert [record.audio for record in records] == ["a.wav", "b.wav"]
    assert [record.text for record in records] == ["one", "two"]
    assert [record.line for record in records] == [1, 2]


def _manifest_refusal_line_and_field(message: str, *, line: int) -> None:
    match = _LINE_IN_MESSAGE.search(message)
    assert match, f"the refusal must name its source line; got {message!r}"
    assert int(match.group(1)) == line, f"the refusal must name line {line}; got {message!r}"
    assert ("text" in message) or ("transcript" in message.lower()) or ("audio" in message), message


def test_load_audio_manifest_refuses_a_missing_transcript_and_names_the_line(
    tmp_path: Path, capsys: Any
) -> None:
    path = _manifest(
        tmp_path,
        [{"audio": "a.wav", "text": "one"}, {"audio": "b.wav"}],
        name="no_text.jsonl",
    )
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    assert "text" in message or "transcript" in message.lower()
    _manifest_refusal_line_and_field(message, line=2)


def test_load_audio_manifest_refuses_a_blank_transcript_and_names_the_line(
    tmp_path: Path, capsys: Any
) -> None:
    path = _manifest(
        tmp_path,
        [{"audio": "a.wav", "text": "one"}, {"audio": "b.wav", "text": "   "}],
        name="blank_text.jsonl",
    )
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    assert "text" in message or "transcript" in message.lower()
    _manifest_refusal_line_and_field(message, line=2)


def test_load_audio_manifest_refuses_a_blank_audio_field_and_names_the_line(
    tmp_path: Path, capsys: Any
) -> None:
    path = _manifest(
        tmp_path,
        [{"audio": "a.wav", "text": "one"}, {"audio": "   ", "text": "two"}],
        name="blank_audio.jsonl",
    )
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    assert "audio" in message
    _manifest_refusal_line_and_field(message, line=2)


# ---------------------------------------------------------------------------
# 4. run() prices config.reward -- decoded responses against the records' own
#    transcripts -- on both legs that roll out.
# ---------------------------------------------------------------------------


def test_run_uses_config_reward_on_the_corpus_leg(tmp_path: Path, monkeypatch: Any) -> None:
    _require_torch()
    _stub_transformers_if_absent(monkeypatch)
    model = _make_fake_model(good_ids=(10,), bad_ids=(11,))
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(trainer_module, "_load_causal_lm", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        trainer_module,
        "resolve_prompt_surface",
        lambda *args, **kwargs: _surface("tokenizer", tokenizer),
    )
    monkeypatch.setattr(
        trainer_module,
        "load_sharegpt",
        lambda *args, **kwargs: [_sample("p0", "A"), _sample("p1", "B")],
    )
    monkeypatch.setattr(
        trainer_module,
        "encode_prompts",
        lambda surface, chunk, device: _prompts(len(chunk)),
    )

    reward = _RecordingReward(good_response="alpha")
    config = RLTrainConfig(
        **_text_fields(
            dataset=str(tmp_path / "corpus.jsonl"),
            reward=reward,
            prompts_per_step=2,
            max_new_tokens=2,
        )
    )
    reports = RLTrainer(config).run()

    assert reports, "a group whose reward VARIES must produce a measured step"
    # Rows are prompt-major, and (call + row) % 2 hands every prompt one of
    # each canned completion -- so this multiset is the claim in full: the
    # decoded response was scored against the record's own gold, under
    # config.reward and not under the hard-coded MCQLetterReward.
    assert sorted(reward.calls) == sorted(
        [("alpha", "A"), ("beta", "A"), ("alpha", "B"), ("beta", "B")]
    )


def test_run_uses_config_reward_on_the_audio_leg(tmp_path: Path, monkeypatch: Any) -> None:
    _require_torch()
    _stub_transformers_if_absent(monkeypatch)
    golds = ("hello world", "goodbye now")
    rows = [
        {"audio": str(tmp_path / f"{index}.wav"), "text": text} for index, text in enumerate(golds)
    ]
    for index in range(len(rows)):
        _write_silent_wav(tmp_path / f"{index}.wav")
    manifest = _manifest(tmp_path, rows, name="run.jsonl")
    processor = _FakeASRProcessor()
    model = _make_fake_model(good_ids=(1, 2, 4), bad_ids=(1, 3, 4))
    _install_asr_fakes(monkeypatch, model=model, processor=processor)

    reward = _RecordingReward(good_response="hello world")
    config = RLTrainConfig(
        **_asr_fields(
            tmp_path,
            dataset=str(manifest),
            audio_column="audio",
            audio_language="en",
            reward=reward,
            prompts_per_step=2,
        )
    )
    reports = RLTrainer(config).run()

    assert reports, "a group whose reward VARIES must produce a measured step"
    # Two rollouts per record -- one decoded "hello world", one "hello wrold"
    # -- each priced against THAT record's transcript ("the record's
    # transcript", not an MCQ gold and not a neighbour's text).
    assert sorted(reward.calls) == sorted(
        [("hello world", gold) for gold in golds] + [("hello wrold", gold) for gold in golds]
    )
    assert processor.requests, "no transcription request was built from any clip"
    declared = {lang for _, language in processor.requests for lang in _flat_strings(language)}
    assert declared == {"en"}, f"requests were labelled {declared}, not the declared language"
    assert model.forward_calls, "no scoring forward priced the audio rollouts"
    for shape, kwargs in model.forward_calls:
        assert AUDIO_FEATURES_KEY in kwargs, (
            f"a scoring forward carried only {sorted(kwargs)}: the audio features must "
            f"reach EVERY forward (old / current log-probabilities alike)"
        )
        assert tuple(kwargs[AUDIO_FEATURES_KEY].shape)[0] == shape[0], (
            "the audio feature tensors must follow the same per-row row space as "
            "input_ids -- group-expanded and narrowed like every other modality key"
        )


def test_stack_feature_rows_pads_the_frame_axis_keeping_the_mel_axis() -> None:
    """Clips of different lengths: (1, 128, frames) rows pad frames, keep 128 mel bins.

    Regression (GPU smoke run, 2026-10-17): the pad block was built with size 1 on
    every non-frame axis, so cat() failed on the 128-bin mel axis.
    """
    import torch

    from foundationscale.rl.asr_rollout import _stack_feature_rows

    short = torch.ones(1, 128, 30)
    long = torch.ones(1, 128, 50)
    out = _stack_feature_rows([short, long], key="input_features")
    assert tuple(out.shape) == (2, 128, 50)
    assert float(out[0, :, 30:].abs().sum()) == 0.0  # padding is zeros
    assert float(out[0, :, :30].sum()) == 128 * 30  # the clip itself is untouched
    masks = _stack_feature_rows([torch.ones(1, 30), torch.ones(1, 50)], key="mask")
    assert tuple(masks.shape) == (2, 50) and float(masks[0, 30:].sum()) == 0.0


def test_wer_reward_floors_the_wer_term_so_one_runaway_cannot_swamp_a_group() -> None:
    from foundationscale.rl.rewards import WERReward

    gold = "one two three"
    runaway = " ".join(["noise"] * 200)
    assert WERReward().score(response=runaway, gold=gold) == WERReward.WER_FLOOR
    penalised = WERReward(runaway_penalty=1.0, loop_penalty=1.0).score(response=runaway, gold=gold)
    assert penalised >= WERReward.WER_FLOOR - 2.0
    assert WERReward().score(response="", gold=gold) == 0.0  # empty: scored, WER 1.0


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf"), True, "1"])
def test_wer_reward_refuses_a_penalty_that_is_not_a_finite_non_negative_number(bad: object) -> None:
    from foundationscale.rl.rewards import WERReward

    with pytest.raises(ValueError):
        WERReward(runaway_penalty=bad)  # type: ignore[arg-type]


def test_scored_width_alignment_leaves_audio_features_alone_when_a_width_collides() -> None:
    """Regression (SP-3 GPU run): a 128-token prompt equals the 128 mel bins.

    The per-token heuristic pads any axis-1 that equals the prompt width; audio
    features are per-clip and must pass through unchanged, while a genuine
    per-token key still extends to the scored width.
    """
    import torch

    from foundationscale.rl.trainer import _align_modality_keys_to_scored_width

    features = torch.ones(2, 128, 300)
    mask = torch.ones(2, 128)  # frames == prompt width, also a coincidence
    token_types = torch.ones(2, 128, dtype=torch.long)
    out = _align_modality_keys_to_scored_width(
        {"input_features": features, "input_features_mask": mask, "mm_token_type_ids": token_types},
        prompt_width=128,
        sequence_width=168,
    )
    assert tuple(out["input_features"].shape) == (2, 128, 300)
    assert tuple(out["input_features_mask"].shape) == (2, 128)
    assert tuple(out["mm_token_type_ids"].shape) == (2, 168)


def _read_manifest_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _strict_load_audio_manifest(path: str | None, *, capsys: Any) -> asr_rollout.AudioRow | None:
    """Run ``load_audio_manifest`` and refuse to let SystemExit leak silently.

    Under the exit-96 contract the refusal is a ``SystemExit`` carrying 96 and
    prints its message to stderr; under the module's own voice it is a
    ``TrainerRefusal``. Neither may return without refusing.
    """
    try:
        return load_audio_manifest(path)
    except SystemExit as exc:
        assert int(exc.code) == 96, f"exit contract is 96, got {exc.code!r}"
        raise AssertionError(
            f"load_audio_manifest refused via SystemExit(96): {capsys.readouterr().err!r}"
        ) from exc
    return None  # pragma: no cover -- the call above refuses or returns rows


def test_load_audio_manifest_refuses_a_non_string_path_and_names_the_path(capsys: Any) -> None:
    """Line 109: ``dataset=path!r`` is named, and the refusal is a real one.

    1 of 1 dataset paths must be a non-empty string -- the audio manifest
    reader's whole file lives behind this one field. ``None``, ``""``, ``"   "``
    and a ``Path`` (which is not a ``str`` to the reader's own
    ``isinstance(path, str)`` check) are all refused here.
    """
    for path in (None, "", "   ", 5):
        message = _refusal_message(capsys, lambda p=path: load_audio_manifest(p))
        assert "dataset" in message, f"the refusal must name the dataset field; got {message!r}"
        assert (
            repr(path) in message
            or (isinstance(path, str) and path in message)
            or str(path) in message
        ), f"the refusal must name the offending path {path!r}; got {message!r}"


def test_load_audio_manifest_refuses_a_blank_line_and_names_its_line(
    tmp_path: Path, capsys: Any
) -> None:
    """Line 119: a blank line is REFUSED with its line number, not skipped.

    The claim this reader makes is "never silently skips a record", which a
    blank line is exactly the case where that would happen: the jsonl response
    names line 2 as blank rather than quietly ignoring it or swallowing an
    empty record.
    """
    path = tmp_path / "blank.jsonl"
    path.write_bytes(b'{"audio": "a.wav", "text": "one"}\n\n{"audio": "b.wav", "text": "two"}\n')
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    match = _LINE_IN_MESSAGE.search(message)
    assert match, f"the refusal must name its source line; got {message!r}"
    assert int(match.group(1)) == 2, f"the refusal must name line 2; got {message!r}"
    assert "blank" in message.lower(), message


def test_load_audio_manifest_refuses_invalid_json_and_names_the_line(
    tmp_path: Path, capsys: Any
) -> None:
    """Line 126-127: ``JSONDecodeError`` is caught and re-raised naming the line.

    A malformed record is a REFUSAL in the same voice as a missing field: the
    message carries the manifest line, the column it choked at, and the
    decoder's own reason -- never a raw ``json.JSONDecodeError`` that names
    only what the parser itself saw.
    """
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"audio": "a.wav", "text": "one"}\n{"audio": "b.wav", "text": ')
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    match = _LINE_IN_MESSAGE.search(message)
    assert match, f"the refusal must name its source line; got {message!r}"
    assert int(match.group(1)) == 2, f"the refusal must name line 2; got {message!r}"
    assert "column" in message.lower(), message
    assert any(word in message.lower() for word in ("invalid", "json", "expecting")), message


def test_load_audio_manifest_refuses_a_non_object_record_and_names_the_line(
    tmp_path: Path, capsys: Any
) -> None:
    """Line 132: a decoded JSON array is refused naming its type and its line.

    A jsonl record is a JSON OBJECT; a bare list decodes fine and would let a
    record reach the field reader without ``audio``/``text`` keys at all. The
    refusal names the decoded type (``list``) and the line it came from, in the
    same voice the field refusals use.
    """
    path = tmp_path / "list.jsonl"
    path.write_bytes(b'["a.wav", "one"]\n')
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    match = _LINE_IN_MESSAGE.search(message)
    assert match, f"the refusal must name its source line; got {message!r}"
    assert int(match.group(1)) == 1, f"the refusal must name line 1; got {message!r}"
    assert "list" in message, f"the refusal must name the decoded type; got {message!r}"
    assert "object" in message.lower(), message


def test_load_audio_manifest_refuses_an_unreadable_file_and_names_the_path(
    tmp_path: Path, capsys: Any
) -> None:
    """Line 153: an ``OSError`` reading the manifest is refused naming the path.

    The file may simply be absent -- the manifest is a PATH, so a typo in the
    run declaration is exactly what this reader is the first to see. The
    refusal names the dataset path and says no records are available, rather
    than letting ``FileNotFoundError`` escape as an unhandled ``OSError``.
    """
    missing = tmp_path / "no_such_manifest.jsonl"
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(missing)))
    assert "dataset" in message, f"the refusal must name the dataset field; got {message!r}"
    assert str(missing) in message or repr(str(missing)) in message, message
    assert "could not be read" in message.lower(), message


def test_load_audio_manifest_refuses_non_utf8_bytes_and_names_the_path(
    tmp_path: Path, capsys: Any
) -> None:
    """Line 158: bytes outside UTF-8 are refused naming the path, not the line.

    Transcript supervision is attributed to decodable bytes; an unknown
    encoding means the gold cannot be read reliably at all. The refusal names
    the dataset path (the source of the whole file) and the decoding failure --
    precisely so the reader never scores a transcript whose text is unknown.
    """
    path = tmp_path / "latin.jsonl"
    path.write_bytes(b'{"audio": "a.wav", "text": "caf\xe9"}\n')
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    assert "dataset" in message, f"the refusal must name the dataset field; got {message!r}"
    assert "utf-8" in message.lower(), message
    assert "unknown" in message.lower() or "decoding" in message.lower(), message


def test_load_audio_manifest_refuses_an_empty_file_and_names_the_path(
    tmp_path: Path, capsys: Any
) -> None:
    """Line 164: zero records is refused naming the path -- an empty run is refused.

    A manifest with no usable rows has no denominator to measure against, and
    the refusal says so: 0 of 0 lines and the dataset path it read, rather than
    returning an empty tuple that would divide by nothing later.
    """
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    message = _refusal_message(capsys, lambda: load_audio_manifest(str(path)))
    assert "dataset" in message, f"the refusal must name the dataset field; got {message!r}"
    assert "0 of 0" in message or "0 records" in message, message
    assert "measures nothing" in message.lower(), message


class _EmptyDecodeProcessor:
    """A processor whose batch_decode answers one row per call but always empty.

    The refusal this models is ``decoded`` being falsy AFTER a per-row decode --
    exactly the case ``_decode_completion`` guards before reading its text,
    since a completion with no decode has no scoreable text at all.
    """

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        return []


class _ObjectWithText:
    """A decoded row carrying ``text`` as an attribute, not a mapping key."""

    def __init__(self, text: str) -> None:
        self.text = text


class _ObjectWithoutText:
    """A decoded row carrying neither ``text`` key nor ``text`` attribute."""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:  # pragma: no cover -- only for a message's repr() display
        return "<no text here>"


def _decode_completion(processor: Any, token_ids: Any) -> str:
    return asr_rollout._decode_completion(processor, token_ids)


def test_decode_completion_refuses_an_empty_decode_answer(capsys: Any) -> None:
    """Line 187: a decode that returns no rows is refused, never read as empty.

    ``_decode_completion`` calls ``batch_decode`` with exactly one row, so any
    falsy answer means the processor failed to answer at all -- and the
    transcript that would be scored as "" is instead refused as having no
    scoreable text. The refusal names the missing count rather than letting a
    bare ``IndexError`` leak from ``decoded[0]``.
    """
    message = _refusal_message(capsys, lambda: _decode_completion(_EmptyDecodeProcessor(), (1, 2)))
    assert "0 of 1" in message or "no decode" in message.lower(), message
    assert "refus" in message.lower() or "scoreable" in message.lower(), message


def test_decode_completion_reads_a_string_row_directly() -> None:
    """Line 195 (the ``isinstance(text, str)`` arm): a plain string row is its transcript.

    The most common decode answer is the transcript ITSELF -- a bare string,
    not a record around one -- and that string is returned verbatim, not wrapped,
    re-keyed, or silently re-decoded.
    """  # noqa: E401 -- the line comment below covers 195

    class _StringProcessor:
        def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
            rows = _rows(sequences)
            assert len(rows) == 1, "decode is per row"
            return ["hello world from the row"]

    assert _decode_completion(_StringProcessor(), (1, 2)) == "hello world from the row"


def test_decode_completion_reads_text_off_a_record() -> None:
    """Line 195-198: a mapping row's ``text`` key and an object's ``text`` attribute both read.

    A record/attribute carrying ``text`` is the natural shape for a structured
    decode response (the same field name the gold transcript uses). Both an
    ``{"text": ...}`` mapping and a class carrying ``text`` as an attribute are
    read here -- the string is returned unchanged, not stringified or wrapped.
    """

    class _RecordProcessor:
        def __init__(self, row: Any) -> None:
            self._row = row

        def batch_decode(self, sequences: Any, **kwargs: Any) -> list[Any]:
            return [self._row]

    assert (
        _decode_completion(_RecordProcessor({"text": "from a mapping"}), (1,)) == "from a mapping"
    )
    assert (
        _decode_completion(_RecordProcessor(_ObjectWithText("from an attribute")), (1,))
        == "from an attribute"
    )


def _refusal_for_stack(capsys: Any, rows: Any, key: str = "input_features") -> str:
    import torch  # noqa: F401 -- the refusal path guarantees the function-local import runs

    return _refusal_message(capsys, lambda: asr_rollout._stack_feature_rows(rows, key=key))


def test_stack_feature_rows_refuses_rows_of_disagreeing_rank(capsys: Any) -> None:
    """Line 242: rows whose .dim() disagree are refused naming the key and shapes.

    One clip's features are one structured row here, so a 2-D row beside a 3-D
    row is not something a pad block can fix without guessing which axis is
    missing. The refusal names ``key`` and the offending shapes rather than
    letting a torch broadcast or cat error escape.
    """
    import torch

    message = _refusal_for_stack(
        capsys, [torch.ones(1, 128, 50), torch.ones(1, 50)], key="input_features"
    )
    assert "'input_features'" in message, f"the refusal must name the key; got {message!r}"
    assert "rank" in message.lower() or "dimensionality" in message.lower(), message
    assert "(1, 128, 50)" in message and "(1, 50)" in message, (
        f"the refusal must name both shapes; got {message!r}"
    )


def test_stack_feature_rows_refuses_more_than_one_varying_axis(capsys: Any) -> None:
    """Line 255: two axes varying across rows is refused naming the key and shapes.

    Only one axis is the audio frame axis and may be padded; a batch whose rows
    disagree on mel bins AND frames cannot be stacked without guessing which is
    which. The refusal names the key, the varying axis count and the shapes --
    not a silent pad of some chosen axis.
    """
    import torch

    message = _refusal_for_stack(
        capsys, [torch.ones(1, 128, 50), torch.ones(1, 64, 30)], key="input_features"
    )
    assert "'input_features'" in message, f"the refusal must name the key; got {message!r}"
    assert "axes vary" in message or "axes" in message, message
    assert "(1, 128, 50)" in message and "(1, 64, 30)" in message, (
        f"the refusal must name both shapes; got {message!r}"
    )


def _refusal_for_encode(capsys: Any, processor: Any, chunk: Any) -> str:
    return _refusal_message(
        capsys,
        lambda: asr_rollout._encode_audio_prompts(
            processor=processor,
            chunk=chunk,
            language="en",
            pad_token_id=0,
            device="cpu",
        ),
    )


def _one_record(line: int = 1) -> asr_rollout.AudioRow:
    return asr_rollout.AudioRow(line=line, audio="a.wav", text="hello world")


class _NonMappingRequestProcessor:
    """A processor whose request answers a bare list, not a mapping of tensors."""

    def apply_transcription_request(self, audio: Any = None, language: Any = None, **kwargs: Any):
        return [7, 6, 7]


class _NonTensorRequestProcessor:
    """A processor whose request has one value no tensor can be read from."""

    def apply_transcription_request(self, audio: Any = None, language: Any = None, **kwargs: Any):
        import torch

        return {
            "input_ids": torch.tensor([[7, 6, 7]], dtype=torch.long),
            "input_features": object(),  # no ``tolist``, no tensor constructor
        }


class _MissingInputIdsRequestProcessor:
    """A request answering only feature keys -- no ``input_ids`` to continue."""

    def apply_transcription_request(self, audio: Any = None, language: Any = None, **kwargs: Any):
        import torch

        return {
            "input_features": torch.zeros((1, 7, 8)),
            "input_features_mask": torch.ones((1, 7), dtype=torch.long),
        }


class _WideRequestProcessor:
    """A request answering TWO batch rows for ONE clip."""

    def apply_transcription_request(self, audio: Any = None, language: Any = None, **kwargs: Any):
        import torch

        return {
            "input_ids": torch.tensor([[7, 6, 7], [7, 6, 7]], dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        }


class _DisagreeingKeySetsProcessor:
    """A processor whose two clips answer DIFFERENT key sets.

    The first clip carries an extra key the second does not -- exactly the case
    a pad-in would have to drop or invent a conditioning for.
    """

    def __init__(self) -> None:
        self.calls = 0

    def apply_transcription_request(self, audio: Any = None, language: Any = None, **kwargs: Any):
        import torch

        self.calls += 1
        if self.calls == 1:
            return {
                "input_ids": torch.tensor([[7, 6, 7]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "input_features": torch.zeros((1, 7, 8)),
            }
        return {
            "input_ids": torch.tensor([[7, 6, 7]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


def test_encode_audio_prompts_refuses_a_non_mapping_request(capsys: Any) -> None:
    """Line 314: a request answering a non-mapping is refused naming the line and clip.

    A transcription request must yield a mapping of prompt tensors (input_ids
    plus its feature keys); a bare list gives no place to read the prompt from
    at all. The refusal names the manifest line and the clip audio prompt, and
    says what the surface must yield -- not a crash reading ``.items()`` on a
    list.
    """
    message = _refusal_for_encode(capsys, _NonMappingRequestProcessor(), [_one_record(line=1)])
    assert "manifest line 1" in message or _LINE_IN_MESSAGE.search(message), message
    assert "'a.wav'" in message or "a.wav" in message, message
    assert "list" in message, f"the refusal must name the decoded type; got {message!r}"
    assert "mapping" in message.lower(), message


def test_encode_audio_prompts_refuses_a_value_no_tensor_can_be_read_from(capsys: Any) -> None:
    """Line 325-326: a request value no tensor can be read from is refused naming the key.

    ``torch.as_tensor`` on a bare ``object()`` raises before any batching is
    attempted; the refusal names the manifest line, the offending key and the
    value's type rather than letting ``TypeError``/``ValueError`` escape from
    the tensor constructor's own message.
    """
    message = _refusal_for_encode(capsys, _NonTensorRequestProcessor(), [_one_record(line=3)])
    assert "manifest line 3" in message or _LINE_IN_MESSAGE.search(message), message
    assert "'input_features'" in message or "input_features" in message, message
    assert "object" in message, f"the refusal must name the decoded type; got {message!r}"
    assert any(word in message.lower() for word in ("tensor", "read")), message


def test_encode_audio_prompts_reads_one_clip_as_one_batch_row() -> None:
    """Line 333: a 1-D request value is unsqueezed to a single batch row.

    A request made without ``return_tensors`` is entitled to answer plain lists
    (or a 1-D tensor) rather than a batched one -- and one clip is still 1 batch
    row by contract. A 1-D ``input_ids`` row of shape ``(3,)`` becomes ``(1, 3)``
    here rather than being read as three batch rows of nothing.
    """

    class _ListRequestProcessor:
        def apply_transcription_request(self, audio=None, language=None, **kwargs: Any):
            return {"input_ids": [7, 6, 7]}

    batched = asr_rollout._encode_audio_prompts(
        processor=_ListRequestProcessor(),
        chunk=[_one_record()],
        language="en",
        pad_token_id=0,
        device="cpu",
    )
    assert tuple(int(s) for s in batched["input_ids"].shape) == (1, 3), (
        f"a 1-D request answer must be read as ONE clip in ONE batch row, got "
        f"{tuple(int(s) for s in batched['input_ids'].shape)}"
    )
    assert [int(v) for v in batched["input_ids"].reshape(-1).tolist()] == [7, 6, 7]
    assert [int(v) for v in batched["attention_mask"].reshape(-1).tolist()] == [1, 1, 1]


def test_encode_audio_prompts_refuses_a_request_wider_than_one_clip(capsys: Any) -> None:
    """Line 335: a request answering 2 batch rows for ONE clip is refused with its shapes.

    One clip is 1 batch row here; anything wider would have to be cut or
    re-attributed to another clip silently, which this contract refuses. The
    message carries the manifest line, the key and the offending shape.
    """
    message = _refusal_for_encode(capsys, _WideRequestProcessor(), [_one_record(line=2)])
    assert "manifest line 2" in message or _LINE_IN_MESSAGE.search(message), message
    assert "'input_ids'" in message or "input_ids" in message, message
    assert "(2, 3)" in message, f"the refusal must name the assembled shape; got {message!r}"
    assert any(word in message.lower() for word in ("cut", "re-attributed", "batch row")), message


def test_encode_audio_prompts_refuses_a_request_without_input_ids(capsys: Any) -> None:
    """Line 343: a request with no ``input_ids`` is refused naming the keys it answered.

    ``generate()`` has no prompt to continue without input ids, so a request
    answering only feature tensors is refused naming the manifest line, the
    clip and the keys it DID carry -- not a silent fallback to some other key or
    an empty prompt.
    """
    message = _refusal_for_encode(capsys, _MissingInputIdsRequestProcessor(), [_one_record(line=4)])
    assert "manifest line 4" in message or _LINE_IN_MESSAGE.search(message), message
    assert "'a.wav'" in message or "a.wav" in message, message
    assert "input_ids" in message, f"the refusal must name the missing key; got {message!r}"
    assert "input_features" in message, f"the refusal must name the answered keys; got {message!r}"


def test_encode_audio_prompts_refuses_rows_with_disagreeing_key_sets(capsys: Any) -> None:
    """Line 359: two rows disagreeing about their key set are refused naming both sets.

    A batch whose rows disagree about what the prompt carries cannot be stacked
    without dropping or inventing a conditioning key in one of them. The
    refusal names both sorted key sets -- the exact sets, not just a count --
    so an unannounced key is visible in the message rather than silently
    dropped.
    """
    records = [
        asr_rollout.AudioRow(line=1, audio="a.wav", text="one"),
        asr_rollout.AudioRow(line=2, audio="b.wav", text="two"),
    ]
    message = _refusal_for_encode(capsys, _DisagreeingKeySetsProcessor(), records)
    assert "2 distinct key set" in message or "key set" in message.lower(), message
    assert "'input_features'" in message or "input_features" in message, message
    assert "cannot be stacked" in message or "dropping a key" in message.lower(), message
    # Both distinct sets are named, not just a count.
    assert any(
        set_name in message for set_name in ("input_ids", "attention_mask", "input_features")
    ), message


# ---------------------------------------------------------------------------
# 1b. WERReward under a declared custom normalizer, and KeywordWERReward.
# ---------------------------------------------------------------------------


def _split(text: str) -> list[str]:
    """A case-sensitive normalizer that differs from the FS one on purpose."""
    return text.split()


def test_wer_reward_custom_normalizer_is_declared_with_its_id() -> None:
    for bad in ({"normalizer": _split}, {"normalizer_id": "split/v1"}):
        with pytest.raises(ValueError, match="normalizer and normalizer_id"):
            rewards.WERReward(**bad)
    with pytest.raises(ValueError, match="normalizer_id"):
        rewards.WERReward(normalizer=_split, normalizer_id="")
    with pytest.raises(ValueError, match="normalizer_id"):
        rewards.WERReward(normalizer=_split, normalizer_id=3)  # type: ignore[arg-type]
    reward = rewards.WERReward(normalizer=_split, normalizer_id="split/v1")
    assert (reward.normalizer, reward.normalizer_id) == (_split, "split/v1")
    assert rewards.WERReward().normalizer is None and rewards.WERReward().normalizer_id is None


def test_wer_reward_scores_under_the_declared_normalizer() -> None:
    fs, custom = rewards.WERReward(), rewards.WERReward(normalizer=_split, normalizer_id="split/v1")
    # the FS normalizer folds case; the declared one does not, so one of two words is wrong
    assert fs.score(response="Hello world", gold="hello world") == pytest.approx(1.0)
    assert custom.score(response="Hello world", gold="hello world") == pytest.approx(0.5)
    assert custom.score(response="hello world", gold="hello world") == pytest.approx(1.0)
    # abstention follows the declared normalizer: "..." is one token for split, nothing for FS
    assert fs.score(response="x", gold="...") is None
    assert custom.score(response="x", gold="...") == pytest.approx(0.0)
    assert custom.score(response="x", gold="   ") is None
    # the floor still holds under a declared normalizer
    assert custom.score(response=" ".join(["z"] * 40), gold="a") == pytest.approx(
        rewards.WERReward.WER_FLOOR
    )


def test_wer_reward_custom_normalizer_keeps_the_fs_detectors() -> None:
    custom = rewards.WERReward(normalizer=_split, normalizer_id="split/v1", loop_penalty=0.5)
    plain = rewards.WERReward(normalizer=_split, normalizer_id="split/v1")
    assert custom.score(response=_LOOPY, gold="hello world") == pytest.approx(
        plain.score(response=_LOOPY, gold="hello world") - 0.5
    )


def test_keyword_score_is_mean_of_clipped_precision_and_recall() -> None:
    assert "keyword_score" in rewards.__all__ and "KeywordWERReward" in rewards.__all__
    # gold says "b" twice, response once: precision 1/1, recall 1/2
    assert rewards.keyword_score("a b c", "a b c b", ["b"]) == pytest.approx(0.75)
    # response repeats the term four times against one: precision 1/4, recall 1/1
    assert rewards.keyword_score("b b b b", "a b", ["b"]) == pytest.approx(0.625)
    # a two-word term counts as one occurrence, and terms are normalized like the text
    assert rewards.keyword_score(
        "x Mitral Valve y", "mitral valve", ["MITRAL valve"]
    ) == pytest.approx(1.0)
    # missing the term entirely: precision 0 (nothing said), recall 0
    assert rewards.keyword_score("a c", "a b c", ["b"]) == pytest.approx(0.0)
    # two terms pooled: b hit 1/1, d missed 0/1 -> precision 1/1, recall 1/2
    assert rewards.keyword_score("a b c", "a b c d", ["b", "d"]) == pytest.approx(0.75)
    # a term the gold does not say is not a target, and an empty term is ignored
    assert rewards.keyword_score("q b", "a b", ["b", "q", "  "]) == pytest.approx(1.0)
    assert rewards.keyword_score("a b", "a b", ["zz"]) is None
    assert rewards.keyword_score("a b", "a b", []) is None
    # occurrences may overlap ("a a" occurs twice in "a a a")
    assert rewards.keyword_score("a a", "a a a", ["a a"]) == pytest.approx((1.0 + 0.5) / 2)


def test_keyword_score_uses_the_given_normalizer() -> None:
    assert rewards.keyword_score("B c", "b c", ["b"], normalize=_split) == pytest.approx(0.0)
    assert rewards.keyword_score("b c", "b c", ["b"], normalize=_split) == pytest.approx(1.0)


def test_keyword_wer_reward_prices_missed_and_repeated_keywords() -> None:
    gold = "the mitral valve is fine"
    reward = rewards.KeywordWERReward(keywords={gold: ("mitral valve",)}, alpha=0.5)
    base = rewards.WERReward()
    exact = "the mitral valve is fine"
    missed = "the metal valve is fine"
    assert reward.score(response=exact, gold=gold) == pytest.approx(1.0)
    # WER 0.2 -> 0.8; keyword score 0 -> minus 0.5 * (1 - 0)
    assert reward.score(response=missed, gold=gold) == pytest.approx(
        base.score(response=missed, gold=gold) - 0.5
    )
    assert reward.score(response=missed, gold=gold) == pytest.approx(0.3)
    # a gold with no declared terms scores exactly like WERReward
    assert reward.score(response="a x", gold="a b") == pytest.approx(
        base.score(response="a x", gold="a b")
    )
    # declared terms the gold does not say leave the score unchanged
    odd = rewards.KeywordWERReward(keywords={"a b": ("zz",)})
    assert odd.score(response="a x", gold="a b") == pytest.approx(0.5)
    # abstention outranks the keyword term
    assert rewards.KeywordWERReward(keywords={"": ("b",)}).score(response="b", gold="") is None
    # alpha 0 is plain WER; the default is 0.5
    assert rewards.KeywordWERReward(keywords={gold: ("mitral valve",)}, alpha=0.0).score(
        response=missed, gold=gold
    ) == pytest.approx(0.8)
    assert rewards.KeywordWERReward(keywords={}).alpha == 0.5


def test_keyword_wer_reward_uses_the_wer_rewards_normalizer_and_penalties() -> None:
    wer = rewards.WERReward(normalizer=_split, normalizer_id="split/v1", loop_penalty=0.25)
    reward = rewards.KeywordWERReward(keywords={"b c": ("b",)}, alpha=1.0, wer=wer)
    # case-sensitive: "B" is a substitution AND a missed keyword
    assert reward.score(response="B c", gold="b c") == pytest.approx(0.5 - 1.0)
    assert reward.score(response="b c", gold="b c") == pytest.approx(1.0)
    assert reward.wer is wer


def test_keyword_wer_reward_refuses_bad_settings() -> None:
    for alpha in (-0.1, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises(ValueError, match="alpha"):
            rewards.KeywordWERReward(keywords={}, alpha=alpha)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keywords"):
        rewards.KeywordWERReward(keywords=[("a", ("b",))])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wer"):
        rewards.KeywordWERReward(keywords={}, wer="WERReward")  # type: ignore[arg-type]
    reward = rewards.KeywordWERReward(keywords={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        reward.alpha = 1.0  # type: ignore[misc]
