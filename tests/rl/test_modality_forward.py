"""Control for #371: the scored log-prob forward MUST receive `pixel_values`.

Hermetic: no GPU, no checkpoint, no network. `transformers` is replaced in
sys.modules with a fake whose Auto classes hand back a recording VLM and a
tokenizer that fabricates the measured gemma-4-E4B processor surface
(input_ids, attention_mask, mm_token_type_ids, pixel_values,
image_position_ids) with shrunken shapes -- the control is about FORWARDING,
not about reproducing the measured [1, 2520, 768] geometry.

RED MUTATION (exact, one of):
  1. In trainer.py, widen ``_TEXT_KEYS`` to
     ``{"input_ids", "attention_mask", "pixel_values"}``.
     The filter then drops pixel_values from modality_kwargs, every recorded
     forward-call kwargs dict lacks the key, and
     ``assert "pixel_values" in call`` goes red.
  2. Alternatively delete ``**modality_kwargs`` (however spelled) from the
     model(...) call in the log-prob closure. Same failure.

GREEN contract additionally verified: the values are prompt-major expanded
(prompt row repeated group_size times, THEN narrowed to kept rows) -- with a
non-abstaining reward the kept rows are all B*G rows in original order, so
the received tensor must equal base_pixel.repeat_interleave(G, dim=0).
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from types import SimpleNamespace

import pytest

from foundationscale.rl.corpus import Sample

torch = pytest.importorskip("torch", reason="trainer.py itself refuses torch-free hosts")

import foundationscale.rl.trainer as trainer  # noqa: E402 -- must follow importorskip; see module docstring

# --- dimensions (local to the control; deliberately NOT the measured 273/2520) ---
_B, _G, _PROMPT_W, _NEW_TOKENS, _VOCAB = 2, 3, 5, 4, 32

# One tag per prompt row so expansion ORDER is observable.
_BASE_PIXEL = torch.arange(_B * 4 * 3, dtype=torch.float32).reshape(_B, 4, 3) + 100.0


class _FakeBatch(dict):
    """Dict that also answers .to(device) like a transformers BatchEncoding."""

    def to(self, device):
        return _FakeBatch({k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()})


class _FakeTokenizer:
    # The trainer REFUSES a tokenizer with no chat_template rather than
    # concatenating strings silently, so the double must carry one or the
    # control never reaches the forward it exists to measure. A non-empty
    # value is all the guard checks; the template body is not exercised here.
    chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

    """Processor surface measured on gemma-4-E4B, shrunken."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    # The parameter is `text`, matching PreTrainedTokenizerBase.__call__ and
    # ProcessorMixin.__call__. It was `texts` here, which is a double NARROWER
    # than the type it replaces -- the same shape as #252/#372 -- and it went
    # unnoticed only because the caller passed positionally. encode_prompts
    # now calls by keyword on purpose, so the mismatch surfaced immediately.
    def __call__(self, text, *, return_tensors=None, padding=True, add_special_tokens=False):
        ids = torch.arange(1, _B * _PROMPT_W + 1, dtype=torch.long).reshape(_B, _PROMPT_W)
        return _FakeBatch(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.zeros_like(ids),
            image_position_ids=torch.arange(_PROMPT_W).expand(_B, _PROMPT_W).clone(),
            pixel_values=_BASE_PIXEL.clone(),
        )

    def batch_decode(self, rows, skip_special_tokens=True):
        # Row order is preserved: r0..r(B*G-1), prompt-major as generate() emits.
        return [f"r{i}" for i in range(len(rows))]


class _RecordingVLM(torch.nn.Module):
    """Fake VLM. forward() records every kwarg it is called with; generate()
    fabricates B*G prompt-major rows. CPU-only, gradients flow via embed/proj
    so the trainer's backward/optimiser step, if reached, is real."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(_VOCAB, 8)
        self.proj = torch.nn.Linear(8, _VOCAB)
        self.config = SimpleNamespace(use_cache=False)
        self.calls: list[dict] = []

    def forward(self, *, input_ids, attention_mask=None, **kwargs):
        self.calls.append(
            {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        )
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))

    def generate(self, *, input_ids, max_new_tokens, num_return_sequences, **kwargs):
        B, W = input_ids.shape
        G = num_return_sequences
        prompts = input_ids.repeat_interleave(G, dim=0)
        tail = (
            torch.arange(1, B * G * max_new_tokens + 1, dtype=torch.long).reshape(
                B * G, max_new_tokens
            )
            % (_VOCAB - 1)
        ) + 1  # never pad_token_id(0): attention mask stays all-ones
        return torch.cat([prompts, tail], dim=1)


class _FakeReward:
    """Never abstains (every row is kept), and varies WITHIN each group:
    rows p0g0..p0gG-1 score 0.0, 0.25, 0.5 -- a degenerate identical-reward
    group could trip an unrelated refusal and mask this control."""

    def score(self, *, response: str, gold):
        return float(int(response[1:]) % _G) * 0.25


def _build_config():
    """Fill RLTrainConfig from a name->value table using introspection. Any
    required field with no table entry is a LOUD failure naming what is
    needed -- never a guess."""
    table = {
        "max_steps": 1,
        "max_new_tokens": _NEW_TOKENS,
        "group_size": _G,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "device": "cpu",
        # The REAL field names, confirmed against dataclasses.fields():
        # `model` and `dataset`. The generator guessed model_name/model_id/
        # corpus_path; its own required-field guard refused rather than
        # silently constructing a config from a wrong key, which is why the
        # miss surfaced here instead of as a confusing downstream error.
        "model": "fake/vlm",
        "dataset": "unused",
        "model_name": "fake/vlm",
        "model_id": "fake/vlm",
        "tokenizer_name": "fake/vlm",
        "batch_size": _B,
        "prompts_per_step": _B,
        "chunk_size": _B,
        "learning_rate": 1e-3,
        "lr": 1e-3,
        "corpus_path": "unused",
    }
    kwargs = {}
    for field in dataclasses.fields(trainer.RLTrainConfig):
        if field.name in table:
            kwargs[field.name] = table[field.name]
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            pytest.fail(
                f"UNMEASURED: RLTrainConfig field {field.name!r} has no default and no "
                "table entry; extend the table in this control with its real meaning."
            )
    return trainer.RLTrainConfig(**kwargs)


def _execute(config):
    """Locate the run() surface without assuming its name."""
    if hasattr(config, "run"):
        return config.run()
    runners = [
        v
        for v in vars(trainer).values()
        if inspect.isclass(v)
        and v.__module__ == trainer.__name__
        and callable(getattr(v, "run", None))
    ]
    if not runners:
        pytest.fail("UNMEASURED: trainer.py exposes no run() surface this control can drive.")
    return runners[0](config).run()


def test_logprob_forward_receives_pixel_values(monkeypatch):
    vlm = _RecordingVLM()
    tokenizer = _FakeTokenizer()

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: vlm),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **k: vlm),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
    )
    # trainer.py imports torch/transformers INSIDE run(), so a sys.modules swap
    # is enough -- no checkpoint is ever touched.
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        trainer,
        "load_sharegpt",
        # The REAL Sample dataclass, not a SimpleNamespace. A stub is NARROWER
        # than the type it replaces -- this one lacked `.images`/`.video` and
        # the #371 modality guard died on AttributeError instead of running.
        # That is the #252/#372 shape: a double that cannot exercise the code
        # it stands in for turns a real control into a fixture bug. Using the
        # real type means the guard sees exactly what production gives it.
        lambda *a, **k: [
            Sample(
                sample_id=f"s{index}",
                prompt_turns=(("user", "q"),),
                response="A",
                gold="A",
            )
            for index in range(64)
        ],
    )
    monkeypatch.setattr(trainer, "MCQLetterReward", lambda *a, **k: _FakeReward())

    _execute(_build_config())

    assert vlm.calls, "model.__call__ was never invoked; step produced no forward pass"

    # THE control: EVERY forward -- the no_grad old-logprob pass AND the
    # graph-carrying current pass -- must receive pixel_values. The trainer
    # builds modality_kwargs once precisely so these two cannot drift.
    for i, call in enumerate(vlm.calls):
        assert "pixel_values" in call, (
            f"forward call {i} received no pixel_values; keys seen: {sorted(call)}. "
            "The image conditional is being scored image-free (importance ratio "
            "compares two different distributions) and nothing downstream can see it."
        )

    received = vlm.calls[0]["pixel_values"]
    assert received.shape[0] == _B * _G, (
        f"expected {_B * _G} prompt-major repeated rows, got {received.shape[0]}"
    )
    # repeat-to-group THEN narrow-to-kept, in that order. With zero abstentions
    # the kept rows are 0..B*G-1, so this is exact.
    expected = _BASE_PIXEL.repeat_interleave(_G, dim=0)
    assert torch.equal(received, expected), (
        "pixel_values rows do not match prompt-major group expansion; the image "
        "conditioning is attached to the wrong completions."
    )
