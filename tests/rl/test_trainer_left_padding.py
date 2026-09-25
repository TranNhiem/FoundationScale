# SPDX-License-Identifier: Apache-2.0
"""#546: the RL trainer must LEFT-pad batched prompts before ``generate``.

WHAT IS CLAIMED: ``RLTrainer.run()`` sets ``padding_side == "left"`` on the
tokenizer -- and on the processor's tokenizer for the processor surface --
before any prompt batch is encoded for ``model.generate``. The claim is
measured AT generate time through a fake tokenizer/model seam: the fake model
records the padding side in effect when ``generate`` is invoked, and the fake
tokenizer pads whichever side is in effect when it is called, so the encoded
rows themselves show where the pads landed. The fakes start on "right" (the
transformers default), so a trainer that set nothing would be caught, not
assumed away -- that is the control.

WHAT IS NOT CLAIMED: anything about real transformers models, real tokenizers,
loss values, or convergence. The seam stubs the model, the surface and the
corpus and drives the real trainer code around them; nothing here says the
recipes train well.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.trainer as trainer_module  # noqa: E402 (torch importorskip first)
from foundationscale.rl.trainer import (  # noqa: E402 (torch importorskip first)
    RLTrainConfig,
    RLTrainer,
)

_PAD = 0
_TOK_A = 61  # decodes to "A" in the fake -- a row the reward scores 1.0
_TOK_B = 62  # decodes to "B" -- scored 0.0, so each group has variance
_VOCAB = 96


class _FakeTokenizer:
    """Tokenizer stand-in that HONOURS padding_side.

    It starts on "right" -- the transformers default the defect came from --
    so that "the batch was left-padded at generate time" is a measured
    property of the run rather than a restated configuration.
    """

    def __init__(self) -> None:
        self.chat_template = "fake-template"
        self.pad_token_id = _PAD
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.padding_side_at_encode: list[str] = []
        self.last_input_ids = None

    def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=True):
        return [
            ";".join(f"{turn['role']}:{turn['content']}" for turn in conversation)
            for conversation in conversations
        ]

    def __call__(
        self, *, text, return_tensors=None, padding=None, add_special_tokens=None, images=None
    ):
        self.padding_side_at_encode.append(self.padding_side)
        rows = [[(ord(ch) % 32) + 8 for ch in line] for line in text]
        width = max(len(row) for row in rows)
        padded = []
        for row in rows:
            pads = [_PAD] * (width - len(row))
            padded.append(pads + row if self.padding_side == "left" else row + pads)
        input_ids = torch.tensor(padded, dtype=torch.long)
        self.last_input_ids = input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != _PAD).long(),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            "".join("A" if token == _TOK_A else "B" for token in row) for row in sequences.tolist()
        ]


class _FakeProcessor:
    """Processor stand-in whose whole surface is its inner tokenizer.

    This is how the trainer reads the processor surface: ``tokenizer`` binds
    to ``surface.tokenizer``, so the left-padding invariant has to reach the
    INNER tokenizer, not just the processor object.
    """

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            conversations, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def __call__(self, *, text, **kwargs):
        return self.tokenizer(text=text, **kwargs)


class _FakeModel(torch.nn.Module):
    """One embedding table whose lookups double as logits.

    Deterministic, so the no-grad old pass and the graph-carrying current
    pass read identically until the optimizer moves something -- which is
    what lets the same seam serve the step-1 metrics invariant in
    test_trainer_step_metrics.py.
    """

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(_VOCAB, _VOCAB)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(_VOCAB) * 2.0)
        self._tokenizer = tokenizer
        self.padding_side_at_generate: list[str] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return types.SimpleNamespace(logits=self.embedding(input_ids))

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        max_new_tokens,
        num_return_sequences,
        do_sample,
        temperature,
        top_p,
        top_k,
        pad_token_id,
        **kwargs,
    ):
        self.padding_side_at_generate.append(self._tokenizer.padding_side)
        expanded = input_ids.repeat_interleave(num_return_sequences, dim=0)
        # Rows alternate continuation letters, so within every group rewards
        # vary (gold is "A") and the step is measurable rather than saturated.
        continuation = torch.tensor(
            [
                [_TOK_A if row % 2 == 0 else _TOK_B] * max_new_tokens
                for row in range(expanded.shape[0])
            ],
            dtype=torch.long,
        )
        return torch.cat([expanded, continuation], dim=1)


def _run_seam(monkeypatch, *, surface_kind: str):
    """Drive the REAL trainer over fakes and return (tokenizer, model, reports)."""
    tokenizer = _FakeTokenizer()
    model = _FakeModel(tokenizer)
    samples = (
        types.SimpleNamespace(
            sample_id="s0",
            prompt_turns=(("user", "Pick A."),),  # the shorter prompt
            gold="A",
            images=(),
            video=None,
        ),
        types.SimpleNamespace(
            sample_id="s1",
            prompt_turns=(("user", "Now pick the letter A, please."),),
            gold="A",
            images=(),
            video=None,
        ),
    )
    monkeypatch.setattr(
        trainer_module, "load_sharegpt", lambda dataset, gold_key=None: list(samples)
    )
    if surface_kind == "processor":
        surface = types.SimpleNamespace(
            kind="processor",
            surface=_FakeProcessor(tokenizer),
            reason="fake processor",
            supports_images=True,
        )
    else:
        surface = types.SimpleNamespace(
            kind="tokenizer",
            surface=tokenizer,
            reason="fake tokenizer",
            supports_images=False,
        )
    monkeypatch.setattr(
        trainer_module, "resolve_prompt_surface", lambda model_id, needs_images: surface
    )
    loader = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = loader
    fake_transformers.AutoModelForImageTextToText = loader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = RLTrainConfig(
        model="fake/model",
        dataset="unused.jsonl",
        group_size=2,
        prompts_per_step=2,
        max_steps=1,
        max_new_tokens=2,
        device="cpu",
    )
    reports = RLTrainer(config).run()
    assert len(reports) == 1, "the seam is built so every group varies; a step must be measured"
    return tokenizer, model, reports


def test_the_fake_defaults_to_right_padding_so_the_claim_is_not_vacuous() -> None:
    # Control: if the trainer set nothing, generate() would observe "right".
    assert _FakeTokenizer().padding_side == "right"


def test_generate_runs_with_left_padding_on_the_tokenizer_surface(monkeypatch) -> None:
    tokenizer, model, reports = _run_seam(monkeypatch, surface_kind="tokenizer")
    assert tokenizer.padding_side == "left"
    assert model.padding_side_at_generate == ["left"], (
        "the trainer encoded prompts for generate() while padding was still "
        f"{model.padding_side_at_generate!r}"
    )
    assert tokenizer.padding_side_at_encode
    assert all(side == "left" for side in tokenizer.padding_side_at_encode)
    # The encoded batch itself shows the property that matters: pads form a
    # PREFIX on the shorter row, never a wedge between prompt and continuation.
    ids = tokenizer.last_input_ids.tolist()
    assert ids[0][0] == _PAD, "shorter prompt is not left-padded"
    assert ids[0][-1] != _PAD, "shorter prompt's real tokens must reach the slice boundary"
    assert ids[1][0] != _PAD, "longest row must carry no padding at all"
    assert reports[0].rows == 4


def test_generate_runs_with_left_padding_on_the_processor_surface(monkeypatch) -> None:
    tokenizer, model, reports = _run_seam(monkeypatch, surface_kind="processor")
    # The trainer binds tokenizer := processor.tokenizer; the invariant must
    # hold on that inner object, which is what this asserts.
    assert tokenizer.padding_side == "left"
    assert model.padding_side_at_generate == ["left"]
    assert reports[0].rows == 4
