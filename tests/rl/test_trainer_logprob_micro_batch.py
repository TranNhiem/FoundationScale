# SPDX-License-Identifier: Apache-2.0
"""Micro-batched log-prob scoring preserves the surrogate gradient.

WHAT IS CLAIMED: ``_token_logprobs`` agrees with the explicit
``log_softmax`` plus ``gather`` reading; for row budgets 1 and 3 over five
kept rows, the trainer's ``_micro_batched_backward`` helper accumulates the
same parameter gradients as one whole-batch graph; and a negative row
budget is rejected as invalid configuration.

WHAT IS NOT CLAIMED: anything about real model loading, convergence, end-
to-end reward accounting, or a specific memory saving. The helper-level
model below is deterministic precisely so the gradient equality being
tested is arithmetic rather than sampling luck.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.trainer import (  # noqa: E402 (torch importorskip first)
    RLTrainConfig,
    RLTrainer,
    _micro_batched_backward,
    _token_logprobs,
)


class _TinyPolicy(torch.nn.Module):
    """A deterministic embedding plus linear head exposing ``.logits``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(4907)
        self.embedding = torch.nn.Embedding(32, 11)
        self.head = torch.nn.Linear(11, 32)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        assert input_ids is not None
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return types.SimpleNamespace(logits=self.head(hidden))


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.tensor(
        [
            [4, 11, 14, 9, 23, 7],
            [6, 8, 13, 27, 5, 12],
            [10, 20, 3, 17, 19, 21],
            [2, 25, 15, 16, 18, 22],
            [24, 26, 28, 29, 1, 30],
        ],
        dtype=torch.long,
    )
    attention = torch.ones_like(token_ids)
    attention[1, -2:] = 0
    attention[3, -1] = 0
    target_ids = token_ids[:, 1:]
    supervision = attention[:, 1:].to(dtype=torch.float32)
    return token_ids, attention, target_ids, supervision


def _forward_slice(
    model: _TinyPolicy,
    token_ids: torch.Tensor,
    attention: torch.Tensor,
    target_ids: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    width = end - start
    output = model(
        input_ids=token_ids.narrow(0, start, width),
        attention_mask=attention.narrow(0, start, width),
    )
    return _token_logprobs(output.logits, target_ids.narrow(0, start, width))


def _row_slices(n_rows: int, micro_batch: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + micro_batch, n_rows)) for start in range(0, n_rows, micro_batch)
    )


def _surrogate(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    row_weights = torch.linspace(
        0.75, 1.35, steps=logprobs.shape[0], dtype=torch.float32
    ).unsqueeze(-1)
    position_offsets = torch.linspace(-0.4, 0.7, steps=logprobs.shape[1], dtype=torch.float32)
    centred = row_weights * (logprobs - position_offsets)
    return (centred.square() * mask).sum() / mask.sum()


def _parameter_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        grads[name] = parameter.grad.detach().clone()
    return grads


def test_token_logprobs_matches_log_softmax_then_gather() -> None:
    torch.manual_seed(101)
    logits = torch.randn(3, 7, 19, dtype=torch.float32)
    target_ids = torch.randint(0, 19, (3, 7), dtype=torch.long)

    expected = torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    actual = _token_logprobs(logits, target_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("micro_batch", (1, 3))
def test_micro_batched_two_pass_gradient_equals_whole_batch(
    micro_batch: int,
) -> None:
    token_ids, attention, target_ids, supervision = _fixed_batch()
    n_rows = token_ids.shape[0]
    assert n_rows == 5

    whole_model = _TinyPolicy()
    whole_current = _forward_slice(whole_model, token_ids, attention, target_ids, 0, n_rows)
    whole_loss = _surrogate(whole_current, supervision)
    whole_loss.backward()
    whole_grads = _parameter_grads(whole_model)

    micro_model = _TinyPolicy()
    row_slices = _row_slices(n_rows, micro_batch)
    with torch.no_grad():
        micro_current = (
            torch.cat(
                [
                    _forward_slice(
                        micro_model,
                        token_ids,
                        attention,
                        target_ids,
                        start,
                        end,
                    )
                    for start, end in row_slices
                ],
                dim=0,
            )
            .detach()
            .requires_grad_(True)
        )
    micro_loss = _surrogate(micro_current, supervision)

    def forward_slice(start: int, end: int) -> torch.Tensor:
        return _forward_slice(micro_model, token_ids, attention, target_ids, start, end)

    _micro_batched_backward(
        loss_tensor=micro_loss,
        current_logprobs=micro_current,
        row_slices=row_slices,
        forward_slice=forward_slice,
    )
    micro_grads = _parameter_grads(micro_model)

    torch.testing.assert_close(micro_loss.detach(), whole_loss.detach(), rtol=2e-6, atol=2e-7)
    assert micro_current.grad is not None
    assert micro_grads.keys() == whole_grads.keys()
    for name, whole_grad in whole_grads.items():
        torch.testing.assert_close(micro_grads[name], whole_grad, rtol=2e-5, atol=2e-6)


def test_negative_logprob_micro_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="logprob_micro_batch"):
        RLTrainer(
            RLTrainConfig(
                model="unused/model",
                dataset="unused.jsonl",
                logprob_micro_batch=-1,
            )
        )
