"""logprob_chunk_tokens: chunked LM-head scoring equals the full-logits path (log-probs and gradients)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.trainer import TrainerRefusal, _chunked_token_logprobs, _token_logprobs  # noqa: E402


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=37, dim=16, softcap=None):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.body = torch.nn.Linear(dim, dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.config = type("C", (), {"final_logit_softcapping": softcap})()

    class _Dec(torch.nn.Module):
        def __init__(self, outer):
            super().__init__()
            self.outer = [outer]

        def forward(self, input_ids, attention_mask=None):
            o = self.outer[0]
            return type("O", (), {"last_hidden_state": torch.tanh(o.body(o.embed(input_ids)))})()

    def get_decoder(self):
        return TinyLM._Dec(self)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": self.lm_head(self.get_decoder()(input_ids).last_hidden_state)})()


@pytest.mark.parametrize("chunk", [1, 3, 7, 64])
@pytest.mark.parametrize("sampling", [(1.0, 1.0, 0), (0.7, 0.9, 5)])
def test_chunked_equals_full(chunk, sampling):
    torch.manual_seed(0)
    m = TinyLM()
    ids = torch.randint(0, 37, (2, 12))
    tgt = ids[:, 1:]
    temperature, top_p, top_k = sampling
    full = _token_logprobs(m(ids).logits, tgt, temperature=temperature, top_p=top_p, top_k=top_k)
    full.sum().backward()
    g_full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    part = _chunked_token_logprobs(m, ids, None, tgt, chunk=chunk, temperature=temperature, top_p=top_p, top_k=top_k)
    part.sum().backward()
    assert part.shape == full.shape and torch.allclose(part, full, atol=1e-5)
    for a, b in zip(g_full, [p.grad for p in m.parameters()]):
        assert torch.allclose(a, b, atol=1e-5)


def test_softcapping_and_non_linear_heads_are_refused():
    ids = torch.randint(0, 37, (1, 6))
    with pytest.raises(TrainerRefusal, match="softcapping"):
        _chunked_token_logprobs(TinyLM(softcap=30.0), ids, None, ids[:, 1:], chunk=2,
                                temperature=1.0, top_p=1.0, top_k=0)
    m = TinyLM()
    m.get_output_embeddings = lambda: torch.nn.Identity()
    with pytest.raises(TrainerRefusal, match="nn.Linear"):
        _chunked_token_logprobs(m, ids, None, ids[:, 1:], chunk=2, temperature=1.0, top_p=1.0, top_k=0)
