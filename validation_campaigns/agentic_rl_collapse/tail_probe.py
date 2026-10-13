"""Mechanism probe for the agentic music collapse: does the policy's tail inflate over training?

    python3 tail_probe.py <base_model> <tasks.jsonl> <out.json> <label=ckpt_dir> [<label=ckpt_dir> ...]

Hypothesis: with top_p 0.95 sampling and raw-logit scoring, negative-advantage updates raise every
unsampled token's probability (softmax gradient e_a - p), and tail tokens are never sampled to be
corrected, so the distribution's tail inflates until outputs turn malformed. Signature: per-token
entropy and the 0.95-nucleus size grow across a raw-scoring run's checkpoints ahead of its collapse, and
stay flat for consistent scoring.

Responses are FIXED across checkpoints (sampled once from the base model), so every checkpoint is
measured on identical inputs; only the distribution over the next token changes.
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base, tasks_path, out_path = sys.argv[1:4]
ckpts = [a.split("=", 1) for a in sys.argv[4:]]
tok = AutoTokenizer.from_pretrained(base)
tasks = [json.loads(line) for line in open(tasks_path) if line.strip()][:4]
torch.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
seqs = []
for t in tasks:
    prompt = tok.apply_chat_template(t["messages"], add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        gen = model.generate(ids, do_sample=True, top_p=0.95, temperature=1.0, max_new_tokens=768)
    seqs.append((ids.shape[1], gen))
del model
torch.cuda.empty_cache()


def stats(model):
    ent, nuc, top1 = [], [], []
    with torch.no_grad():
        for plen, gen in seqs:
            logits = model(gen).logits[0, plen - 1 : -1].float()
            p = logits.softmax(-1)
            ent.append(-(p * p.clamp_min(1e-30).log()).sum(-1))
            sp = p.sort(-1, descending=True).values
            nuc.append(((sp.cumsum(-1) - sp) < 0.95).sum(-1).float())
            top1.append(sp[:, 0])
    cat = lambda xs: torch.cat(xs)  # noqa: E731
    e, n, t = cat(ent), cat(nuc), cat(top1)
    return {
        "tokens": int(e.numel()),
        "entropy_mean": round(float(e.mean()), 4),
        "nucleus95_size_mean": round(float(n.mean()), 2),
        "nucleus95_size_p90": round(float(n.quantile(0.9)), 1),
        "top1_prob_mean": round(float(t.mean()), 4),
    }


rep = {}
for label, path in [("base", base), *ckpts]:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    rep[label] = stats(m)
    print(label, rep[label], flush=True)
    del m
    torch.cuda.empty_cache()
json.dump(rep, open(out_path, "w"), indent=1)
