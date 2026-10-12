"""Cost of sampled-distribution scoring on the real policy (GB200)."""

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM

from foundationscale.rl.trainer import _token_logprobs

model_path, L = sys.argv[1], int(sys.argv[2])
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
model.train()
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, 150000, (1, L + 1), generator=g).cuda()
rep = {"L": L}
for name, kw in [
    ("raw", {}),
    ("top_p0.95", {"top_p": 0.95}),
    ("T0.8_top_p0.95_top_k20", {"temperature": 0.8, "top_p": 0.95, "top_k": 20}),
]:
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    logits = model(input_ids=ids[:, :-1], use_cache=False).logits
    rep.setdefault("after_forward_GB", round(torch.cuda.max_memory_allocated() / 2**30, 1))
    rep.setdefault("logits_dtype", str(logits.dtype))
    torch.cuda.synchronize()
    t0 = time.time()
    lp = _token_logprobs(logits, ids[:, 1:], **kw)
    torch.cuda.synchronize()
    t_score = time.time() - t0
    lp.sum().backward()
    torch.cuda.synchronize()
    t_total = time.time() - t0
    rep[name] = {
        "score_s": round(t_score, 3),
        "score+backward_s": round(t_total, 2),
        "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "mean_logprob": round(float(lp.mean()), 4),
        "finite": bool(torch.isfinite(lp).all()),
    }
    del logits, lp
print(json.dumps(rep, indent=1))
