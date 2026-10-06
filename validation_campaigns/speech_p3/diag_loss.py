"""Why is the first-step loss ~15 for a model that already transcribes at ~4% WER?

Takes one real batch from FoundationScale's audio collator and measures the loss three ways
on the BASE model, then prints the most expensive supervised tokens.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from foundationscale.rl.prompt_surface import PromptSurface
from foundationscale.train.audio import train_audio_collator_or_refuse

model_dir, manifest = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()][:4]
proc = AutoProcessor.from_pretrained(model_dir)
surface = PromptSurface(kind="processor", surface=proc, reason="diag", supports_images=False)
collate = train_audio_collator_or_refuse(surface, audio_column="audio", max_length=2048)
batch = collate(rows)
model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).cuda()
dev = {
    k: (v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda())
    for k, v in batch.items()
}
labels = dev["labels"]
print("keys", sorted(dev), "label tokens per row", (labels != -100).sum(1).tolist())
for mode in ("eval", "train"):
    getattr(model, mode)()
    with torch.no_grad():
        out = model(**dev)
    logits = out.logits.float()
    manual = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    print(f"{mode}: model loss {out.loss.item():.3f}  manual shifted CE {manual.item():.3f}")
model.eval()
with torch.no_grad():
    logits = model(**dev).logits.float()
tok = proc.tokenizer
row = 0
pos = (labels[row, 1:] != -100).nonzero().flatten()
nll = F.cross_entropy(logits[row, :-1][pos], labels[row, 1:][pos], reduction="none")
print("row0 supervised text:", repr(tok.decode(labels[row][labels[row] != -100])))
print("row0 argmax at those positions:", repr(tok.decode(logits[row, :-1][pos].argmax(-1))))
worst = nll.argsort(descending=True)[:8]
for i in worst.tolist():
    p = pos[i].item()
    print(
        f"  pos {p}: target {tok.decode([labels[row, p + 1].item()])!r} nll {nll[i].item():.2f}"
        f" argmax {tok.decode([logits[row, p].argmax().item()])!r}"
    )
print("row0 mean nll", nll.mean().item())

# Does the Trainer's extra kwarg change the loss? (the only input the Trainer adds)
n_items = (labels[:, 1:] != -100).sum()
for mode in ("eval", "train"):
    getattr(model, mode)()
    with torch.no_grad():
        with_kw = model(**dev, num_items_in_batch=n_items).loss.item()
        without = model(**dev).loss.item()
    print(
        f"{mode}: loss without num_items {without:.3f}"
        f"  with num_items={n_items.item()} {with_kw:.3f}"
    )

# The Trainer wraps forward in bf16 autocast when bf16=True. Same batch, autocast on vs off.
model.train()
with torch.no_grad():
    plain = model(**dev).loss.item()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        auto = model(**dev).loss.item()
print(f"AUTOCAST: loss without autocast {plain:.3f}  with bf16 autocast {auto:.3f}")

# Direct call, float32 input_features exactly as the collator emits them.
model.train()
dev32 = {k: (v.cuda() if v.is_floating_point() else v.cuda()) for k, v in batch.items()}
with torch.no_grad():
    l32 = model(**dev32).loss.item()
    lbf = model(**dev).loss.item()
print(f"DTYPE: input_features float32 -> loss {l32:.3f}   bf16 -> loss {lbf:.3f}")

# Grad enabled vs no_grad, same batch.
model.train()
with torch.no_grad():
    l_ng = model(**dev32).loss.item()
l_g = model(**dev32).loss.item()
model.eval()
l_ge = model(**dev32).loss.item()
print(f"GRAD: no_grad {l_ng:.3f}   grad-enabled train {l_g:.3f}   grad-enabled eval {l_ge:.3f}")

# use_cache is the only thing Trainer() changes. Pin it both ways on the same batch.
model.train()
with torch.no_grad():
    for uc in (None, True, False):
        model.config.use_cache = uc
        print(f"USECACHE config.use_cache={uc}: loss {model(**dev32).loss.item():.3f}")
    model.config.use_cache = False
    print(
        "USECACHE config False + forward(use_cache=True): loss "
        f"{model(**dev32, use_cache=True).loss.item():.3f}"
    )
tc = model.config.get_text_config()
print(
    "USECACHE text config:",
    {k: getattr(tc, k, None) for k in ("num_kv_shared_layers", "num_hidden_layers", "use_cache")},
)
