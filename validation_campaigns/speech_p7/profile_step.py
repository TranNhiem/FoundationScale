"""Where does a Gemma-4 audio training step spend its time? Collate (CPU) vs compute (GPU)."""

import json
import sys
import time

import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from foundationscale.rl.prompt_surface import PromptSurface
from foundationscale.train.audio import train_audio_collator_or_refuse

model_dir, manifest = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(manifest)][:128]
proc = AutoProcessor.from_pretrained(model_dir)
surface = PromptSurface(kind="processor", surface=proc, reason="profile", supports_images=False)
collate = train_audio_collator_or_refuse(surface, audio_column="audio", max_length=1024)

# (a) collate: break it down by stage on one batch, then time whole batches.
import soundfile as sf

batch_rows = rows[:8]
t = time.perf_counter()
for r in batch_rows:
    sf.read(r["audio"], dtype="float32")
t_decode = time.perf_counter() - t
waves = [sf.read(r["audio"], dtype="float32")[0] for r in batch_rows]
msgs = [[{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": r["text"]}]},
         {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}] for w, r in zip(waves, batch_rows)]
t = time.perf_counter()
proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
t_template = time.perf_counter() - t
times = []
for i in range(0, 64, 8):
    t = time.perf_counter()
    batch = collate(rows[i : i + 8])
    times.append(time.perf_counter() - t)
print(f"PROF collate_batch8 mean={sum(times)/len(times):.3f}s (decode 8 files {t_decode:.3f}s, one chat-template pass {t_template:.3f}s)")

# (b) GPU step on a ready batch: forward+backward+AdamW, bf16 full fine-tune.
model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).cuda()
model.config.use_cache = True  # KV-sharing forward needs it (see loop._kv_shared_layer_count)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
dev = {k: (v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda()) for k, v in batch.items()}
for step in range(6):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss = model(**dev).loss
    torch.cuda.synchronize(); t1 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    if step >= 2:
        print(f"PROF gpu step{step}: fwd {t1-t0:.3f}s bwd {t2-t1:.3f}s opt {t3-t2:.3f}s total {t3-t0:.3f}s "
              f"seq {dev['input_ids'].shape[1]} feats {tuple(dev['input_features'].shape)}")

# (c) what the Trainer adds: global grad-norm clipping (default max_grad_norm=1.0) and bf16 autocast.
for step in range(4):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**dev).loss
    loss.backward()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t3 = time.perf_counter()
    if step >= 1:
        print(f"PROF trainerlike step{step}: fwd+bwd(autocast) {t1-t0:.3f}s clip {t2-t1:.3f}s opt {t3-t2:.3f}s total {t3-t0:.3f}s")

# (d) average over 16 DIFFERENT real batches: lengths vary, so one batch is not representative.
tot, seqs = [], []
for i in range(0, 128, 8):
    b = collate(rows[i % len(rows) : i % len(rows) + 8]) if i + 8 <= len(rows) else collate(rows[:8])
    d = {k: (v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda()) for k, v in b.items()}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**d).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); tot.append(time.perf_counter() - t0); seqs.append(d["input_ids"].shape[1])
print(f"PROF avg16 step {sum(tot)/len(tot):.3f}s (min {min(tot):.3f} max {max(tot):.3f}) seq len avg {sum(seqs)/len(seqs):.0f} range {min(seqs)}-{max(seqs)}")
