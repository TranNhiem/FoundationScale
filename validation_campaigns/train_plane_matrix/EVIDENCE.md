# Training plane — method × precision evidence matrix (A3, #331)

Only things actually executed. Each cell states what was run and what came back.
Inferences are marked INFERRED and kept separate from measurements. A control
that turned out to be unable to fail is named as such rather than counted as a
pass.

Produced by `run_matrix.sh` in this directory, in one invocation against one
working tree, on one GB200 GPU (`<gb200-node>`).

    tree   loop.py            247df092180d86c8b15fe052e782d4b1
           checkpoint_gates.py 4337df355a02346d516a248dd2d5f518
           = commit c6b6c5ea4aea7eea5aa4ea290e71c788d6276e1a
    env    peft 0.19.1 · transformers 5.13.0 · torch 2.12.1+cu130
    run    Qwen/Qwen2.5-1.5B · fancyzhx/ag_news · 20 steps · batch 1 · 1 GPU
           save-interval 10, so each green cell writes 3 checkpoints

All six cells come from the SAME tree. This matters more than it sounds: the
readings these cells replaced were taken across three different trees, and cells
measured on different trees cannot be compared to each other, which is the one
property a matrix exists to provide.

---

## M1 — the matrix

| method | precision | rc | wall | artifact bytes | saved tensor dtypes | post-load precision reading |
|---|---|---|---|---|---|---|
| full | bf16  | 0  | 27s | 21,647,069,015 | `{'BF16': 1014}` | verified `torch.bfloat16` |
| full | fp32  | 0  | 36s | 43,259,074,294 | `{'F32': 1014}`  | verified `torch.float32` |
| full | nvfp4 | 96 | 10s | 10,401         | no artifact      | absent — refused before load |
| lora | bf16  | 0  | 21s | 65,114,191     | `{'F32': 336}`   | verified `torch.bfloat16` |
| lora | fp32  | 0  | 21s | 65,114,197     | `{'F32': 336}`   | verified `torch.float32` |
| lora | nvfp4 | 96 | 11s | 10,437         | no artifact      | absent — refused before load |

Save-completeness denominators, quoted from the run logs: full-finetune
`338/338 tensors`, LoRA `112/112 tensors`, both "all declared tensors present
(excluding 0 `_extra_state` metadata blobs)".

`bytes` is the whole output directory, which includes JSON sidecars carrying the
run's timestamp and output path. Small deltas between otherwise identical runs
(the 6 bytes between the two LoRA cells) live in those sidecars, not in weights;
the tensor payload is pinned by the (count, dtype) reading, not by the byte total.

---

## M2 — the METHOD control DISCRIMINATES

    full-finetune   21.6 GB (bf16) / 43.3 GB (fp32)
    LoRA            65.1 MB
    ratio           ~332x

A LoRA cell that began writing full-model weights, or a full-finetune cell that
wrote only adapter-sized output, would be a mislabelled run at any exit code, and
this reading catches it without reference to any log line. The tensor counts
agree independently: 1014 saved tensors under full-finetune against 336 under
LoRA, and declared denominators of 338 against 112.

## M3 — the PRECISION control DISCRIMINATES on full-finetune

    fp32 / bf16 bytes      43,259,074,294 / 21,647,069,015 = 1.9983
    saved dtypes           F32 x1014  vs  BF16 x1014

Two independent readings, both moving the right way. The dtype reading is taken
from the safetensors headers — the bytes a downstream consumer actually loads —
rather than from a log line claiming a precision. That distinction is the whole
finding of #422: a run declared fp32, trained bf16, and said nothing.

## M4 — the PRECISION control is INERT on LoRA at the SAVE boundary   [#424]

Both LoRA cells save `{'F32': 336}`. Identical. The 6-byte difference between
them is sidecar text, not weights.

This is correct peft behaviour — adapter parameters are kept in fp32 over a bf16
base — so it is not a training defect. But it does mean the save-side precision
comparison cannot discriminate the two LoRA cells, and a gate that reports PASS
over both is reporting a comparison it did not make. That is the #375/#382 shape:
a control that cannot fail, counted as a control that passed.

The load-side reading, added by #422, DOES discriminate here: `torch.bfloat16`
against `torch.float32` in the two cells. So the honest statement is narrower
than "precision is unverified on LoRA" —

    LoRA precision is VERIFIED AT LOAD and UNMEASURABLE AT SAVE.

Filed as #424. The fix is for the save-side precision gate to declare UNMEASURED
(95) on an adapter-only artifact rather than PASS, because "the artifact does not
contain the tensors whose dtype I was asked about" is an abstention, not a
confirmation.

## M5 — NVFP4 REFUSES on both methods, and that closes #268/#345

    full x nvfp4    rc 96, 10,401 bytes, no safetensors
    lora x nvfp4    rc 96, 10,437 bytes, no safetensors

Both refuse before any model is loaded — hence the absent post-load reading, which
is the correct rendering of "this never got far enough to have a dtype". The ~10 KB
written is the refusal record, not weights.

#268 held that the nvfp4/bf16 "control pair" declared ONE configuration under two
names. On this tree that is no longer true, and the evidence is behavioural rather
than textual: the two names now produce different outcomes — bf16 trains and saves
1014 BF16 tensors, nvfp4 refuses and saves nothing. A pair whose arms diverge by
an exit code and by the existence of an artifact is not one configuration.

INFERRED (not measured here): nvfp4 refuses because this tree has no NVFP4
execution path bound, not because the hardware lacks the capability. sm100 does
support NVFP4. Distinguishing "unimplemented" from "unsupported" needs a reading
this matrix does not take, and is left to #345.

---

## What this matrix does NOT show

- **Convergence.** 20 steps on 1 GPU proves the plane executes, saves, and labels
  honestly. It says nothing about whether the resulting weights are good.
- **Multi-GPU or multi-node.** Every cell is `--nodes 1 --gpus-per-node 1`.
- **Any model but one.** Qwen2.5-1.5B only. Model-agnosticism is claimed elsewhere
  and is not evidence produced here.
- **NVFP4 numerics.** Nothing was computed in NVFP4; the cells refused.
