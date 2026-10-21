---
name: fskills-vla
license: MIT
description: "Plans and judges vision-language-action (VLA) policy work strictly against FoundationScale's VLA registry: an eval of a SUPPORTED policy (LIBERO reproductions of gr00t, openpi, Long-WAM on GB200) becomes a launch spec behind the plan hash, and its report is judged against the published success rate and tolerance. Fine-tune, RL and driving requests are UNMEASURED (adapters exist, nothing reproduced) and unregistered policies are REFUSED. Do NOT use for LLM/VLM language training (fskills-training) or text benchmarks (fskills-evaluation)."
compatibility: "Needs Python>=3.10 and the foundationskills package (fskills CLI) plus FoundationScale importable (pip install -e <foundation-scale repo>) so the VLA registry (VLA_MODELS) and levels.judge can be read; eval additionally needs the backend's policy server environment for foundationscale.vla.eval (checkpoint/serve env per backend). Plan and verify run on CPU; the rollout runs on the estate and is only started through `fskills launch --confirm`. The skill never launches and never echoes a hash."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Plan the openpi-pi05 libero_spatial eval as an FS launch spec — I read the hash myself before confirming","Verify the gr00t-n1.7 libero_10 report: did the run reproduce 94.5 within its tolerance, or is the rate a lower bound?","I want to fine-tune pi05 on my own demos — is that measured in FoundationScale, or unmeasured?","Evaluate octo-base on LIBERO — has that policy ever been reproduced in the FS VLA registry?"]'
---

# vla — FoundationSkills VLA policy planning and judging (against the FoundationScale registry)

## Purpose

Plan and judge vision-language-action policy work **strictly** against FoundationScale's own VLA registry
(`foundationscale.upstream.vla_models.VLA_MODELS`). A row is SUPPORTED only when FS reproduced the upstream's published
LIBERO success rate on GB200 within a two-sample binomial tolerance — and this skill advertises exactly that.
`plan` renders FS's own Level 3 step (`levels.level3_plan`: entry `foundationscale.vla.eval`, backend's published
protocol) as an `fs_launch_spec`; nothing runs without `fskills launch --confirm <hash>` with the hash you read.
`verify` reads the report `foundationscale.vla.eval` wrote and judges it with FS's own `levels.judge`
(|measured − card| ≤ tolerance), writing a `vla_verdict`. Fine-tune / RL / driving are UNMEASURED; unregistered policies are REFUSED.

## When to use

- To plan an eval of a SUPPORTED policy: "evaluate openpi-pi05 on libero_spatial and judge it against the card".
- To judge a finished run: "does this report reproduce 98.8 ± 1.4, or is it a lower bound because episodes raised?".
- To see what FS actually measures: "can I fine-tune pi05 here, or is that unmeasured?".
- Not for LLM/VLM language training (fskills-training) or text benchmarks (fskills-evaluation).

## Inputs

| Field | Required | Meaning |
|---|---|---|
| `action` | no | `plan` (default) or `verify` |
| `model_id` | yes | policy id from FoundationScale's VLA registry |
| `mode` | no | `eval` (default), `finetune`, `rl` or `driving` |
| `checkpoint` | no | override the registry row's upstream checkpoint ref |
| `out` | yes (`plan` eval) | report path `foundationscale.vla.eval` will write (a file, not a directory) |
| `serve_cmd` | no | command that serves the policy (forwarded to the eval step) |
| `report` | yes (`verify`) | path to the report the eval wrote |

## Outputs

`plan` (eval of a SUPPORTED row) → `fs_launch_spec`: the FS Level 3 step (`python -m foundationscale.vla.eval --backend <b>
--model <ref> --suite <manifest> --out <report>`) plus the judgment block (card, tolerance, profile). It is never launched
by the skill. `verify` → `vla_verdict` (measured, gap, card, tolerance, episode_errors, verdict).

Exit codes: 0 PASS (a rendered eval spec; or a verify within tolerance), 5 RED (VL-HO-003), 95 UNMEASURED
(VL-HO-001/002), 96 REFUSED (VL-IN-001..005; nothing written).

## Running it on GB200

Plan/verify are CPU-side; the rollout is the backend's eval on the estate, behind the universal launch gate:

1. `fskills vla plan --model-id openpi-pi05-libero --mode eval --out vla/openpi_spatial.json [--checkpoint <ref>] [--serve-cmd <cmd>]`
   renders FS's Level 3 step and writes the `fs_launch_spec`. It never launches.
2. `fskills hash --spec <spec>`, then `fskills launch --spec <spec> --confirm <the hash you read> --no-submit`
   (also inside `srun --overlap --jobid=<hold> --gres=gpu:1`). Without the hash the launch is refused (96).
3. `fskills vla verify --model-id openpi-pi05-libero --report vla/openpi_spatial.json` reads the report, judges it with
   `levels.judge` against the registry row, and writes the `vla_verdict`.

## Scope

LIBERO checkpoint evals of the registry's SUPPORTED rows on GB200 aarch64 (gr00t-n1.7, openpi-pi05, Long-WAM), judged at
the published success rate and tolerance. Fine-tune, RL and driving lie inside FS's *planning* surface but outside its
*measurement* surface — always UNMEASURED. Language training is fskills-training; text benchmarks are fskills-evaluation.

## Validation rules

| Rule | Phase | Severity | When |
|---|---|---|---|
| VL-IN-001 | input | BLOCK | the policy has no row in FoundationScale's VLA registry (the refusal names the SUPPORTED ids) |
| VL-IN-002 | input | BLOCK | FoundationScale's VLA registry is not importable here, or registry_problems(VLA_MODELS) is non-empty |
| VL-IN-003 | input | BLOCK | plan eval: no report path (out), or out is an existing directory |
| VL-IN-004 | input | BLOCK | verify: the report is missing or unreadable, or lacks success_rate_pct/episode_errors |
| VL-IN-005 | input | BLOCK | verify: the report's backend or suite does not match the registry row being judged |
| VL-HO-001 | handoff | WARN | verify: an episode raised instead of reaching a verdict (kept in the denominator as a failure, so the rate is a lower bound) — UNMEASURED, never PASS |
| VL-HO-002 | handoff | WARN | fine-tune / RL / driving request, or a registry row that is not SUPPORTED: no reproduction or acceptance tolerance exists — UNMEASURED, never advertised |
| VL-HO-003 | handoff | BLOCK | verify: \|measured − published\| exceeds the row's tolerance — RED |

## What FoundationScale supports

14 SUPPORTED rows — LIBERO checkpoint evals reproduced on GB200 aarch64 within a two-sample binomial tolerance:

| Backend | Rows |
|---|---|
| gr00t-n1.7 | 4 LIBERO suites (goal, object, spatial, libero_10) |
| openpi-pi05 | 4 LIBERO suites (checkpoint evals, pi05) |
| Long-WAM | idm × 2 (spatial, goal) + cod × 4 (spatial, object, goal, libero_10) |

Declared-but-unreproduced adapters (pure renderers, **never** measured, hence UNMEASURED): fine-tune `Gr00tFinetuneSpec`,
`OpenPiFinetuneSpec`, `LongWamFinetuneSpec` (no fine-tune acceptance tolerance exists — the binomial margin is a
rollout-rate margin, and one measured fine-tune sits 3.8 pt under its card); RL `VerlVlaRlSpec` (no row, no declared
margin — published curves are success counts); driving `AlpamayoSftSpec` (evidence is eval_loss, not minADE).

## Verified on real reports

A4.10 (2026-10-21, CPU plan/verify over 43 real reports the VLA lane wrote on GB200):

- plan eval openpi-pi05 → PASS, renders `python -m foundationscale.vla.eval --backend openpi --model gs://openpi-assets/checkpoints/pi05_libero --suite libero_spatial --out <report>`; `fskills launch` without hash → 96.
- plan finetune → UNMEASURED 95 naming `foundationscale.vla.adapters.openpi.finetune.OpenPiFinetuneSpec`; unregistered id → REFUSED 96 (VL-IN-001).
- Measured openpi runs reproduce exactly: 98.8 gap 0.0; 92.4 gap 0.0 — all PASS; gr00t goal/object/spatial PASS.
- 4 openpi runs with 500/500 episode errors → UNMEASURED (a naive scorer would call them 0% RED).
- gr00t libero_10 is marginal: 6 of 10 runs RED (89.5–90.5 vs 94.5 ± 3.7) though the row is SUPPORTED at a 91.33 seed mean.
- 2 old-format gr00t reports lacking `episode_errors` → REFUSED (VL-IN-004).

## Failure handling

- Policy not registered / registry broken: REFUSED (96) with VL-IN-001 (naming the SUPPORTED ids) or VL-IN-002 (fix FS
  first). Nothing is written.
- plan eval without `out`, or verify with a missing/unreadable/mismatched report: REFUSED (96) with VL-IN-003/004/005 —
  run the eval spec first, or verify a report against its own row.
- Fine-tune / RL / driving, or a non-SUPPORTED row: UNMEASURED (95) with VL-HO-002 naming the adapter and the measurement
  gap; no spec is rendered.
- Episodes raised during the eval: UNMEASURED (95) with VL-HO-001 — the 0% is a lower bound, never RED and never PASS;
  fix the server/adapter and re-run every episode.
- Measured outside the row's tolerance: RED (5) with VL-HO-003 — do not claim the reproduction (see gr00t libero_10).

## FS interface

- Entry: `fskills vla` (`plan` / `verify`); emits `fs_launch_spec` and `vla_verdict`.
- APIs (FS-owned, never re-implemented): `foundationscale.upstream.levels.level3_plan` (spec), `levels.judge` (verdict),
  `foundationscale.upstream.vla_models.VLA_MODELS` (registry + `registry_problems`).
- Launch: `fskills launch --confirm <the hash you read from fskills hash> --no-submit` — the same universal gate.

## Worked examples

- **PASS (eval):** plan openpi-pi05-libero → spec rendered from FS's Level 3 step; hash, launch, verify → 98.8 vs 98.8
  (gap +0.0, tolerance 1.4), 0 episode errors → `vla_verdict` PASS; the 92.4 row passes the same way.
- **UNMEASURED (finetune):** `--mode finetune` on openpi-pi05 → 95 with VL-HO-002 naming `OpenPiFinetuneSpec` and the
  missing fine-tune acceptance tolerance; no spec rendered.
- **REFUSED (unregistered):** `--model-id octo-base-libero` → 96 with VL-IN-001 listing the 14 SUPPORTED ids.
- **UNMEASURED (broken run):** verify a run whose report has 500/500 `episode_errors` → 95 with VL-HO-001: the 0% rate
  is a lower bound, called a failure-in-denominator, not a RED verdict.
