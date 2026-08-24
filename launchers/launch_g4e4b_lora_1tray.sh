#!/bin/bash
# ============================================================================
# Gemma-4-E4B (VL, DENSE + PLE — enable_moe_block=False and num_experts=null,
# both measured off the base config.json; fix28 A1) — SFT-Taiwan-AIEC v3
# LoRA (AdamW) — 1 TRAY
# FIRST PRODUCTION RUN. Modeled on launch_omni_lora_foxbrain_4tray_mbs2.sh
# (4-tray LoRA, job-680 lineage) scaled DOWN: world = 1 node x 4 GPUs = 4.
#
# Scaling vs the 4-tray reference:
#   world 16 -> 4     DP 8 -> 4     EP 1 (NOT EP=world: that convention is
#   MoE-only and this base is measured dense; EP=world is reachable only if
#   cfg_get measures enable_moe_block=true on some future base)
#   GBS 16 MBS 1 kept (estate LoRA fixed knobs) -> ga = 16/(4/1)/1 = 4
#
# LoRA design (per g4_sft/LORA_EXPERIMENT_PLAN.md): on E4B the plan's L1/base4
# distinction COLLAPSES — L1's whole point is the MoE-expert adapter targets
# and this model has ZERO expert modules (0 of 2,130 base-header tensor names
# match any expert classifier). On a dense base this launcher DROPS the expert
# targets and RELABELS the arm base4, loudly, so no OUT_DIR or manifest ever
# wears the L1 name on an expert-free adapter set. What the plan fixes holds:
#   r=32, alpha=64, dropout=0.0, lr=1e-4  (Estate AdamW LoRA baseline)
#   targets = attention (linear_qkv, linear_proj) + dense MLP fc1/fc2
#   r/lr/alpha are NOT changed in the same run -- attribution (plan's rule).
#
# RUN ORDER (enforced by gates, not politeness):
#   1) PROBE=1 sbatch ...   -> 20 iters, save at 10+20, writes to *_probe dir,
#      verifies attach counts, trainable-param census, CoT survival, save path.
#   2) sbatch ...           -> production into a STABLE OUT_DIR (no job id)
#   3) sbatch --dependency=afterany:<prod_jobid> ...   (resume no-op chain;
#      exits 0 once last_iter >= train_iters, estate-proven pattern)
# ============================================================================

#SBATCH --job-name=g4e4b-lora-taiwan-L1
#SBATCH --nodes=1
#SBATCH --nodelist=<compute-node>
# PUBLICATION PLACEHOLDER: sbatch directives are static and cannot read
# environment variables, so this name could not be parameterized the way the
# runtime guards below were (FS_ALLOWED_NODE / FS_FORBIDDEN_NODES). Before any
# real sbatch submission, replace <compute-node> above with the short hostname
# of YOUR one allowed tray — the same value you export as FS_ALLOWED_NODE.
# sbatch no longer exists on the login nodes (finding #51), so this header is
# currently vestigial; it is retained deliberately (the sbatch path is a
# standing rule and is never deleted), and the runtime node guards below
# enforce the allow/deny policy independently of whether this line is ever
# read by a scheduler.
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=30
#SBATCH --mem=400G
#SBATCH --time=10-00:00:00
#SBATCH --partition=<group>
#SBATCH --output=logs/g4e4b_lora_taiwan_%j.out
#SBATCH --error=logs/g4e4b_lora_taiwan_%j.err
#SBATCH --exclusive

set -euo pipefail

# ----------------------------------------------------------------------------
# ESTATE ROOT + NODE-GUARD INPUTS — publication-safe parameterization.
# docs/DECISIONS.md forbids real account names, home paths and hostnames in
# this repo; the values live in the environment, never in source.
# ----------------------------------------------------------------------------
# CLUSTER_HOME: estate root. The ${CLUSTER_HOME:-$HOME} default is
# behaviour-preserving on the allowed tray — fs_container_backend.sh documents
# at its own line 587 that there $HOME IS the estate home — and the contract
# suite drives this script through env -i HOME=<sandbox>. Every path default
# below is rerooted on this variable.
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"

# FS_ALLOWED_NODE: REQUIRED, NO DEFAULT. The short hostname (prefix) of the
# ONE tray this estate may run on. An unset or empty value must REFUSE — a
# guard that cannot fire is not a guard, and defaulting it to anything would
# let a typo silently disable the standing rule that keeps this estate off
# another team's hardware.
# FS_FORBIDDEN_NODES: optional, space-separated denylist encoding the "never
# the other team's tray" rule without naming a real host in source. It is
# checked BEFORE the allowlist at both guard sites, so an explicit denial
# always beats a sloppy allow-prefix. Unset => the allowlist alone governs.
if [[ -z "${FS_ALLOWED_NODE:-}" ]]; then
  echo "FATAL: FS_ALLOWED_NODE is unset and has NO default. Set it to the short hostname (prefix) of your ONE allowed tray, e.g. FS_ALLOWED_NODE=<compute-node>; optionally export FS_FORBIDDEN_NODES='<other-team-node>' for the denylist. Refusing to run: an unconfigured node guard is a disabled standing rule." >&2
  exit 1
fi
FS_FORBIDDEN_NODES="${FS_FORBIDDEN_NODES:-}"

# _fs_node_forbidden HOST — succeeds if HOST prefix-matches any token of the
# space-separated FS_FORBIDDEN_NODES. $FS_FORBIDDEN_NODES is deliberately
# UNQUOTED in the for-list so word-splitting yields one host per token, and
# each ${_d}* is deliberately UNQUOTED in the case arm so it stays a prefix
# glob with the same semantics as the allowlist. Quoting either one breaks
# the denylist — do not "fix" them.
_fs_node_forbidden() {
  local _h=$1 _d
  for _d in $FS_FORBIDDEN_NODES; do
    case "$_h" in
      ${_d}*) return 0 ;;
    esac
  done
  return 1
}

# ----------------------------------------------------------------------------
# Paths — accel workspace (native Gemma4 lives here, per g4_sft/README.md)
# ----------------------------------------------------------------------------
WORKSPACE=$CLUSTER_HOME/Training-model/FoxBrain-omni-accel
REPO=$WORKSPACE/Megatron-Bridge
EXTRAS=$WORKSPACE/python-extras-mbridge                     # stays FIRST on PYTHONPATH (README trap 1)
LOG_DIR=$WORKSPACE/logs

HF_MODEL_PATH=$CLUSTER_HOME/pretraining_weights/Vision-Language-Models/Google/Gemma4/gemma-4-E4B-it
MEGATRON_CKPT=$WORKSPACE/converted_ckpts/gemma-4-E4B-it      # VERIFIED exists 2026-08-23 per the sibling full-FT launcher's in-file measurement (iter_0000000/ = 15 GB torch_dist DCP, 1,252 tensors, dense: zero expert FQNs). The iter_0000000/ preflight below stands regardless: an existing path is not proof of a complete conversion.

SQSH_DIR=$CLUSTER_HOME/SQSH-env
CONTAINER_SQSH=$SQSH_DIR/nemo-automodel-26-04_compute.sqsh

# Taiwan-AIEC v3 corpus — TRAIN split only. Test split exists but eval stays OFF
# (val loader is not modality-bucketed -> silent deadlock; SFT_TAIWAN_AIEC.md §6).
DATA_TRAIN=$CLUSTER_HOME/Post-training-Data/SFT-Taiwan-AIEC/formatted-gemma4-v3/train
FOXBRAIN_SFT_JSONLS="$DATA_TRAIN/foxbrain_identification.jsonl,$DATA_TRAIN/kenny_cot_think.jsonl,$DATA_TRAIN/kenny_notag_nothink.jsonl,$DATA_TRAIN/lilian_taiwan_multicat.jsonl"

# ----------------------------------------------------------------------------
# EXECUTION BACKEND — selected and PROVEN before anything below reads SLURM_*
# ----------------------------------------------------------------------------
# fix22 (s1, measured 2026-08-23): sbatch/srun AND the pyxis container plugin
# are absent from this estate, so the #SBATCH path below is currently
# unexecutable but PRESERVED (Slurm may come back). fs_container_backend.sh
# carries both backends: fs_backend_init selects FS_BACKEND (auto, or
# FS_BACKEND=slurm|enroot by hand), enforces the node guard — slurm arm: the
# allocation allowlist exactly as before; enroot arm: `hostname -s` ground
# truth, which nothing about this process can influence — and only afterwards
# mints the SLURM_* values the rest of this script still reads. Off-Slurm the
# minted id is numeric (MASTER_PORT math), unique per invocation, and recorded
# (it flows into the run manifest's --job-id and the gate-parsed log filename).
# shellcheck source=launchers/fs_container_backend.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fs_container_backend.sh"
fs_backend_init "$WORKSPACE"

# ----------------------------------------------------------------------------
# Recipe / parallelism / batch
# ----------------------------------------------------------------------------
RECIPE=${RECIPE:-gemma4_vl_e4b_peft_config}       # fix28 VERIFIED: registered by the estate patch landing with this change (recipes/gemma4_vl/gemma4_vl.py + __init__.py export; registry grep 4/4 -> 6/6 when applied). Preflight (4) greps the MEASURED recipe root $REPO/src/megatron/bridge/recipes/ — the old guard grepped $REPO/recipes/, which does not exist.
STEP_FUNC=${STEP_FUNC:-vlm_step}                  # estate-proven for Gemma4-VL (smoke header: llava_step -> KeyError 'images')
PEFT_SCHEME=lora

NODES=1
GPUS_PER_NODE=4
WORLD=$((NODES * GPUS_PER_NODE))

# MoE-ness is MEASURED from the base config, never assumed — this branch is
# ported from the sibling full-FT launcher, which solved the dense/MoE split
# first (fix28 Blocker 2). Measured for THIS base (A1): enable_moe_block is
# affirmatively False and num_experts is present-but-null, i.e. DENSE — so the
# old unconditional EP=$WORLD default below was MoE-shaped on a dense model.
# cfg_get reads text_config first (the keys live there), python3 first, grep
# last (UNVERIFIED single-occurrence assumption in the fallback arm).
cfg_get() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$HF_MODEL_PATH/config.json" "$1" <<'PY'
import json,sys
with open(sys.argv[1]) as f: cfg=json.load(f)
tc=cfg.get("text_config",{}) or {}
print(tc.get(sys.argv[2], cfg.get(sys.argv[2],"")))
PY
  else  # UNVERIFIED fallback: assumes the key appears exactly once in config.json
    grep -m1 -oE "\"$1\"[[:space:]]*:[[:space:]]*[^,}]+" "$HF_MODEL_PATH/config.json" | sed -E 's/^[^:]*:[[:space:]]*//' | tr -d '" '
  fi
}
[[ -f "$HF_MODEL_PATH/config.json" ]] || { echo "FATAL: HF model missing config.json: $HF_MODEL_PATH" >&2; exit 1; }
ENABLE_MOE_BLOCK=$(cfg_get enable_moe_block)
NUM_EXPERTS=$(cfg_get num_experts)
[[ -n "$ENABLE_MOE_BLOCK" ]] || { echo "FATAL: could not parse text_config.enable_moe_block from $HF_MODEL_PATH/config.json — refusing to GUESS the parallel geometry (an unreadable key BLOCKS; it never defaults)" >&2; exit 1; }

TP=${TP:-1}
ETP=${ETP:-1}
CP=${CP:-1}
# The TP cap is MEASURED, not inherited: num_key_value_heads=2 on BOTH layer
# types (A1; HF num_global_key_value_heads null, bridge coerces to 2), and TP
# must divide the KV-head count of every layer type -> TP in {1,2}; TP=4 is
# impossible here. PP is provider-forbidden (PLE needs input_ids on every
# stage; Gemma4E4BModelProvider.__post_init__ hard-asserts pp==1), and CP/SP
# mis-shard PLE by construction (the forward stashes full-sequence
# ple_per_layer while CP/SP shard hidden states — TODO P1b-harden), so the
# tray's stacked parallelism is DP.
[[ "$TP" =~ ^[12]$ ]] || { echo "FATAL: TP=$TP — E4B caps TP at 2 (measured num_key_value_heads=2, sliding AND full). Only TP=1|2." >&2; exit 1; }
[[ "$TP" != 2 ]] || echo "WARN: TP=2 is arithmetically legal (KV heads=2) but NOT tray-validated. PLE survives it ONLY because the PLE linears are replicated and the residual stream is unsharded between layers when sequence_parallel stays False (pinned). Prove with PROBE before production."
[[ "$CP" == 1 ]] || { echo "FATAL: CP=$CP — CP mis-shards PLE inputs (TODO P1b-harden). CP=1 only for this model today." >&2; exit 1; }
[[ "$ETP" =~ ^[0-9]+$ ]] || { echo "FATAL: ETP must be an integer, got '$ETP'" >&2; exit 1; }

if [[ "$ENABLE_MOE_BLOCK" == "true" || "$ENABLE_MOE_BLOCK" == "True" ]]; then
  MOE=1
  EP=${EP:-$WORLD}          # EP=world, estate convention — reachable ONLY on a measured-MoE base now
  [[ "$NUM_EXPERTS" =~ ^[0-9]+$ ]] || { echo "FATAL: enable_moe_block=$ENABLE_MOE_BLOCK but num_experts unparseable ('$NUM_EXPERTS') — refusing to guess EP geometry" >&2; exit 1; }
  (( NUM_EXPERTS % EP == 0 )) || { echo "FATAL: num_experts=$NUM_EXPERTS not divisible by EP=$EP" >&2; exit 1; }
  (( WORLD % (EP * TP * CP) == 0 )) || { echo "FATAL: WORLD=$WORLD not divisible by EP*TP*CP=$((EP*TP*CP))" >&2; exit 1; }
  MOE_OVERRIDES="model.expert_tensor_parallel_size=$ETP model.expert_model_parallel_size=$EP model.moe_token_dispatcher_type=alltoall model.moe_grouped_gemm=True"
else
  MOE=0
  # fix28 Q2: an EXPLICIT EP/ETP on a measured-dense base is REFUSED, never
  # silently coerced — honoring it would reintroduce exactly the mis-shard this
  # branch exists to prevent (zero experts; the 31B recipe hard-asserts the
  # same), and coercing quietly would make the banner and the run manifest
  # record a knob the operator did not write. The refusal names both the
  # instruction and the measured fact; unset the var to proceed dense.
  if [[ -n "${EP:-}" && "$EP" != "1" ]]; then
    echo "FATAL: EP=$EP set explicitly, but the base is DENSE (enable_moe_block=$ENABLE_MOE_BLOCK, num_experts=${NUM_EXPERTS:-null}). With zero experts, EP>1 silently mis-shards the model. Unset EP to launch dense (EP=1), or point HF_MODEL_PATH at a MoE base." >&2; exit 1
  fi
  if [[ "$ETP" != "1" ]]; then
    echo "FATAL: ETP=$ETP set explicitly, but the base is DENSE — expert tensor parallelism has nothing to shard. Unset ETP to launch dense (ETP=1)." >&2; exit 1
  fi
  EP=1
  MOE_OVERRIDES=""          # dense launcher carries NO MoE overrides by design; the omission is STATED (branch echo, preflight banner, manifest --ep 1) and the recipe pins EP=ETP=1 with a hard-assert
  echo "config says enable_moe_block=$ENABLE_MOE_BLOCK -> treating text tower as DENSE (EP=1, no MoE CLI overrides emitted)"
fi
DP=$((WORLD / TP / CP))
SEQ_LENGTH=${SEQ_LENGTH:-8192}        # matches official Taiwan base runs (SFT_TAIWAN_AIEC.md §5)
GLOBAL_BATCH_SIZE=${GBS:-16}          # estate LoRA fixed knob (LORA_EXPERIMENT_PLAN "Fixed for every arm")
MICRO_BATCH_SIZE=${MBS:-1}

# LoRA hyperparams — estate baseline (L1 arm), per LORA_EXPERIMENT_PLAN.md
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}          # scale alpha/r = 2.0; L3 arm would be 32 — deliberate, not here
LORA_DROPOUT=${LORA_DROPOUT:-0.0}     # UNVERIFIED: confirm recipe default before trusting; we pin explicitly
LORA_LR=${LORA_LR:-1e-4}              # AdamW LoRA baseline lr (README results table)
# fix42 — KNOB-PATH MUST_FIRE DRILL (env knob FS_PEFT_DRILL_RANK, fix41 idiom).
# Doctrine 3 applied to the four peft.* overrides: the recipe defaults equal
# this launcher's shipped defaults at 4 of 4 knobs (dim 32, alpha 64, dropout
# 0.0, and the four measured target strings), so on an undrilled launch
# "peft.dim landed at 32" and "the recipe default sat at 32 untouched" are
# INDISTINGUISHABLE — the happy path has zero discriminatory power over the
# exact silent-revert class fix42 repairs (an override accepted and then
# reverted post-composition leaves 32 either way). This drill perturbs
# LORA_RANK to a value NEITHER the recipe NOR this launcher defaults to, so
# the override-replay step below can only report the drill value if the
# override channel genuinely carried it. One source of truth: LORA_RANK alone
# feeds the banner, RUN_TAG/OUT_DIR (so a drill run can never contaminate the
# r32 checkpoint or its resume chain), the manifest's --effective peft.dim,
# CLI_OVERRIDES, and the replay probe's expectation. Scope, stated in the
# fix41 idiom: the drill discriminates 1 of the 4 knobs (dim); alpha/dropout/
# targets share the same carrier and composition call, and their coverage is
# asserted carrier-wide, never silently assumed.
if [[ -n "${FS_PEFT_DRILL_RANK:-}" ]]; then
  [[ "$FS_PEFT_DRILL_RANK" =~ ^[1-9][0-9]*$ ]] || \
    { echo "FATAL: FS_PEFT_DRILL_RANK='$FS_PEFT_DRILL_RANK' must be a positive integer rank — a non-integer drill would fail downstream for reasons unrelated to the knob path it exists to exercise." >&2; exit 1; }
  [[ "$FS_PEFT_DRILL_RANK" != "32" ]] || \
    { echo "FATAL: FS_PEFT_DRILL_RANK=32 equals BOTH the recipe default and this launcher's default — a drill that perturbs nothing can prove nothing about whether the override landed. Choose a rank the defaults are NOT (96 is the rehearsed value)." >&2; exit 1; }
  echo "============================================================"
  echo "DRILL: FS_PEFT_DRILL_RANK=$FS_PEFT_DRILL_RANK — LORA_RANK perturbed off"
  echo "both defaults (32). The override-replay step MUST resolve peft.dim to"
  echo "$FS_PEFT_DRILL_RANK and the CLEAR arm refuses to proceed on any other"
  echo "value — including 32, which would mean the override channel reverted to"
  echo "the default and every undrilled green on this launcher was vacuous."
  echo "============================================================"
  LORA_RANK=$FS_PEFT_DRILL_RANK
fi
# EXTRACTION CONTRACT — READ BEFORE INSERTING ANYTHING BELOW THIS LINE.
# test_launcher_contracts.sh's lora_arm_block() lifts the arm-switching logic
# with `sed -n '/^EXPERT_TARGETS=/,/^fi$/p'` and EVALS it verbatim, on purpose:
# the harness must exercise the shipped branch, never a paraphrase of it. That
# makes the span from the line below to the arm block's closing `fi` (:258) a
# contract surface. A top-level `if ... fi` inserted inside that span silently
# truncates the extraction at ITS `fi`, so the harness evals the intruder and
# LORA_ARM never gets set — measured: fix42's drill block landed here and took
# 5 contract legs red at once, including a MUST_FIRE that went UNREACHABLE
# because the mined WARN needle was no longer in the extracted text. The drill
# above therefore sits BEFORE this line. Put new top-level blocks above it or
# below :258; never between.
EXPERT_TARGETS=${EXPERT_TARGETS:-1}   # 1 = L1 arm (adds MoE expert targets); 0 = legacy 4-module set — MEANINGFUL ONLY on a MoE base
# fix39 (measured 2026-08-24 on the real E4B module tree — 1556 modules offered
# to the matcher, controls green): the pre-fix dense strings 'mlp.linear_fc1' /
# 'mlp.linear_fc2' attached ZERO modules. ModuleMatcher.match decides by
# (leaf_name == pattern) OR wildcard_match(pattern, full_FQN), and
# wildcard_match anchors the whole pattern at both ends with '*' as the ONLY
# wildcard (peft/utils.py:208) — so a dotted pattern with no '*' matches only
# an FQN byte-equal to itself, and no module's FQN is 'mlp.linear_fc1' (the
# real spelling doubles the segment: '...mlp.mlp.linear_fc1', a Gemma4MLP-ish
# wrapper owning an inner mlp). The run trained 84 of the 168 modules the
# header above declares, with no error anywhere — the founding-bug shape.
# Two spellings measured attaching 42/42 each on the dense base:
#   bare leaf names (estate default): linear_fc1 / linear_fc2
#   anchored FQN wildcards:           *.mlp.mlp.linear_fc1 / *.mlp.mlp.linear_fc2
# Measured identical HERE (0 experts of 1556), they differ on a MoE base —
# stated as PREDICTION, only the dense base is measured tonight: bare leaf
# names would ALSO attach expert linear_fc1/fc2 (the expert strings' own
# spelling tells us those modules share the leaf names), collapsing the
# plan's base4/L1 partition into the base list and silently re-arming the
# one-run-two-labels defect with module sets instead of run names. The
# doubled-mlp wildcard selects the DENSE MLP only on either base kind:
# '^(.*)\.mlp\.mlp\.linear_fc1$' cannot absorb an '.experts.experts.'
# segment. Chosen: the wildcard, so 'dense MLP fc1/fc2' in the header means
# the same module set on every base — attribution is the whole run plan.
# linear_qkv / linear_proj stay bare leaf names (estate-default spelling,
# measured 42/42 by leaf equality; attention has no expert twin to confuse).
LORA_TARGETS_BASE="linear_qkv,linear_proj,*.mlp.mlp.linear_fc1,*.mlp.mlp.linear_fc2"
# fix39 expert-arm strings — PREDICTION, NOT measurement: the pre-fix expert
# strings were the same unmatchable shape (dotted, no '*') and would attach
# ZERO expert modules on a real MoE base — the L1 arm silently training zero
# expert adapters. Tonight's base is dense so these stay dropped by the
# MOE!=1 branch below regardless (verify: 'enable_moe_block' measured False,
# 0 of 1556 is_expert_linear). Repaired in the same anchored-wildcard shape
# as the dense strings. Their discharge on any future MoE base is the step
# (5) census: it runs the real matcher over that tree's live population
# pre-GPU and BLOCKS the launch if this prediction is wrong. Note the second
# unmeasured dependency this carries honestly: the spelling
# 'mlp.experts.experts.' itself is unverified for a live E4B-MoE tree — the
# census, not this comment, is trusted to catch that.
LORA_TARGETS_EXPERT="*.mlp.experts.experts.linear_fc1,*.mlp.experts.experts.linear_fc2"
if [[ "$MOE" != "1" ]]; then
  # DENSE base (measured above): the expert strings would match ZERO modules —
  # appending them is a vacuous instruction, and keeping the "L1" label on the
  # resulting expert-free adapter set would be the one-run-two-labels lie the
  # LORA_ARM comment below exists to prevent. Coerce the list and RELABEL the
  # arm, loudly (fix28 Q2: vacuous instructions are coerced with a label fix;
  # geometry-corrupting ones like EP>1 are refused — see the branch above).
  if [[ "$EXPERT_TARGETS" == "1" ]]; then
    echo "WARN: EXPERT_TARGETS=1 but the base is DENSE (enable_moe_block=$ENABLE_MOE_BLOCK, num_experts=${NUM_EXPERTS:-null}) — dropping expert targets (0 modules would match) and relabelling arm L1->base4 so OUT_DIR/manifest cannot wear the L1 name on an expert-free run."
  fi
  LORA_TARGETS="${LORA_TARGETS_BASE}"
  LORA_ARM=base4
elif [[ "$EXPERT_TARGETS" == "1" ]]; then
  LORA_TARGETS="${LORA_TARGETS_BASE},${LORA_TARGETS_EXPERT}"
  LORA_ARM=L1
else
  LORA_TARGETS="${LORA_TARGETS_BASE}"
  LORA_ARM=base4
fi
# FQN SPELLING — MEASURED, NO LONGER AN ASSUMPTION (fix39, 2026-08-24): fix28
# Q3's "0/N verified against a live E4B tree" phrasing is retired by
# measurement, not by assertion. Every shipped string is re-proven against
# the LIVE module tree on EVERY launch by the step-(5) census, which asks the
# shipped ModuleMatcher itself — the only oracle that decides attachment —
# over the full module population before any GPU time is spent. Measured on
# this dense base (1556 modules offered; controls green): linear_qkv 42,
# linear_proj 42, *.mlp.mlp.linear_fc1 42, *.mlp.mlp.linear_fc2 42 — 168
# attachments of the 168 the header declares. The expert strings are the
# same repaired shape and remain a stated PREDICTION here (this base carries
# 0 expert modules; they stay dropped by the MOE!=1 branch above), with the
# same census as their pre-GPU discharge on any future MoE base.
# LORA_ARM is load-bearing, not cosmetic: it feeds RUN_TAG below, which feeds
# OUTPUT_DIR *and* the manifest --run-id. Before this variable existed the tag
# spelled "L1" as a literal, so EXPERT_TARGETS=0 and EXPERT_TARGETS=1 resolved
# to the SAME OUTPUT_DIR — and because the auto-resume state machine keys off
# $CKPT_DIR/latest_checkpointed_iteration.txt, launching the second arm would
# have silently RESUMED the first arm's optimizer state under a different LoRA
# target set. That is not a bookkeeping gap; it is one run wearing two labels,
# which is exactly the confusion the 24-run-split audit could not untangle.

# Early-save requirement: PROBE forces a real save at iters 10 and 20 (trap 9:
# a run looks healthy until its FIRST save; confirm iter_* + latest file before
# trusting). Production save cadence matches the official Taiwan runs.
if [[ "${PROBE:-0}" == "1" ]]; then
  TRAIN_ITERS=20
  SAVE_INTERVAL=10
  RUN_SUFFIX=_probe
else
  SAVE_INTERVAL=${SAVE_INTERVAL:-250} # official Taiwan base runs; TRAIN_ITERS computed below from row count
  RUN_SUFFIX=""
fi
EVAL_INTERVAL=100000
EVAL_ITERS=0
LOG_INTERVAL=1
EPOCHS=${EPOCHS:-2}

# STABLE output dir (NO ${SLURM_JOB_ID}) — required for the afterany resume
# chain to land in the same OUT_DIR (SFT_TAIWAN_AIEC.md §5).
RUN_TAG=1t_lora_${LORA_ARM}_r${LORA_RANK}a${LORA_ALPHA}_ep${EP}_gbs${GLOBAL_BATCH_SIZE}_seq${SEQ_LENGTH}
OUTPUT_DIR=$WORKSPACE/results/${RECIPE}_${RUN_TAG}${RUN_SUFFIX}
PREFLIGHT_DIR=$OUTPUT_DIR/preflight
mkdir -p "$OUTPUT_DIR" "$PREFLIGHT_DIR" "$LOG_DIR"

# enroot-arm runtime setup (slurm arm returns immediately): provenance-checked
# idempotent enroot create (the g4export lesson: a name match is not origin),
# the GPU-drain gate (s8d), then our own stdout/stderr tee into the SAME file
# the post-run gates parse below ($OUTLOG). Without the tee there is no
# #SBATCH --output capture off-Slurm and the gates would grep another run's
# log — or no log at all, which they treat as a failure anyway.
fs_backend_runtime_setup "$CONTAINER_SQSH" "$GPUS_PER_NODE" "$LOG_DIR/g4e4b_lora_taiwan_${SLURM_JOB_ID}.out"

# ----------------------------------------------------------------------------
# Environment — GB200 single tray. bond0/IB pins from the smoke are for
# INTER-NODE NCCL on r04; world=1 tray does not cross trays, so they are
# omitted. <other-team-node> must NEVER appear (other team's node).
# (2026-08-23 addendum: "omitted" describes the pyxis-era reasoning for this
# env block. The OFF-SLURM enroot arm additionally sets the bond0 pins and
# NCCL_MNNVL_ENABLE=0 per s8b/s8c — inside fs_container_backend.sh, with the
# single-tray denominator of s8c stated honestly there. On one tray they are
# at worst inert insurance.)
# ----------------------------------------------------------------------------
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch"

# Gemma4: full-attn layers use head_dim=512 -> cuDNN fused attn required
# (FlashAttention caps at 256); container defaults NVTE_FUSED_ATTN=0 (smoke).
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=1
export MASTER_PORT=${MASTER_PORT:-$(( 29400 + ${SLURM_JOB_ID:-211} % 1000 ))}
# The backend pre-sets MASTER_ADDR=127.0.0.1 on the enroot arm (single-tray
# rendezvous, resolvable inside the container regardless of /etc/hosts) and
# scontrol is measured ABSENT off-Slurm (s1). Under sbatch the scontrol
# derivation is kept; `|| true` turns a missing scontrol into the hostname
# fallback instead of a pipefail abort on this very assignment under set -e.
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null | head -n1 || true)
  MASTER_ADDR=${MASTER_ADDR:-$(hostname)}   # hostname is correct for 1 node
  export MASTER_ADDR
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH=${EXTRAS}:${REPO}/src:${REPO}/3rdparty/Megatron-LM
export HF_HOME=$CLUSTER_HOME/.hf_cache   # shared torch/HF stack
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_PROJECT=${WANDB_PROJECT:-g4e4b-sft-taiwan}

# THE CoT TRAP GUARD. collate.py's training template patch is on by default;
# export explicitly so a stray FOXBRAIN_GEMMA4_KEEP_COT=0 in the environment
# cannot silently strip <|channel>thought CoT -> zero supervised tokens.
export FOXBRAIN_GEMMA4_KEEP_COT=1
export FOXBRAIN_SFT_JSONLS

# ----------------------------------------------------------------------------
# PREFLIGHT — fails loudly BEFORE GPU time is burned
# ----------------------------------------------------------------------------
echo "============================================================"
echo "Gemma-4-E4B — Taiwan-AIEC v3 LoRA (${LORA_ARM} arm — a dense base coerces L1 to base4, fix28) — 1 TRAY (world=$WORLD)"
echo "============================================================"
echo "Backend:       $FS_BACKEND$([[ "$FS_BACKEND" == enroot ]] && printf ' (enroot+torchrun; job-id minted locally, scontrol absent per s1)')"
echo "Job:           ${SLURM_JOB_ID:-N/A}  node=${SLURM_JOB_NODELIST:-N/A}  PROBE=${PROBE:-0}"
echo "Recipe:        $RECIPE (peft=$PEFT_SCHEME step=$STEP_FUNC)"
echo "HF:            $HF_MODEL_PATH"
echo "Ckpt:          $MEGATRON_CKPT -> $OUTPUT_DIR/checkpoints"
echo "MoE:           $MOE (enable_moe_block=$ENABLE_MOE_BLOCK num_experts=${NUM_EXPERTS:-null} — measured from config.json; a dense verdict emits NO MoE CLI overrides)"
echo "Parallelism:   TP=$TP (cap 2, measured KV heads=2) ETP=$ETP EP=$EP CP=$CP  DP=$DP"
echo "Batch:         GBS=$GLOBAL_BATCH_SIZE MBS=$MICRO_BATCH_SIZE seq=$SEQ_LENGTH ga=$((GLOBAL_BATCH_SIZE / DP / MICRO_BATCH_SIZE))"
echo "LoRA:          r=$LORA_RANK alpha=$LORA_ALPHA drop=$LORA_DROPOUT lr=$LORA_LR targets=[$LORA_TARGETS]"
echo "LoRA-FQN:      target spellings re-proven against the LIVE module tree by the step-(5) census on every launch (oracle = shipped ModuleMatcher over the full population, controls first; fix39 retires fix28-Q3's UNVERIFIED)"
echo "Recompute:     OFF for E4B by construction (recipe pins + no CLI override; A2: recompute would train WITHOUT PLE gradients)"
echo "CoT:           FOXBRAIN_GEMMA4_KEEP_COT=$FOXBRAIN_GEMMA4_KEEP_COT"
echo "Eval:          DISABLED ($EVAL_INTERVAL/$EVAL_ITERS) — val loader deadlocks"
echo "============================================================"

# (0) allowed-node guard — standing rule
# ALLOWLIST, not blocklist, and no default for the unset case. The earlier shape
# here was `${SLURM_JOB_NODELIST:-<compute-node>} == *<other-team-node>*`, which failed OPEN
# twice over: an unset nodelist substituted the allowed node and passed, and any
# node that simply is not <other-team-node> passed. A guard that cannot fire on the state
# it was written to catch is the vacuous shape this repository refuses; the
# sibling full-FT launcher already uses the closed form and this now matches it.
# Both arms of this guard now run inside fs_backend_init (sourced above),
# BEFORE any SLURM_* value was minted — so the guard can no longer be
# satisfied by a string the launcher itself wrote (the failure mode the
# ALLOWLIST comment above was written to prevent). The slurm arm kept the two
# original checks verbatim in effect; the enroot arm compares the allowlist
# against `hostname -s` (kernel ground truth) and REFUSES any pre-set
# SLURM_JOB_ID/SLURM_JOB_NODELIST off-Slurm. What remains here is a cheap
# defense-in-depth re-check: on the enroot arm it still reads the kernel live
# and never the minted copy.
if [[ "$FS_BACKEND" == slurm ]]; then
  # Denylist FIRST (defence in depth): an explicit denial must beat a sloppy
  # allow-prefix, so a denied tray refuses even if FS_ALLOWED_NODE would match.
  if _fs_node_forbidden "${SLURM_JOB_NODELIST:-}"; then
    echo "FATAL: STANDING RULE VIOLATION: landed on '${SLURM_JOB_NODELIST:-<unset>}', which matches an entry in FS_FORBIDDEN_NODES (currently '$FS_FORBIDDEN_NODES'). That tray belongs to another team and is never allowed, whatever the allowlist says — scancel this job." >&2; exit 1
  fi
  case "${SLURM_JOB_NODELIST:-}" in
    # ${FS_ALLOWED_NODE} is deliberately UNQUOTED: the trailing * must act as a
    # prefix glob so a nodelist like '<tray>,<tray2>' still matches — the exact
    # semantics of the original literal prefix. Quoting it would force
    # full-string equality and silently break the guard. Do not "fix" this.
    ${FS_ALLOWED_NODE}*) ;;
    *) echo "FATAL: STANDING RULE VIOLATION: landed on '${SLURM_JOB_NODELIST:-<unset>}'. Only nodelists matching the prefix in FS_ALLOWED_NODE (currently '${FS_ALLOWED_NODE}') are allowed, and anything in FS_FORBIDDEN_NODES (currently '$FS_FORBIDDEN_NODES', another team's tray) refuses before the allowlist is even consulted. scancel this job." >&2; exit 1 ;;
  esac
else
  _node_now=$(hostname -s)
  # Denylist FIRST, same rule as the Slurm arm above (see _fs_node_forbidden).
  if _fs_node_forbidden "$_node_now"; then
    echo "FATAL: STANDING RULE VIOLATION (off-Slurm re-check): hostname -s reports '$_node_now', which matches an entry in FS_FORBIDDEN_NODES (currently '$FS_FORBIDDEN_NODES'). That tray belongs to another team — denied outright, before the allowlist is even consulted. Run this estate only on your allowed tray." >&2; exit 1
  fi
  case "$_node_now" in
    # ${FS_ALLOWED_NODE} is deliberately UNQUOTED: prefix-glob semantics, same
    # as the Slurm arm above — quoting would turn the allowlist into an
    # exact-match and silently break the guard. Do not "fix" this.
    ${FS_ALLOWED_NODE}*) ;;
    *) echo "FATAL: STANDING RULE VIOLATION (off-Slurm re-check): hostname -s reports '$_node_now'. Only hosts matching the FS_ALLOWED_NODE prefix (currently '${FS_ALLOWED_NODE}') are allowed." >&2; exit 1 ;;
  esac
fi

# (1) files / dirs
[[ -f "$CONTAINER_SQSH" ]] || { echo "FATAL: container missing: $CONTAINER_SQSH" >&2; exit 1; }
[[ -d "$HF_MODEL_PATH" ]] || { echo "FATAL: HF source missing: $HF_MODEL_PATH" >&2; exit 1; }
PYTHONPATH_HEAD=${PYTHONPATH%%:*}
[[ "$PYTHONPATH_HEAD" == "$EXTRAS" ]] || { echo "FATAL: \$EXTRAS is not FIRST on PYTHONPATH (README trap 1)" >&2; exit 1; }
for f in ${FOXBRAIN_SFT_JSONLS//,/ }; do
  [[ -s "$f" ]] || { echo "FATAL: corpus file missing/empty: $f (missing files must be a hard error)" >&2; exit 1; }
done
ROWS=$(cat ${FOXBRAIN_SFT_JSONLS//,/ } | wc -l)
echo "Preflight: $ROWS supervised rows across 4 train JSONLs"
# train_iters sized to the REAL corpus so the cosine actually anneals (trap:
# a run shorter than train_iters never reaches the LR floor).
STEPS_PER_EPOCH=$(( (ROWS + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
TRAIN_ITERS=${TRAIN_ITERS:-$(( EPOCHS * STEPS_PER_EPOCH ))}
echo "train_iters=$TRAIN_ITERS ($EPOCHS epochs x ceil($ROWS/GBS)) save=$SAVE_INTERVAL"

# (2) E4B identity + KV-head sanity, read from the HF config, not assumed.
#     EP geometry and divisibility are already decided (branch above, MoE arm
#     only); this step proves we loaded the E4B config and records the measured
#     keys in the log — including BOTH KV counts, so a future base swap forces a
#     human to re-derive the TP cap rather than inherit it. It reads only keys
#     A1 verified present (hidden_activation exists; hidden_act is ABSENT — the
#     spelling trap is a provider concern and stays out of launchers).
if command -v python3 >/dev/null; then
python3 - "$HF_MODEL_PATH/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
t = cfg["text_config"]
assert t["num_hidden_layers"] == 42, f"expected 42 layers, got {t['num_hidden_layers']}"
assert t["hidden_size"] == 2560, "hidden != 2560 — not the E4B config we think we loaded"
nkvh = t["num_key_value_heads"]; ng = t.get("num_global_key_value_heads")
print(f"config.json: layers=42 hidden=2560 num_key_value_heads={nkvh} num_global_key_value_heads={ng} enable_moe_block={t.get('enable_moe_block')} num_experts={t.get('num_experts')}")
assert nkvh == 2 and (ng in (None, 2)), f"KV head counts changed ({nkvh}/{ng}) — re-derive the TP cap (currently 2) before launching"
PY
else
  { echo "FATAL: no host python3 — cannot confirm E4B config identity pre-launch. The previous behaviour (skip this check behind an UNVERIFIED comment) was a fail-open path: the geometry branch above may then have run on the unverified grep fallback." >&2; exit 1; }
fi

# (3) converted Megatron ckpt must exist (bridge recipes here load pre-converted).
if [[ ! -d "$MEGATRON_CKPT/iter_0000000" ]]; then
  echo "FATAL: no converted E4B Megatron ckpt at $MEGATRON_CKPT/iter_0000000" >&2
  echo "  Convert first, mirroring how converted_ckpts/gemma-4-31B-base (job 1952)" >&2
  echo "  was built. UNVERIFIED exact CLI — locate with: ls \$REPO/scripts | grep -i convert" >&2
  echo "  Then re-run with PROBE=1." >&2
  exit 1
fi

# (4) recipe must actually be registered, and must be an E4B-shaped config.
#     fix28: the guard now greps the MEASURED recipe root —
#     $REPO/src/megatron/bridge/recipes/ exists; $REPO/recipes/ does NOT, which
#     is why the old guard FATALed on every launch AND misdirected the operator
#     into a non-path. The guard must still be able to fail, and it can: it
#     fires whenever RECIPE names (a) something with no `def` in the real tree,
#     or (b) a def missing from the package __init__ export — state (b) is a
#     REAL halfway state (E1 applied, E2 forgotten), and the message says so.
RECIPE_ROOT="$REPO/src/megatron/bridge/recipes"
if ! grep -RqE "def +${RECIPE}\b" "$RECIPE_ROOT/" 2>/dev/null; then
  echo "FATAL: recipe '$RECIPE' has no def under $RECIPE_ROOT/ (the measured recipe root)" >&2
  echo "  E4B-shaped candidates found:"; grep -RhoE "def [a-zA-Z0-9_]*(e4b|E4B)[a-zA-Z0-9_]*" "$RECIPE_ROOT" 2>/dev/null | sort -u >&2
  echo "  If none: apply the fix28 estate patch first (it adds gemma4_vl_e4b_peft_config / gemma4_vl_e4b_foxbrain_sft_config)." >&2
  echo "  Do NOT point RECIPE at gemma4_vl_26b_*: MoE-shaped defaults on a dense base — and the 26B SMOKE configs train on synthetic mock data and ignore FOXBRAIN_SFT_JSONLS entirely (fix28 A3)." >&2
  exit 1
fi
if ! grep -qE "\b${RECIPE}\b" "$RECIPE_ROOT/gemma4_vl/__init__.py" 2>/dev/null; then
  echo "FATAL: recipe '$RECIPE' is defined under $RECIPE_ROOT but NOT exported from $RECIPE_ROOT/gemma4_vl/__init__.py — the fix28 estate patch is half-applied (E1 without E2)." >&2
  exit 1
fi

# (5) PROVE THE TARGET STRINGS ATTACH — with the matcher that will attach them.
#     fix39 (measured 2026-08-24, <compute-node>, in the training container): the
#     pre-fix census scored each target with `grep -cF "$t"` over a module
#     dump — a SUBSTRING oracle certifying a decision made by
#     ModuleMatcher.match, whose wildcard compiles to a FULLY ANCHORED regex
#     with '*' as the only wildcard (peft/utils.py:208). The oracles disagreed
#     on this very tree in BOTH directions: grep scored the shipped
#     'mlp.linear_fc1' 42 (substring of the real doubled '.mlp.mlp.' spelling)
#     while the matcher scored it 0 of 1556 — and grep scored the CORRECT
#     wildcard repairs 0, so the step that claimed to prevent the silent
#     no-op both certified it AND blocked its own repair. The census below
#     asks the shipped ModuleMatcher itself, over the same population PEFT's
#     walk offers (every module: walk defaults leaf_only=False —
#     peft/base.py:120-124, walk_utils.py:224,232; a leaf-only census read
#     'linear_proj' as 0 because the real TERowParallelLinear OWNS a child —
#     control 3 of the probe exists to catch exactly that narrowing class),
#     in the training interpreter stack, and the probe refuses to render any
#     verdict unless its MUST_FIRE / MUST_NOT_FIRE / anti-narrowing controls
#     pass. Exit vocabulary is the shipped one, unrenumbered:
#       0 CLEAR      — every shipped target attaches >= 1 module;
#       1 BLOCKED    — a NAMED target attaches ZERO under the real matcher
#                      (fix the strings, not the census);
#       3 UNMEASURED — the probe ABSTAINED (controls failed / build failed /
#                      matcher API drifted): fix the census, never bypass it.
#     fix40 addendum (MEASURED 2026-08-24 on <compute-node>): the vocabulary above
#     is the PROBE's own exit code, and the launcher never sees it. torchrun
#     maps ANY nonzero child exit to 1 (ChildFailedError; the receipt is
#     tonight's broken run, whose log carried CENSUS_VERDICT=UNMEASURED
#     beside PROBE_RC=1). The triage below therefore keys on the printed
#     CENSUS_VERDICT= line — the only three-valued signal measured to
#     survive the wrapper — with rc demoted to one corroborating bit. The
#     rc=3 arm this note retired had been certifying coverage that could
#     never once run: unreachable code reading as a shipped control.
CENSUS_PROBE=""
for _cp in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lora_target_census.py" \
           "$WORKSPACE/g4_sft/lora_target_census.py"; do
  [[ -f "$_cp" ]] && { CENSUS_PROBE="$_cp"; break; }
done
[[ -n "$CENSUS_PROBE" ]] || CENSUS_PROBE=$(find "$REPO" -name lora_target_census.py -print -quit 2>/dev/null)
[[ -n "$CENSUS_PROBE" ]] || \
  { echo "FATAL: lora_target_census.py not found. Searched: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ (shipped beside the launcher, fix39), $WORKSPACE/g4_sft/ (container-visible location; the old dump tool's measured precedent), then all of $REPO." >&2; exit 1; }
echo "Preflight: LoRA target census via $CENSUS_PROBE (oracle = the shipped ModuleMatcher itself; grep kept only as a labelled secondary column that decides nothing)"
CENSUS_OUT=$PREFLIGHT_DIR/target_census.txt
# #78 — MACHINE-READABLE HALF of this census (the sibling --out patch on
# lora_target_census.py, assumed landed). The SAME probe run below also
# writes $ADAPTER_MODULES, the JSON attachment census fs_live_save_gate()
# now demands via --adapter-modules. Three lines below are load-bearing:
#  - LOCATION $OUTPUT_DIR/fs_gate/: OUTSIDE the judged iter_* tree on
#    purpose — written inside it, the loader's own tautology guard
#    (tools/live_save_gate.py:742) refuses the census as self-certifying.
#  - mkdir BEFORE the probe runs: the probe's write is atomic (tmp file +
#    rename inside the target dir) and cannot create the parent itself; a
#    missing parent would surface only AFTER a CLEAR verdict, as the
#    claim-vs-disk gap this estate already measured once (doctrine 4).
#  - rm -f of any STALE artifact: the CLEAR arm below certifies PRESENCE.
#    A previous run's surviving file would let a probe that wrote NOTHING
#    this run pass that check — presence must mean THIS probe wrote it, or
#    the check certifies a foreign census and zero units were measured
#    tonight (doctrine 1). This is the line that makes the check below
#    unfakeable by leftovers.
# Container visibility is inherited from the same bind-mount precedent as
# $CENSUS_PROBE itself (fix39, below): the probe writes this path from
# INSIDE the container. If that precedent ever breaks, the probe cannot
# write where the host reads and the CLEAR-arm check fails CLOSED — never
# open, and never a launch failure on an UNMEASURED verdict: that check
# lives inside the CLEAR arm only, so the verdict triage below stays
# authoritative over every non-CLEAR outcome (the fix41 drill included).
ADAPTER_MODULES="$OUTPUT_DIR/fs_gate/adapter-modules.json"
mkdir -p "$OUTPUT_DIR/fs_gate" || \
  { echo "FATAL: cannot create $OUTPUT_DIR/fs_gate (for ADAPTER_MODULES=$ADAPTER_MODULES) — the census probe's atomic --out write needs the parent directory; refusing now rather than failing a CLEAR census on a missing artifact afterwards (doctrine 4)." >&2; exit 1; }
rm -f "$ADAPTER_MODULES" || \
  { echo "FATAL: cannot remove stale $ADAPTER_MODULES before the census — a leftover artifact would let the CLEAR-arm presence check below certify a census THIS probe never wrote (the measured claim-vs-disk gap). Refusing (doctrine 4)." >&2; exit 1; }
echo "Preflight: machine-readable adapter census -> $ADAPTER_MODULES (--out on the same probe run; deliberately outside the judged iter_* tree per live_save_gate.py:742's tautology guard)"
# Invocation MEASURED (fix39): the probe's argparse is --hf_model_path
# (required) plus --ep ONLY — the pre-fix attempt passed --hf_path/--recipe
# and would have died in argparse before censusing anything — and the provider
# calls mp.initialize_model_parallel, which without RANK dies on the env://
# rendezvous ('environment variable RANK expected, but not set') — so the
# probe runs under torchrun with a self-contained single-process rendezvous
# (the measured spell: --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1;
# port = the launcher's own minted MASTER_PORT, safe because this rendezvous
# dies with the probe — the measurement's literal 29517 was only an
# unminted-shell substitute). The single quotes around '$LORA_TARGETS' are
# load-bearing: they belong to the CONTAINER's bash (the outer double quotes
# are the host shell's, which still expands the variables), so the '*' now
# legitimate in the repaired patterns cannot glob-expand against $REPO's
# file listing — a file named like a pattern must never rewrite a preflight
# answer. Container visibility of the probe path is inherited from the old
# dump tool's measured precedent ($WORKSPACE paths are in-container
# readable); if that precedent ever stops holding, the rc triage below fails
# CLOSED, never open.
# fix41 — CENSUS MUST_FIRE DRILL (env knob FS_CENSUS_DRILL_BUILD_FAILURE=1).
# Doctrine 3 is not satisfied by WRITING the UNMEASURED arm; it is satisfied
# by WATCHING it fire. fix40 made the arm reachable on paper by keying the
# triage on the CENSUS_VERDICT line, but no genuine abstention has ever
# travelled that arm through the real launcher — and an arm never observed
# to fire reads as coverage exactly the way the retired rc=3 arm did. This
# drill re-creates the one MEASURED genuine abstention trigger (fix40
# receipt, 2026-08-24, <compute-node>): CUDA_VISIBLE_DEVICES= in front of the
# in-container torchrun makes the probe's build_model die with
# "RuntimeError: No CUDA GPUs are available", and the real probe then prints
# its OWN CENSUS_VERDICT=UNMEASURED (0 of N targets certified: model build
# failed) and exits 3 (observed rc: 1 past torchrun's ChildFailedError — the
# one measured bit). SCOPE, and it is load-bearing: the empty assignment is
# prepended INSIDE this census payload string only. It is never exported, so
# it cannot ride the run_in_container --env loop into the env probe or the
# training step, and the drain gate reads host-side nvidia-smi regardless —
# a drill that blinds the launcher upstream of the census dies before the
# arm it exists to exercise, reporting a "fire" that measured nothing (the
# fix41-§4 harness-error class: legs that pass because the experiment never
# ran). SCOPE LIMIT, stated honestly: of the probe's six abstention paths
# this drills ONE (model build failed) — the only path reachable tonight
# without a patched probe or a falsified artifact, both forbidden. The five
# undrilled paths share this one's carrier byte-for-byte (same one-print
# shape, same exit 3, same laundering), and the carrier is the thing fix41
# exists to prove; what remains unproven is named in the fix41 reply, never
# silently assumed.
census_cuda_prefix=""
if [[ "${FS_CENSUS_DRILL_BUILD_FAILURE:-0}" == "1" ]]; then
  census_cuda_prefix="CUDA_VISIBLE_DEVICES= "
  census_drill_n=$(tr ',' '\n' <<<"$LORA_TARGETS" | grep -c . || true)
  echo "============================================================"
  echo "DRILL: FS_CENSUS_DRILL_BUILD_FAILURE=1 — re-creating the MEASURED census"
  echo "abstention on purpose: the census invocation below runs with"
  echo "CUDA_VISIBLE_DEVICES= (empty, assignment-scoped, NEVER exported — the"
  echo "launcher's other steps keep GPU visibility), so the real probe's model"
  echo "build must fail, the probe must print its OWN CENSUS_VERDICT=UNMEASURED"
  echo "(0 of $census_drill_n targets certified: model build failed), and this"
  echo "launcher MUST land on the verdict-keyed UNMEASURED arm and BLOCK."
  echo "A launch that PROCEEDS with this drill set is itself a stated failure,"
  echo "refused by name in the CLEAR arm below: a drill that cannot fire is a"
  echo "dead tripwire wearing a banner."
  echo "============================================================"
fi
# #78: one probe run, two artifacts — the preflight TEXT above and the JSON
# census at --out. The single quotes around '$ADAPTER_MODULES' belong to the
# CONTAINER bash exactly like '$LORA_TARGETS' (the fix39 idiom): the host
# expands the variable into the payload, the container receives one literal
# word. The drill prefix stays scoped ahead of torchrun, so an armed
# FS_CENSUS_DRILL_BUILD_FAILURE=1 still writes NOTHING — and what blocks
# that drill remains the verdict-keyed triage below, never the CLEAR-only
# artifact check (an UNMEASURED verdict must not be re-scored by it).
census_rc=0
run_in_container --slurm-ntasks 1 --workdir "$REPO" \
  bash -lc "cd $REPO && ${census_cuda_prefix}torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=$MASTER_PORT $CENSUS_PROBE --hf_model_path $HF_MODEL_PATH --ep $EP --targets '$LORA_TARGETS' --out '$ADAPTER_MODULES'" \
    >"$CENSUS_OUT" 2>&1 || census_rc=$?
# fix40 — VERDICT-LINE TRIAGE (deciding), rc (corroborating only).
# MEASURED 2026-08-24 on <compute-node> (the CUDA_VISIBLE_DEVICES="" run): the
# probe reached an abstention path, printed CENSUS_VERDICT=UNMEASURED,
# exited 3 — and the observed rc was 1. torchrun raises ChildFailedError
# for ANY nonzero child exit and its entrypoint maps that to exit 1; the
# child's own 0/1/3 is NOT propagated. Past this wrapper rc therefore
# carries exactly ONE BIT — zero vs nonzero — except rc=127, which names
# the wrapper/tool itself unreachable. A three-arm `case` keyed on that
# one bit left the UNMEASURED arm dead and reported every real abstention
# as a conviction, instructing the operator to re-spell strings the probe
# had never measured; that is tonight's defect. The probe prints exactly
# one CENSUS_VERDICT= line on EVERY exit path (six abstention paths +
# BLOCKED + CLEAR — the 8-prints/8-returns accounting is now contract-
# pinned in test_launcher_contracts.sh), and that line is measured to
# survive the wrapper intact, so the line decides and rc corroborates.
# Rule: a corroborated CLEAR (verdict CLEAR AND rc=0) proceeds — into the
# population/control/per-target re-verification below, which stays because
# the launch never rests on the probe honouring its own contract. Every
# other combination BLOCKS with a named message; every verdict/rc
# disagreement is a vocabulary breach whose fix targets the carrier
# (probe/wrapper), never the strings. A stated PREDICTION, unmeasured
# tonight: exporting RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
# MASTER_PORT=$MASTER_PORT would let plain python3 satisfy
# mp.initialize_model_parallel's env rendezvous, retiring the wrapper and
# restoring true child codes (0/2 required measurements exist: a direct
# --ep 0 forced-abstention must print UNMEASURED and exit 3; a direct
# CLEAR must reproduce the torchrun counts row-for-row). This triage is
# deliberately wrapper-independent so it stays correct under either
# executor; it is the belt that does not depend on that guess.
[[ -r "$CENSUS_OUT" ]] || \
  { echo "FATAL: LoRA target census output $CENSUS_OUT is missing or unreadable — no verdict line, population count, or control row can be read off an unreadable artifact, and an unreadable artifact BLOCKS; it never reads as an empty one (doctrines 1/4)." >&2; exit 1; }
# grep -c over the (now provably readable) file prints an honest 0 and
# exits 1 on no match; || true keeps that 0. The -r guard above is what
# makes the 0 evidence rather than a vacuous count.
census_verdicts_n=$(grep -c '^CENSUS_VERDICT=' "$CENSUS_OUT" || true)
if [[ "$census_verdicts_n" -ne 1 ]]; then
  cat "$CENSUS_OUT" >&2 || true
  if [[ "$census_verdicts_n" -eq 0 && "$census_rc" -eq 0 ]]; then
    echo "FATAL: census printed ZERO CENSUS_VERDICT lines yet rc=0 — DISAGREEMENT: rc=0 claims a CLEAR the contract says must be printed on every path. A census with no stated verdict BLOCKS (doctrine 4): inspect $CENSUS_OUT above (truncation? overwrite? a probe edit that dropped the print?). Do NOT re-spell the strings — nothing was measured." >&2
  elif [[ "$census_verdicts_n" -eq 0 ]]; then
    echo "FATAL: LoRA target census infrastructure failure (rc=$census_rc — ONE BIT past torchrun: nonzero says only 'the child failed somehow', with the child's own code laundered, so argparse's 2, a traceback's 1, and an OOM-kill are indistinguishable here; rc=127 instead names the wrapper itself). The probe died BEFORE printing any verdict: 0 of the shipped targets were certified. Do NOT re-spell LORA_TARGETS_BASE / LORA_TARGETS_EXPERT — none were measured tonight." >&2
  else
    echo "FATAL: $census_verdicts_n CENSUS_VERDICT lines in one census run — DISAGREEMENT: the probe renders exactly one per contract and the launcher truncates \$CENSUS_OUT each run, so a doubled verdict is a probe-side breach or a foreign writer. The file is unreadable as evidence — which run would each row belong to? Fail closed." >&2
  fi
  exit 1
fi
# Exactly one verdict line: extract the verdict WORD (first token after
# 'CENSUS_VERDICT='). sub() resplits fields, so $1 is the word even with
# the parenthesised payload the probe appends; an empty word lands in the
# drift arm below, never in a pass.
census_verdict=$(awk '/^CENSUS_VERDICT=/{sub(/^CENSUS_VERDICT=/, ""); print $1; exit}' "$CENSUS_OUT")
case "$census_verdict" in
  CLEAR)
    if [[ "$census_rc" -eq 0 ]]; then
      if [[ "${FS_CENSUS_DRILL_BUILD_FAILURE:-0}" == "1" ]]; then
        cat "$CENSUS_OUT" >&2 || true
        echo "FATAL: CENSUS DRILL ARMED BUT THE LAUNCH PROCEEDED — FS_CENSUS_DRILL_BUILD_FAILURE=1 was set for this launch, yet the census returned a corroborated CLEAR. Either the scoped CUDA_VISIBLE_DEVICES= prefix silently dropped out of the census invocation (the drill is vacuous — an armed no-op is precisely the unread-tripwire class fix41 exists to end), or the probe no longer fails to build with zero GPUs visible (the measured trigger drifted). Either way the UNMEASURED arm just demonstrably failed to fire on demand, so this run's green census is not evidence the tripwire is live. Fix the drill or the census — never delete this refusal to make a drill run launch." >&2
        exit 1
      fi
      # #78 CLEAR-ONLY artifact re-verification (doctrines 1/2/4). A
      # corroborated CLEAR whose machine-readable census is absent is the
      # claim-vs-disk gap this estate already measured once, and the verdict
      # triage above vouches for the probe's TEXT only — the launcher never
      # rests on the probe honouring its own contract (the same principle
      # the population/control greps below enforce on the text artifact), so
      # it re-verifies the JSON half itself. STRONGER THAN SPECIFIED, stated
      # plainly: mere presence is fakeable — an empty or garbage file is
      # PRESENT — so this arm also parses the file and requires a positive
      # module count, refusing otherwise. Scope, load-bearing: the check
      # sits INSIDE the CLEAR arm only, so an UNMEASURED or BLOCKED verdict
      # is never re-scored by it — the triage stays authoritative, and an
      # armed fix41 drill is already refused above, before this line.
      # MUST_FIRE (broken to see red): aim --out where the container cannot
      # write (delete the mkdir above, or point ADAPTER_MODULES outside the
      # bind-mounted tree) and every CLEAR census MUST die at the first
      # refusal below — a CLEAR that sails on with no readable JSON on disk
      # is the fix35-era 'verdict survives, evidence doesn't' shape this arm
      # exists to end. MUST_PASS (observed green on a healthy input): an
      # ordinary CLEAR launch in which THIS probe wrote its --out artifact
      # MUST pass the -r/parse/count gauntlet below and print the parsed
      # denominator off the artifact; a healthy CLEAR that dies here is
      # doctrine 5's symmetric defect — a false red costs what a false green
      # costs — and the arm gets repaired, never bypassed. Both halves ship
      # with this detector because a control observed only red is as
      # uncalibrated as one observed only green (doctrine 3).
      if [[ ! -r "$ADAPTER_MODULES" ]]; then
        cat "$CENSUS_OUT" >&2 || true
        echo "FATAL: census verdict CLEAR (rc=0 corroborated) yet ADAPTER_MODULES=$ADAPTER_MODULES is absent or unreadable — the probe's text claims CLEAR but the JSON census the gate will demand via --adapter-modules is not on disk where the launcher reads. Missing is not zero and unreadable is not empty (doctrine 4). Suspects, in order: the bind-mount precedent broke (the probe wrote where the host cannot see), the atomic write never committed, or a probe edit dropped --out. The launch BLOCKS — and do NOT hand-create the file: a census no probe wrote certifies nothing." >&2
        exit 1
      fi
      # Denominator, measured off the artifact (doctrine 2): the launcher
      # parses the JSON and counts what THIS file declares — 168 is tonight's
      # expectation, never a number the launcher restates from memory
      # (doctrine 5). The sibling --out contract is a flat name->record
      # mapping with the leading 'module.' stripped (or an array of such
      # names), so len(root) IS the declared count; if that shape ever grows
      # a metadata wrapper around the mapping, this count goes wrong LOUDLY
      # at the next CLEAR and the repair lands here, not in the gate. Host
      # python3 is safe for this read: it needs only stdlib json, so fix44's
      # torch-less-host hazard does not apply — no torch import exists on
      # this path.
      census_modules_n=$(python3 -c '
import json, sys
with open(sys.argv[1]) as fh:
    d = json.load(fh)
if isinstance(d, (dict, list)):
    print(len(d))
else:
    raise SystemExit("census root is %s, not an object/array of module records" % type(d).__name__)
' "$ADAPTER_MODULES" 2>&1) || {
        echo "FATAL: census verdict CLEAR but ADAPTER_MODULES=$ADAPTER_MODULES is not countable JSON — a census the launcher cannot parse states no denominator (doctrine 2), and the launcher does not rest on the probe honouring its own contract. python3's own words: $census_modules_n. The launch BLOCKS." >&2
        exit 1
      }
      # The 2>&1 capture above is double-duty: on failure the operator reads
      # python's own message, and on success stdout carries ONLY the bare
      # count — which the regex below enforces, so any stray print turns the
      # run red here instead of being narrated as a module count (doctrine 4
      # over a pretty story).
      [[ "$census_modules_n" =~ ^[1-9][0-9]*$ ]] || {
        echo "FATAL: census verdict CLEAR but ADAPTER_MODULES=$ADAPTER_MODULES declares ${census_modules_n:-<nothing countable>} modules — zero declared modules is UNMEASURED, never PASS (doctrine 1): with an empty census every downstream --adapter-modules demand is all([])==True. The launch BLOCKS." >&2
        exit 1
      }
      echo "Preflight census: $ADAPTER_MODULES declares $census_modules_n adapter modules (written-census denominator, parsed off the artifact itself — doctrine 2); this file is the --adapter-modules demand fs_live_save_gate() now receives on every gate call."
      : # the population/control greps and per-target arithmetic below still
        # re-verify the probe's TEXT artifact exactly as before — nothing
        # above replaces the verdict triage or those greps; it gates only the
        # machine-readable half the sibling --out patch added.
    else
      cat "$CENSUS_OUT" >&2 || true
      echo "FATAL: census verdict CLEAR but rc=$census_rc — DISAGREEMENT: past torchrun rc!=0 means the child genuinely exited nonzero, so a process printed CLEAR and then failed, or the wrapper failed around a completed child (this arm includes any rc outside {0,1}). The printed CLEAR and the exit status contradict; the census is untrustworthy — investigate the probe/wrapper, never 'rescue' this into a pass." >&2
      exit 1
    fi
    ;;
  BLOCKED)
    cat "$CENSUS_OUT" >&2 || true
    if [[ "$census_rc" -ne 0 ]]; then
      echo "FATAL: LoRA target census BLOCKED (verdict line; rc=$census_rc corroborates nonzero — the child's 1 arrives laundered but unanimous) — a shipped target string attaches ZERO modules under the REAL matcher." >&2
      echo "  The probe output above names the target(s) and the population denominator. Re-spell LORA_TARGETS_BASE / LORA_TARGETS_EXPERT. Do NOT relaunch on a red census, and do NOT 'repair' the strings against the old grep oracle — on 2026-08-24 grep scored the CORRECT spellings 0 and the broken ones 42." >&2
    else
      echo "FATAL: census verdict BLOCKED with rc=0 — DISAGREEMENT: BLOCKED must exit 1 by the shipped vocabulary, so the verdict carrier itself is broken. The rows above still name zero-attaching targets and stand as SUSPECTS, but a probe whose report and exit disagree convicts nothing and clears nothing: fix the probe's exit path and re-measure. The launch BLOCKS either way (doctrine 4)." >&2
    fi
    exit 1 ;;
  UNMEASURED)
    cat "$CENSUS_OUT" >&2 || true
    if [[ "$census_rc" -ne 0 ]]; then
      echo "FATAL: LoRA target census UNMEASURED (verdict line — the arm that was UNREACHABLE while keyed on rc: torchrun launders the child's 3 to rc=1, measured 2026-08-24, so this arm must never demand 3) — the probe ABSTAINED. Its own report above names which: controls failed, model build failed (a stated possibility at --ep>1 under this single-process invocation), empty target list, bad --ep, or matcher API drift." >&2
      if [[ "${FS_CENSUS_DRILL_BUILD_FAILURE:-0}" == "1" ]]; then
        # fix41 drill scoring — POSITIVE EVIDENCE, never rc-shape alone (the
        # fix41-§4 lesson: a leg that passes because the experiment never ran
        # is a vacuous pass, and so is a drill that takes credit for a fire
        # it did not cause). The drill succeeded iff the abstention this arm
        # just received NAMES the one path the drill forced. rc only
        # corroborates — the one bit torchrun is measured to preserve (if the
        # pending enroot exit-propagation measurement changes that map, the
        # re-scoping belongs to the same edit that lands the fact, per the
        # fix41 reply's conditional).
        drill_line=$(grep -m1 '^CENSUS_VERDICT=' "$CENSUS_OUT" || true)
        case "$drill_line" in
          *"model build failed"*)
            echo "DRILL FIRED: FS_CENSUS_DRILL_BUILD_FAILURE rehearsed the model-build-failed abstention end-to-end through the REAL probe, torchrun, the container executor, and this triage — the UNMEASURED arm fix40 made reachable is now PROVEN able to fire (1 of the probe's 6 abstention paths exercised through the full launcher stack; the 5 undrilled paths and their price are stated in the fix41 reply). Observed rc=$census_rc nonzero, as torchrun's measured laundering predicts." >&2
            ;;
          *)
            echo "DRILL ANOMALY: FS_CENSUS_DRILL_BUILD_FAILURE=1 is armed for the model-build-failed path, but the probe abstained on a DIFFERENT path (verdict line: '${drill_line:-<none>}'). The abstention still BLOCKS below, as every abstention must — but this run must NOT be recorded as the drill's MUST_FIRE: either the trigger drifted or the census has a second live abstention tonight. Investigate before re-drilling." >&2
            ;;
        esac
      fi
      echo "  0 of the shipped targets were certified. A stated abstention is a first-class outcome (doctrine 5) and it BLOCKS the launch (doctrine 4): repair the census mechanism; never trade an abstention for a guessed string, and never mint a divergence that was never observed. The strings measured at 42/42/42/42 stand — nothing tonight re-measured them in either direction." >&2
    else
      echo "FATAL: census verdict UNMEASURED with rc=0 — DISAGREEMENT: an abstaining probe exits 3 (-> 1 past torchrun); rc=0 breaches the vocabulary. The abstention still BLOCKS — no targets were certified — and what gets fixed is the probe/wrapper, never the strings, which remain unmeasured either way." >&2
    fi
    exit 1 ;;
  *)
    cat "$CENSUS_OUT" >&2 || true
    echo "FATAL: census verdict line states an unknown verdict '$census_verdict' (shipped vocabulary: CLEAR / BLOCKED / UNMEASURED) — DISAGREEMENT: probe and launcher have drifted, and this triage refuses to guess a classification. Fail closed (rc=$census_rc, one bit past the wrapper)." >&2
    exit 1 ;;
esac
# Denominator and control lines into the launch log BEFORE any per-target
# arithmetic (doctrines 2/3): a census whose population count or control
# record is missing is a census that did not happen, and the launch treats it
# that way.
grep '^CENSUS_POPULATION ' "$CENSUS_OUT" || { echo "FATAL: $CENSUS_OUT carries no CENSUS_POPULATION line — a census without a stated denominator BLOCKS; it never reads as a pass (doctrine 4)." >&2; exit 1; }
grep '^CENSUS_CONTROL ' "$CENSUS_OUT"    || { echo "FATAL: $CENSUS_OUT carries no CENSUS_CONTROL lines — an uncontrolled census is worth nothing and BLOCKS the launch (doctrine 3)." >&2; exit 1; }
# Per-target re-verification reads ONLY the real-matcher column (awk field 3).
# The grep column (field 4) is the pre-fix oracle kept as a DECLARED-SECONDARY
# signal for forensics: by construction it can neither pass nor fail this
# step. set -f: the repaired patterns legitimately contain '*', and even an
# absurdly-named host-cwd file must not glob-rewrite a launch decision.
set -f
census_bad=0
for t in ${LORA_TARGETS//,/ }; do
  real_n=$(awk -v t="$t" '$1=="CENSUS_TARGET" && $2==t {print $3; exit}' "$CENSUS_OUT")
  grep_n=$(awk -v t="$t" '$1=="CENSUS_TARGET" && $2==t {print $4; exit}' "$CENSUS_OUT")
  if [[ -z "$real_n" ]]; then
    echo "FATAL: no CENSUS_TARGET row for shipped target '$t' in $CENSUS_OUT — what is launched must be exactly what was censused; a missing row BLOCKS (doctrine 4)." >&2
    census_bad=1
    continue
  fi
  note=""
  if [[ "$real_n" -eq 0 && "$grep_n" -gt 0 ]] || [[ "$real_n" -gt 0 && "$grep_n" -eq 0 ]]; then
    note="  *** ORACLE DIVERGENCE on this target — the weaker substring oracle and ModuleMatcher disagree on the live tree; the matcher count is the only one this launch reads (fix39 evidence signature) ***"
  fi
  echo "  target '$t' -> real-matcher=$real_n, grep-oracle=$grep_n (secondary, non-deciding)$note"
  # Unreachable while the probe honours CLEAR==all-attach; kept because the
  # launch decision must not rest on that honour.
  if [[ "$real_n" -eq 0 ]]; then
    echo "FATAL: target '$t' attaches 0 modules under the real matcher despite a CLEAR probe rc — the census file and the probe disagree; fail closed." >&2
    census_bad=1
  fi
done
set +f
[[ "$census_bad" -eq 0 ]] || exit 1

# (6) EXTRAS-shadowing tripwire (README trap 1): torch/transformer_engine must import coherently
if ! run_in_container --slurm-ntasks 1 \
    bash -lc "python3 -c \"import torch, transformer_engine.pytorch as te; print('torch', torch.__version__, '| TE OK')\"" \
      > "$PREFLIGHT_DIR/env_probe.txt" 2>&1; then
  echo "FATAL: torch/TE import broken — check \$EXTRAS for pip --target debris (README trap 1)" >&2; exit 1
fi
cat "$PREFLIGHT_DIR/env_probe.txt"

# ----------------------------------------------------------------------------
# Provenance: RunManifest WITHOUT a declared block — an abstention, stated
# ----------------------------------------------------------------------------
# The manifest itself (code provenance, FOXBRAIN_-prefixed env slice,
# topology, effective knobs, fingerprint) is now mandatory: it is the record
# the 24-run-split audit needed and could not produce. The `declared` block
# is deliberately NOT emitted for LoRA: the FQN set of an adapter save is a
# property of the peft implementation inside the training repository — not
# importable on a login node, not verifiable on this estate today — and the
# base ckpt census is the WRONG denominator for adapter tensors. Emitting
# r×targets×suffix strings from our own CLI would fabricate a denominator
# from the very strings this launcher could misconfigure. checkpoint.save_
# complete keeps its honest VACUOUS on LoRA runs; an honest VACUOUS beats a
# false PASS. Emission failure is still fatal: a run whose provenance cannot
# even ABSTAIN in writing is not a run we may launch.

# OPT_LR_KEY is defined HERE, not down with the other recipe overrides, because
# the manifest emission below references it. `set -euo pipefail` (line 40) makes
# a definition 30 lines further down not "later" but an unbound-variable ABORT —
# and the abort landed exactly ON this provenance gate, so the LoRA launcher
# could never reach a GPU. Keep this above its first use.
OPT_LR_KEY=${OPT_LR_KEY:-optimizer.lr}   # UNVERIFIED: bridge container may spell this 'optim.lr'.
                                         # Check once: python3 -c 'from megatron.bridge.training.config import ConfigContainer as C; print(sorted(C.__dataclass_fields__))'

# FS_ROOT: search the known deploy locations instead of asserting one. The old
# default ($WORKSPACE/FoundationScale) does not exist on this estate — the
# checkout is at $HOME/foundationscale (measured 2026-08-23) — so the FATAL below
# fired on every launch. Candidates are tried in order and the failure message
# names ALL of them, so an operator is told what was searched rather than handed
# a single wrong path to puzzle over.
FS_CANDIDATES=("${FOUNDATIONSCALE_ROOT:-}" "$HOME/foundationscale" "$WORKSPACE/FoundationScale")
FS_ROOT=""
for _fs_c in "${FS_CANDIDATES[@]}"; do
  [[ -n "$_fs_c" && -f "$_fs_c/tools/emit_run_manifest.py" ]] || continue
  FS_ROOT=$_fs_c; break
done
[[ -n "$FS_ROOT" ]] || \
  { printf 'FATAL: tools/emit_run_manifest.py not found under any of: %s\n' "${FS_CANDIDATES[*]}" >&2; \
    echo "Set FOUNDATIONSCALE_ROOT; launching without recorded provenance is not an option." >&2; exit 1; }
echo "provenance gate: FoundationScale checkout resolved to $FS_ROOT"
# fix35: the adjudicator ships in the same tools/ directory as the emitter;
# one-present-one-absent means a partial checkout. Even though this run's
# lora adjudication ABSTAINS by design today (see the post-run section), the
# abstention must be delivered by the tool itself on the record — a missing
# tool cannot even abstain, and an unrecorded abstention is not one.
[[ -f "$FS_ROOT/tools/live_save_gate.py" ]] || \
  { echo "FATAL: tools/live_save_gate.py not found under $FS_ROOT/tools — the manifest emitter beside it resolved, so this checkout is PARTIAL. Fix the checkout or set FOUNDATIONSCALE_ROOT." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || \
  { echo "FATAL: host python3 unavailable — cannot emit the run manifest; provenance emission is not optional." >&2; exit 1; }
PYTHONPATH="$FS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 "$FS_ROOT/tools/emit_run_manifest.py" \
  --run-id "${RECIPE}_${RUN_TAG}${RUN_SUFFIX}" \
  --out-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
  --job-id "${SLURM_JOB_ID}" \
  --nodes 1 --gpus-per-node 4 --tp $TP --pp 1 --cp $CP --dp $DP --ep $EP \
  --code-root "$REPO" --entrypoint "$REPO/scripts/training/run_recipe.py" \
  --env-prefix FOXBRAIN_ \
  --watch-env FOXBRAIN_GEMMA4_KEEP_COT --watch-env FOXBRAIN_SFT_JSONLS \
  --effective recipe=$RECIPE --effective peft.dim=$LORA_RANK --effective peft.alpha=$LORA_ALPHA \
  --effective "$OPT_LR_KEY=$LORA_LR" --effective train.train_iters=$TRAIN_ITERS \
  --effective train.global_batch_size=$GLOBAL_BATCH_SIZE --effective model.seq_length=$SEQ_LENGTH \
  --effective checkpoint.save_interval=$SAVE_INTERVAL \
  --lora \
  || { echo "FATAL: run-manifest emission FAILED — provenance could not be recorded even with its stated LoRA abstention; refusing to launch." >&2; exit 1; }
echo "provenance gate: run manifest emitted (declared block: abstained — LoRA, see emitter output above)"

# ----------------------------------------------------------------------------
# fix35 — produce the gate's --train-config (fs_gate/resolved-train-config.json).
# Keys are chosen from the vocabulary tools/live_save_gate.py actually reads
# (_KIND_KEYS/_RANK_KEYS/_TARGET_KEYS): today the lora path refuses on the
# unpinned adapter prefix BEFORE any derivation, but the config must already
# exist and be truthful the day that prefix is measured (the gate's own
# measurement recipe in its --adapter-prefix help); a producer that ships
# later than its consumer is how this estate got an unreachable adjudicator.
# No --fqn-map here BY DESIGN: for lora runs the gate explicitly IGNORES the
# map flag (the adapter declared set derives from base header x targets x
# rank), and this launcher emits no declared block to read one back from.
# ----------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR/fs_gate" || { echo "FATAL: cannot create $OUTPUT_DIR/fs_gate" >&2; exit 1; }
RESOLVED_CFG=$OUTPUT_DIR/fs_gate/resolved-train-config.json
python3 - "$RESOLVED_CFG" "$PEFT_SCHEME" "$RECIPE" "$LORA_RANK" "$LORA_ALPHA" "$LORA_DROPOUT" "$LORA_LR" "$LORA_TARGETS" "$TRAIN_ITERS" "$GLOBAL_BATCH_SIZE" "$SEQ_LENGTH" "$SAVE_INTERVAL" <<'PY' || { echo "FATAL: resolved-train-config write failed — the gate would run on a tolerated absence; refusing (doctrine 4)" >&2; exit 1; }
import json, sys
(out, scheme, recipe, rank, alpha, drop, lr, targets, iters, gbs, seq,
 save_iv) = sys.argv[1:13]
doc = {
    "peft_scheme": scheme,
    "run_kind": "lora",
    "recipe": recipe,
    "lora_rank": int(rank),
    "lora_alpha": int(alpha),
    "lora_dropout": float(drop),
    "lora_lr": lr,
    "lora_targets": targets,
    "train.train_iters": int(iters),
    "train.global_batch_size": int(gbs),
    "model.seq_length": int(seq),
    "checkpoint.save_interval": int(save_iv),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
print(f"resolved train config: {out} ({len(doc)} keys incl. peft_scheme/run_kind/lora_rank/lora_targets — the gate's own key vocabulary)")
PY

# ----------------------------------------------------------------------------
# Auto-resume state machine (estate-proven semantics): fresh / resume / done.
# ----------------------------------------------------------------------------
CKPT_DIR=$OUTPUT_DIR/checkpoints
LATEST_FILE=$CKPT_DIR/latest_checkpointed_iteration.txt
if [[ -f "$LATEST_FILE" ]]; then
  LAST=$(tr -dc '0-9' < "$LATEST_FILE"); LAST=${LAST:-0}
  if (( LAST >= TRAIN_ITERS )); then
    echo "Already at train_iters ($LAST >= $TRAIN_ITERS) — chain no-op, exit 0."; exit 0
  fi
  echo "Resuming from iter $LAST (optimizer + RNG loaded)."
  LOAD_STATE="checkpoint.load_optim=True checkpoint.load_rng=True"
else
  if [[ -d "$OUTPUT_DIR/tb_logs" ]] && compgen -G "$OUTPUT_DIR/tb_logs/*" >/dev/null; then
    echo "FATAL: $OUTPUT_DIR has run artifacts but NO checkpoint — refusing to" >&2
    echo "silently restart from scratch. Investigate or rm -rf the dir deliberately." >&2
    exit 1
  fi
  echo "Fresh run from pretrained checkpoint."
  LOAD_STATE="checkpoint.load_optim=False checkpoint.load_rng=False"
fi

# ----------------------------------------------------------------------------
# Recipe CLI overrides
# ----------------------------------------------------------------------------
# OPT_LR_KEY is set above, before the provenance gate that consumes it.
CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$MEGATRON_CKPT \
    $LOAD_STATE \
    checkpoint.load=$CKPT_DIR \
    checkpoint.save=$CKPT_DIR \
    checkpoint.save_interval=$SAVE_INTERVAL \
    logger.tensorboard_dir=$OUTPUT_DIR/tb_logs \
    model.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.context_parallel_size=$CP \
    model.sequence_parallel=False \
    $MOE_OVERRIDES \
    peft.dim=$LORA_RANK \
    peft.alpha=$LORA_ALPHA \
    peft.dropout=$LORA_DROPOUT \
    peft.target_modules=[$LORA_TARGETS] \
    $OPT_LR_KEY=$LORA_LR \
    scheduler.lr_decay_iters=$TRAIN_ITERS \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    dataset.trust_remote_code=True \
    dataset.pack_sequences_in_batch=False \
    dataset.num_workers=${NUM_WORKERS:-8} \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${RECIPE}_${RUN_TAG}"
# TWO KNOB GROUPS ARE ABSENT FROM THE OVERRIDE LINE ABOVE, DELIBERATELY (fix28):
#  - recompute: A2 mechanism — the E4B PLE slice is stashed per decoder layer at
#    modeling_gemma4_e4b_vl.py:82 and cleared to None by the finally at :93
#    BEFORE backward, so ANY recompute (selective/core_attn included) re-runs
#    the layer forward during backward with _ple_input=None and the PLE
#    parameters get NO gradient. The E4B recipe pins granularity/method/
#    num_layers=None and hard-asserts off; restating any recompute override here
#    would re-arm the defect. The old trap-5 reason for preferring core_attn
#    alone (the fuller list diverged LoRA at ~iter 190, peft/recompute.py:71) is
#    moot — NO recompute is legal on this model.
#  - MoE transport/parallelism: comes from $MOE_OVERRIDES, which is EMPTY by
#    design on the measured-dense base (port of the full-FT launcher's branch);
#    on a measured-MoE base it carries exactly the four knobs this launcher
#    used to send unconditionally.
# num_workers=8 — the 26B recipe shipped 2 (plan bug #4). lr_decay_iters=
# train_iters so the cosine COMPLETES (trap 2).

# ONE training-command definition, TWO executors: LAUNCH_PY is the only
# backend-dependent word. sbatch -> python3 (the allocation's
# --ntasks-per-node=4 supplies four ranks, one python3 each). enroot ->
# torchrun forks the same 4 ranks inside the ONE enroot start. Everything
# after the launcher word is backend-invariant by construction.
LAUNCH_PY=$(fs_launch_python "$GPUS_PER_NODE")

CMD="cd $REPO && \
    $LAUNCH_PY scripts/training/run_recipe.py \
        --recipe $RECIPE \
        --peft_scheme $PEFT_SCHEME \
        --hf_path $HF_MODEL_PATH \
        --step_func $STEP_FUNC \
        $CLI_OVERRIDES"

# The resolved command is printed in full so the log the gates parse (and the
# humans reading it later) record WHICH executor ran — the backend library
# exports the whole job environment into the container, a superset of the
# hand-maintained --export list the srun shape carried.
echo "Training command ($FS_BACKEND): $CMD"

# ----------------------------------------------------------------------------
# (7) OVERRIDE-REPLAY — prove the composed config carries the SHIPPED peft
#     knobs before any GPU time is burned (fix42 / #73).
#     The founding failure of this fix was silent in the config layer: the
#     recipe set cfg.peft, the override mechanism dropped it, and 4 ranks
#     learned that only inside torchrun. The repair makes peft.* addressable,
#     but "the overrides work" is a claim with no denominator unless the run
#     produces positive evidence that the SHIPPED values are what composition
#     resolved. This step drives the REAL process_config_with_overrides over
#     the BYTE-IDENTICAL token stream the training command receives — the
#     $CLI_OVERRIDES expansion below is deliberately unquoted, so the replay
#     sees exactly the tokens training will see, glob risk and all; the single
#     quotes around '$LORA_TARGETS' belong to the CONTAINER bash (the census
#     invocation's measured idiom) so repaired '*' patterns cannot glob-
#     rewrite the expectation. A full-set replay also surfaces every other
#     composition defect pre-GPU with a loud, named BLOCK instead of a
#     4-rank crash — tonight's crash aborted at peft.dim, so every override
#     after it (optimizer.lr's UNVERIFIED spelling included) was never once
#     exercised until this step existed. What this step deliberately does NOT
#     prove: that the model builder reads dim/alpha/dropout (that bridge is
#     the drill's G3 trainable-params comparison and G2's attach lines — the
#     replay's evidence ends at composition, and the verdict line says so).
#     Invocation shape (torchrun single-process rendezvous, MASTER_PORT
#     reuse-while-free, container path search order, -r guard, verdict-count
#     check) mirrors the step-(5) census byte-for-byte for the same measured
#     reason: torchrun launders ANY nonzero child exit to 1 (fix40 receipt),
#     so rc past the wrapper is exactly one bit and the printed REPLAY_VERDICT=
#     line is the only signal that decides; rc corroborates. Vocabulary is the
#     shipped one, unrenumbered: 0 CLEAR, 1 BLOCKED, 3 UNMEASURED.
# ----------------------------------------------------------------------------
REPLAY_PROBE=""
for _rp in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/peft_override_replay.py" \
           "$WORKSPACE/g4_sft/peft_override_replay.py"; do
  [[ -f "$_rp" ]] && { REPLAY_PROBE="$_rp"; break; }
done
[[ -n "$REPLAY_PROBE" ]] || REPLAY_PROBE=$(find "$REPO" -name peft_override_replay.py -print -quit 2>/dev/null)
[[ -n "$REPLAY_PROBE" ]] || \
  { echo "FATAL: peft_override_replay.py not found. Searched: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ (shipped beside the launcher, fix42), $WORKSPACE/g4_sft/ (container-visible precedent), then all of $REPO. Launching with peft.* overrides that nothing has replayed is exactly how #73 reached torchrun — fail closed (doctrine 4)." >&2; exit 1; }
echo "Preflight: peft override replay via $REPLAY_PROBE (oracle = the real process_config_with_overrides over the byte-identical \$CLI_OVERRIDES; verdict line decides, rc — one bit past torchrun — corroborates)"
REPLAY_OUT=$PREFLIGHT_DIR/override_replay.txt
replay_rc=0
run_in_container --slurm-ntasks 1 --workdir "$REPO" \
  bash -lc "cd $REPO && torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=$MASTER_PORT $REPLAY_PROBE --recipe $RECIPE --peft_scheme $PEFT_SCHEME --hf_path $HF_MODEL_PATH --seq_length $SEQ_LENGTH --expect-dim $LORA_RANK --expect-alpha $LORA_ALPHA --expect-dropout $LORA_DROPOUT --expect-targets-csv '$LORA_TARGETS' --overrides $CLI_OVERRIDES" \
    >"$REPLAY_OUT" 2>&1 || replay_rc=$?
[[ -r "$REPLAY_OUT" ]] || \
  { echo "FATAL: override-replay output $REPLAY_OUT is missing or unreadable — no verdict line or evidence row can be read off an unreadable artifact, and an unreadable artifact BLOCKS; it never reads as an empty one (doctrines 1/4)." >&2; exit 1; }
# fix43 — TOKEN NAMESPACE IS CONTRACT. The six verdict/rc disagreement arms
# below carry the token CONTRADICTION; the step-(5) census triage's six keep
# DISAGREEMENT. fix42 shipped its CLEAR-vs-rc arm with the census token, so
# one grep tally silently spanned two triages (measured: the fix40 census
# legs read 7, with a deleted census arm compensated by an added replay arm
# invisible inside the merged count). grep matches substrings and \b is not
# portable to the BSD grep the contract suite runs under on macOS, which is
# why the replay token shares NO substring with the census token: the two
# populations are censused separately in test_launcher_contracts.sh, each
# with its own drop-one MUST_FIRE. The token lives in CODE (echo lines),
# never in these comments — the suite reads the comment-stripped view.
replay_verdicts_n=$(grep -c '^REPLAY_VERDICT=' "$REPLAY_OUT" || true)
if [[ "$replay_verdicts_n" -ne 1 ]]; then
  cat "$REPLAY_OUT" >&2 || true
  # Splintered to parity with the census triage (fix43): silent-with-rc0 and
  # doubled-verdict are carrier breaches and are NAMED as such; the plain
  # nonzero-rc death-before-verdict stays the infrastructure arm and
  # deliberately carries no disagreement token, exactly as the census's
  # plain infrastructure arm does not.
  if [[ "$replay_verdicts_n" -eq 0 && "$replay_rc" -eq 0 ]]; then
    echo "FATAL: replay printed ZERO REPLAY_VERDICT lines yet rc=0 — CONTRADICTION: rc=0 claims a CLEAR the probe contract says must be printed on every exit path. A verdict-less replay measured 0 of the 4 shipped peft knobs; BLOCK and inspect the probe/wrapper, never the strings (truncation? overwrite? a probe edit that dropped the print?)." >&2
  elif [[ "$replay_verdicts_n" -eq 0 ]]; then
    echo "FATAL: peft override replay infrastructure failure (rc=$replay_rc — ONE BIT past torchrun: nonzero says only 'the child failed somehow'; the child's own code is laundered, and rc=127 would name the wrapper itself unreachable). The probe died BEFORE printing any verdict: 0 of the 4 shipped peft knobs were verified. Repair the replay mechanism; do NOT relaunch on it." >&2
  else
    echo "FATAL: $replay_verdicts_n REPLAY_VERDICT lines in one replay run — CONTRADICTION: the probe renders exactly one per contract and the launcher truncates \$REPLAY_OUT each run, so a doubled verdict is a probe-side breach or a foreign writer. The file is unreadable as evidence — which run would each evidence row belong to? Fail closed." >&2
  fi
  exit 1
fi
replay_verdict=$(awk '/^REPLAY_VERDICT=/{sub(/^REPLAY_VERDICT=/, ""); print $1; exit}' "$REPLAY_OUT")
# Word extraction, not field 2 of an unbounded '='-split (fix43, measured by
# the contract suite's MUST_PASS leg going red on a healthy fixture): the
# discriminating line's payload contains further '=' bytes ('...dim=32
# alpha=64...'), so the old awk -F= '{print $2}' yielded '1 (pre-composition
# dim' — never equal to "0" — and the drill-hint note in the CLEAR arm below
# was unreachable dead code. sub() resplits fields, so $1 is the word,
# whatever parenthesised payload follows; the census_verdict idiom one step
# up, reused verbatim.
replay_discrim=$(awk '/^REPLAY_KNOB_DISCRIMINATING=/{sub(/^REPLAY_KNOB_DISCRIMINATING=/, ""); print $1; exit}' "$REPLAY_OUT")
case "$replay_verdict" in
  CLEAR)
    if [[ "$replay_rc" -ne 0 ]]; then
      cat "$REPLAY_OUT" >&2 || true
      echo "FATAL: replay verdict CLEAR but rc=$replay_rc — CONTRADICTION: past torchrun rc!=0 means the child genuinely exited nonzero, so a process printed CLEAR and then failed, or the wrapper failed around a completed child (this arm includes any rc outside {0,1}). The printed CLEAR and the exit status disagree; the composition evidence is untrustworthy — investigate the probe/wrapper, never rescue this into a pass." >&2
      exit 1
    fi
    grep '^REPLAY_PEFT ' "$REPLAY_OUT" || { echo "FATAL: CLEAR without a REPLAY_PEFT evidence line — a verdict with no stated knobs_checked BLOCKS (doctrine 2): the claim 'the overrides resolve' is meaningless without the values it examined." >&2; exit 1; }
    if [[ -n "${FS_PEFT_DRILL_RANK:-}" ]]; then
      if grep -q "^REPLAY_PEFT dim=$FS_PEFT_DRILL_RANK " "$REPLAY_OUT"; then
        echo "DRILL FIRED: FS_PEFT_DRILL_RANK=$FS_PEFT_DRILL_RANK resolved through the REAL composition path while both defaults sit at 32 — the peft.* override channel is proven live on a run where landed-vs-default would otherwise be indistinguishable (1 of 4 knobs discriminated; the other 3 share this carrier, scope stated at the drill arming)."
      else
        cat "$REPLAY_OUT" >&2 || true
        echo "FATAL: KNOB DRILL ARMED BUT RESOLVED dim != $FS_PEFT_DRILL_RANK — the replay CLEARed while reporting a value the drill did not ship (the default 32 included). That is the silent-revert signature fix42 exists to end: the override was accepted and then discarded, or never applied. This run's CLEAR is therefore not evidence the knob path is live. Fix composition; never delete this refusal to make a drill run launch." >&2
        exit 1
      fi
    elif [[ "$replay_discrim" == "0" ]]; then
      echo "replay note: shipped knobs EQUAL the recipe defaults (4 of 4), so tonight's CLEAR cannot by itself distinguish 'override landed' from 'default sat there' — run FS_PEFT_DRILL_RANK=96 PROBE=1 once to discriminate; the drill's result is recorded in this same preflight artifact."
    fi
    ;;
  BLOCKED)
    cat "$REPLAY_OUT" >&2 || true
    if [[ "$replay_rc" -ne 0 ]]; then
      echo "FATAL: override replay BLOCKED (verdict line; rc=$replay_rc corroborates nonzero past the wrapper) — a NAMED override failed composition or resolved to a value other than the one shipped. The probe output above names field, shipped value, resolved value." >&2
      if [[ -n "${FS_PEFT_DRILL_RANK:-}" ]]; then
        echo "FATAL-AND-DRILL-FIRED: with FS_PEFT_DRILL_RANK=$FS_PEFT_DRILL_RANK armed, this BLOCK is the tripwire doing its job — the drill's perturbation did NOT survive composition, which is exactly the defect class the drill exists to detect. Do not 'rescue' it." >&2
      fi
    else
      echo "FATAL: replay verdict BLOCKED with rc=0 — CONTRADICTION: BLOCKED must exit 1 by the shipped vocabulary, so the verdict carrier itself is broken. The mismatch rows above stand as SUSPECTS, but a probe whose report and exit disagree convicts nothing and clears nothing: fix the probe's exit path and re-measure. The launch BLOCKS either way (doctrine 4)." >&2
    fi
    exit 1 ;;
  UNMEASURED)
    cat "$REPLAY_OUT" >&2 || true
    if [[ "$replay_rc" -ne 0 ]]; then
      echo "FATAL: override replay UNMEASURED (verdict line) — the probe ABSTAINED: its report above names which of imports / recipe registration / recipe construction / empty override list failed. 0 of the 4 shipped knobs were verified, and a stated abstention BLOCKS (doctrines 4/5); repair the replay mechanism, never bypass it." >&2
    else
      echo "FATAL: replay verdict UNMEASURED with rc=0 — CONTRADICTION: an abstaining probe exits 3 (-> 1 past torchrun); rc=0 breaches the vocabulary. The abstention still BLOCKS — no knobs were verified — and what gets fixed is the probe/wrapper, never the overrides, which remain unreplayed either way." >&2
    fi
    exit 1 ;;
  *)
    cat "$REPLAY_OUT" >&2 || true
    echo "FATAL: replay verdict line states an unknown verdict '$replay_verdict' (shipped vocabulary: CLEAR / BLOCKED / UNMEASURED) — CONTRADICTION: probe and launcher have drifted and this triage refuses to guess a classification. Fail closed (rc=$replay_rc, one bit past the wrapper)." >&2
    exit 1 ;;
esac

RC=0
run_in_container --workdir "$REPO" bash -lc "$CMD" || RC=$?

# ----------------------------------------------------------------------------
# POST-RUN GATES — parse OUR OWN log; any failure marks the job failed.
# Positive control: adapters attached AND trainable AND base frozen.
# fix35 addition, stated rather than implied: this launcher has NO live
# watcher (training is synchronous above), so unlike the full-FT sibling —
# which kills a doomed run at its first save via tripwire (d) — every
# adjudication here is POST-RUN ONLY, including the live_save_gate artifact
# verdict appended below (G-artifact). G1-G5 remain this run's adjudication
# of record while the lora gate abstains on the unpinned adapter prefix
# (exit 3 by design; see the mapping next to gate_fail). The full-FT
# launcher states this same asymmetry in its own watcher comment.
# ----------------------------------------------------------------------------
OUTLOG="${SLURM_SUBMIT_DIR:-$PWD}/logs/g4e4b_lora_taiwan_${SLURM_JOB_ID}.out"
GATE=0
gate_fail(){ echo "GATE FAIL: $1" >&2; GATE=1; }

# ---------------------------------------------------------------------------
# fix35 — the artifact adjudicator, wired POST-RUN ONLY (asymmetry, stated).
# The sibling full-FT launcher backgrounds training behind a watcher and runs
# this same gate LIVE at first save, killing a doomed run at (d); this
# launcher trains synchronously with no watcher, so nothing here can stop a
# run mid-flight, and pretending otherwise in either file is how siblings
# diverge silently. Today the lora path of tools/live_save_gate.py exits 3
# BY DESIGN (the adapter-prefix refusal; measurement recipe in its
# --adapter-prefix help — pinning '' unmeasured would assert a save layout
# never observed, the doctrine-5 symmetric defect), so the mapping below
# keeps exactly ONE calibrated member of the exit-3 class: the prefix
# abstention lands rc 0 ONLY when the gate's own record names it, while 1
# (measured BLOCK), every OTHER member of exit 3, and any off-contract rc
# (wiring bug/crash) stop the job. When the prefix is measured and pinned in
# fs_live_save_gate, delete the prefix arm's "expected" phrasing in the same
# edit.
# fix44 / #77-B1 addendum: until this fix the wrapper invoked HOST python3 —
# the one python call site in this launcher that did not go through the
# executor. Host python3 has no torch (fix32 measurement: the host anaconda
# interpreter shadows the image stack), so both PROBE runs' exit 3s were a
# torch-less host reading a HEALTHY DCP at live_save_gate.py:1149 — never the
# adapter-prefix abstention the log claimed. The gate now runs through
# run_in_container like the four sites the contract suite already pins.
# ---------------------------------------------------------------------------
fs_live_save_gate() { # $1=iter dir  $2=event  $3=report path  $4=capture log; returns the gate's rc UNTOUCHED.
  # #78 — the wiring this wrapper used to deliberately OMIT. The comment
  # this replaces read "NO --adapter-prefix: deliberately unpinned today",
  # under which a gate exit 3 with refusal_class adapter_prefix_unpinned was
  # the calibrated rc-0 'expected lora state' in the mapper below. Both
  # flags are now pinned ON EVERY CALL, and the calibrated arm is retired in
  # the SAME edit — one edit, not two — because the gate raises its prefix
  # demand BEFORE its census demand (live_save_gate.py:505-531, order
  # load-bearing; the :511-523 note licenses exactly this COORDINATED
  # change), so wiring --adapter-modules while leaving the prefix arm in
  # place would have changed NOTHING: the prefix refusal would still
  # short-circuit first and the run would still abstain. --adapter-prefix ''
  # is a PIN, not an omission: the decision is 'no prefix' — the census
  # strips the leading 'module.', so adapter names attach to base-model
  # module names bare. The '' rides the same host->container quote layering
  # as the census probe's '$LORA_TARGETS' (fix39): the outer double quotes
  # are the host's, the container bash sees a literal --adapter-prefix '',
  # and argv carries a genuinely EMPTY word — neither a missing flag (the
  # gate's demand distinguishes an empty pin from None) nor the two-
  # character string "''". --adapter-modules hands the gate the preflight
  # probe's --out artifact, written OUTSIDE the judged iter_* tree on
  # purpose, because live_save_gate.py:742's tautology guard refuses any
  # census inside it.
  # Executor-routed (fix44 / #77-B1), invocation shape copied from the two
  # measured precedents: --slurm-ntasks 1 (a single-CPU adjudicator, like the
  # env probe and the census/replay probes), --workdir "$REPO" (the replay
  # probe's idiom; inert on the enroot arm). Inside the payload the
  # established PYTHONPATH idiom is honored and EXTENDED, never replaced: the
  # forwarded PYTHONPATH (EXTRAS:REPO/src:REPO/3rdparty/Megatron-LM) keeps
  # its order and FS_ROOT/src is prepended, exactly the layering the retired
  # host call had. The single quotes belong to the CONTAINER bash (the
  # measured census idiom): $FS_ROOT, $HF_MODEL_PATH, $RESOLVED_CFG and the
  # args expand on the HOST into the payload string — every FS_ROOT
  # candidate ($FOUNDATIONSCALE_ROOT or $HOME/foundationscale or
  # $WORKSPACE/FoundationScale, with $WORKSPACE itself under $HOME) lives
  # under $HOME, and both executor arms bind-mount $HOME:$HOME, so the
  # byte-identical paths are visible in-container — while ${PYTHONPATH} is
  # backslash-deferred so the CONTAINER expands its own forwarded value. If a
  # future FOUNDATIONSCALE_ROOT ever points outside the mounted $HOME tree,
  # python3 exits 2 ('can't open file') and the rc-92 arm blocks the launch:
  # fail closed, never a silent host fallback. PYTHONNOUSERSITE=1 is already
  # exported and forwarded (s7); restating it scoped to this payload keeps
  # the load-bearing property true even if a future edit drops the export.
  # rc hygiene: the output capture is a REDIRECT, not a pipe — there is no
  # process between the gate and its exit status (the #72 torchrun laundering
  # class is structurally absent here), '|| fs_gate_rc=$?' preserves the
  # executor's rc bit-exact (the fix41 measurement: run_in_container does not
  # launder), and the bare 'return' hands it on UNTOUCHED: the 0/1/3/other
  # contract the mapper below decodes is the gate's own, never a re-score.
  # The capture file is not bookkeeping: it is the only place the gate's own
  # words survive, and the mapper reads the exit-3 CAUSE from it (#77-B2).
  local fs_gate_rc=0
  run_in_container --slurm-ntasks 1 --workdir "$REPO" \
    bash -lc "PYTHONPATH='$FS_ROOT/src':\${PYTHONPATH} PYTHONNOUSERSITE=1 python3 '$FS_ROOT/tools/live_save_gate.py' '$1' --event '$2' --run-kind lora --base-model-dir '$HF_MODEL_PATH' --train-config '$RESOLVED_CFG' --adapter-modules '$ADAPTER_MODULES' --adapter-prefix '' --json '$3'" \
    >"$4" 2>&1 || fs_gate_rc=$?
  return "$fs_gate_rc"
}
fs_lora_gate_verdict_to_rc() { # $1=gate rc  $2=which  $3=report path  $4=capture log -> echoes; sets FS_ART_GATE_STATE + FS_ART_GATE_RC
  # fix44 / #77-B2: gate exit 3 is a CLASS of refusals, not a cause (the
  # tool's own :79 docstring: unreadable artifact, missing base files,
  # unresolvable run mode, tool bug, OR the deliberately unpinned adapter
  # prefix). The pre-fix44 arm narrated ONE member — the prefix abstention —
  # for every 3, defended as 'stated every run so it cannot fade into a
  # habit'. Taken seriously, that argument is an argument for stating the
  # OBSERVED cause every run, not one member of the class: a permanent,
  # confident, WRONG explanation does not fight habituation, it
  # industrializes it — both PROBE runs were told 'adapter-prefix unpinned'
  # while the gate had actually died at :1149 on a torch-less HOST python
  # (fix44 / #77-B1) reading a healthy DCP. The cause is now read from the
  # gate's own evidence, never assumed: the refusal_class field of the
  # on-disk refusal report first (written by the tool at the point of
  # refusal — the single source of truth, fix44 / #77-B3), the captured
  # stderr line as corroboration and as the ONLY signal available when the
  # record itself is missing. Calibration, RE-DECIDED (#78, this same edit):
  # NO member of the exit-3 class is an rc-0 abstention any longer. The
  # comment this replaces calibrated exactly ONE member — the adapter-prefix
  # refusal — to rc 0, because WE chose not to pin the prefix and the gate
  # correctly declined a job it was never configured to do. That choice
  # ended when --adapter-prefix '' and --adapter-modules were wired in this
  # same edit (fs_live_save_gate above): a chosen abstention cannot survive
  # the choice being made, and leaving the rc-0 arm armed after the wiring
  # landed would have silently kept the abstention alive (the byte-for-byte
  # hazard live_save_gate.py:511-523 warns about). Every member of the class
  # now rides the rc-92 infrastructure class: unreadable checkpoint, missing
  # base files, tool crash, unreadable capture, missing record, missing
  # census — AND the formerly-calibrated prefix refusal, which can now only
  # mean the flags dropped out of the payload, i.e., a wiring failure. An
  # unchosen non-measurement that lets the afterany resume chain continue
  # converts a broken adjudicator into a decorative one; this arm is where
  # the production chain is decided.
  local fs_cause="" fs_rec_class=""
  if [[ -r "${4:-}" ]]; then
    fs_cause=$(grep -m1 'live_gate could not measure:' "$4" 2>/dev/null || true)
  fi
  if [[ -s "${3:-}" ]]; then
    fs_rec_class=$(sed -n 's/.*"refusal_class": "\([a-z_]*\)".*/\1/p' "$3" | head -n1)
  fi
  case "$1" in
    0) # CLEAR must be corroborated, both ways: the gate printed its own
       # verdict into the capture AND the report it implies exists on disk
       # does. An rc of 0 without either is the founding bug's shape — a
       # process claiming success its own evidence does not carry — and it
       # maps to the infrastructure class, never to a pass (doctrines 4/5).
       if [[ -s "$3" ]] && [[ -r "${4:-}" ]] && grep -qF 'LIVE GATE VERDICT: CLEAR' "$4"; then
         FS_ART_GATE_STATE="CLEAR (corroborated by the gate's own printed verdict; report verified present: $3)"
         FS_ART_GATE_RC=0
       else
         FS_ART_GATE_STATE="OVERCLAIM (gate rc 0 uncorroborated — no self-printed CLEAR in capture ${4:-<none>} or no report at $3; class rc-92)"
         FS_ART_GATE_RC=92
         echo "artifact gate ($2): rc=0 yet the gate never printed its CLEAR verdict into ${4:-<no capture>} or its report $3 is missing/empty — an uncorroborated pass is the founding bug's shape; mapping to the infrastructure class, chain stops." >&2
       fi ;;
    1) FS_ART_GATE_STATE="BLOCKED (gate exit 1 — MEASURED and not clear; blocking reasons: $3)"
       FS_ART_GATE_RC=91
       echo "ARTIFACT GATE BLOCKED ($2): the adapter artifact was measured and did not clear, or a MUST_FIRE detector is unproven on it. Report: $3. This run's checkpoints are NOT cleared for resume, eval, or export." >&2
       if [[ -r "${4:-}" ]]; then grep -m4 '^      ' "$4" >&2 || true; fi ;;
    3) if [[ "$fs_rec_class" == "adapter_prefix_unpinned" ]]; then
         # RETIRED CALIBRATION (#78, this edit) — kept as an explicit RED
         # arm, never deleted, because a silently deleted calibration is
         # unreviewable. What this arm WAS: the ONE calibrated member of the
         # exit-3 class, mapped to rc 0 as "TODAY THE EXPECTED lora state",
         # because WE chose not to pin --adapter-prefix and the gate
         # correctly declined a job it was never configured to do (the
         # pre-#78 text, fix44 / #77-B3, earned even that green only as a
         # CONFIRMED fact from this same refusal record). Why the
         # calibration ended: the choice it calibrated no longer exists —
         # fs_live_save_gate now pins --adapter-prefix '' AND
         # --adapter-modules "$ADAPTER_MODULES" on every call, and the
         # gate's own contract (live_save_gate.py:511-523) licensed only
         # this COORDINATED edit, byte-for-byte. A gate that STILL raises
         # the prefix refusal is a gate whose payload silently lost the
         # flags — the same armed-no-op class as the drill that cannot fire
         # — and calibrating THAT to rc 0 is exactly how a gate goes
         # decorative. MUST_FIRE for the wiring (broken to see red): delete
         # --adapter-prefix '' from the fs_live_save_gate payload and every
         # gate call MUST land HERE and stop the chain — a launch that still
         # self-narrates UNMEASURED-by-prefix after this edit is the
         # measurement that the wiring did not land. Denominator, stated on
         # every fire: 0 of the gate's gates and 0 of its controls ran.
         FS_ART_GATE_STATE="UNMEASURED-INFRA (gate exit 3, record CONFIRMED adapter_prefix_unpinned at $3 — a refused-the-refusal now that --adapter-prefix '' and --adapter-modules are wired, #78; class rc-92)"
         FS_ART_GATE_RC=92
         echo "artifact gate ($2): exit 3 classified adapter_prefix_unpinned — CONFIRMED from the tool's own refusal record (verified present at $3), and now a FAILURE, not an abstention: --adapter-prefix '' and --adapter-modules $ADAPTER_MODULES are wired on every call since #78, so this refusal can only mean the flags silently dropped out of the fs_live_save_gate payload (or the gate drifted). Formerly the calibrated rc-0 'expected lora state' (fix44 / #77-B3); that calibration ended in the same edit that wired the census, per the byte-for-byte contract at live_save_gate.py:511-523. 0 of 3 gates and 0 of 3 controls ran; mapping to the infrastructure class, chain stops. Do NOT restore rc 0 here — resurrecting the abstention without retiring the wiring is the one-sided edit live_save_gate.py:511-523 forbids." >&2
       elif [[ "$fs_cause" == *"--adapter-prefix was not pinned"* ]]; then
         # The gate's words name the prefix abstention, but the refusal
         # record it must have written is absent or unmarked — precisely the
         # fix35-era state, now an indictment instead of a narration: a
         # claimed-but-absent record is not a record (#77-B3). Post-#78 the
         # words themselves are ALSO a wiring alarm (the arm above); the
         # missing record stays the deciding defect here.
         FS_ART_GATE_STATE="OVERCLAIM (gate's own line names the adapter-prefix abstention, but the refusal record it must have written is absent or unmarked at $3; class rc-92)"
         FS_ART_GATE_RC=92
         echo "artifact gate ($2): the gate's own output names the adapter-prefix abstention, but no refusal record with that classification exists at $3 — the refusal record it must have written is absent. A stated-on-stdout abstention without its on-disk record is the exact claim-vs-disk gap measured on both PROBE runs (#77-B3); mapping to the infrastructure class, chain stops." >&2
       else
         FS_ART_GATE_STATE="UNMEASURED-INFRA (gate exit 3, cause not the retired prefix refusal; class rc-92)"
         FS_ART_GATE_RC=92
         echo "artifact gate ($2): exit 3 whose cause is OUTSIDE the (retired, #78) adapter-prefix refusal — class rc-92 (wiring/tool/artifact failure), never an rc-0 abstention. Cause, quoted from the gate's own output: ${fs_cause:-<no 'live_gate could not measure:' line captured in ${4:-<none>} — an unreadable or silent capture is itself fail-closed>}. Since #78 NO member of the exit-3 class is a chosen abstention — the prefix is pinned on every call; every non-measurement stops the afterany chain, because calibrating a broken adjudicator to rc 0 is how a gate becomes decorative." >&2
       fi ;;
    *) FS_ART_GATE_STATE="INFRASTRUCTURE FAILURE (gate rc=$1 — outside 0/1/3: 2=argparse/wiring, 127=tool vanished, else=crash)"
       FS_ART_GATE_RC=92
       echo "artifact gate ($2): infrastructure failure (rc=$1) — the wiring, not the checkpoint, is broken. Fail closed." >&2 ;;
  esac
}

echo "============================================================"; echo "POST-RUN GATES ($OUTLOG)"
if [[ -r "$OUTLOG" ]]; then
  # G1 zero supervised tokens — CoT was stripped (the defining trap of this corpus)
  zst=$(grep -c "ZERO supervised tokens" "$OUTLOG" || true)
  echo "  G1 'ZERO supervised tokens' warnings: $zst (must be 0)"
  [[ "$zst" -eq 0 ]] || gate_fail "CoP patch inactive — chat template stripped <|channel>thought"

  # G2 attach census: peft logs one 'Adding lora to <module>' line per site
  att=$(grep -c "Adding lora to" "$OUTLOG" || true)
  att_exp=$(grep "Adding lora to" "$OUTLOG" | grep -c "experts" || true)
  echo "  G2 'Adding lora to' lines: $att total, $att_exp expert"
  [[ "$att" -gt 0 ]] || gate_fail "zero adapters attached — target names did not match"
  # fix39: this expectation keys on LORA_ARM, not $EXPERT_TARGETS. On a dense
  # base the arm branch deliberately DROPS the expert strings and relabels
  # L1->base4 while the EXPERT_TARGETS variable itself stays "1" — keying the
  # check on the raw request would demand expert attachments the branch
  # intentionally removed, a GUARANTEED false red on exactly the
  # measured-dense configuration (this estate, tonight) that the relabel
  # exists to make honest. LORA_ARM is the post-decision record the
  # contract suite's own arm-identity legs pin as authoritative: L1 <=>
  # expert strings were actually appended (MOE=1 && EXPERT_TARGETS=1).
  # Doctrine 5 is symmetric — a false failure is not the cure for a false
  # pass — and a red here would have marked the first repaired production
  # run failed for training the CORRECT module set.
  if [[ "$LORA_ARM" == "L1" ]]; then
    [[ "$att_exp" -gt 0 ]] || gate_fail "expert targets matched nothing (LORA plan STEP 0 check)"
  fi
  # expected sites == what the step-(5) census measured with the REAL matcher
  # (fix39: these lines read the grep-substring module dump — the oracle
  # measured to score the broken strings 42 and the correct ones 0, i.e. the
  # defect certifying itself). The census file carries one CENSUS_TARGET row
  # per shipped target; a MISSING row is a missing denominator and BLOCKS —
  # it never reads as zero (doctrines 1/4). got_n counts attach lines
  # containing the pattern's TAIL as a literal substring (the leading '*.'
  # stripped): grep -F treats '*' as a literal byte, so the raw wildcard
  # pattern could never match any attach line and would mint a false red
  # here — the same doctrine-5 defect in the failure direction. Substring is
  # a lawful oracle ONLY in this cheap confirming direction, over modules the
  # census already proved attachable; it is never again the certifying
  # direction. got == exp is deliberately NOT pinned: the multiplicity of
  # peft's 'Adding lora to' logging (per rank? per module?) is UNMEASURED on
  # this estate (0 repaired-run logs exist), and the PROBE run's own log is
  # the first honest measurement — a tighter bound asserted before that
  # measurement would be a claim broader than its evidence.
  set -f
  for t in ${LORA_TARGETS//,/ }; do
    exp_n=$(awk -v t="$t" '$1=="CENSUS_TARGET" && $2==t {print $3; exit}' "$PREFLIGHT_DIR/target_census.txt")
    if [[ -z "$exp_n" ]]; then
      gate_fail "no CENSUS_TARGET row for '$t' in $PREFLIGHT_DIR/target_census.txt — the preflight census cannot vouch for what it did not count"
      exp_n="MISSING"
    fi
    tail_t=${t#\*.}
    got_n=$(grep "Adding lora to" "$OUTLOG" | grep -cF "$tail_t" || true)
    echo "    target '$t': census=$exp_n attach-lines=$got_n (tail '$tail_t')"
    [[ "$got_n" -gt 0 ]] || gate_fail "attached 0 for '$t' (by tail substring over attach lines)"
  done
  set +f

  # G3 trainable census (fix44 / #76, extended by fix45 §A5 into a
  # three-field census of the trainer's FOUR-line block): small AND
  # nonzero, AND the realized rank must equal the requested rank, AND — new
  # — the rank-INVARIANT frozen base must equal its twice-measured
  # constant. NEEDLE RE-ANCHORED from HuggingFace PEFT's single-line
  # 'trainable params: N || all params: M || trainable%: P' — a format THIS
  # trainer never prints: the pre-fix44 grep matched 0 lines of BOTH
  # healthy PROBE logs and gate-failed two correct runs into exit 90 (jobs
  # 1787517960364 and 1787518637847). This stack (Megatron-Bridge) prints
  # FOUR lines, measured verbatim on <compute-node> (and the two gated numbers on
  # those healthy-run logs):
  #     PEFT Statistics:
  #       Total parameters: 7,750,478,080
  #       Trainable parameters: 63,078,400
  #       Trainable percentage: 0.81%
  # fix44 pinned the latter two (the fix44 packet had only those two
  # lines). fix45 adds 'Total parameters', the one measured line that buys
  # a STRICTLY BETTER frozen-base test than the percentage window, because
  # Total - Trainable IS the frozen base population and it is
  # rank-invariant (arithmetic in the identity check below), whereas the
  # percentage's denominator moves with rank. Extraction is one three-field
  # census under one counting discipline — a third grep is NOT bolted onto
  # a two-field parse: counts are of DISTINCT values per field (identical
  # per-rank repeats are one fact; distinct per-rank values are a defect
  # arm), all three counts travel together through EVERY branch below, and
  # a PARTIAL census (any strict subset of the three lines: a log cut
  # mid-print, or a format drift on one line only) lands on the named
  # ambiguous arm instead of laundering into 'missing' or 'found'. Measured
  # multiplicity facts (<compute-node>): each of the three lines appears EXACTLY
  # ONCE per run (rank 0 only), so -m1 would be safe though unused, and
  # `grep -c Trainable` on a healthy log = 2 — the tee's per-rank
  # duplication that hits the 672 attach lines does not reach the census
  # lines. The 'PEFT Statistics:' banner is documented, not parsed.
  g3_total_vals=$(grep -oE 'Total parameters: *[0-9,]+' "$OUTLOG" | sort -u || true)
  g3_tr_vals=$(grep -oE 'Trainable parameters: *[0-9,]+' "$OUTLOG" | sort -u || true)
  g3_pc_vals=$(grep -oE 'Trainable percentage: *[0-9.]+%' "$OUTLOG" | sort -u || true)
  g3_total_n=$(printf '%s\n' "$g3_total_vals" | grep -c . || true)
  g3_tr_n=$(printf '%s\n' "$g3_tr_vals" | grep -c . || true)
  g3_pc_n=$(printf '%s\n' "$g3_pc_vals" | grep -c . || true)
  echo "  G3 parameter census: $g3_total_n distinct 'Total parameters' value(s), $g3_tr_n distinct 'Trainable parameters' value(s), $g3_pc_n distinct 'Trainable percentage' value(s) in $OUTLOG"
  if [[ "$g3_total_n" -eq 0 && "$g3_tr_n" -eq 0 && "$g3_pc_n" -eq 0 ]]; then
    if grep -qF 'trainable params:' "$OUTLOG"; then
      gate_fail "the log carries the HuggingFace PEFT single-line census ('trainable params: N || all params: M || trainable%: P'), not this stack's measured four-line census block — the trainer stack or its logging CHANGED under a pinned needle. Re-derive the census against a real log and pin ONE format deliberately; the pre-fix44 needle matched 0 lines of 2 healthy-run logs, and accepting both formats sight-unseen is how a dead needle came to guard a stack this launcher never runs"
    else
      gate_fail "no parameter census in $OUTLOG — 0 of the 3 census lines this stack is measured to print are present (needles derived from 2 healthy-run logs plus the fix45 <compute-node> measurement of the full four-line block) — LoRA never initialized, or the log was cut before the census. An absent census is never a pass (doctrines 1/4)"
    fi
  elif [[ "$g3_total_n" -ne 1 || "$g3_tr_n" -ne 1 || "$g3_pc_n" -ne 1 ]]; then
    gate_fail "parameter census is ambiguous: $g3_total_n distinct 'Total parameters' value(s), $g3_tr_n distinct 'Trainable parameters' value(s), $g3_pc_n distinct 'Trainable percentage' value(s) where exactly 1 of each is required — a PARTIAL census (one or two of the three lines: a log cut mid-print, or a format drift on one line only) or per-rank values that disagree. Reading whichever value appeared first would mint a verdict from a coin flip; BLOCKED, all three counts named (doctrine 4)"
  else
    total_n=$(printf '%s\n' "$g3_total_vals" | sed -E 's/Total parameters: *([0-9,]+)/\1/' | tr -d ,)
    tr_n=$(printf '%s\n' "$g3_tr_vals" | sed -E 's/Trainable parameters: *([0-9,]+)/\1/' | tr -d ,)
    pct=$(printf '%s\n' "$g3_pc_vals" | sed -E 's/Trainable percentage: *([0-9.]+)%/\1/')
    if [[ "$total_n" =~ ^[0-9]+$ && "$tr_n" =~ ^[0-9]+$ && "$pct" =~ ^[0-9.]+$ ]]; then
      [[ "$tr_n" -gt 0 ]] || gate_fail "ZERO trainable params — classic silent LoRA failure"
      # Percentage sanity window (0.10, 10.0) — this window's job is BASE
      # FROZEN, and only that. It is deliberately NOT a rank check, because
      # the percentage is a function of rank: pct(r) = 100*1,971,200*r /
      # (7,687,399,680 + 1,971,200*r), where 7,687,399,680 is the measured
      # frozen base (Total - Trainable, identical at r=32 and r=96 — the
      # denominator moves with rank, which is exactly why a percentage
      # cannot pin one). Edges derived by evaluating that function over the
      # ranks an operator may legitimately set, since FS_PEFT_DRILL_RANK is
      # operator-chosen and arbitrary: r=8 -> 0.205%, r=16 -> 0.409%, r=32
      # -> 0.814% (production, measured), r=96 -> 2.403% (drill, measured),
      # r=128 -> 3.178%, r=256 -> 6.16%. A tighter (0.50, 3.0) was drafted
      # and REJECTED on this arithmetic: it false-reds r=16 under its floor
      # and every r >= 122 over its ceiling, and it leaves the measured
      # r=96 drill sitting 0.6 pp under the edge — a gate that reddens on
      # honest configurations trains operators to ignore it, which is the
      # doctrine-5 symmetric defect (a false alarm is as much a defect as a
      # false green). (0.10, 10.0) holds every rank 8..512 and still catches
      # the classes this check exists for by an order of magnitude: base
      # un-frozen prints ~100%, and a per-expert attach explosion on an MoE
      # base prints >= 0.81 x num_experts (plan watch-out 1; >= 26% at 32
      # experts). Stated explicitly, because a window is not a
      # discriminator: EVERY interior wrong-rank build passes this window BY
      # DESIGN; discriminating rank is the integer-equality check below, and
      # quoting the window as proof of the right rank would be a claim
      # broader than its evidence. fix45 addendum: the window is now also
      # the ONLY param check that survives outside the calibrated estate —
      # the scoped frozen-base identity below (Total - Trainable ==
      # 7,687,399,680) is strictly stronger where it applies because the
      # window's denominator moves with rank while the identity's does not;
      # outside that scope BOTH identities abstain by name and this window
      # stands alone as the coarse frozen-base guard, which is exactly why
      # its edges stay wide.
      awk -v p="$pct" 'BEGIN{exit !(p>0.10 && p<10.0)}' || gate_fail "trainable%=$pct outside (0.10,10.0) — base not frozen (an unfrozen base prints ~100%), or per-expert attach explosion (plan watch-out 1: could go x num_experts). This window does NOT check rank; see the rank-identity check below"
      # Rank identity (fix44). Trainable count is EXACTLY linear in rank on
      # this estate: each adapted linear contributes r*(in+out) params (A is
      # (rank,in), B is (out,rank); biases None, alpha a scalar, dropout
      # stateless, base frozen — the freeze asserted two checks up), so
      # trainable = r * K with K = sum of (in+out) over the adapted set. For
      # the base4 168-module set on this base K is MEASURED TWICE on
      # hardware and agrees to the integer: 63,078,400/32 = 1,971,200 =
      # 189,235,200/96 (and 189,235,200/63,078,400 = 3.0 = 96/32 exactly), so
      # the honest check is integer equality, not a tolerance window — the
      # quantity is deterministic, and equality is exactness by arithmetic,
      # not bit-exactness by luck. Exactness assumptions, stated: (i) the
      # adapted set is tonight's 168-module base4 census (step-(5) and G2
      # pin it), (ii) E4B module dims are fixed by the measured config,
      # (iii) nothing outside the adapters is trainable (a future
      # modules_to_save or unfrozen scope breaks equality LOUDLY — the
      # desired direction), (iv) the printed count is the global count,
      # verified only under the shipped geometry TP=CP=EP=1. This is the
      # ARTIFACT-level rank discriminator the config-level drill (fix42)
      # cannot be: composition may resolve peft.dim=$LORA_RANK while the
      # optimizer was handed something else — the silent-revert signature —
      # and a percentage window cannot see that difference.
      # Frozen-base identity (fix45 / §A5). Total - Trainable IS the frozen
      # base population, and it is rank-INVARIANT — measured identical at
      # both extremes the estate has run, on hardware:
      #     7,750,478,080 -  63,078,400 = 7,687,399,680   (r=32, production)
      #     7,876,634,880 - 189,235,200 = 7,687,399,680   (r=96, drill)
      # That invariance is exactly what the percentage lacks — the
      # window's denominator (Total) includes the adapters, so the
      # window's basis moves with rank — and it is why this check is
      # strictly stronger wherever it applies: the window catches a base
      # that CATASTROPHICALLY thawed (~100% trainable), while this integer
      # catches ANY thaw from one param upward, and any trainable scope
      # beyond the adapter set — the precise classes that hide inside the
      # window's interior by design. Same exactness assumptions as the rank
      # identity above, stated per its own list: (i) the adapted set is
      # tonight's 168-module base4 census (step-(5) and G2 pin it), (ii)
      # E4B module dims are fixed by the measured config, (iii) nothing
      # outside the adapters is trainable (a future modules_to_save or
      # unfrozen scope breaks equality LOUDLY — the desired direction),
      # (iv) BOTH printed numbers are assumed global under the shipped
      # geometry TP=CP=EP=1. It inherits the rank identity's scope guard
      # verbatim, and outside that scope it ABSTAINS BY NAME rather than
      # inventing a constant for a population nobody measured twice
      # (doctrine 5): the (0.10,10.0) window alone stands there, and that
      # is now stated in its comment as well.
      if [[ "$LORA_TARGETS" == "$LORA_TARGETS_BASE" && "$MOE" != "1" && "$TP" == "1" && "$CP" == "1" ]]; then
        g3_expect=$(( LORA_RANK * 1971200 ))
        if [[ "$tr_n" -ne "$g3_expect" ]]; then
          if (( tr_n % 1971200 == 0 )); then g3_realized=$(( tr_n / 1971200 )); else g3_realized="non-integral ($tr_n/1971200)"; fi
          gate_fail "REALIZED RANK != REQUESTED RANK: the trainer reports $tr_n trainable params, i.e. realized rank $g3_realized at the exact 1,971,200 params/rank of the 168-module base4 set, but rank $LORA_RANK was requested (expected $g3_expect params). This is the silent-revert signature at the artifact level: the layer that CLAIMS peft.dim=$LORA_RANK is not the layer that BUILT it. Investigate the override/composition path (fix42 drill territory); never rescue this into a pass"
        fi
        g3_frozen=$(( total_n - tr_n ))
        if [[ "$g3_frozen" -ne 7687399680 ]]; then
          gate_fail "frozen-base identity BROKEN: Total - Trainable = $g3_frozen (Total=$total_n, Trainable=$tr_n) != 7,687,399,680, the frozen base measured TWICE on hardware and reconciled to the integer (r=32 and r=96). Suspects, in order: (i) the base THAWED — every param that left the frozen pool moves this integer, which is precisely what the (0.10,10.0) window's interior hides by design; (ii) trainable scope beyond the adapters (a modules_to_save or an unfrozen horizon) — symmetric with assumption (iii) above: the two identities stand or fall on the same premise; (iii) a census that is not one run's census (interleaved or grafted lines). Never rescue this into a pass: the percentage window alone CANNOT see this difference."
        fi
        echo "    trainable=$tr_n ($pct%) of total=$total_n — adapters live; realized rank == requested rank $LORA_RANK exactly (168-module base4 set, 1,971,200 params/rank, measured twice on hardware); frozen-base identity holds exactly (Total - Trainable = $g3_frozen = 7,687,399,680, rank-invariant across both hardware measurements)"
      else
        echo "    trainable=$tr_n ($pct%) of total=$total_n — nonzero and inside the gross window. Rank AND frozen-base identities ABSTAIN, by name (doctrine 5): the 1,971,200 params/rank constant and the 7,687,399,680 frozen-base constant each carry a two-measurement provenance ONLY for the base4 168-module set at TP=CP=EP=1 on a measured-dense base, and this run sits outside that scope (targets-are-base4=$([[ "$LORA_TARGETS" == "$LORA_TARGETS_BASE" ]] && echo yes || echo no) MOE=$MOE TP=$TP CP=$CP). The zero-check and the (0.10,10.0) base-frozen window above remain tonight's only param-count guard here — a coarse FROZEN-BASE guard, with NEITHER the realized rank NOR the frozen base pinned to an integer."
      fi
    else
      gate_fail "trainable census did not parse (count='$tr_n', pct='$pct') — an unreadable denominator BLOCKS (doctrine 4)"
    fi
  fi

  # G4 schedule honesty: resolved config must anneal on this budget
  grep -m1 "lr_decay_iters" "$OUTLOG" || echo "  G4 WARN: lr_decay_iters not in log — verify resolved config by eye"
else
  gate_fail "cannot read own log $OUTLOG — gates could not run"
fi

# G5 PROBE-only: prove the SAVE path end-to-end on disk (traps 9/11)
if [[ "${PROBE:-0}" == "1" ]]; then
  [[ -f "$LATEST_FILE" ]] || gate_fail "PROBE produced no latest_checkpointed_iteration.txt"
  last=$(tr -dc '0-9' < "$LATEST_FILE" 2>/dev/null || echo 0)
  [[ "$last" -eq "$TRAIN_ITERS" ]] || gate_fail "PROBE last_iter=$last != $TRAIN_ITERS"
  nd=$(ls -d "$CKPT_DIR"/iter_* 2>/dev/null | wc -l)
  [[ "$nd" -ge 2 ]] || gate_fail "PROBE expected >=2 iter_* dirs (saves at 10 and 20), found $nd"
  sz_mb=$(du -sm "$CKPT_DIR"/iter_*"$last" 2>/dev/null | cut -f1 || echo 0)
  echo "  G5 PROBE ckpt: $nd dirs, last=$last, iter dir ${sz_mb} MB"
  # LoRA kind check: adapter ckpt of an ~8B model should be ~10 MB — a few GB.
  # (>100 GB would mean we saved FULL weights+optimizer = not a LoRA ckpt.)
  [[ "$sz_mb" -ge 5 && "$sz_mb" -le 20000 ]] || gate_fail "ckpt size ${sz_mb}MB implausible for an adapter checkpoint"
  echo "PROBE GATES DONE — inspect '$CKPT_DIR' and module_dump attach ratios BEFORE launching production."
fi

# --- G-artifact (fix35): tools/live_save_gate.py over the realized save ----
# Target resolution mirrors the full-FT exact-match rule (no zero-padding
# assumption). Zero iteration dirs means the adjudicator examined 0 of 1
# artifacts: stated as a named abstention-shaped state with the denominator,
# never read as a pass — and G5 already polices save existence under PROBE.
# The rc is captured and tested on every arm of the 0/1/3/other contract;
# a `|| true` here would re-launder the founding bug verbatim.
FS_ART_GATE_RC=0
FS_ART_GATE_STATE="not-run"
ART_CKPT=""
if [[ -f "$LATEST_FILE" ]]; then
  ART_LAST=$(tr -dc '0-9' < "$LATEST_FILE")
  if [[ -n "$ART_LAST" ]]; then
    for d in "$CKPT_DIR"/iter_*; do
      [[ -d "$d" ]] || continue
      if [[ "${d##*/iter_}" =~ ^0*${ART_LAST}$ ]]; then ART_CKPT=$d; break; fi
    done
  fi
fi
if [[ -n "$ART_CKPT" ]]; then
  ART_REPORT=$OUTPUT_DIR/fs_gate/report-lora.json
  # The capture log is the gate's own voice (fix44 / #77-B2/B3): the mapper
  # decodes the exit-3 CLASS from what the gate actually said and corroborates
  # CLEAR from what the gate actually printed — the job log's narration no
  # longer asserts a cause it never read.
  ART_CAPTURE=$OUTPUT_DIR/fs_gate/capture-lora.log
  fs_ag=0
  fs_live_save_gate "$ART_CKPT" save "$ART_REPORT" "$ART_CAPTURE" || fs_ag=$?
  fs_lora_gate_verdict_to_rc "$fs_ag" "post-run adapter ($ART_CKPT)" "$ART_REPORT" "$ART_CAPTURE"
else
  FS_ART_GATE_STATE="NO-ARTIFACT — 0 iter_* dirs under $CKPT_DIR; adjudicator examined 0 of 1 checkpoints (named abstention shape, not a pass)"
  echo "artifact gate: $FS_ART_GATE_STATE" >&2
fi

echo "Job ${SLURM_JOB_ID:-N/A} finished: train_rc=$RC gate_rc=$GATE artifact_gate=$FS_ART_GATE_STATE (rc=$FS_ART_GATE_RC)"
[[ "$GATE" -eq 0 ]] || exit 90
# 91=adjudicator measured a blocking verdict (stops the afterany resume
# chain: poisoned adapters must not be resumed from); 92=adjudicator
# infrastructure failed. Both are distinct from 90 (log gates) on purpose.
[[ "$FS_ART_GATE_RC" -eq 0 ]] || exit "$FS_ART_GATE_RC"
exit $RC
