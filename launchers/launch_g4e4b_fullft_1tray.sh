#!/bin/bash
# ============================================================================
# Gemma-4-E4B (VL-family, ~4B-active text + vision/audio) — FULL fine-tune
# on the SFT-Taiwan-AIEC v3 corpus. FIRST real FoundationScale production job.
#
# Modelled on:
#   - launch_gemma4_moe_smoke.sh   (container / NCCL / CLI-override conventions)
#   - launch_g4dense31b_sft.sh     (validation helpers, resume detection, eval-off)
#   - run_26b_fullft_v3.sh         (FOXBRAIN_SFT_JSONLS, rows->iters arithmetic)
#
# Deliberate deviations from the bigger siblings (justified in the rationale):
#   TP=1 CP=1 PP=1 DP=4; EP=4 only if config.text_config.enable_moe_block=true.
#
# Env overrides: TP CP EP ETP SEQ_LENGTH GBS MBS EPOCHS TRAIN_ITERS
#                SAVE_INTERVAL MASTER_PORT OUT_DIR BASE_CKPT EXTRA_OVERRIDES
#                PROBE=1 -> 20 iters, saves at 10 AND 20, OUT_DIR gains a
#                _probe suffix, G5 save-path gate on exit (mirrors the LoRA
#                launcher). Unset reads as 0; any other value is refused --
#                silence about this knob was finding #81 and must not return.
# ============================================================================
#SBATCH --job-name=g4e4b-tw-v3-fullft
#SBATCH --partition=<group>
#SBATCH --nodes=1
#SBATCH --nodelist=<compute-node>        # STANDING RULE: <compute-node> only; <other-team-node> is another team's node
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=30
#SBATCH --mem=400G
# STANDING RULE: --time=10-00:00:00 on EVERY sbatch. A RUNNING job's limit cannot
# be raised, only a pending one's; the old 24 h caps silently ended runs mid-anneal.
#SBATCH --time=10-00:00:00
#SBATCH --exclusive
#SBATCH --output=<CLUSTER_HOME>/Training-model/FoxBrain-omni-accel/logs/g4e4b_sft_%j.out
#SBATCH --error=<CLUSTER_HOME>/Training-model/FoxBrain-omni-accel/logs/g4e4b_sft_%j.err

set -uo pipefail
export PATH=/cm/local/apps/slurm/current/bin:$PATH      # slurm bins are only on PATH under bash -lc
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf

die() { echo "ERROR: $*" >&2; exit 1; }
require_pos_int() { local n=$1 v=$2; [[ $v =~ ^[0-9]+$ ]] && (( v >= 1 )) || die "$n must be a positive integer, got '$v'"; }
require_nonneg_int() { local n=$1 v=$2; [[ $v =~ ^[0-9]+$ ]] || die "$n must be a non-negative integer, got '$v'"; }

# The standing-rule node guard MOVED into fs_backend_init, sourced a few lines
# below immediately after the paths it needs. Nothing is weakened: the slurm
# arm there keeps these two checks verbatim in effect, and the enroot arm
# proves node identity from `hostname -s` instead of an allocation string —
# off-Slurm there IS no allocator, so any SLURM_JOB_NODELIST found in the
# environment would be a caller-written string, and checking it would be the
# self-satisfying guard this repository refuses. The init must precede the
# geometry section because that section divides by SLURM_NTASKS (minted
# off-Slurm) and MASTER_PORT math reads SLURM_JOB_ID (minted numeric).

# Estate root: prose pseudonym <CLUSTER_HOME>, code mechanism $CLUSTER_HOME.
# Default $HOME is behaviour-preserving on the published tray (measured:
# $HOME IS the estate home there -- fs_container_backend.sh, fix32 precedent)
# and lets the contract suite drive this launcher through a sandbox HOME.
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
A=$CLUSTER_HOME/Training-model/FoxBrain-omni-accel
REPO=$A/Megatron-Bridge
EXTRAS=$A/python-extras-mbridge     # trap1: $EXTRAS is FIRST on PYTHONPATH; never pip --target into it without dep pruning
SQSH=$CLUSTER_HOME/SQSH-env/nemo-automodel-26-04_compute.sqsh
G=$CLUSTER_HOME/pretraining_weights/Vision-Language-Models/Google/Gemma4
# fix45-A2 / #82: this assignment MUST be exported. The in-container preflight
# probe ($COT_PROBE_PY below) reads os.environ["HF_MODEL"], and run_in_container
# forwards EXPORTED env only (s7). Measured on <compute-node>: unexported, the probe
# died KeyError: 'HF_MODEL' on every launch ("preflight tokenizer/CoT probe
# FAILED — job stopped before any training GPU-seconds") — a second
# unconditional hard block in series behind the emitter, invisible until the
# emitter was unblocked, which is why it had never been seen. grep -c
# "export HF_MODEL" on the shipped launcher was 0. The generalized rule and
# its census live in the contract suite: EVERY variable a container-side
# python body reads from os.environ must be exported by this launcher (the
# FOXBRAIN_SFT_JSONLS export in the Paths block above is the standing
# MUST_PASS member of that census).
export HF_MODEL=${HF_MODEL:-$G/gemma-4-E4B-it}          # -it: ships chat_template.jinja; v3 corpus needs NO inference-side template patch
V3=$CLUSTER_HOME/Post-training-Data/SFT-Taiwan-AIEC/formatted-gemma4-v3/train

# Converted Megatron checkpoint. Estate convention (smoke + dense + 31B-base job 1952):
# training loads a PRE-CONVERTED ckpt (iter_0000000/), never the raw HF dir.
# VERIFIED 2026-08-23: this path EXISTS — iter_0000000/ is a 15 GB torch_dist DCP
# (1,252 tensors + 423 _extra_state blobs, 14.32 GiB implied, bfloat16, dense:
# zero "expert" FQNs). Superseded the earlier "does not exist yet per estate
# notes" claim, which came from exports/EXPORT_STATUS.md — a document the
# architecture review already flagged as superseded-but-still-present. This
# launcher still REFUSES to run without it; the guard is not the stale part.
BASE_CKPT=${BASE_CKPT:-$A/converted_ckpts/gemma-4-E4B-it}

# The ONLY supported way to point a recipe at a custom corpus in this estate
# (run_recipe.py has no --jsonl_paths; read by _env_jsonls() in recipes/gemma4_vl/gemma4_vl.py)
export FOXBRAIN_SFT_JSONLS="$V3/foxbrain_identification.jsonl,$V3/kenny_cot_think.jsonl,$V3/kenny_notag_nothink.jsonl,$V3/lilian_taiwan_multicat.jsonl"

# ----------------------------------------------------------------------------
# EXECUTION BACKEND (see fs_container_backend.sh; selection rationale in the
# comment where the old top-level guard stood, above)
# ----------------------------------------------------------------------------
# fs_backend_init: backend select + node-identity proof + SLURM_* minting.
# fs_backend_runtime_setup: enroot arm only — provenance-checked idempotent
# container unpack (a name match is not origin), GPU-drain gate (s8d), then a
# tee of everything below into the exact file LOG_OUT names below; off-Slurm
# nothing captures our stdout for us, and the tripwires/epilogue grep LOG_OUT.
# shellcheck source=launchers/fs_container_backend.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fs_container_backend.sh"
fs_backend_init "$A"
fs_backend_runtime_setup "$SQSH" "${SLURM_NTASKS}" "$A/logs/g4e4b_sft_${SLURM_JOB_ID}.out"

# ----------------------------------------------------------------------------
# Read the HF config to derive MoE-ness — parallel geometry is decided from
# MEASURED config keys (enable_moe_block / num_experts in text_config),
# not assumptions. (Facts: model_type gemma4, text_config 42 layers / hidden 2560.)
# ----------------------------------------------------------------------------
[[ -f "$HF_MODEL/config.json" ]] || die "HF model missing config.json: $HF_MODEL"
[[ -f "$HF_MODEL/model.safetensors" ]] || die "HF model missing model.safetensors (expected single 14.89 GiB shard): $HF_MODEL"

cfg_get() {  # key -> value from text_config (top-level fallback); python3 first, grep fallback
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$HF_MODEL/config.json" "$1" <<'PY'
import json,sys
with open(sys.argv[1]) as f: cfg=json.load(f)
tc=cfg.get("text_config",{}) or {}
print(tc.get(sys.argv[2], cfg.get(sys.argv[2],"")))
PY
  else  # UNVERIFIED fallback: assumes the key appears exactly once in config.json
    grep -m1 -oE "\"$1\"[[:space:]]*:[[:space:]]*[^,}]+" "$HF_MODEL/config.json" | sed -E 's/^[^:]*:[[:space:]]*//' | tr -d '" '
  fi
}
ENABLE_MOE_BLOCK=$(cfg_get enable_moe_block)
NUM_EXPERTS=$(cfg_get num_experts)
TEXT_LAYERS=$(cfg_get num_hidden_layers)
TEXT_HIDDEN=$(cfg_get hidden_size)
[[ -z "$ENABLE_MOE_BLOCK" ]] && die "could not parse text_config.enable_moe_block from $HF_MODEL/config.json — refusing to GUESS the parallel geometry"
{ [[ -z "$TEXT_LAYERS" ]] || [[ "$TEXT_LAYERS" == "42" ]]; } || echo "WARN: text_config layers=$TEXT_LAYERS (estate facts said 42) — continuing, but confirm you have the right weights"
{ [[ -z "$TEXT_HIDDEN" ]] || [[ "$TEXT_HIDDEN" == "2560" ]]; } || echo "WARN: text_config hidden=$TEXT_HIDDEN (estate facts said 2560)"

# ----------------------------------------------------------------------------
# Geometry: TP=1 CP=1 (PP=1 = recipe default), DP=world=4.
#  - ~4B-active model with hidden 2560: tensor parallelism buys nothing and
#    TP>1 is NOT pre-validated (E4B KV-head count UNVERIFIED) -> hard-refused.
#  - MoE text tower -> EP=world on one tray (Estate convention: smoke runner
#    used EP=world for the 26B); num_experts must divide evenly by EP.
#  - Dense text tower -> EP=ETP=1 with the same hard guard the dense 31B uses.
# ----------------------------------------------------------------------------
TP=${TP:-1}; CP=${CP:-1}; ETP=${ETP:-1}
require_pos_int TP "$TP"; require_pos_int CP "$CP"; require_pos_int ETP "$ETP"
(( ETP == 1 )) || die "ETP>1 not pre-validated for E4B"
(( TP == 1 )) || die "TP>1 NOT pre-validated for E4B (num KV heads UNVERIFIED). TP=1 is the only supported geometry for this first run."
(( CP == 1 )) || die "CP>1 unnecessary at seq 8K on one tray and not pre-validated"

if [[ "$ENABLE_MOE_BLOCK" == "true" || "$ENABLE_MOE_BLOCK" == "True" ]]; then
  MOE=1
  EP=${EP:-4}                                   # EP=world: num_experts/4 per rank
  require_pos_int EP "$EP"
  require_pos_int NUM_EXPERTS "$NUM_EXPERTS"    # MoE but no parseable num_experts -> stop
  (( NUM_EXPERTS % EP == 0 )) || die "num_experts=$NUM_EXPERTS not divisible by EP=$EP"
  (( (SLURM_NTASKS) % (EP * TP * CP) == 0 )) || die "world=$SLURM_NTASKS not divisible by EP*TP*CP=$((EP*TP*CP))"
  MOE_OVERRIDES="model.expert_tensor_parallel_size=$ETP model.expert_model_parallel_size=$EP \
                 model.moe_token_dispatcher_type=alltoall model.moe_grouped_gemm=True"
else
  MOE=0
  # fix28 Q2: an EXPLICIT EP/ETP on a measured-dense base is REFUSED, never
  # silently coerced. Honoring EP>1 with zero experts is the mis-shard this
  # branch exists to prevent (the 31B recipe hard-asserts the same); silently
  # coercing it would make the banner and the run manifest record a knob the
  # operator never wrote. One loud t=0 error that names both the instruction
  # and the measured fact beats either. Unsetting the var is the operator's
  # one-word acknowledgement that they read it. ($ETP is already defaulted to
  # 1 by the time this branch runs; $EP is read un-defaulted precisely so an
  # explicit EP=1 is accepted but distinguishable from unset.)
  if [[ -n "${EP:-}" && "${EP}" != "1" ]]; then
    die "EP=$EP set explicitly, but the base is DENSE (enable_moe_block=$ENABLE_MOE_BLOCK, num_experts=${NUM_EXPERTS:-null}). With zero experts, EP>1 silently mis-shards the model. Unset EP to launch dense (EP=1), or point HF_MODEL at a MoE base."
  fi
  if [[ "${ETP}" != "1" ]]; then
    die "ETP=$ETP set explicitly, but the base is DENSE — expert tensor parallelism has nothing to shard. Unset ETP to launch dense (ETP=1)."
  fi
  EP=1
  MOE_OVERRIDES=""                              # dense launcher carries NO MoE overrides by design
  echo "config says enable_moe_block=$ENABLE_MOE_BLOCK -> treating text tower as DENSE (EP=1)"
fi
DP=$(( SLURM_NTASKS / (TP * CP) ))

GBS=${GBS:-16}; MBS=${MBS:-1}
require_pos_int GBS "$GBS"; require_pos_int MBS "$MBS"
(( GBS % (DP * MBS) == 0 )) || die "GBS=$GBS not divisible by DP*MBS=$((DP*MBS)) (would silently change grad-accum)"
SEQ_LENGTH=${SEQ_LENGTH:-8192}                  # official Taiwan-AIEC setting (32K was retested and rejected; 16K was for the bigger models' earlier runs)
require_pos_int SEQ_LENGTH "$SEQ_LENGTH"
SEQ_K=$(( (SEQ_LENGTH + 512) / 1024 ))

# fix28 A2 (measured mechanism, modeling_gemma4_e4b_vl.py:79-93): the E4B PLE
# slice is stashed per decoder layer at :82 and cleared to None by the finally
# at :93, which fires BEFORE backward. ANY recompute re-runs the layer forward
# during backward, sees _ple_input=None, takes the non-PLE branch, and the PLE
# parameters receive NO gradient — hard error under DDP all-params-used, or
# SILENTLY WRONG training without it. The old default "selective" was calibrated
# for the 26B/31B, whose layers have no PLE stash; on E4B it trains wrong with
# no error. 'full' remains additionally broken for the older reason
# (save_for_backward tuple error). So "none" is the ONLY legal value here, and
# when it resolves we emit NO recompute override below — the E4B recipe pins
# granularity/method/num_layers=None and hard-asserts off, and passing the
# literal string "none" on a CLI whose None-parsing is unverified would itself
# be an assumption (fail closed by omission). If OOM ever appears, the levers
# are SEQ_LENGTH/MBS — recompute is not one of them for this model.
RECOMPUTE=${RECOMPUTE:-none}
case "$RECOMPUTE" in
  none) ;;
  selective) die "RECOMPUTE=selective trains E4B WITHOUT PLE gradients (fix28 A2: stash at modeling_gemma4_e4b_vl.py:82, cleared at :93 finally; the backward-time re-forward sees _ple_input=None). Refusing." ;;
  full) die "RECOMPUTE=full is BROKEN on gemma4 (save_for_backward tuple error) AND hits the A2 PLE no-gradient mechanism. Refusing." ;;
  *) die "RECOMPUTE must be none for E4B, got '$RECOMPUTE'" ;;
esac

# CoT trap #1: the in-tree training-time template patch MUST be active or
# think rows supervise ZERO tokens. Default ON; refusing to run without it.
export FOXBRAIN_GEMMA4_KEEP_COT=${FOXBRAIN_GEMMA4_KEEP_COT:-1}
[[ "$FOXBRAIN_GEMMA4_KEEP_COT" == "1" ]] || die "FOXBRAIN_GEMMA4_KEEP_COT=0 would re-enable strip_thinking on supervised targets (trap #1). Refusing."

# ----------------------------------------------------------------------------
# Data gates (rows > 0 per file, schema spot-check) + iteration arithmetic
# ----------------------------------------------------------------------------
ROWS=0
for f in ${FOXBRAIN_SFT_JSONLS//,/ }; do
  [[ -f "$f" ]] || die "missing corpus file: $f (missing files are a HARD error — a shortened corpus looks exactly like a healthy short run)"
  n=$(wc -l < "$f" | tr -d ' ')
  require_pos_int "rows($f)" "$n"
  ROWS=$(( ROWS + n ))
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$f" <<'PY' || die "schema check failed on $f (expected {\"id\", \"conversations\":[{\"from\": human|gpt, \"value\"}]})"
import json,sys
row=json.loads(open(sys.argv[1]).readline())
assert isinstance(row.get("id"),str) and isinstance(row.get("conversations"),list) and row["conversations"], "bad schema"
assert all(c.get("from") in ("human","gpt") and isinstance(c.get("value"),str) for c in row["conversations"]), "bad role/value"
PY
  else
    echo "WARN: python3 unavailable on host — skipped JSON schema spot-check for $f"   # UNVERIFIED: compute-node host python3 presence
  fi
done
EPOCHS=${EPOCHS:-2}                              # estate standard for this corpus
ITERS=$(( (ROWS * EPOCHS + GBS - 1) / GBS ))
# fix #81 finding 3: snapshot the operator's EXPLICIT TRAIN_ITERS before the
# default fills it in below, so the probe arm can LOG a displacement rather
# than silently overwrite one -- the run manifest records only final values.
# SAVE_INTERVAL gets its own snapshot immediately before ITS default line:
# each snapshot sits provably upstream of the default it protects.
OPERATOR_TRAIN_ITERS=${TRAIN_ITERS:-}
TRAIN_ITERS=${TRAIN_ITERS:-$ITERS}
require_pos_int TRAIN_ITERS "$TRAIN_ITERS"
# ~16.1k rows * 2 / GBS 16 = ~2016 iters. Wall-clock is NOT the binding constraint
# (10-day cap on 4x GB200 for a ~4B-active model), so train_iters == the real
# budget and scheduler.lr_decay_iters=$TRAIN_ITERS COMPLETES the cosine. This is
# NOT a smoke test; the LR schedule is meaningful.

# fix #81 finding 3: companion to OPERATOR_TRAIN_ITERS above -- capture the
# operator's EXPLICIT SAVE_INTERVAL before the default folds in (empty reads
# as unset: an empty assignment defaults anyway and deserves no warning).
OPERATOR_SAVE_INTERVAL=${SAVE_INTERVAL:-}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}              # FIRST_SAVE EARLY: see rationale (trap 9/11 lesson)
require_pos_int SAVE_INTERVAL "$SAVE_INTERVAL"
EVAL_INTERVAL=100000; EVAL_ITERS=0               # val loader is NOT modality-bucketed -> any eval pass deadlocks silently. Keep OFF.

# PROBE mode (fix #81), with probe parity to the LoRA launcher's probe-mode
# block. The original defect was SILENCE plus a name collision: `PROBE=1
# sbatch` bound the mode knob, then the old preflight path assignment (now
# COT_PROBE_PY, below) overwrote the operator's `1` with a path before
# anything read it as a mode, handing the operator a ~2016-iter production
# run. So no read of this knob may be quiet again: unset reads as 0, 1
# selects the probe arm, and any OTHER value is refused outright (fail
# closed -- silently mapping "yes" onto production would be the finding
# re-born with better spelling). The arms write back a normalized 0/1, so
# the banner below and the G5 gate read the RESOLVED $PROBE and never
# re-read the environment independently.
case "${PROBE:-0}" in
  0|1) ;;
  *) die "PROBE must be 0 or 1, got '${PROBE}' -- refusing to choose between a 20-iter probe and a ~2016-iter production run on the operator's behalf" ;;
esac
if [[ "${PROBE:-0}" == "1" ]]; then
  PROBE=1
  # Displacement is LOGGED, never silent (fix #81 finding 3): the probe
  # budget forces 20/10 over any explicitly-set knobs, and the emitted run
  # manifest records only the resolved values -- this echo is the record.
  # The ==20/==10 no-op cases warn NOTHING: nothing was displaced, and a
  # false alarm costs what a false green costs.
  [[ -z "$OPERATOR_TRAIN_ITERS" || "$OPERATOR_TRAIN_ITERS" == 20 ]] || \
    echo "WARN: PROBE=1 forces TRAIN_ITERS=20 over explicit $OPERATOR_TRAIN_ITERS (fix #81)"
  [[ -z "$OPERATOR_SAVE_INTERVAL" || "$OPERATOR_SAVE_INTERVAL" == 10 ]] || \
    echo "WARN: PROBE=1 forces SAVE_INTERVAL=10 over explicit $OPERATOR_SAVE_INTERVAL (fix #81)"
  TRAIN_ITERS=20        # literal, self-validating; the row-count arithmetic above sized PRODUCTION only
  SAVE_INTERVAL=10      # a run is healthy only after its FIRST save -- probe forces saves at 10 AND 20
  RUN_SUFFIX=_probe
else
  PROBE=0
  RUN_SUFFIX=""
fi

MASTER_PORT=${MASTER_PORT:-$(( 29400 + SLURM_JOB_ID % 1000 ))}
(( MASTER_PORT >= 1024 && MASTER_PORT <= 65535 )) || die "MASTER_PORT out of range: $MASTER_PORT"
# ${RUN_SUFFIX} is folded in where OUT_DIR is BORN, ahead of every consumer
# (the mkdir/write-probe below, the disk watermark check, WANDB_DIR, the
# checkpoint.load/save paths, the resume read of checkpoints/ at ~635): a
# probe must be PHYSICALLY unable to write into the stable auto-resume
# chain, else its latest_checkpointed_iteration.txt (=20) seeds that chain
# and the next production launch "resumes" a throwaway probe at 20/2016 --
# the probe's optimizer state silently worn by the run it was meant to
# precede. The suffix sits OUTSIDE the default: an operator-set OUT_DIR
# takes it too, because a custom path is no proof the mode knob was
# remembered. Suffixing any later than this line reopens exactly that
# collision; this ordering is the load-bearing part. (logger.wandb_exp_name
# at ~669 does NOT inherit the suffix -- stated, not hidden.) MUST_FIRE
# (broken to see red): delete ${RUN_SUFFIX} from this line and the banner
# below prints PROBE=1 beside an unsuffixed out= path -- the loud
# contradiction this arrangement exists to produce.
OUT_DIR=${OUT_DIR:-$A/results/g4e4b_twaiec_it_fullft_v3_${SEQ_K}k}${RUN_SUFFIX}   # STABLE dir -> auto-resume chain works; PROBE=1 detours to *_probe, out of that chain's reach
# PROBE<->OUT_DIR pairing enforced IN CODE (fix #81 finding 2): until now,
# dropping the suffix expansion from the line above produced only a
# contradictory BANNER line -- no rc, no consumer. This gate reads the
# RESOLVED knob and path, never source text (this file's own comments carry
# literal ${RUN_SUFFIX} decoys; a token check would eat them), so the
# stated MUST_FIRE is a real failure, not a described one. Both arms are
# checked (scope is symmetric): a production OUT_DIR that ends in _probe
# is the same coupling defect, other side.
if [[ "$PROBE" == 1 ]]; then
  [[ "$OUT_DIR" == *_probe ]] || \
    die "PROBE=1 but OUT_DIR=$OUT_DIR lacks the _probe suffix (#81 pairing gate)"
else
  [[ "$OUT_DIR" != *_probe ]] || \
    die "PROBE=0 but OUT_DIR=$OUT_DIR ends in _probe (#81 pairing gate)"
fi
LOG_OUT=$A/logs/g4e4b_sft_${SLURM_JOB_ID}.out
LOG_ERR=$A/logs/g4e4b_sft_${SLURM_JOB_ID}.err
mkdir -p "$OUT_DIR/checkpoints" "$A/logs"
touch "$OUT_DIR/.preflight_write_ok" || die "OUT_DIR not writable: $OUT_DIR"
rm -f "$OUT_DIR/.preflight_write_ok"

# Post-resolution re-assertion of the mode knob (fix #81 finding 1): the
# 0/1 case above guarded only the t=0 read, and readonly is no remedy --
# this file ships `set -uo pipefail` WITHOUT -e, so a failed readonly
# assignment prints and CONTINUES. A PROBE overwritten between the case and
# this line now dies LOUDLY before the banner prints, instead of surfacing
# later as an absent armed-gate echo (detection-by-absence, refused).
[[ "$PROBE" =~ ^[01]$ ]] || \
  die "PROBE='$PROBE' at banner -- post-resolution overwrite of the mode knob (#81)"

echo "============================================================"
echo " Gemma4-E4B  FULL fine-tune  (SFT-Taiwan-AIEC v3)  FoundationScale prod #1"
echo " backend=$FS_BACKEND$([[ "$FS_BACKEND" == enroot ]] && printf ' (enroot+torchrun; job-id minted locally, scontrol absent per s1)')"
echo " job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST  world=$SLURM_NTASKS"
echo " MoE=$MOE (enable_moe_block=$ENABLE_MOE_BLOCK num_experts=${NUM_EXPERTS:-n/a})"
echo " TP=$TP PP=1 CP=$CP DP=$DP ET P=$ETP EP=$EP  seq=$SEQ_LENGTH  recompute=$RECOMPUTE"
echo " rows=$ROWS gbs=$GBS mbs=$MBS epochs=$EPOCHS -> train_iters=$TRAIN_ITERS (lr_decay_iters==train_iters: anneal COMPLETES)"
echo " save_interval=$SAVE_INTERVAL (first ckpt at iter $SAVE_INTERVAL)"
echo " hf=$HF_MODEL"
echo " base_ckpt=$BASE_CKPT"
# PROBE= prints the mode as RESOLVED in the probe-mode block above
# (normalized to exactly 0 or 1 there), not a fresh environment read --
# re-reading the knob here would print the operator's intent instead of the
# script's decision, which is precisely the gap a swallowed flag hides in
# (fix #81). Under PROBE=1 the out= path on this same line ends in _probe;
# a PROBE/out= pairing that disagrees proves the suffix plumbing regressed.
echo " out=$OUT_DIR  port=$MASTER_PORT  PROBE=$PROBE"
echo "============================================================"

# ----------------------------------------------------------------------------
# Preflight gates — fail LOUDLY before burning GPU time
# ----------------------------------------------------------------------------
[[ -d "$REPO" ]] || die "repo missing $REPO"
[[ -d "$EXTRAS" ]] || die "python-extras missing $EXTRAS"
[[ -f "$SQSH" ]] || die "container missing $SQSH"
[[ -f "$REPO/scripts/training/run_recipe.py" ]] || die "run_recipe.py missing under $REPO/scripts/training"

# Recipe name. fix28 VERIFIED: registered by the estate patch landing with this
# change (recipes/gemma4_vl/gemma4_vl.py + __init__.py export). The old grep over
# $REPO/src was the RIGHT tree but the WRONG test — any occurrence of the string
# (a comment, this launcher) satisfied it. It must still fail, and it can: it
# now requires (a) a `def` under the MEASURED recipe root and (b) the package
# __init__ export — a def without an export is exactly the half-applied-patch
# state.
RECIPE=${RECIPE:-gemma4_vl_e4b_foxbrain_sft_config}
grep -RqE "def +${RECIPE}\b" "$REPO/src/megatron/bridge/recipes/" 2>/dev/null || \
  die "recipe '$RECIPE' has no def under $REPO/src/megatron/bridge/recipes/ (the measured recipe root; \$REPO/recipes/ does not exist). Candidates: grep -RhoE 'def +gemma4[a-z0-9_]*config' \$REPO/src/megatron/bridge/recipes/ | sort -u. If the E4B pair is absent: apply the fix28 estate patch first."
grep -qE "\b${RECIPE}\b" "$REPO/src/megatron/bridge/recipes/gemma4_vl/__init__.py" 2>/dev/null || \
  die "recipe '$RECIPE' defined but NOT exported from recipes/gemma4_vl/__init__.py — the fix28 estate patch is half-applied (E1 without E2)."

# Converted checkpoint gates. Trap 2 lesson: never trust rc=0 on a conversion —
# compare BYTES, not exit codes.
[[ -d "$BASE_CKPT/iter_0000000" ]] || \
  die "converted base ckpt missing: $BASE_CKPT/iter_0000000 — run the HF->Megatron conversion first (same flow as converted_ckpts/gemma-4-31B-base, job 1952; see exports/EXPORT_STATUS.md). Raw HF import at train time is not estate convention."
HF_BYTES=$(stat -c%s "$HF_MODEL/model.safetensors")
CKPT_BYTES=$(du -sb "$BASE_CKPT" | cut -f1)
# UNVERIFIED heuristic: a healthy bf16 dist-ckpt of this model should be >=~70%
# of the HF safetensors bytes (weights + dist metadata). Catches empty/partial
# conversions. Calibrate once against the first real E4B conversion.
MIN_BYTES=$(( HF_BYTES * 7 / 10 ))
(( CKPT_BYTES >= MIN_BYTES )) || die "converted ckpt looks PARTIAL: $CKPT_BYTES bytes vs HF $HF_BYTES (min accepted $MIN_BYTES). Do not trust its rc=0; re-convert."
echo "ckpt byte gate: hf=$HF_BYTES converted=$CKPT_BYTES ok"

# Gate: checkpoint.load MUST differ from checkpoint.pretrained_checkpoint,
# otherwise resume semantics silently disengage (dense-launcher trap).
_norm_load=${OUT_DIR%/}/checkpoints; _norm_pre=${BASE_CKPT%/}
[[ "$_norm_load" != "$_norm_pre" ]] || die "checkpoint.load == checkpoint.pretrained_checkpoint — resume would silently not engage"

# Gate: CoT-preserving training template must be IN THE TREE (collate.py) and
# wired to the env escape hatch. Static proof before any GPU is touched.
COLLATE_FILE=$(grep -rl --include='*.py' "_gemma4_training_chat_template" "$REPO/src" "$REPO/scripts" 2>/dev/null | head -n1)
[[ -n "$COLLATE_FILE" ]] || \
  die "in-tree CoT patch (_gemma4_training_chat_template) NOT FOUND under $REPO — without it apply_chat_template strips <|channel>thought from supervised targets and this run learns nothing. Check for a clobbered submodule (trap 6 pattern)."
grep -q "FOXBRAIN_GEMMA4_KEEP_COT" "$COLLATE_FILE" || \
  die "$COLLATE_FILE does not read FOXBRAIN_GEMMA4_KEEP_COT — escape-hatch wiring changed; inspect before running"
echo "CoT gate: patch present in $COLLATE_FILE ; FOXBRAIN_GEMMA4_KEEP_COT=$FOXBRAIN_GEMMA4_KEEP_COT"

# ----------------------------------------------------------------------------
# Provenance: emit the RunManifest WITH the declared block, before any GPU-second
# ----------------------------------------------------------------------------
# The measured estate state (first real probe run): 0/3 first-save sub-gates
# reached a verdict on the E4B base ckpt — save_complete VACUOUS because
# nothing in this estate ever wrote RunManifest.declared. This block is the
# missing producer side, and it is placed AFTER the byte gate above so the
# denominator is censused from a base the launcher has already proven
# substantial, never from a conversion we are merely hoping is complete.
#
# DENOMINATOR INDEPENDENCE (the load-bearing property): declared_fqns comes
# from the tensor census of the BASE checkpoint ($BASE_CKPT/iter_0000000) —
# produced by conversion BEFORE any training step, a physically separate tree
# from the $OUT_DIR/checkpoints this run will write. A truncated or
# expert-dropping save fails against it. The emitter REFUSES (rc=1) to derive
# from the judged dir or anything nested with it; rc=3 means the denominator
# could not even be read. Both die below: training with gates that cannot
# adjudicate is the pre-FoundationScale state this job exists to end.
# EXTRA_OVERRIDES is a documented escape hatch (line 15) appended VERBATIM to the
# recipe CLI further down — so it changes the run. Until now the manifest never
# saw it: an operator could pass model.seq_length=8192 and the recorded
# provenance would still read $SEQ_LENGTH. A manifest that disagrees with the run
# it describes is worse than no manifest, so every extra override is recorded.
#
# A key that is ALSO recorded explicitly below is REFUSED rather than merged.
# record_effective() raises on a duplicate key by design ("recording twice would
# silently overwrite the first resolution"), and the operator's intent in that
# case is to change the knob, not to shadow it — so name the dedicated variable
# and stop, instead of letting the emitter fail with a message about provenance
# internals that says nothing about which env var to set.
declare -a EXTRA_EFFECTIVE=()
_FIRST_CLASS_KEYS="recipe train.train_iters train.global_batch_size model.seq_length checkpoint.save_interval model.recompute_granularity"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  for _ov in $EXTRA_OVERRIDES; do
    [[ "$_ov" == *=* ]] || \
      die "EXTRA_OVERRIDES entry '$_ov' is not KEY=VALUE — refusing to launch with an override the manifest cannot record"
    _k=${_ov%%=*}
    for _fc in $_FIRST_CLASS_KEYS; do
      if [[ "$_k" == "$_fc" ]]; then
        die "EXTRA_OVERRIDES sets '$_k', which this launcher already records from its own variable. Set the dedicated env knob (RECIPE / TRAIN_ITERS / GBS / SEQ_LENGTH / SAVE_INTERVAL / RECOMPUTE) instead of shadowing it — two sources for one knob is precisely the divergence the manifest exists to catch."
      fi
    done
    EXTRA_EFFECTIVE+=(--effective "$_ov")
  done
  echo "provenance gate: ${#EXTRA_EFFECTIVE[@]} extra override(s) will be recorded: $EXTRA_OVERRIDES"
fi

# FS_ROOT: search the known deploy locations instead of asserting one. The old
# default ($A/FoundationScale) does not exist on this estate — the checkout is at
# $HOME/foundationscale (measured 2026-08-23) — so the die below fired on every
# launch. Candidates are tried in order and the failure names ALL of them, so an
# operator is told what was searched rather than handed one wrong path.
FS_CANDIDATES=("${FOUNDATIONSCALE_ROOT:-}" "$HOME/foundationscale" "$A/FoundationScale")
FS_ROOT=""
for _fs_c in "${FS_CANDIDATES[@]}"; do
  [[ -n "$_fs_c" && -f "$_fs_c/tools/emit_run_manifest.py" ]] || continue
  FS_ROOT=$_fs_c; break
done
[[ -n "$FS_ROOT" ]] || \
  die "tools/emit_run_manifest.py not found under any of: ${FS_CANDIDATES[*]} — set FOUNDATIONSCALE_ROOT; launching without recorded provenance is not an option for this run"
echo "provenance gate: FoundationScale checkout resolved to $FS_ROOT"
# fix35: the same checkout must also carry the ARTIFACT adjudicator. The
# hard-refusal idiom this block already runs for the emitter applies with
# equal force here: a launch that can emit declared_fqns but cannot adjudicate
# the saves those denominators describe recreates the exact pre-fix35 estate —
# provenance produced, never read, gate unreachable. Both tools live in the
# same tools/ directory of one checkout, so one-present-one-absent means a
# partial checkout, and that is a refusal, never a skipped step (doctrine 4).
[[ -f "$FS_ROOT/tools/live_save_gate.py" ]] || \
  die "tools/live_save_gate.py not found under $FS_ROOT/tools — the manifest emitter beside it resolved, so this checkout is PARTIAL. Launching unadjudicated is the pre-fix35 state this run exists to end; fix the checkout or set FOUNDATIONSCALE_ROOT to a complete one"
# fix45-A2 (a), decided and defended: the `command -v python3` guard that
# stood here asserted the WRONG plane. The emitter no longer runs on the
# host (below), so host python3's presence says nothing about whether
# emission can happen; and the two stdlib host-python writers further down
# (the file's only remaining hard host-python3 dependencies — censused with
# the gate comment below) each carry their own `|| die` naming exactly what
# failed, so an absent interpreter surfaces there as a named refusal, never
# a silent skip. No container-executor availability assertion is shipped in
# its place, by decision, and the reason is on record: the emitter
# invocation itself is the FIRST executor call this launcher makes and it
# is fail-closed (`|| die`) before any GPU-second, so a dead executor
# surfaces at this line with the executor's own stderr and a named refusal;
# a separate trivial-payload probe would pay a container start to pre-learn
# what the very next call learns anyway. What IS asserted, because a
# container-side failure would otherwise narrate itself as an emission
# failure (the misattribution class doctrine 5 exists to refuse): the
# bind-mount invariant.
#
# fix45-A2 (b): both executor arms bind-mount $HOME:$HOME byte-identically,
# so a payload path resolves in-container iff it sits under $HOME as
# spelled. The <compute-node> run worked BECAUSE $OUT_DIR is under $HOME — an
# unwritten precondition until now — and the gate edit's report/capture
# paths (under $OUT_DIR/fs_gate) rely on the same contract. $OUT_DIR is an
# env-overridable knob, FS_ROOT's candidates are convention not guarantee,
# and an override outside $HOME is exactly how this breaks silently.
# Scope, stated: the five paths the executor-routed provenance/adjudication
# calls consume (training's corpus/image paths are out of this guard's
# claim, by name, not by oversight — an exhaustive mount audit is not what
# this block asserts). Paths are compared AS SPELLED; a symlinked $HOME
# resolving elsewhere is unmeasured and named so.
for _bm in "$OUT_DIR" "$FS_ROOT" "$REPO" "$HF_MODEL" "$BASE_CKPT"; do
  case "$_bm" in
    "$HOME"|"$HOME"/*) ;;
    *) die "bind-mount invariant broken: '$_bm' is not under \$HOME='$HOME' as spelled — both executor arms mount \$HOME:\$HOME byte-identically, so this path would not resolve in-container and the failure would surface misattributed as an emission/gate error. Point the knob back under \$HOME (the measured <compute-node> run relied on exactly this)." ;;
  esac
done

# fix45-A2 / #77's emitter site: with --full-ft --base-checkpoint the
# emitter CENSUSES THE BASE DCP — it imports the torch.distributed.checkpoint
# stack (measured on <compute-node>: this exact call hard-blocked every full-FT
# launch in the estate's history, dying "torch.distributed.checkpoint is
# unavailable; cannot read DCP" before one GPU-second of training). It is
# therefore NOT one of the legitimate host-python sites — fix45-A's census
# of five named it "torch-free by design", and that mislabel is part of the
# record kept with the gate below. The discriminator is "reads a DCP",
# never "is python on the host" (measured: the host CAN import torch when
# the user site is visible). It now runs where the DCP stack lives: the
# container, with the same torch build that writes the saves. The cheap
# host alternative (~/.local's CPU-only 2.10.0+cpu, exposed by unsetting
# PYTHONNOUSERSITE) is REFUSED on record with the gate below; all three
# reasons apply here unchanged. Payload idiom mirrors fs_live_save_gate
# exactly (the measured single-quote layering: $FS_ROOT and $(printf '%q'
# ...) expand on the HOST into the payload string; ${PYTHONPATH} is
# backslash-deferred so the CONTAINER expands its own forwarded value;
# PYTHONNOUSERSITE=1 restated payload-scoped). argv is assembled as an
# ARRAY and flattened through printf '%q' so no argument is ever re-split
# or hand-re-quoted. rc hygiene: `|| die` reads the executor's own rc, no
# pipe in between. This is the invocation shape that produced the first
# successful full-FT run, adapted only in this comment.
FS_EMIT_ARGS=(
  --run-id g4e4b-tw-v3-fullft
  --out-dir "$OUT_DIR"
  --checkpoint-dir "$OUT_DIR/checkpoints"
  --job-id "$SLURM_JOB_ID"
  --nodes 1 --gpus-per-node 4 --tp "$TP" --pp 1 --cp "$CP" --dp "$DP" --ep "$EP"
  --code-root "$REPO" --entrypoint "$REPO/scripts/training/run_recipe.py"
  --env-prefix FOXBRAIN_
  --watch-env FOXBRAIN_GEMMA4_KEEP_COT --watch-env FOXBRAIN_SFT_JSONLS
  --effective "recipe=$RECIPE" --effective "train.train_iters=$TRAIN_ITERS"
  --effective "train.global_batch_size=$GBS" --effective "model.seq_length=$SEQ_LENGTH"
  --effective "checkpoint.save_interval=$SAVE_INTERVAL" --effective "model.recompute_granularity=$RECOMPUTE"
  ${EXTRA_EFFECTIVE[@]+"${EXTRA_EFFECTIVE[@]}"}
  --full-ft --base-checkpoint "$BASE_CKPT/iter_0000000" --hf-config "$HF_MODEL/config.json"
)
run_in_container --slurm-ntasks 1 --workdir "$REPO" \
  bash -lc "PYTHONPATH='$FS_ROOT/src':\${PYTHONPATH} PYTHONNOUSERSITE=1 python3 '$FS_ROOT/tools/emit_run_manifest.py' $(printf '%q ' "${FS_EMIT_ARGS[@]}")" \
  || die "run-manifest emission FAILED — the first-save gates would stay VACUOUS for want of declared denominators, which is exactly the measured estate defect; this launcher fails closed"
echo "provenance gate: run manifest emitted (declared censused from $BASE_CKPT/iter_0000000)"

# ----------------------------------------------------------------------------
# fix35 — materialize the live_save_gate inputs at SUBMIT time, before any GPU
# ----------------------------------------------------------------------------
# Two producers the estate never had, both written under $OUT_DIR/fs_gate/ and
# consumed by the watcher tripwire (d) and the epilogue below:
#  (1) resolved-train-config.json — the gate's --train-config. The gate
#      tolerates its absence ({}, "no --train-config supplied"); a tolerated
#      absence is not a produced input, and an input that exists makes the
#      --run-kind flag corroborated rather than bare. Written via python3
#      json.dump — printf-assembled JSON is one quoting slip from an
#      unparseable config, and an unparseable config would read as the gate's
#      KEY=VALUE fallback, silently changing which keys exist.
#  (2) fqn-map.json — the gate's --fqn-map: artifact-namespace declared FQNs.
#      WITHOUT this, on the estate's DCP saves the gate's HF-vs-Megatron
#      overlap check (<0.90) leaves declared_fqns=None, completeness
#      VACUOUS-blocks, the drop control is unconstructable, and EVERY first
#      save exits 1 healthy-or-not — a permanent red that would get this
#      wiring uninstalled. The map is read back verbatim from the manifest
#      attempt record this block just emitted: censused at submit time by
#      emit_run_manifest from the INDEPENDENT base ($BASE_CKPT/iter_0000000),
#      a tree the run under judgment never writes — exactly the provenance
#      --fqn-map's own contract demands. This is the first reader of the
#      declared block; the finding was that nothing ever read it. What this
#      comment no longer does is ASSERT the namespace (#78 re-scope): the
#      PYNS block below MEASURES it at submit by live_save_gate.py's own
#      overlap rule run the other way — >=0.90 of the record's FQNs found
#      in the $HF_MODEL model*.safetensors header key set means the census
#      IS the HF namespace and the launch refuses there (C1's first-save
#      BLOCK moved to submit, where the evidence lives); zero overlap is
#      the disjoint-segments shape of this estate's DCP census against HF
#      names, and proceeds; anything between abstains BY NAME and refuses.
#      So "artifact-namespace" above names the measurement's verdict, and
#      the printed gate line carries the denominators that verdict rests
#      on (R2). The materializer's own confident print line is beyond the
#      window this repair was measured against and is deliberately left
#      unedited: it can now print only over a measured denominator, and an
#      ambiguous census never reaches it (the named abstention refuses the
#      launch) — supersession stated, not hidden.
# Both writes fail closed: launch-time failure means the first save cannot be
# honestly adjudicated, and launching anyway is the defect being repaired.
FS_GATE_DIR=$OUT_DIR/fs_gate
RESOLVED_CFG=$FS_GATE_DIR/resolved-train-config.json
FQN_MAP=$FS_GATE_DIR/fqn-map.json
mkdir -p "$FS_GATE_DIR" || die "cannot create $FS_GATE_DIR — the gate reports and adjudication inputs have nowhere to land"

# ---------------------------------------------------------------------------
# #78 re-scope (R1/R2) — MEASURE the fqn-map namespace at submit; never assert it
# ---------------------------------------------------------------------------
# The fqn-map materializer further down copies declared.declared_fqns out of
# the emitter's attempt record into $FQN_MAP verbatim, and its only
# validation is non-emptiness: the strings wear whatever namespace $BASE_CKPT
# resolved to when emit_run_manifest censused it. Estate today: the
# Megatron/DCP base. But C1: if $BASE_CKPT ever resolves to an HF-layout
# tree, the map is HF names under an artifact-namespace label and the gate
# blocks at the FIRST SAVE — hours into a paid multi-node run — on evidence
# fully available HERE (doctrine 4: the cheap refusal belongs at submit).
# The block below reads the freshest attempt-*.json out of
# $OUT_DIR/checkpoints, the same directory the materializer reads its map
# out of, so the measurement covers exactly the consumer's input. The
# discriminator is NOT a new regex: it is live_save_gate.py's own overlap
# rule — two FQN sets share a namespace iff >=0.90 of one overlaps the
# other, the 0.90 the gate applies when IT decides header-vs-artifact — run
# in the other direction over that census against the HF pole exactly as
# this estate defines it: the model*.safetensors HEADER key set under
# $HF_MODEL (8-byte little-endian length + JSON: stdlib reads, no torch, no
# DCP access, so under f45's own discriminator — "reads a DCP", never "is
# python on the host" — this is a legitimate host site, enumerated as the
# fifth exception in the f45 census leg). Three-way verdict, every branch
# printing its denominators (doctrine 2):
#   ratio >= 0.90 -> the census IS the HF namespace: REFUSE at submit. C1's
#                    block moves from first save to submit, where the
#                    evidence lives — the measured cheap refusal.
#   ratio == 0    -> the disjoint-leading-segments shape this estate's DCP
#                    census has against HF names: consistent with the
#                    artifact namespace the --fqn-map contract requires.
#                    Print the MEASUREMENT, then let the materializer run.
#   anything else -> genuinely ambiguous (a partially-converted base tree):
#                    ABSTAIN BY NAME and refuse — the confident materializer
#                    line must never print over an abstention, so abstaining
#                    here means not launching.
# Missing/unreadable record, missing/unreadable header, or an empty key set
# on either side is UNMEASURED, never empty (doctrine 4: missing is not
# zero, unreadable is not empty): refuse. The invocation is ONE line with
# the heredoc opener so the contract leg's sed-extract of this body drops
# exactly one header line (a line-continued opener would leave the || die
# line inside the extracted python).
python3 - "$HF_MODEL" "$OUT_DIR/checkpoints" <<'PYNS' || die "fqn-map namespace gate: the map's namespace could not be ESTABLISHED as the artifact namespace at submit (see stderr for the measured basis — established-HF refusal, named abstention, or an unreadable/unmeasurable input; doctrines 1/4) — refusing to launch so the materializer's artifact-namespace line below never prints over an unmeasured denominator"
import glob
import json
import os
import struct
import sys

hf_root, ckpt_dir = sys.argv[1], sys.argv[2]


def refuse(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


records = sorted(glob.glob(os.path.join(ckpt_dir, "attempt-*.json")))
if not records:
    refuse(
        f"fqn-map namespace UNMEASURED: no attempt-*.json record in "
        f"{ckpt_dir} — the materializer below reads its map out of this "
        "same directory, so absence is a broken emission (doctrine 4: "
        "missing is not zero), not an empty namespace"
    )
try:
    rec = max(records, key=lambda p: (os.path.getmtime(p), p))
except OSError as exc:
    refuse(
        f"fqn-map namespace UNMEASURED: attempt record stat failed under "
        f"{ckpt_dir} ({exc!r}) — unreadable is not empty (doctrine 4)"
    )
try:
    with open(rec, encoding="utf-8") as fh:
        attempt = json.load(fh)
except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
    refuse(
        f"fqn-map namespace UNMEASURED: attempt record {rec} unreadable "
        f"({exc!r}) — unreadable is not empty (doctrine 4)"
    )
declared = attempt.get("declared") if isinstance(attempt, dict) else None
fqns = declared.get("declared_fqns") if isinstance(declared, dict) else None
if (
    not isinstance(fqns, list)
    or not fqns
    or any(not isinstance(f, str) or not f for f in fqns)
):
    refuse(
        f"fqn-map namespace UNMEASURED: declared.declared_fqns in {rec} "
        "is absent, empty, or holds non-string entries — a census over "
        "zero units is UNMEASURED (doctrine 1), never a denominator"
    )
shards = sorted(
    glob.glob(os.path.join(hf_root, "**", "model*.safetensors"), recursive=True)
)
hf_keys = set()
for shard in shards:
    try:
        with open(shard, "rb") as fh:
            (hdr_len,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(hdr_len).decode("utf-8"))
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        refuse(
            f"fqn-map namespace UNMEASURED: HF safetensors header {shard} "
            f"unreadable ({exc!r}) — the HF pole of the discriminator is "
            "unmeasured; unreadable is not empty (doctrine 4)"
        )
    hf_keys.update(k for k in hdr if k != "__metadata__")
if not shards or not hf_keys:
    refuse(
        f"fqn-map namespace UNMEASURED: {len(shards)} model*.safetensors "
        f"under {hf_root} yielded {len(hf_keys)} header keys — an empty "
        "reference cannot discriminate a namespace (doctrine 1)"
    )
hits = sum(1 for f in fqns if f in hf_keys)
n = len(fqns)
ratio = hits / n
basis = (
    f"{hits}/{n} declared FQNs from {rec} overlap the {len(hf_keys)}-key "
    f"HF safetensors header set ({len(shards)} model*.safetensors under "
    f"{hf_root}; ratio {ratio:.3f}; the 0.90 same-namespace threshold is "
    "live_save_gate.py's own overlap rule, run here in the other direction)"
)
if ratio >= 0.90:
    refuse(
        f"FQN-MAP NAMESPACE REFUSED at submit: {basis} — the census "
        "already IS the HF namespace: $BASE_CKPT resolved to an HF-layout "
        "tree and fqn-map.json would be HF names wearing the --fqn-map "
        "artifact-namespace label, blocking at the FIRST SAVE hours into a "
        "paid run (C1). Point $BASE_CKPT at the DCP conversion, re-emit, "
        "resubmit."
    )
if hits:
    refuse(
        f"FQN-MAP NAMESPACE ABSTENTION: {basis} — neither 0 nor >=0.90, "
        "so the namespace cannot be established from measurement: a mixed "
        "census means a partially-converted base tree. This launcher "
        "abstains BY NAME and refuses to certify what it has not measured; "
        "the confident materializer line below must never print over an "
        "abstention, so abstaining here means not launching."
    )
print(
    f"fqn-map namespace measured: {basis} — zero overlap is the disjoint-"
    "namespace shape this estate's DCP census must have against the HF "
    "reference; the materializer below reads this same directory and "
    f"copies the freshest attempt record's census verbatim, so these {n} "
    "FQNs are the map's namespace is measured over — and this line, not "
    "the copy, is the measurement"
)
PYNS
python3 - "$RESOLVED_CFG" "$RECIPE" "$TRAIN_ITERS" "$GBS" "$SEQ_LENGTH" "$SAVE_INTERVAL" "$RECOMPUTE" "$HF_MODEL" "$BASE_CKPT" <<'PY' || die "resolved-train-config write failed — the gate would run on a tolerated absence again; refusing (doctrine 4)"
import json, sys
out, recipe, iters, gbs, seq, save_iv, recompute, hf, base = sys.argv[1:10]
doc = {
    # run_kind is stated affirmatively so even an --run-kind auto invocation
    # resolves from a recorded source instead of artifact-marker inference.
    "run_kind": "full",
    "recipe": recipe,
    "train.train_iters": int(iters),
    "train.global_batch_size": int(gbs),
    "model.seq_length": int(seq),
    "checkpoint.save_interval": int(save_iv),
    "model.recompute_granularity": recompute,
    "hf_model": hf,
    "base_checkpoint": base,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
print(f"resolved train config: {out} ({len(doc)} keys, run_kind='full' stated; for a full run the gate consults it only for kind corroboration — no lora rank/target derivation applies)")
PY
python3 - "$OUT_DIR/checkpoints" "$FQN_MAP" <<'PY' || die "fqn-map materialization failed — the declared denominator cannot reach the adjudicator; launching now would be another unadjudicated run (the fix35 defect). Refusing."
import glob, json, os, sys
ckpts, out = sys.argv[1], sys.argv[2]
cands = sorted(glob.glob(os.path.join(ckpts, "attempt-*.json")))
if not cands:
    raise SystemExit(
        "no attempt-*.json manifest record beside the future saves — the "
        "emitter's discovery link is missing seconds after a successful "
        "emission; the provenance plane is broken underneath this launch")
rec = json.load(open(cands[-1], encoding="utf-8"))
fqns = (rec.get("declared") or {}).get("declared_fqns") or []
if not fqns:
    raise SystemExit(
        f"{cands[-1]} carries an empty/absent declared.declared_fqns for a "
        f"--full-ft emission — that combination is refused at emission by "
        f"design, so seeing it means the record was tampered with or the "
        f"emitter silently changed contract; 0 FQNs cannot adjudicate "
        f"anything (doctrine 1)")
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"declared_fqns": fqns}, fh)
    fh.write("\n")
print(f"fqn-map materialized: {len(fqns)} artifact-namespace declared FQNs read back verbatim from {cands[-1]} (censused from the INDEPENDENT base at submit time; the run under judgment never wrote any of it)")
PY

# Disk: E4B ckpt w/ optim est. ~90-110 GB (26B was ~290 GB/ckpt -> ~35%). # UNVERIFIED actual size after first save.
FREE_BYTES=$(df -B1 --output=avail "$OUT_DIR" | tail -n1 | tr -d ' ')
(( FREE_BYTES >= 120000000000 )) || die "<120 GB free under $OUT_DIR — cannot guarantee ONE checkpoint"
(( FREE_BYTES >= 2500000000000 )) || echo "WARN: <2.5 TB free — ~$((FREE_BYTES/100000000000)) ckpts would fit; save_interval=$SAVE_INTERVAL over $TRAIN_ITERS iters implies $((TRAIN_ITERS/SAVE_INTERVAL+1)) saves"

# Dynamic probe (1 CPU task, in-container): tokenizer loads; trap is present in
# the stock template; the two-replacement patch mechanics keep the CoT in the
# render of a REAL think row. Fails the job if the render-under-patch drops CoT.
# fix #81: this path variable MUST NOT be named PROBE. PROBE is the
# operator-facing mode knob (PROBE=1 -> 20-iter probe run, resolved above);
# the old line bound a path to the same name and silently overwrote the
# operator's `1` with a path before anything read it as a mode -- that
# overwrite WAS the finding, turning `PROBE=1 sbatch` into a ~2016-iter
# production run. COT_PROBE_PY makes the collision structurally impossible,
# not just unlikely: after this patch every remaining PROBE use in this file
# names the mode knob exactly (header doc, mode block, banner, G5 gate), so
# no corner is left for a second meaning to hide in. Renaming this back is
# how that defect returns -- one name, one meaning.
COT_PROBE_PY=$OUT_DIR/preflight_cot_probe.py
cat > "$COT_PROBE_PY" <<'PY'
import json, os, sys
hf = os.environ["HF_MODEL"]
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
except Exception as e:
    print("FATAL: tokenizer failed to load:", e); sys.exit(1)
n = len(tok("hello FoundationScale", add_special_tokens=False).input_ids)
assert n > 0, "tokenizer produced 0 tokens"
print(f"tokenizer ok (sample len={n})")
tmpl = getattr(tok, "chat_template", None)
assert tmpl, "no chat_template on an -it model?!"
fx = os.environ["FOXBRAIN_SFT_JSONLS"].split(",")[0]     # foxbrain_identification.jsonl
row = None
for line in open(fx):
    if "<|channel>thought" in line:
        row = json.loads(line); break
assert row, "no think-format row found in foxbrain_identification.jsonl?!"
msgs = [{"role": ("user" if c["from"] == "human" else "assistant"), "content": c["value"]} for c in row["conversations"]]
reasoning = row["conversations"][1]["value"].split("<|channel>thought", 1)[1]
snippet = reasoning.strip()[:80]
assert len(snippet) >= 40, "reasoning snippet implausibly short"
stock = tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False)
if "strip_thinking" in tmpl:
    patched = tmpl.replace("strip_thinking(message['content'])", "message['content']") \
                  .replace("strip_thinking(item['text'])", "item['text']")
    assert patched != tmpl, "FATAL: patch replacements are a NO-OP on this template string (upstream drift)"
    fixed = tok.apply_chat_template(msgs, chat_template=patched, tokenize=False)
    print("trap confirmed present in stock render (CoT stripped):", snippet not in stock)
    print("patched render keeps CoT:", snippet in fixed, f"({len(stock)} -> {len(fixed)} chars)")
    if snippet not in fixed:
        print("FATAL: even under the patched template, the CoT does not survive the render"); sys.exit(1)
else:
    print("WARN: stock template has no strip_thinking — trap not visible; in-tree grep gate + tripwire remain the guards")  # UNVERIFIED: E4B-it template assumed identical in kind to 26b-it
print("PREFLIGHT-PROBE-PASS")
PY
# Same interpreter stack as training (see the module-dump comment in the LoRA
# launcher); --slurm-ntasks 1 preserves the historic single-task probe exactly.
# $COT_PROBE_PY crosses into the container as DATA, not as inner-shell SOURCE:
# 'python3 "$1"' is a fixed literal for the inner bash and the path rides along
# as positional $1, so an OUT_DIR containing whitespace stays one word and one
# containing $(...) text is never executed by the inner shell (BLOCKER 2). The
# :513/:1002 nesting idiom ('$VAR' inside a double-quoted source) fixes the
# split and substitution but still splices the path INTO source text -- a
# single quote in the path reopens the injection; passing arguments is the
# only form that treats the path as data. The BASH-LC standing leg in
# .github/workflows/ci.yml now enforces the boundary; do not "simplify" this
# back to a splice.
run_in_container --slurm-ntasks 1 --workdir "$REPO" \
     bash -lc 'python3 "$1"' _ "$COT_PROBE_PY" || die "preflight tokenizer/CoT probe FAILED — job stopped before any training GPU-seconds"

# ----------------------------------------------------------------------------
# G5-equivalent post-run save-path gate (fix #81; mirrors the LoRA launcher's
# own post-run save-path gate)
# ----------------------------------------------------------------------------
# Probe mode without this gate burns 20 iterations and asserts NOTHING -- an
# unmeasured run (doctrine 1), the same defect the finding was filed against.
# Written as an EXIT trap installed HERE, ahead of every later exit path
# (die() everywhere below, the resume-complete `exit 0`, the training rc
# passthrough, later post-run gate calls): a linear insertion at the script
# tail could be orphaned by any future early exit, and an install point
# AFTER the resume-complete exit would silently stop covering it -- on that
# path a second `PROBE=1 sbatch` costs zero GPU-seconds and this gate
# re-measures the EXISTING saves instead of trusting them. The gate
# adjudicates ONLY a run that claims success: it captures the pending rc
# first and passes a nonzero rc straight through, so a crashed or
# preflight-red run keeps its own single clean error message and never gets
# green check text printed beside its red status (scope is symmetric: a
# false alarm costs what a false green costs).
# HAZARD (stated, not hidden): bash keeps exactly ONE EXIT trap. If any
# later section installs its own, it MUST chain fs_probe_save_gate when
# PROBE=1 -- the armed-banner echo below is what an operator greps the log
# for to notice the gate went missing; a detector that never RUNS is not a
# control. Armed ONLY under PROBE=1: production runs keep a byte-identical
# trap state and never adjudicate probe evidence.
fs_probe_save_gate() {
  local rc=$?    # the status the script was ALREADY exiting with; captured first, before any test below clobbers it
  # Re-assert the knob's 0/1 shape BEFORE the scope lock (fix #81 finding
  # 1): a PROBE re-assigned to a path AFTER arming would fail the scope lock
  # and exit with rc -- a green run would pass green, a red run would pass
  # its rc, and either way the corruption would fail NOTHING. That silent
  # disarm is exactly what this gate exists to refuse; the knob dies loudly.
  [[ "$PROBE" =~ ^[01]$ ]] || \
    die "PROBE='$PROBE' inside fs_probe_save_gate -- mode knob corrupted after arming (#81)"
  # Scope lock: even if someone later re-arms this trap on a production
  # path, a non-probe run must never adjudicate probe evidence.
  [[ "${PROBE:-0}" == "1" ]] || exit "$rc"
  # A red run never reaches the checks: rc!=0 means something upstream
  # already failed, and inspecting the save dir could only double-report --
  # or print green text beside a red run when the iter-10 save happened to
  # land. The original failure passes through with its rc untouched.
  (( rc == 0 )) || exit "$rc"
  local ckpt_dir=$OUT_DIR/checkpoints
  local latest=$ckpt_dir/latest_checkpointed_iteration.txt
  [[ -f "$latest" ]] || die "PROBE=1 reached rc 0 but $latest is missing -- no save ever landed; the probe run examined 0 save artifacts, and zero units is UNMEASURED, never PASS"
  local raw; raw=$(tr -dc '0-9' < "$latest")
  [[ -n "$raw" ]] || die "PROBE=1: $latest carries no iteration digits -- unreadable is not zero"
  local last=$((10#$raw))    # 10# pins base: a zero-padded write would otherwise parse octal and report nonsense like 16!=20
  (( last == TRAIN_ITERS )) || die "PROBE=1 last_iter=$last != TRAIN_ITERS=$TRAIN_ITERS in $latest -- saved short of the probe budget; not measured"
  local nd; nd=$(ls -d "$ckpt_dir"/iter_* 2>/dev/null | wc -l | tr -d ' ')
  (( nd >= 2 )) || die "PROBE=1 expected >=2 iter_* dirs under $ckpt_dir (forced saves at iters 10 and 20), found $nd"
  echo "PROBE G5 PASS: latest_checkpointed_iteration=$last == TRAIN_ITERS=$TRAIN_ITERS, $nd iter_* dirs under $ckpt_dir -- save path measured end-to-end on $TRAIN_ITERS iters"
  exit 0
}
# Fail-closed ARM decision (fix #81 finding 1): after the mode block PROBE
# is exactly 0 or 1, so any other value AT THIS LINE is a post-resolution
# re-assignment (the COT_PROBE_PY collision shape, which lands after the
# banner re-check above and before this arm -- the gate-body check above
# never runs if the trap is never installed, so this site needs its own).
# Arming nothing on a corrupted knob was an absent echo, not a failure:
# detection-by-absence, refused.
[[ "$PROBE" =~ ^[01]$ ]] || \
  die "PROBE='$PROBE' at G5-gate arming -- mode knob overwritten after resolution (#81)"
if [[ "$PROBE" == "1" ]]; then
  trap fs_probe_save_gate EXIT
  # The echo below is the gate's proof-of-RUN marker in the log: no banner,
  # no armed gate.
  # MUST_FIRE (broken to see red): set TRAIN_ITERS=21 in the probe block
  #   above -- an honest probe dies here with last_iter=20 != 21; or delete
  #   latest_checkpointed_iteration.txt after a probe save and the missing
  #   file goes red on its own line.
  # MUST_PASS: an honest PROBE=1 run -- saves at 10 and 20 on disk, latest
  #   file reading 20, PASS line printed, exit 0 preserved.
  # The OBSERVED red/green evidence for both legs belongs to the #81
  # contract suite (a separate task); until it lands this gate is
  # specified, not witnessed -- sign no green off this file alone.
  echo "PROBE=1: G5 save-path gate armed (EXIT trap) -- will require $OUT_DIR/checkpoints/latest_checkpointed_iteration.txt == $TRAIN_ITERS and >=2 iter_* dirs"
fi

# ----------------------------------------------------------------------------
# Environment (estate conventions, smoke runner + dense launcher)
# ----------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PYTHONPATH=$EXTRAS:$REPO/src:$REPO/3rdparty/Megatron-LM
export HF_HOME=$CLUSTER_HOME/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=1      # gemma4 full-attn head_dim 512 needs cuDNN fused attn
export NVTE_CPU_OFFLOAD_V1=1                       # TE>=2.10 requires it set even when off
# s8b (measured on the off-Slurm run that worked): the bond0 pins keep NCCL
# off the unroutable IPoIB address, and NCCL_IB_HCA=mlx5 PREFIX-matches all
# eight mlx5_* devices and BREAKS NCCL. This single line predates that
# measurement. The author's intent — pin NCCL's interface choice — is served
# on both arms; only the enroot arm drops the variable s8b measured harmful.
# The slurm/pyxis arm keeps its historical environment byte-identical.
if [[ "$FS_BACKEND" == slurm ]]; then
  export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5
else
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
  unset NCCL_IB_HCA   # s8b; already unset by fs_backend_init, restated here so a later edit to this block cannot silently reintroduce it
fi
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT
# enroot arm comes in with MASTER_ADDR pre-set to 127.0.0.1 (single-tray
# rendezvous that resolves inside the container no matter what /etc/hosts
# holds); sbatch keeps the historical scontrol derivation. scontrol is
# measured ABSENT off-Slurm (s1); the `|| true` keeps a missing scontrol a
# fallback-to-hostname, not a pipefail failure, under set -u -o pipefail.
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n1 || true)
  export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
fi
unset WANDB_MODE                                                  # dense-launcher W&B handling
# fs204: the secrets file is an OPERATOR-SITE convention, not a framework fact. This
# was hard-coded to one person's filename in $HOME, which did two bad things at once:
# it published a personal credential path from a public repository, and on every other
# machine the test silently failed so the launcher looked configured while sourcing
# nothing. Now it is a knob with NO default -- a training framework must not invent a
# path it will read credentials from. Unset means source nothing, which is correct.
if [[ -n "${FS_SECRETS_FILE:-}" ]]; then
  if [[ -f "${FS_SECRETS_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${FS_SECRETS_FILE}" >/dev/null 2>&1 || true
  else
    echo "WARN: FS_SECRETS_FILE=${FS_SECRETS_FILE} is set but not a file; sourcing nothing" >&2
  fi
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then echo "WARN: no WANDB_API_KEY -> WANDB_MODE=offline (wandb sync later)"; export WANDB_MODE=offline; fi
export WANDB_API_KEY="${WANDB_API_KEY:-}" WANDB_PROJECT=FoxBrain-Gemma4-E4B-SFT
export WANDB_DIR=$OUT_DIR/wandb WANDB_SILENT=true

# Resume detection (dense-launcher semantics): ckpt present -> reload optim+RNG;
# never silently restart; exit 0 when already complete. Trap 8: size any resume
# chain from steady-state timings, never iteration-10 numbers.
LATEST_ITER_FILE=$OUT_DIR/checkpoints/latest_checkpointed_iteration.txt
if [[ -f "$LATEST_ITER_FILE" ]]; then
  LAST_ITER=$(tr -dc '0-9' < "$LATEST_ITER_FILE")
  LOAD_OPTIM=True; LOAD_RNG=True
  echo "RESUME: ckpt at iter ${LAST_ITER:-?} -> load_optim=True load_rng=True"
  if [[ -n "$LAST_ITER" ]] && (( LAST_ITER >= TRAIN_ITERS )); then
    echo "RESUME: iter $LAST_ITER >= train_iters $TRAIN_ITERS -- complete, exiting 0"; exit 0
  fi
else
  LOAD_OPTIM=False; LOAD_RNG=False
  [[ "${RESUME:-0}" == "1" ]] && die "RESUME=1 but no checkpoint under $OUT_DIR/checkpoints — refusing to restart from scratch"
  echo "FRESH: base weights from $BASE_CKPT"
fi

# LR is left to the recipe default.
# fix28 VERIFIED: gemma4_vl_e4b_foxbrain_sft_config (estate patch landing with
# this change) selects max_lr 5e-6 exactly when peft is None — the same band,
# selected the same way (absence of --peft_scheme), as the 26B/31B foxbrain
# configs. NOTE below the overrides: NO recompute override is emitted for E4B
# (fix28 A2) — RECOMPUTE is pinned to "none" by the case-guard above and the
# recipe pins the knobs themselves; the old `model.recompute_granularity=
# $RECOMPUTE` line passed the literal 'selective' here, i.e. it re-armed the
# PLE no-gradient defect on every launch of this file until now.

OVERRIDES="\
    checkpoint.pretrained_checkpoint=$BASE_CKPT \
    checkpoint.load=$OUT_DIR/checkpoints \
    checkpoint.save=$OUT_DIR/checkpoints \
    checkpoint.save_interval=$SAVE_INTERVAL \
    checkpoint.fully_parallel_save=True \
    checkpoint.load_optim=$LOAD_OPTIM \
    checkpoint.load_rng=$LOAD_RNG \
    logger.tensorboard_dir=$OUT_DIR/tb_logs \
    logger.wandb_project=FoxBrain-Gemma4-E4B-SFT \
    logger.wandb_exp_name=g4e4b_it_fullft_v3_${SEQ_K}k \
    logger.log_interval=10 \
    model.seq_length=$SEQ_LENGTH \
    dataset.seq_length=$SEQ_LENGTH \
    model.tensor_model_parallel_size=$TP \
    model.context_parallel_size=$CP \
    model.sequence_parallel=False \
    $MOE_OVERRIDES \
    train.train_iters=$TRAIN_ITERS \
    scheduler.lr_decay_iters=$TRAIN_ITERS \
    train.global_batch_size=$GBS \
    train.micro_batch_size=$MBS \
    dataset.num_workers=8 \
    dataset.trust_remote_code=True \
    dataset.pack_sequences_in_batch=False \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    ${EXTRA_OVERRIDES:-}"

# One training-command definition, two executors, as in the LoRA launcher:
# sbatch -> python3 once per allocation task (4 ranks); enroot -> a single
# torchrun --nproc_per_node=$SLURM_NTASKS inside one enroot start.
LAUNCH_PY=$(fs_launch_python "${SLURM_NTASKS}")

CMD="cd $REPO && $LAUNCH_PY scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --hf_path $HF_MODEL \
    --step_func vlm_step \
    $OVERRIDES"

# ---------------------------------------------------------------------------
# fix35 — tools/live_save_gate.py, WIRED. One invocation shape, two call
# sites: the watcher tripwire (d) at first save, and the epilogue on the
# final artifact. Both go through fs_live_save_gate so the event/flags can
# never drift between the two adjudications of the same run. The gate's exit
# contract (verified against tools/live_save_gate.py: EXIT_CLEAR=0 /
# EXIT_BLOCKED=1 / EXIT_UNMEASURED=3; 2 is argparse's, i.e. wiring misuse):
#   0 CLEAR       — every applicable gate reached a real verdict over
#                   non-vacuous coverage on THIS artifact; consumed as such.
#   1 BLOCKED     — a measured blocking verdict or an unproven detector on
#                   real content. Live: kills training (the 87.5% class).
#                   Epilogue: the run exits 44 and the artifacts are poison.
#   3 UNMEASURED  — the TOOL could not measure (or crashed); never a
#                   checkpoint verdict. Live: too important to kill a healthy
#                   run over a broken verifier — the live leg disarms loudly
#                   and the epilogue remains authoritative, where a 3 exits
#                   the run 45 (unverified is never laundered into green).
#   other         — off-contract (2=argparse/wiring bug, 127=tool vanished
#                   mid-run, a bare crash, 124=the launcher's own wall-clock
#                   bound fired — fix45; the tool has no 124 and never mints
#                   one, it comes only from fs_live_save_gate's watchdog):
#                   epilogue exit 46.
# fix45-A2 / #77-B1: TWO falsified texts are kept on record here, each next
# to the measurement that killed it — the estate keeps its refuted
# arguments, corrected. First the JUSTIFICATION paragraph, verbatim as it
# shipped:
#   "Host-side invocation, deliberately NOT run_in_container: read_metadata
#   is torch-free by design and this is the same login-node verification
#   plane + PYTHONPATH="$FS_ROOT/src" idiom the manifest emitter above
#   already runs under; a container-side gate would add a second
#   interpreter question to a tool whose inputs are all host-mounted
#   files."
# "read_metadata is torch-free by design" is true for the SAFETENSORS
# HEADER path and FALSE for the DCP path — and a training save is a DCP.
# That much of fix45-A survived. What did NOT survive is the causal story
# built on it, so the DIAGNOSIS is preserved next to the table that refuted
# it. fix45-A said the host arm fails because the host has no DCP-capable
# interpreter. Measured on <compute-node>, 2026-08-24 (host python3 = anaconda3
# Python 3.12.2; user site ~/.local/lib/python3.12/site-packages):
#   bare                                     rc=0  torch 2.10.0+cpu @ ~/.local/...
#   PYTHONNOUSERSITE=1 only                  rc=3  ModuleNotFoundError: No module named 'torch'
#   PYTHONPATH=launcher only                 rc=0  torch 2.10.0+cpu @ ~/.local/...
#   BOTH (the launcher's env at the gate)    rc=3  ModuleNotFoundError
# torch IS installed on the host (2.10.0+cpu, in the ~/.local user site);
# the discriminator is PYTHONNOUSERSITE=1 — the single `export
# PYTHONNOUSERSITE=1` in the Environment section below, correct and
# load-bearing for the training payload — which hides the user site from
# every host-side gate call that follows. It was caught by a MUST_FIRE that
# refused to fire: the gate's host arm, run bare against the real 99 GB
# iter_0000020 save, CLEARED (rc 0, a full report, drop control fired) —
# the arm meant to confirm the diagnosis refuted it. Three arms, one
# extracted function, one artifact:
#   A  host + launcher env (NOUSERSITE=1)  rc=3  report=ABSENT
#      "torch.distributed.checkpoint is unavailable; cannot read DCP"
#   B  host - NOUSERSITE                   rc=0  report=CLEAR
#   C  container (the fix, below)          rc=0  report=CLEAR
# The corrected claim: the host python3 CAN parse the training save; under
# the launcher's own environment it must NOT be asked to. The fix is
# unchanged in mechanism (executor-routed, captured, wall-clock-bounded —
# the function below) and stronger in reason. The cheap alternative the new
# diagnosis invites — unset PYTHONNOUSERSITE around the gate — is REFUSED
# here, on record, for three load-bearing reasons (each applies unchanged
# to the emitter's container routing above):
#   1. it adjudicates a CUDA-written checkpoint with a CPU-only 2.10.0
#      wheel that has no relationship to the 2.11.0a0 nv26.02 stack that
#      wrote it — certifying an artifact with an instrument never
#      certified against it is the failure this estate exists to prevent;
#   2. ~/.local is UNMANAGED — nothing declares it, nothing pins it; a
#      `pip uninstall torch` in a user's home silently flips every future
#      gate from CLEAR to UNMEASURED with no code change and no diff to
#      point at;
#   3. PYTHONNOUSERSITE=1 exists precisely to keep user-site packages out
#      of the training stack; re-enabling it for the ADJUDICATOR
#      reintroduces the contamination it exists to prevent, in the one
#      component whose whole job is to be trustworthy.
# Consequence of the defect as it shipped, on every full-FT run: the gate
# returned 3 with the arm-A shape (the <compute-node> run reproduced exactly that
# in situ: two adjudications, rc 3, report ABSENT both times), live
# tripwire (d) disarmed at its first invocation having examined nothing,
# the run trained unwatched by the one tripwire that can see the
# founding-incident class, and the epilogue then failed the job with the
# cause narrated as a guess. The protection never ran and the job still
# failed at the end, misattributed.
# What legitimately REMAINS host python in this file, enumerated so the
# contract suite can census it (fix44's complement-census idiom, mirrored).
# The discriminator is NOT "is python on the host" — it is "reads a DCP":
#   cfg_get(1) — reads config.json text; torch-free; runs while the
#     container is not yet the question, and part of deciding whether to
#     launch at all.
#   the JSON schema spot-check(1) — data-file text; torch-free; its WARN on
#     python3-absence is a TOLERATED ABSENCE, marked UNVERIFIED at its own
#     site and now counted here rather than forgotten.
#   the resolved-train-config writer(1) — json.dump of the resolved knobs;
#     stdlib only (json/sys), correctly on the host, and it MUST NOT move:
#     relocating it would pay container-start latency for a block with no
#     reason to be there.
#   the fqn-map materializer(1) — reads the manifest attempt record and
#     writes the gate's map; stdlib only (glob/json/os/sys); same ruling,
#     same reason it stays.
# FOUR sites, each host-sufficient because its inputs are plain-text files
# on the host plane and it imports only stdlib. The file's OTHER two python
# calls are the torch-importing pair, and both run in the container: the
# manifest emitter (with --full-ft it censuses the base DCP — the site that
# hard-blocked every full-FT launch in the estate's history; fix45-A
# wrongly enumerated it here as "torch-free by design", and that mislabel
# is part of the record this block keeps) and this gate. No sentence in
# this block asserts that the host interpreter cannot parse the artifact —
# it can (arm B, measured); ROUTING is the pinned property, because an
# incapacity claim here would be a dead control reading like a live one.
# ASYMMETRY DISCLOSURE (sibling honesty, required): the LoRA launcher runs
# this same gate POST-RUN ONLY — it trains synchronously with no watcher, and
# its lora adjudication exits 3 BY DESIGN until the adapter prefix is
# measured from a PROBE save; that difference is stated in its own text too.
# ---------------------------------------------------------------------------
fs_live_save_gate() { # $1=iter dir  $2=event(save|first_save)  $3=report path  $4=capture log; returns the gate's rc UNTOUCHED
  # Executor-routed (fix45 / #77-B1, porting fix44's LoRA-side repair; the
  # measured host-env refusal — arm A — and the <compute-node> correction table
  # are quoted in the comment block above; the routing stands on arm C and
  # the refusal of the cheap host alternative, not on any incapacity).
  # Invocation shape copied from the two measured precedents:
  # --slurm-ntasks 1 (a single-CPU adjudicator, like the env probe and the
  # census/replay probes), --workdir "$REPO" (inert on the enroot arm,
  # which has no workdir flag). Inside the payload the forwarded PYTHONPATH
  # (EXTRAS:$REPO/src:$REPO/3rdparty/Megatron-LM) keeps its order and
  # $FS_ROOT/src is prepended — the exact layering the retired host call
  # had. The single quotes belong to the CONTAINER bash (fix39's measured
  # idiom): $FS_ROOT, $HF_MODEL, $RESOLVED_CFG, $FQN_MAP and the positional
  # args expand on the HOST into the payload string — all of them live
  # under $HOME ($HF_MODEL under .../pretraining_weights, the fs_gate dir
  # under $OUT_DIR, every FS_ROOT candidate under $HOME or $A), and both
  # executor arms bind-mount $HOME:$HOME, so the byte-identical paths
  # resolve in-container (the bind-mount invariant is now asserted at
  # launch, at the emitter call site, fix45-A2 (b)) — while ${PYTHONPATH}
  # is backslash-deferred so the CONTAINER expands its own forwarded value.
  # A FOUNDATIONSCALE_ROOT outside the mounted tree makes python3 exit 2,
  # landing on the off-contract arms (live: disarm; epilogue: 46): fail
  # closed, never a silent host fallback. PYTHONNOUSERSITE=1 is already
  # exported and forwarded (s7); restating it payload-scoped keeps the
  # load-bearing property true even if a future edit drops the export.
  # rc hygiene: the capture is a REDIRECT, not a pipe — no process stands
  # between the gate and its exit status (the #72 torchrun-laundering class
  # is structurally absent); the rc that survives is the executor's own,
  # bit-exact (fix41 measured run_in_container does not launder).
  # WALL-CLOCK BOUND, and why it exists HERE and not in the LoRA sibling:
  # this function runs INSIDE the tripwire loop, and a wedged gate would
  # never return, so tripwires (a)-(c) would silently never check again —
  # a loss of protection invisible in any log. timeout(1) cannot do the
  # job (it execs a FILE; run_in_container is a shell function), so a
  # watchdog subshell does it: at the deadline it first APPENDS a marker
  # line to the capture (a token the tool itself can never emit — it
  # exists only in this file), then TERM, then KILL. Ordering is
  # load-bearing: append before any kill, so a marker found after the wait
  # means the budget genuinely fired; the minted rc is 124, deliberately
  # OUTSIDE the tool's 0/1/3 contract so every caller's off-contract arm
  # names it. FS_GATE_TIMEOUT_S (default 600) is generous against the
  # measured cost of `import torch` plus a DCP-metadata read (seconds to
  # tens of seconds in this estate's probes); it is a budget, not a tuned
  # edge. An invalid knob refuses (running UNBOUNDED mid-run re-opens the
  # watcher-stall hole this watchdog exists to close).
  # CONCURRENCY, stated honestly (fix45 Task-A3.4): on the enroot arm this
  # call runs a SECOND `enroot start` into the live training container
  # while training holds the GPUs. That concurrency is UNMEASURED on this
  # estate (0 observed concurrent starts — the <compute-node> scratch run shipped
  # only the emitter and #82 deltas, so its two rc-3 adjudications ran the
  # OLD host arm; this live routed call has still never been observed). It
  # cannot create a second container (same $ENROOT_NAME — enroot start
  # attaches), so RIC_ACTIVE_CONTAINER accounting and fs_hard_stop_training's
  # refusal to act without it are untouched. The failure direction is
  # bounded and loud BY CONSTRUCTION: the watcher maps any off-contract
  # outcome (timeout, 2, 127, any non-0/1, even an uncorroborated 0) to
  # disarm-with-narration — never a kill of training, never a pass — and
  # the epilogue re-adjudicates authoritatively post-run, where no
  # training container is live. The measurement that settles it is the
  # next run's own first-save capture; disarm-with-narration is the
  # designed posture until that measurement exists.
  local fs_gate_rc=0 fs_gate_pid fs_watch_pid
  if [[ ! "${FS_GATE_TIMEOUT_S:-600}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'FS_GATE_TIMEOUT_S=%s is not a positive integer — the wall-clock bound could not be applied; refusing to run the gate UNBOUNDED inside the tripwire loop and minting 124 (the budget is the load-bearing property, not the invocation)\n' "${FS_GATE_TIMEOUT_S:-600}" >>"$4" 2>/dev/null || true
    return 124
  fi
  run_in_container --slurm-ntasks 1 --workdir "$REPO" \
    bash -lc "PYTHONPATH='$FS_ROOT/src':\${PYTHONPATH} PYTHONNOUSERSITE=1 python3 '$FS_ROOT/tools/live_save_gate.py' '$1' --event '$2' --run-kind full --base-model-dir '$HF_MODEL' --train-config '$RESOLVED_CFG' --fqn-map '$FQN_MAP' --json '$3'" \
    >"$4" 2>&1 &
  fs_gate_pid=$!
  (
    # defect-2 repair (orphan-leak). OLD SHAPE: the budget sleep ran in
    # the FOREGROUND of this subshell, so `kill "$fs_watch_pid"` below
    # reaped only bash — the `sleep` GRANDCHILD was orphaned holding our
    # inherited stdout (any $( ) caller then blocked the whole budget —
    # the measured 600s wedge at contracts:2520), and on any platform
    # where TERM to a subshell in foreground-wait defers, this subshell
    # itself outlived its dismissal and ran the marker-append and the
    # kill of $fs_gate_pid against a possibly-RECYCLED pid, stamping a
    # false cause into a CLEARED gate's capture. NEW SHAPE: the sleeps
    # are BACKGROUND children whose pid we record and WAIT on — bash
    # regains control at the wait's signal-checkpoints, so the TERM trap
    # fires promptly, kills the sleep, and exits. The trap's `exit 0` is
    # load-bearing: without it the handler would RETURN, wait would come
    # back signalled, and the subshell would run on into the marker/TERM
    # lines below on a gate that finished IN BUDGET — re-introducing the
    # exact false-cause bug this repair kills. Race, stated: a TERM
    # landing after `sleep &` forks but before the pid assignment runs
    # would orphan one mute sleep; the detached stdio (</dev/null
    # >/dev/null 2>&1 on this subshell) caps that worst case at a sleep
    # that can write nothing, signal nothing, and hold no caller's pipe
    # — the two defences are belt-and-braces BECAUSE each alone leaves
    # one of the leak's signatures open, and because any future child
    # added to this watchdog reopens the wedge only if it inherits our
    # stdio. The one sanctioned output channel remains the explicit
    # >>"$4" marker append below; nothing else may print.
    fs_watch_sleep_pid=""
    trap 'if [[ -n "$fs_watch_sleep_pid" ]]; then kill "$fs_watch_sleep_pid" 2>/dev/null; fi; exit 0' TERM
    sleep "${FS_GATE_TIMEOUT_S:-600}" &
    fs_watch_sleep_pid=$!
    wait "$fs_watch_sleep_pid"
    printf 'live_gate wall-clock budget exhausted (%ss, FS_GATE_TIMEOUT_S) while the gate was still running — a wedged verifier is infrastructure, never a checkpoint verdict (recorded here so the rc-124 arms carry a cause, not just a number)\n' "${FS_GATE_TIMEOUT_S:-600}" >>"$4" 2>/dev/null
    kill -TERM "$fs_gate_pid" 2>/dev/null
    sleep 5 &
    fs_watch_sleep_pid=$!
    wait "$fs_watch_sleep_pid"
    kill -KILL "$fs_gate_pid" 2>/dev/null
  ) </dev/null >/dev/null 2>&1 &
  fs_watch_pid=$!
  wait "$fs_gate_pid" || fs_gate_rc=$?
  kill "$fs_watch_pid" 2>/dev/null
  wait "$fs_watch_pid" 2>/dev/null || true
  if grep -qF 'live_gate wall-clock budget exhausted' "$4" 2>/dev/null; then
    fs_gate_rc=124
  fi
  return "$fs_gate_rc"
}
fs_gate_refusal_class() { # $1=report path -> stdout: the tool's refusal_class token, or empty
  # fix44 / #77-B3's on-disk contract, consumed: the tool writes a refusal
  # record carrying "refusal_class" on EVERY exit 3, at the point of
  # refusal. sed, not python, because the interpreter stack may be exactly
  # what is being adjudicated. EMPTY IS EVIDENCE here — the caller's 46 arm
  # reads it as the claim-vs-disk state — never a default, never a guess
  # (doctrines 1/4). The [a-z_]* class matches the tool's refusal-
  # vocabulary constants by construction; a record whose payload sits
  # outside that vocabulary classifies as absent, because a record that
  # does not parse cannot vouch for a cause.
  local fs_class=""
  if [[ -s "${1:-}" ]]; then
    fs_class=$(sed -n 's/.*"refusal_class": "\([a-z_]*\)".*/\1/p' "$1" | head -n1)
  fi
  printf '%s' "$fs_class"
}
fs_gate_verdict_to_rc() { # $1=gate rc  $2=which adjudication  $3=report path  $4=capture log -> echoes meaning; sets GATE_JOB_RC
  # The 0/44/45/46 table is CONTINUOUS with the pre-fix45 mapping — same
  # four job rcs, same meanings. What changes is WHERE the evidence comes
  # from (fix44's #77-B2/B3 ported): the exit-3 CAUSE is read from the
  # tool's own on-disk refusal record, never narrated from a guess-list; a
  # 3 with no usable record is the claim-vs-disk gap and is indicted at 46
  # instead of laundered into 45; and an exit 0 earns 0 only when
  # corroborated by the gate's OWN printed verdict plus an on-disk report
  # (fix44 m6 — an rc claiming a success its own evidence does not carry
  # is the founding bug's shape, and that shape is tool-independent). The
  # <compute-node> run's fs_gate/ held only fqn-map.json and
  # resolved-train-config.json after two in-situ adjudications — the
  # rc-3-record-ABSENT shape this mapper indicts, measured live.
  # CALIBRATION, stated (fix45 Task-A3.1): on this --run-kind full path NO
  # member of the exit-3 class is a chosen abstention — there is no
  # analogue of the LoRA adapter prefix that we deliberately left
  # unpinned, because every input this call consumes is launcher-produced
  # and fail-closed at launch. Every 3 is therefore a wiring/tool/artifact
  # failure and fails the job 45 with its named cause. Introducing any
  # future rc-0 member of the class would be a deliberate act: record it
  # here and census it in the contract suite, the way the LoRA sibling's
  # m2 leg does, or this comment is a lie. What this mapper deliberately
  # never sees: the epilogue's OWN minted UNMEASURED (iter dir or fqn map
  # missing) — the tool never ran, so no refusal record exists to demand,
  # and feeding that minted 3 through here would mint a false 46
  # indictment, the doctrine-5 symmetric defect.
  local fs_gcls="" fs_gcause=""
  fs_gcls=$(fs_gate_refusal_class "${3:-}")
  if [[ -r "${4:-}" ]]; then
    fs_gcause=$(grep -m1 'live_gate could not measure:' "$4" 2>/dev/null || true)
  fi
  case "$1" in
    0)
      if [[ -s "${3:-}" && -r "${4:-}" ]] && grep -qF 'LIVE GATE VERDICT: CLEAR' "$4"; then
        echo "artifact gate ($2): CLEAR (exit 0) — every applicable gate reached a real verdict over non-vacuous coverage; corroborated by the gate's own printed verdict in $4; report verified present: $3"
        GATE_JOB_RC=0
      else
        echo "artifact gate ($2): OVERCLAIM — gate rc 0, but the gate never printed its CLEAR verdict into ${4:-<none>} or its report $3 is missing/empty. An rc claiming a success its own evidence does not carry is the founding bug's shape; mapped to the infrastructure class (46), never a pass (doctrines 4/5)." >&2
        GATE_JOB_RC=46
      fi ;;
    1)
      echo "artifact gate ($2): BLOCKED (exit 1) — the artifact was MEASURED and did not clear (or a MUST_FIRE detector is unproven on it). Blocking reasons: $3. These artifacts are NOT cleared for resume, eval, or export." >&2
      GATE_JOB_RC=44 ;;
    3)
      if [[ -n "$fs_gcls" ]]; then
        echo "artifact gate ($2): UNMEASURED (exit 3) — cause read from the tool's OWN refusal record at $3: refusal_class=$fs_gcls${fs_gcause:+ (verbatim from the tool: $fs_gcause)}. The artifact is UNVERIFIED: never read that as clear. On this run-kind=full path NO member of the exit-3 class is a chosen abstention, so every 3 is a wiring/tool/artifact failure and fails the job at 45." >&2
        GATE_JOB_RC=45
      else
        echo "artifact gate ($2): UNMEASURED-CLAIMED, RECORD ABSENT — exit 3 with no usable refusal record at $3 (missing, empty, or carrying no refusal_class in the tool's vocabulary): the tool claims it could not measure but left no record of it. That is the exact claim-vs-disk gap measured on the LoRA PROBE runs (#77-B3), indicted here, not narrated — a 3 with no evidence is infrastructure, not a measurement of its own inability. Infrastructure class 46${fs_gcause:+; the capture carries a cause fragment the record cannot vouch for: $fs_gcause}." >&2
        GATE_JOB_RC=46
      fi ;;
    *)
      echo "artifact gate ($2): INFRASTRUCTURE FAILURE (gate rc=$1, outside the gate's own 0/1/3 contract — 2=argparse/wiring, 124=the fix45 wall-clock bound fired (a wedged verifier is infrastructure, never a checkpoint verdict; cause is in the capture), 127=tool vanished mid-run, anything else=uncaught crash) — fail closed." >&2
      GATE_JOB_RC=46 ;;
  esac
}

# ----------------------------------------------------------------------------
# Launch with LIVE TRIPWIRES (this run is a verification fixture as much as a
# training run — fail loud and early, not after 6 hours of garbage):
#   (a) systemic "ZERO supervised tokens" (>= threshold) -> template patch dead
#   (b) lm loss NaN
#   (c) lm loss < collapse floor (job-1668 signature: 3.8 -> 0.01 by iter 160)
#   (d) FIRST-SAVE ARTIFACT adjudication: the first latest_checkpointed_
#       iteration.txt triggers tools/live_save_gate.py --event first_save on
#       the realized iter dir. (a)-(c) grep logs; (d) is the first tripwire
#       that adjudicates the ARTIFACT — the founding-incident class the three
#       log-greps cannot see. BLOCKED (exit 1) kills; UNMEASURED (exit 3)
#       disarms this leg loudly without killing (a broken verifier is not
#       evidence of a doomed run — the epilogue re-adjudicates and a
#       persistent 3 fails the job there, doctrine-symmetric). fix45
#       addenda, all three load-bearing and all three pointing at the live
#       arms below, with the diagnosis as CORRECTED by fix45-A2 (the full
#       record and measurement table stand above fs_live_save_gate): the
#       gate now runs EXECUTOR-ROUTED — the #77-B1 sibling defect; the host
#       python3 CAN read a DCP when the user site is visible (measured
#       CLEAR on the real 99 GB save), so the pinned property is ROUTING to
#       the adjudicator stack that wrote the artifact plus three refused
#       reasons not to expose the user site, NEVER host incapacity —
#       routing under a WALL-CLOCK bound (a wedged live gate would silently
#       stop (a)-(c) by never letting this loop roll; minted as rc 124 —
#       NOT the tool's contract — and disarmed like any other off-contract
#       outcome); an rc 0 is accepted only beside the gate's OWN printed
#       CLEAR (an uncorroborated 0 disarms, and the epilogue indicts a
#       repeat at 46); and every exit-3 narration names the cause read
#       from the tool's refusal record — never a guessed one.
# ----------------------------------------------------------------------------
ZERO_WARN_MAX=${ZERO_WARN_MAX:-20}       # threshold = systemic, not one pathological row
COLLAPSE_FLOOR=${COLLAPSE_FLOOR:-0.05}   # healthy Taiwan runs stayed >=~1.0 by iter 80

echo "Training command ($FS_BACKEND): $CMD"
run_in_container --workdir "$REPO" bash -lc "$CMD" &
# SRUN_PID is the historical name; it is now the run_in_container pid (srun on
# the sbatch arm, enroot start on the enroot arm).
SRUN_PID=$!
# A TERM to the launcher-side wrapper does not reliably propagate into an
# enroot payload. The tripwires below may therefore need to force-remove a
# container, and they may only ever remove the one THIS invocation accounted
# for — shared tray, other people's containers exist. RIC_ACTIVE_CONTAINER is
# that accounting; fs_hard_stop_training refuses to act without it.
[[ "$FS_BACKEND" == enroot ]] && RIC_ACTIVE_CONTAINER=$ENROOT_NAME || true

(
  while kill -0 $SRUN_PID 2>/dev/null; do
    sleep 45
    [[ -f "$LOG_OUT" ]] || continue
    # fix45-A2 / #84: on zero matches `grep -c` PRINTS "0" AND exits 1, so
    # the old `|| echo 0` also fired and ZC became the two-line string
    # "0\n0" — every healthy run logged "((: 0\n0: syntax error in
    # expression" every 45 s (measured <compute-node>, whole run + the epilogue
    # twin). Behaviour was ACCIDENTALLY correct — the malformed arithmetic
    # evaluated false, the right answer for zero matches, and the tripwire
    # still fired when matches existed: this is noise-hygiene, not a dead
    # tripwire, and is scoped low on exactly that evidence. The cost it
    # actually bills: a permanent syntax error in every healthy run's log —
    # the noise that trains an operator to ignore watcher errors. Take
    # grep's own printed count always (`|| true`, consume the rc); default
    # only when grep printed nothing at all (missing log), which preserves
    # the old echo's one honest job. The epilogue site carries the same fix
    # and points here.
    ZC=$(grep -c "ZERO supervised tokens" "$LOG_OUT" 2>/dev/null || true); ZC=${ZC:-0}
    if (( ZC >= ZERO_WARN_MAX )); then
      echo "TRIPWIRE: $ZC 'ZERO supervised tokens' warnings — CoT-preserving template is NOT effective (trap #1). Killing job $SLURM_JOB_ID." >&2
      # Same TERM->grace->KILL escalation; the enroot arm additionally
      # force-removes exactly the container this job launched, so a tripped
      # run cannot orphan ranks on the tray.
      fs_hard_stop_training "$SRUN_PID"; break
    fi
    if grep -qiE "lm loss: nan" "$LOG_OUT" 2>/dev/null; then
      echo "TRIPWIRE: NaN lm loss — suspect stale/mis-converted base ckpt (see 31B STALE-2026-06-23 incident). Killing." >&2
      fs_hard_stop_training "$SRUN_PID"; break   # see tripwire (a) above
    fi
    LV=$(grep -oE "lm loss: [0-9]+\.[0-9]+(E[-+][0-9]+)?" "$LOG_OUT" 2>/dev/null | tail -1 | awk '{print $3}')
    if [[ -n "${LV:-}" ]] && LC_ALL=C awk -v v="$LV" -v f="$COLLAPSE_FLOOR" 'BEGIN{exit !(v<f)}'; then
      echo "TRIPWIRE: lm loss $LV < floor $COLLAPSE_FLOOR (loss-collapse signature). Killing." >&2
      fs_hard_stop_training "$SRUN_PID"; break   # see tripwire (a) above
    fi
    # Tripwire (d) — ADDED, nothing above modified. First appearance of the
    # tracker file is the first_save event the gate's CLI names; Megatron
    # writes the tracker after the save finalizes, which is the only ordering
    # that makes reading the iter dir inside a live run safe. The iter dir is
    # resolved by EXACT iteration match (no zero-padding assumption), and if
    # it is not visible yet this poll round just retries — the leg disarms
    # only after a real invocation, on UNMEASURED (loudly), or on BLOCKED
    # (after killing). $FQN_MAP absence mid-run (deleted after launch) simply
    # never arms the leg; the epilogue's own missing-map path reports that
    # infrastructure drift as UNMEASURED-shaped rather than passing over it.
    # The rc is captured with `|| FS1_RC=$?` and tested on all three arms of
    # the contract — a `|| true` here would re-launder the founding bug.
    if [[ -z "${FS_FIRST_SAVE_ADJUDICATED:-}" && -f "$LATEST_ITER_FILE" && -f "$FQN_MAP" ]]; then
      FS1_ITER=$(tr -dc '0-9' < "$LATEST_ITER_FILE")
      FS1_CKPT=""
      for d in "$OUT_DIR/checkpoints"/iter_*; do
        [[ -d "$d" ]] || continue
        if [[ "${d##*/iter_}" =~ ^0*${FS1_ITER}$ ]]; then FS1_CKPT=$d; break; fi
      done
      if [[ -n "$FS1_CKPT" ]]; then
        FS_FIRST_SAVE_ADJUDICATED=1
        FS1_REPORT=$OUT_DIR/fs_gate/report-first-save.json
        # The capture is the gate's OWN voice (fix44 / #77-B2-B3 ported):
        # the arms below read the exit-3 CAUSE from the tool's on-disk
        # refusal record and quote the capture, instead of this launcher
        # narrating 'refusal/crash' for every member of the exit-3 class —
        # the multiplexed-decode defect #77-B2 was measured against.
        FS1_CAPTURE=$OUT_DIR/fs_gate/capture-first-save.log
        FS1_RC=0
        fs_live_save_gate "$FS1_CKPT" first_save "$FS1_REPORT" "$FS1_CAPTURE" || FS1_RC=$?
        case "$FS1_RC" in
          0) # Corroborated CLEAR only (fix44 m6 ported): rc 0 is evidence
             # only beside the gate's own printed verdict in the capture
             # and the report on disk. An uncorroborated 0 does NOT pass
             # and does NOT kill training: it disarms like any other
             # off-contract outcome, and the epilogue's corroborated-CLEAR
             # check indicts a repeat there at 46. The live leg never
             # mints a pass on evidence that does not exist.
             if [[ -s "$FS1_REPORT" ]] && grep -qF 'LIVE GATE VERDICT: CLEAR' "$FS1_CAPTURE"; then
               echo "first-save artifact gate: CLEAR (iter $FS1_ITER) — corroborated by the gate's own printed verdict; report $FS1_REPORT"
             else
               echo "first-save artifact gate returned rc=0 WITHOUT its own printed verdict in $FS1_CAPTURE or with no report at $FS1_REPORT — an uncorroborated pass is the founding bug's shape; live leg DISARMED, and the epilogue's corroborated-CLEAR check will indict this state at 46 if it repeats there." >&2
             fi ;;
          1) echo "TRIPWIRE: first-save artifact adjudication BLOCKED (gate exit 1, iter $FS1_ITER) — a measured defect or an unproven MUST_FIRE detector on the first real artifact; blocking reasons in $FS1_REPORT. Killing job $SLURM_JOB_ID." >&2
             fs_hard_stop_training "$SRUN_PID"; break ;;
          *) fs1_class=$(fs_gate_refusal_class "$FS1_REPORT")
             fs1_cause=""
             [[ -r "$FS1_CAPTURE" ]] && fs1_cause=$(grep -m1 'live_gate could not measure:' "$FS1_CAPTURE" 2>/dev/null || true)
             echo "first-save artifact gate returned rc=$FS1_RC — NOT a checkpoint verdict (3=UNMEASURED: a tool-side refusal; refusal_class from the tool's own record: ${fs1_class:-<none — no refusal record at $FS1_REPORT; the claim-vs-disk state #77-B3 indicts, live>}; 124=the fix45 wall-clock bound fired; 2=argparse/wiring; 127=tool vanished). The tool's own words: ${fs1_cause:-<no 'live_gate could not measure:' line captured in $FS1_CAPTURE>}. Live leg DISARMED WITHOUT KILLING; the epilogue re-adjudicates authoritatively and a persistent non-CLEAR fails the job there (a healthy run must not die for a broken verifier, and a doomed artifact must not pass for one either)." >&2 ;;
        esac
      fi
    fi
  done
) &
WATCH_PID=$!

wait $SRUN_PID; RC=$?
kill $WATCH_PID 2>/dev/null || true

# ----------------------------------------------------------------------------
# Epilogue — never trust rc=0 alone (trap 2). The gates adjudge ARTIFACTS.
# ----------------------------------------------------------------------------
echo "=== g4e4b fullft job $SLURM_JOB_ID finished rc=$RC ==="
# fix45-A2 / #84: the epilogue twin of the watcher's counter — `grep -c`
# prints its own count on rc 1, so `|| echo 0` double-printed on every
# healthy run (measured <compute-node>: the syntax error fired here too). grep's
# own count always; default only on silence (LOG_OUT may not exist here).
ZC=$(grep -c "ZERO supervised tokens" "$LOG_OUT" 2>/dev/null || true); ZC=${ZC:-0}
echo "post-run: 'ZERO supervised tokens' count=$ZC"
if [[ -f "$LATEST_ITER_FILE" ]]; then
  LAST=$(cat "$LATEST_ITER_FILE")
  BYTES=$(du -sb "$OUT_DIR/checkpoints" | cut -f1)
  echo "post-run: latest_checkpointed_iteration=$LAST  checkpoints_bytes=$BYTES"
  echo "ARTIFACT(for gates): $OUT_DIR/checkpoints (iter $LAST)"
  # fix35: that line used to END the script — artifact announced, gates never
  # invoked (the finding). Now it introduces the adjudication. Runs even when
  # training's own RC!=0: a crashed run's last finalized save is exactly the
  # artifact a resume would consume, so it must be judged either way. CLEAR
  # below clears the ARTIFACT only — a crashed run keeps its nonzero RC.
  # Latest-iter resolution repeats the watcher's exact-match rule (no
  # zero-padding guess). Missing iter dir or missing denominator map is
  # treated as the gate's own exit-3 shape (UNMEASURED): loud, job-failing,
  # never a pass — an adjudicator that cannot run blocks (doctrine 4) even
  # though the tool was verified present at launch.
  LAST_D=$(tr -dc '0-9' <<<"$LAST")
  FINAL_CKPT=""
  for d in "$OUT_DIR/checkpoints"/iter_*; do
    [[ -d "$d" ]] || continue
    if [[ "${d##*/iter_}" =~ ^0*${LAST_D}$ ]]; then FINAL_CKPT=$d; break; fi
  done
  FINAL_REPORT=$OUT_DIR/fs_gate/report-final.json
  FINAL_CAPTURE=$OUT_DIR/fs_gate/capture-final.log
  FS_GATE_RC=0
  if [[ -n "$FINAL_CKPT" && -f "$FQN_MAP" ]]; then
    fs_live_save_gate "$FINAL_CKPT" save "$FINAL_REPORT" "$FINAL_CAPTURE" || FS_GATE_RC=$?
    fs_gate_verdict_to_rc "$FS_GATE_RC" "final save (iter $LAST)" "$FINAL_REPORT" "$FINAL_CAPTURE"
  else
    # The LAUNCHER's own minted UNMEASURED, deliberately kept OUT of
    # fs_gate_verdict_to_rc (fix45): that mapper's contract takes a
    # tool-emitted rc plus the tool's on-disk evidence. Demanding a
    # refusal record from a run that never happened would mint a false 46
    # indictment (the doctrine-5 symmetric defect), so the cause is named
    # HERE, with its denominator, and it fails the job 45 like every
    # UNMEASURED on this path.
    echo "artifact gate (final save, iter $LAST): UNMEASURED — the gate was never invoked: iter dir for ${LAST_D:-?} or denominator map $FQN_MAP missing (adjudicated 0 of 1 artifacts; launcher-minted, NOT a tool refusal, so no refusal record exists to read and none is demanded). Never a pass." >&2
    GATE_JOB_RC=45
  fi
  (( GATE_JOB_RC == 0 )) || RC=$GATE_JOB_RC
  [[ -f "$OUT_DIR/fs_gate/report-first-save.json" ]] && \
    echo "post-run: first-save gate report also on disk: $OUT_DIR/fs_gate/report-first-save.json" || true
else
  echo "post-run: NO latest_checkpointed_iteration.txt — treat ANY artifact as ABSENT even though rc=$RC (trap 2/9 lesson)" >&2
  (( RC == 0 )) && RC=43
fi
(( ZC > 0 )) && (( ZC < ZERO_WARN_MAX )) && echo "WARN: $ZC isolated zero-supervised-token samples — inspect before trusting per-token loss accounting"
exit $RC
