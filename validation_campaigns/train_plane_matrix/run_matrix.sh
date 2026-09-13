#!/bin/bash
# A3 (#331) -- the method x precision evidence matrix for the training plane.
#
# Sweeps {full-finetune, LoRA} x {bf16, fp32, nvfp4} through the SHIPPED entry
# point (`python -m foundationscale.train`), on one GPU, and records four
# independent readings per cell. It exists so the framework's precision and
# adapter claims rest on artifacts rather than on flags that were accepted.
#
# WHY FOUR READINGS. An exit code cannot tell a run that worked from a run that
# was mislabelled, and every defect this matrix was built after was of the second
# kind:
#
#   rc       the four-state verdict: 0 GREEN, 5 RED, 95 UNMEASURED, 96 REFUSE.
#   bytes    the METHOD control. Full-finetune and LoRA artifacts differ by ~330x,
#            so a LoRA cell writing full-model bytes -- or the reverse -- is a
#            mislabelled run even at rc=0, and no log line can disguise it.
#   saved    the PRECISION control, read off the safetensors HEADERS rather than
#            off a log line that claims them. This is the only reading that can
#            contradict the declaration, which is the whole point: #422 was a run
#            that declared fp32, trained bf16, and said nothing.
#   markers  the post-load precision verification and the save-completeness
#            denominator, quoted VERBATIM. An absent reading must render as
#            absent; a grep for wording the code does not emit would render a
#            silent no-op as a pass.
#
# WHY ONE PASS. Cells measured on different trees cannot be compared to each
# other, which is the one property a matrix exists to provide. Every cell here is
# produced by a single invocation against a single working tree, and the tree's
# file hashes are recorded in the header so a reader can bind the numbers to a
# commit.
#
# No estate literals: every path comes from the environment or from a default
# that is relative to this checkout. Set FS_MATRIX_PYTHON, FS_MATRIX_PROFILE,
# FS_MATRIX_MODEL and FS_MATRIX_DATASET for your site.
set -u

PY=${FS_MATRIX_PYTHON:-python3}
PROFILE=${FS_MATRIX_PROFILE:-}
MODEL=${FS_MATRIX_MODEL:-Qwen/Qwen2.5-1.5B}
DATASET=${FS_MATRIX_DATASET:-fancyzhx/ag_news}
SUMMARY=${FS_MATRIX_OUT:-./fs_a3_matrix.txt}
# Node-local scratch, never a network filesystem: an NFS output directory
# produced ENOLCK and a FALSE RED once already (#409), which is an environment
# failure wearing a scientific verdict.
SCRATCH=${FS_MATRIX_SCRATCH:-/tmp}
STEPS=${FS_MATRIX_STEPS:-20}

if [ -z "$PROFILE" ]; then
  echo "REFUSE 96: set FS_MATRIX_PROFILE to a run profile path" >&2
  exit 96
fi

: > "$SUMMARY"
{
  echo "A3 matrix start $(date -Iseconds)"
  echo "tree: loop=$(md5sum src/foundationscale/train/loop.py | cut -d' ' -f1) gates=$(md5sum src/foundationscale/gates/checkpoint_gates.py | cut -d' ' -f1)"
  echo "env : peft=$("$PY" -c 'import peft;print(peft.__version__)' 2>&1) transformers=$("$PY" -c 'import transformers;print(transformers.__version__)' 2>&1) torch=$("$PY" -c 'import torch;print(torch.__version__)' 2>&1)"
} >> "$SUMMARY"

run_cell () {
  METHOD=$1; PREC=$2
  TAG="${METHOD}_${PREC}"
  OUT="${SCRATCH}/fs_a3_${TAG}_$$"
  LOG="${SCRATCH}/fs_a3_${TAG}.log"
  ADAPTER_ARGS=""
  [ "$METHOD" = "lora" ] && ADAPTER_ARGS="--adapter lora --adapter-rank 8 --adapter-alpha 16"
  START=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" -u -m foundationscale.train \
    --model "$MODEL" --dataset "$DATASET" \
    --output-dir "$OUT" --max-steps "$STEPS" --per-device-batch-size 1 \
    --save-interval 10 --nodes 1 --gpus-per-node 1 \
    --profile-path "$PROFILE" --precision "$PREC" $ADAPTER_ARGS > "$LOG" 2>&1
  RC=$?
  DUR=$(( $(date +%s) - START ))
  BYTES=$(du -sb "$OUT" 2>/dev/null | cut -f1)
  OBS=$("$PY" - "$OUT" <<'PY' 2>&1
import collections
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
shards = sorted(out.rglob("*.safetensors"))
if not shards:
    print("no-artifact")
    raise SystemExit
counts: collections.Counter[str] = collections.Counter()
for shard in shards:
    with shard.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    for key, spec in header.items():
        if key != "__metadata__":
            counts[spec["dtype"]] += 1
print(f"{dict(counts)} over {len(shards)} file(s)")
PY
)
  POST=$(grep -m1 'consistency.*precision=\|unverified precision declaration' "$LOG" | sed 's/.*\] *//' | cut -c1-110)
  COMP=$(grep -m1 'save_complete' "$LOG" | sed 's/.*save_complete: *//' | cut -c1-110)
  {
    echo "--- ${TAG}"
    echo "    rc=${RC} wall=${DUR}s bytes=${BYTES:-none}"
    echo "    saved   : ${OBS}"
    echo "    postload: ${POST:-<ABSENT -- the precision reading did not run>}"
    echo "    complete: ${COMP:-<no save_complete line>}"
  } >> "$SUMMARY"
  # A full-finetune fp32 cell is ~43 GB and the scratch is node-local; drop each
  # artifact once its readings are taken rather than filling the node.
  rm -rf "$OUT"
}

for method in full lora; do
  for precision in bf16 fp32 nvfp4; do
    run_cell "$method" "$precision"
  done
done

echo "=== A3 MATRIX DONE $(date -Iseconds) ===" >> "$SUMMARY"
