#!/bin/bash
# Evaluate a published VLA checkpoint on one LIBERO suite through the FoundationScale harness.
#   bash eval_libero.sh <openpi|gr00t|longwam> <suite> [trials-per-task]
# Run inside the LIBERO eval environment; the backend's policy server is started in ITS environment by --serve-cmd.
# Required env: FS_ROOT (FS checkout), VLA_RUNS (outputs), and per backend:
#   openpi : OPENPI_SRC (openpi checkout with its .venv)
#   gr00t  : GR00T_SRC (Isaac-GR00T checkout), GR00T_PY (GR00T env python), GR00T_CKPT (e.g. .../GR00T-N1.7-LIBERO/libero_spatial)
#   longwam: LONGWAM_SRC (LongLive/Long-WAM), LONGWAM_PY (Long-WAM env python), LONGWAM_CKPT (checkpoint dir),
#            LONGWAM_EVAL_CFG and LONGWAM_MODEL_CFG (configs; see README "Long-WAM" for the two fixes they carry)
set -euo pipefail
B=$1; SUITE=$2; TRIALS=${3:-}
OUT=$VLA_RUNS/eval/${B}_${SUITE}; mkdir -p "$OUT"
case $B in
  openpi)  MODEL=pi05_libero
           SERVE="$OPENPI_SRC/.venv/bin/python $OPENPI_SRC/scripts/serve_policy.py --env LIBERO --port {port}" ;;
  gr00t)   MODEL=$GR00T_CKPT
           SERVE="env PYTHONPATH=$GR00T_SRC $GR00T_PY $GR00T_SRC/gr00t/eval/run_gr00t_server.py --model-path {model} --embodiment-tag libero_sim --use-sim-policy-wrapper --port {port}" ;;
  longwam) MODEL=$LONGWAM_CKPT
           SERVE="env PYTHONPATH=$FS_ROOT/src TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $LONGWAM_PY -m foundationscale.vla.adapters.longwam.server --config $LONGWAM_EVAL_CFG --port {port} checkpoint={model}/model.pt model_config=$LONGWAM_MODEL_CFG stats={model}/dataset_stats.json action_horizon=32 num_inference_steps=10 replan_steps=10 gpu_id=0 hardware=reference" ;;
  *) echo "backend must be openpi, gr00t or longwam" >&2; exit 96 ;;
esac
python -m foundationscale.vla.eval --backend "$B" --model "$MODEL" --suite "$SUITE" \
  ${TRIALS:+--trials-per-task $TRIALS} --out "$OUT/report.json" --serve-cmd "$SERVE"
