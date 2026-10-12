#!/bin/bash
# Native worker env for VoxCPM2 (voxcpm 2.0.3): standalone venv, official torch/torchaudio 2.11.0+cu130.
set -eu
V=$HOME/envs/tts-voxcpm
rm -rf $V
/opt/venv/bin/python3 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0" "torchaudio==2.11.0"
echo "torch==2.11.0" > $V/constraints.txt; echo "torchaudio==2.11.0" >> $V/constraints.txt
$V/bin/pip install -q -c $V/constraints.txt "voxcpm==2.0.3"
$V/bin/pip freeze > $V/requirements.lock
$V/bin/python - <<'PY'
import importlib.metadata as m
for p in ["torch","torchaudio","torchcodec","transformers","voxcpm","funasr"]:
    try: print(p, m.version(p))
    except Exception: print(p, None)
from voxcpm import VoxCPM; print("voxcpm import OK")
PY
