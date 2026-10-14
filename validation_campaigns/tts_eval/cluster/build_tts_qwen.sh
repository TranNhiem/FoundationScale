#!/bin/bash
# Native worker env for qwen-tts 0.1.1: standalone venv, official torch/torchaudio 2.11.0+cu130 (aarch64).
set -eu
V=$HOME/envs/tts-qwen
rm -rf $V
/opt/venv/bin/python3 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0" "torchaudio==2.11.0"
$V/bin/pip install -q "qwen-tts==0.1.1"
$V/bin/pip freeze > $V/requirements.lock
$V/bin/python - <<'PY'
import importlib.metadata as m, torch
for p in ["torch","torchaudio","transformers","accelerate","qwen-tts","onnxruntime"]: print(p, m.version(p))
import torchaudio.compliance.kaldi
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
print("qwen_tts import OK; cuda built", torch.version.cuda, "archs", torch.cuda.get_arch_list()[-3:])
PY
