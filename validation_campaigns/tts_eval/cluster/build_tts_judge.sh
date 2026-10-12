#!/bin/bash
# Seed-TTS-eval judge env: Whisper-large-v3 (transformers), Paraformer-zh (funasr), WavLM-large SV (UniSpeech + s3prl hub).
set -eu
V=$HOME/envs/tts-judge
rm -rf $V
/opt/venv/bin/python3 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0" "torchaudio==2.11.0"
echo torch==2.11.0 > $V/constraints.txt; echo torchaudio==2.11.0 >> $V/constraints.txt
$V/bin/pip install -q -c $V/constraints.txt "transformers==4.57.3" "jiwer==3.0.5" "fire==0.7.1" funasr modelscope zhconv zhon scipy soundfile librosa tqdm easydict joblib omegaconf
$V/bin/pip freeze > $V/requirements.lock
$V/bin/python - <<'PY'
import importlib.metadata as m
for p in ["torch","torchaudio","transformers","funasr","modelscope","jiwer","zhconv"]: print(p, m.version(p))
from funasr import AutoModel; import jiwer, zhconv, zhon; print("judge imports OK")
PY
# SIM: s3prl is pinned in TORCH_HOME/hub (commit ec8064b) with hubconf narrowed to wavlm_large; see FS_PIN.txt there.
