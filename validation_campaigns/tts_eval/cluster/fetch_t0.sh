#!/bin/bash
# T0/O0 downloads (TTS + omni plan). Public repos only; no token. Writes sha256 manifests.
set -u
W=$HOME/pretraining_weights/Speech-Models; E=$HOME/datasets/speech/tts_eval; PY=$HOME/envs/hf-cand-519/bin/python
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e HF_HUB_DISABLE_TELEMETRY=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
$CT $PY - <<'PY'
import os
from huggingface_hub import snapshot_download
W=os.path.expanduser("~/pretraining_weights/Speech-Models"); E=os.path.expanduser("~/datasets/speech/tts_eval")
jobs=[("Qwen/Qwen3-TTS-12Hz-1.7B-Base",f"{W}/TTS/Qwen3-TTS-12Hz-1.7B-Base","model"),
      ("Qwen/Qwen3-TTS-12Hz-0.6B-Base",f"{W}/TTS/Qwen3-TTS-12Hz-0.6B-Base","model"),
      ("Qwen/Qwen3-TTS-Tokenizer-12Hz",f"{W}/TTS/Qwen3-TTS-Tokenizer-12Hz","model"),
      ("openbmb/VoxCPM2",f"{W}/TTS/VoxCPM2","model"),
      ("funasr/paraformer-zh",f"{W}/paraformer-zh","model"),
      ("MiniMaxAI/TTS-Multilingual-Test-Set",f"{E}/minimax_multilingual","dataset"),
      ("Qwen/Qwen3-Omni-30B-A3B-Instruct",f"{W}/Qwen3-Omni-30B-A3B-Instruct","model")]
for rid,dst,kind in jobs:
    try:
        snapshot_download(rid, repo_type=kind, local_dir=dst); print("OK",rid,flush=True)
    except Exception as e:
        print("FAIL",rid,type(e).__name__,str(e)[:200],flush=True)
PY
# Google Drive artefacts of seed-tts-eval (test set + WavLM-large SV checkpoint)
mkdir -p $E/gd && cd $E/gd
python3 -m pip install --quiet --target $E/.gdown gdown >/dev/null 2>&1
PYTHONPATH=$E/.gdown python3 -m gdown 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP -O seedtts_testset.zip && echo OK seedtts_testset || echo FAIL seedtts_testset
PYTHONPATH=$E/.gdown python3 -m gdown 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP -O wavlm_large_finetune.pth && echo OK wavlm || echo FAIL wavlm
sha256sum $E/gd/* > $E/gd/SHA256SUMS 2>/dev/null
echo DONE
