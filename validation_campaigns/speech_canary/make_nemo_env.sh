#!/bin/bash
# Run INSIDE the NeMo Framework image (26.08 measured): a venv layered on the image's /opt/venv
# with NeMo 3.1 [asr] installed from the image's own /opt/NeMo source, pinned to the image's
# torch/transformers/numpy stack so pip cannot swap the CUDA builds. Idempotent.
# The image ships the NeMo source but not the installed package (measured: `import nemo` fails).
set -euo pipefail
V="${NEMO_SPEECH_VENV:-${CLUSTER_HOME:-$HOME}/envs/nemo-speech-asr}"
[ -x "$V/bin/python" ] || /opt/venv/bin/python -m venv --system-site-packages "$V"
/opt/venv/bin/python -m pip freeze 2>/dev/null \
  | grep -i -E "^(torch|torchaudio|torchvision|triton|transformers|tokenizers|safetensors|numpy|accelerate|huggingface[-_]hub|nvidia-|cuda-)" \
  | grep -v " @ " > "$V/constraints.txt" || true
"$V/bin/python" -m pip install --quiet --upgrade pip
"$V/bin/python" -m pip install --quiet -c "$V/constraints.txt" "/opt/NeMo[asr]"
# speechlm2 (Canary-Qwen SALM) imports peft; --no-deps keeps the container torch untouched.
"$V/bin/python" -m pip install --quiet --no-deps peft accelerate
"$V/bin/python" -c "import torch, transformers, nemo; print('torch', torch.__version__, 'transformers', transformers.__version__, 'nemo', nemo.__version__)"
