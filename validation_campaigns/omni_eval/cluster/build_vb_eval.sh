#!/bin/bash
# VoiceBench evaluator env (CPU): only what src/evaluator imports (mcq, ifeval, harm, bbh).
set -eu
V=$HOME/envs/vb-eval
rm -rf $V; /opt/venv/bin/python3 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q absl-py immutabledict langdetect "nltk>=3.9" loguru pandas numpy "qa_metrics==0.2.17" openai
$V/bin/python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
$V/bin/pip freeze > $V/requirements.lock
cd $HOME/datasets/speech/voicebench/code && $V/bin/python -c "from src.evaluator import evaluator_mapping; print('evaluators', sorted(evaluator_mapping))"
