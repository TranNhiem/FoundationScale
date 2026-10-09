#!/bin/bash
# Re-score HF-native speech results on the large sets. Usage: run_hf_bigeval.sh <host-index 0|1>
FS=/home/hhri-ai/hh28144/Project-Developments/FoundationScale
W=/home/hhri-ai/hh28144/pretraining_weights; R=$FS/runs; M=/home/hhri-ai/hh28144/datasets/speech/manifests
G4=$W/Vision-Language-Models/Google/Gemma4/gemma-4-E4B-it; WH=$W/Speech-Models/whisper-large-v3; QA=$W/Speech-Models/Qwen2-Audio-7B-Instruct
# name | model | extra args
JOBS=(
"gemma_base|$G4|--processor $G4"
"gemma_full|$R/speech_p3_full_20261006_165415/final|--processor $G4"
"gemma_lora|$G4|--processor $G4 --adapter $R/speech_p3_lora_20261006_165416/final"
"whisper_base|$WH|--processor $WH --language en"
"whisper_ft|$R/speech_dgx18_whisper500_20261006_193123/final|--processor $WH --language en"
"qwen2audio_ft|$R/speech_dgx18_qwen2audio500_20261006_193423/final|--processor $QA"
)
SETS=(devclean_full ood_ami2000)
H=$1; all=(); for j in "${JOBS[@]}"; do for s in "${SETS[@]}"; do all+=("$j|$s"); done; done
mine=(); for i in "${!all[@]}"; do [ $((i % 2)) -eq "$H" ] && mine+=("${all[$i]}"); done
cd $FS; mkdir -p artifacts/speech/big2
worker() {
  g=$1; shift
  for item in "$@"; do
    IFS='|' read -r name model extra set <<< "$item"
    [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || { echo "GPU$g busy, skip $name $set"; continue; }
    ENROOT_DATA_PATH=/home/hhri-ai/hh28144/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e PYTHONPATH=$FS/checkouts/fs-speech/src -e CUDA_VISIBLE_DEVICES=$g -e HF_HUB_OFFLINE=1 --mount /home/hhri-ai/hh28144:/home/hhri-ai/hh28144 fs-g4e4b-nemo-automodel-26-04_compute python3 $FS/scripts/speech_eval_wer.py --model $model $extra --manifest $M/$set.jsonl --out $FS/artifacts/speech/big2/${name}_${set}.json > logs/big2_${name}_${set}.log 2>&1
    echo "$name $set rc=$? $(grep -E '^WER' logs/big2_${name}_${set}.log)"
  done
}
for g in 0 1 2 3; do
  q=(); for i in "${!mine[@]}"; do [ $((i % 4)) -eq $g ] && q+=("${mine[$i]}"); done
  [ ${#q[@]} -gt 0 ] && { worker $g "${q[@]}" & sleep 30; }
done
wait; echo DONE
