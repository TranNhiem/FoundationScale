#!/bin/bash
# T0 + T1 chain: lane per language, each step on its own GPU with an idle check. usage: run_t1_chain.sh GPU_EN GPU_ZH
E=$HOME/datasets/speech/tts_eval
lane() {  # lane GPU LANG
  local g=$1 l=$2
  $E/run_gt.sh $g $l meta.lst gt_$l || return
  for m in Qwen3-TTS-12Hz-1.7B-Base Qwen3-TTS-12Hz-0.6B-Base; do
    t=$(echo $m | tr 'A-Z.' 'a-z_')_$l
    $E/run_synth_qwen.sh $g $m $l meta.lst $t && $E/run_score.sh $g $l meta.lst $t || return
  done
  $E/run_synth_voxcpm.sh $g $l meta.lst voxcpm2_$l && $E/run_score.sh $g $l meta.lst voxcpm2_$l
}
lane $1 en > $E/chain_en.log 2>&1 &
lane $2 zh > $E/chain_zh.log 2>&1 &
wait
echo "chain done $(date)" >> $E/chain_en.log
