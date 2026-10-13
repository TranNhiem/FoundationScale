#!/bin/bash
# O0: cascade baseline over several VoiceBench subsets on one GPU, sequentially. usage: run_o0_lane.sh GPU SUBSET...
G=$1; shift; V=$HOME/datasets/speech/voicebench
for s in "$@"; do $V/run_vb.sh $G cascade $s || echo "subset $s exit $?"; done
