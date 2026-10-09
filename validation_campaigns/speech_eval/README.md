# Speech eval comparison: the default way to report a fine-tune

`compare.py --manifest M.jsonl --base BASE.json --tuned TUNED.json [--json OUT]` reports one
base-vs-tuned comparison on one eval set:

- both checkpoints, read from each eval file's own recorded model (and adapter), never from a run
  directory name;
- rows used against rows expected (expected comes from the manifest);
- both corpus rates, and the paired-bootstrap difference with its 95% CI
  (`foundationscale.train.speech_metrics.paired_bootstrap`: the same utterances under both
  models, one draw for both);
- the `speech.runaway_hypotheses` verdict (exit 5 if it blocks).

It replaces `speech_p7/paired_bootstrap.py` and `speech_p8/runaway_verdict.py` for reporting.
Use it on the large sets (full dev-clean 2,683, AMI-2000, Earnings-22 2,504). The 300-row sets
cannot resolve shifts under about 1 point.

Cross-check (2026-10-09), Earnings-22, Canary-1B-flash:

- LS+AMI mix vs base: -0.75 points, CI [-1.65, +0.45], not significant, runaway PASS (7 vs a
  limit of 19). This matches the earlier campaign bootstrap.
- The corrupted-data run vs base: +24.56 points, runaway FAIL (122 vs 19), exit 5.

The provenance line also caught a mislabelled artifact on its first use. The retracted
"1500-step" eval file records its model as `runs/canary_ft500_mix_earn/finetuned.nemo`: the
launcher's path substitution never matched, so the 1500-step model overwrote the 500-step one.
Both belong to the retracted round in `speech_p8`, so no reported result changes.
