"""Paired bootstrap of corpus WER difference (fine-tune minus base) over the same utterances."""

import json
import random
import sys

from foundationscale.train.speech_metrics import corpus_error_rate


def per_row(path):
    rows = json.load(open(path))["all"]
    out = {}
    for r in rows:
        m = corpus_error_rate([(r["reference"], r["hypothesis"])], metric="wer", expected=1).as_manifest()
        out[r["id"]] = m
    return out


base, tuned = per_row(sys.argv[1]), per_row(sys.argv[2])
keys = sorted(base)
assert keys == sorted(tuned)
k_err = [k for k in base[keys[0]] if "err" in k or "edit" in k][0]
k_ref = [k for k in base[keys[0]] if "ref" in k or "word" in k or "unit" in k and k != k_err][0]
b = [(base[k][k_err], base[k][k_ref]) for k in keys]
t = [(tuned[k][k_err], tuned[k][k_ref]) for k in keys]
def wer(rows, idx):
    return sum(rows[i][0] for i in idx) / sum(rows[i][1] for i in idx)
full = list(range(len(keys)))
d0 = wer(t, full) - wer(b, full)
rng = random.Random(0)
ds = sorted(wer(t, idx) - wer(b, idx) for idx in ([rng.randrange(len(keys)) for _ in keys] for _ in range(2000)))
print(f"n={len(keys)} fields=({k_err},{k_ref}) base={wer(b, full):.4f} tuned={wer(t, full):.4f} "
      f"diff={d0:+.4f} 95%CI=[{ds[50]:+.4f},{ds[1949]:+.4f}] P(diff<0)={sum(d < 0 for d in ds)/len(ds):.3f}")
