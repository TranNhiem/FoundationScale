"""SP-2/SP-3 selection and held-out test, with the rule fixed in advance.

Candidates: every ``<run>/val_<ckpt>/eval.json`` plus the SFT model's own validation decode. The
lowest FS-normalizer WER on ``earnings_validation1000`` wins; if that is the SFT model, the verdict
is "no improvement" and nothing is decoded on the test set. Otherwise the winner's test decode
(``<run>/test_<ckpt>/eval.json``, written by decode.sh) is compared with the SFT test decode by a
paired bootstrap on identical rows. usage:
  select_and_test.py select RUN_DIR SFT_VAL_EVAL            -> prints the winner (or SFT)
  select_and_test.py compare SFT_TEST_EVAL WINNER_TEST_EVAL -> WERs + paired 95% CI
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from foundationscale.train.speech_metrics import (
    corpus_error_rate,
    normalize_transcript,
    word_errors,
)


def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["all"]


def _wer(rows: list[dict]) -> float:
    m = corpus_error_rate(
        [(r["reference"], r["hypothesis"]) for r in rows], metric="wer", expected=len(rows)
    ).as_manifest()
    if m["verdict"] != "COVERED":
        raise SystemExit(f"REFUSE (95): coverage {m['verdict']}")
    return 100.0 * m["error_rate"]


def select(run_dir: Path, sft_val: Path) -> None:
    scores = {"SFT": _wer(_rows(sft_val))}
    for ev in sorted(run_dir.glob("val_*/eval.json")):
        scores[ev.parent.name.removeprefix("val_")] = _wer(_rows(ev))
    best = min(scores, key=scores.get)
    print(
        json.dumps({"validation_wer": {k: round(v, 2) for k, v in scores.items()}, "winner": best})
    )


def compare(sft_test: Path, cand_test: Path, n_boot: int = 2000) -> None:
    b = {r["id"]: r for r in _rows(sft_test)}
    t = {r["id"]: r for r in _rows(cand_test)}
    if sorted(b) != sorted(t):
        raise SystemExit("REFUSE (96): the two test decodes cover different rows")
    keys = [k for k in sorted(b) if normalize_transcript(b[k]["reference"])]
    e = {
        k: (
            word_errors(b[k]["reference"], b[k]["hypothesis"]),
            word_errors(t[k]["reference"], t[k]["hypothesis"]),
        )
        for k in keys
    }

    def pooled(ks: list[str], side: int) -> float:
        errs = sum(e[k][side].errors for k in ks)
        refs = sum(e[k][side].reference_length for k in ks)
        return 100.0 * errs / refs

    rng = random.Random(0)
    diffs = []
    for _ in range(n_boot):
        s = [rng.choice(keys) for _ in keys]
        diffs.append(pooled(s, 1) - pooled(s, 0))
    diffs.sort()
    print(
        json.dumps(
            {
                "rows": len(keys),
                "sft_wer": round(_wer(list(b.values())), 2),
                "candidate_wer": round(_wer(list(t.values())), 2),
                "paired_diff_95ci": [
                    round(diffs[int(0.025 * n_boot)], 2),
                    round(diffs[int(0.975 * n_boot) - 1], 2),
                ],
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
