"""Whole-call Earnings-22 long-form set: the 6 held-out test calls, full audio and transcript.

Source: distil-whisper/earnings22 config "full" (one row per call). Only the calls of the
Open ASR Leaderboard Earnings-22 test split are kept, so nothing overlaps the Earnings training
data (speech_p8 excluded exactly these calls). Audio is written as 16 kHz mono FLAC; a clip at
another rate or channel count is converted here, at build time, and the conversion is recorded
in its manifest row and printed (the eval itself never resamples).
--dev N picks N other calls instead (a long-form DEV set for checkpoint/seed selection): the
first N non-test calls by file_id whose length is 25-75 min. Their segments must then be removed from
training (see filter_calls.py).
Usage: build_longform.py [--dev 6]   (writes manifests/longform_earnings22_{test6|devN}.json)
"""

from __future__ import annotations

import io
import json
import os
from math import gcd
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem, hf_hub_download
from scipy.signal import resample_poly

REPO = "distil-whisper/earnings22"
HOME = Path(os.environ.get("CLUSTER_HOME", str(Path.home())))
ROOT = HOME / "datasets/speech/earnings22_full"
MAN = HOME / "datasets/speech/manifests"


def test_calls(fs: HfFileSystem) -> set[str]:
    calls: set[str] = set()
    for f in fs.glob("datasets/hf-audio/open-asr-leaderboard/earnings22/test-*.parquet"):
        with fs.open(f) as h:
            calls |= {i.split("/")[0] for i in pq.read_table(h, columns=["id"]).column("id").to_pylist()}
    return calls


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=0)
    args = ap.parse_args()
    fs = HfFileSystem(token=False)
    test = test_calls(fs)
    if len(test) != 6:
        raise SystemExit(f"REFUSE (96): expected the 6 Earnings-22 test calls, found {sorted(test)}")
    wanted = set(test)
    if args.dev:
        meta: list[tuple[str, int]] = []
        for f in sorted(fs.glob(f"datasets/{REPO}/full/test-*.parquet")):
            with fs.open(f) as h:
                t = pq.read_table(h, columns=["file_id", "file_length"]).to_pylist()
            meta += [(r["file_id"], int(r["file_length"])) for r in t]
        pool = sorted(fid for fid, sec in meta if fid not in test and 1500 <= sec <= 4500)
        wanted = set(pool[: args.dev])
        if len(wanted) != args.dev:
            raise SystemExit(f"REFUSE (96): only {len(wanted)} eligible dev calls")
    (ROOT / "flac").mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for f in sorted(fs.glob(f"datasets/{REPO}/full/test-*.parquet")):
        local = hf_hub_download(REPO, f.split(f"{REPO}/")[1], repo_type="dataset", local_dir=str(ROOT), token=False)
        for r in pq.read_table(local).to_pylist():
            if r["file_id"] not in wanted:
                continue
            wave, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            note = None
            if wave.shape[1] != 1 or sr != 16000:
                note = f"converted_from_{sr}Hz_{wave.shape[1]}ch"
            wave = wave.mean(axis=1)
            if sr != 16000:
                g = gcd(sr, 16000)
                wave, sr = resample_poly(wave, 16000 // g, sr // g).astype("float32"), 16000
            path = ROOT / "flac" / f"call_{r['file_id']}.flac"
            sf.write(str(path), wave, sr)
            seen.add(r["file_id"])
            rows.append({"audio_filepath": str(path), "duration": round(len(wave) / sr, 3),
                         "text": r["transcription"], "source_lang": "en", "target_lang": "en",
                         "taskname": "asr", "pnc": "no", "id": r["file_id"], "conversion": note})
    missing = wanted - seen
    if missing:
        raise SystemExit(f"REFUSE (96): test calls absent from the full set: {sorted(missing)}")
    out = MAN / (f"longform_earnings22_dev{args.dev}.json" if args.dev else "longform_earnings22_test6.json")
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"manifest": str(out), "calls": len(rows),
                      "hours": round(sum(x["duration"] for x in rows) / 3600, 2),
                      "conversions": {x["id"]: x["conversion"] for x in rows if x["conversion"]}}))


if __name__ == "__main__":
    main()
