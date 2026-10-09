"""Whole-call Earnings-22 long-form set: the 6 held-out test calls, full audio and transcript.

Source: distil-whisper/earnings22 config "full" (one row per call). Only the calls of the
Open ASR Leaderboard Earnings-22 test split are kept, so nothing overlaps the Earnings training
data (speech_p8 excluded exactly these calls). Audio is written as 16 kHz mono FLAC; a clip at
another rate or channel count is converted here, at build time, and the conversion is recorded
in its manifest row and printed (the eval itself never resamples).
Usage: build_longform.py   (writes manifests/longform_earnings22_test6.json, NeMo format)
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
    fs = HfFileSystem(token=False)
    wanted = test_calls(fs)
    if len(wanted) != 6:
        raise SystemExit(f"REFUSE (96): expected the 6 Earnings-22 test calls, found {sorted(wanted)}")
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
    out = MAN / "longform_earnings22_test6.json"
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"manifest": str(out), "calls": len(rows),
                      "hours": round(sum(x["duration"] for x in rows) / 3600, 2),
                      "conversions": {x["id"]: x["conversion"] for x in rows if x["conversion"]}}))


if __name__ == "__main__":
    main()
