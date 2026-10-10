"""Where do the long-form loops come from? Per-chunk transcription of whole calls, with audio stats.

Splits each call into the same non-overlapping chunks NeMo's chunked inference uses, transcribes
every chunk with one model (Canary prompt, pnc=no), and prints per chunk: loop words
(speech_metrics.loop_words), RMS level (dBFS), share of near-silent 25 ms frames, and the chunk's
word count. A loop that tracks silence or low level points at the audio; one that does not points
at the decoder.
Usage: loop_chunks.py --model M.nemo --manifest longform.json --calls 4432298,4479741 [--chunk 40]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from foundationscale.train.speech_metrics import loop_words, normalize_transcript


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--chunk", type=float, default=40.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from nemo.collections.asr.models import ASRModel

    import torch

    # Loaded exactly as speech_canary/eval_wer_nemo.py loads it (bf16 on CUDA), so the chunks see
    # the same numerics as the scored runs.
    if args.model.endswith(".nemo"):
        model = ASRModel.restore_from(args.model, map_location="cuda")
    else:
        model = ASRModel.from_pretrained(args.model, map_location="cuda")
    model = model.to(torch.bfloat16).eval()
    rows = {json.loads(line)["id"]: json.loads(line) for line in Path(args.manifest).read_text().splitlines()}
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        for call in args.calls.split(","):
            wave, sr = sf.read(rows[call]["audio_filepath"], dtype="float32")
            n = int(args.chunk * sr)
            paths, stats = [], []
            for k, start in enumerate(range(0, len(wave), n)):
                seg = wave[start : start + n]
                if len(seg) < sr:  # under a second: NeMo's chunker keeps it, but it carries no speech
                    continue
                p = Path(tmp) / f"{call}_{k:04d}.wav"
                sf.write(str(p), seg, sr)
                frames = seg[: len(seg) // 400 * 400].reshape(-1, 400)
                rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
                stats.append({"chunk": k, "start_s": round(start / sr, 1),
                              "dbfs": round(float(20 * np.log10(np.sqrt((seg**2).mean()) + 1e-12)), 1),
                              "silent_frac": round(float((rms < 10 ** (-45 / 20)).mean()), 2)})
                paths.append(str(p))
            hyps = model.transcribe(paths, batch_size=16, pnc="no", source_lang="en", target_lang="en")
            for st, h in zip(stats, hyps, strict=True):
                text = h.text if hasattr(h, "text") else str(h)
                words = normalize_transcript(text)
                st.update({"call": call, "words": len(words), "loop_words": loop_words(words),
                           "hyp_head": " ".join(words[:12])})
                report.append(st)
    Path(args.out).write_text(json.dumps(report, indent=1))
    looping = [r for r in report if r["loop_words"]]
    clean = [r for r in report if not r["loop_words"]]

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 2) if xs else float("nan")

    print(f"chunks {len(report)}: looping {len(looping)}, clean {len(clean)}")
    for name, group in (("looping", looping), ("clean", clean)):
        print(f"{name:8s} mean dBFS {mean([r['dbfs'] for r in group])} silent_frac "
              f"{mean([r['silent_frac'] for r in group])} words {mean([r['words'] for r in group])}")
    for r in sorted(looping, key=lambda r: -r["loop_words"])[:8]:
        print(r)


if __name__ == "__main__":
    main()
