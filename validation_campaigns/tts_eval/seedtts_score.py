from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _tail(text: str, n: int = 40) -> str:
    """Return the last ``n`` lines of ``text`` for concise failure reporting."""
    return "\n".join(text.splitlines()[-n:])


def _run(cmd: list[str], cwd: Path) -> None:
    """Run an upstream seed-tts-eval script unmodified; exit 96 on non-zero exit.

    The upstream scripts define the scoring semantics (text normalisation, WER
    formula, similarity averaging) and are only invoked through their own CLIs so
    those semantics are used verbatim; on failure we abort rather than invent
    numbers.
    """
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"seedtts_score: upstream command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"--- stderr tail ---\n{_tail(proc.stderr)}\n"
            f"--- stdout tail ---\n{_tail(proc.stdout)}\n"
        )
        sys.exit(96)


def _git_head(seed_code: Path) -> str | None:
    """Return the git HEAD of ``seed_code`` or None (recorded for provenance)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(seed_code),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def _parse_meta(line: str, lineno: int) -> tuple[str, str, str, str, str]:
    """Parse "utt|prompt_text|prompt_wav|target[|gt]" (4 or 5 fields)."""
    parts = line.strip().split("|")
    if len(parts) not in (4, 5):
        sys.stderr.write(
            f"seedtts_score: line {lineno}: expected 4 or 5 '|' fields, "
            f"got {len(parts)}: {line!r}\n"
        )
        sys.exit(96)
    gt = parts[4] if len(parts) == 5 else ""
    return parts[0], parts[1], parts[2], parts[3], gt


def _ref_length(raw_truth: str, lang: str) -> int:
    """Reference token count exactly as run_wer.process_one post-normalisation.

    process_one strips every CJK/ASCII punctuation mark except apostrophe,
    collapses double spaces and then splits on spaces; for ``en`` the truth is
    lowercased first (words = tokens), for ``zh`` it is space-joined per char so
    tokens are single characters (the join inserts spaces that we exclude here).
    """
    import string

    from zhon.hanzi import punctuation as zh_punct

    truth = raw_truth
    for x in zh_punct + string.punctuation:
        if x == "'":
            continue
        truth = truth.replace(x, "")
    truth = truth.replace("  ", " ")
    if lang == "en":
        return len(truth.lower().split(" "))
    if lang == "zh":
        # process_one does " ".join(list(truth)); the resulting token count is
        # the number of non-space characters in the (already stripped) truth.
        return sum(1 for tok in truth if tok.strip() != "")
    raise NotImplementedError(lang)


def _run_wer(
    rows: list[dict[str, Any]],
    seed_code: Path,
    lang: str,
    python: str,
    workdir: Path,
) -> None:
    """Run upstream run_wer.py and parse its TSV rows in place.

    The upstream TSV is ``wav\\twer\\traw_truth\\traw_hypo\\tinse\\tdele\\tsubs``;
    hypothesis/reference keep the raw strings and errors = inse+dele+subs (these
    are already rates per reference length in run_wer.py, so we back out the raw
    edit counts as rate * reference_length). Corpus WER later mirrors
    average_wer.py: the unweighted mean of per-utterance WER values (its TSV is
    read line by line and averaged with np.mean of the ``wer`` column) × 100.
    """
    list_path = workdir / f"wer_list_{lang}.txt"
    out_path = workdir / f"wer_out_{lang}.tsv"
    with list_path.open("w") as f:
        for r in rows:
            f.write(f"{r['wav']}|{r['target']}\n")
    _run([python, str(seed_code / "run_wer.py"), str(list_path), str(out_path), lang], seed_code)

    by_wav: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 7:
                continue
            wav_path, wer_s, raw_truth, raw_hypo, inse_s, dele_s, subs_s = fields[:7]
            wer_val = float(wer_s)
            ref_len = _ref_length(raw_truth, lang)
            if ref_len <= 0 and wer_val > 0:
                # Derivation fallback: inse/dele/subs in run_wer.py are counts /
                # len(reference); with errors = (ins+del+sub)/len and wer = errors/len
                # we recover len = round(errors_rate_sum / wer).
                errors_rate_sum = float(inse_s) + float(dele_s) + float(subs_s)
                ref_len = round(errors_rate_sum / wer_val) if wer_val else 0
            errors = (
                (float(inse_s) + float(dele_s) + float(subs_s)) * ref_len if wer_val > 0 else 0.0
            )
            # run_wer.py computes subs/del/ins as counts / len(ref_list) where
            # ref_list = normalised truth.split(" ") — for zh this counts spaces
            # too; use that same length for count recovery when available.
            by_wav[wav_path] = {
                "reference": raw_truth,
                "hypothesis": raw_hypo,
                "errors": errors,
                "reference_length": ref_len,
                "wer": wer_val,
            }
    for r in rows:
        r["wer_record"] = by_wav.get(r["wav"])


def _run_sim(
    rows: list[dict[str, Any]], seed_code: Path, ckpt: str, python: str, workdir: Path, device: str
) -> dict[str, float]:
    """Run upstream verification_pair_list_v2.py and map wav1-path prefix to score.

    Pairs use the 2-field "<wav1>|<wav2>" form; wav1 = synth wav, wav2 = prompt wav.
    The score lines key on "<wav1>_0_-1|<wav2>_0_-1".  Corpus SIM = mean of the
    per-pair scores × 100, matching average.py (np.mean of its score column).
    """
    sv_dir = seed_code / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"
    pairs = workdir / "sim_pairs.txt"
    scores = workdir / "sim_scores.txt"
    with pairs.open("w") as f:
        for r in rows:
            f.write(f"{r['wav']}|{r['prompt']}\n")
    cmd = [
        python,
        str(sv_dir / "verification_pair_list_v2.py"),
        str(pairs),
        "--model_name",
        "wavlm_large",
        "--checkpoint",
        ckpt,
        "--scores",
        str(scores),
        "--wav1_start_sr",
        "0",
        "--wav2_start_sr",
        "0",
        "--wav1_end_sr",
        "-1",
        "--wav2_end_sr",
        "-1",
        "--device",
        device,
    ]
    _run(cmd, sv_dir)
    out: dict[str, float] = {}
    if scores.exists():
        for line in scores.read_text().splitlines():
            if "\t" not in line or line.startswith("avg score"):
                continue
            key, val = line.split("\t", 1)
            try:
                out[key.split("|", 1)[0]] = float(val)
            except ValueError:
                continue
    score_by_wav: dict[str, float] = {}
    for r in rows:
        for key, val in out.items():
            if key.startswith(f"{r['wav']}_"):
                score_by_wav[r["wav"]] = val
                break
    return score_by_wav


def main(argv: Sequence[str] | None = None) -> int:
    """Score Seed-TTS wavs (synth or ground truth) and emit a WER/SIM JSON manifest.

    WER and speaker-verification (SIM) metrics are measured by running the
    upstream ``seed-tts-eval`` scripts as subprocesses and parsing their file
    outputs, so the released normalisation/averaging semantics are preserved for
    reproducible TTS evaluations.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset-dir", required=True)
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--list", required=True, help="relative to <testset-dir>/<lang>")
    ap.add_argument("--synth-dir")
    ap.add_argument("--gt", action="store_true", help="use <testset-dir>/<lang>/wavs")
    ap.add_argument("--seed-code", required=True)
    ap.add_argument("--wavlm-ckpt", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workdir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sim", action="store_true")
    args = ap.parse_args(argv)

    if bool(args.synth_dir) == bool(args.gt):
        sys.stderr.write("seedtts_score: provide exactly one of --synth-dir or --gt\n")
        return 96

    lang_dir = Path(args.testset_dir) / args.lang
    meta_path = lang_dir / args.list
    seed_code = Path(args.seed_code).resolve()
    out_path = Path(args.out).resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    use_lines = meta_path.read_text().splitlines()
    if args.limit is not None:
        use_lines = use_lines[: args.limit]
    rows_expected = len([ln for ln in use_lines if ln.strip()])

    wav_root = Path(args.synth_dir) if args.synth_dir else lang_dir / "wavs"
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(use_lines, 1):
        if not line.strip():
            continue
        utt, _prompt_text, prompt_wav, target_text, _gt = _parse_meta(line, i)
        name = utt if utt.endswith(".wav") else f"{utt}.wav"
        utt_id = Path(name).stem
        wav = (lang_dir / "wavs" / name) if args.gt else (wav_root / f"{utt_id}.wav")
        if not wav.exists():
            missing.append(str(wav))
            continue
        prompt_abs = Path(prompt_wav) if Path(prompt_wav).is_absolute() else lang_dir / prompt_wav
        rows.append(
            {
                "id": utt_id,
                "wav": str(wav.resolve()),
                "target": target_text,
                "prompt": str(prompt_abs.resolve()),
            }
        )

    _run_wer(rows, seed_code, args.lang, args.python, workdir)
    scored: list[dict[str, Any]] = [r for r in rows if r.get("wer_record")]
    scored_wavs = {r["wav"] for r in scored}
    # Rows the upstream run_wer.py silently skipped (missing files etc.).
    missing.extend(r["wav"] for r in rows if r["wav"] not in scored_wavs)

    sims: dict[str, float] = {}
    if not args.no_sim:
        sims = _run_sim(scored, seed_code, args.wavlm_ckpt, args.python, workdir, args.device)

    all_rows: list[dict[str, Any]] = []
    for r in scored:
        rec = r["wer_record"]
        all_rows.append(
            {
                "id": r["id"],
                "reference": rec["reference"],
                "hypothesis": rec["hypothesis"],
                "errors": rec["errors"],
                "reference_length": rec["reference_length"],
                "wer": rec["wer"],
                "sim": sims.get(r["wav"]) * 100.0 if r["wav"] in sims else None,
            }
        )

    # average_wer.py averages the per-line ``wer`` column with np.mean × 100.
    corpus_wer = (
        round(sum(x["wer"] for x in all_rows) / len(all_rows) * 100, 3) if all_rows else 0.0
    )
    sim_vals = [x["sim"] for x in all_rows if x["sim"] is not None]
    corpus_sim = round(sum(sim_vals) / len(sim_vals), 3) if sim_vals else None

    result = {
        "lang": args.lang,
        "list": str(meta_path),
        "source": "gt" if args.gt else "synth",
        "n_rows": rows_expected,
        "rows_scored": len(all_rows),
        "missing": missing,
        "wer": corpus_wer,
        "sim": corpus_sim,
        "upstream": {
            "seed_code_commit": _git_head(seed_code),
            "wavlm_ckpt": str(args.wavlm_ckpt),
        },
        "all": all_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return 95 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
