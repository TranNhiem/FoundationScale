"""Item 6 -- LoRA probe (CONTRACT-BOUND)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _finalize,
)

# ---------------------------------------------------------------------------
# Item 6 — LoRA probe (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_lora_probe(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 6: the 20-iter LoRA probe actually attached, trained, and merged.

    INPUT CONTRACT (unverified against the FoxBrain repo):
      * lora.run_log: text log of the probe run. Must contain >0 lines
        matching 'Adding lora to', and >0 such lines naming EACH class in
        lora.target_classes (substring). Must contain a trainable-params line
        matching r'trainable[^\\n]*?params?[^\\n]*?([0-9][0-9,]*)' whose value
        sits inside lora.trainable_band.
      * lora.probe_metrics_jsonl: JSONL {"iteration": i, ...}; iterations must
        be exactly 1..lora.expected_iters complete; the max-iteration record
        must carry "lora_b_norm": {class: float} with every target class > 0.
      * lora.delta_audit_json: {class: {"delta_l2": >0, "tensors_checked":
        int>=1}} for EVERY target class — the merged Δ-audit, with its own
        per-class denominator.
      * lora.merged_dir: merged HF export. Parity = sum of st_size over every
        regular file beneath it MUST equal lora.pinned_merged_total_bytes.
        NEVER the dir's own index.json metadata_size/total_size: a self-index
        is the artifact attesting about itself, and the design says so in so
        many words.
    """
    evidence: dict[str, Any] = {}
    classes = list(s["target_classes"])
    expected = len(classes)

    try:
        log_text = Path(s["run_log"]).read_text(encoding="utf-8")
    except OSError as exc:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage.none("LoRA target classes"),
            f"run log unreadable: {s['run_log']}: {exc}",
            evidence,
        )
    lora_lines = [ln for ln in log_text.splitlines() if "Adding lora to" in ln]
    evidence["adding_lora_lines"] = len(lora_lines)
    if not lora_lines:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage.none("LoRA target classes"),
            "grep -c 'Adding lora to' == 0 — LoRA attached to nothing",
            evidence,
        )
    per_class_lines = {c: sum(1 for ln in lora_lines if c in ln) for c in classes}
    evidence["per_class_attach_lines"] = per_class_lines
    silent = [c for c, n in per_class_lines.items() if n == 0]
    if silent:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected - len(silent), "LoRA target classes", expected=expected),
            f"zero 'Adding lora to' lines name intended classes: {silent} — attached somewhere "
            f"else, or nowhere",
            evidence,
        )

    m = re.search(r"trainable[^\n]*?params?[^\n]*?([0-9][0-9,]*)", log_text)
    if not m:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            "no trainable-params line in the run log — the band was never evaluated",
            evidence,
        )
    trainable = int(m.group(1).replace(",", ""))
    lo, hi = s["trainable_band"]
    evidence["trainable_params"] = trainable
    evidence["trainable_band"] = [lo, hi]
    if not lo <= trainable <= hi:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"trainable params {trainable:,} outside band [{lo:,}, {hi:,}] — adapters are "
            f"not the size the layout implies",
            evidence,
        )

    try:
        probe_rows = [
            json.loads(ln)
            for ln in Path(s["probe_metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"probe metrics unreadable: {s['probe_metrics_jsonl']}: {exc}",
            evidence,
        )
    iters = sorted(
        {
            int(r["iteration"])
            for r in probe_rows
            if isinstance(r, dict) and isinstance(r.get("iteration"), int)
        }
    )
    evidence["probe_iterations_observed"] = iters
    if iters != list(range(1, int(s["expected_iters"]) + 1)):
        missing = sorted(set(range(1, int(s["expected_iters"]) + 1)) - set(iters))[:5]
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"probe ran iterations {iters[:3]}…{iters[-3:] if iters else ''} ({len(iters)} of "
            f"{s['expected_iters']}); missing e.g. {missing} — a {s['expected_iters']}-iter "
            f"claim over {len(iters)} iters is under-covered",
            evidence,
        )
    final_rec = max(
        (r for r in probe_rows if isinstance(r, dict) and isinstance(r.get("iteration"), int)),
        key=lambda r: r["iteration"],
    )
    bnorms = final_rec.get("lora_b_norm") if isinstance(final_rec, dict) else None
    if not isinstance(bnorms, dict):
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"final probe record (iter {final_rec.get('iteration')}) carries no lora_b_norm map "
            f"— B>0 was asserted, never measured",
            evidence,
        )
    evidence["lora_b_norm"] = bnorms
    b_bad = [
        c
        for c in classes
        if not isinstance(bnorms.get(c), (int, float))
        or isinstance(bnorms.get(c), bool)
        or bnorms.get(c) <= 0
    ]
    if b_bad:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected - len(b_bad), "LoRA target classes", expected=expected),
            f"lora_b_norm not > 0 for classes {b_bad} — zero-init survived the probe; the "
            f"B>0 assertion exists because exactly this once shipped silently",
            evidence,
        )

    try:
        delta = json.loads(Path(s["delta_audit_json"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"delta audit unreadable: {s['delta_audit_json']}: {exc}",
            evidence,
        )
    evidence["delta_audit"] = delta if isinstance(delta, dict) else {}
    d_bad = []
    for c in classes:
        rec = delta.get(c) if isinstance(delta, dict) else None
        if (
            not isinstance(rec, dict)
            or not isinstance(rec.get("delta_l2"), (int, float))
            or rec.get("delta_l2", 0) <= 0
        ):
            d_bad.append(f"{c} (delta_l2 missing or <= 0)")
        elif not isinstance(rec.get("tensors_checked"), int) or rec.get("tensors_checked", 0) < 1:
            d_bad.append(
                f"{c} (tensors_checked denominator missing or 0 — an unqualified 'Δ nonzero')"
            )
    if d_bad:
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected - len(d_bad), "LoRA target classes", expected=expected),
            f"merged Δ-audit failed for: {', '.join(d_bad)}",
            evidence,
        )

    merged = Path(s["merged_dir"])
    if not merged.is_dir():
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.ERROR,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"merged dir not present: {merged}",
            evidence,
        )
    observed = 0
    n_files = 0
    for dirpath, _dirnames, filenames in os.walk(merged):
        for fn in filenames:
            p = Path(dirpath) / fn
            # Deliberately stat ONLY: any model.safetensors.index.json in this
            # walk is summed as BYTES-ON-DISK, never parsed for its claimed
            # total_size. Design: parity is vs. the pinned 14.89 GiB — never
            # vs. self-index.
            observed += p.stat().st_size
            n_files += 1
    evidence["merged"] = {
        "dir": str(merged),
        "files": n_files,
        "observed_bytes": observed,
        "pinned_total_bytes": s["pinned_merged_total_bytes"],
        "comparison_source": "external pin (sum of st_size), never self-index",
    }
    if observed != int(s["pinned_merged_total_bytes"]):
        return _finalize(
            "lora_probe",
            "LoRA probe",
            Verdict.FAIL,
            Coverage(expected, "LoRA target classes", expected=expected),
            f"merged HF bytes {observed:,} != pinned {int(s['pinned_merged_total_bytes']):,} "
            f"across {n_files} files",
            evidence,
        )

    return _finalize(
        "lora_probe",
        "LoRA probe",
        Verdict.PASS,
        Coverage(expected, "LoRA target classes", expected=expected),
        f"{expected}/{expected} classes attached, B>0, Δ>0; {len(iters)}/"
        f"{s['expected_iters']} iters; trainable {trainable:,} in band; merged "
        f"{observed:,} B == pin",
        evidence,
    )
