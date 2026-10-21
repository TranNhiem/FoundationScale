"""Judge one RTN compression measurement against its plan's ``compress`` block.
Rules CP-IN-005, CP-HO-003, CP-IN-006, CP-HO-001/002/006, CP-IN-004 (budget) and CP-HO-011 run in that order until
the first hit.
Pure stdlib; the only I/O is reading the worker's results JSON file.
"""

from __future__ import annotations

import json
import math

EFFICIENCY_NOTE = (
    "CP-HO-005: efficiency (tok/s, latency, memory, energy) is not measured by v1; "
    "quantize-dequantize keeps bf16 weights, so no speed claim follows (W3)."
)
_SCHEMA = "fs.compress.rtn_results.v1"


def _finite(value) -> bool:
    """True for a real, non-bool number that math.isfinite accepts."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _load(results_path):
    """Read the worker's results JSON; None when missing or unparseable."""
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def judge_results(compress: dict, results_path) -> dict:
    """Verify verdict of one RTN measurement against its plan."""
    res = _load(results_path)

    def out(verdict, rule, note=None):
        report = None
        if verdict != "REFUSED":
            report = {"model": compress.get("model"), "bits": compress.get("bits"),
                      "group_size": compress.get("group_size"),
                      "budget_ppl_ratio": compress.get("budget_ppl_ratio"),
                      "base_ppl": res.get("base_ppl_a"),
                      "compressed_ppl": res.get("compressed_ppl"),
                      "ppl_ratio": res.get("ppl_ratio"),
                      "instrument_certified": bool(res.get("instrument_certified")),
                      "instrument_rel_diff": res.get("instrument_rel_diff"),
                      "effective_bits_total": res.get("effective_bits_total"),
                      "effective_bits_quantized": res.get("effective_bits_quantized"),
                      "changed_fraction": res.get("changed_fraction"),
                      "eval_tokens": res.get("eval_tokens"), "text_sha256": res.get("text_sha256"),
                      "gpu_name": res.get("gpu_name"), "gpu_uuid": res.get("gpu_uuid"),
                      "hostname": res.get("hostname"), "efficiency": "UNMEASURED",
                      "verdict": verdict}
        return {"verdict": verdict, "rule": rule, "note": note, "report": report}

    if res is None or res.get("schema") != _SCHEMA:
        return out("REFUSED", "CP-IN-005", "results missing, unreadable JSON, or bad schema")

    if res.get("status") == "UNMEASURED":
        return out("UNMEASURED", "CP-HO-003", str(res.get("reason") or "")[:400])

    differ = [k for k in ("model", "bits", "group_size", "text_sha256")
              if res.get(k) != compress.get(k)]
    if differ:
        return out("REFUSED", "CP-IN-006", "plan-identity mismatch: " + ", ".join(differ))

    if res.get("instrument_certified") is not True:
        diff = res.get("instrument_rel_diff")
        return out("UNMEASURED", "CP-HO-001",
                   f"instrument_certified is not True; instrument_rel_diff={diff!r}")

    base = res.get("base_ppl_a")
    if not _finite(base) or base <= 1:
        return out("RED", "CP-HO-002", f"base_ppl_a invalid: {base!r}")

    quant, changed = res.get("quantized_linears"), res.get("changed_fraction")
    if quant is None or quant == 0 or changed is None or changed == 0:
        return out("RED", "CP-HO-006",
                   f"quantized_linears={quant!r}, changed_fraction={changed!r}")

    ratio, budget = res.get("ppl_ratio"), compress.get("budget_ppl_ratio")
    if not _finite(budget) or not 1.0 < budget <= 2.0:  # an edited spec cannot drop the pre-registered budget
        return out("REFUSED", "CP-IN-004", f"plan budget_ppl_ratio={budget!r} is not in (1.0, 2.0]")
    if not _finite(ratio) or ratio > budget:
        return out("RED", "CP-HO-011", f"ppl_ratio={ratio!r} vs budget_ppl_ratio={budget!r}")

    return out("PASS", None)
