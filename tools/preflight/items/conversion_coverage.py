"""Item 5 -- conversion coverage (CONTRACT-BOUND)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._artifacts import (
    _read_safetensors_header,
)
from .._base import (
    _MISSING,
    _SAFETENSORS_DTYPE_BYTES,
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _finalize,
)
from .._errors import (
    ArtifactError,
)

# ---------------------------------------------------------------------------
# Item 5 — conversion coverage (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_conversion_coverage(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 5: every one of the pinned tensors is converted or EXPLICITLY allow-listed.

    INPUT CONTRACT (unverified against the FoxBrain / Megatron-Bridge repo):
      * frozen.model.files (safetensors) supply the tensor NAMESPACE via
        headers; the count must re-equal the manifest pin here too (denomina-
        tors source from the manifest, but re-deriving them at the point of
        use is the proof they were not swapped).
      * conversion.coverage_map_json: {"tensors": [...]} produced by the
        Megatron-Bridge conversion probe, one entry per name:
          {"name": str, "bytes": int, "coverage": "converted"}             — or —
          {"name": str, "coverage": "allowlist", "rule": "tied"|"shared_kv"}
        'converted' names must exist in the headers with EQUAL stored bytes.
        'allowlist' names may be absent from headers (they share storage) but
        their rule must be GROUNDED: 'tied' requires the HF config key
        conversion.tied_grounding (e.g. tie_word_embeddings) == true;
        'shared_kv' requires conversion.shared_kv_grounding (the E4B key a
        human must supply; null ⇒ any shared_kv entry FAILS). Unknown rules
        FAIL: an allow-list entry nobody can ground is an unexamined tensor
        wearing permission.
      * conversion.hf_config_json: E4B config.json. conversion.divisibility is
        the design's "settles TP-divisibility, MoE-block pattern, EP=4
        divisibility": a list of {"field": dotted.key, "divisible_by": N} or
        {"field": ..., "equals": X} assertions, each evaluated here.
      * conversion.iter_metrics_jsonl: JSONL records; the iteration==1 record
        must carry "loss" within conversion.iter1_loss_band (design pins
        ≈[1.0, 3.0]); exactly one record must carry "param_count", and it must
        equal conversion.expected_param_count (the pinned ~8.0e9 arithmetic).
    """
    evidence: dict[str, Any] = {}

    # -- namespace from headers ------------------------------------------------
    headers: dict[str, dict[str, Any]] = {}
    for rel in cfg["frozen"]["model"]["files"]:
        try:
            for name, meta in _read_safetensors_header(Path(rel)).items():
                headers[name] = meta
        except ArtifactError as exc:
            return _finalize(
                "conversion_coverage",
                "Conversion coverage",
                Verdict.ERROR,
                Coverage.none("tensors"),
                str(exc),
                evidence,
            )
    expected = len(headers)
    pinned = cfg["frozen"]["model"]["tensor_count"]
    evidence["header_tensor_count"] = expected
    evidence["pinned_tensor_count"] = pinned
    if expected != pinned:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            Coverage(0, "tensors", expected=pinned),
            f"safetensors headers name {expected} tensors; the frozen manifest pins "
            f"{pinned} — the model under test is not the model that was frozen",
            evidence,
        )

    # -- HF config facts ---------------------------------------------------------
    hf_path = Path(s["hf_config_json"])
    try:
        hf = json.loads(hf_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"HF config unreadable: {hf_path}: {exc}",
            evidence,
        )
    if not isinstance(hf, dict):
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"{hf_path} is not a JSON object",
            evidence,
        )

    def dotted(key: str) -> Any:
        node: Any = hf
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return _MISSING
            node = node[part]
        return node

    _MISSING_LOCAL = _MISSING
    assertion_failures: list[str] = []
    assertion_count = 0
    for assertion in s["divisibility"]:
        assertion_count += 1
        value = dotted(assertion["field"])
        if value is _MISSING_LOCAL:
            assertion_failures.append(
                f"HF config has no key {assertion['field']!r} — assertion unevaluable"
            )
            continue
        if "divisible_by" in assertion:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value % assertion["divisible_by"] != 0
            ):
                assertion_failures.append(
                    f"{assertion['field']}={value!r} is not divisible by "
                    f"{assertion['divisible_by']}"
                )
        else:
            if value != assertion["equals"]:
                assertion_failures.append(
                    f"{assertion['field']}={value!r} != required {assertion['equals']!r}"
                )
    evidence["divisibility_assertions"] = {"count": assertion_count, "failures": assertion_failures}

    # -- coverage map -------------------------------------------------------------
    cm_path = Path(s["coverage_map_json"])
    try:
        coverage_map = json.loads(cm_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"coverage map unreadable: {cm_path}: {exc}",
            evidence,
        )
    entries = coverage_map.get("tensors") if isinstance(coverage_map, dict) else None
    if not isinstance(entries, list):
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(0, "tensors", expected=expected),
            f"{cm_path} carries no 'tensors' list — the map covers nothing",
            evidence,
        )

    covered: set[str] = set()
    allowlist_count = 0
    uncovered: list[str] = []
    problems: list[str] = []
    by_name: dict[str, dict] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("name"), str):
            by_name[e["name"]] = e
    for name in by_name:
        if name not in headers and by_name[name].get("coverage") == "converted":
            problems.append(
                f"map converts {name!r}, which no model header declares — "
                f"numerator outruns denominator"
            )
    for name, meta in headers.items():
        e = by_name.get(name)
        if e is None:
            uncovered.append(name)
            continue
        if e.get("coverage") != "converted":
            uncovered.append(f"{name} (map entry is {e.get('coverage')!r}, not 'converted')")
            continue
        want = meta["numel"] * _SAFETENSORS_DTYPE_BYTES[meta["dtype"]]
        if e.get("bytes") != want:
            problems.append(f"{name}: map claims {e.get('bytes')} bytes; header implies {want}")
            continue
        covered.add(name)
    for name, e in by_name.items():
        if e.get("coverage") != "allowlist":
            continue
        rule = e.get("rule")
        allowlist_count += 1
        if rule == "tied":
            ground = dotted(s["tied_grounding"])
            if ground is not True:
                problems.append(
                    f"allow-listed {name!r} under rule 'tied', but HF config "
                    f"{s['tied_grounding']!r} is {ground!r} — the ground is absent or false"
                )
        elif rule == "shared_kv":
            key = s.get("shared_kv_grounding")
            ground = dotted(key) if key else _MISSING_LOCAL
            if ground is not True:
                problems.append(
                    f"allow-listed {name!r} under rule 'shared_kv', but grounding key "
                    f"{key!r} is {None if key is None else ground!r} — "
                    f"declare and confirm the E4B key"
                )
        else:
            problems.append(
                f"allow-listed {name!r} under unknown rule {rule!r} — no ground exists for it"
            )
    evidence["converted"] = len(covered)
    evidence["allowlisted"] = allowlist_count
    evidence["uncovered_count"] = len(uncovered)
    evidence["uncovered_sample"] = uncovered[:8]

    # -- iter-1 band + param echo ---------------------------------------------------
    try:
        metrics = [
            json.loads(line)
            for line in Path(s["iter_metrics_jsonl"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(len(covered), "tensors", expected=expected),
            f"iter metrics unreadable: {s['iter_metrics_jsonl']}: {exc}",
            evidence,
        )
    evidence["metrics_records"] = len(metrics)
    iter1 = [r for r in metrics if isinstance(r, dict) and r.get("iteration") == 1]
    if not iter1:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            Coverage(len(covered), "tensors", expected=expected),
            f"no iteration==1 record in {len(metrics)} metric rows — the loss band was "
            f"never evaluated, and absent evidence is not in-band evidence",
            evidence,
        )
    lo, hi = s["iter1_loss_band"]
    loss1 = iter1[0].get("loss")
    evidence["iter1_loss"] = loss1
    evidence["iter1_band"] = [lo, hi]
    if not isinstance(loss1, (int, float)) or isinstance(loss1, bool):
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.ERROR,
            Coverage(len(covered), "tensors", expected=expected),
            f"iter-1 loss is not numeric: {loss1!r}",
            evidence,
        )
    if len(iter1) > 1:
        problems.append(
            f"{len(iter1)} records claim iteration==1 — a duplicated band row contradicts itself"
        )
    echoes = [r["param_count"] for r in metrics if isinstance(r, dict) and "param_count" in r]
    evidence["param_echoes"] = echoes
    if not echoes:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            Coverage(len(covered), "tensors", expected=expected),
            f"no param_count echo in {len(metrics)} metric rows — the ~8.0e9 pin was "
            f"never compared against the trainer's own count",
            evidence,
        )
    if len(set(echoes)) != 1:
        problems.append(
            f"param_count echo is self-contradictory across rows: {sorted(set(echoes))!r}"
        )

    band_ok = lo <= loss1 <= hi
    param_ok = len(set(echoes)) == 1 and echoes[0] == s["expected_param_count"]
    cov = Coverage(len(covered), "tensors", expected=expected)
    reasons = []
    if uncovered:
        reasons.append(
            f"{len(uncovered)} of {expected} header tensors are neither converted nor allow-listed"
        )
    if problems:
        reasons.append("; ".join(problems[:3]) + (" …" if len(problems) > 3 else ""))
    if assertion_failures:
        reasons.append(
            f"{len(assertion_failures)}/{assertion_count} divisibility assertions failed"
        )
    if not band_ok:
        reasons.append(f"iter-1 loss {loss1} outside pinned band [{lo}, {hi}]")
    if not param_ok and not problems:
        reasons.append(f"param echo {echoes[0]} != pinned {s['expected_param_count']}")
    if reasons:
        return _finalize(
            "conversion_coverage",
            "Conversion coverage",
            Verdict.FAIL,
            cov,
            " | ".join(reasons),
            evidence,
        )
    return _finalize(
        "conversion_coverage",
        "Conversion coverage",
        Verdict.PASS,
        cov,
        f"{len(covered)}/{expected} tensors converted ({allowlist_count} grounded allow-list "
        f"entries); iter-1 loss {loss1} in [{lo}, {hi}]; param echo == {s['expected_param_count']}",
        evidence,
    )
