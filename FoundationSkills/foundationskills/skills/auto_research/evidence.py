"""Evidence-bound trial results (M6): named evidence in, declared derivations out."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

EVIDENCE_FIELDS = ("evidence", "evidence_class", "metric_sources", "derived")

_RL_MANIFEST = "fskills_rl_manifest.json"
_PREFERENCE_MANIFEST = "fskills_preference_manifest.json"
_SFT_MANIFEST = "run_manifest.json"
_LAUNCH_LOG = "fskills_launch.log"
_DONE_RE = re.compile(r"\[fs:train:done\]\s+(\S+)")
_SAVED_RE = re.compile(r"^saved:\s+(\S+)$")
_TOLERANCE = 1e-9


class EvidenceError(ValueError):
    """Named evidence-integrity failure carrying its doctrine rule id."""

    def __init__(self, rule_id: str, message: str, detail: dict | None = None) -> None:
        super().__init__(f"{rule_id}: {message}")
        self.rule_id = rule_id
        self.message = message
        self.detail = dict(detail) if detail else {}


def sha256_file(path: Any) -> str:
    """Digest of a file's bytes as "'sha256:<64 hex>'"."""
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def rehash_problems(result: dict) -> list[str]:
    """Close-time (D3) re-hash of every bound evidence file; non-bound -> []."""
    evidence = result.get("evidence") if isinstance(result, dict) else None
    if not isinstance(evidence, dict) or result.get("evidence_class") != "bound":
        return []
    records = [r for r in (evidence.get("run_manifests") or []) if isinstance(r, dict)]
    if isinstance(evidence.get("eval_report"), dict):
        records.append(evidence["eval_report"])
    records += [r for r in (evidence.get("launch_logs") or []) if isinstance(r, dict)]
    problems: list[str] = []
    for record in records:
        recorded = str(record.get("sha256", ""))
        sha12 = recorded.split(":", 1)[-1][:12]
        try:
            current = sha256_file(record.get("path"))
        except (OSError, TypeError, ValueError):
            problems.append(f"evidence_file_missing:{sha12}")
            continue
        if current != recorded:
            problems.append(f"evidence_file_changed:{sha12}")
    return problems


def derive_result(
    identity: dict,
    evidence: dict,
    *,
    metric_names: Sequence[str],
    rl_floor: float | None = None,
) -> dict:
    """Derive one complete, evidence-bound trial result.

    A manifest whose checkpoint is not a "'saved: <path>'" record (or an SFT
    run without <output_dir>/final on disk) degrades its row to "unmeasured";
    there is then no trained checkpoint the eval report must bind to (D2/D3).
    Every other eval/manifest mismatch is refused as AR-IN-012.
    """
    _check_identity(identity)
    manifest_paths, eval_path = _check_evidence(evidence)
    manifests = [_parse_manifest(path) for path in manifest_paths]
    report = _parse_eval(eval_path)
    _bind_checkpoints(report, [m["checkpoint"] for m in manifests])
    _check_floor(manifests, rl_floor)

    unmeasured = report["status"] == "unmeasured" or any(m["status"] == "unmeasured" for m in manifests) or any(
        m["kind"] == "rl" and _is_number(m["measured_fraction"]) and _is_number(m["min_measured_fraction"])
        and m["measured_fraction"] < m["min_measured_fraction"] for m in manifests)
    status = "unmeasured" if unmeasured else "ok"
    steps = sum(m["steps"] for m in manifests)
    derived_scalars = {"status": status, "limited": report["limited"],
                       "steps": steps, "eval_policy_fingerprint": report["fingerprint"]}
    claimed_metrics = identity.get("metrics") or {}
    derivable, rl_values, rl_floors = _derivable_metrics(manifests, report)
    _check_claimed_metrics(claimed_metrics, derivable)
    _check_claimed_scalars(identity, derived_scalars)
    metrics, metric_sources = _assemble_metrics(metric_names, claimed_metrics, derivable)
    return {
        "trial": identity["trial"],
        "role": identity["role"],
        "seed": identity["seed"],
        "status": status,
        "limited": report["limited"],
        "steps": steps,
        "eval_policy_fingerprint": report["fingerprint"],
        "metrics": metrics,
        "evidence_class": "bound",
        "evidence": {"run_manifests": [m["record"] for m in manifests],
                     "eval_report": report["record"],
                     "launch_logs": [r for m in manifests for r in m["launch_logs"]]},
        "metric_sources": metric_sources,
        "derived": {"measured_fraction": min(rl_values) if rl_values else None,
                    "min_measured_fraction": min(rl_floors) if rl_floors else None,
                    "checkpoint": manifests[-1]["checkpoint"] if manifests else None},
    }


def _check_identity(identity: dict) -> None:
    """trial/role/seed are required; asserted metrics must be a mapping."""
    if not isinstance(identity, dict):
        raise EvidenceError("AR-IN-011", "evidence_malformed:identity_not_dict", {"identity": identity})
    for key in ("trial", "role", "seed"):
        if key not in identity:
            raise EvidenceError("AR-IN-011", f"evidence_malformed:identity_missing_{key}", {"field": key})
    claimed = identity.get("metrics")
    if claimed is not None and not isinstance(claimed, dict):
        raise EvidenceError("AR-IN-011", "evidence_malformed:metrics_not_dict", {"metrics": claimed})


def _check_evidence(evidence: dict) -> tuple[list[str], str]:
    """Named evidence: manifest path list plus the required eval report path."""
    if not isinstance(evidence, dict):
        raise EvidenceError("AR-IN-011", "evidence_malformed:evidence_not_dict", {"evidence": evidence})
    manifest_paths = evidence.get("run_manifests", [])
    if (not isinstance(manifest_paths, list)
            or not all(isinstance(p, str) and p for p in manifest_paths)):
        raise EvidenceError("AR-IN-011", "evidence_malformed:run_manifests_not_list_of_str",
                            {"run_manifests": manifest_paths})
    eval_path = evidence.get("eval_report")
    if not isinstance(eval_path, str) or not eval_path:
        raise EvidenceError("AR-IN-011", "evidence_malformed:eval_report_absent", {"eval_report": eval_path})
    return list(manifest_paths), eval_path


def _read_json_object(path: Any) -> dict:
    """Load one named JSON evidence file, or raise AR-IN-011."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError("AR-IN-011", f"evidence_file_unreadable:{path}",
                            {"path": str(path), "error": str(exc)}) from exc
    if not isinstance(payload, dict):
        raise EvidenceError("AR-IN-011", f"evidence_file_unreadable:{path}",
                            {"path": str(path), "error": "not_a_json_object"})
    return payload


def _manifest_kind(filename: str) -> str:
    """Manifest kind from its schema filename; unrecognised -> AR-IN-011."""
    if filename == _RL_MANIFEST:
        return "rl"
    if filename == _PREFERENCE_MANIFEST:
        return "preference"
    if filename == _SFT_MANIFEST:
        return "sft"
    raise EvidenceError("AR-IN-011", f"evidence_manifest_unrecognised:{filename}", {"filename": filename})


def _verdict_status(verdict: Any, label: str) -> str:
    """Recordable verdict -> 'ok' / 'unmeasured'; anything else -> AR-IN-011."""
    if verdict == "PASS":
        return "ok"
    if isinstance(verdict, str) and verdict.startswith("UNMEASURED"):
        return "unmeasured"
    raise EvidenceError("AR-IN-011", f"{label}_verdict_not_recordable:{verdict}", {"verdict": verdict})


def _parse_manifest(path: str) -> dict:
    """Parse one training manifest into its record and derivation inputs."""
    kind = _manifest_kind(Path(path).name)
    manifest = _read_json_object(path)
    launch_logs: list = []
    checkpoint = None
    steps = 0
    measured = None
    min_floor = None
    if kind in ("rl", "preference"):
        status = _verdict_status(manifest.get("status"), "training")
        raw_steps = manifest.get("steps_measured")
        if not _is_int(raw_steps):  # absent/mistyped is never read as 0 (declared, never guessed)
            raise EvidenceError("AR-IN-011", f"evidence_malformed:steps_measured:{raw_steps}", {"path": path})
        steps = int(raw_steps)
        saved = manifest.get("checkpoint")
        match = _SAVED_RE.match(saved) if isinstance(saved, str) else None
        if match and Path(match.group(1)).is_dir():
            checkpoint = match.group(1)
        else:
            status = "unmeasured"  # no saved checkpoint on disk: nothing trained can be bound
        if kind == "rl":
            measured = manifest.get("measured_fraction")
            min_floor = manifest.get("min_measured_fraction")
    else:
        verdict, log_record = _sft_launch_verdict(Path(path))
        launch_logs.append(log_record)
        status = _verdict_status(verdict, "training")
        steps = _sft_steps(manifest)
        final_dir = _sft_final_dir(manifest)
        if final_dir is not None and final_dir.is_dir():
            checkpoint = str(final_dir)
        else:
            status = "unmeasured"
    return {
        "record": {"path": path, "sha256": sha256_file(path), "kind": kind, "status": status},
        "status": status,
        "kind": kind,
        "steps": steps,
        "checkpoint": checkpoint,
        "measured_fraction": measured,
        "min_measured_fraction": min_floor,
        "launch_logs": launch_logs,
    }


def _sft_launch_verdict(manifest_path: Path) -> tuple[str, dict]:
    """Last '[fs:train:done]' verdict token in the sibling fskills_launch.log."""
    log_path = manifest_path.parent / _LAUNCH_LOG
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceError("AR-IN-011", f"evidence_file_unreadable:{log_path}",
                            {"path": str(log_path), "error": str(exc)}) from exc
    tokens = _DONE_RE.findall(text)
    if not tokens:
        raise EvidenceError("AR-IN-011", "training_verdict_not_recordable:fs:train:done_line_missing",
                            {"path": str(log_path)})
    return tokens[-1], {"path": str(log_path), "sha256": sha256_file(log_path)}


def _sft_steps(manifest: dict) -> int:
    """Measured steps from telemetry, else the configured maximum, else 0."""
    for source, key in (("telemetry", "perf_steps_observed"), ("config", "max_steps")):
        block = manifest.get(source)
        candidate = block.get(key) if isinstance(block, dict) else None
        if _is_int(candidate):
            return int(candidate)
    return 0


def _sft_final_dir(manifest: dict) -> Path | None:
    """<artifact_paths.output_dir>/final, or None when output_dir is undeclared."""
    artifact_paths = manifest.get("artifact_paths")
    output_dir = artifact_paths.get("output_dir") if isinstance(artifact_paths, dict) else None
    if isinstance(output_dir, str) and output_dir:
        return Path(output_dir) / "final"
    return None


def _parse_eval(path: str) -> dict:
    """Parse the eval report: verdict, declared limits, fingerprint, binding fields."""
    report = _read_json_object(path)
    status = _verdict_status(report.get("verdict"), "eval")
    return {
        "status": status,
        "limited": _eval_limited(report),
        "fingerprint": _eval_fingerprint(report),
        "checkpoint": report.get("checkpoint") if isinstance(report.get("checkpoint"), str) else None,
        "base": report.get("base") if isinstance(report.get("base"), str) else None,
        "benchmarks": report.get("benchmarks") if isinstance(report.get("benchmarks"), list) else [],
        "record": {"path": path, "sha256": sha256_file(path),
                   "verdict": report.get("verdict"), "checkpoint": report.get("checkpoint")},
    }


def _eval_limited(report: dict) -> bool:
    """D4: only the eval report's declared limit marker counts."""
    if "limited" not in report:
        return False
    value = report["limited"]
    if isinstance(value, bool):
        return value
    raise EvidenceError("AR-IN-011", f"eval_limited_not_bool:{value}", {"value": value})


def _eval_fingerprint(report: dict) -> str:
    """Policy fingerprint, always 'sha256:'-prefixed."""
    policy = report.get("policy")
    fingerprint = policy.get("fingerprint") if isinstance(policy, dict) else None
    if not isinstance(fingerprint, str) or not fingerprint:
        raise EvidenceError("AR-IN-011", "eval_policy_fingerprint_missing:policy.fingerprint",
                            {"path": report.get("path")})
    return fingerprint if fingerprint.startswith("sha256:") else f"sha256:{fingerprint}"


def _bind_checkpoints(report: dict, checkpoints: list) -> None:
    """Realpath-bind the eval report to the trained (or base) checkpoint."""
    if checkpoints:
        trained = checkpoints[-1]
        if trained is None:
            return
        bound_to, rule = trained, "eval_not_bound_to_trained_checkpoint"
    else:
        bound_to, rule = report["base"], "eval_of_trained_checkpoint_without_manifest"
    if not _same_path(report["checkpoint"], bound_to):
        raise EvidenceError("AR-IN-012", rule,
                            {"eval_checkpoint": report["checkpoint"], "bound_to": bound_to})


def _check_floor(manifests: list, rl_floor: float | None) -> None:
    """rl campaigns must have trained under the declared floor (AR-IN-012 else)."""
    if rl_floor is None:
        return
    for entry in manifests:
        if entry["kind"] != "rl":
            continue
        value = entry["min_measured_fraction"]
        if not _is_number(value) or abs(value - rl_floor) > _TOLERANCE:
            raise EvidenceError("AR-IN-012", f"trained_under_different_floor:{value}",
                                {"min_measured_fraction": value, "rl_floor": rl_floor})


def _derivable_metrics(manifests: list, report: dict) -> tuple[dict, list, list]:
    """Metric payloads derivable from evidence, plus rl ledger aggregates."""
    derivable: dict[str, tuple[dict, str]] = {}
    for entry in report["benchmarks"]:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            for part in ("score", "stderr"):
                if entry.get(part) is not None and not _is_number(entry[part]):
                    raise EvidenceError("AR-IN-011", f"evidence_malformed:{entry['name']}.{part}:{entry[part]}",
                                        {"benchmark": entry["name"]})
            derivable[f"{entry['name']}_acc"] = (
                {"value": entry.get("score"), "se": entry.get("stderr")}, "eval_report")
    rl_values = [float(m["measured_fraction"]) for m in manifests
                 if m["kind"] == "rl" and _is_number(m["measured_fraction"])]
    rl_floors = [float(m["min_measured_fraction"]) for m in manifests
                 if m["kind"] == "rl" and _is_number(m["min_measured_fraction"])]
    if rl_values:
        derivable["rl_measured_fraction"] = ({"value": min(rl_values), "se": 0.0}, "run_manifest")
    return derivable, rl_values, rl_floors


def _check_claimed_metrics(claimed: dict, derivable: dict) -> None:
    """Refuse caller metrics that contradict derivable ones (AR-IN-012)."""
    for name, payload in claimed.items():
        if name not in derivable:
            continue
        entry = derivable[name][0]
        contradicts = not isinstance(payload, dict) or any(
            part in payload and _differs(payload[part], entry[part]) for part in ("value", "se"))
        if contradicts:
            raise EvidenceError("AR-IN-012", f"assertion_contradicts_evidence:metrics:{name}",
                                {"metric": name, "claimed": payload, "derived": entry})


def _check_claimed_scalars(identity: dict, derived: dict) -> None:
    """Refuse caller scalars that contradict derived ones (AR-IN-012)."""
    for field, value in derived.items():
        claimed = identity.get(field, None)
        if claimed is not None and claimed != value:
            raise EvidenceError("AR-IN-012", f"assertion_contradicts_evidence:{field}",
                                {"field": field, "claimed": claimed, "derived": value})


def _assemble_metrics(metric_names: Sequence[str], claimed: dict, derivable: dict) -> tuple[dict, dict]:
    """Derived payloads win; caller-supplied extras stay asserted; anything else is absent (value None)."""
    metrics: dict[str, dict] = {}
    metric_sources: dict[str, str] = {}
    ordered = [str(name) for name in metric_names]
    ordered += [name for name in claimed if name not in ordered]
    for name in ordered:
        if name in derivable:
            metrics[name], metric_sources[name] = derivable[name]
        elif name in claimed:
            metrics[name], metric_sources[name] = claimed[name], "asserted"
        else:
            metrics[name], metric_sources[name] = {"value": None, "se": None}, "absent"
    return metrics, metric_sources


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _differs(claimed: Any, derived: Any) -> bool:
    """Whether an asserted value contradicts a derived one beyond 1e-9."""
    if claimed is None or derived is None:
        return claimed is not None or derived is not None
    try:
        return abs(float(claimed) - float(derived)) > _TOLERANCE
    except (TypeError, ValueError):
        return True


def _same_path(left: Any, right: Any) -> bool:
    return (isinstance(left, str) and isinstance(right, str)
            and os.path.realpath(left) == os.path.realpath(right))
