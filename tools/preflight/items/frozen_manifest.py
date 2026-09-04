"""Item 1 -- frozen manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .._artifacts import (
    _manifest_hash_for,
    _read_safetensors_header,
    _sha256_and_lines,
)
from .._base import (
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
# Item 1 — frozen manifest
# ---------------------------------------------------------------------------


def _check_frozen_manifest(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 1: the run's denominators, frozen and hashed into one manifest.

    Verifies, against PINS IN THE CONFIG (nowhere else — item 1's final
    sentence), that: every declared model file exists with the pinned st_size;
    the safetensors headers across all model files contain exactly the pinned
    tensor count; every declared corpus file's streamed sha256 and wc-l line
    count match; the run config's sha256 matches. Then computes THE manifest
    hash that the banner, launch_provenance and any later checkpoint bind to.

    INPUT CONTRACT: generic (design text only). Model files MUST be
    safetensors — anything else is a BLOCK naming the file, never a guess at
    its tensor count.
    """
    expected_files = len(s["model"]["files"]) + len(s["corpus"]["files"]) + 1
    evidence: dict[str, Any] = {}
    checked = 0

    # Three failure classes reach this loop and only two verdict vocabularies
    # were previously used, so one class was misfiled. They are kept distinct:
    #   * absent or unparseable shard — a defect of the ARTIFACT under
    #     clearance: a FAIL this check exists to NAME ("your checkpoint is
    #     missing a shard" sends the operator to the artifact);
    #   * an OS-level refusal (EACCES, EIO, a file vanishing mid-sweep) — a
    #     defect of the CHECK's ability to run: ERROR, fail closed ("the
    #     preflight cannot read this filesystem" sends them to the machine).
    #     The model-file-missing MUST_FIRE lane caught the conflation by
    #     rightly demanding the NAMED verdict, not merely any blocking one.
    # stat() orders the evidence: FileNotFoundError is absence; any other
    # OSError is environmental; once stat succeeds, an ArtifactError from
    # _read_safetensors_header WITHOUT a chained OSError cause is a bytes-level
    # defect, while one carrying an OSError __cause__ is an open/read refusal
    # that arrived (or a file that vanished) after stat succeeded — still
    # environmental, because the filesystem moved under the check mid-sweep.
    mismatches: list[str] = []
    tensor_total = 0
    param_total = 0
    model_files_detail = []
    model_unexamined: list[dict[str, Any]] = []
    for rel in s["model"]["files"]:
        p = Path(rel)
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            # Counted as a mismatch, never as examined: 'checked' stays honest
            # AND 'expected' does not move — a FAIL that read 6 of 7 declared
            # artifacts says "6/7", it does not re-target its denominator.
            mismatches.append(
                f"declared model file absent: {rel} — the shard the frozen "
                f"manifest pins is not on disk"
            )
            model_unexamined.append({"path": rel, "state": "absent"})
            continue
        except OSError as exc:
            return _finalize(
                "frozen_manifest",
                "Frozen manifest",
                Verdict.ERROR,
                Coverage(checked, "artifact files", expected=expected_files),
                f"model artifact unreadable (environmental): {rel}: {exc}",
                evidence,
            )
        try:
            header = _read_safetensors_header(p)
        except ArtifactError as exc:
            if isinstance(exc.__cause__, OSError):
                return _finalize(
                    "frozen_manifest",
                    "Frozen manifest",
                    Verdict.ERROR,
                    Coverage(checked, "artifact files", expected=expected_files),
                    f"model artifact unreadable (environmental): {rel}: {exc}",
                    evidence,
                )
            mismatches.append(f"model shard corrupt or unparseable as safetensors: {exc}")
            model_unexamined.append({"path": rel, "state": "corrupt", "problem": str(exc)})
            # st_size is knowable even when the header won't parse; recording
            # the bytes keeps sum(st_size) literally true, while the tensor
            # pricing below (correctly) never sees this shard.
            model_files_detail.append(
                {"path": rel, "bytes": size, "tensors": None, "state": "corrupt"}
            )
            continue
        tensor_total += len(header)
        param_total += sum(t["numel"] for t in header.values())
        model_files_detail.append({"path": rel, "bytes": size, "tensors": len(header)})
        checked += 1
    observed_bytes = sum(d["bytes"] for d in model_files_detail)
    evidence["model"] = {
        "files": model_files_detail,
        "tensors_observed": tensor_total,
        "tensors_pinned": s["model"]["tensor_count"],
        "bytes_observed": observed_bytes,
        "bytes_pinned": s["model"]["total_bytes"],
        "header_param_sum": param_total,
        # Files declared but never priced: absent/corrupt shards land here so
        # the FAIL evidence names exactly what was NOT examined — 'checked'
        # counts only fully parsed units, so this list is where the under-count
        # stays legible to both humans and JSON consumers.
        "unexamined": model_unexamined,
    }
    if tensor_total != s["model"]["tensor_count"]:
        mismatches.append(f"tensor count {tensor_total} != pinned {s['model']['tensor_count']}")
    if observed_bytes != s["model"]["total_bytes"]:
        mismatches.append(f"sum(st_size) {observed_bytes} != pinned {s['model']['total_bytes']}")

    corpus_detail = []
    for entry in s["corpus"]["files"]:
        p = Path(entry["path"])
        try:
            sha, lines = _sha256_and_lines(p)
        except OSError as exc:
            return _finalize(
                "frozen_manifest",
                "Frozen manifest",
                Verdict.ERROR,
                Coverage(checked, "artifact files", expected=expected_files),
                f"corpus file unreadable: {entry['path']}: {exc}",
                evidence,
            )
        corpus_detail.append(
            {
                "path": entry["path"],
                "sha256": sha,
                "lines": lines,
                "sha_matches_pin": sha == entry["sha256"],
                "lines_match_pin": lines == entry["lines"],
            }
        )
        if sha != entry["sha256"]:
            mismatches.append(
                f"corpus sha256 mismatch on {entry['path']} "
                f"(pinned {entry['sha256'][:12]}…, observed {sha[:12]}…)"
            )
        if lines != entry["lines"]:
            mismatches.append(
                f"corpus line count {lines} != pinned {entry['lines']} on {entry['path']}"
            )
        checked += 1
    evidence["corpus"] = {"files": corpus_detail, "count": len(corpus_detail)}

    rc_path = Path(s["run_config"]["path"])
    try:
        rc_sha = hashlib.sha256(rc_path.read_bytes()).hexdigest()
    except OSError as exc:
        return _finalize(
            "frozen_manifest",
            "Frozen manifest",
            Verdict.ERROR,
            Coverage(checked, "artifact files", expected=expected_files),
            f"run config unreadable: {rc_path}: {exc}",
            evidence,
        )
    checked += 1
    evidence["run_config"] = {
        "path": str(rc_path),
        "sha256": rc_sha,
        "matches_pin": rc_sha == s["run_config"]["sha256"],
    }
    if rc_sha != s["run_config"]["sha256"]:
        mismatches.append(
            "run config sha256 mismatch — the recipe being launched is not "
            "the recipe that was frozen"
        )

    manifest_sha, payload = _manifest_hash_for(s, shared["_config_sha256"])
    evidence["manifest_sha256"] = manifest_sha
    evidence["config_sha256"] = shared["_config_sha256"]
    # Publish the run's ONLY sanctioned denominator source.
    shared["manifest_sha256"] = manifest_sha
    shared["manifest_payload"] = payload
    shared["corpus_files"] = [f["path"] for f in s["corpus"]["files"]]

    cov = Coverage(checked, "artifact files", expected=expected_files)
    if mismatches:
        return _finalize(
            "frozen_manifest",
            "Frozen manifest",
            Verdict.FAIL,
            cov,
            "; ".join(mismatches[:4]) + (" …" if len(mismatches) > 4 else ""),
            evidence,
        )
    return _finalize(
        "frozen_manifest",
        "Frozen manifest",
        Verdict.PASS,
        cov,
        f"all pins verified; manifest_sha256={manifest_sha[:16]}…",
        evidence,
    )
