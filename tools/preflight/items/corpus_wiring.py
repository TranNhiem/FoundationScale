"""Item 3 -- corpus wiring (CONTRACT-BOUND)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .._artifacts import (
    _canonical_sample_sha256,
)
from .._base import (
    Coverage,
    Verdict,
)
from .._core import (
    CheckResult,
    _Check,
    _finalize,
    _shared_or_error,
    _stub_fn,
)
from .._errors import (
    ArtifactError,
)

# ---------------------------------------------------------------------------
# Item 3 — corpus wiring (CONTRACT-BOUND)
# ---------------------------------------------------------------------------


def _check_corpus_wiring(cfg, s, env, shared, registry=None) -> CheckResult:
    """Item 3: the corpus the launcher EXPORTS is the corpus the manifest PINNED.

    INPUT CONTRACT (unverified against the FoxBrain repo):
      * env[corpus_wiring.env_var] (design: FOXBRAIN_SFT_JSONLS): os.pathsep-
        separated list of JSONL paths IN TRAINING ORDER; entry 0 is the
        batch-0 source. Empty entries refused.
      * corpus_wiring.recipe_files: at least one must textually call
        ``_env_jsonls(`` — a grep only: it shows the code path names the env
        reader, NOT that the call is live on every branch. A human must
        confirm that before first launch.
      * corpus_wiring.attestation_path: JSON {"reader": str, "sample_sha256":
        hex, "note": str?}. The sample hash is MACHINE-VERIFIED: we decode the
        first row of the first resolved corpus file and require equality — a
        hash over a different sample or a file that changed since FAILS.
        'reader' itself cannot be verified; the artifact pins WHO asserted it.
    """
    err = _shared_or_error(_Check("corpus_wiring", "", None, _stub_fn), shared, ["corpus_files"])
    if err:
        return err

    evidence: dict[str, Any] = {}
    raw = env.get(s["env_var"])
    if raw is None:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage.none("corpus files"),
            f"--export does not carry {s['env_var']}: the corpus list is not in "
            f"the launch environment",
            evidence,
        )
    resolved = [part for part in raw.split(os.pathsep) if part.strip()]
    if not resolved:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage.none("corpus files"),
            f"{s['env_var']} is set but empty — a zero-file corpus list is the vacuous case",
            evidence,
        )
    pinned = list(shared["corpus_files"])
    evidence["resolved_files"] = resolved
    evidence["pinned_files"] = pinned
    expected = len(pinned)
    if resolved != pinned:
        # Name the drift, not just the cardinality: name the phantom/dropped
        # entries so the operator can diff by eye. Every failure claim carries
        # its denominator AND names its offenders.
        extras = [r for r in resolved if r not in pinned]
        dropped = [f for f in pinned if f not in resolved]
        named = []
        if extras:
            named.append(f"resolved names files the manifest never pinned: {extras[:3]}")
        if dropped:
            named.append(f"pinned files missing from the resolved list: {dropped[:3]}")
        if not named:
            # Identical sets, different order: batch-0 is entry zero of the
            # resolved list, so ORDER is part of the corpus pin, not a detail.
            named.append(
                "the same files in a different order — order is pinned (batch-0 is entry 0)"
            )
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len([r for r in resolved if r in pinned]), "corpus files", expected=expected),
            f"resolved {s['env_var']} ({len(resolved)} files, in order) != frozen manifest corpus "
            f"({len(pinned)} files): "
            + "; ".join(named)
            + " — the banner would not match the manifest",
            evidence,
        )

    recipe_hits: dict[str, int] = {}
    for path in s["recipe_files"]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return _finalize(
                "corpus_wiring",
                "Corpus wiring",
                Verdict.ERROR,
                Coverage(len(resolved), "corpus files", expected=expected),
                f"recipe source unreadable: {path}: {exc}",
                evidence,
            )
        recipe_hits[path] = text.count("_env_jsonls(")
    evidence["recipe_files_examined"] = len(recipe_hits)
    evidence["recipe_hits"] = recipe_hits
    if not any(recipe_hits.values()):
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len(resolved), "corpus files", expected=expected),
            f"no call to _env_jsonls( found in {len(recipe_hits)} declared recipe files — "
            f"the recipe is not proven to read {s['env_var']}",
            evidence,
        )

    attest_path = Path(s["attestation_path"])
    try:
        attestation = json.loads(attest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage(len(resolved), "corpus files", expected=expected),
            f"batch-0 human attestation unreadable: {attest_path}: {exc}",
            evidence,
        )
    reader = str(attestation.get("reader", "")).strip()
    claimed = str(attestation.get("sample_sha256", ""))
    evidence["attestation"] = {
        "reader": reader,
        "claimed_sample_sha256": claimed,
        "path": str(attest_path),
    }
    if not reader:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len(resolved), "corpus files", expected=expected),
            "attestation names no reader — a claim of human review with no human named",
            evidence,
        )
    try:
        actual = _canonical_sample_sha256(Path(resolved[0]))
    except ArtifactError as exc:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.ERROR,
            Coverage(len(resolved), "corpus files", expected=expected),
            str(exc),
            evidence,
        )
    evidence["attestation"]["recomputed_sample_sha256"] = actual
    if claimed != actual:
        return _finalize(
            "corpus_wiring",
            "Corpus wiring",
            Verdict.FAIL,
            Coverage(len(resolved), "corpus files", expected=expected),
            f"attestation sample hash {claimed[:12]}… != recomputed {actual[:12]}… over "
            f"{resolved[0]} — either the sample read was not batch-0 of this corpus, or the "
            f"file changed since; both void the human-review claim",
            evidence,
        )

    return _finalize(
        "corpus_wiring",
        "Corpus wiring",
        Verdict.PASS,
        Coverage(len(resolved), "corpus files", expected=expected),
        f"env export matches manifest {len(resolved)}/{expected}; _env_jsonls( wired in "
        f"{sum(1 for v in recipe_hits.values() if v)}/{len(recipe_hits)} recipe files; "
        f"batch-0 sample attested by {reader} and machine-confirmed",
        evidence,
    )
