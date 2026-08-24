"""Point the shipped checkpoint gates at a REAL on-disk checkpoint.

Why this tool exists
--------------------
Until this probe, every fact known about ``foundationscale.gates.checkpoint_gates``
came from synthetic fixtures (``gates/fixtures.py``) written by the same hand that
wrote the detectors. ``verify_controls`` proves each gate *can* fire on corruption its
author *imagined*. It has never said what the gates conclude — or abstain from — on a
real artifact: multi-GB, written by a real trainer, described imperfectly by an
upstream ``config.json``. A detector suite validated only by its author's fixtures
finds exactly the defects its author thought to fabricate. This probe is the first
time the gates meet content they were not designed around.

Two rules make the probe itself safe to believe:

1. **The denominator is independent or absent.** ``declared_*`` fields are derived
   from the HF ``config.json`` beside the weights — never from the checkpoint's own
   metadata, and never from the framework's run manifest (which is written by the
   same training stack that wrote the checkpoint, so it cannot adjudicate it).
   Denominators the config does not *state* are left ``None`` and said to be
   ``None``; the affected gate then abstains (SKIP) or blocks (VACUOUS), and the
   probe reports that. Fabricating a denominator so every gate "runs" would be this
   repository committing its own founding defect inside its own probe.

2. **The exit code is the deliverable.** ``0`` only when every gate reached a real
   verdict (PASS over non-vacuous coverage, or SKIP with a stated reason — and
   "stated" is re-verified at the summary, not inherited: a SKIP whose reason
   string is blank is printed as reason-MISSING and blocks CLEAR) and every
   requested control fired. Any blocking verdict — FAIL, VACUOUS, UNDERCOVERED,
   ERROR — means NOT CLEARED and exit ``1``. "Fired" is a CAUSAL claim,
   adjudicated in :func:`run_alias_control`: the unmodified artifact must NOT
   have blocked the gate, and the injected artifact must have returned FAIL.
   A block with verdict ERROR (a crash) or VACUOUS/UNDERCOVERED (a coverage
   tripwire), and any block the baseline already produced, is reported
   INCONCLUSIVE — shown, named, and weighed against CLEAR like any unmet
   control, because crediting it would certify a detector through a confound.
   A VACUOUS gate on a real 16 GB
   checkpoint is not a footnote; it is the ``all([])`` result this repository exists
   to prevent, arriving from the framework itself, and the probe must be able to say
   that about its own employer. Exit ``3`` means *the probe could not measure*
   (unreadable checkpoint, missing/invalid config) — distinct from argparse's ``2``
   (used wrongly) and from ``1`` (measured, and the answer was not "clear").

The probe is read-only: nothing here writes to or modifies the checkpoint.

Usage:
    python tools/real_checkpoint_probe.py CKPT_DIR [--config CONFIG_JSON] \\
        [--json OUT] [--inject-alias N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from foundationscale.checkpoint.dcp import CheckpointFormatError
from foundationscale.checkpoint.dcp_meta import CheckpointMetadata, read_metadata

# Private names, imported deliberately. The --inject-alias control must target
# exactly the tensor population ExpertDistinctnessGate examines, and the inventory
# byte total must price dtypes by exactly the table the gates use — paraphrasing
# either here would satisfy a reader but violate the controls contract ("a control
# must exercise the real code path, not a paraphrase"). If these drift, this import
# failing loudly is the point.
# The import comment above promises "the gate's own selection logic, not a
# paraphrase". The alias control broke that promise in one place: it grouped
# experts by a hand-rolled _SHARD_SUFFIX_RE loop written when only the
# Megatron LOCAL spelling existed. _split_expert_layouts -- the one classifier
# both expert gates actually run -- replaces it below; _SHARD_SUFFIX_RE has no
# remaining use in this file, and keeping an unused private import would keep
# the paraphrase on life support.
# The same contract now covers the dense-declaration census (_expert_named /
# _matches_expert_family): the census that corroborates or refutes a dense
# declaration must name exactly the population the gates would examine, or
# "census: 0 expert tensors" could certify dense over a population the gates'
# own empty-set door would have indicted.
from foundationscale.gates.checkpoint_gates import (
    _DTYPE_BYTES,
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    FirstSaveGate,
    SaveCompletenessGate,
    TensorMeta,
    _expert_named,
    _expert_weights,
    _matches_expert_family,
    _split_expert_layouts,
)
from foundationscale.gates.core import Gate, GateResult, Verdict

# The config-key vocabulary is likewise imported once, from the provenance module
# that owns it. This probe once kept a two-name local copy of the routed-expert
# count keys beside derive_declared; it drifted from the library's three-name list
# (num_local_experts / n_routed_experts / num_experts), and a DeepSeek-family
# config stating its experts only under n_routed_experts was MoE to the library
# and "dense" to this probe — a false dense declaration minted by nothing but
# key-list drift, at the exact altitude where a minted 0 now shrinks
# FirstSaveGate's denominator. Two copies cannot drift when one does not exist;
# if either constant is renamed, this import failing loudly is the intended alarm,
# the same rationale as the private imports above.
from foundationscale.provenance.manifest import _ENABLE_MOE_BLOCK_KEY, _EXPERT_COUNT_KEYS

EXIT_CLEAR = 0
EXIT_BLOCKED = 1
EXIT_UNMEASURED = 3  # 2 belongs to argparse; 3 keeps "could not measure" its own signal

_ALIAS_STORAGE_ID = "probe://injected-alias/many-names-one-storage"

# The only keys this probe will read, per its own rule: derive what the config
# states and nothing else. ``text_config`` scope is searched first (multimodal
# models nest the LM config there), then the top level. Everything absent or null
# stays absent — and, since the denominator-shrink repair, an ABSENT routed-expert
# count stays absent (num_experts=None, UNKNOWN, gates block); only an affirmative
# dense statement corroborated by the artifact census may mint the 0.
# _EXPERT_COUNT_KEYS is imported from foundationscale.provenance.manifest (see the
# import block): it is the library's wider list, shared, and therefore undriftable.
_MOE_LAYER_KEYS = ("num_moe_layers",)

# Enumerated, not pulled from REGISTRY: the sweep's population must be a returned
# fact printed below, not an inference from import side effects (importing
# gates.example or verify.parity would silently swell a registry-driven sweep, and
# this file's whole job is exact statements about what ran).
_CHECKPOINT_GATES: tuple[type[Gate], ...] = (
    ExpertDistinctnessGate,
    ExpertByteVolumeGate,
    SaveCompletenessGate,
    FirstSaveGate,
)


class ProbeUnmeasured(RuntimeError):
    """The probe could not measure. Distinct from 'measured, and the answer blocks'."""


# ---------------------------------------------------------------------------
# Measurement: checkpoint metadata + the independent declared block
# ---------------------------------------------------------------------------


def _measure_checkpoint(path: Path) -> CheckpointMetadata:
    try:
        return read_metadata(path)
    except (CheckpointFormatError, OSError) as exc:
        # read_metadata already refuses non-checkpoints and zero-key sources with
        # precise messages; the probe adds nothing by re-explaining, so relay.
        raise ProbeUnmeasured(f"checkpoint unreadable: {path}: {exc}") from exc


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ProbeUnmeasured(
            f"config file not found: {config_path} — the declared block has no "
            f"source. Pass --config, or accept that without an independent "
            f"denominator this probe has nothing to compare the checkpoint against"
        )
    try:
        raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProbeUnmeasured(
            f"config unreadable or not valid JSON: {config_path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(raw, dict):
        raise ProbeUnmeasured(f"config is not a JSON object: {config_path}")
    return raw


def _scoped_int(
    config: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
    notes: list[str],
) -> tuple[int | None, str | None]:
    """First non-null integer among ``keys``, ``text_config`` scope before top level.

    Returns ``(value, basis)``; ``(None, None)`` means the config states nothing
    this probe can responsibly use. Only JSON integers count as declared counts:
    ``int(7.5) == 7`` would mint a denominator the config never stated, and a
    printed "7" wearing a malformed ``7.5``'s clothes is this probe committing
    its own founding defect inside its own denominator derivation. Integral
    floats are accepted *with the coercion recorded in the notes*; every other
    wrong-typed or out-of-domain value is treated as absent and recorded —
    "silently coerced" and "silently dropped" are both ways a probe starts
    telling stories about its denominator.
    """
    scopes: list[tuple[str, dict[str, Any]]] = []
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        scopes.append(("text_config", text_cfg))
    scopes.append(("", config))
    for scope_name, scope in scopes:
        for key in keys:
            if key not in scope:
                continue
            dotted = f"{scope_name}.{key}" if scope_name else key
            raw = scope[key]
            if raw is None:
                notes.append(f"{label}: {dotted} is null — treated as absent (HF convention)")
                continue
            # bool before int: in Python isinstance(True, int) is True, and a
            # JSON true is not a count however eagerly it converts like one.
            if isinstance(raw, bool):
                notes.append(
                    f"{label}: {dotted} is a boolean ({raw!r}), not a count — treated as absent"
                )
                continue
            if isinstance(raw, float):
                if not raw.is_integer():
                    # inf/NaN also land here; int(inf) raising OverflowError is
                    # why nothing below this branch calls int() on a float.
                    notes.append(
                        f"{label}: {dotted} = {raw!r} is not an integral number — "
                        f"treated as absent rather than truncated into a declaration"
                    )
                    continue
                value = int(raw)
                notes.append(
                    f"{label}: {dotted} is the float {raw!r}, not a JSON integer — "
                    f"used as {value} and recorded as a coercion, because a silent "
                    f"one would read as a declaration"
                )
            elif isinstance(raw, int):
                value = raw
            else:
                notes.append(f"{label}: {dotted} is not an integer ({raw!r}) — treated as absent")
                continue
            if value < 0:
                notes.append(
                    f"{label}: {dotted} = {value} is negative — a negative count is "
                    f"not a denominator; treated as absent"
                )
                continue
            return value, f"{dotted} = {value}"
    return None, None


def _enable_moe_block_flag(
    config: dict[str, Any], notes: list[str]
) -> tuple[bool | None, str | None]:
    """Read the AFFIRMATIVE dense/MoE discriminator, ``text_config`` before top level.

    Returns ``(flag, scope_label)``; ``(None, None)`` means the key is absent in
    both scopes. Present-but-non-boolean is treated as absent WITH A NOTE,
    mirroring :func:`_scoped_int`'s discipline: a quoted ``"false"`` must never
    truthy-parse into an affirmative dense declaration, and recording the
    malformed value is how this probe avoids telling stories about its
    denominator. The key name is read through the provenance module's single
    definition (:data:`_ENABLE_MOE_BLOCK_KEY`) — the emitter writes to the same
    constant, so probe and emitter can never drift on WHICH key is affirmative.
    """
    seen: list[tuple[str, dict[str, Any]]] = []
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        seen.append(("text_config", text_cfg))
    seen.append(("top level", config))
    for scope_name, scope in seen:
        if _ENABLE_MOE_BLOCK_KEY not in scope:
            continue
        raw = scope[_ENABLE_MOE_BLOCK_KEY]
        if isinstance(raw, bool):
            return raw, scope_name
        notes.append(
            f"enable_moe_block: {scope_name}.{_ENABLE_MOE_BLOCK_KEY} is present "
            f"but not a JSON boolean ({raw!r}) — treated as absent, because a "
            "stated declaration of the wrong type is never coerced into one"
        )
    return None, None


def _census_expert_family(meta: CheckpointMetadata) -> tuple[int, tuple[str, ...]]:
    """Count expert-family REAL tensors in the measured artifact.

    This census is the artifact half of the dense-declaration corroboration: a
    config that says dense is only believed when the artifact also contains ZERO
    expert-family tensors — two independent sources. The classifiers are the
    gates' own atoms (imported above, per the controls contract). The
    ``_expert_named`` half is included deliberately: an unrecognized MoE layout
    must still trip the contradiction, matching the gates' rule that an
    unrecognized layout is not a dense model. Blobs are excluded by the same rule
    ``build_inventory`` applies, because keys are not tensors.
    """
    matched = sorted(
        fqn
        for fqn, tm in meta.tensors.items()
        if not (tm.is_extra_state or "_extra_state" in fqn)
        and (_expert_named(fqn) or _matches_expert_family(fqn))
    )
    return len(matched), tuple(matched[:8])


def derive_declared(
    config: dict[str, Any],
    *,
    expert_family_census: int | None = None,
    expert_family_sample: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The whole declared block, with every field's provenance written next to it.

    Anything not literally stated by the config is None — and for ``num_experts``
    absence STAYS None. This function once rewrote absence into ``0``; ``0`` is a
    positive dense declaration the gates honor as NOT_APPLICABLE and
    FirstSaveGate honors by removing both expert properties from its
    denominator, so a MoE model whose count lived under a key this probe did
    not know (``n_routed_experts`` — now covered by the shared list) reached a
    1/1 PASS over an artifact no expert gate examined. That is the founding
    all([]) defect wearing a config parser. The mint rule is now: ``0``
    requires an AFFIRMATIVE statement (``enable_moe_block == false``, or the
    count key itself carrying an explicit ``0``) CORROBORATED by
    ``expert_family_census == 0`` over the real artifact — two independent
    sources. Absence yields UNKNOWN (None, and the gates block). A
    contradiction (config says dense, census finds experts; config affirms MoE
    without an understood count) also stays None, with the contradiction
    written into the basis — stated, never silently adjudicated in the
    config's favor.

    ``declared_fqns`` is None *always*: an HF config states hyperparameters,
    not module FQNs, and guessing the FQN set from layer counts would
    manufacture the very denominator SaveCompletenessGate exists to demand.

    ``expert_family_census=None`` means the caller supplied no census (direct
    library-style use); an uncorroborated affirmative statement then stays None
    too — one source is an assertion, two independent sources are evidence.
    """
    notes: list[str] = []

    count, count_basis = _scoped_int(config, _EXPERT_COUNT_KEYS, "num_experts", notes)
    flag, flag_scope = _enable_moe_block_flag(config, notes)

    census = expert_family_census if expert_family_census is not None else 0
    sample_hint = f" (first: {expert_family_sample[0]})" if expert_family_sample else ""
    if expert_family_census is not None:
        # The corroboration denominator travels in the notes, so "census: 0"
        # below never reads as an unqualified count (doctrine 2).
        notes.append(
            f"expert-family census: {census} expert-named-or-family real "
            f"tensor(s) in the measured artifact — the artifact half of the "
            f"dense/MoE corroboration"
        )

    # Collect what the config AFFIRMATIVELY says, on both sides. An explicit
    # zero under a count key is a dense statement; a positive count is an MoE
    # statement; the discriminator is one of either. Absence is neither.
    dense_sources: list[str] = []
    if count == 0:
        dense_sources.append(count_basis or "num_experts = 0")
    if flag is False:
        dense_sources.append(f"{flag_scope}.{_ENABLE_MOE_BLOCK_KEY}=false")
    moe_sources: list[str] = []
    if count is not None and count > 0:
        moe_sources.append(count_basis or f"num_experts = {count}")
    if flag is True:
        moe_sources.append(f"{flag_scope}.{_ENABLE_MOE_BLOCK_KEY}=true")

    num_experts: int | None
    experts_basis: str
    if moe_sources and dense_sources:
        # The config contradicts itself (enable_moe_block=true beside an
        # explicit routed count of 0 is the reachable shape). Refusing to
        # adjudicate inside a self-contradicting config is the same rule the
        # emitter holds: UNKNOWN blocks, loudly, naming both sides.
        num_experts = None
        experts_basis = (
            f"the config contradicts itself: {' and '.join(moe_sources)} point(s) "
            f"at MoE while {' and '.join(dense_sources)} declare(s) dense — "
            f"refusing to pick a winner; num_experts stays None (UNKNOWN; the "
            f"gates block)"
        )
    elif moe_sources:
        if count is None:
            # The discriminator affirms MoE but no understood key states how
            # many routed experts: minting dense from that is the measured
            # Gemma-4-26B defect; minting a count is fabrication.
            num_experts = None
            experts_basis = (
                f"{' and '.join(moe_sources)} affirm(s) MoE structure, but no "
                f"routed-expert count was found under the keys this probe shares "
                f"with the library ({'/'.join(_EXPERT_COUNT_KEYS)}). num_experts "
                f"stays None (UNKNOWN; the gates block). Extend the shared key "
                f"list, or correct the config"
            )
        else:
            num_experts = count
            experts_basis = count_basis or f"num_experts = {count}"
            if flag is True:
                experts_basis += f"; MoE affirmed by {flag_scope}.{_ENABLE_MOE_BLOCK_KEY}=true"
            if expert_family_census is not None and census == 0:
                # The mirror contradiction: MoE declared, no expert tensors
                # present. The gates' empty-set doors already block this shape
                # as VACUOUS; the note makes those blocking verdicts read as
                # ONE contradiction instead of two coincidences.
                notes.append(
                    f"contradiction recorded: {experts_basis} declares {count} "
                    f"routed experts per MoE layer, but the artifact census "
                    f"found 0 expert-family tensors — the expert gates already "
                    f"block this shape as VACUOUS; stated at the denominator so "
                    f"the disagreement reads as one loud fact, not two quiet "
                    f"verdicts"
                )
    elif dense_sources:
        if expert_family_census is None:
            num_experts = None
            experts_basis = (
                f"{' and '.join(dense_sources)} read(s) as an affirmative dense "
                f"declaration, but no artifact census was supplied to corroborate "
                f"it against — one source is an assertion, two independent "
                f"sources are evidence. num_experts stays None (UNKNOWN; the "
                f"gates block)"
            )
        elif census > 0:
            # Config says dense, artifact holds experts: stated, never
            # adjudicated in the config's favor.
            num_experts = None
            experts_basis = (
                f"CONTRADICTION, stated rather than adjudicated: "
                f"{' and '.join(dense_sources)} declare(s) dense, yet the "
                f"artifact holds {census} expert-family tensor(s){sample_hint}. "
                f"Believing the config mints dense over a live-expert artifact "
                f"(the denominator-shrink incident, verbatim); believing the "
                f"artifact fabricates a count the config never stated. "
                f"num_experts stays None (UNKNOWN) so every expert gate blocks"
            )
        else:
            num_experts = 0
            experts_basis = (
                f"dense, corroborated: {' and '.join(dense_sources)} — "
                f"affirmative statement(s) READ, not inferred from a key's "
                f"absence — AND the artifact census found 0 expert-family "
                f"tensors: two independent sources agree, so num_experts = 0 is "
                f"recorded as a positive dense declaration the gates may honor "
                f"(SKIP NOT_APPLICABLE), not an absence laundered into one"
            )
    else:
        num_experts = None
        experts_basis = (
            f"no {'/'.join(_EXPERT_COUNT_KEYS)} key (non-null) in text_config or "
            f"at top level, and no {_ENABLE_MOE_BLOCK_KEY} discriminator: the "
            f"config STATES NOTHING about routed experts, and saying nothing is "
            f"not declaring dense. num_experts stays None (UNKNOWN; the gates "
            f"block). This line replaced the probe's own founding-bug relapse: "
            f"it used to mint num_experts=0 from exactly this absence"
        )

    num_moe_layers, layers_basis = _scoped_int(config, _MOE_LAYER_KEYS, "num_moe_layers", notes)
    if num_moe_layers is None:
        layers_basis = (
            "config states no num_moe_layers — left None. Unknown is not zero: "
            "gates will report coverage without a fabricated denominator rather "
            "than count every layer as MoE by architectural assumption."
        )

    return {
        "num_experts": num_experts,
        "num_moe_layers": num_moe_layers,
        "declared_fqns": None,
        "expected_expert_bytes": None,
        "basis": {
            "num_experts": experts_basis,
            "num_moe_layers": layers_basis,
            "declared_fqns": (
                "an HF config.json states hyperparameters, not module FQNs — the "
                "full declared tensor set is NOT derivable from it. Left None. "
                "SaveCompletenessGate will answer VACUOUS (blocking), which is the "
                "honest result here: 'what is there matches what is there' is not "
                "a check, and this probe will not be the place that pretends it is."
            ),
            "expected_expert_bytes": (
                "expert byte volume is not *stated* anywhere in config.json. It is "
                "computable from hidden dims in principle, but a computed "
                "denominator is inference wearing declaration's clothes — left "
                "None. Consequence stated once, plainly: on a real MoE checkpoint "
                "the byte-volume gate abstains (SKIP). That gap is reported, not "
                "papered over."
            ),
        },
        "notes": notes,
    }


def build_inventory(meta: CheckpointMetadata) -> dict[str, Any]:
    """One-line facts about the real artifact, counted the way the gates count.

    Real tensors vs ``_extra_state`` byte blobs are separated with the same rule the
    gates apply, because the 8,042-of-8,970 lesson is that keys are not tensors.
    """
    real = [
        (fqn, tm)
        for fqn, tm in meta.tensors.items()
        if not (tm.is_extra_state or "_extra_state" in fqn)
    ]
    dtype_hist: dict[str, int] = {}
    total_bytes = 0
    uncounted = 0
    with_sid = 0
    for _fqn, tm in real:
        dtype_hist[tm.dtype] = dtype_hist.get(tm.dtype, 0) + 1
        width = _DTYPE_BYTES.get(tm.dtype)
        if width is None:
            # TensorMeta.nbytes prices an unknown dtype at 4 B/elem without saying
            # so (its .get default). The byte gate inherits that approximation;
            # this inventory refuses to print it as fact.
            uncounted += 1
        else:
            total_bytes += math.prod(tm.shape) * width
        if tm.storage_id is not None:
            with_sid += 1
    return {
        "origin": meta.origin,
        "format": meta.format,
        "entries_total": len(meta.tensors),
        "real_tensors": len(real),
        "extra_state_blobs": len(meta.tensors) - len(real),
        "metadata_implied_bytes": total_bytes,
        "uncounted_unknown_dtype_tensors": uncounted,
        "dtypes": dict(sorted(dtype_hist.items())),
        "with_storage_id": with_sid,
        "without_storage_id": len(real) - with_sid,
    }


def build_context(
    meta: CheckpointMetadata, declared: dict[str, Any], config_path: Path
) -> CheckpointGateContext:
    """Assemble the gate context from the two independent halves.

    Per-tensor mapping mirrors ``CheckpointGateContext.from_path`` exactly, but the
    declared fields come from ``derive_declared`` — NOT from ``load_manifest``. The
    run manifest is produced by the same training stack that produced the
    checkpoint; a denominator that can be wrong in exactly the same way as the thing
    it adjudicates is not a denominator.
    """
    tensors = tuple(
        TensorMeta(
            fqn=fqn,
            shape=tuple(tm.shape),
            dtype=tm.dtype,
            storage_id=tm.storage_id,
            kind="extra_state" if (tm.is_extra_state or "_extra_state" in fqn) else "tensor",
        )
        for fqn, tm in meta.tensors.items()
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=declared["declared_fqns"],
        num_experts=declared["num_experts"],
        num_moe_layers=declared["num_moe_layers"],
        expected_expert_bytes=declared["expected_expert_bytes"],
        # One origin field, both provenances: what was measured vs what was declared.
        origin=f"{meta.origin} [metadata={meta.format}; declared={config_path}]",
    )


# ---------------------------------------------------------------------------
# MUST_FIRE control on real content: --inject-alias N
# ---------------------------------------------------------------------------


def run_alias_control(
    ctx: CheckpointGateContext, n: int, baseline: GateResult | None
) -> dict[str, Any]:
    """Collapse N real expert-shaped tensors onto one storage_id; demand a block.

    This is the incident's mechanism (many names, one byte span) reconstructed
    inside a context built from the real checkpoint — the fixture lesson, re-proven
    on tensors the framework's author never synthesized. The gate must BLOCK. A
    quiet gate here means the aliasing detector is dead on real metadata, and that
    is a finding about the framework, not the checkpoint.

    But "the gate blocked" is not yet "the control fired". Firing is a CAUSAL
    claim — the injected aliasing caused the block — and this function is where
    that claim is adjudicated, because ``result.blocking`` alone once stood in
    for it and credited confounded runs, crash verdicts and tripwires alike.
    Attribution requires all three legs, observed:

    1. the injected run blocked (necessary, never sufficient);
    2. with verdict FAIL — the only verdict asserting "the units were examined
       and the defect was found". ERROR is a crash on the injected metadata;
       VACUOUS/UNDERCOVERED are coverage tripwires. All three stop a sweep
       WITHOUT detecting anything, so crediting any of them certifies the
       detector by its own malfunction;
    3. against a clean base rate: the baseline (unmodified artifact) ran under
       this gate and did NOT block — otherwise the block may pre-date the
       injection. A non-blocking baseline reads as "no objection to the
       unmodified artifact" whether it is PASS or SKIP; both verdicts are
       printed side by side in the report so the reader judges the toggle.

    Legs that cannot be established yield ``"inconclusive"`` with the failed
    leg named in ``inconclusive_reason``: a stated abstention, first-class in
    the output, and per the module's exit contract it weighs against CLEAR
    exactly as a quiet gate does — the claim this control exists to test stays
    unverified on this artifact. Crediting it anyway would repair a false
    negative by minting a false positive, which is the same defect.
    """
    experts = _expert_weights(ctx)  # the gate's own selection, verbatim
    if not experts:
        return {
            "status": "skipped",
            "reason": (
                f"the checkpoint holds zero expert-shaped tensors by the gate's "
                f"own selection (declared num_experts={ctx.num_experts}); there is "
                f"nothing to alias. On a dense artifact this control is "
                f"inapplicable — and an inapplicable control must say so, because "
                f"a control that quietly 'passes' over nothing is the all([]) "
                f"result one level up."
            ),
        }
    # Group with the gate's OWN classifier, verbatim -- the hand-rolled loop
    # this replaces knew only the Megatron LOCAL suffix spelling
    # (...linear_fc1.weight7, the incident) and so read the GLOBAL per-expert
    # spelling (...experts.<i>.<proj>.weight -- Mixtral, Qwen-MoE, and this
    # estate's Gemma-4 conversion) as "fused", reporting inapplicable exactly
    # where ExpertDistinctnessGate DOES check within-group storage aliasing.
    # While the live gate routed by _split_expert_layouts and this control
    # grouped by the paraphrase, the router said "shardable work exists", the
    # control said "nothing to alias", and the disagreement resolved to a
    # recorded-only skip -- the founding incident wearing a router. Group
    # membership for the LOCAL spelling is identical under both groupers (the
    # splitter's shard branch wraps the same regex; only the group key string
    # gains a "<i>" suffix, and the key is never read past this line), so this
    # is a strict superset of the old behavior, not a reinterpretation of it.
    shard_groups, _stacked, _unrecognized = _split_expert_layouts(experts)
    eligible = [members for members in shard_groups.values() if len(members) >= 2]
    if not eligible:
        return {
            "status": "skipped",
            "reason": (
                f"{len(experts)} expert-shaped tensor(s) exist, but none sit in a "
                f"sharded group of >= 2 (layout reads as fused). "
                f"ExpertDistinctnessGate checks storage distinctness only within "
                f"sharded groups; there is no fused-layout aliasing check for this "
                f"control to exercise (see # NOTES). The control is inapplicable "
                f"here, stated."
            ),
        }

    members = max(eligible, key=len)
    targets = members[: min(n, len(members))]
    target_fqns = {t.fqn for t in targets}
    injected_ctx = dataclasses.replace(
        ctx,
        tensors=tuple(
            dataclasses.replace(t, storage_id=_ALIAS_STORAGE_ID) if t.fqn in target_fqns else t
            for t in ctx.tensors
        ),
    )
    result = ExpertDistinctnessGate().run(injected_ctx)
    confounded = baseline is not None and baseline.blocking
    if not result.blocking:
        # PASS or SKIP on experts aliased BY CONSTRUCTION: the gate saw (or
        # declined to see) the defect and raised no objection. The only honest
        # reading of a non-blocking MUST_FIRE control.
        status = "not_fired"
        inconclusive_reason = ""
    elif result.verdict is not Verdict.FAIL:
        # Blocking is not detecting. FAIL means the units were examined and the
        # defect was found; ERROR means the gate crashed on the injected
        # metadata; VACUOUS/UNDERCOVERED are coverage tripwires. The last three
        # stop the sweep without ever rendering on the aliasing.
        status = "inconclusive"
        inconclusive_reason = (
            f"the injected run blocked with verdict {result.verdict.value}, which "
            f"is not detection: FAIL asserts the aliasing was found; ERROR means "
            f"the gate crashed on the injected metadata; VACUOUS/UNDERCOVERED are "
            f"coverage tripwires"
        )
    elif baseline is None:
        # No base rate was recorded for this gate, so nothing distinguishes
        # "the injection caused the block" from "the block pre-dates the
        # injection". With attribution unobservable, the causal claim has no
        # evidence; abstain rather than assert.
        status = "inconclusive"
        inconclusive_reason = (
            "no baseline (unmodified) result for this gate was supplied, so the "
            "block cannot be shown to post-date the injection — attribution is "
            "unobservable on this run"
        )
    elif confounded:
        status = "inconclusive"
        inconclusive_reason = (
            f"the baseline (unmodified) artifact already blocked this gate "
            f"({baseline.verdict.value} — {baseline.detail}); a gate that blocks "
            f"with and without the defect shows it blocks, not that the injected "
            f"aliasing is why"
        )
    else:
        # All three legs observed: clean base rate, blocking run, FAIL verdict.
        # The injected storage_id rewrite is the only difference between the two
        # contexts, so the causal claim is earned rather than asserted.
        status = "fired"
        inconclusive_reason = ""
    return {
        "status": status,
        "requested": n,
        "aliased": len(targets),
        "verdict": result.verdict.value,
        "detail": result.detail,
        "aliased_fqns": sorted(target_fqns)[:8],
        # The base rate is half the attribution evidence; it travels with the
        # record so nobody has to re-derive "fired" from a single verdict.
        "baseline_verdict": baseline.verdict.value if baseline is not None else None,
        # Base rates matter: if the unmodified artifact already blocks this gate,
        # the control shows the gate blocks, not that the injection caused it.
        "confounded": confounded,
        "confound_note": (
            "baseline (unmodified) run was already blocking; this run shows the "
            "gate blocks on the injected artifact, not that aliasing is why"
        )
        if confounded
        else "",
        # Populated exactly when the injected run blocked but could not be
        # credited; names which attribution leg failed.
        "inconclusive_reason": inconclusive_reason,
        "aliasing_leg_observed": ("storages" in result.detail) or ("alias" in result.detail),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _h(title: str) -> None:
    print(f"\n== {title} " + "=" * max(4, 68 - len(title)))


def _print_inventory(inv: dict[str, Any]) -> None:
    _h(f"inventory: {inv['origin']}")
    gib = inv["metadata_implied_bytes"] / float(1 << 30)
    pct = (100.0 * inv["with_storage_id"] / inv["real_tensors"]) if inv["real_tensors"] else 0.0
    print(f"  format                  : {inv['format']}")
    print(
        f"  named metadata entries  : {inv['entries_total']:,} total -> "
        f"{inv['real_tensors']:,} tensors + {inv['extra_state_blobs']:,} extra_state "
        f"byte blobs (blobs excluded from every tensor count below)"
    )
    caveat = (
        f"; {inv['uncounted_unknown_dtype_tensors']:,} tensor(s) EXCLUDED — dtype not "
        f"in the gates' byte table"
        if inv["uncounted_unknown_dtype_tensors"]
        else ""
    )
    print(
        f"  metadata-implied bytes  : {inv['metadata_implied_bytes']:,} "
        f"({gib:.2f} GiB) over known-dtype tensors{caveat}"
    )
    dtypes = ", ".join(f"{name} x{count:,}" for name, count in inv["dtypes"].items()) or "<none>"
    print(f"  dtypes                  : {dtypes}")
    print(
        f"  storage_id coverage     : {inv['with_storage_id']:,}/{inv['real_tensors']:,} "
        f"real tensors ({pct:.1f}%)"
    )
    if inv["without_storage_id"]:
        print(
            f"  WARNING: {inv['without_storage_id']:,} tensor(s) carry storage_id=None; "
            f"for those, storage aliasing is UNDETECTABLE from metadata — None reads "
            f"as 'unknown', never as 'distinct'"
        )


def _print_declared(declared: dict[str, Any], config_path: Path, config: dict[str, Any]) -> None:
    _h(f"declared block (independent source: {config_path})")
    bits = []
    if isinstance(config.get("model_type"), str):
        bits.append(f"model_type={config['model_type']!r}")
    arch = config.get("architectures")
    if isinstance(arch, list) and arch:
        bits.append(f"architectures={arch!r}")
    if bits:
        # Display only. A MoE model whose expert keys this probe does not know is
        # classified dense; this line is the evidence left in view to falsify that.
        print(
            "  identity                : "
            + ", ".join(bits)
            + "  (display only — never fed to a gate)"
        )
    basis = declared["basis"]
    print(f"  num_experts             : {declared['num_experts']}\n      <- {basis['num_experts']}")
    print(
        f"  num_moe_layers          : {declared['num_moe_layers']}"
        f"\n      <- {basis['num_moe_layers']}"
    )
    print(f"  declared_fqns           : None\n      <- {basis['declared_fqns']}")
    print(f"  expected_expert_bytes   : None\n      <- {basis['expected_expert_bytes']}")
    for note in declared["notes"]:
        print(f"  note: {note}")


def _print_control(control: dict[str, Any]) -> None:
    _h("MUST_FIRE control: --inject-alias")
    if control["status"] == "skipped":
        print(f"  SKIPPED — {control['reason']}")
        return
    print(
        f"  rewrote {control['aliased']} expert tensor(s) (requested {control['requested']}) "
        f"onto one storage_id:"
    )
    for fqn in control["aliased_fqns"]:
        print(f"      {fqn}")
    # The two verdicts print side by side BEFORE any interpretation, because
    # "the gate blocked" and "the injection made the gate block" are different
    # claims with different denominators, and only the second credits the control.
    baseline_text = (
        control["baseline_verdict"] if control["baseline_verdict"] is not None else "<not recorded>"
    )
    print(f"  baseline verdict (unmodified artifact)  : {baseline_text}")
    print(f"  verdict on the alias-injected artifact  : {control['verdict']} — {control['detail']}")
    if control["status"] == "fired":
        print(
            "  control FIRED: clean baseline, FAIL on injection — the block is "
            "attributable to the injected aliasing, on evidence printed above."
        )
    elif control["status"] == "inconclusive":
        print(f"  control INCONCLUSIVE: {control['inconclusive_reason']}.")
        print(
            "  An inconclusive control has NOT fired: the run blocked, but crediting "
            "the detection would assert a cause that was never observed. The exit "
            "code below treats it accordingly."
        )
    else:
        print(
            "  control DID NOT FIRE: the detector returned a non-blocking verdict on "
            "artificially aliased experts built from real metadata — "
            "checkpoint.expert_distinctness cannot be trusted on this artifact. "
            "Exit blocks regardless of the sweep."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="real_checkpoint_probe",
        description="Run the shipped checkpoint gates against a real on-disk "
        "checkpoint, with the declared block derived from an independent HF "
        "config.json. Exit 0 = cleared; 1 = blocking verdict reached; "
        "3 = the probe could not measure.",
    )
    p.add_argument("ckpt_dir", help="DCP directory, or safetensors file/directory (read-only)")
    p.add_argument(
        "--config",
        default=None,
        metavar="CONFIG_JSON",
        help="HF config.json supplying the declared block "
        "(default: CKPT_DIR/config.json, or beside a single-file checkpoint)",
    )
    p.add_argument(
        "--json",
        dest="json_out",
        default=None,
        metavar="OUT",
        help="write the full machine-readable report to OUT",
    )
    p.add_argument(
        "--inject-alias",
        type=int,
        default=None,
        metavar="N",
        help="MUST_FIRE control: force N expert-shaped tensors to share one "
        "storage_id and assert checkpoint.expert_distinctness blocks (N >= 2)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.inject_alias is not None and args.inject_alias < 2:
        # A span cannot alias with itself; N=1 would control nothing while looking
        # like a control. Usage errors are argparse's exit code (2), on purpose.
        parser.error("--inject-alias N requires N >= 2")

    ckpt_path = Path(args.ckpt_dir)
    config_path = (
        Path(args.config)
        if args.config
        else (
            ckpt_path / "config.json" if ckpt_path.is_dir() else ckpt_path.with_name("config.json")
        )
    )

    try:
        meta = _measure_checkpoint(ckpt_path)
        config = _read_config(config_path)
        inventory = build_inventory(meta)
        # The census is measured BEFORE the declared block is derived: a dense
        # declaration is only minted when the config affirmatively says dense
        # AND the artifact holds zero expert-family tensors, so derive_declared
        # needs the artifact half measured up front.
        census, census_sample = _census_expert_family(meta)
        declared = derive_declared(
            config, expert_family_census=census, expert_family_sample=census_sample
        )
        ctx = build_context(meta, declared, config_path)
    except ProbeUnmeasured as exc:
        print(f"probe could not measure: {exc}", file=sys.stderr)
        return EXIT_UNMEASURED
    except Exception:
        # A bug in the probe is also "could not measure" — never "clear".
        traceback.print_exc()
        print(
            "probe could not measure: unexpected failure above "
            "(a probe bug is not a checkpoint verdict)",
            file=sys.stderr,
        )
        return EXIT_UNMEASURED

    try:
        return _deliver(
            args=args,
            ckpt_path=ckpt_path,
            config_path=config_path,
            config=config,
            inventory=inventory,
            declared=declared,
            ctx=ctx,
        )
    except ProbeUnmeasured as exc:
        print(f"probe could not measure: {exc}", file=sys.stderr)
        return EXIT_UNMEASURED
    except Exception:
        # The fail-closed rule cannot stop at measurement. From here the probe is
        # rendering and ruling on real measurements; if it dies in this phase the
        # interpreter's own exit status is 1, and every pipeline that has read
        # the contract files exit 1 as "measured, and the answer blocks". A
        # probe bug must never wear a checkpoint verdict — it is exit 3, with
        # the traceback in view.
        traceback.print_exc()
        print(
            "probe could not measure: unexpected failure after measurement above "
            "(a probe bug is not a checkpoint verdict)",
            file=sys.stderr,
        )
        return EXIT_UNMEASURED


def _deliver(
    *,
    args: argparse.Namespace,
    ckpt_path: Path,
    config_path: Path,
    config: dict[str, Any],
    inventory: dict[str, Any],
    declared: dict[str, Any],
    ctx: CheckpointGateContext,
) -> int:
    """Render, rule, and report on a checkpoint that is already measured.

    Everything after measurement lives behind this one call so ``main`` can
    hold a single invariant around the entire phase: a failure inside is a
    probe failure (exit 3), never a checkpoint verdict (exit 1). The gate
    sweep and summary wording grew from the body that lived inline in ``main``;
    the blocking-reason list and report schema have since gained two honesty
    rules the inline version lacked, both enforced below rather than assumed:
    a skip is "stated" only if its reason string is non-blank, and a MUST_FIRE
    control is credited only when its block is attributable (see
    :func:`run_alias_control`).
    """
    _print_inventory(inventory)
    _print_declared(declared, config_path, config)

    results: list[GateResult] = []
    invariant_breaches: list[str] = []
    _h(f"gates ({len(_CHECKPOINT_GATES)} shipped checkpoint gates; population explicit)")
    for gate_cls in _CHECKPOINT_GATES:
        result = gate_cls().run(ctx)  # run(), not check(): ERROR on exception, timed
        results.append(result)
        print(f"  {result.render()}")
        # The framework states Gate.ok cannot return PASS over zero coverage.
        # "States" is doing work in that sentence, so verify it: the day this
        # fires, the framework has its own founding bug, and the probe reports it
        # instead of inheriting it.
        if result.verdict is Verdict.PASS and result.coverage.checked == 0:
            invariant_breaches.append(result.gate_id)

    control: dict[str, Any] | None = None
    if args.inject_alias is not None:
        baseline = next((r for r in results if r.gate_id == ExpertDistinctnessGate.id), None)
        control = run_alias_control(ctx, args.inject_alias, baseline)
        _print_control(control)

    blocking = [r for r in results if r.blocking]
    skipped = [r for r in results if r.verdict is Verdict.SKIP]
    passed = [r for r in results if r.verdict is Verdict.PASS]
    # An abstention is only as good as its stated reason. Gate.skip() enforces
    # that at construction, but a hand-built GateResult — the exact move the
    # gate-contract docstring warns against — can carry Verdict.SKIP with a
    # blank detail, and Gate.run forwards it untouched. The probe prints
    # "SKIP (reason stated)" and counts SKIP toward the CLEAR wording "stated
    # why it abstained"; both are CLAIMS MADE HERE, so both are VERIFIED HERE,
    # at the point of claim, rather than inherited from the helper some gate
    # may have bypassed. A blank-reason SKIP is not a stated abstention: it
    # prints under its own honest header and weighs against CLEAR.
    unstated_skips = [r for r in skipped if not r.detail.strip()]
    stated_skips = [r for r in skipped if r.detail.strip()]
    reasons = [f"{r.gate_id}={r.verdict.value}" for r in blocking]
    if control is not None and control["status"] == "not_fired":
        reasons.append("--inject-alias MUST_FIRE control stayed quiet on real content")
    elif control is not None and control["status"] == "inconclusive":
        # The injected run blocked but never earned attribution to the
        # injection (run_alias_control names which leg failed). The exit-0
        # contract is "every requested control FIRED"; inconclusive is not
        # fired, and crediting it would repair a confound by minting a pass.
        reasons.append(
            "--inject-alias MUST_FIRE control was INCONCLUSIVE, not credited as fired — "
            + control["inconclusive_reason"]
        )
    elif control is not None and control["status"] == "skipped":
        # A requested MUST_FIRE control that could not run leaves its claim —
        # that the aliasing detector fires on real metadata — unverified on
        # this artifact. The module's exit-0 contract is "every requested
        # control fired", so a skipped control must weigh against CLEAR exactly
        # as a quiet one does; anything softer is the control "passing" over
        # zero exercised units. (As shipped, checkpoint.save_complete blocks
        # every run first, so this branch cannot change today's exit code. It
        # exists for the day an independent declared_fqns source makes CLEAR
        # reachable — which is precisely when it will matter.)
        reasons.append(
            "--inject-alias MUST_FIRE control was requested but reported 'skipped' — "
            "the claim it exists to test is unverified on this artifact"
        )
    reasons.extend(
        f"{r.gate_id}: SKIP with no stated reason — an unexplained abstention is a "
        f"hole. Gate.skip rejects blank reasons, so this result bypassed the "
        f"helper, and the probe re-checks the claims it prints rather than "
        f"inheriting them"
        for r in unstated_skips
    )
    reasons.extend(
        f"{gid}: PASS over 0 checked units — the framework's own coverage rule did not hold"
        for gid in invariant_breaches
    )
    exit_code = EXIT_BLOCKED if reasons else EXIT_CLEAR

    _h("summary")
    print(
        f"  gates run               : {len(results)} "
        f"(population enumerated by this file, not inferred from the registry)"
    )
    print(f"  PASS (non-vacuous)      : {len(passed)}")
    print(f"  SKIP (reason stated)    : {len(stated_skips)}")
    for r in stated_skips:
        print(f"      {r.gate_id}: {r.detail}")
    if unstated_skips:
        # Its own header, because "reason stated" is a claim with a denominator
        # and these results are outside it. They already entered `reasons` above.
        print(f"  SKIP (reason MISSING)   : {len(unstated_skips)}")
        for r in unstated_skips:
            print(f"      {r.gate_id}: <no reason stated>")
    print(f"  blocking                : {len(blocking)}")
    for reason in reasons:
        print(f"      {reason}")
    if control is not None:
        print(
            "  --inject-alias control  : "
            + {
                "fired": "FIRED",
                "not_fired": "DID NOT FIRE",
                "inconclusive": "INCONCLUSIVE (blocked, but not attributable — not credited)",
                "skipped": "SKIPPED",
            }[control["status"]]
        )
    if exit_code == EXIT_CLEAR:
        print(
            "PROBE VERDICT: CLEAR (exit 0) — every gate reached a real verdict or "
            "stated why it abstained; nothing passed over zero examined units."
        )
    else:
        print(f"PROBE VERDICT: NOT CLEARED (exit {EXIT_BLOCKED})")
        print(
            "  NOT CLEARED is not an accusation of corruption. It is the honest "
            "extent of what is verifiable about this artifact on independent "
            "evidence: gates above either found defects or could not run for want "
            "of a denominator the config does not state. Any consumer requiring "
            "'verified' is unserved until the blocking items close — a run "
            "manifest carrying declared_fqns would let checkpoint.save_complete "
            "actually check completeness instead of refusing to."
        )

    report = {
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "inventory": inventory,
        "declared": declared,
        "gates": [r.to_dict() for r in results],
        "control": control,
        "framework_invariant_breaches": invariant_breaches,
        # Downstream consumers must not have to re-derive WHICH gates were
        # counted as abstentions from per-gate verdicts and blank details —
        # the denominator of the "stated" claim travels explicitly.
        "unstated_skip_gates": [r.gate_id for r in unstated_skips],
        "blocking_reasons": reasons,
        "exit_code": exit_code,
    }
    if args.json_out:
        out = Path(args.json_out)
        try:
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"could not write --json report {out}: {exc}", file=sys.stderr)
            return EXIT_UNMEASURED  # measurement happened, the deliverable did not
        print(f"\n  report written: {out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
