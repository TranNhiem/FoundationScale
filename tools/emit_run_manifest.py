"""Emit the launch-time RunManifest the checkpoint gates need to be anything at all.

Why this tool exists
--------------------
The estate's first real FoundationScale probe run measured it: on the converted
Gemma-4-E4B base checkpoint the shipped ``tools/real_checkpoint_probe.py`` reached
a real verdict in **0 of 3** first-save sub-gates. The two expert gates SKIPPED
honestly (the model is dense), but ``checkpoint.save_complete`` reported VACUOUS
— "0 declared tensors … there is no declared tensor set to compare the present
set against" — and ``checkpoint.first_save`` inherited the block. The machinery
was never the gap: :class:`DeclaredCheckpoint`, :class:`ManifestStore` and
:func:`foundationscale.checkpoint.dcp_meta.load_manifest` all predate this tool.
The gap is that *nothing at launch time ever populated* ``RunManifest.declared``.
This tool is the missing producer side. It runs on the login node, before a single
GPU-second is burned, stdlib + this package only (the verification plane must run
with nothing installed; the DCP safetensors header parse it relies on is
torch-free by design). #83 addendum: torch is an OPTIONAL runtime sample, never an
import requirement — where the emitting interpreter can import it (the fix46
container-routed invocations), the build that will TRAIN is recorded in-manifest
for cross-check against the gate's record of the build that ADJUDICATED; where it
cannot (a host interpreter whose only torch lives in a user-site hidden by
PYTHONNOUSERSITE=1, by design), the absence is recorded as a stated abstention
with the import error in-band. An invented version string is worse than no field,
so no field is ever guessed.

What it writes, and where
-------------------------
Two placements, one inode:

* **Canonical record.** A :class:`ManifestStore` under
  ``<out_dir>/provenance/<run_id>/attempt-NNNN.json`` — attempt-keyed, atomic,
  refuse-to-overwrite, exactly the append-only semantics the 62-launches/35-bundles
  audit demanded.
* **Discovery copy.** A hard link of that same file to
  ``<checkpoint_dir>/attempt-NNNN.json``. ``load_manifest`` searches the judged
  artifact directory and *its parent only*; judged artifacts live at
  ``$CKPT_DIR/iter_NNNNNNN``, so the parent IS ``$CKPT_DIR``, and attempt-keyed
  basenames feed its newest-first glob. A hard link rather than a copy so the two
  paths can never diverge — and so the discovery side inherits the store's
  no-clobber refusal. The reserved basename ``run_manifest.json`` is deliberately
  NOT used: it is a single slot that a second attempt could only update by
  overwriting, and overwrite-in-place is the defect class the store exists to
  end.

Denominator policy (the load-bearing half of the design)
--------------------------------------------------------
* ``--full-ft`` derives ``declared_fqns`` from the tensor census of an
  INDEPENDENT base checkpoint (the converted artifact training starts from),
  never from the directory this run will save into. Deriving the declared set
  from the judged tree is the tautology the probe refuses in words — "what is
  there matches what is there" is not a check — so
  :func:`ensure_declaration_is_independent` makes it refused in fact: equal,
  nested, or samefile-aliased base/judged paths are a hard error.
* Dense-vs-MoE classification reads the AFFIRMATIVE key
  (``text_config.enable_moe_block``, top-level fallback) as a statement. A
  config cannot talk a tool into declaring dense merely by omitting a count
  key: ``enable_moe_block=true`` with no routed count this tool understands is
  a refusal, not a dense declaration (that confusion is the measured
  Gemma-4-26B false-dense defect). Conversely ``enable_moe_block=false`` is no
  longer thrown away as ``None`` (which the gates read as UNKNOWN and which
  blocked every healthy dense first save at 1/3): with the independent base
  census showing ZERO expert-family tensors as the second source, the tool
  records ``num_experts=0`` — a positive dense declaration — so dense runs
  verify 1/1 applicable instead of blocking forever. Either source ALONE
  stays ``None``; disagreement between them is a refusal (rc=1), never a
  silent winner.
* ``--lora`` emits NO declared block. The FQN set of an adapter save is a
  property of the PEFT implementation in the training repository — not
  importable here, not verifiable on this estate — and the base checkpoint's
  FQN set is the wrong denominator for adapter tensors. Minting
  r×targets×suffix strings from the launcher's own CLI would fabricate a
  denominator out of the very strings a misconfigured launcher would
  misconfigure. ``declared=None`` is the honest emission; the gates keep their
  honest VACUOUS/SKIP, and the abstention is stated in argv and stdout — and,
  since #79, also IN the manifest, as five ``declared.*`` config entries (who
  abstained, why, what derives the set instead, and how many ``iter_*`` save
  dirs pre-existed at emission), because a deliberate abstention and a field
  nobody populated must never again be the same bytes on disk. Two controls
  hold that line: emission re-reads the serialized record and REFUSES any LoRA
  record whose abstention markers did not survive to disk (#56's "a declared
  state, not a silent pass", one layer out), drillable end-to-end via
  ``FS_EMIT_DRILL_BARE_NULL=1``; and :func:`check_saved_run_declaration`
  carries the rule wherever a manifest faces realized saves: a run that SAVED
  tensors on a bare-null declared FAILS, and a zero-save observation is a
  NAMED not-exercised, never a pass (doctrines 1 and 3).

Exit codes: ``0`` emitted; ``1`` REFUSED (a policy or contradiction refusal —
measured and declined; #79 adds: a LoRA record whose declared abstention is a
bare null while save directories already exist); ``2`` usage (argparse); ``3``
could not establish inputs (unreadable base checkpoint, missing config,
unplaceable discovery link; #79 adds: a serialized record whose abstention
markers did not survive to disk — a measurement failure, never a verdict).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

# The probe (tools/real_checkpoint_probe.py) assumes an installed package. This
# tool additionally bootstraps ``src/`` when one is absent, because its primary
# execution site is a launcher on a login node, where the estate never ran
# ``pip install -e .``. Failure of THAT fallback import is itself a loud emitter
# failure — provenance half-emitted is provenance worth nothing.
try:
    import foundationscale  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - depends on install state
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foundationscale.provenance.manifest import (
    _ENABLE_MOE_BLOCK_KEY,
    DEFAULT_ENV_PREFIXES,
    CaptureStatus,
    CodeProvenance,
    ConfigResolver,
    DeclaredCheckpoint,
    ManifestError,
    ManifestStore,
    RunManifest,
    Topology,
    capture_code_provenance,
    capture_environment,
    declared_from_hf_config,
)

EXIT_OK = 0
EXIT_REFUSED = 1
# 2 belongs to argparse; 3 keeps "could not establish inputs" its own signal,
# mirroring the probe's convention so operators learn one vocabulary.
EXIT_UNMEASURED = 3

_STORE_SUBDIR = "provenance"
"""Store root within the run dir. Fixed deliberately: one knob fewer to drift
between launchers, and the discovery contract lives at the checkpoints dir,
not here."""

_LORA_ABSTENTION = (
    "LoRA run: NO declared block emitted. The tensor-FQN set of an adapter "
    "checkpoint is a property of the PEFT implementation inside the training "
    "repository (not importable on a login node, not verifiable on this "
    "estate today), and the base checkpoint's FQN census is the WRONG "
    "denominator for adapter tensors — it counts the frozen base, which an "
    "adapter save does not contain. Fabricating r x targets x suffix strings "
    "from this launcher's own CLI would manufacture a denominator out of the "
    "very strings a misconfigured launcher would get wrong: self-certification. "
    "declared=None is the honest emission; checkpoint.save_complete keeps its "
    "honest VACUOUS, the expert gates keep their honest SKIP, and this line is "
    "the stated abstention. An honest VACUOUS beats a false PASS."
)

# ---------------------------------------------------------------------------
# #79 — the abstention RECORD. A deliberate abstention and a field nobody ever
# populated used to be the same bytes on disk ("declared": null), yet they
# demand opposite responses. The null DECLARED field itself stays exactly as
# the gates know it (their honest VACUOUS/SKIP depends on it); what changes is
# that a --lora emission now also records, in the manifest's own config block,
# the five entries named below: who abstained, why, what derives the set
# instead, and how many save dirs pre-existed. The config block is the channel
# this file can defend without inventing schema: the watched-env mechanism
# already records ABSENCES there as facts rather than invisible holes ("a
# recorded absence is a fact"). The keys tuple is the single source both the
# producer and the on-disk verifier zip over, so the two cannot drift.
# ---------------------------------------------------------------------------
_LORA_ABSTENTION_SOURCE = "measured:lora-abstention"
_LORA_ABSTENTION_RECORD_KEYS: tuple[str, ...] = (
    "declared.status",
    "declared.abstained_by",
    "declared.abstention_reason",
    "declared.superseded_by",
    "declared.preexisting_iter_dirs",
)
_LORA_ABSTENTION_WHO = (
    "tools/emit_run_manifest.py (--lora mode, at launch time; the LoRA "
    "launcher deliberately passes neither --base-checkpoint nor --hf-config — "
    "the designed abstention of the 'abstention, stated' provenance block in "
    "launchers/launch_g4e4b_lora_1tray.sh)"
)
_LORA_ABSTENTION_REASON = (
    "the tensor-FQN set of an adapter save is a property of the PEFT "
    "implementation inside the training repository (not importable on the "
    "emission plane, not verifiable on this estate); the converted base "
    "checkpoint census counts the FROZEN base an adapter save does not "
    "contain; and minting r x targets x suffix strings from the launcher's own "
    "CLI would self-certify the denominator out of the very strings a "
    "misconfigured launcher would get wrong. Full rationale: _LORA_ABSTENTION "
    "in this module and the launcher block named in declared.abstained_by."
)
# What supersedes the abstention — named on the record even before its wiring
# lands (fix46 shard T1 owns the gate side), with the measured numbers and
# their denominators, because this record is the artifact a future auditor
# reads to decide whether this null was a choice or an omission.
_LORA_ABSTENTION_SUPERSEDED_BY = (
    "launchers/lora_target_census.py (#78 census oracle; gate-side wiring in "
    "progress under fix46 shard T1): declared_fqns = census attachment parent "
    "FQNs (168 measured on Gemma-4-E4B, every one carrying the single "
    "'module.' Megatron-save wrapper segment, which is stripped) x "
    "'.adapter.linear_in.weight' + '.adapter.linear_out.weight' = 336 declared "
    "adapter tensors vs 336 actual (missing 0 of 336, extra 0 of 336 — "
    "EQUIVALENCE EXACT; oracle control drop-1-real+inject-1-phantom FIRED "
    "missing=1 extra=1). Until that wiring lands, checkpoint.save_complete "
    "keeps its honest VACUOUS — an honest VACUOUS beats a false PASS."
)
_TRAINING_STACK_SOURCE = "measured:training-stack"
# The drill that proves the #79 controls can fire (#56's rule one layer out):
# suppress the abstention record so the on-disk verifier MUST refuse the save.
# Estate drill idiom: an armed drill that the run survives is itself the
# control's reported failure, never a pass with extra steps.
_DRILL_BARE_NULL_ENV = "FS_EMIT_DRILL_BARE_NULL"
_DECLARED_NULL_RE = re.compile(r'"declared"\s*:\s*null')
# A declared_fqns array serialized EMPTY on disk. The in-memory derivation
# refuses a zero-tensor census, so this shape can only be a store/serializer
# regression — the founding all([]) one persistence layer out (B3).
_EMPTY_DECLARED_FQNS_RE = re.compile(r'"declared_fqns"\s*:\s*\[\s*\]')
# The in-manifest record of the one abstaining field of a full-ft declaration
# (B2): status, reason, who, what-supersedes, and the measured context,
# carried as config entries in the SAME five-field shape as the LoRA
# abstention record so the artifact itself distinguishes "abstained" from
# "never populated" and the same object-key oracle verifies both.
_FULL_FT_BYTES_ABSTENTION_RECORD_KEYS: tuple[str, ...] = (
    "declared.expected_expert_bytes.status",
    "declared.expected_expert_bytes.reason",
    "declared.expected_expert_bytes.abstained_by",
    "declared.expected_expert_bytes.superseded_by",
    "declared.expected_expert_bytes.context",
)
# ConfigResolver source for those entries; matches _SOURCE_RE's measured:
# class — stated in-band by the recording process at record time.
_FULL_FT_BYTES_ABSTENTION_SOURCE = "measured:full-ft-bytes-abstention"


class EmitRefused(RuntimeError):
    """Measured, and declined on policy or contradiction. Exit 1."""


class EmitUnmeasured(RuntimeError):
    """Could not establish an input. Exit 3 — never a verdict, never a skip."""


class BareNullDeclarationError(RuntimeError):
    """A run with saved tensors carries a bare-null declared field (#79).

    #56's rule one layer out: an abstaining declaration is a DECLARED STATE,
    not a silent pass — so the abstention has an in-manifest record, and the
    absence of BOTH declaration and record on a run with saves is a refusal,
    never a green.

    Reach, stated honestly (doctrine 5): NOTHING post-run raises or catches
    this today. The only live caller is the emitter auditing the bytes it
    just wrote; at resume/eval/export judgment time no consumer reads the
    UNCLEARED discrimination yet — the realized-save-count call site named
    for it is follow-on wiring that does NOT exist in this tree. Public so
    that wiring can land without a radius change, never to imply it has.
    """


def _positive_int(text: str) -> int:
    """argparse type for counts that must be positive; misuse is exit 2, loudly."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return value


def ensure_declaration_is_independent(base_ckpt: Path, judged_dir: Path) -> None:
    """Refuse the one derivation that would make the whole exercise vacuous.

    A completeness gate whose denominator is censused from the artifact under
    judgment passes a truncated save perfectly: the present set always equals
    whatever was present. That is the ``all([])`` shape wearing a file path.
    The full-FT denominator must be the BASE checkpoint — temporally prior
    (conversion predates training), artifactually separate. This function is
    where "must" stops being prose. ``resolve()`` collapses symlinks and
    ``..``; the samefile probe additionally collapses hardlink/alias spellings
    for the case where both paths exist. Nesting is refused in BOTH
    directions: a base inside the judged tree is contaminated; a judged tree
    inside the "base" means the caller pointed us at the output side of some
    earlier run and mislabeled it.
    """
    try:
        same = Path(base_ckpt).samefile(judged_dir)
    except OSError:
        same = False  # either side absent; the stat-identity layer below owns that case
    resolved_base = base_ckpt.resolve()
    resolved_judged = judged_dir.resolve()
    if same or resolved_base == resolved_judged:
        raise EmitRefused(
            f"refusing to derive the declared tensor set from the directory under "
            f"judgment: --base-checkpoint {base_ckpt} and --checkpoint-dir "
            f"{judged_dir} are the same location. The denominator must come from "
            f"an INDEPENDENT artifact (the converted base checkpoint), because "
            f"'what is there matches what is there' is not a check"
        )
    if resolved_base.is_relative_to(resolved_judged) or resolved_judged.is_relative_to(
        resolved_base
    ):
        raise EmitRefused(
            f"refusing to derive the declared tensor set from a path nested with "
            f"the directory under judgment: --base-checkpoint {base_ckpt} and "
            f"--checkpoint-dir {judged_dir} overlap (resolved: {resolved_base} vs "
            f"{resolved_judged}). A denominator drawn from inside the judged tree "
            f"— or one whose tree CONTAINS the judged tree — is the tautology "
            f"this tool exists to prevent"
        )

    # Stat-identity layer. Everything above compares SPELLINGS of a resolved
    # path, and the audit walked straight through it: one filesystem exported
    # under two mount points (/lustre/x vs /mnt/lustre/x), or a case-differing
    # spelling on a case-insensitive filesystem, yields unequal strings and
    # un-is_relative_to paths for what is byte-for-byte the same directory —
    # and the "independent" denominator was then censused from the judged tree
    # itself, certifying a truncated save complete against itself. Strings
    # describe paths; (st_dev, st_ino) NAMES the object. This layer refuses
    # whenever it cannot name both objects, because fail-closed is the rule for
    # the one property the whole tool stands on.
    def _identity(path: Path, label: str) -> tuple[int, int]:
        try:
            info = path.stat()
        except OSError as exc:
            raise EmitRefused(
                f"cannot establish independence: stat({label} {path}) failed "
                f"({type(exc).__name__}: {exc}). The launcher creates both "
                f"directories before emission, so an unstat-able side means the "
                f"mount layout itself is suspect; 'could not verify they "
                f"differ' must refuse rather than allow by assumption"
            ) from exc
        return info.st_dev, info.st_ino

    base_id = _identity(base_ckpt, "--base-checkpoint")
    judged_id = _identity(judged_dir, "--checkpoint-dir")
    # Nesting as identity, the st_dev/st_ino form of is_relative_to: the judged
    # tree sits inside the base tree iff the base's identity appears among the
    # judged path's ancestors; symmetrically the other way. String-identical
    # cases already exited above; what remains are mount aliases and case
    # spellings, which resolve() cannot canonicalize but os.stat collapses.
    judged_chain: set[tuple[int, int]] = {
        _identity(ancestor, "--checkpoint-dir ancestor")
        for ancestor in (resolved_judged, *resolved_judged.parents)
    }
    if base_id in judged_chain:
        raise EmitRefused(
            f"refusing to derive the declared tensor set: --base-checkpoint "
            f"{base_ckpt} and --checkpoint-dir {judged_dir} are the same object "
            f"or nested by device/inode identity (a bind mount, a second mount "
            f"path for one filesystem, or a case-variant spelling — all "
            f"invisible to resolved-string comparison). The tautology does not "
            f"become a check at a different mount spelling"
        )
    base_chain: set[tuple[int, int]] = {
        _identity(ancestor, "--base-checkpoint ancestor")
        for ancestor in (resolved_base, *resolved_base.parents)
    }
    if judged_id in base_chain:
        raise EmitRefused(
            f"refusing to derive the declared tensor set from a tree CONTAINED "
            f"by the directory under judgment: --base-checkpoint {base_ckpt} "
            f"lies inside --checkpoint-dir {judged_dir} by device/inode "
            f"identity (mount-alias nested). The caller pointed the denominator "
            f"at the output side of some earlier tree"
        )


def _read_json_mapping(path: Path, owner: str) -> dict[str, Any]:
    """Read and shape-check a JSON object, failing as UNMEASURED throughout.

    A config that cannot be read is not a config that declares dense: every
    failure here is exit 3, so a corrupt or missing ``config.json`` can never
    launder itself into a classification.
    """
    if not path.is_file():
        raise EmitUnmeasured(
            f"{owner} not found: {path} — without it the run cannot be honestly "
            f"classified dense-vs-MoE, and an assumed classification is the "
            f"defect class this tool refuses"
        )
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EmitUnmeasured(
            f"{owner} unreadable or not valid JSON: {path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(data, dict):
        raise EmitUnmeasured(f"{owner} is not a JSON object: {path}")
    return data


def _enable_moe_block_flag(config: dict[str, Any]) -> tuple[bool | None, str]:
    """Read the AFFIRMATIVE MoE declaration, ``text_config`` scope before top level.

    Returns ``(flag, dotted_scope_name)``; ``(None, "")`` means the key is
    absent in both scopes (Mixtral-class configs never carry it — absence is
    fine, it just means classification rests on the routed-count keys alone).
    A present-but-non-boolean value is a REFUSAL: a string ``"false"`` that
    truthy-parses as MoE — or a typo'd ``ture`` that parses as nothing — is
    exactly the kind of quiet misclassification this flag was read to prevent.
    """
    # The key name is read through the manifest module's single definition
    # (imported above): the probe reads the same constant, so the emitter and
    # the probe can never drift on WHICH key is affirmative. The rendered
    # strings are byte-identical to the literals they replace.
    scopes: list[tuple[str, dict[str, Any]]] = []
    nested = config.get("text_config")
    if isinstance(nested, dict):
        scopes.append((f"text_config.{_ENABLE_MOE_BLOCK_KEY}", nested))
    scopes.append((_ENABLE_MOE_BLOCK_KEY, config))
    for dotted, scope in scopes:
        if _ENABLE_MOE_BLOCK_KEY not in scope:
            continue
        raw = scope[_ENABLE_MOE_BLOCK_KEY]
        if not isinstance(raw, bool):
            raise EmitRefused(
                f"{dotted} is present but not a JSON boolean ({raw!r}); a stated "
                f"declaration of the wrong type is refused, not coerced — coerce "
                f"it and a quoted 'false' would classify as MoE while looking "
                f"like a config fact"
            )
        return raw, dotted.rsplit(f".{_ENABLE_MOE_BLOCK_KEY}", 1)[0] or "top level"
    return None, ""


def _declared_fqns_from_base(
    base_ckpt: Path,
) -> tuple[tuple[str, ...], str, int, tuple[str, ...]]:
    """Census the independent base artifact's REAL tensors (never the blobs).

    The inventory read is lazy-imported so this module stays loadable even
    where the checkpoint subpackage cannot be imported at all; on the estate's
    login node the header-parse path is torch-free BY DESIGN (see the
    dcp_meta module docstring), so absence of that import is a broken
    deployment, surfaced as UNMEASURED rather than worked around.

    Returns ``(fqns, format, excluded_blob_count, expert_family_fqns)``; the
    fourth element is the artifact half of the dense/MoE corroboration contract
    in :func:`derive_declared_full_ft`. The vacuity guard is the
    point of the return shape: a base whose census filters down to ZERO real
    tensors yields ``declared_fqns=()`` downstream, which the gates read as
    *no denominator* — while a casual reader sees a manifest that "has a
    declared block". ``read_metadata`` already refuses zero-KEY sources; this
    refusal covers the subtler shape, a nonempty listing that is all
    ``_extra_state`` bookkeeping (the 8,042-of-8,970 lesson: keys are not
    tensors).
    """
    try:
        from foundationscale.checkpoint.dcp import CheckpointFormatError
        from foundationscale.checkpoint.dcp_meta import read_metadata
        from foundationscale.gates.checkpoint_gates import (
            _expert_named,
            _matches_expert_family,
        )
    except ImportError as exc:  # pragma: no cover - import wiring is exercised in tests
        raise EmitUnmeasured(
            f"cannot import the checkpoint metadata reader and the gates' own "
            f"expert-family classifiers ({exc}); neither the declared FQN "
            f"census nor its expert-family count can be taken, so the declared "
            f"block cannot be built honestly — refusing to emit a hollow one"
        ) from exc
    try:
        meta = read_metadata(os.fspath(base_ckpt))
    except (CheckpointFormatError, OSError) as exc:
        raise EmitUnmeasured(
            f"base checkpoint unreadable as a checkpoint: {base_ckpt}: {exc}. The "
            f"denominator comes from this artifact, so if it cannot be read the "
            f"run has no honest declared set. Re-run the conversion (never trust "
            f"its rc=0) and emit again"
        ) from exc
    # The exclusion rule is stated twice and must stay twice-consistent:
    # capture_state_dict_keys drops any FQN containing "_extra_state"; the
    # gates filter on the is_extra_state FLAG plus the same substring
    # (checkpoint_gates.py:1567 per the task measurement). Declaring with any
    # other rule would let the declared set and the judged set count different
    # populations while both calling themselves tensors.
    real = [
        fqn
        for fqn, tm in meta.tensors.items()
        if not (tm.is_extra_state or "_extra_state" in fqn)
    ]
    fqns = sorted(real)
    excluded = len(meta.tensors) - len(fqns)
    if not fqns:
        raise EmitRefused(
            f"base checkpoint {base_ckpt} yields 0 declarable tensors "
            f"({len(meta.tensors)} named entries, all of them extra_state "
            f"bookkeeping/blobs): a zero-length declared_fqns reads downstream "
            f"as NO denominator while looking like a completed census — the "
            f"founding all([]) shape. Refusing"
        )
    # The expert-family census over the same real-only population, classified
    # by the gates' OWN name-level atoms — never a paraphrased pattern, per the
    # controls contract the probe already holds. This count is the artifact
    # half of the dense corroboration in derive_declared_full_ft: "census found
    # 0" must mean "of exactly the population the gates would examine", or a
    # diverged local regex could zero a population the gates' own selector
    # would indict, and the corroboration would certify dense over live
    # experts. The _expert_named half is included on purpose: an unrecognized
    # MoE layout must still count, because fail-closed-on-unknown is the
    # gates' rule too.
    expert_fqns = sorted(
        fqn for fqn in real if _expert_named(fqn) or _matches_expert_family(fqn)
    )
    return tuple(fqns), meta.format, excluded, tuple(expert_fqns)


def derive_declared_full_ft(
    base_ckpt: Path, judged_dir: Path, hf_config: Path
) -> tuple[DeclaredCheckpoint, dict[str, Any]]:
    """Build the declared block for a FULL fine-tune, or refuse to fake one.

    Independence of the FQN denominator is enforced first and separately from
    value assembly, so no future edit to the classification logic can bury the
    guard. Counts (num_experts / num_moe_layers) reuse
    :func:`declared_from_hf_config`, which already refuses invented depths
    (interleave keys force an abstaining None with the refusal stated in
    ``moe_layer_basis``); the affirmative ``enable_moe_block`` overlay below
    decides WHETHER those counts may stand.
    """
    ensure_declaration_is_independent(base_ckpt, judged_dir)
    config_data = _read_json_mapping(hf_config, "HF config")
    flag, flag_scope = _enable_moe_block_flag(config_data)
    try:
        producer = declared_from_hf_config(config_data)
    except ValueError as exc:
        # The config STATED something invalid (a non-positive count, an
        # unpriced dtype). That is a refusal, not an absence: the file exists
        # and declares, and what it declares cannot be honoured as-is.
        raise EmitRefused(
            f"HF config {hf_config} declares invalidly ({exc}); fix the config or "
            f"teach the producer (extend _KNOWN_DTYPE_WIDTHS / the understood "
            f"count keys) — do not narrow the declaration to whatever parses"
        ) from exc

    # The census is taken BEFORE classification: an affirmative dense statement
    # is only honorably writable as num_experts=0 when the INDEPENDENT base
    # artifact is ALSO free of expert-family tensors. Two sources, both
    # measured, before one declaration.
    fqns, fmt, excluded, expert_fqns = _declared_fqns_from_base(base_ckpt)
    census_count = len(expert_fqns)

    if flag is True and producer.num_experts is None:
        raise EmitRefused(
            f"{flag_scope}.enable_moe_block is TRUE — an affirmative declaration "
            f"of MoE structure — but no routed-expert count was found under the "
            f"keys this tool understands (num_local_experts, n_routed_experts, "
            f"num_experts). Declaring dense here is the measured top-level-only "
            f"Gemma-4-26B defect; declaring experts without a count is "
            f"fabrication. Refusing: extend the understood keys or correct the "
            f"config"
        )
    if flag is True and census_count == 0:
        # The mirror-image disagreement: the config affirms MoE but the base
        # holds no expert-family tensor at all. Either the config mislabels the
        # model or --base-checkpoint points at the wrong artifact; refusing
        # names both rather than letting the gates discover it run by run.
        raise EmitRefused(
            f"{flag_scope}.{_ENABLE_MOE_BLOCK_KEY} is TRUE with routed count "
            f"{producer.num_experts}, but the independent base census found 0 "
            f"expert-family tensors in {base_ckpt} (of {len(fqns)} real "
            f"tensors) — the two sources this tool trusts disagree. Believing "
            f"the config arms expert gates against a base that is not MoE; "
            f"believing the census contradicts a stated config. Refusing"
        )
    if flag is False:
        # The affirmative key GOVERNS, even over a dormant count. Some estate
        # configs carry an expert count the block flag disables; obeying the
        # count would declare MoE for a dense run (a false MoE declaration
        # arms expert gates against a checkpoint that contains no experts —
        # VACUOUS-by-mismatch, a false failure), and silently dropping the
        # count would erase what the config said. The count is therefore
        # recorded in the basis text while the DECLARED fields follow the flag.
        if census_count > 0:
            flag_and_count = producer.num_experts
            raise EmitRefused(
                f"{flag_scope}.{_ENABLE_MOE_BLOCK_KEY} is false — an "
                f"affirmative dense declaration (dormant count key: "
                f"{flag_and_count}) — but the independent base census found "
                f"{census_count} expert-family tensor(s) (first: "
                f"{expert_fqns[0]}). The config and the artifact contradict "
                f"each other, and either silent winner is a defect: declaring "
                f"dense disarms the expert gates over live experts (the "
                f"Gemma-4-26B trap inverted); declaring MoE fabricates a count "
                f"the config refused to state. Refusing: fix the config, or "
                f"point --base-checkpoint at the right artifact"
            )
        # Both sources agree dense, so the declaration is recorded POSITIVELY
        # as num_experts=0 — the change that lets a healthy dense first save
        # verify 1/1 applicable instead of blocking 1/3 forever (the gates read
        # None as UNKNOWN and fail closed on it; a gate that cannot pass on a
        # healthy artifact is a gate operators switch off).
        basis = (
            f"dense: {flag_scope}.enable_moe_block is false — an affirmative "
            f"declaration READ, not inferred from a count key's absence — AND "
            f"the independent base census found 0 expert-family tensors (of "
            f"{len(fqns)} real tensors): two independent sources agree, so the "
            f"dense declaration stands as num_experts=0"
        )
        if producer.num_experts is not None:
            basis += (
                f"; the config also carries a dormant routed-expert count "
                f"({producer.num_experts}) which the false flag disables — "
                f"recorded here instead of silently obeyed or silently dropped"
            )
        core = replace(
            producer, num_experts=0, num_moe_layers=None, moe_layer_basis=basis
        )
        mode_note = "dense (enable_moe_block=false; base census: 0 expert-family tensors)"
    elif flag is True:
        suffix = f"; MoE affirmed by {flag_scope}.enable_moe_block=true"
        core = replace(
            producer,
            moe_layer_basis=(producer.moe_layer_basis or "depth unresolved") + suffix,
        )
        mode_note = (
            f"MoE ({producer.num_experts} experts, affirmed; base census: "
            f"{census_count} expert-family tensors)"
        )
    else:
        # No affirmative key anywhere: the producer's routed-count verdict is
        # the only statement the config makes, and it is kept verbatim. The
        # census is stated but CANNOT promote absence into a 0: the two-source
        # rule needs one affirmative source from the config by construction.
        core = producer
        mode_note = (
            "classified by routed-count keys alone (no enable_moe_block key "
            f"present; base census: {census_count} expert-family tensors)"
        )

    declared = replace(
        core,
        declared_fqns=fqns,
        # Both the census source (conversion output) and the artifacts under
        # judgment (this run's saves) are Megatron dist-checkpoints, so the
        # FQN space is megatron-core; the producer's "hf-moe" would describe
        # the HF tree the run does NOT save.
        naming_convention="megatron-core",
        tensors_per_expert_layer=2,
        # expected_expert_bytes stays None — an abstention now recorded IN
        # THE ARTIFACT under declared.expected_expert_bytes.* (see
        # _full_ft_bytes_abstention_entries); this comment only explains the
        # why, the entries ARE the record. Pricing expert bytes honestly
        # requires knowing which FQN pattern the byte gate's selection
        # applies; guessing a regex against a gate this tool cannot see
        # would mint a denominator from a pattern-space paraphrase, and a
        # wrong denominator makes that gate lie quietly where None makes
        # it SKIP loudly.
    )
    info = {
        "mode": mode_note,
        "base": str(base_ckpt),
        "base_format": fmt,
        "declared_fqn_count": len(fqns),
        "excluded_blob_count": excluded,
        # The denominator of the dense corroboration travels with the record;
        # "0 expert-family tensors" in the basis is meaningless without "of N".
        "expert_family_census": census_count,
    }
    return declared, info


def _record_effective_pairs(
    resolver: ConfigResolver, pairs: list[str]
) -> None:
    """Enter the launcher's resolved knobs into the config block.

    Values arrive stringified from argv, which is honest: the emitter records
    what the launch actually used, at the point it was decided, and does not
    pretend to have watched the trainer resolve anything. Any richer claim
    belongs to a resolver inside the training process, which this estate does
    not yet have.
    """
    for pair in pairs:
        key, eq, value = pair.partition("=")
        if not eq or not key.strip():
            raise EmitRefused(
                f"--effective expects KEY=VALUE, got {pair!r} (this is a launcher "
                f"bug; refusing to record an anonymous value)"
            )
        try:
            resolver.record_effective(key.strip(), value, "cli")
        except ValueError as exc:
            raise EmitRefused(f"invalid effective-config entry {pair!r}: {exc}") from exc


def _record_watched_env(resolver: ConfigResolver, names: list[str]) -> None:
    """Record env-sourced knobs WITH the resolver's drift/absence findings.

    The estate's CoT switch (FOXBRAIN_GEMMA4_KEEP_COT) and corpus list are the
    local analog of the 24-run-split variable: decisive, env-carried, and
    historically unrecorded. When the variable is unset, the entry is still
    recorded — with a ``<unset>`` value and the resolver's own
    'unverifiable env source' finding — because a silently absent entry is an
    invisible-hole repetition of the split, while a recorded absence is a fact.
    """
    for name in names:
        live = os.environ.get(name)
        try:
            resolver.record_effective(name, live if live is not None else "<unset>", f"env:{name}")
        except ValueError as exc:
            raise EmitRefused(f"invalid --watch-env name {name!r}: {exc}") from exc


def _count_save_dirs(ckpt_dir: Path) -> int:
    """Count iter_* save directories already present (0 on any fresh launch).

    An OBSERVATION, never a judgment: the iter_NNNNNNN naming is the training
    stack's contract and completeness over those dirs is the gates' business.
    The count exists so the abstention record and the bare-null control carry
    their denominator (doctrine 2) — above all on RESUME launches, where the
    checkpoint dir is non-empty BY DESIGN at emission time, which makes the
    emitter itself the first honest witness to 'a run that SAVED tensors'.
    """
    return sum(1 for p in ckpt_dir.glob("iter_*") if p.is_dir())


def _record_stated_entries(
    resolver: ConfigResolver, entries: list[tuple[str, str]], source: str
) -> None:
    """Enter emitter-authored stated records into the config block.

    Same channel and discipline as the --effective pairs, but the source names
    the emitter rather than the CLI: these facts are authored here, at the
    point they were decided, and the source string must say so — a record
    whose provenance is itself ambiguous would restate the defect it repairs.
    """
    for key, value in entries:
        try:
            resolver.record_effective(key, value, source)
        except ValueError as exc:
            raise EmitRefused(
                f"invalid in-manifest record entry {key!r}: {exc}"
            ) from exc


def _training_stack_entries() -> list[tuple[str, str]]:
    """#83, manifest side: the torch build that TRAINS, for gate cross-check.

    The gate records which torch ADJUDICATED the artifact on every exit path
    (#83's gate half); an artifact adjudicated by a different build than the
    one that wrote it is exactly the 'good compression measured badly' class
    this estate exists to catch — but only if the training side is on record
    too. This samples the EMITTING interpreter at launch time, before one
    GPU-second: under the fix46 routing the emitter runs in the training
    container, so the sample is the stack training will use. Where torch
    cannot be imported here (a host python whose only torch sits in a
    user-site hidden by PYTHONNOUSERSITE=1 — correct and load-bearing for the
    training payload), the absence is recorded as a STATED abstention with the
    exception in-band; an invented version string would be worse than no
    field, so none is ever guessed. python_executable/python_version are
    always recorded: partial honest attribution beats none.
    """
    entries = [
        ("training_stack.python_executable", sys.executable or "<unset>"),
        ("training_stack.python_version", platform.python_version()),
    ]
    try:
        import torch  # optional BY DESIGN — never an import requirement here
    except ImportError as exc:
        entries.append(
            (
                "training_stack.torch_record",
                "ABSTAINED: torch is not importable in the emitting "
                f"interpreter ({type(exc).__name__}: {exc}); the training "
                "stack is the container's, which a host-side emission cannot "
                "honestly sample — recorded as an absence rather than guessed "
                "(#83). The adjudicating gate DOES record torch provenance "
                "on every exit path, UNMEASURED included: its foundationscale.gates."
                "adjudication._interpreter_provenance writes a torch_record under this "
                "entry's spelling by design, with torch LOCATED via importlib "
                "find_spec + dist metadata, never imported, so the "
                "measurement cannot move the measured. This arm carries no "
                "in-emitter torch reading, so there is nothing here to set "
                "against the gate's record. No automated comparison runs in "
                "this shard: the emitter reads no gate report, and the "
                "comparison is deferred to the shard that adds that reader "
                "(named abstention, #83 follow-up).",
            )
        )
    else:
        entries.append(
            (
                "training_stack.torch_record",
                f"measured in-emitter at launch time (pre-GPU): torch "
                f"{torch.__version__} at {torch.__file__} — this samples the "
                "emitting interpreter only, by IMPORTING torch and reading "
                "__version__/__file__. The adjudicating gate also writes a "
                "torch_record for the same fact under the same spelling on "
                "every exit path — foundationscale.gates.adjudication._interpreter_provenance — "
                "but with a deliberately different instrument: torch is "
                "LOCATED via importlib find_spec + dist metadata, never "
                "imported, so the measurement cannot move the measured. The "
                "two instruments can disagree for honest reasons (module "
                "__version__ vs dist metadata; __file__ vs find_spec "
                "origin), so a mismatch between the two records indicts the "
                "instrument pairing first and demands reconciliation; by "
                "itself it convicts neither the training build nor the "
                "model, and agreement is corroboration, not certainty. No "
                "automated comparison runs in this shard: the emitter reads "
                "no gate report; both records exist so the shard that adds "
                "that reader can set them against each other (named "
                "abstention, #83 follow-up).",
            )
        )
    return entries


def _lora_abstention_record_entries(preexisting_saves: int) -> list[tuple[str, str]]:
    """The five-entry record that makes this null a DECLARED state (#79).

    Keys and values are zipped from the module's single keys tuple (strict:
    lengths cannot diverge silently), so the on-disk marker predicate and the
    producer share one enumerable definition by construction.
    """
    values = (
        "abstained",
        _LORA_ABSTENTION_WHO,
        _LORA_ABSTENTION_REASON,
        _LORA_ABSTENTION_SUPERSEDED_BY,
        str(preexisting_saves),
    )
    return list(zip(_LORA_ABSTENTION_RECORD_KEYS, values, strict=True))


def _abstention_markers_absent(record_text: str) -> list[str]:
    """Abstention-record FIELD keys missing from the SERIALIZED record text.

    Serialized-text key matching is a deliberate, stated oracle: it checks
    exactly the property the on-disk control stands on — that the record's
    fields survived serialization — and presumes exactly ONE measured fact
    about the manifest's JSON layout: each record field is an OBJECT KEY.
    Measured on real emitted bytes: every record key appears TWICE, once as
    the object key (``"declared.status": {`` …) and once echoed as the value
    of its entry's inner ``key`` field; quoted-anywhere count 2, with-colon
    count exactly 1 per field. A bare quoted-substring predicate matches BOTH
    copies, so a dropped field still read present on the strength of its own
    echo — the drop-one control caught exactly that. The predicate below
    therefore matches only the object-key spelling (``"<key>":``); the echo
    is never counted. A spill from ANOTHER field remains non-confusable (the
    dotted spellings are this module's own); the confusable second copy was
    always the record's OWN echo, and the sentence this docstring previously
    shipped denying that was broader than its evidence (doctrine 5) — removed,
    not softened. Drift mode, named: if the serializer ever stops writing
    these fields as object keys, ALL FIVE read missing at once — fail-closed,
    a false alarm priced like a false green, but diagnosable from the message
    alone (5-of-5 wholesale reads as serializer drift; a subset reads as a
    genuine partial record). Field presence is ALL this oracle verifies: an
    intact key over an emptied entry payload, or a lost echo whose field
    survives, is outside this control's class and is not claimed here.
    """
    return [k for k in _LORA_ABSTENTION_RECORD_KEYS if f'"{k}":' not in record_text]


def check_saved_run_declaration(record_text: str, *, saves_observed: int) -> str:
    """The #79 control: bare-null declared + saved tensors = FAIL.

    #56's fix applied one layer out, to the manifest. A run that SAVED tensors
    and whose manifest carries a bare-null ``declared`` FAILS — now that the
    emitter records its abstentions, a bare null can only be an omission, and
    an omission on a run with saves means the first-save gates were VACUOUS
    without anyone ever choosing that. Zero-denominator policy (doctrines
    1/3): ``saves_observed == 0`` returns a NAMED not-exercised state that
    must never be calibrated into a pass.

    Returns a state string, never a bare bool — every outcome here is a named
    state, matching the estate's exit-vocabulary discipline — and raises
    :class:`BareNullDeclarationError` on the failing shapes. Call-site
    honesty (doctrine 5): the ONLY invocation in this tree is the LoRA
    emission path below, auditing the record it just wrote — launch-time
    coverage that already includes RESUME launches (their checkpoint dir is
    non-empty by design, so their pre-existing saves DO form a denominator
    here). FULL-FT launches never invoke this control: they carry a
    populated declared block, so the bare-null class does not apply, and
    they answer to the serialized-record enforce in main instead
    (_enforce_full_ft_declared_on_disk). NO post-run consumer exists: at
    resume/eval/export judgment time nothing calls this with the realized
    save count, so today UNCLEARED can fire only at emission (the drill, or
    a store/serializer regression). That judgment-time wiring is follow-on
    work outside this file, and this docstring must not be read as claiming
    it has landed.
    """
    if saves_observed < 0:
        raise ValueError(f"saves_observed must be >= 0, got {saves_observed}")
    if saves_observed == 0:
        return (
            "NOT-EXERCISED: 0 saved-artifact directories observed — there is no "
            "denominator for this control to adjudicate, and zero units "
            "examined is never a pass (stated by name, never implied)"
        )
    if _DECLARED_NULL_RE.search(record_text) is None:
        if '"declared"' not in record_text:
            raise BareNullDeclarationError(
                f"{saves_observed} saved-artifact dir(s) observed, and the "
                f"manifest carries NO 'declared' key at all — a serialized "
                f"shape this control does not recognize (record predates the "
                f"schema, or is corrupted). Fail closed"
            )
        return (
            "DECLARED: a declared block is present; completeness over it is "
            "checkpoint.save_complete's own gated business with its own "
            "controls — this control's class (the bare-null abstention) does "
            "not apply"
        )
    missing = _abstention_markers_absent(record_text)
    if missing:
        raise BareNullDeclarationError(
            f"{saves_observed} saved-artifact dir(s) observed and the manifest's "
            f"declared field is a BARE null: {len(missing)} of "
            f"{len(_LORA_ABSTENTION_RECORD_KEYS)} abstention-record fields "
            f"missing ({', '.join(missing)}). A deliberate abstention and a "
            f"field nobody populated were once the same bytes; after #79 that "
            f"ambiguity is a defect BY CONSTRUCTION. These saves are UNCLEARED "
            f"for resume/eval/export until the manifest's abstention record is "
            f"repaired"
        )
    return (
        "STATED-ABSTENTION: declared is null WITH the in-manifest abstention "
        "record verified complete at the FIELD level on the SERIALIZED "
        "bytes: each field matched by its object-key spelling only, never by "
        "the key-echo the serializer writes inside every entry (that echo "
        "survives the loss of its own field — measured on real emitted bytes "
        "— so it is not evidence for presence) "
        f"({len(_LORA_ABSTENTION_RECORD_KEYS)} of "
        f"{len(_LORA_ABSTENTION_RECORD_KEYS)} record-field keys verified "
        "present in the serialized manifest: who, why, "
        "what-derives, save-count context) — a DECLARED STATE, not a silent "
        "pass"
    )


def _enforce_lora_abstention_record(
    record_text: str, *, saves_observed: int, drill_armed: bool
) -> tuple[str, int, int]:
    """Refuse the just-written record if the abstention did not survive to disk.

    Two layers, one fixed order: (1) the save-semantics control
    (:func:`check_saved_run_declaration` — named states, bare-null refusal;
    exit 1 class), then (2) serializer fidelity: the markers the in-memory
    manifest held MUST be present in the bytes on disk, because provenance
    that lives only in stdout is the #79 defect restated (exit 3 class).
    Returns ``(state, present, total)`` so the emit readout carries its own
    denominator. A suppressed record under ``FS_EMIT_DRILL_BARE_NULL=1`` is
    reported with the DRILL FIRED token on whichever arm catches it — the
    token, not a single rc, is the drill's signature: which arm fires depends
    on how many saves pre-existed, and both arms fail closed.
    """
    drill_prefix = f"DRILL FIRED ({_DRILL_BARE_NULL_ENV}=1): " if drill_armed else ""
    try:
        state = check_saved_run_declaration(record_text, saves_observed=saves_observed)
    except BareNullDeclarationError as exc:
        raise EmitRefused(
            f"{drill_prefix}the LoRA abstention-record control refused the "
            f"manifest this emission just wrote: {exc}"
            + (
                " — the control fires end-to-end through the real store path; "
                "unset the drill to launch (an armed drill that launches "
                "anyway is the control's failure, never its pass)"
                if drill_armed
                else ""
            )
        ) from exc
    missing = _abstention_markers_absent(record_text)
    if missing:
        raise EmitUnmeasured(
            f"{drill_prefix}the serialized record lacks {len(missing)} of "
            f"{len(_LORA_ABSTENTION_RECORD_KEYS)} abstention markers it held "
            f"in memory ({', '.join(missing)}) — either the drill suppressed "
            f"the record (expected under the drill) or the store/serializer "
            f"dropped fields: provenance on stdout that does not survive to "
            f"disk is #79 restated, so emission fails closed"
        )
    total = len(_LORA_ABSTENTION_RECORD_KEYS)
    return state, total - len(missing), total


def _full_ft_bytes_abstention_entries(
    *, declared_fqn_count: int, expert_family_census: int
) -> list[tuple[str, str]]:
    """The full-ft ``expected_expert_bytes`` abstention, as in-manifest facts (#79).

    Full-ft derivation leaves ``declared.expected_expert_bytes`` as ``None``
    — a deliberate abstention whose statement used to live ONLY in a source
    comment and a stdout line, i.e. nowhere in the artifact: on disk it was
    indistinguishable from "never populated", which is #79's own defect
    class on an in-scope field. These entries record the abstention in the
    same five-field shape as the LoRA record (status, reason, who,
    what-supersedes, measured context), zipped against
    :data:`_FULL_FT_BYTES_ABSTENTION_RECORD_KEYS` so a drifted value list
    dies at build, not on disk. The context entry carries the denominators
    measured where the decision was made: an abstention that does not state
    what it measured is a claim without a denominator.
    """
    values = [
        "abstained",
        (
            "pricing expert bytes requires knowing which FQN pattern the byte "
            "gate selects on; minting one from a regex paraphrased off a gate "
            "this emitter cannot see would fabricate a denominator that makes "
            "that gate lie quietly, where None makes it SKIP loudly"
        ),
        "tools/emit_run_manifest.py derive_declared_full_ft",
        (
            "the trainer supplying expected_expert_bytes from resolved shapes "
            "and dtype widths post-launch (the DeclaredCheckpoint contract: "
            "None leaves that denominator for the trainer to supply)"
        ),
        (
            f"measured at emission: declared_fqns={declared_fqn_count} real "
            f"base-checkpoint tensors censused, {expert_family_census} of "
            f"them expert-family by the gates' own census atoms"
        ),
    ]
    return list(zip(_FULL_FT_BYTES_ABSTENTION_RECORD_KEYS, values, strict=True))


def _enforce_full_ft_declared_on_disk(record_text: str) -> tuple[int, int]:
    """Serialized-bytes re-verification for full-ft emissions (the B3 hole).

    The in-memory derivation refuses a zero-tensor census, but that guard
    dies at the store boundary: a store or serializer regression that wrote
    the populated declared block as bare null, or emptied ``declared_fqns``
    ON DISK, would have emitted green and handed the checkpoint gates a
    silent ``all([])`` — until now the whole re-read + enforce block was
    gated ``if not args.full_ft``. This control reads the persisted bytes
    with the same object-key oracle as the LoRA fidelity layer (a key's
    echo inside its own entry is never counted: only the with-colon
    spelling proves presence, a discrimination the B1 tests pin against the
    real serializer) and fails closed as EmitUnmeasured — the emission's
    own persistence failed, which is exit-3 unmeasured, never a verdict.

    Scope stated honestly (doctrine 5): this control refuses three measured
    shapes — (1) the declared block serialized as bare null, (2)
    ``declared_fqns`` absent or emptied on disk, (3) abstention fields
    dropped from the serialized config. A PARTIAL field rewrite short of
    those shapes is outside its evidence and is not claimed: full
    in-memory/on-disk equality would need a pinned round-trip-fidelity
    guarantee through the store's load path, which this tree does not yet
    have. Returns (present, total) over the five abstention fields so the
    readout carries its own denominator (doctrine 2).
    """
    if _DECLARED_NULL_RE.search(record_text) is not None:
        raise EmitUnmeasured(
            "a full-ft emission held a POPULATED declared block in memory, "
            'but the serialized manifest reads "declared": null — the '
            "store or serializer dropped the census the derivation refused "
            "to emit empty, and the gates would now read a fabricated "
            "UNKNOWN where a denominator existed. Failing closed"
        )
    if '"declared_fqns":' not in record_text or _EMPTY_DECLARED_FQNS_RE.search(
        record_text
    ):
        raise EmitUnmeasured(
            "a full-ft emission censused a NON-EMPTY declared_fqns in "
            "memory, but the serialized manifest has the key missing or "
            "empty: on disk that reads downstream as NO denominator while "
            "looking like a completed census — the founding all([]) shape, "
            "one persistence layer out. Failing closed"
        )
    missing = [
        key
        for key in _FULL_FT_BYTES_ABSTENTION_RECORD_KEYS
        if f'"{key}":' not in record_text
    ]
    if missing:
        raise EmitUnmeasured(
            f"the serialized full-ft record lacks {len(missing)} of "
            f"{len(_FULL_FT_BYTES_ABSTENTION_RECORD_KEYS)} "
            f"expected_expert_bytes abstention fields it held in memory "
            f"({', '.join(missing)}) — the store/serializer dropped "
            f"stated-abstention fields, so the artifact no longer "
            f"distinguishes abstained from never populated. Failing closed"
        )
    total = len(_FULL_FT_BYTES_ABSTENTION_RECORD_KEYS)
    return total - len(missing), total


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="emit_run_manifest",
        description="Emit the launch-time RunManifest (with the declared "
        "checkpoint block for full-FT runs) beside the run's future "
        "checkpoints, where load_manifest will find it. Exit 0 = emitted; "
        "1 = refused on policy/contradiction; 3 = inputs not establishable.",
    )
    p.add_argument("--run-id", required=True, help="logical run name; attempts key off it")
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="run root; the canonical ManifestStore lands at <out-dir>/provenance",
    )
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="the directory that will hold iter_* saves; the attempt-keyed "
        "discovery link lands HERE, because load_manifest searches a probed "
        "artifact's directory and its parent only",
    )
    p.add_argument("--job-id", default=None)
    p.add_argument("--nodes", required=True, type=_positive_int)
    p.add_argument("--gpus-per-node", required=True, type=_positive_int)
    p.add_argument("--tp", required=True, type=_positive_int)
    p.add_argument("--pp", required=True, type=_positive_int)
    p.add_argument(
        "--cp",
        required=True,
        type=_positive_int,
        help="context parallel size; recorded in the manifest's topology and "
        "its fingerprint — a launcher knob that is accepted but silently "
        "dropped is how CP=1 and CP=8 attempts once fingerprinted equal",
    )
    p.add_argument("--dp", required=True, type=_positive_int)
    p.add_argument(
        "--ep",
        default=None,
        type=_positive_int,
        help="expert parallel size, when expert parallelism is configured at all",
    )
    p.add_argument(
        "--env-prefix",
        action="append",
        default=[],
        metavar="PFX",
        help="extra environment-capture prefix, repeatable; appended to the "
        "module default allowlist (pass FOXBRAIN_ — the estate's decisive "
        "switches live under it and the default prefixes do not cover it)",
    )
    p.add_argument(
        "--watch-env",
        action="append",
        default=[],
        metavar="NAME",
        help="record a single environment variable as an effective-config "
        "entry with drift/absence findings, repeatable",
    )
    p.add_argument(
        "--effective",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="resolved config knob to record with source 'cli', repeatable",
    )
    p.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="output location to record (excluded from the fingerprint), repeatable",
    )
    p.add_argument(
        "--code-root",
        default=None,
        type=Path,
        help="repository root for git provenance; when omitted the record "
        "says NOT_A_REPOSITORY in-band instead of pretending a capture happened",
    )
    p.add_argument("--entrypoint", default=None, help="script actually launched")
    p.add_argument(
        "--diff-path",
        action="append",
        default=[],
        metavar="REL",
        help="repo-relative capture scope, repeatable; default is the whole repo",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--full-ft",
        action="store_true",
        help="full fine-tune: declared set censused from --base-checkpoint "
        "(independent artifact; nesting/equality with --checkpoint-dir refused)",
    )
    mode.add_argument(
        "--lora",
        action="store_true",
        help="adapter run: emit the manifest WITHOUT a declared block — the "
        "honest abstention; see the module docstring for the full argument",
    )
    p.add_argument(
        "--base-checkpoint",
        default=None,
        type=Path,
        help="(--full-ft only) the converted base checkpoint dir or safetensors "
        "file the run initializes from",
    )
    p.add_argument(
        "--hf-config",
        default=None,
        type=Path,
        help="(--full-ft only) HF config.json supplying the affirmative "
        "dense/MoE classification and routed-expert counts",
    )
    return p


def _emit(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    ckpt_dir: Path = args.checkpoint_dir
    if not out_dir.is_dir():
        raise EmitUnmeasured(
            f"--out-dir {out_dir} does not exist — the launcher creates the run "
            f"dir before emission; inventing it here would record provenance for "
            f"a tree the launcher has not validated"
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    declared: DeclaredCheckpoint | None
    declared_line: str
    # #79 drill state, decided once for the whole emission: an armed drill
    # suppresses the LoRA abstention record so the on-disk verifier after
    # store.save MUST refuse. Set-but-inert under --full-ft is stated in the
    # readout, never silent — an unread drill is the tripwire-with-a-banner
    # class. Both branch-local facts (entries, save count) are initialized for
    # every mode so no path can read them unbound under set -u-like discipline.
    drill_bare_null = os.environ.get(_DRILL_BARE_NULL_ENV) == "1"
    lora_abstention_entries: list[tuple[str, str]] | None = None
    lora_preexisting_saves = 0
    full_ft_bytes_entries: list[tuple[str, str]] | None = None
    if args.full_ft:
        declared, info = derive_declared_full_ft(
            args.base_checkpoint, ckpt_dir, args.hf_config
        )
        full_ft_bytes_entries = _full_ft_bytes_abstention_entries(
            declared_fqn_count=info["declared_fqn_count"],
            expert_family_census=info["expert_family_census"],
        )
        declared_line = (
            f"declared block: mode={info['mode']} | declared_fqns="
            f"{info['declared_fqn_count']} real tensors censused from INDEPENDENT "
            f"base {info['base']} ({info['base_format']}, {info['excluded_blob_count']} "
            f"extra_state/blob entries excluded with the gates' own counting "
            f"rule; expert-family census: {info['expert_family_census']} of "
            f"them — the artifact half of the dense corroboration) | "
            f"num_experts={declared.num_experts} num_moe_layers="
            f"{declared.num_moe_layers} expected_expert_bytes=None (stated "
            f"abstention — recorded in-manifest under "
            f"declared.expected_expert_bytes.*)"
        )
    else:
        declared = None
        declared_line = f"declared block: {_LORA_ABSTENTION}"
        lora_preexisting_saves = _count_save_dirs(ckpt_dir)
        if not drill_bare_null:
            lora_abstention_entries = _lora_abstention_record_entries(
                lora_preexisting_saves
            )
        # else: the drill IS the omission. With the record suppressed, the
        # on-disk verifier after store.save MUST refuse this emission; if it
        # does not, every non-drill assurance this file gives about #79 is
        # unproven, and a drill run reaching the readout is the control's own
        # failure, loudly.

    resolver = ConfigResolver()
    _record_watched_env(resolver, list(args.watch_env))
    _record_effective_pairs(resolver, list(args.effective))
    # #83 (manifest side) on EVERY run; #79's abstention record on every
    # non-drilled LoRA run. Both are in-manifest facts authored here, at the
    # point they were decided, with sources that name the emitter, not the CLI.
    _record_stated_entries(
        resolver, _training_stack_entries(), _TRAINING_STACK_SOURCE
    )
    if lora_abstention_entries is not None:
        _record_stated_entries(
            resolver, lora_abstention_entries, _LORA_ABSTENTION_SOURCE
        )
    if full_ft_bytes_entries is not None:
        _record_stated_entries(
            resolver, full_ft_bytes_entries, _FULL_FT_BYTES_ABSTENTION_SOURCE
        )

    # Deduplicated allowlist: the module default first, launcher extras after;
    # the allowlist is stored IN the manifest, so what was excluded stays
    # readable forever.
    prefixes = list(DEFAULT_ENV_PREFIXES)
    for pfx in args.env_prefix:
        if pfx and pfx not in prefixes:
            prefixes.append(pfx)
    environment = capture_environment(tuple(prefixes))

    if args.code_root is None:
        # Honest non-capture, in-band: the status says no repository was
        # probed, and _derive_findings converts that to a stated finding.
        code = CodeProvenance(
            status=CaptureStatus.NOT_A_REPOSITORY,
            root=None,
            commit=None,
            dirty_files=0,
            untracked_files=0,
            diff_sha256=None,
            diff_bytes=0,
            paths=(),
            entrypoint=args.entrypoint,
            entrypoint_captured=None,
        )
    else:
        code = capture_code_provenance(
            args.code_root,
            tuple(args.diff_path) or (),
            entrypoint=args.entrypoint,
        )

    artifacts: dict[str, str] = {}
    for pair in args.artifact:
        name, eq, value = pair.partition("=")
        if not eq or not name.strip():
            raise EmitRefused(f"--artifact expects NAME=PATH, got {pair!r}")
        artifacts[name.strip()] = value

    store = ManifestStore(out_dir / _STORE_SUBDIR)
    attempt = store.allocate_attempt(args.run_id)
    try:
        manifest = RunManifest(
            run_id=args.run_id,
            attempt=attempt,
            code=code,
            config=resolver.freeze(),
            environment=environment,
            topology=Topology(
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                tensor_parallel=args.tp,
                pipeline_parallel=args.pp,
                data_parallel=args.dp,
                expert_parallel=args.ep,
                context_parallel=args.cp,
            ),
            job_id=args.job_id,
            artifact_paths=artifacts,
            declared=declared,
        )
        saved = store.save(manifest)
    except (ManifestError, ValueError) as exc:
        raise EmitRefused(
            f"manifest build/store refused the record: {type(exc).__name__}: {exc}"
        ) from exc

    # #79 controls run against the bytes just persisted, BEFORE the discovery
    # link is placed: a record these controls refuse must never reach the
    # location the gates glob newest-first. The canonical store record itself
    # stays under provenance/ as append-only forensic evidence of the refused
    # attempt, per the store's no-clobber contract.
    lora_record_report = ""
    # Both modes re-read the bytes just persisted: an unreadable record must
    # never be fatal in one mode and unguarded in the other (doctrine 4).
    try:
        record_text = saved.read_text(encoding="utf-8")
    except OSError as exc:
        raise EmitUnmeasured(
            f"could not re-read the manifest record just written at {saved} "
            f"({type(exc).__name__}: {exc}). Provenance that cannot be "
            f"re-read at emission time is not provenance, and the on-disk "
            f"declaration controls would have nothing to stand on"
        ) from exc
    if not args.full_ft:
        state, present, total = _enforce_lora_abstention_record(
            record_text,
            saves_observed=lora_preexisting_saves,
            drill_armed=drill_bare_null,
        )
        lora_record_report = (
            f"abstention record : {present} of {total} record fields verified "
            f"present in the serialized manifest at {saved.name} (pre-existing "
            f"iter_* dirs at emission: {lora_preexisting_saves}) | declaration "
            f"control: {state}"
        )
    else:
        present, total = _enforce_full_ft_declared_on_disk(record_text)
        lora_record_report = (
            f"declared record   : {present} of {total} expected_expert_bytes "
            f"abstention fields verified present in the serialized manifest "
            f"at {saved.name}; declared block confirmed NON-NULL with "
            f"declared_fqns present and NON-EMPTY on the serialized bytes — "
            f"an on-disk all([]) fails closed above, never reaching the "
            f"gates. Measured shapes, stated honestly: a null/absent "
            f"declared block, an emptied declared_fqns, and dropped "
            f"abstention fields; persistence of those shapes is re-verified, "
            f"field-by-field equality with the in-memory record is not"
        )
        if drill_bare_null:
            lora_record_report += (
                f" | drill note   : {_DRILL_BARE_NULL_ENV}=1 is set but mode "
                f"is --full-ft; the drill targets the LoRA bare-null "
                f"abstention record and is INERT here — stated so an inert "
                f"drill can never read as an exercised control"
            )

    # Discovery copy: SAME content in the directory load_manifest searches.
    # os.link over a copy because two paths holding two byte streams are two
    # sources of truth; the link keeps one. FileExistsError is handled the
    # store's own way: byte-identical converges, divergent refuses.
    link = ckpt_dir / saved.name
    try:
        os.link(saved, link)
        discovery_note = "linked"
    except FileExistsError:
        if link.read_bytes() == saved.read_bytes():
            discovery_note = "already present byte-identically (converged)"
        else:
            raise EmitRefused(
                f"{link} already exists with DIFFERENT content — the checkpoints "
                f"dir carries an attempt record this launch did not write. "
                f"Refusing to shadow it: overwrite-in-place is the 27-lost-"
                f"manifests defect, even at the discovery layer"
            ) from None
    except OSError as exc:
        raise EmitUnmeasured(
            f"could not place the discovery hard link {link} (store and "
            f"checkpoints dir must share a filesystem; both live under the run "
            f"dir by launcher construction, so investigate the mount layout): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # The readout carries its own denominators: a reviewer never has to trust
    # "captured" without "N of M".
    print(f"run manifest emitted for run_id={args.run_id!r} attempt={attempt}")
    print(f"  canonical store record : {saved}")
    print(f"  discovery link         : {link} ({discovery_note}; newest-first "
          f"attempt-*.json glob in load_manifest finds it beside any iter_* save)")
    print(f"  fingerprint            : {manifest.fingerprint()}")
    print(f"  environment            : {environment.captured}/{environment.source_var_count} "
          f"variables captured under {len(prefixes)} prefixes ({', '.join(prefixes)})")
    print(f"  effective config       : {len(manifest.config)} entr(ies) recorded")
    print(f"  code provenance        : status={code.status.value} commit={code.commit}")
    print(f"  {declared_line}")
    if lora_record_report:
        print(f"  {lora_record_report}")
    print(f"  findings               : {len(manifest.findings)}")
    for finding in manifest.findings:
        print(f"      - {finding}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Mutual exclusivity of MODE is argparse's job; cross-constraints of the
    # mode with its inputs are usage too, so they die as exit 2 here, not as
    # refusals downstream.
    if args.full_ft:
        for flag_name in ("base_checkpoint", "hf_config"):
            if getattr(args, flag_name) is None:
                _build_parser().error(
                    f"--full-ft requires --{flag_name.replace('_', '-')} (the "
                    f"declared block cannot be built honestly without it)"
                )
    else:
        for flag_name in ("base_checkpoint", "hf_config"):
            if getattr(args, flag_name) is not None:
                _build_parser().error(
                    f"--lora takes no --{flag_name.replace('_', '-')} — see the "
                    f"module docstring: the base artifact is the WRONG "
                    f"denominator for an adapter save"
                )
    if args.entrypoint is not None and args.code_root is None:
        _build_parser().error(
            "--entrypoint without --code-root would record a containment answer "
            "against a capture that never happened"
        )
    try:
        return _emit(args)
    except EmitRefused as exc:
        print(f"manifest emission REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except EmitUnmeasured as exc:
        print(f"manifest emitter could not establish its inputs: {exc}", file=sys.stderr)
        return EXIT_UNMEASURED
    except Exception:
        # Same rule the probe holds itself to: a bug in the emission path is
        # not a verdict and not a refusal — it is a measurement failure, with
        # the traceback in view, never exit 1 wearing a policy's clothes.
        traceback.print_exc()
        print(
            "manifest emitter FAILED with an internal error above (a tool bug "
            "is not a run verdict)",
            file=sys.stderr,
        )
        return EXIT_UNMEASURED


if __name__ == "__main__":
    sys.exit(main())
