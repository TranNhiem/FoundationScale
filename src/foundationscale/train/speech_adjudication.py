"""Speech adjudication at SAVE: raw-byte digests plus the speech gates under one exit code.

A speech checkpoint is honest exactly when two independent claims hold at SAVE:
the artifact covered every audio row the run declared, and the trainable towers
moved *precisely* as the run declared they would. The second claim needs a signal
over the parameter bytes themselves — this module owns that signal
(:func:`tensor_digest`) and the adjudication that folds the speech gates'
results plus the reasons measurement was refused into one JSON-ready manifest and
one process exit code.

Nothing here re-implements a verdict. The gates of
:mod:`foundationscale.gates.speech_gates` are invoked through :meth:`Gate.run`, so
a gate whose ``check`` raises yields an ``ERROR`` result (blocking) instead of
tearing down the runner: a failed measurement must adjudicate, never crash the
adjudication.

torch and safetensors are imported inside the functions. This is the one layer
that must touch tensors (to hash them), while the gates themselves must run on a
bare stdlib interpreter (see ``speech_gates.py``); a module-level torch import
would also make ``import foundationscale.train.speech_adjudication`` drag a heavy
build into every consumer that merely wants a digest's string shape or the
manifest schema.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundationscale import EXIT_PASS, EXIT_RED, EXIT_UNMEASURED
from foundationscale.gates.core import GateResult
from foundationscale.gates.speech_gates import (
    AudioPlaceholderContext,
    AudioPlaceholderCoverageGate,
    AudioRowCoverageContext,
    AudioRowCoverageGate,
    TowerMovementContext,
    TowerMovementGate,
)


def tensor_digest(t: Any) -> str:
    """``"<dtype>:<sha256hex>"`` over a tensor's dtype and its raw bytes.

    The dtype rides in the digest on purpose. Base digests (read from the
    substitute base at substitution) and saved digests (read from the checkpoint
    at SAVE) are produced by different code paths and can land at different
    precisions — and that is exactly the comparison that decides "moved".
    Compared *without* the dtype the two sides can only be called equal or not,
    and every cross-dtype pair hashes different bytes and so would read as
    movement that training never produced. Tagged with the dtype (normalized:
    ``"torch.bfloat16" -> "bfloat16"``) the pair is recognizably a dtype change,
    and :func:`dtype_mismatches` refuses to price it as movement in either
    direction (the run adjudicates ``NOT_ESTABLISHED``, never "moved").

    The payload is the tensor's raw bytes, reinterpreted byte-for-byte
    (``reshape(-1).view(torch.uint8)``) rather than converted through numpy
    values — bfloat16 has no numpy value conversion at all, and the bytes are
    literally what the checkpoint carries. Consequences worth relying on: a
    one-bit edit anywhere in the tensor flips the digest, and two
    identically-stored copies (memory vs. a safetensors shard) hash identically.
    """
    import torch  # local: only this path needs the tensor/byte bridge

    raw: Any = t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return f"{_dtype_name(t)}:{hashlib.sha256(raw).hexdigest()}"


def _dtype_name(t: Any) -> str:
    """Bare dtype name for the digest tag.

    ``str(t.dtype)`` spells ``"torch.bfloat16"``; the tag drops the ``torch.``
    prefix so the digest reads as a dtype rather than as a module attribute path.
    """
    name = str(t.dtype)
    if name.startswith("torch."):
        return name[len("torch.") :]
    return name


def _digest_dtype(digest: str) -> str:
    """The dtype tag of a ``"<dtype>:<sha256hex>"`` digest.

    The payload is hex and so cannot contain a ``:`` — splitting on the first
    colon is therefore unambiguous no matter how the digest was built.
    """
    return digest.split(":", 1)[0]


_PEFT_ROOT = "base_model.model."


def canonical_param_name(name: str) -> str | None:
    """The base-model name for a parameter, through peft's wrapping; None for a trainable copy.

    Measured on peft 0.18.1 with ``modules_to_save``: the wrapped model names the
    frozen original ``base_model.model.<name with .original_module inserted>`` and the
    trainable copy ``...modules_to_save.default...``, and ``save_pretrained`` writes the
    copy as plain ``base_model.model.<name>``. Mapping all of them back to ``<name>``
    is what lets an adapter run's save be compared with the model as loaded. The copy
    is skipped at capture time (it is byte-identical to the original until training
    starts), so each base name is digested exactly once.
    """
    if ".modules_to_save." in name:
        return None
    if name.startswith(_PEFT_ROOT):
        name = name[len(_PEFT_ROOT) :]
    return name.replace(".original_module.", ".")


def _under_prefixes(name: str, prefixes: Sequence[str]) -> bool:
    """True when ``name`` sits under some prefix as a whole dotted segment.

    The name must be the prefix itself or extend it with a ``.``; a substring
    match would claim ``audio.tower_proj.w`` for the prefix ``audio.tower`` and
    grade one tower against another's bytes.
    """
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def digests_from_named_tensors(
    named: Iterable[tuple[str, Any]],
    prefixes: Sequence[str],
) -> dict[str, str]:
    """Digest the ``(name, tensor)`` pairs whose name lives under any ``prefixes`` entry.

    Scoping is a whole dotted segment (see :func:`_under_prefixes`): a name is
    claimed here iff the ``TowerMovementGate`` could carry it under that prefix,
    plus the costless ``name == prefix`` case (a flat parameter named exactly
    after its tower). Dict keyed by name — that key is the only join key
    ``TowerMovementGate`` has between the base map and the saved map, so the two
    sides must name parameters identically.
    """
    out: dict[str, str] = {}
    for raw, value in named:
        name = canonical_param_name(raw)
        if name is not None and _under_prefixes(name, prefixes):
            out[name] = tensor_digest(value)
    return out


def digests_from_safetensors_dir(
    directory: Path,
    prefixes: Sequence[str],
) -> dict[str, str]:
    """Digest every matching tensor in a flat directory of ``*.safetensors`` shards.

    Only the shards sitting directly in ``directory`` are read (not recursive: a
    run's checkpoint is one flat layout, and descending into nested dirs would
    silently widen the audit and re-hash leftovers of earlier saves). Each shard
    is opened with ``safetensors.safe_open(path, framework="pt")`` so the bytes
    hashed are exactly the bytes on disk — the same channel :func:`tensor_digest`
    hashes from memory, which is what makes the base/saved comparison fair.

    A key found in two shards raises :class:`ValueError`: two files claim one
    name and its digest is then ambiguous, which is worse than absent — an
    ambiguous digest still *looks* like a measurement and would be compared as
    one. The check covers every key in the shards and not only the reported
    prefixes, because a shard layout that already disagrees with itself cannot be
    graded honestly for half its names. A directory with no shard file simply
    yields the empty map (nothing saved is a coverage fact, not an ambiguity), but
    a path that is not a directory raises — silently returning the empty map there
    would transmute a lost checkpoint into "nothing to compare".
    """
    import safetensors  # local: only this path needs to read a shard

    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory of safetensors shards")

    out: dict[str, str] = {}
    first_seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safetensors.safe_open(str(path), framework="pt") as shard:
            for key in sorted(shard.keys()):
                # Keyed on the CANONICAL name, so a full save and an adapter save
                # both carrying one parameter (under two spellings) is caught as
                # the ambiguity it is, not silently resolved by file order.
                name = canonical_param_name(key)
                if name is None:
                    continue
                previous = first_seen.get(name)
                if previous is not None:
                    raise ValueError(
                        f"ambiguous shard layout: {name!r} is present in both "
                        f"{previous!r} and {path.name!r} — a key living in two "
                        "shards has no single digest"
                    )
                first_seen[name] = path.name
                if _under_prefixes(name, prefixes):
                    out[name] = tensor_digest(shard.get_tensor(key))
    return out


def dtype_mismatches(base: Mapping[str, str], saved: Mapping[str, str]) -> list[str]:
    """Names present in both maps whose digest dtype tags disagree.

    This is the referee for :func:`tensor_digest`'s dtype tag. A name listed here
    was read at one precision in the base map and another in the saved map, so
    its byte payloads describe different objects and can neither confirm nor
    refute movement — the caller must refuse the measurement ("the count and the
    first names" is all an adjudication can honestly report). Sorted, so "the
    first names" is a stable claim.

    Names present in only one map are deliberately *not* listed: their problem is
    coverage, which ``TowerMovementGate`` already adjudicates as UNDERCOVERED /
    VACUOUS — reporting them here would label a missing measurement as a dtype
    disagreement and send the caller looking in the wrong layer.
    """
    bad = [
        name
        for name in base
        if name in saved and _digest_dtype(base[name]) != _digest_dtype(saved[name])
    ]
    return sorted(bad)


@dataclass(frozen=True)
class SpeechAdjudication:
    """The adjudicated SAVE verdict over one speech checkpoint.

    ``results`` holds the gates that actually ran, in adjudication order;
    ``unmeasured`` records why the gates that did NOT run are absent (one
    human-readable reason per refused measurement). A refused measurement is
    deliberately not folded into ``results`` as a soft row: this type is named for
    carrying those refusals out loud, because "not run" must never render as
    "passed".

    ``exit_code`` ranks the three states the way a reviewer would: RED outranks
    UNMEASURED outranks PASS. A blocking result is a settled refutation and wins
    over a *refused* measurement (diagnose the refutation first); if a refusal
    outranked a fail, a badly measured run could smear a proven defect into a
    polite 95 where 5 is the truth.
    """

    results: tuple[GateResult, ...]
    unmeasured: tuple[str, ...] = ()
    # NOT_APPLICABLE abstentions: checks that cannot apply to this run by
    # construction (they leave the denominator and do not move the exit code),
    # recorded so the reader sees what was not checked and why.
    notes: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        """True when at least one executed gate blocks the save."""
        return any(r.blocking for r in self.results)

    @property
    def exit_code(self) -> int:
        """``EXIT_RED`` (5) when something blocks, ``EXIT_UNMEASURED`` (95) when
        measurement was refused and nothing blocked, else ``EXIT_PASS`` (0).
        """
        if self.blocking:
            return EXIT_RED
        if self.unmeasured:
            return EXIT_UNMEASURED
        return EXIT_PASS

    def as_manifest(self) -> dict[str, Any]:
        """JSON-ready manifest: ``{"gates": [...], "unmeasured": [...]}``.

        One row per executed gate carrying ``gate_id``, ``verdict``, the coverage
        triple (``checked`` / ``expected`` / ``unit`` - the denominator goes to
        JSON too: a verdict over "3 tensors" is unqualified without "out of how
        many") and ``detail``. ``verdict`` is serialized as its plain string value
        so an upstream runner can embed this with ``json.dumps`` and no custom
        encoder.
        """
        gates: list[dict[str, Any]] = [
            {
                "gate_id": r.gate_id,
                "verdict": str(r.verdict.value),
                "checked": r.coverage.checked,
                "expected": r.coverage.expected,
                "unit": r.coverage.unit,
                "detail": r.detail,
            }
            for r in self.results
        ]
        return {"gates": gates, "unmeasured": list(self.unmeasured), "notes": list(self.notes)}

    def render_lines(self) -> list[str]:
        """One line per gate result:
        ``"[<verdict>] <gate_id>: <checked>/<expected> <unit> -- <detail>"``.

        Result lines only. The ``unmeasured`` reasons carry no coverage
        denominator of their own and would have to be invented as a fake
        "0/0 units" to fit this format — fabricating the one measurement the
        adjudication admits it does not have. They live in :meth:`as_manifest`
        (key ``"unmeasured"``) and force the exit code to ``EXIT_UNMEASURED``;
        a renderer that prints these lines must print that list too.
        """
        lines: list[str] = []
        for r in self.results:
            lines.append(
                f"[{str(r.verdict.value)}] {r.gate_id}: "
                f"{r.coverage.checked}/{r.coverage.expected} {r.coverage.unit} "
                f"-- {r.detail}"
            )
        # An abstention nobody prints is a gap in the record: each unmeasured
        # reason and each NOT_APPLICABLE note gets its own line.
        lines.extend(f"[UNMEASURED] speech: {reason}" for reason in self.unmeasured)
        lines.extend(f"[n/a] speech: {note}" for note in self.notes)
        return lines


def adjudicate_speech(
    *,
    coverage_manifest: Mapping[str, Any],
    base_digests: Mapping[str, str] | None,
    saved_digests: Mapping[str, str] | None,
    towers: Sequence[tuple[str, bool]],
    adapter: str | None,
    placeholder_applicable: bool = True,
) -> SpeechAdjudication:
    """Adjudicate one speech checkpoint at SAVE: verdicts plus why anything is absent.

    ``placeholder_applicable=False`` for the seq2seq and ctc kinds: their batches carry no
    audio placeholder tokens (the audio enters as encoder features), so the placeholder
    gate has nothing to compare and is recorded as a NOT_APPLICABLE note instead of a
    denominator mismatch.

    Result order:

    1. :class:`AudioRowCoverageGate` and :class:`AudioPlaceholderCoverageGate` run
       always. Coverage is the denominator of every other claim, so it is measured
       even when tower movement is refused — an unmeasured gate is never a passed
       gate. Both are invoked through ``Gate.run(ctx)`` (not ``check``) so a gate
       that raises stays an ``ERROR`` result (blocking) instead of aborting the
       whole adjudication.

    2. Tower movement runs per declared ``(tower_prefix, exercised)`` tower, and
       only where the measurement is comparable. Three states refuse it and
       record a reason instead:

       * ``adapter is not None``: the saved checkpoint carries adapter weights, so
         base-vs-saved tower digests are not comparable and movement is
         ``NOT_ESTABLISHED``;
       * ``base_digests`` or ``saved_digests`` is ``None`` (e.g. a sharded save
         layout, or a base checkpoint that could not be resolved) — the reason
         names which side is missing;
       * :func:`dtype_mismatches` is non-empty (a cross-dtype pair can only
         "prove" movement that training never produced) — the reason names the
         count and the first names.

       Refusing instead of running is the point: a comparison over bytes that
       measure the wrong object adjudicates nothing, and the framework prices a
       declared ``NOT_ESTABLISHED`` far below a silent PASS.

    ``coverage_manifest`` must carry its own compact census: ``rows_expected``,
    ``rows_checked``, ``rows_refused``, ``refused`` (defaults to the empty map),
    ``placeholder_rows_verified`` and ``placeholder_rows_unmeasured``. A missing
    key raises rather than defaulting to zero: a zeroed census would transmute a
    missing measurement into VACUOUS coverage, which is a verdict with teeth —
    precisely the wrong one to read for an absent manifest field.
    """
    results: list[GateResult] = []
    unmeasured: list[str] = []

    rows_checked = int(coverage_manifest["rows_checked"])
    refused_counts: Mapping[str, int] = dict(coverage_manifest.get("refused") or {})

    # Coverage first and unconditional: whichever way tower movement is handled
    # below, the row and placeholder censuses are the denominators everything
    # else is reported over.
    results.append(
        AudioRowCoverageGate().run(
            AudioRowCoverageContext(
                rows_expected=int(coverage_manifest["rows_expected"]),
                rows_checked=rows_checked,
                rows_refused=int(coverage_manifest["rows_refused"]),
                refused=refused_counts,
            )
        )
    )
    notes: list[str] = []
    if placeholder_applicable:
        results.append(
            AudioPlaceholderCoverageGate().run(
                AudioPlaceholderContext(
                    rows_checked=rows_checked,
                    placeholder_rows_verified=int(coverage_manifest["placeholder_rows_verified"]),
                    placeholder_rows_unmeasured=int(
                        coverage_manifest["placeholder_rows_unmeasured"]
                    ),
                )
            )
        )
    else:
        notes.append(
            "placeholder coverage is NOT_APPLICABLE for this model kind: the audio enters "
            "as encoder features, so the batch carries no audio placeholder tokens"
        )

    if adapter is not None:
        # An adapter run saves only what trained: the adapter and the
        # modules_to_save copies of the exercised towers. peft freezes every other
        # parameter by construction, so the dormant-tower control cannot fail and
        # the frozen towers are not in the save to compare -- NOT_APPLICABLE, not
        # unmeasured. The exercised towers are still judged.
        dormant = [prefix for prefix, exercised in towers if not exercised]
        towers = [(prefix, exercised) for prefix, exercised in towers if exercised]
        if dormant:
            notes.append(
                f"adapter run: dormant-tower control for {', '.join(dormant)} is "
                "NOT_APPLICABLE -- peft freezes every non-target parameter by "
                "construction, and the adapter checkpoint does not carry frozen towers"
            )
    if base_digests is None or saved_digests is None:
        # Names the absent side so the operator knows which half of the pair to
        # rebuild — "digests unavailable" alone does not say what to fix.
        missing = [
            label
            for label, present in (
                ("base_digests", base_digests),
                ("saved_digests", saved_digests),
            )
            if present is None
        ]
        unmeasured.append(
            f"tower movement not measured: {', '.join(missing)} unavailable "
            "(e.g. sharded save layout) — base-vs-saved tower digests cannot be "
            "compared (tower movement is NOT_ESTABLISHED)"
        )
    else:
        inconsistencies = dtype_mismatches(base_digests, saved_digests)
        if inconsistencies:
            # Counts and names the first offenders so the run can be re-read at
            # one precision; the measurement is refused, not computed over bytes
            # that disagree about their own dtype.
            shown = ", ".join(inconsistencies[:3])
            tail = ", ..." if len(inconsistencies) > 3 else ""
            unmeasured.append(
                f"tower movement not measured: {len(inconsistencies)} name(s) "
                f"read at different dtypes ({shown}{tail}) — a digest is "
                "dtype-qualified, so a cross-dtype pair can only 'prove' movement "
                "that training never produced (tower movement is NOT_ESTABLISHED)"
            )
        else:
            for prefix, exercised in towers:
                results.append(
                    TowerMovementGate().run(
                        TowerMovementContext(
                            tower_prefix=prefix,
                            base_digests=base_digests,
                            saved_digests=saved_digests,
                            exercised=exercised,
                        )
                    )
                )

    return SpeechAdjudication(
        results=tuple(results), unmeasured=tuple(unmeasured), notes=tuple(notes)
    )


SUPERSEDE_NOTE = (
    "speech plane: the speech.* gates reported SKIP in the save sweep "
    "(no context there) are adjudicated here with their contexts"
)


def capture_speech_inputs(
    family: Any, state_items: Callable[[], Iterable[tuple[str, Any]]]
) -> tuple[list[tuple[str, bool]], dict[str, str] | None]:
    """The towers to judge and their as-loaded digests, taken BEFORE a step is run.

    Every modality tower the family declares is judged: the audio ones as exercised,
    the rest as the dormant negative control. ``state_items`` is called only when a
    digest is needed (it is ``model.state_dict().items`` in the plane, and walking it
    is not free). Adapter runs are digested too: names are canonicalised through
    peft's wrapping (:func:`canonical_param_name`), so they compare with the save.
    """
    if family is None:
        return [], None
    towers = [(prefix, modality == "audio") for prefix, modality in family.towers if modality]
    if not towers:
        return towers, None
    return towers, digests_from_named_tensors(state_items(), [prefix for prefix, _ in towers])


def run_final_speech_adjudication(
    *,
    final_dir: Path,
    has_safetensors: bool,
    coverage_manifest: Mapping[str, Any],
    base_digests: Mapping[str, str] | None,
    towers: Sequence[tuple[str, bool]],
    adapter: str | None,
    placeholder_applicable: bool = True,
) -> SpeechAdjudication:
    """Read the final save's tower digests (safetensors layout only) and adjudicate.

    A sharded (DCP) save or an adapter run leaves the saved side unread, which
    :func:`adjudicate_speech` reports as UNMEASURED rather than guessing.
    """
    saved = (
        digests_from_safetensors_dir(final_dir, [prefix for prefix, _ in towers])
        if has_safetensors and towers
        else None
    )
    return adjudicate_speech(
        coverage_manifest=coverage_manifest,
        base_digests=base_digests,
        saved_digests=saved,
        towers=towers,
        adapter=adapter,
        placeholder_applicable=placeholder_applicable,
    )


def fold_speech_verdict(rc: int, done: str, speech_exit: int) -> tuple[int, str]:
    """Fold the speech verdict into the run's: RED outranks PASS; UNMEASURED moves only PASS.

    A blocking speech gate is a measurement and outranks a clean save sweep; an
    abstention is not a measurement, so it can stop a PASS from being claimed but
    never softens a RED that was measured elsewhere.
    """
    if speech_exit == EXIT_RED and rc != EXIT_RED:
        return EXIT_RED, "RED: blocking speech-plane gate verdict"
    if speech_exit == EXIT_UNMEASURED and rc == EXIT_PASS:
        return EXIT_UNMEASURED, (
            "UNMEASURED: the run trained and saved, but a speech-plane gate "
            "could not be run (see the speech lines above)"
        )
    return rc, done
