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
import sys
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


# The counts reduce_coverage_across_ranks sums, in the order its packed row
# carries them. The order IS the contract: a rank that packs another order sums
# its rows into another rank's seconds and publishes refusals no row produced.
_COVERAGE_COUNT_KEYS = (
    "rows_expected",
    "rows_checked",
    "rows_refused",
    "seconds_total",
    "placeholder_rows_verified",
    "placeholder_rows_unmeasured",
)

# The compact census adjudicate_speech's docstring names as the manifest's
# required keys: all of them are refused (ValueError) when absent, because a
# count that is absent cannot be summed and zero-filling it would publish a
# census no row produced. seconds_total is deliberately NOT among them -- it
# rides AudioCoverage.as_manifest's schema and is summed when present, but a
# manifest that declares no seconds stays without them rather than being handed
# an invented 0.0.
_REQUIRED_COVERAGE_KEYS = (
    "rows_expected",
    "rows_checked",
    "rows_refused",
    "refused",
    "placeholder_rows_verified",
    "placeholder_rows_unmeasured",
)


def _coverage_counts_vector(coverage_manifest: Mapping[str, Any]) -> list[float]:
    """One packed float64 row: the six counts, then one slot per refusal reason.

    The refusal buckets ride as one slot per ``AUDIO_LOAD_REASONS`` entry --
    audio.py's canonical vocabulary, exported there (it is the same tuple
    ``SharedAudioCoverage`` keys its shared refusal Array by) -- and NOT as "the
    reasons this rank saw". Per-rank buckets would give ranks of different
    widths (gloo would fail the shape and nccl would sum whatever it is handed)
    and, worse, a rank whose ``unreadable`` lands in another rank's ``too_long``
    slot manufactures a refusal reason no row ever produced.

    A reason outside the vocabulary therefore raises: it has no bucket to be
    merged into, and dropping it would report a run that lost no rows -- the
    same refusal ``SharedAudioCoverage.record_refused`` makes, and for the same
    reason. ``seconds_total`` packs as 0.0 when the manifest does not carry it
    (a wire slot, never a claim on the manifest): see
    :func:`reduce_coverage_across_ranks`, where a census without seconds stays
    without them.
    """
    from foundationscale.train.audio import AUDIO_LOAD_REASONS  # noqa: PLC0415

    vector = [
        # Only seconds_total can be missing here -- reduce_coverage_across_ranks
        # already refused an absent count key.
        float(coverage_manifest[key]) if key in coverage_manifest else 0.0
        for key in _COVERAGE_COUNT_KEYS
    ]
    index_of = {reason: index for index, reason in enumerate(AUDIO_LOAD_REASONS)}
    buckets = [0.0] * len(AUDIO_LOAD_REASONS)
    for reason, count in dict(coverage_manifest["refused"]).items():
        index = index_of.get(reason)
        if index is None:
            raise ValueError(
                f"refusal reason {reason!r} is not one of {list(AUDIO_LOAD_REASONS)}: "
                "the ranks merge refusals one bucket per vocabulary token, so a reason "
                "with no bucket would lose its rows from the manifest (the same "
                "refusal SharedAudioCoverage.record_refused makes)"
            )
        buckets[index] += float(count)
    return vector + buckets


def _all_reduce_coverage_vector(vector: list[float]) -> tuple[list[float], int]:
    """``vector`` summed over the default process group; ``(vector, 1)`` when there is no group.

    torch is read through ``sys.modules`` and imported nowhere here -- the way
    :mod:`foundationscale.train.loop`'s rank helper finds the process's rank --
    because this module must import under a bare interpreter (see the module
    docstring) and a run that is not distributed must be able to pass a manifest
    through without a tensor framework arriving to copy six integers. "torch is
    loaded but its group is not initialized" is the same case: there are no peers
    to sum with.

    The row goes through ONE float64 tensor and ONE ``all_reduce(SUM)``. float64
    so the sums stay exact up to 2**53 rows (float32 rounds at 2**24 and would
    publish a coverage claim nobody measured). The device follows the backend:
    the current CUDA device for NCCL (NCCL has no CPU tensors, and a CPU tensor
    there is an error rather than a slow path) and CPU otherwise (gloo sums CPU
    tensors and refuses a CUDA one out loud).
    """
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return list(vector), 1
    dist = getattr(torch_module, "distributed", None)
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return list(vector), 1
    ranks = int(dist.get_world_size())
    if ranks < 2:
        return list(vector), 1
    # torch is loaded by now (its distributed submodule just answered). Backend
    # spellings differ between builds -- "nccl" in some, "Backend.NCCL" from the
    # Backend object's __repr__ in others -- so the stable part is the suffix.
    device = (
        torch_module.device("cuda", torch_module.cuda.current_device())
        if str(dist.get_backend()).lower().endswith("nccl")
        else torch_module.device("cpu")
    )
    row = torch_module.tensor(vector, dtype=torch_module.float64, device=device)
    dist.all_reduce(row, op=dist.ReduceOp.SUM)
    return [float(value) for value in row.to("cpu").tolist()], ranks


def reduce_coverage_across_ranks(
    coverage_manifest: Mapping[str, Any],
    *,
    all_reduce_sum: Callable[[list[float]], list[float]] | None = None,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Sum one rank's coverage manifest over the ranks of the run. A COLLECTIVE.

    EVERY RANK MUST CALL THIS, in the same order. :func:`coverage_after_train` is
    the caller: ``train()`` reaches it on every rank right after ``Trainer.train()``
    returns. :func:`finish_speech_run` must NOT call it -- only the writing rank gets
    there, and a collective on one rank deadlocks (measured). The default path is ONE
    ``all_reduce(SUM)`` over one packed float64 row: it blocks until every rank
    of the group arrives and pairs the entries POSITIONALLY, so a rank that
    skips the call hangs the run (NCCL would rather time out in flames than let
    it continue with half a census), and a rank that packs a different width or
    order sums one rank's rows into another rank's seconds.

    MEASURED defect (GB200, 2026-10-09): the speech gates audited only the LOCAL
    rank's rows. ``SharedAudioCoverage`` shares counters across one rank's
    DataLoader WORKERS and never across ranks, so under ``torchrun --nproc-per-node
    2`` every rank handed :func:`finish_speech_run` its own
    ``data_collator.coverage.as_manifest()`` and that was adjudicated as the
    corpus: a 150-step run at per-device batch 8 trained 2,400 rows on two GPUs
    while ``speech.audio_row_coverage`` claimed "1272/1272 audio rows" -- the
    1-GPU count (1,200 trained rows plus 72 prefetched) -- and rank 1's rows,
    refusals and placeholder checks were never adjudicated at all. COVERED over
    half the census is the most expensive lie this framework can publish, and the
    placeholder gate carried the same defect, repaired by the same call.

    The return is a NEW manifest (the input is never written to): ``rows_expected``,
    ``rows_checked``, ``rows_refused``, ``seconds_total``,
    ``placeholder_rows_verified``, ``placeholder_rows_unmeasured`` and every
    ``refused[reason]`` are the SUM over the ranks (buckets only where the sum is
    non-zero, as ``AudioCoverage.as_manifest`` reports them); ``"verdict"`` is
    RECOMPUTED from the summed counts with audio.py's own
    :func:`foundationscale.train.audio._coverage_verdict` (a verdict stamped
    before the ranks were summed is a verdict over one rank and must not
    survive); ``sampling_rate`` is kept as it was measured; a ``seconds_total``
    the manifest never carried stays absent -- sums invent no duration; and
    ``"ranks"`` is the world size the sums describe.

    "Required" keys are :func:`adjudicate_speech`'s compact census --
    ``rows_expected``, ``rows_checked``, ``rows_refused``, ``refused``,
    ``placeholder_rows_verified``, ``placeholder_rows_unmeasured`` -- and one
    missing raises :class:`ValueError` before anything is summed. ``seconds_total``
    is counted when it is present and is not one of them: it is no gate's
    denominator, so a compact census has none to give and none is fabricated for
    it (the collator's ``as_manifest`` always writes one, and that one is summed
    like every other count).

    ``all_reduce_sum``/``world_size`` stand in for the ``torch.distributed``
    handle -- this is how unit tests drive the sum: ``all_reduce_sum`` takes the
    local packed row and returns the summed one. The default path never imports
    torch (see :func:`_all_reduce_coverage_vector`), so a single-process run --
    or one where no group is initialized -- gets a copy of its manifest with
    ``"ranks": 1``, which is a scope statement and not a coverage claim made
    smaller.
    """
    from foundationscale.train.audio import AUDIO_LOAD_REASONS, _coverage_verdict  # noqa: PLC0415

    missing = [key for key in _REQUIRED_COVERAGE_KEYS if key not in coverage_manifest]
    if missing:
        raise ValueError(
            f"coverage manifest is missing required count key(s): {', '.join(missing)} -- "
            "a count that is absent cannot be summed across ranks, and zero-filling it "
            "would publish a census no row produced (the reduction refuses, it does not guess)"
        )
    packed = _coverage_counts_vector(coverage_manifest)
    if all_reduce_sum is None:
        summed, ranks = _all_reduce_coverage_vector(packed)
    else:
        if world_size is None:
            raise ValueError(
                "all_reduce_sum is given without world_size: a count summed without saying "
                "over how many ranks is not a coverage claim, and 'ranks' is how the reduced "
                "manifest states it"
            )
        summed = [float(value) for value in all_reduce_sum(packed)]
        ranks = int(world_size)
    if len(summed) != len(packed):
        raise ValueError(
            f"the reduction returned {len(summed)} values for a packed row of {len(packed)}: "
            "a row whose width changes mid-collective pairs one rank's rows with another "
            "rank's seconds"
        )
    if ranks < 2:
        # No peers to sum with: the manifest IS the census. Exactly a copy, so a
        # caller that keeps counting cannot rewrite what the adjudication read.
        copy = dict(coverage_manifest)
        copy["refused"] = dict(coverage_manifest["refused"])
        copy["ranks"] = 1
        return copy

    reduced = dict(coverage_manifest)
    for index, key in enumerate(_COVERAGE_COUNT_KEYS):
        if key == "seconds_total" and key not in coverage_manifest:
            continue  # measured nowhere -> reported nowhere (no invented 0.0)
        reduced[key] = float(summed[index]) if key == "seconds_total" else int(summed[index])
    merged: dict[str, int] = {}
    for reason, value in zip(AUDIO_LOAD_REASONS, summed[len(_COVERAGE_COUNT_KEYS) :], strict=True):
        count = int(value)
        if count:
            merged[reason] = count
    reduced["refused"] = merged
    reduced["verdict"] = _coverage_verdict(
        rows_checked=reduced["rows_checked"],
        rows_expected=reduced["rows_expected"],
        refused_total=reduced["rows_refused"],
    )
    reduced["ranks"] = ranks
    return reduced


SUPERSEDE_NOTE = (
    "speech plane: the speech.* gates reported SKIP in the save sweep "
    "(no context there) are adjudicated here with their contexts"
)


def capture_speech_inputs(
    family: Any,
    state_items: Callable[[], Iterable[tuple[str, Any]]],
    full_train: Sequence[str] | None = None,
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
    if full_train:
        # Adapter run: only the declared full-train modules are saved whole (peft
        # modules_to_save), so THEY are what must move; LoRA-wrapped layers keep their base
        # weights frozen and absent from the save. Non-audio towers stay the dormant control.
        towers = [(module, True) for module in full_train] + [t for t in towers if not t[1]]
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


def prepare_speech_run(
    model: Any, family: Any, full_train: Sequence[str] | None
) -> tuple[list[tuple[str, bool]], dict[str, str] | None, str | None]:
    """Before training: freeze BatchNorm statistics and capture the tower digests.

    Returns ``(towers, base_digests, batchnorm_announcement_or_None)`` so the train loop
    holds one call. The BatchNorm freeze is measured necessary on parakeet-ctc-1.1b (see
    ``speech_kinds.freeze_batchnorm_statistics``); digests come from parameters only.
    """
    from foundationscale.train.speech_kinds import freeze_batchnorm_statistics  # noqa: PLC0415

    frozen = freeze_batchnorm_statistics(model)
    line = (
        f"[   ok] speech.batchnorm: {frozen} BatchNorm layer(s) keep their stored running "
        "statistics during training (affine weights still train): small, zero-padded audio "
        "batches otherwise corrupt them (measured on parakeet-ctc)"
        if frozen
        else None
    )
    towers, base = capture_speech_inputs(family, lambda: model.named_parameters(), full_train)
    return towers, base, line


def coverage_after_train(collator: Any, *, audio_declared: bool) -> dict[str, Any] | None:
    """The run-wide audio census, summed over the ranks; ``None`` when no audio is declared.

    ``train()`` calls this on EVERY rank right after ``Trainer.train()`` returns -- the
    last point all ranks are guaranteed to reach. The final adjudication runs on the
    writing rank alone, so a census read there is that rank's rows reported as the run:
    measured on GB200 with 2 ranks, 1272 of 2544 loaded rows audited as "1272/1272".
    """
    if not audio_declared:
        return None
    return reduce_coverage_across_ranks(collator.coverage.as_manifest())


def finish_speech_run(
    *,
    rc: int,
    done: str,
    final_dir: Path,
    has_safetensors: bool,
    coverage_manifest: Mapping[str, Any] | None,
    base_digests: Mapping[str, str] | None,
    towers: Sequence[tuple[str, bool]],
    adapter: str | None,
    placeholder_applicable: bool,
) -> tuple[int, str, list[str], str]:
    """After the final save: adjudicate, fold into the run verdict, and render.

    ``coverage_manifest`` is THIS RANK's collator manifest and is reduced across
    the ranks before anything reads it
    (:func:`reduce_coverage_across_ranks` -- the collective lands here, where
    every rank arrives exactly once), so the gates adjudicate the corpus the run
    trained and not the half of it this rank happens to hold.

    Returns ``(rc, done, lines_to_print, manifest_json)``.
    """
    import json  # noqa: PLC0415

    # No collective here: this runs on the WRITING rank only (the other ranks abstain
    # before the final adjudication), so an all_reduce in this function deadlocks --
    # measured on GB200, rank 0 spinning at 100% while rank 1 had exited. The census
    # arrives already summed by coverage_after_train, which train() calls right after
    # Trainer.train() returns, while every rank is still present.
    if coverage_manifest is None:
        raise ValueError(
            "no run-wide audio coverage census: train() must sum it over the ranks "
            "(coverage_after_train) before the final adjudication"
        )
    speech = run_final_speech_adjudication(
        final_dir=final_dir,
        has_safetensors=has_safetensors,
        coverage_manifest=coverage_manifest,
        base_digests=base_digests,
        towers=towers,
        adapter=adapter,
        placeholder_applicable=placeholder_applicable,
    )
    rc, done = fold_speech_verdict(rc, done, speech.exit_code)
    lines = [SUPERSEDE_NOTE, *speech.render_lines()]
    return rc, done, lines, json.dumps(speech.as_manifest(), sort_keys=True)
