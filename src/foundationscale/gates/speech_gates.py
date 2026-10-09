"""Speech gates: audio-row coverage, placeholder coverage, tower movement, runaways, loops.

Five gates auditing a speech training job's checkpoint at :attr:`Lifecycle.SAVE`:

* :class:`AudioRowCoverageGate` — every declared audio row was either loaded into
  the artifact or refused-and-counted (and any refusal leaking into the save under
  strict handling is a defect that blocks).
* :class:`AudioPlaceholderCoverageGate` — placeholder-audio rows have had their
  placeholder markers verified against a count where the processor offers one,
  with a declared ``NOT_ESTABLISHED`` abstention where it does not (never silently
  PASSing over unmeasured rows).
* :class:`TowerMovementGate` — the speech tower's trainable parameters (everything
  under ``tower_prefix`` that is not a runtime buffer) actually moved off their base
  digests iff the run declared them exercised; the dormant negative control
  (``exercised=False``) requires the inverse.
* :class:`RunawayHypothesisGate` — the eval rows the run measured at save stay
  within ``2 * base_runaway + ceil(0.005 * rows_checked)`` runaway hypotheses: a
  decoder that stopped listening (repetition loops, hallucinated domain text) is
  invisible to every artifact gate, and the base model's own rate is the yardstick.
* :class:`RepetitionLoopGate` — the loop words the run measured at save stay within
  ``2 * base_loop_words + ceil(0.005 * reference_words)``: ``is_runaway`` judges
  WHOLE rows (past twice the row's reference length) and a repetition loop inside
  one 40 s chunk of an hour-long call adds only 4-12% of that call's words (the
  measured 1,428-loop-word Canary-1B Earnings collapse passed it), so loops are
  counted locally, in words, against the base model's own loop count.

Registration at import time is doctrine (mirroring ``checkpoint_gates.py``): the
``@register`` class decorator adds each gate to the process-wide :data:`REGISTRY`,
so ``foundationscale-controls`` and other consumers see these gates the moment the
module is imported. No torch / numpy is imported here — gates must run on a bare
stdlib interpreter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from typing import Any, ClassVar

from .core import (
    AbstentionKind,
    Control,
    ControlKind,
    Coverage,
    Gate,
    GateResult,
    Lifecycle,
    register,
)


@dataclass(frozen=True)
class AudioRowCoverageContext:
    """Census of the audio rows the save was expected to contain.

    The denominator ``rows_expected`` must come from outside the artifact (the
    training manifest or dataloader config — a corrupt save could claim any
    number of rows, which is exactly the audited disaster the framework names).
    ``rows_checked`` counts rows actually loaded from the artifact. ``rows_refused``
    counts rows refused during load; ``refused`` maps each refusal reason to its
    count. In strict handling the loader refuses at load time and the save should
    never contain a refused row, so a non-zero ``rows_refused`` reaching this gate
    is a smoking gun that tolerant handling leaked into the strict path.
    """

    rows_expected: int
    rows_checked: int
    rows_refused: int
    refused: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioPlaceholderContext:
    """Placeholder-audio verification census across the rows the save covered.

    ``rows_checked`` is the row count for this save; ``placeholder_rows_verified``
    is how many rows' placeholder markers were actually checked against a
    placeholder count offered by the processor (or trivially carry nothing to
    check); ``placeholder_rows_unmeasured`` is how many rows the processor offered
    no count for and hence could not be verified against.

    The accounting invariant ``verified + unmeasured == rows_checked`` is the
    gate's denominator hygiene: a caller whose counts do not add up to the row
    census is a miscounted denominator and must block before any PASS can be
    claimed.
    """

    rows_checked: int
    placeholder_rows_verified: int
    placeholder_rows_unmeasured: int


@dataclass(frozen=True)
class TowerMovementContext:
    """Digest census of the tower's trainable parameters across a save.

    ``base_digests`` is the pre-training (substitute base) state keyed by
    fully-qualified parameter name; ``saved_digests`` is the state carried into
    the checkpoint. ``tower_prefix`` scopes the measurement to one tower (e.g.
    ``"speech.tower"``). Parameters whose last dotted component is one of
    ``buffer_suffixes`` (``input_min`` / ``input_max`` / ``output_min`` /
    ``output_max`` by default) are quantization-stat runtime buffers and are
    excluded from the movement measurement: counting them as movement would
    fabricate the very signal the gate exists to check.

    ``exercised`` is the run's declaration: ``True`` means the tower was meant to
    be updated during training (so at least one trainable parameter must have
    moved by save); ``False`` is the dormant negative control (nothing may have
    moved).
    """

    tower_prefix: str
    base_digests: Mapping[str, str]
    saved_digests: Mapping[str, str]
    buffer_suffixes: tuple[str, ...] = (
        "input_min",
        "input_max",
        "output_min",
        "output_max",
    )
    exercised: bool = True


@register
class AudioRowCoverageGate(Gate):
    """Every expected audio row was either loaded or refused-and-counted.

    Defect class: a save that silently drops audio rows, or a strict-mode loader
    that quietly tolerates row-level refusals that should have blocked the job at
    load. Coverage counts every row the artifact either accepted or refused; the
    denominator is the run's external ``rows_expected`` so a save cannot inflate
    its own row count to make the check pass.

    ``rows_checked == 0`` yields ``VACUOUS`` (blocking) — no attestation is
    possible over zero loaded rows. A non-zero ``rows_refused`` reaching the
    gate under strict handling is a leak that must ``FAIL`` with the reasons
    named; strict handling refuses at load, so a refusal that arrives here means
    tolerant handling leaked in.

    Because ``docs/CUSTOM_GATES.md`` mandates the ``Coverage.unit`` string be
    ``"plural, lowercase"``, this module uses ``"audio rows"`` / ``"tower
    parameters"`` (plural) rather than the singular shapes the prose spec wrote
    informally.
    """

    id: ClassVar[str] = "speech.audio_row_coverage"
    description: ClassVar[str] = (
        "Every audio row the run declared is accounted for at save: either "
        "loaded into the artifact or refused-and-counted (and any refusal "
        "under strict handling is a leak that must block)"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.SAVE,)
    context_type: ClassVar[type | None] = AudioRowCoverageContext

    def check(self, ctx: Any) -> GateResult:
        c = ctx
        rows_expected = c.rows_expected
        rows_checked = c.rows_checked
        rows_refused = c.rows_refused
        refused = dict(c.refused) if c.refused else {}

        # The per-spec Coverage formula (checked = loaded + refused, expected =
        # manifest). This exact formula serves the two branches that actually
        # measure data (refusals-found FAIL and the general pass/under/over
        # chain). The rows_checked==0 branch deliberately overrides it to
        # Coverage.none: forcing checked=0 is the only way to make the
        # framework's ok() downgrade yield the mandated VACUOUS verdict even
        # when rows_refused>0 (since the formula would produce checked>0 in
        # that case and ok() would downgrade to UNDERCOVERED / OVERCOVERED /
        # PASS instead of VACUOUS). Refusals still surface in evidence.
        coverage = Coverage(
            checked=rows_checked + rows_refused,
            unit="audio rows",
            expected=rows_expected,
        )

        if rows_checked == 0:
            # "rows_checked == 0 -> VACUOUS (blocks)" is an absolute rule of
            # this gate: zero loaded rows means no attestation is possible no
            # matter how many refusals preceded them. The spec slot for this
            # case is first; the refusal-leak FAIL branch below casts a wider
            # (rows_checked > 0, rows_refused > 0) net because rows_checked == 0
            # already returned. Returning through self.ok over zero coverage
            # routes this to Verdict.VACUOUS via the framework's downgrade
            # chain — the only sanctioned way to produce VACUOUS without
            # hand-assembling GateResult.
            return self.ok(
                f"no audio row was loaded (0 of {rows_expected} expected; "
                f"{rows_refused} refused) — nothing to attest about",
                Coverage.none("audio rows"),
                evidence={
                    "rows_expected": rows_expected,
                    "rows_refused": rows_refused,
                    "refused": refused,
                },
            )

        if rows_refused > 0:
            # Strict handling refuses at load; a refusal that reaches this gate
            # means the loader tolerated instead of blocking. The leak is named
            # and must FAIL with the reasons surfaced so the caller can see
            # which row-level failure slipped through.
            return self.fail(
                f"{rows_refused} of {rows_expected} audio rows were refused "
                f"under strict-mode handling ({refused}) — strict mode refuses "
                "at load, so a refused count reaching the save means tolerant "
                "handling leaked in",
                coverage,
                evidence={
                    "rows_refused": rows_refused,
                    "refused": refused,
                    "rows_checked": rows_checked,
                },
            )

        # rows_refused == 0 and rows_checked > 0 here. Coverage reduces to
        # (rows_checked, ..., rows_expected) since rows_refused == 0; the
        # framework's self.ok() downgrade resolves shortfall (UNDERCOVERED) or
        # overage (OVERCOVERED) and blocks in both cases. The clean pass state
        # is rows_checked == rows_expected with zero refusals.
        return self.ok(
            f"{rows_checked} of {rows_expected} expected audio rows loaded (no refusals)",
            coverage,
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                name="no-rows-loaded",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: AudioRowCoverageContext(
                    rows_expected=5,
                    rows_checked=0,
                    rows_refused=0,
                    refused={},
                ),
                note="0 rows loaded against 5 expected: must block as VACUOUS "
                "with zero coverage (no attestation over zero loaded rows)",
            ),
            Control(
                name="strict-refusal-leaked",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: AudioRowCoverageContext(
                    rows_expected=5,
                    rows_checked=3,
                    rows_refused=2,
                    refused={"decoder_failure": 2},
                ),
                note="2 rows refused under strict handling that should have "
                "blocked at load: must FAIL and name 'decoder_failure' as the "
                "leaked tolerant-handling reason",
            ),
            Control(
                name="undercover-row-count",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: AudioRowCoverageContext(
                    rows_expected=5,
                    rows_checked=3,
                    rows_refused=0,
                    refused={},
                ),
                note="3 of 5 rows loaded without refusals: shortfall of 2 must "
                "block as UNDERCOVERED (framework downgrade over Coverage(3,5))",
            ),
            Control(
                name="all-rows-loaded",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: AudioRowCoverageContext(
                    rows_expected=5,
                    rows_checked=5,
                    rows_refused=0,
                    refused={},
                ),
                note="5 of 5 rows loaded and 0 refused: the clean pass state",
            ),
        ]


@register
class AudioPlaceholderCoverageGate(Gate):
    """Placeholder rows have had their placeholder markers verified or counted.

    Defect class: a save hiding placeholder audio behind fabricated "verified"
    claims it never actually measured, or a placeholder accounting that does not
    add up to the row census it claims to describe. The framework's doctrine
    requires any claim to carry its own coverage: if the processor offers no
    placeholder count to verify against (every row unmeasured), the gate must
    STAY SILENT via a declared ``AbstentionKind.NOT_ESTABLISHED`` skip — it is
    forbidden to PASS over rows it could not adjudicate. If the accounting
    itself does not add up (``verified + unmeasured != rows_checked``), the
    denominator is malformed and must FAIL before any ratio is computed.
    """

    id: ClassVar[str] = "speech.audio_placeholder_coverage"
    description: ClassVar[str] = (
        "Every audio row's placeholder marker is verified at save: either "
        "actually checked against a placeholder count (verified) or counted "
        "as unmeasured behind a declared NOT_ESTABLISHED abstention — never "
        "silently PASSing over rows the processor could not verify"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.SAVE,)
    context_type: ClassVar[type | None] = AudioPlaceholderContext

    def check(self, ctx: Any) -> GateResult:
        c = ctx
        rows_checked = c.rows_checked
        verified = c.placeholder_rows_verified
        unmeasured = c.placeholder_rows_unmeasured

        # Denominator hygiene first: if the placeholder accounting does not add
        # up to the row census, the Coverage denominator this gate would report
        # is itself false, and no subsequent ratio computed over the two could
        # be trusted. Route through fail() — always blocks and never softens.
        if verified + unmeasured != rows_checked:
            return self.fail(
                f"denominator mismatch: {verified} placeholder rows verified "
                f"+ {unmeasured} unmeasured = {verified + unmeasured} but "
                f"rows_checked declares {rows_checked} — the placeholder "
                "accounting does not add up to the row census",
                Coverage(
                    checked=verified,
                    unit="audio rows",
                    expected=rows_checked,
                ),
                evidence={
                    "rows_checked": rows_checked,
                    "placeholder_rows_verified": verified,
                    "placeholder_rows_unmeasured": unmeasured,
                },
            )

        if rows_checked == 0:
            # Nothing to verify over zero rows. The denominator check above
            # already confirmed 0 verified + 0 unmeasured == 0; returning
            # through self.ok over Coverage.none() routes to VACUOUS via the
            # framework's enforced downgrade.
            return self.ok(
                "no rows were sent through placeholder measurement for this "
                "save — nothing to attest about",
                Coverage.none("audio rows"),
            )

        if unmeasured == rows_checked:
            # rows_checked > 0 here (the ==0 case routed above to VACUOUS).
            # Every row is unmeasured and the processor offers no placeholder
            # count to verify against. Take the declared NOT_ESTABLISHED
            # abstention — never PASS over entirely unmeasured rows.
            return self.skip(
                "the processor offers no placeholder count to verify against: "
                f"all {rows_checked} rows are rows the gate cannot adjudicate "
                "— NOT_ESTABLISHED",
                kind=AbstentionKind.NOT_ESTABLISHED,
            )

        # Mixed state (some verified, some unmeasured) or fully verified: totals
        # already add up (checked above). Coverage is (verified, ..., rows_checked);
        # the framework's self.ok() downgrade handles shortfall as UNDERCOVERED
        # (verified < rows_checked blocks) and full coverage as PASS (verified ==
        # rows_checked).
        return self.ok(
            f"{verified} of {rows_checked} audio row placeholder markers "
            f"verified ({unmeasured} unmeasured)",
            Coverage(
                checked=verified,
                unit="audio rows",
                expected=rows_checked,
            ),
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                name="denominator-mismatch",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: AudioPlaceholderContext(
                    rows_checked=10,
                    placeholder_rows_verified=5,
                    placeholder_rows_unmeasured=2,
                ),
                note="5+2=7 but rows_checked declares 10: placeholder accounting "
                "does not add up to the row census; must FAIL and name the "
                "mismatch before any ratio is computed",
            ),
            Control(
                name="zero-rows-vacuous",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: AudioPlaceholderContext(
                    rows_checked=0,
                    placeholder_rows_verified=0,
                    placeholder_rows_unmeasured=0,
                ),
                note="0 rows and the addition holds trivially (0+0==0): must "
                "still block as VACUOUS — no attestation over zero rows",
            ),
            Control(
                name="partial-unmeasured-undercovered",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: AudioPlaceholderContext(
                    rows_checked=5,
                    placeholder_rows_verified=3,
                    placeholder_rows_unmeasured=2,
                ),
                note="3 of 5 rows verified, 2 unmeasured and totals add: must "
                "block as UNDERCOVERED — partial evidence is not full evidence "
                "when the framework does not declare a sample",
            ),
            Control(
                name="all-unmeasured-NOT_ESTABLISHED",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: AudioPlaceholderContext(
                    rows_checked=5,
                    placeholder_rows_verified=0,
                    placeholder_rows_unmeasured=5,
                ),
                note="0 of 5 rows verified because the processor offers no "
                "placeholder count to verify against: the gate must take its "
                "declared NOT_ESTABLISHED abstention and must not PASS over "
                "entirely unmeasured rows",
                expect_skip=(
                    "the processor offers no placeholder count to verify "
                    "against: every row is unmeasured and cannot be "
                    "adjudicated — NOT_ESTABLISHED is the correct abstention"
                ),
            ),
            Control(
                name="all-verified-PASS",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: AudioPlaceholderContext(
                    rows_checked=5,
                    placeholder_rows_verified=5,
                    placeholder_rows_unmeasured=0,
                ),
                note="5 of 5 placeholder markers verified, 0 unmeasured: the "
                "clean pass state (this is the MUST_PASS fixture set's required "
                "affirmative healthy-input pass)",
            ),
        ]


@register
class TowerMovementGate(Gate):
    """The tower's trainable parameters moved off their base digests as declared.

    Defect class: a save that writes the speech tower byte-for-byte unchanged
    into the checkpoint while the run declared the tower exercised for training
    (the update silently did not happen), or conversely a dormant negative
    control that mutated despite declaring no training. Runtime buffers (last
    dotted component in ``buffer_suffixes``) are excluded from the movement
    measurement: they track quantization stats, not training, and counting them
    as movement would fabricate the exact signal the gate exists to verify.

    Parameters = keys under ``tower_prefix`` whose last dotted component is NOT
    a buffer suffix. Coverage counts params present in both ``base_digests`` and
    ``saved_digests``; ``expected`` is the full base parameter count, so any
    parameter missing from the saved map is UNDERCOVERED (or VACUOUS if checked
    drops to 0 — every base param missing at once) and must block.

    When ``exercised=True``, at least one parameter must have moved (digest must
    differ between base and saved) by the time of save or the run's claim is
    false ("declared exercised, carried unchanged"). When ``exercised=False``
    (the dormant negative control), NO parameter may have moved — any movement
    is a contradiction of the declaration and must block.
    """

    id: ClassVar[str] = "speech.tower_movement"
    description: ClassVar[str] = (
        "The speech tower's trainable parameters actually moved off their base "
        "digests at save iff the run declared them exercised (and the converse "
        "for the dormant negative control: nothing moves when the tower is "
        "declared not-exercised)"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.SAVE,)
    context_type: ClassVar[type | None] = TowerMovementContext

    def check(self, ctx: Any) -> GateResult:
        pre = ctx.tower_prefix
        base = dict(ctx.base_digests)
        saved = dict(ctx.saved_digests)
        buffer_suffixes = set(ctx.buffer_suffixes)
        exercised = ctx.exercised

        def _is_param(k: str) -> bool:
            # A whole dotted segment, so "model.audio_tower" does not claim a
            # sibling such as "model.audio_tower_proj".
            if not k.startswith(pre + "."):
                return False
            last = k.rsplit(".", 1)[-1]
            return last not in buffer_suffixes

        base_params = sorted(k for k in base if _is_param(k))
        saved_params = {k for k in saved if _is_param(k)}

        if not base_params:
            # No trainable parameters found under the prefix in the base (or
            # only buffers). Nothing to attest about; returning through self.ok
            # over zero coverage routes this to VACUOUS via the framework's
            # enforced downgrade.
            return self.ok(
                f"no trainable parameters found under tower_prefix {pre!r} in "
                "the base (after excluding runtime buffers) — nothing to "
                "attest about for tower movement",
                Coverage.none("tower parameters"),
                evidence={
                    "tower_prefix": pre,
                    "base_key_count": len(base),
                    "buffer_suffixes": sorted(buffer_suffixes),
                },
            )

        both = [k for k in base_params if k in saved_params]
        missing_in_saved = [k for k in base_params if k not in saved_params]

        # Expected = the gate's external denominator (# trainable params in
        # base). Checked = # params present in BOTH maps — params dropped from
        # the save are a shortfall and must block (framework downgrade of
        # self.ok: UNDERCOVERED when 0 < checked < expected, VACUOUS when
        # checked == 0).
        coverage = Coverage(
            checked=len(both),
            unit="tower parameters",
            expected=len(base_params),
        )

        if missing_in_saved:
            # Base params absent from the saved digest map. The framework's
            # self.ok() downgrade handles the shortfall verdict (UNDERCOVERED
            # when some params made it, VACUOUS when every base param is
            # missing at once — both blocking). Route through ok() so the
            # enforced downgrade applies.
            return self.ok(
                f"{len(missing_in_saved)} of {len(base_params)} tower "
                f"parameters under {pre!r} missing from the saved digest map: "
                f"{missing_in_saved[:8]}",
                coverage,
                evidence={
                    "missing_in_saved": missing_in_saved[:16],
                    "missing_total": len(missing_in_saved),
                    "tower_prefix": pre,
                },
            )

        # Every base param made it into the saved map: the two maps agree on
        # the same set of trainable parameters. Now measure movement.
        moved = [k for k in both if base[k] != saved[k]]
        unchanged = [k for k in both if base[k] == saved[k]]
        total = len(both)

        if exercised:
            if moved:
                # Spec: "PASS iff moved (digest differs) > 0; report moved/total
                # in the message."
                return self.ok(
                    f"{len(moved)} of {total} tower parameters moved from base "
                    "to saved — declared exercised, and at least one moved",
                    coverage,
                    evidence={
                        "moved_sample": moved[:8],
                        "moved_total": len(moved),
                    },
                )
            # "FAIL when 0 moved ('declared exercised, carried unchanged')."
            return self.fail(
                f"declared exercised but 0 of {total} tower parameters moved — "
                "every trainable parameter carried unchanged from base to saved",
                coverage,
                evidence={
                    "unchanged_sample": unchanged[:8],
                    "unchanged_total": len(unchanged),
                    "exercised": True,
                },
            )

        # Not exercised — the dormant negative control. "PASS iff moved == 0";
        # "FAIL if any moved."
        if moved:
            return self.fail(
                f"dormant negative control (exercised=False) but {len(moved)} "
                f"of {total} tower parameters moved — the tower was declared "
                "not-exercised yet its parameters changed digests",
                coverage,
                evidence={
                    "moved_sample": moved[:8],
                    "moved_total": len(moved),
                    "exercised": False,
                },
            )
        return self.ok(
            f"dormant negative control (exercised=False): 0 of {total} tower "
            "parameters moved — as declared",
            coverage,
        )

    def controls(self) -> list[Control]:
        def _single_digest_pair(
            base_digest: str,
            saved_digest: str,
            *,
            exercised: bool,
        ) -> TowerMovementContext:
            return TowerMovementContext(
                tower_prefix="speech.tower",
                base_digests={"speech.tower.encoder.w": base_digest},
                saved_digests={"speech.tower.encoder.w": saved_digest},
                exercised=exercised,
            )

        return [
            Control(
                name="no-tower-params-vacuous",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: TowerMovementContext(
                    tower_prefix="speech.tower",
                    base_digests={},
                    saved_digests={},
                ),
                note="no parameters under tower_prefix at all: cannot attest "
                "movement, must block as VACUOUS with zero coverage",
            ),
            Control(
                name="missing-in-saved-undercovered",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: TowerMovementContext(
                    tower_prefix="speech.tower",
                    base_digests={
                        "speech.tower.encoder.w": "abc",
                        "speech.tower.encoder.b": "def",
                        "speech.tower.decoder.w": "ghi",
                    },
                    saved_digests={
                        # encoder.b and decoder.w absent entirely.
                        "speech.tower.encoder.w": "abc",
                    },
                    exercised=True,
                ),
                note="3 tower params in base, 1 in saved: 2 missing must block "
                "as UNDERCOVERED (framework downgrade over Coverage(1,3))",
            ),
            Control(
                name="exercised-but-unchanged",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: _single_digest_pair(
                    "same-digest",
                    "same-digest",
                    exercised=True,
                ),
                note="declared exercised=True but every parameter carried the "
                "identical digest from base to saved: must FAIL naming "
                "'declared exercised, carried unchanged'",
            ),
            Control(
                name="dormant-but-moved",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: _single_digest_pair(
                    "base-digest",
                    "different-digest",
                    exercised=False,
                ),
                note="declared exercised=False (dormant negative control) but a "
                "parameter digest changed: must FAIL",
            ),
            Control(
                name="exercised-and-moved",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: _single_digest_pair(
                    "base-digest",
                    "different-digest",
                    exercised=True,
                ),
                note="declared exercised=True and at least one parameter moved: "
                "the clean affirmative pass state under exercised=True",
            ),
            Control(
                name="dormant-and-static",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: _single_digest_pair(
                    "same-digest",
                    "same-digest",
                    exercised=False,
                ),
                note="declared exercised=False and no parameter moved: the "
                "clean pass state under the dormant negative control (still "
                "affirms healthy input — the gate accepts a correctly-static "
                "untrained tower)",
            ),
        ]


@dataclass(frozen=True)
class RunawayHypothesisContext:
    """Runaway census over one eval run's rows, for the base and the tuned model.

    ``rows_expected`` is the corpus's declared size (2504 for Earnings-22) and
    must come from OUTSIDE the eval artifact — a run's summary can claim any
    number of rows, which is exactly the audited disaster the framework names.
    ``rows_checked`` counts the rows whose (reference, hypothesis) pairs were
    actually measured; ``base_runaway`` and ``tuned_runaway`` count the rows
    ``train.speech_metrics.is_runaway`` flagged for the base and the tuned
    model over those same rows.

    The accounting invariants — every count non-negative, neither runaway count
    larger than ``rows_checked`` — are REFUSED (``ValueError``), never priced: a
    census that cannot describe any real run corrupts the measurement, and a
    limit derived from it would launder the miscount into a verdict about the
    model. That is the same refusal ``Coverage``'s own constructor makes over a
    negative count, one field earlier.
    """

    rows_expected: int
    rows_checked: int
    base_runaway: int
    tuned_runaway: int


@register
class RunawayHypothesisGate(Gate):
    """The tuned model's runaway rows stay near the base rate, or the run blocks.

    Defect class: a decoder that stopped listening and kept emitting. The
    measured case is a Canary-1B fine-tune whose training rows pointed at the
    wrong audio (a data-builder bug left 6,000 rows on 820 files). It passed
    every adjudication gate and still ran 122 of its 2504 Earnings-22
    hypotheses into repetition loops and hallucinated domain text
    (``"So, uh, that's a good question."``) where the base model ran 3 and a
    clean LibriSpeech+AMI fine-tune ran 7.
    Nothing is malformed in such a run: the tower moved, every audio row is
    accounted for, the save lines up — so movement and coverage gates report a
    healthy artifact. The hypothesis LENGTHS against the base model's own
    runaway count are the only trace a stopped decoder leaves behind.

    The allowance is ``limit = 2 * base_runaway + ceil(0.005 * rows_checked)``:
    double the base drift, plus half a percent of the measured rows for
    segmentation noise (the per-row slack of ``RUNAWAY_SLACK_WORDS`` is spent
    inside ``is_runaway``; this is the corpus-level slack). RED (FAIL, blocking)
    iff ``tuned_runaway > limit`` — landing exactly on the limit is inside it.
    The base model is the yardstick rather than a hard ceiling because a corpus
    the base reads just as badly must not block on length alone.

    Coverage is the row census: (rows_checked, rows_expected) in ``"eval rows"``,
    so zero measured rows is VACUOUS and a short sweep is UNDERCOVERED — both
    blocking, exactly as :class:`AudioRowCoverageGate` reports its rows.

    Counts that cannot describe a run are refused ``ValueError`` before any
    limit is derived: a negative count, or a runaway count over more rows than
    were checked. Malformed input is corruption of the measurement, not a fact
    about the model.
    """

    id: ClassVar[str] = "speech.runaway_hypotheses"
    description: ClassVar[str] = (
        "The tuned model's runaway hypotheses stay within the limit of "
        "2 * base_runaway + ceil(0.005 * rows_checked) at save: a decoder that "
        "stopped listening and ran away into repetition loops or hallucinated "
        "domain text is invisible to every artifact gate and only this count "
        "sees it"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.SAVE,)
    context_type: ClassVar[type | None] = RunawayHypothesisContext

    def check(self, ctx: Any) -> GateResult:
        c = ctx
        rows_expected = c.rows_expected
        rows_checked = c.rows_checked
        base_runaway = c.base_runaway
        tuned_runaway = c.tuned_runaway

        # Refusing impossible input BEFORE any limit is derived, exactly as
        # Coverage's constructor refuses a negative count. A census that cannot
        # describe any real run is corruption of the measurement; computing a
        # limit over it would launder the miscount into a verdict about the
        # model, so this is a ValueError (fail closed through Gate.run), never a
        # priced FAIL about rows that were never measured.
        if min(rows_expected, rows_checked, base_runaway, tuned_runaway) < 0:
            raise ValueError(
                "runaway counts cannot be negative: rows_expected="
                f"{rows_expected}, rows_checked={rows_checked}, "
                f"base_runaway={base_runaway}, tuned_runaway={tuned_runaway} — "
                "a negative count describes no run at all"
            )
        if base_runaway > rows_checked or tuned_runaway > rows_checked:
            raise ValueError(
                "a runaway count cannot outrun the rows it counts over: "
                f"rows_checked={rows_checked}, base_runaway={base_runaway}, "
                f"tuned_runaway={tuned_runaway} — the census is impossible and "
                "is refused, never priced"
            )

        # The per-spec Coverage formula (checked = rows measured, expected =
        # manifest). It serves every branch below except rows_checked == 0,
        # which deliberately overrides it to Coverage.none for the same reason
        # AudioRowCoverageGate does: forcing checked=0 is the only way to make
        # the framework's ok() downgrade yield the mandated VACUOUS verdict.
        coverage = Coverage(
            checked=rows_checked,
            unit="eval rows",
            expected=rows_expected,
        )

        if rows_checked == 0:
            # First rule of this gate and absolute: zero measured rows attests
            # nothing, whatever hangs off it (the refusals above have already
            # forbidden any runaway count here). Through self.ok over
            # Coverage.none this is the framework's VACUOUS downgrade — the only
            # sanctioned way to produce it without hand-assembling a result.
            return self.ok(
                f"no eval row was measured (0 of {rows_expected} expected) — "
                "nothing to attest about for runaway hypotheses",
                Coverage.none("eval rows"),
                evidence={
                    "rows_expected": rows_expected,
                    "rows_checked": rows_checked,
                    "base_runaway": base_runaway,
                    "tuned_runaway": tuned_runaway,
                },
            )

        # Double the base drift plus the corpus slack. Both arms are spelled out
        # in every message: the limit IS the claim, and a threshold a reader
        # cannot recompute is a threshold nobody can audit.
        limit = 2 * base_runaway + ceil(0.005 * rows_checked)

        if tuned_runaway > limit:
            return self.fail(
                f"{tuned_runaway} of {rows_checked} rows ran away against the "
                f"limit of {limit} (2 * base_runaway {base_runaway} + "
                f"ceil(0.005 * {rows_checked})) — the tuned decoder stopped "
                f"listening where the base model stayed with the transcript",
                coverage,
                evidence={
                    "rows_checked": rows_checked,
                    "base_runaway": base_runaway,
                    "tuned_runaway": tuned_runaway,
                    "limit": limit,
                },
            )

        return self.ok(
            f"{tuned_runaway} of {rows_checked} rows ran away against the limit "
            f"of {limit} (2 * base_runaway {base_runaway} + ceil(0.005 * "
            f"{rows_checked})) — the decoder kept listening",
            coverage,
            evidence={
                "rows_checked": rows_checked,
                "base_runaway": base_runaway,
                "tuned_runaway": tuned_runaway,
                "limit": limit,
            },
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                name="earnings-22-collapse",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RunawayHypothesisContext(
                    rows_expected=2504,
                    rows_checked=2504,
                    base_runaway=3,
                    tuned_runaway=122,
                ),
                note="the measured collapse (a fine-tune trained on rows whose "
                "audio did not match their transcripts): 122 of 2504 "
                "Earnings-22 rows ran away where the base model ran 3 (limit 19 = 2 * 3 + "
                "ceil(0.005 * 2504)) — must RED-block naming both counts and "
                "the limit",
            ),
            Control(
                name="earnings-22-clean-finetune",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: RunawayHypothesisContext(
                    rows_expected=2504,
                    rows_checked=2504,
                    base_runaway=3,
                    tuned_runaway=7,
                ),
                note="the measured healthy run over the same corpus: a clean "
                "LibriSpeech+AMI fine-tune, 7 rows against 3 base runaways, "
                "inside the limit of 19",
            ),
            Control(
                name="no-rows-vacuous",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RunawayHypothesisContext(
                    rows_expected=5,
                    rows_checked=0,
                    base_runaway=0,
                    tuned_runaway=0,
                ),
                note="0 rows measured against 5 expected: must block as VACUOUS "
                "with zero coverage (no attestation over zero rows)",
            ),
            Control(
                name="undercover-row-count",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RunawayHypothesisContext(
                    rows_expected=5,
                    rows_checked=3,
                    base_runaway=1,
                    tuned_runaway=1,
                ),
                note="3 of 5 rows measured, runaways inside the limit (2 * 1 + "
                "ceil(0.005 * 3) = 3): the shortfall of 2 must block as "
                "UNDERCOVERED (framework downgrade over Coverage(3, 5))",
            ),
            Control(
                name="runaway-beyond-checked",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RunawayHypothesisContext(
                    rows_expected=5,
                    rows_checked=3,
                    base_runaway=0,
                    tuned_runaway=4,
                ),
                note="4 runaway rows counted over 3 checked rows: the census "
                "cannot describe any run and must be refused (ValueError, "
                "blocking) before any limit is derived from it",
            ),
        ]


@dataclass(frozen=True)
class RepetitionLoopContext:
    """Repetition-loop census over one eval run's rows, for the base and the tuned model.

    ``rows_expected`` is the corpus's declared size (2,504 for a Earnings-22
    clip sweep, 6 whole calls for the long-form one) and must come from OUTSIDE
    the eval artifact — a run's summary can claim any number of rows, which is
    exactly the audited disaster the framework names. ``rows_checked`` counts the
    rows whose (reference, hypothesis) pairs were actually measured;
    ``reference_words`` is their summed NORMALIZED reference length
    (``LoopCount.reference_words``) and ``base_loop_words`` /
    ``tuned_loop_words`` are the words ``train.speech_metrics.loop_words``
    flagged inside the base and the tuned hypotheses over those same rows.

    Loop words are HYPOTHESIS words and routinely outnumber the reference's words
    in a collapse — a decoder stuck on ``"the"`` emits words the reference never
    had — so ``loop words <= reference words`` is NOT an invariant here and is not
    refused. Only counts that cannot describe any run (a negative number) are
    refused ``ValueError``, never priced: the same refusal ``Coverage``'s own
    constructor makes over a negative count, one field earlier.
    """

    rows_expected: int
    rows_checked: int
    reference_words: int
    base_loop_words: int
    tuned_loop_words: int


@register
class RepetitionLoopGate(Gate):
    """The tuned model's local repetition loop words stay near the base count, or the run blocks.

    Defect class: a decoder that fell into a LOCAL repetition loop INSIDE one
    chunk. The measured case is a Canary-1B Earnings fine-tune decoded with
    NeMo's chunked long-form inference over six whole Earnings-22 calls: the
    loops (``"the the the ..."``, ``"uh uh uh ..."``) made 1,428 words beside
    50,400 reference words where the base model looped none, turning a
    17.30 -> 14.14 WER gain (the loops collapsed, a diagnostic) into the real
    17.30 -> 16.77. ``speech.runaway_hypotheses`` passed over that output and
    could not have fired: it judges whole rows (past twice the row's reference
    length) and a loop inside one 40 s chunk of an hour-long call adds 4-12% of
    that call's words. Loops must be counted locally, in words, against the base
    model.

    The allowance is ``limit = 2 * base_loop_words + ceil(0.005 *
    reference_words)``: double the base drift, plus half a percent of the
    reference words for the repetition every corpus carries in its speech and its
    transcript alike (the per-run run-length floor is spent inside ``loop_words``;
    this is the corpus-level slack). RED (FAIL, blocking) iff
    ``tuned_loop_words > limit`` — landing exactly on the limit is inside it. The
    base model is the yardstick rather than a hard ceiling because a corpus the
    base reads with the same repetitions must not block on repetition alone (AMI's
    base model loops 165 words in 15,194, and a fine-tune sitting at 188 is
    drift, not a defect).

    Coverage is the row census: (rows_checked, rows_expected) in ``"eval rows"``,
    so zero measured rows is VACUOUS and a short sweep is UNDERCOVERED — both
    blocking, exactly as :class:`AudioRowCoverageGate` reports its rows.

    Counts that cannot describe a run are refused ``ValueError`` before any limit
    is derived: anything negative — and nothing else, because loop words are
    hypothesis words and a collapse CAN carry more of them than the reference has
    words. Malformed input is corruption of the measurement, not a fact about the
    model.
    """

    id: ClassVar[str] = "speech.repetition_loops"
    description: ClassVar[str] = (
        "The tuned model's local repetition loop words stay within the limit of "
        "2 * base_loop_words + ceil(0.005 * reference_words) at save: a decoder "
        "that fell into a repetition loop inside one chunk is invisible to every "
        "artifact gate and sails under the whole-row runaway detector, and only "
        "this local count sees it"
    )
    events: ClassVar[tuple[Lifecycle, ...]] = (Lifecycle.SAVE,)
    context_type: ClassVar[type | None] = RepetitionLoopContext

    def check(self, ctx: Any) -> GateResult:
        c = ctx
        rows_expected = c.rows_expected
        rows_checked = c.rows_checked
        reference_words = c.reference_words
        base_loop_words = c.base_loop_words
        tuned_loop_words = c.tuned_loop_words

        # Refusing impossible input BEFORE any limit is derived, exactly as
        # Coverage's constructor refuses a negative count. Only negatives here:
        # loop words are HYPOTHESIS words and a collapse can emit more of them
        # than the reference has words (the measured 1,428 beside a reference
        # carrying its own 50,400), so bounding them by the reference would
        # refuse the very measurement this gate exists to make. A negative count,
        # though, describes no run at all — refused, never priced.
        if (
            min(
                rows_expected,
                rows_checked,
                reference_words,
                base_loop_words,
                tuned_loop_words,
            )
            < 0
        ):
            raise ValueError(
                "loop counts cannot be negative: rows_expected="
                f"{rows_expected}, rows_checked={rows_checked}, "
                f"reference_words={reference_words}, "
                f"base_loop_words={base_loop_words}, "
                f"tuned_loop_words={tuned_loop_words} — "
                "a negative count describes no run at all"
            )

        # The Coverage formula (rows checked against the manifest's expected),
        # with rows_checked == 0 deliberately overridden to Coverage.none below
        # for the same reason AudioRowCoverageGate does it: forcing checked=0 is
        # the only way to make the framework's ok() downgrade yield the mandated
        # VACUOUS verdict.
        coverage = Coverage(
            checked=rows_checked,
            unit="eval rows",
            expected=rows_expected,
        )

        if rows_checked == 0:
            # First rule of this gate and absolute: zero measured rows attests
            # nothing (the refusal above has already forbidden any loop count
            # here). Through self.ok over Coverage.none this is the framework's
            # VACUOUS downgrade — the only sanctioned way to produce it without
            # hand-assembling a result.
            return self.ok(
                f"no eval row was measured (0 of {rows_expected} expected) — "
                "nothing to attest about for repetition loops",
                Coverage.none("eval rows"),
                evidence={
                    "rows_expected": rows_expected,
                    "rows_checked": rows_checked,
                    "reference_words": reference_words,
                    "base_loop_words": base_loop_words,
                    "tuned_loop_words": tuned_loop_words,
                },
            )

        # Double the base drift plus the corpus slack over the reference's
        # words. Both arms are spelled out in every message: the limit IS the
        # claim, and a threshold a reader cannot recompute is a threshold nobody
        # can audit.
        limit = 2 * base_loop_words + ceil(0.005 * reference_words)

        if tuned_loop_words > limit:
            return self.fail(
                f"{tuned_loop_words} loop words over {reference_words} reference "
                f"words (the base model produced {base_loop_words}) exceeded the "
                f"limit of {limit} (2 * base_loop_words {base_loop_words} + "
                f"ceil(0.005 * {reference_words})) — the tuned decoder fell into "
                "local repetition loops where the base model stayed with the "
                "transcript",
                coverage,
                evidence={
                    "rows_checked": rows_checked,
                    "reference_words": reference_words,
                    "base_loop_words": base_loop_words,
                    "tuned_loop_words": tuned_loop_words,
                    "limit": limit,
                },
            )

        return self.ok(
            f"{tuned_loop_words} loop words over {reference_words} reference "
            f"words (the base model produced {base_loop_words}) stayed within "
            f"the limit of {limit} (2 * base_loop_words {base_loop_words} + "
            f"ceil(0.005 * {reference_words})) — the decoder kept out of the "
            "repetition loops",
            coverage,
            evidence={
                "rows_checked": rows_checked,
                "reference_words": reference_words,
                "base_loop_words": base_loop_words,
                "tuned_loop_words": tuned_loop_words,
                "limit": limit,
            },
        )

    def controls(self) -> list[Control]:
        return [
            Control(
                name="earnings-22-long-form-loops",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=6,
                    rows_checked=6,
                    reference_words=50400,
                    base_loop_words=0,
                    tuned_loop_words=1428,
                ),
                note="the measured NeMo chunked collapse (Canary-1B Earnings "
                "fine-tune over 6 whole calls): 1,428 loop words beside 50,400 "
                "reference words where the base model looped none (limit 252 = "
                "2 * 0 + ceil(0.005 * 50400)) — must RED-block naming both "
                "counts, the reference words and the limit",
            ),
            Control(
                name="earnings-22-corrupted-data-clips",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=2504,
                    rows_checked=2504,
                    reference_words=47865,
                    base_loop_words=0,
                    tuned_loop_words=2936,
                ),
                note="the corrupted-data fine-tune's clip eval: 2,936 loop words "
                "beside 47,865 reference words where the base model looped none "
                "(limit 240 = 2 * 0 + ceil(0.005 * 47865)) — must RED-block "
                "naming both counts and the limit",
            ),
            Control(
                name="earnings-22-clip-finetune",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=2504,
                    rows_checked=2504,
                    reference_words=47865,
                    base_loop_words=0,
                    tuned_loop_words=151,
                ),
                note="the clean clip fine-tune over the same corpus: 151 loop "
                "words inside the limit of 240 — ordinary drift over 47,865 "
                "reference words, not a decoder in a loop",
            ),
            Control(
                name="ami-base-loops-too",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=2000,
                    rows_checked=2000,
                    reference_words=15194,
                    base_loop_words=165,
                    tuned_loop_words=188,
                ),
                note="a corpus the BASE model also loops on: 165 base loop words "
                "in 15,194 reference words, and a fine-tune at 188 is drift "
                "inside the limit of 406 (2 * 165 + ceil(0.005 * 15194)) — the "
                "base model is the yardstick, not a zero",
            ),
            Control(
                name="no-rows-vacuous",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=5,
                    rows_checked=0,
                    reference_words=0,
                    base_loop_words=0,
                    tuned_loop_words=0,
                ),
                note="0 rows measured against 5 expected: must block as VACUOUS "
                "with zero coverage (no attestation over zero rows)",
            ),
            Control(
                name="undercover-row-count",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=5,
                    rows_checked=3,
                    reference_words=100,
                    base_loop_words=1,
                    tuned_loop_words=1,
                ),
                note="3 of 5 rows measured with the loops inside the limit "
                "(2 * 1 + ceil(0.005 * 100) = 3): the shortfall of 2 must block "
                "as UNDERCOVERED (framework downgrade over Coverage(3, 5))",
            ),
            Control(
                name="negative-count-refused",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: RepetitionLoopContext(
                    rows_expected=5,
                    rows_checked=5,
                    reference_words=100,
                    base_loop_words=0,
                    tuned_loop_words=-1,
                ),
                note="a negative loop-word count describes no run and must be "
                "refused (ValueError, blocking) before any limit is derived "
                "from it",
            ),
        ]
