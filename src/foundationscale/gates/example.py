"""The reference gate: expert-content aliasing in MoE checkpoints.

This module is deliberately read more than it is run. It is the worked example every
future gate author copies, so its structure is the teaching material: a small context
type decoupling the gate from any real checkpoint reader, a check that returns only
through ``self.ok``/``self.fail``/``self.skip``, and a controls list with a MUST_FIRE
for every defect class — including the empty one.

The incident it encodes
-----------------------
A Mixture-of-Experts model class was not registered as a parallel-aware module, so
at save time its experts were written under LOCAL names. 128 distinct experts came
out on disk as 16 experts replicated 8 times. Every structural check in the estate
passed: rc=0, resume, healthy loss, tensor counts, dtypes, tensor byte sizes — all
of them, because every one of those properties was genuinely correct. The only
thing wrong was the content: expert ``i`` and expert ``i % 16`` were bitwise
identical. Two full training runs ran on a model that was 87.5% wrong.

Two independent, checkable signatures fell out of that defect, and this gate checks
both:

1. **Name signature** — keys spelled ``...linear_fc1.weight0`` .. ``...weight15``,
   a trailing local index where a global expert path should be. Detectable from the
   key set alone, before a single tensor byte is read, and it means global expert
   identity is already unrecoverable.
2. **Content signature** — experts whose bytes are bitwise identical to a
   lower-indexed expert. This is the one no shape/count/dtype check can ever see.

The gate also encodes the audit's sharpest lesson by *not* encoding it: there is no
special case for an empty expert set ON A DECLARED-MoE MODEL. ``self.ok`` is called
with zero coverage and the contract in :mod:`~foundationscale.gates.core` downgrades
it to VACUOUS. The MUST_FIRE control ``empty-expert-set`` exists to prove that stays
true.

Exactly one exception stands beside that lesson, and it is narrow enough to state in
one sentence: a POSITIVE dense declaration — ``declared_expert_count == 0``, never
an absence of evidence — corroborated by zero expert tensors in the artifact is the
one shape in which "no experts" means "no experts", so only that shape earns the
gate's declared ``NOT_APPLICABLE`` abstention. Every other arrival at "no experts"
(undeclared-but-present, declared-but-malformed) blocks. Look-alike zeros
(``False``, ``0.0``, ``0j``) satisfy ``== 0`` in Python without being dense
declarations — the checkpoint gates learned that from a YAML ``moe: false`` — so a
type guard blocks malformed counts BEFORE the dense door can be purchased with one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
from .fixtures import (
    ExpertSet,
    make_aliased_experts,
    make_empty_experts,
    make_healthy_experts,
    make_local_name_experts,
)

__all__ = ["ExpertCheckContext", "ExpertAliasGate"]


FingerprintFn = Callable[[bytes], str]
"""Content fingerprint for an expert tensor.

Injectable so the gate — and, more importantly, its controls — never needs torch or
a real reader. The default is a stable hash of the raw bytes. A fingerprint MUST
distinguish content: comparing pointers, ids or tensor *objects* is the exact bug
(a replicated view of the same storage compares equal to itself) this gate exists
to catch.
"""


def _default_fingerprint(data: bytes) -> str:
    """Stable content hash of raw tensor bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ExpertCheckContext:
    """Everything :class:`ExpertAliasGate` needs from a checkpoint, and nothing more.

    The gate never opens a file or imports a framework; whoever owns the reader
    (checkpoint writer, converter, promotion pipeline) adapts their format into
    this shape. That indirection is deliberate: in the audited estate, checks that
    knew their file format got copy-pasted per script and quietly omitted from one
    of them; checks that know an interface can run identically at FIRST_SAVE, SAVE
    and anywhere else the context can be built.

    Args:
        tensors: Expert tensor checkpoint key -> raw content bytes.
        expert_index: Key -> *global* expert index for every key in ``tensors``
            whose name carries one. Keys spelled with local indices have no entry —
            the gate treats that as the name-signature failure, not as missing data.
        declared_expert_count: The expert count from the model config, i.e. what the
            artifact is supposed to hold. Never derive this by counting keys in the
            artifact itself — the corrupt artifact "contained" exactly as many expert
            keys as its bogus index.json claimed it should.
    """

    tensors: Mapping[str, bytes]
    expert_index: Mapping[str, int]
    declared_expert_count: int

    @classmethod
    def from_expert_set(cls, experts: ExpertSet) -> ExpertCheckContext:
        """Adapt a synthetic control fixture. Production callers write their own adapter."""
        return cls(
            tensors=experts.tensors,
            expert_index=experts.expert_index,
            declared_expert_count=experts.declared_expert_count,
        )


def _ranges(indices: Sequence[int]) -> str:
    """Render sorted integer indices compactly, e.g. ``[0,1,2,5,7,8]`` -> ``0-2, 5, 7-8``.

    A failure detail that must name 112 offending indices has to be readable to be
    actionable; ranges keep the message one line instead of a wall of numbers nobody
    checks — and a detail nobody reads is one step from a gate nobody trusts.
    """
    if not indices:
        return "none"
    parts: list[str] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = i
    parts.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(parts)


@register
class ExpertAliasGate(Gate):
    """Blocks MoE checkpoints whose experts were saved under local (aliased) identity.

    Scope, stated because the repo now produces evidence-backed dense
    declarations this gate will meet for real: expert-content aliasing is a
    property of MoE artifacts. On a POSITIVE dense declaration
    (``declared_expert_count == 0``) corroborated by an artifact containing
    zero expert tensors, the property does not exist, and this gate makes a
    DECLARED ``NOT_APPLICABLE`` abstention — never a PASS over zero examined
    units (doctrine 1), and never a VACUOUS block of a legitimate dense
    checkpoint (a gate that blocks every dense model at first save is a gate
    operators learn to route around). Any other shape that arrives with "no
    experts" blocks: expert tensors beside a declared 0 take the framework's
    OVERCOVERED door, and a count that merely COMPARES equal to zero
    (``False``, ``0.0``) is a malformed denominator, blocked before the dense
    door. Both directions of the abstention are pinned by controls
    (``dense-model`` / ``malformed-dense-count-bool``); until they existed,
    this gate's dense behaviour was unverified in both directions at once.

    Registered with the global :data:`REGISTRY` at import time so the CI controls
    job exercises it without anyone remembering to add it to a list — a gate that
    requires a registration step is a gate that is absent from exactly the run
    where it would have fired.
    """

    id = "checkpoint.expert_alias"
    description = (
        "Expert tensor content is not bitwise-aliased across global expert indices, "
        "and expert keys carry global indices — the 128-saved-as-16 incident"
    )
    events = (Lifecycle.FIRST_SAVE, Lifecycle.SAVE)
    # Declared, because this gate reads a context no other checkpoint gate produces:
    # `check` needs declared_expert_count, which a CheckpointGateContext does not
    # carry. Leaving context_type unset makes the gate LEGACY, and a legacy gate is
    # broadcast whatever context the sweep is holding -- so a run that happened to
    # import this module died at its first save on
    # `AttributeError: 'CheckpointGateContext' object has no attribute
    # 'declared_expert_count'`, scored as a blocking ERROR (#250). Registration is an
    # import side effect, so which runs died was decided by an import graph. Declaring
    # the type makes the mismatch a dispatch fact instead of a traceback: the gate is
    # reported unwired, or SKIP where the caller declared the abstention, and it never
    # reaches `check` with a context it cannot read.
    context_type = ExpertCheckContext

    #: Trailing digits directly on a weight/bias stem: the local-name signature.
    #: Canonical names end in ``.weight`` / ``.bias``; the corrupted module appended
    #: its local shard index instead (``...linear_fc1.weight7``).
    LOCAL_NAME_RE = re.compile(r"\.(?:weight|bias)(\d+)$")

    def __init__(self, fingerprint: FingerprintFn = _default_fingerprint) -> None:
        self._fingerprint = fingerprint

    #: The expert index inside a tensor key, e.g. the ``16`` in
    #: ``layers.0.experts.16.linear_fc1.weight``.
    _EXPERT_SEG_RE = re.compile(r"(?<=\.experts\.)\d+(?=\.)")

    @classmethod
    def _role(cls, name: str) -> str:
        """The tensor's *role*: its key with the expert index blanked out.

        ``layers.0.experts.16.linear_fc1.weight`` -> ``layers.0.experts.*.linear_fc1.weight``
        """
        return cls._EXPERT_SEG_RE.sub("*", name, count=1)

    def _expert_fingerprint(self, tensors: Mapping[str, bytes]) -> str:
        """One fingerprint for an expert's whole tensor set, keyed by role.

        Tensor identity goes into the hash so an expert with the right values under
        the wrong keys (fc1/fc2 swapped) is not silently "healthy". But it goes in as
        the *role* — the key with the expert index blanked — not as the raw name.

        That distinction is the whole gate, and getting it wrong is a live bug this
        file shipped with. Hashing the raw name folds the expert index into the
        fingerprint, so expert 0 and expert 16 differ in their hash *because they are
        expert 0 and expert 16*, no matter what bytes they hold. Every expert is then
        trivially unique, ``alias_map`` is always empty, and the gate returns PASS on
        the exact 128-saved-as-16 artifact it exists to catch. The MUST_FIRE control
        caught it on the first run — which is the argument for controls, made at this
        gate's own expense.
        """
        per = ";".join(
            f"{self._role(name)}={self._fingerprint(blob)}"
            for name, blob in sorted(tensors.items())
        )
        return hashlib.sha256(per.encode("utf-8")).hexdigest()

    def check(self, ctx: ExpertCheckContext) -> GateResult:
        """Look for the local-name signature first, then compare expert content."""
        declared = ctx.declared_expert_count
        # Python launders look-alikes into `== 0`: False == 0, 0.0 == 0 and
        # 0j == 0 are all True, so an unguarded `declared == 0` dense door would
        # admit a YAML `moe: false` flattened into the count field — the lesson
        # the checkpoint gates encoded as _checked_num_experts. The dataclass
        # annotates an int, but adapters are gate-author code (production
        # callers write their own, per ExpertCheckContext), and this framework's
        # premise is that annotations are promises worth checking. A malformed
        # count BLOCKS, routed through ok() over zero coverage so the framework
        # renders it VACUOUS with the raw value named — before it can buy the
        # abstention below or price any expected=coverage downstream.
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            return self.ok(
                f"declared_expert_count is {declared!r} ({type(declared).__name__}), "
                f"not a genuine non-negative integer: a malformed denominator "
                f"establishes nothing, and a value that merely compares equal to "
                f"zero must never buy the dense-model abstention",
                Coverage.none("expert tensors"),
                evidence={"declared_expert_count_raw": repr(declared)},
            )
        names = sorted(ctx.tensors)

        # 1. The name signature is the cheapest, most diagnostic check: it fires
        #    before any content is read, and when it fires, content comparison is
        #    impossible anyway because global expert identity no longer exists.
        local_named = sorted(n for n in names if self.LOCAL_NAME_RE.search(n))
        if local_named:
            return self.fail(
                f"{len(local_named)} expert tensors carry a trailing LOCAL index in "
                f"their name (e.g. {local_named[0]!r}); the checkpoint was written "
                f"by a module saving its local shard under unqualified names, and "
                f"global expert identity is unrecoverable from this artifact",
                Coverage(
                    checked=len(names),
                    unit="expert tensors",
                    expected=ctx.declared_expert_count,
                ),
                evidence={"local_named": local_named[:32]},
            )

        # 2. Group by declared global expert identity.
        by_expert: dict[int, dict[str, bytes]] = {}
        unattributed: list[str] = []
        for name in names:
            idx = ctx.expert_index.get(name)
            if idx is None:
                unattributed.append(name)
            else:
                by_expert.setdefault(idx, {})[name] = ctx.tensors[name]

        coverage = Coverage(
            checked=len(by_expert),
            unit="experts",
            expected=ctx.declared_expert_count,
        )

        # 3. The empty case is NOT special-cased for declared-MoE models: self.ok
        #    with zero coverage is downgraded to VACUOUS by the contract — that
        #    downgrade is the fix for the `all([]) is True` verification tool,
        #    and the `empty-expert-set` control below exists to prove nobody
        #    "helpfully" bypasses it here. The ONE exception is the positive
        #    dense declaration, and it is deliberately double-gated: the config
        #    says 0 experts AND the artifact census finds 0 expert tensors.
        if not by_expert:
            if declared == 0 and not names:
                # A dense model has no experts, so the aliasing property this
                # gate exists to check does not exist here. PASS would be the
                # founding defect verbatim ("no aliases found" over zero
                # examined units); VACUOUS would block every legitimate dense
                # checkpoint at first save, which is how gates get routed
                # around. The honest verdict is a DECLARED abstention with the
                # machine-readable kind attached, exactly the door the sibling
                # checkpoint gates take for num_experts == 0 — and the
                # `dense-model` control proves the door is taken, while
                # `malformed-dense-count-bool` proves a boolean zero cannot
                # sneak through it. Expert tensors present beside the declared
                # 0 fall through to the VACUOUS/UNDERCOVERED doors below or the
                # framework's OVERCOVERED door in step 4: a declaration the
                # artifact contradicts is a finding, never an abstention.
                return self.skip(
                    "context declares 0 experts — a positive dense-model "
                    "declaration — and the artifact census corroborates it "
                    "(zero expert tensors present): expert-content aliasing is "
                    "a property this artifact does not have, so this gate "
                    "abstains rather than passing over zero examined units or "
                    "blocking a legitimate dense checkpoint",
                    kind=AbstentionKind.NOT_APPLICABLE,
                )
            return self.ok(
                f"checkpoint exposes {len(names)} expert tensors total but none with "
                f"a resolvable global expert index",
                coverage,
            )

        if unattributed:
            return self.fail(
                f"{len(unattributed)} expert tensors have no resolvable global expert "
                f"index and are not the known local-index signature (e.g. "
                f"{unattributed[0]!r}); content cannot be attributed to an expert "
                f"identity, so the artifact cannot be vouched for",
                coverage,
                evidence={"unattributed": unattributed[:32]},
            )

        # 4. Content identity: the check no shape/count/dtype/size gate can do.
        #    First occurrence of a fingerprint wins; every later expert with the
        #    same content is an alias of it.
        canonical_of: dict[str, int] = {}
        alias_map: dict[int, int] = {}
        fingerprints: dict[int, str] = {}
        for idx in sorted(by_expert):
            fp = self._expert_fingerprint(by_expert[idx])
            fingerprints[idx] = fp
            if fp in canonical_of:
                alias_map[idx] = canonical_of[fp]
            else:
                canonical_of[fp] = idx

        if alias_map:
            # For the i % period incident, i - alias_of(i) is minimised exactly at
            # the replication period (128 -> 16 yields min diff 16); reporting it
            # turns "aliased" into "aliased the way the known bug aliases", which
            # is the difference between an alarm and a diagnosis.
            period = min(i - c for i, c in alias_map.items())
            offenders = sorted(alias_map)
            distinct = len(fingerprints) - len(alias_map)
            return self.fail(
                f"{len(alias_map)}/{len(fingerprints)} experts are bitwise aliases "
                f"of earlier-indexed experts — offending expert indices: "
                f"{_ranges(offenders)}, aliased to {_ranges(sorted(alias_map.values()))}; "
                f"inferred alias period: {period}; distinct experts on disk: {distinct}. "
                f"Every structural check (count, shape, dtype, size) passes on this "
                f"artifact; only content differs",
                coverage,
                evidence={
                    "alias_map": {str(i): c for i, c in sorted(alias_map.items())},
                    "alias_period": period,
                    "distinct_experts": distinct,
                    "offenders": offenders,
                },
            )

        return self.ok(
            f"all {len(fingerprints)} experts have globally-spelled names and "
            f"pairwise-distinct content",
            coverage,
            evidence={"distinct_experts": len(fingerprints)},
        )

    def controls(self) -> Sequence[Control]:
        """One MUST_FIRE per defect class, plus known-good MUST_PASS fixtures in
        both verdict directions the gate can honestly produce.

        The empty-expert-set control is the one future authors most often question;
        it is the whole point. A gate guarding against silent success must be proven
        not to succeed on the artifact that contains nothing to succeed on.

        The two DENSE controls flank it from the opposite side. ``dense-model``
        proves the positive-declaration door produces the stated, reasoned
        abstention — it declares ``expect_skip`` so the certification layer
        checks the declaration in both directions (an unexpected abstention
        fails; so would a stale one). ``malformed-dense-count-bool`` proves a
        boolean zero — ``False == 0`` is True — is refused at the door as a
        malformed denominator rather than admitted through it.
        """
        return [
            Control(
                name="aliased-128-as-16",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: ExpertCheckContext.from_expert_set(
                    make_aliased_experts(num_experts=128, period=16)
                ),
                note=(
                    "the audited incident: 128 experts stored as 16 replicated 8x; "
                    "names, counts, shapes and dtypes are all correct"
                ),
            ),
            Control(
                name="local-name-signature",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: ExpertCheckContext.from_expert_set(
                    make_local_name_experts(num_local=16, declared_expert_count=128)
                ),
                note=(
                    "keys spelled `...linear_fc1.weight0`..`...weight15`; must fail "
                    "from names alone, before any tensor content is compared"
                ),
            ),
            Control(
                name="empty-expert-set",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: ExpertCheckContext.from_expert_set(
                    make_empty_experts(declared_expert_count=128)
                ),
                note=(
                    "no expert tensors at all: the `all([]) is True` case that made "
                    "the original verification tool report success on a corrupt "
                    "artifact. MUST come back VACUOUS, and VACUOUS blocks"
                ),
            ),
            Control(
                name="healthy",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: ExpertCheckContext.from_expert_set(
                    make_healthy_experts(num_experts=8)
                ),
                note=(
                    "known-good artifact; guards against a gate that blocks on "
                    "everything and gets disabled rather than fixed"
                ),
            ),
            Control(
                name="dense-model",
                kind=ControlKind.MUST_PASS,
                make_ctx=lambda: ExpertCheckContext(
                    tensors={},
                    expert_index={},
                    declared_expert_count=0,
                ),
                note=(
                    "positive dense declaration corroborated by the artifact "
                    "census (0 declared, 0 present): the aliasing property does "
                    "not exist here, so the gate must take its stated "
                    "NOT_APPLICABLE door — never VACUOUS-block a healthy dense "
                    "checkpoint, never PASS over zero examined units"
                ),
                expect_skip=(
                    "expert-content aliasing is a property a declared-dense "
                    "artifact does not have; with the property absent the only "
                    "honest alternatives to this abstention were a blocking "
                    "vacuity (which blocks every dense model at first save and "
                    "teaches operators to route around the gate) or a pass "
                    "over zero examined units (the founding defect)"
                ),
            ),
            Control(
                name="malformed-dense-count-bool",
                kind=ControlKind.MUST_FIRE,
                make_ctx=lambda: ExpertCheckContext(
                    tensors={},
                    expert_index={},
                    declared_expert_count=False,
                ),
                note=(
                    "False == 0 in Python, so a YAML `moe: false` flattened "
                    "into the count field would take the dense door unopposed; "
                    "a boolean denominator is malformed and must VACUOUS-block "
                    "with its raw value named, before any `== 0` test runs"
                ),
            ),
        ]
