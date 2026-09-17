# SPDX-License-Identifier: Apache-2.0
"""#499: pin the KNOWN divergence between the two expert-key vocabularies.

This file does not endorse the divergence. It records it, because #499 is filed
and deliberately not fixed during announcement week: widening the shared tuple
changes what `tools/real_checkpoint_probe.py` declares for real checkpoints and
invalidates matrix rows measured against the narrow list. Re-measuring is the
fix, and it is bigger than the commit that found this.

What is guarded elsewhere, and what is not, is the whole point. The provenance
side genuinely shares ONE constant -- `gates/probe.py`, `models/adapters.py` and
`tools/real_checkpoint_probe.py` all import
`provenance.manifest._EXPERT_COUNT_KEYS`, and `models/adapters.py` holds that
relationship with an import-time `raise` backed by its own test. `train/loop.py`
declares a second, wider copy that nothing imports and no guard ties to the
first. So the two halves of the dense/MoE question can answer differently, and
nothing in the build says so.

Until the lists are unified, these tests are the thing that says so: any edit to
either tuple fails here and has to be made on purpose.
"""

from __future__ import annotations

from foundationscale.provenance.manifest import _EXPERT_COUNT_KEYS as PROVENANCE_KEYS
from foundationscale.train.loop import _EXPERT_COUNT_KEYS as LOOP_KEYS

# The divergence as MEASURED on 2026-09-18. Frozen literals rather than a
# relationship, so that widening EITHER side is what fails -- a test written only
# as `loop >= provenance` would stay green while the gap silently grew.
_PROVENANCE_AT_FILING = ("num_local_experts", "n_routed_experts", "num_experts")
_LOOP_AT_FILING = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_layer",
)
_DIVERGENT_AT_FILING = frozenset({"moe_num_experts", "num_experts_per_layer"})


def test_the_provenance_vocabulary_is_unchanged_since_filing() -> None:
    """Order matters here: this tuple is documented as precedence order."""
    assert PROVENANCE_KEYS == _PROVENANCE_AT_FILING


def test_the_loop_vocabulary_is_unchanged_since_filing() -> None:
    assert LOOP_KEYS == _LOOP_AT_FILING


def test_the_gap_is_exactly_the_two_keys_named_in_the_finding() -> None:
    """The defect #499 describes, stated as the set that causes it.

    A config declaring either of these is MoE to the training loop and dense to
    the manifest and the probe. That is the live consequence, and it is what
    re-measurement has to clear before the lists are unified.
    """
    assert set(LOOP_KEYS) - set(PROVENANCE_KEYS) == _DIVERGENT_AT_FILING


def test_the_loop_never_becomes_NARROWER_than_the_shared_constant() -> None:
    """The one direction that would be a new defect rather than the filed one.

    Wider means the two sides disagree, which is #499 and is written down. But
    a key the provenance producer honours and the training loop drops would mean
    the run that WROTE the checkpoint failed to see an expert count its own
    manifest records -- the declaration would be built from less than the
    evidence already in hand. `models/adapters.py` enforces this same direction
    with an import-time raise; the loop has no such guard, so it is asserted
    here.
    """
    assert set(PROVENANCE_KEYS) <= set(LOOP_KEYS)


def test_num_experts_per_tok_is_in_neither_vocabulary() -> None:
    """The router's top-k is not an expert count, and its absence is load-bearing.

    Every MoE config carries it. Read as a count it would declare an 8-expert
    denominator for a 128-expert layer, and the byte-volume gate would then
    confirm a checkpoint 94% short of its own expectation.
    """
    assert "num_experts_per_tok" not in set(PROVENANCE_KEYS) | set(LOOP_KEYS)
