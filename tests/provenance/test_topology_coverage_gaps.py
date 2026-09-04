"""Coverage-gap tests for the mismatch path of ``declared_vs_effective``.

Why this file exists
--------------------
A mutation battery flipped one line of the frozen module and the whole suite
stayed green::

    if findings:            ->      if False:

above the append of ``topology.effective_comparison_summary``. The trap is that
this does not *delete* the summary: the ``if`` has an ``else:``, so the
mismatch path is re-routed into the branch meant for a clean comparison. The
function returns per-field BLOCK findings followed by
``topology.effective_matches_declared`` — a report that lists differences and
declares "the runtime honours the declared config" in the same breath.

The existing ``test_mismatch_report_also_carries_comparison_coverage`` asked
only ``any(f.details.get("fields_compared") == 7 for f in findings)``, and the
impostor finding carries that exact key, so the mutant passed it. Every test
in this file pins the *code* of the finding that carries the coverage, not
merely the number, and the match path is asserted through the same access
pattern so the two paths cannot drift apart again.
"""

from __future__ import annotations

from collections.abc import Sequence

from foundationscale.topology import (
    Finding,
    Severity,
    Topology,
    declared_vs_effective,
)


def _topo(
    *,
    dp: int = 8,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    cp: int = 1,
    nodes: int = 1,
    gpus_per_node: int = 8,
    tasks_per_node: int | None = None,
) -> Topology:
    """Build a self-consistent topology: 8 GPUs (one node) by default."""
    return Topology(
        dp=dp,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        tasks_per_node=tasks_per_node,
    )


def _codes(findings: Sequence[Finding]) -> list[str]:
    return [f.code for f in findings]


def _by_code(findings: Sequence[Finding], code: str) -> Finding:
    for finding in findings:
        if finding.code == code:
            return finding
    raise AssertionError(f"no finding with code {code!r}; got {_codes(findings)}")


def test_mismatch_report_must_state_how_many_fields_were_compared() -> None:
    """A mismatch report without its comparison summary hides the breadth of the check.

    Wrongly believed if this fails: a caller sees BLOCK findings naming ``dp``
    and ``cp`` and cannot tell whether 7 fields were compared (and 5 matched)
    or only 2 were ever looked at — a partial comparison that happened to
    catch something renders identically to a thorough one. The summary must be
    present by code, carry the field count, and sit after the evidence.
    """
    findings = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=8, cp=1))
    assert "topology.effective_comparison_summary" in _codes(findings)
    summary = _by_code(findings, "topology.effective_comparison_summary")
    assert summary.severity is Severity.OK
    assert summary.details["fields_compared"] == 7  # five degrees + nodes + gpus_per_node
    assert summary.details["blocking"] == 2  # dp and cp, each named by a BLOCK above
    assert findings[-1] is summary  # the summary follows the evidence it summarises


def test_mismatch_report_never_also_claims_everything_matched() -> None:
    """The report must not list differences and declare a match in the same breath.

    The battery's mutant does not remove the summary — the ``if`` it breaks
    has an ``else:``, so the mismatch path falls into the clean-comparison
    branch and appends ``topology.effective_matches_declared`` alongside the
    per-field BLOCKs. That self-contradicting list is the green check mark on
    a lie this framework was built against, and the impostor finding carried
    the exact ``fields_compared`` value the old test scanned for. Pin the
    finding's identity, not the number it happens to carry.
    """
    findings = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=8, cp=1))
    assert "topology.effective_matches_declared" not in _codes(findings)
    assert [f.code for f in findings if f.severity is Severity.OK] == [
        "topology.effective_comparison_summary"
    ]


def test_match_report_carries_the_same_coverage_shape_as_the_mismatch_report() -> None:
    """Positive control, asserted through the same access pattern as the mismatch path.

    The match path must keep working — and its field count must be read with
    the same direct-index access used for the comparison summary above. If the
    two paths ever record their coverage differently (different codes,
    different detail keys, lookup via ``.get`` on one side only), one of these
    two tests goes red instead of both going quietly loose, which is precisely
    how the last mutant slipped through.
    """
    findings = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=2, cp=4))
    assert _codes(findings) == ["topology.effective_matches_declared"]
    only = findings[0]
    assert only.severity is Severity.OK
    assert only.details["fields_compared"] == 7


def test_mismatch_summary_blocking_count_is_the_number_of_fields_that_differ() -> None:
    """The summary's blocking count must be measured, not boilerplate.

    Three differing degrees must yield three per-field BLOCK findings and a
    summary reporting ``blocking == 3``. A summary emitted by the wrong
    branch — or with a hardcoded count — can satisfy one side of this fixture
    but never both, which is the shape of the gap the battery found.
    """
    findings = declared_vs_effective(
        _topo(dp=16, nodes=2),
        _topo(dp=4, tp=2, ep=2, nodes=2),
    )
    override_codes = [
        code for code in _codes(findings) if code.startswith("topology.effective_overrides_")
    ]
    assert override_codes == [
        "topology.effective_overrides_dp",
        "topology.effective_overrides_tp",
        "topology.effective_overrides_ep",
    ]
    summary = _by_code(findings, "topology.effective_comparison_summary")
    assert summary.details["blocking"] == len(override_codes) == 3
