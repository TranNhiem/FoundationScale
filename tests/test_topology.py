"""Tests for ``foundationscale.topology``.

Why this file exists
--------------------
The module under test encodes the measured record of 189 launcher scripts: the
``--ntasks-per-node=8`` vs ``--gpus-per-node=4`` Duplicate-GPU crash that surfaced
2m10s into a job instead of at construction; context parallelism dead in four
entrypoints while 44 launchers exported ``CP=${CP:-1}``; ``MASTER_ADDR=127.0.0.1``
verbatim in 27 multi-node launchers; one partition spelled two ways across a 188/4
file split with nothing ever comparing them; and an 8-node "cluster limit" that was
pure launcher habit while neighbours ran 18 nodes.

Each test here is written to be falsifiable: for every asserted behaviour there is a
concrete code change that would flip it red. Where a check asserts "X does not
happen", the same test first proves the detector can fire — the audit's own verifier
once reported ``all_identity: true`` on a corrupt artifact because ``all([])`` is
``True``, and these tests treat every empty result as that bug until proven
otherwise. Coverage assertions (``checks_run``, ``fields_compared``,
``files_scanned``) are asserted to be non-zero, because a silent clean result is the
exact failure mode the module is designed against.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from foundationscale.topology import (
    PROFILES,
    ClusterProfile,
    Finding,
    Severity,
    Topology,
    blocking,
    declared_vs_effective,
    partition_consistency,
    profile_by_name,
    render_findings,
)

# --------------------------------------------------------------------------------------
# Fixture builders.
# --------------------------------------------------------------------------------------


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


def _profile_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "test-cluster",
        "scheduler": "slurm",
        "partitions": ["gpu"],
        "node_pattern": r"[a-z0-9]+-[0-9]+",
        "gpus_per_node": 8,
        "nccl_socket_ifname": "eth0",
        "ib_hca_pattern": r"mlx5_*",
        "mnnvl_available": False,
        "container_runtime": "enroot",
        "container_image": "nvcr.io/nvidia/pytorch:25.01-py3",
        "filesystem_roots": ["/scratch"],
        "max_nodes": 64,
    }
    base.update(overrides)
    return base


def _profile(**overrides: object) -> ClusterProfile:
    return ClusterProfile.from_dict(_profile_dict(**overrides))


def _codes(findings: Sequence[Finding]) -> list[str]:
    return [f.code for f in findings]


def _by_code(findings: Sequence[Finding], code: str) -> Finding:
    for finding in findings:
        if finding.code == code:
            return finding
    raise AssertionError(f"no finding with code {code!r}; got {_codes(findings)}")


# --------------------------------------------------------------------------------------
# Topology.__post_init__ — crash at construction, not 2m10s into the job.
# --------------------------------------------------------------------------------------


def test_topology_rejects_degree_total_mismatch_and_shows_the_arithmetic() -> None:
    with pytest.raises(ValueError) as excinfo:
        _topo(dp=1, tp=2)  # 1 x 2 = 2, but the layout provides 1 x 8 = 8 GPUs
    msg = str(excinfo.value)
    # The error must show the arithmetic, not just complain — the whole point is that
    # the decomposition vs. layout comparison is visible to the human at launch.
    assert "dp(1) x tp(2) x pp(1) x ep(1) x cp(1) = 2" in msg
    assert "nodes(1) x gpus_per_node(8) = 8" in msg


def test_topology_rejects_nonpositive_degrees() -> None:
    with pytest.raises(ValueError, match="positive int"):
        _topo(tp=0)


def test_topology_rejects_bool_degrees() -> None:
    # bool is a subclass of int; without an explicit guard, dp=True would pass as dp=1
    # and silently break the product check's meaning.
    with pytest.raises(ValueError, match="positive int"):
        Topology(
            dp=True,  # type: ignore[arg-type]
            tp=1,
            pp=1,
            ep=1,
            cp=1,
            nodes=1,
            gpus_per_node=8,
        )


def test_tasks_per_node_defaults_to_gpus_per_node() -> None:
    assert _topo().tasks_per_node == 8


def test_tasks_gpu_mismatch_raises_at_construction_and_names_the_incident() -> None:
    # The measured crash: '#SBATCH --ntasks-per-node=8' against '--gpus-per-node=4'
    # produced NCCL "Duplicate GPU detected" 2m10s in. Construction must refuse first.
    with pytest.raises(ValueError) as excinfo:
        _topo(dp=4, nodes=1, gpus_per_node=4, tasks_per_node=8)
    msg = str(excinfo.value)
    assert "tasks_per_node=8 != gpus_per_node=4" in msg
    assert "Duplicate GPU" in msg


def test_explicit_matching_tasks_per_node_is_accepted() -> None:
    # The must-pass control for the test above: a launcher that sets both correctly
    # must build, or the check blocks everything and gets switched off.
    assert _topo(tasks_per_node=8).tasks_per_node == 8


def test_describe_shows_total_and_flags_pp_cp_scrutiny() -> None:
    out = _topo(dp=1, tp=8, pp=2, nodes=2).describe()
    assert "16 GPUs" in out
    assert "pp/cp > 1" in out  # 0/189 measured launchers ever exercised pp


# --------------------------------------------------------------------------------------
# Cluster profiles as data.
# --------------------------------------------------------------------------------------


def test_builtin_profiles_exist_and_carry_real_cluster_limits() -> None:
    assert set(PROFILES) == {"slurm-generic", "local-single-node"}
    # The estate's 8-node ceiling was a launcher habit; the profile table must state
    # the cluster's real limit as data, and 64 is not that habit.
    assert PROFILES["slurm-generic"].max_nodes == 64
    assert PROFILES["local-single-node"].max_nodes == 1


def test_profiles_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        PROFILES["evil"] = PROFILES["slurm-generic"]  # type: ignore[index]


def test_profile_by_name_unknown_key_says_how_to_add_a_cluster() -> None:
    with pytest.raises(KeyError) as excinfo:
        profile_by_name("no-such-cluster")
    msg = str(excinfo.value)
    assert "slurm-generic" in msg  # lists what IS known
    assert "_PROFILE_DATA" in msg  # says a new cluster is one dict of data


def test_from_dict_rejects_unknown_keys_so_a_typo_cannot_reset_a_field() -> None:
    data = _profile_dict()
    data["gpus_per_ndoe"] = data.pop("gpus_per_node")  # one-character typo
    with pytest.raises(ValueError) as excinfo:
        ClusterProfile.from_dict(data)
    msg = str(excinfo.value)
    assert "gpus_per_ndoe" in msg
    assert "silently" in msg


def test_from_dict_rejects_missing_required_keys() -> None:
    data = _profile_dict()
    data.pop("scheduler")
    with pytest.raises(ValueError, match="missing required keys"):
        ClusterProfile.from_dict(data)


def test_profile_to_dict_round_trips() -> None:
    original = _profile()
    assert ClusterProfile.from_dict(original.to_dict()) == original


def test_from_json_accepts_a_document_or_a_path(tmp_path: Path) -> None:
    from_text = ClusterProfile.from_json(json.dumps(_profile_dict()))
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(_profile_dict()), encoding="utf-8")
    from_path = ClusterProfile.from_json(path)
    assert from_text == from_path
    assert from_path.name == "test-cluster"


def test_profile_rejects_unpinned_image_and_runtime_without_image() -> None:
    with pytest.raises(ValueError, match="pinned"):
        ClusterProfile.from_dict(_profile_dict(container_image=""))
    with pytest.raises(ValueError, match="runtime is 'none'"):
        ClusterProfile.from_dict(_profile_dict(container_runtime="none"))


def test_profile_rejects_invalid_node_pattern_regex() -> None:
    with pytest.raises(ValueError, match="valid regex"):
        ClusterProfile.from_dict(_profile_dict(node_pattern="["))


def test_profile_rejects_nonpositive_gpu_counts_and_node_ceilings() -> None:
    with pytest.raises(ValueError):
        ClusterProfile.from_dict(_profile_dict(gpus_per_node=0))
    with pytest.raises(ValueError):
        ClusterProfile.from_dict(_profile_dict(max_nodes=0))


# --------------------------------------------------------------------------------------
# validate_against — node shape, and the always-on coverage summary.
# --------------------------------------------------------------------------------------


def test_validate_against_always_appends_a_coverage_summary() -> None:
    clean = _topo().validate_against(_profile())
    dirty = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="127.0.0.1")
    # Both the clean and the defective path must end with the summary, and the
    # summary's counts must match the findings it summarises. The dirty path is the
    # anti-vacuity control: if the summary were hardcoded zero, this fails there.
    for findings in (clean, dirty):
        assert findings, "validators must never return an empty list"
        summary = findings[-1]
        assert summary.code == "topology.validate_summary"
        assert summary.severity is Severity.OK
        checks_run = summary.details["checks_run"]
        assert isinstance(checks_run, int)
        assert checks_run > 0
        body = findings[:-1]
        assert summary.details["blocking"] == sum(f.severity is Severity.BLOCK for f in body)
        assert summary.details["warnings"] == sum(f.severity is Severity.WARN for f in body)


def test_clean_single_node_topology_runs_exactly_the_expected_checks() -> None:
    summary = _topo().validate_against(_profile())[-1]
    # gpus-per-node, nodes-limit, tp-boundary, and the pp/cp pair: five. This pins
    # the count so a silently skipped check group cannot keep the suite green.
    assert summary.details["checks_run"] == 5


def test_gpus_per_node_exceeding_profile_blocks_and_fitting_it_clears() -> None:
    findings = _topo(dp=16, nodes=1, gpus_per_node=16).validate_against(_profile())
    block = _by_code(findings, "topology.gpus_per_node_exceeds_profile")
    assert block.severity is Severity.BLOCK
    fits = _topo(dp=16, nodes=1, gpus_per_node=16).validate_against(_profile(gpus_per_node=16))
    assert "topology.gpus_per_node_exceeds_profile" not in _codes(fits)


def test_node_limit_is_profile_data_not_inherited_launcher_habit() -> None:
    # Reproduces the weld: 18 GPU-less? No — 9 nodes against an 8-node ceiling blocks.
    findings = _topo(dp=72, nodes=9).validate_against(_profile(max_nodes=8))
    block = _by_code(findings, "topology.nodes_exceed_profile_limit")
    assert block.severity is Severity.BLOCK
    assert block.details["requested"] == 9
    # ...but other users on the same cluster ran 18 nodes. With honest profile data the
    # same shape must pass: the limit lives in the table, not in a copied script.
    wide = _topo(dp=144, nodes=18).validate_against(
        _profile(max_nodes=64), master_addr="head.cluster.local"
    )
    assert "topology.nodes_exceed_profile_limit" not in _codes(wide)


# --------------------------------------------------------------------------------------
# TP crossing the node boundary — severity is about interconnect, not cluster shape.
# --------------------------------------------------------------------------------------


def test_tp_crossing_blocks_without_mnnvl_and_only_warns_with_it() -> None:
    topo = _topo(dp=1, tp=16, nodes=2)  # 16 GPUs; tp spans both nodes
    plain = _by_code(
        topo.validate_against(_profile(), master_addr="head.cluster.local"),
        "topology.tp_crosses_node_boundary",
    )
    assert plain.severity is Severity.BLOCK
    mnnvl = _by_code(
        topo.validate_against(_profile(mnnvl_available=True), master_addr="head.cluster.local"),
        "topology.tp_crosses_node_boundary",
    )
    assert mnnvl.severity is Severity.WARN
    assert "MNNVL" in mnnvl.message  # the check encodes the reason, not one cluster
    # Must-pass control: a TP group that fits inside a node never produces the finding.
    within = _topo(dp=2, tp=4).validate_against(_profile())
    assert "topology.tp_crosses_node_boundary" not in _codes(within)


# --------------------------------------------------------------------------------------
# Expert parallelism.
# --------------------------------------------------------------------------------------


def test_ep1_on_moe_warns_and_ep2_is_the_fix() -> None:
    findings = _topo(dp=8).validate_against(_profile(), num_experts=8)  # ep=1
    warn = _by_code(findings, "topology.ep1_replicates_experts")
    assert warn.severity is Severity.WARN
    sharded = _topo(dp=4, ep=2).validate_against(_profile(), num_experts=8)
    assert "topology.ep1_replicates_experts" not in _codes(sharded)


def test_ep_uneven_expert_shard_blocks_and_even_shard_clears() -> None:
    findings = _topo(dp=4, ep=2).validate_against(_profile(), num_experts=7)
    assert _by_code(findings, "topology.ep_uneven_expert_shard").severity is Severity.BLOCK
    even = _topo(dp=4, ep=2).validate_against(_profile(), num_experts=8)
    assert "topology.ep_uneven_expert_shard" not in _codes(even)


# --------------------------------------------------------------------------------------
# Rendezvous — the two opposite ways MASTER_ADDR was wrong.
# --------------------------------------------------------------------------------------


def test_multi_node_without_master_addr_warns_unrecorded() -> None:
    findings = _topo(dp=16, nodes=2).validate_against(_profile())
    warn = _by_code(findings, "topology.master_addr_unrecorded")
    assert warn.severity is Severity.WARN


def test_loopback_master_addr_blocks_multi_node_but_not_single_node() -> None:
    # The verbatim incident: MASTER_ADDR=127.0.0.1 in 27 launchers, every node
    # rendezvousing with itself.
    findings = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="127.0.0.1")
    block = _by_code(findings, "topology.master_addr_loopback")
    assert block.severity is Severity.BLOCK
    single = _topo().validate_against(_profile(), master_addr="127.0.0.1")
    assert "topology.master_addr_loopback" not in _codes(single)


def test_bracketed_ipv6_loopback_with_port_is_still_detected() -> None:
    findings = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="[::1]:29400")
    assert "topology.master_addr_loopback" in _codes(findings)


def test_hardcoded_node_name_blocks_even_with_a_port_but_resolved_host_clears() -> None:
    # The opposite defect: MASTER_ADDR welded to one concrete machine (22 launchers).
    findings = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="gpu-14:6000")
    block = _by_code(findings, "topology.master_addr_hardcoded_node")
    assert block.severity is Severity.BLOCK
    assert block.details["host"] == "gpu-14"
    resolved = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="head.cluster.local")
    assert "topology.master_addr_hardcoded_node" not in _codes(resolved)


# --------------------------------------------------------------------------------------
# Requested vs. honoured pp/cp.
# --------------------------------------------------------------------------------------


def test_runtime_override_of_pp_blocks_and_matching_override_clears() -> None:
    topo = _topo(dp=1, tp=8, pp=2, nodes=2)  # 16 GPUs, 2 pipeline stages requested
    findings = topo.validate_against(
        _profile(), master_addr="head.cluster.local", runtime_overrides={"pp": 1}
    )
    block = _by_code(findings, "topology.runtime_overrides_pp")
    assert block.severity is Severity.BLOCK
    assert block.details["requested"] == 2
    assert block.details["forced"] == 1
    honest = topo.validate_against(
        _profile(), master_addr="head.cluster.local", runtime_overrides={"pp": 2}
    )
    assert "topology.runtime_overrides_pp" not in _codes(honest)


def test_cp_gt1_without_evidence_warns_and_matching_override_clears() -> None:
    findings = _topo(dp=2, cp=4).validate_against(_profile())
    assert _by_code(findings, "topology.cp_unverified").severity is Severity.WARN
    honoured = _topo(dp=2, cp=4).validate_against(_profile(), runtime_overrides={"cp": 4})
    assert "topology.cp_unverified" not in _codes(honoured)


# --------------------------------------------------------------------------------------
# declared_vs_effective — the check that catches hardcoded entrypoints.
# --------------------------------------------------------------------------------------


def test_match_reports_a_nonzero_fields_compared() -> None:
    findings = declared_vs_effective(_topo(), _topo())
    assert len(findings) == 1
    only = findings[0]
    assert only.severity is Severity.OK
    assert only.code == "topology.effective_matches_declared"
    # A silent clean result is the failure being designed out; the count must be real:
    # five degrees + nodes + gpus_per_node.
    assert only.details["fields_compared"] == 7


def test_mismatch_blocks_per_differing_degree_with_declared_and_effective() -> None:
    # The measured shape: launcher set CP=4, entrypoint hardcoded cp=1 (and dp shifts
    # with it). Both degrees must be named, each with both values.
    findings = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=8, cp=1))
    codes = _codes(findings)
    assert "topology.effective_overrides_cp" in codes
    assert "topology.effective_overrides_dp" in codes
    cp_finding = _by_code(findings, "topology.effective_overrides_cp")
    assert cp_finding.severity is Severity.BLOCK
    assert cp_finding.details["declared"] == 4
    assert cp_finding.details["effective"] == 1


def test_mismatch_keeps_detector_firing_while_identity_case_stays_clean() -> None:
    # Both fixtures in one place: the blocks-everything and blocks-nothing failure
    # modes are guarded simultaneously.
    broken = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=8, cp=1))
    assert any(f.severity is Severity.BLOCK for f in broken)
    identical = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=2, cp=4))
    assert all(f.severity is Severity.OK for f in identical)


def test_mismatch_report_also_carries_comparison_coverage() -> None:
    findings = declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=8, cp=1))
    assert any(f.severity is Severity.BLOCK for f in findings)  # detector is live

    # This assertion used to be `any(f.details.get("fields_compared") == 7 ...)`, and a
    # mutation battery proved it did not cover the rule it names. Two reasons, both worth
    # keeping in view: `.get()` turns a missing key into None, which contributes a quiet
    # False instead of failing loudly; and `any` pinned "some finding, somewhere, carries
    # the number 7" without ever saying WHICH. Disabling the summary branch re-routed the
    # mismatch path into the else-branch, which appends an OK "effective matches declared"
    # finding carrying exactly that key and value — so the suite stayed green over a
    # report that listed per-field BLOCKs and declared the config honoured in one breath.
    # Name the finding, index the key directly.
    summary = [f for f in findings if f.code == "topology.effective_comparison_summary"]
    assert len(summary) == 1, (
        "a mismatch report without its own coverage summary cannot be told apart "
        f"from a partial comparison; got codes {[f.code for f in findings]}"
    )
    assert summary[0].details["fields_compared"] == 7


# --------------------------------------------------------------------------------------
# partition_consistency — all([]) at module level.
# --------------------------------------------------------------------------------------

_VARIANT_CORPUS: dict[str, str] = {
    "train/shard_a.sh": "#SBATCH --partition=gpu-a100\n",
    "train/shard_b.sh": "#SBATCH --partition=gpu-a100\n",
    "train/shard_c.sh": "sbatch -p gpu-a100 train.py\n",
    "legacy/old.sh": "srun --partition=gpu_a100 python train.py\n",
}


def test_empty_corpus_blocks_instead_of_passing_vacuously() -> None:
    finding = partition_consistency({})
    assert finding.severity is Severity.BLOCK
    assert finding.code == "topology.partition_scan_empty"
    assert "all([])" in finding.message  # names the incident class it refuses to repeat


def test_zero_partitions_in_nonempty_corpus_blocks() -> None:
    finding = partition_consistency(
        {"a.sh": "echo hello\n", "b.sh": "python train.py\n", "c.sh": "# comment\n"}
    )
    assert finding.severity is Severity.BLOCK
    assert finding.code == "topology.partition_not_found"
    assert finding.details["files_scanned"] == 3
    assert finding.details["files_with_partition"] == 0


def test_spelling_variants_block_with_per_spelling_file_counts() -> None:
    # A miniature of the measured 188/4 split: one partition, two spellings.
    finding = partition_consistency(_VARIANT_CORPUS)
    assert finding.severity is Severity.BLOCK
    assert finding.code == "topology.partition_spelling_variants"
    # Keyed by the NORMALISED name: the corpus spells it gpu-a100 and gpu_a100, and the
    # whole point of the check is that those collapse to one group rather than reading as
    # two unrelated partitions.
    variants = finding.details["variants"]["gpua100"]
    assert variants["gpu-a100"]["files"] == 3
    assert variants["gpu_a100"]["files"] == 1
    assert finding.details["files_scanned"] == 4
    # The check must refuse to resolve the split by majority: both spellings survive.
    assert set(variants) == {"gpu-a100", "gpu_a100"}


def test_extraction_accepts_all_four_declaration_forms() -> None:
    # Positive controls on the extractor itself: '#SBATCH --partition=', 'sbatch -p',
    # bare '--partition', and 'partition = "x"'. If any form goes blind, the count
    # drops below 4 and this fails.
    corpus = {
        "a.sh": "#SBATCH --partition=gpu-h100\n",
        "b.sh": "sbatch -p gpu-h100 job\n",
        "c.sh": "python run.py --partition gpu-h100\n",
        "d.sh": 'partition = "gpu-h100"\n',
    }
    finding = partition_consistency(corpus)
    assert finding.severity is Severity.OK
    assert finding.code == "topology.partition_consistent"
    assert finding.details["files_with_partition"] == 4
    assert finding.details["variants"] == 0


def test_bare_dash_p_off_scheduler_lines_is_not_a_partition() -> None:
    # '-p' is overloaded (mkdir, cp, tee); if the sbatch/srun gating were dropped this
    # corpus would manufacture 'partitions' out of flag noise and this test goes red.
    corpus = {
        "ops.sh": "mkdir -p /data/out\n"
        "cp -p weights.bin backup/weights.bin\n"
        "rsync -P logs/ archive/\n"
    }
    finding = partition_consistency(corpus)
    assert finding.code == "topology.partition_not_found"
    assert finding.severity is Severity.BLOCK


# --------------------------------------------------------------------------------------
# Cross-cutting: controls named, blockers filtered, empty never rendered as clean.
# --------------------------------------------------------------------------------------


def test_every_finding_everywhere_names_its_positive_control() -> None:
    scenarios: list[Sequence[Finding]] = [
        _topo().validate_against(_profile()),
        _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="127.0.0.1"),
        _topo(dp=4, ep=2).validate_against(_profile(), num_experts=7),
        declared_vs_effective(_topo(), _topo()),
        declared_vs_effective(_topo(dp=2, cp=4), _topo(dp=8, cp=1)),
        [partition_consistency({})],
        [partition_consistency({"a.sh": "echo hi\n"})],
        [partition_consistency(_VARIANT_CORPUS)],
        [partition_consistency({"a.sh": "#SBATCH --partition=gpu\n"})],
    ]
    total = 0
    for findings in scenarios:
        assert findings, "validators must never return an empty list"
        for finding in findings:
            total += 1
            assert finding.control.strip(), (
                f"{finding.code} cannot say what proves its detector fires — "
                f"an unnamed control is the all([]) incident one level up"
            )
    assert total > 0  # this loop itself must not be vacuous


def test_blocking_filters_to_block_severity_only_and_loses_none() -> None:
    findings = _topo(dp=8).validate_against(_profile(), num_experts=8)  # one WARN
    assert not blocking(findings)
    mixed = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="127.0.0.1")
    blocked = blocking(mixed)
    assert blocked  # non-vacuous: there really is a BLOCK in this fixture
    assert all(f.severity is Severity.BLOCK for f in blocked)
    assert len(blocked) == sum(f.severity is Severity.BLOCK for f in mixed)


def test_render_findings_never_renders_empty_as_clean() -> None:
    assert "suspicious" in render_findings([])


def test_render_findings_lists_every_code() -> None:
    findings = _topo(dp=16, nodes=2).validate_against(_profile(), master_addr="127.0.0.1")
    out = render_findings(findings)
    for finding in findings:
        assert finding.code in out


def test_mapping_import_kept_honest_by_profile_api() -> None:
    # from_dict advertises Mapping; passing a real Mapping (not just dict) must work.
    data: Mapping[str, object] = _profile_dict()
    assert ClusterProfile.from_dict(data).name == "test-cluster"
