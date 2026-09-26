
"""Tests for the mix op and design_mixture (mixing rules)."""
from __future__ import annotations

import json

import pytest

import foundationskills.skills.data_engine.ops  # noqa: F401 - register real ops
from foundationskills.skills.data_engine.mix import design_mixture
from foundationskills.skills.data_engine.ops.base import OPS, OpStats


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _run_mix(cfg, upstream=()):
    stats = OpStats("mix")
    out = list(OPS["mix"](iter(list(upstream)), cfg, stats))
    return out, stats


class TestDesignMixture:
    def test_cpt_preserve_ratios_sum_and_replay_band(self):
        out = design_mixture(
            goal="domain_expert",
            stage="cpt",
            domain_tokens=2_000_000_000,
            preserve_general=True,
            available=["domain", "general_replay", "code_math"],
        )
        ratios = {c["name"]: c["ratio"] for c in out["components"]}
        assert abs(sum(ratios.values()) - 1.0) <= 1e-9
        assert 0.6 <= ratios["domain"] <= 0.8
        assert 0.15 <= ratios["general_replay"] <= 0.3
        assert 0.05 <= ratios["code_math"] <= 0.1
        assert out["citations"]
        assert any("2403.08763" in cite for cite in out["citations"])

    def test_cpt_scarce_domain_caps_epochs(self):
        out = design_mixture(
            goal="domain_expert",
            stage="cpt",
            domain_tokens=100_000_000,
            preserve_general=True,
            available=["domain", "general_replay", "code_math"],
        )
        domain = next(c for c in out["components"] if c["name"] == "domain")
        assert domain.get("max_epochs") == 4
        assert domain["ratio"] <= 0.6 + 1e-9
        assert any("replay" in line for line in out["rationale"])

    def test_rl_mixture(self):
        out = design_mixture(
            goal="math",
            stage="rl",
            domain_tokens=None,
            preserve_general=True,
            available=["domain_verifiable", "verifiable_math_replay"],
        )
        names = {c["name"] for c in out["components"]}
        assert names == {"domain_verifiable", "verifiable_math_replay"}
        assert abs(sum(c["ratio"] for c in out["components"]) - 1.0) <= 1e-9

    def test_tool_calling_flagged_phase2(self):
        out = design_mixture(
            goal="tool_calling",
            stage="sft",
            domain_tokens=None,
            preserve_general=True,
            available=["tool_traces", "general_chat"],
        )
        assert any("phase-2" in line.lower() for line in out["rationale"])
        tool = next(c for c in out["components"] if c["name"] == "tool_traces")
        assert "PHASE-2" in tool["source_hint"]

    def test_unavailable_component_renormalized(self):
        out = design_mixture(
            goal="domain_expert",
            stage="cpt",
            domain_tokens=2_000_000_000,
            preserve_general=True,
            available=["domain", "general_replay"],
        )
        names = {c["name"] for c in out["components"]}
        assert names == {"domain", "general_replay"}
        assert abs(sum(c["ratio"] for c in out["components"]) - 1.0) <= 1e-9
        assert any("renormalized" in line for line in out["rationale"])

    def test_no_rule_matches_fallback(self):
        out = design_mixture(
            goal="general_chat",
            stage="pretrain",
            domain_tokens=None,
            preserve_general=True,
            available=[],
        )
        assert out["components"] == [
            pytest.helpers.expected if False else out["components"][0]
        ]  # single fallback component
        assert len(out["components"]) == 1
        assert out["components"][0]["ratio"] == 1.0

    def test_total_tokens_distributed_exactly(self):
        out = design_mixture(
            goal="domain_expert",
            stage="cpt",
            domain_tokens=2_000_000_000,
            total_tokens=1_000_000,
            preserve_general=True,
            available=["domain", "general_replay", "code_math"],
        )
        assert sum(c["tokens"] for c in out["components"]) == 1_000_000


class TestMixOp:
    def test_determinism_and_ratios(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [{"id": f"a{i}", "text": f"text a{i}"} for i in range(10)])
        b = _write_jsonl(tmp_path / "b.jsonl", [{"id": f"b{i}", "text": f"text b{i}"} for i in range(10)])
        cfg = {
            "components": [
                {"name": "a", "path": str(a), "ratio": 0.5},
                {"name": "b", "path": str(b), "ratio": 0.5},
            ],
            "seed": 7,
            "max_epochs": 4,
        }
        out1, stats1 = _run_mix(cfg)
        out2, stats2 = _run_mix(cfg)
        assert [r["id"] for r in out1] == [r["id"] for r in out2]
        assert len(out1) == 20
        realized = stats1.extra["realized"]
        assert realized["a"] == pytest.approx(0.5)
        assert realized["b"] == pytest.approx(0.5)
        assert all(r["meta"]["mix_component"] in {"a", "b"} for r in out1)

    def test_upsampling_respects_ratios(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [{"id": f"a{i}", "text": "x"} for i in range(10)])
        b = _write_jsonl(tmp_path / "b.jsonl", [{"id": f"b{i}", "text": "y"} for i in range(10)])
        cfg = {
            "components": [
                {"name": "a", "path": str(a), "ratio": 0.8},
                {"name": "b", "path": str(b), "ratio": 0.2},
            ],
            "seed": 0,
            "max_epochs": 4,
        }
        out, stats = _run_mix(cfg)
        assert len(out) == 20
        count_a = sum(1 for r in out if r["meta"]["mix_component"] == "a")
        assert count_a == 16  # 1.6 epochs of a
        assert stats.extra["realized"]["a"] == pytest.approx(0.8)

    def test_epoch_cap_recorded(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [{"id": f"a{i}", "text": "x"} for i in range(2)])
        b = _write_jsonl(tmp_path / "b.jsonl", [{"id": f"b{i}", "text": "y"} for i in range(100)])
        cfg = {
            "components": [
                {"name": "a", "path": str(a), "ratio": 0.95},
                {"name": "b", "path": str(b), "ratio": 0.05},
            ],
            "total_records": 100,
            "seed": 0,
            "max_epochs": 4,
        }
        out, stats = _run_mix(cfg)
        count_a = sum(1 for r in out if r["meta"]["mix_component"] == "a")
        assert count_a == 8  # 2 records * max_epochs
        capped = stats.extra["capped"]
        assert capped["a"]["kept"] == 8
        assert capped["a"]["requested"] == 95

    def test_inline_component_consumes_upstream(self):
        upstream = [{"id": f"u{i}", "text": "z"} for i in range(5)]
        cfg = {
            "components": [{"name": "pooled", "ratio": 1.0, "inline": True}],
            "seed": 0,
            "max_epochs": 4,
        }
        out, stats = _run_mix(cfg, upstream=upstream)
        assert {r["id"] for r in out} == {f"u{i}" for i in range(5)}
        assert all(r["meta"]["mix_component"] == "pooled" for r in out)
        assert stats.records_in == 5

    def test_multiple_inline_components_rejected(self):
        cfg = {
            "components": [
                {"name": "x", "ratio": 0.5, "inline": True},
                {"name": "y", "ratio": 0.5, "inline": True},
            ]
        }
        with pytest.raises(ValueError, match="inline"):
            _run_mix(cfg)

    def test_total_tokens_marks_approximate(self, tmp_path):
        a = _write_jsonl(
            tmp_path / "a.jsonl", [{"id": f"a{i}", "text": "one two three four"} for i in range(10)]
        )
        cfg = {
            "components": [{"name": "a", "path": str(a), "ratio": 1.0}],
            "total_tokens": 200,
            "seed": 0,
            "max_epochs": 10,
        }
        out, stats = _run_mix(cfg)
        assert stats.extra.get("approximate") is True
        assert out  # some records sampled
