
"""Tests for the public dataset catalog and discover()."""
from __future__ import annotations

from foundationskills.skills.data_engine.catalog import discover, load_catalog


class TestCatalogFile:
    def test_loads_and_is_populated(self):
        entries = load_catalog()
        assert len(entries) >= 25
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))
        for entry in entries:
            assert entry.get("id") and isinstance(entry["id"], str)
            assert entry.get("formats"), entry["id"]
            assert isinstance(entry.get("license"), str) and entry["license"]
            assert entry.get("languages"), entry["id"]
            assert entry.get("notes") is not None
            assert "hf_id" in entry
            if entry["hf_id"] is None:
                assert "gap" in entry["notes"].lower() or "absence" in entry["notes"].lower() or "no large open" in entry["notes"].lower()

    def test_known_ids_present(self):
        ids = {e["id"] for e in load_catalog()}
        assert "fineweb" in ids
        assert "openai" in next(e["hf_id"] for e in load_catalog() if e["id"] == "gsm8k")


class TestDiscover:
    def test_format_filter_and_reasons(self):
        out = discover(
            goal="domain_expert",
            stage="pretrain",
            formats=["pretrain"],
            domain=None,
            languages=None,
            k=8,
        )
        assert out
        assert all("pretrain" in e["formats"] for e in out)
        assert all(e["reasons"] for e in out)
        assert len(out) <= 8

    def test_goal_match_ranks_reasoning_rl_first(self):
        out = discover(goal="math", stage="rl", formats=["rl"], domain="math", languages=None, k=5)
        assert out
        assert out[0]["score"] >= out[-1]["score"]
        assert any("goal match" in reason for reason in out[0]["reasons"])

    def test_license_allowlist_filters(self):
        allow = ["apache-2.0", "mit"]
        out = discover(goal=None, stage="sft", formats=["sft"], domain=None, languages=None, license_ok=allow, k=30)
        assert out
        for entry in out:
            assert entry["license"] in allow or entry["license"] == "verify"

    def test_language_filter(self):
        out = discover(goal=None, stage=None, formats=["sft"], domain=None, languages=["fr"], k=10)
        for entry in out:
            assert "fr" in entry["languages"] or "multilingual" in entry["languages"]

    def test_tool_calling_discoverable(self):
        out = discover(goal="tool_calling", stage="sft", formats=["sft"], domain="tools", languages=None, k=5)
        assert out
        assert out[0]["domains"].count("tools") >= 1 or any("tools" in e["domains"] for e in out)
