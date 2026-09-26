
"""Tests for the readiness report: every rule has a must-fire fixture."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from foundationskills.skills.data_engine.report import build_readiness, render_markdown

SFT_MESSAGES = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]


def _op(name, *, records_in=0, records_out=0, dropped=None, extra=None, backend="builtin"):
    return {
        "name": name,
        "records_in": records_in,
        "records_out": records_out,
        "dropped": dict(dropped or {}),
        "modified": {},
        "extra": dict(extra or {}),
        "backend": backend,
    }


@pytest.fixture
def base_case(tmp_path):
    """A clean, fully-measured sft case whose verdict is PASS."""
    shard = tmp_path / "shard-00000.jsonl"
    record = {"id": "r0", "messages": SFT_MESSAGES, "text": "<rendered>", "meta": {"domain": "general"}}
    shard.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = {
        "format": "sft",
        "shards": [{"path": str(shard), "sha256": "0" * 64, "records": 1}],
        "schema": {"columns": ["id", "messages", "text", "meta"]},
        "num_records": 3,
        "num_tokens": 5000,
        "tokenizer": "tok",
        "chat_template_family": "gemma4",
        "fs_columns": {"text_column": "text", "image_column": None, "gold_key": None},
        "name": "base-dataset",
    }
    stats = [
        _op("ingest", records_in=3, records_out=3),
        _op("clean", records_in=3, records_out=3, extra={"pii_remaining": 0}),
        _op("dedup", records_in=10, records_out=8, dropped={"exact": 2}),
        _op("decontam", records_in=8, records_out=8, extra={"hits": 0, "action": "remove"}),
        _op("format", records_in=8, records_out=8, extra={"sft_loss_scope": "full_sequence", "template_source": "gemma4"}),
        _op(
            "tokenize",
            records_in=8,
            records_out=8,
            backend="transformers",
            extra={
                "num_tokens": 5000,
                "length_hist": {"128": 3},
                "truncation_rate": 0.0,
                "domain_counts": {"general": 3},
                "p50": 100,
                "p90": 200,
                "p99": 250,
                "max": 300,
            },
        ),
    ]
    requirements = {
        "target_format": "sft",
        "min_tokens": 1000,
        "min_records": 1,
        "max_truncation_rate": 0.02,
        "require_dedup": True,
        "require_decontam": True,
        "max_pii_remaining": 0,
        "chat_template_family": "gemma4",
    }
    return dataset, stats, requirements


def _check(report, rule_id):
    for check in report["checks"]:
        if check["rule_id"] == rule_id:
            return check
    raise AssertionError(f"check {rule_id} missing from report")


def test_clean_case_passes(base_case):
    dataset, stats, requirements = base_case
    report = build_readiness(dataset, stats, requirements)
    assert report["verdict"] == "PASS"
    assert all(c["passed"] is True for c in report["checks"])
    assert [c["rule_id"] for c in report["checks"]] == [f"DE-RDY-{i:03d}" for i in range(1, 11)]
    # stats section filled from op stats
    s = report["stats"]
    assert s["num_records"] == 3
    assert s["num_tokens"] == 5000
    assert s["length_hist"] == {"128": 3}
    assert s["domain_breakdown"] == {"general": 3}
    assert s["dedup_rate"] == pytest.approx(0.2)
    assert s["decontam_hits"] == 0
    assert s["pii_remaining"] == 0
    assert s["truncation_rate"] == 0.0


def test_de_rdy_001_fires_on_format_mismatch(base_case):
    dataset, stats, requirements = base_case
    dataset = dict(dataset, format="pretrain")
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-001")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_002_flags_insufficient_tokens(base_case):
    dataset, stats, requirements = base_case
    requirements = dict(requirements, min_tokens=99999)
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-002")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_002_is_null_when_counts_are_approximate(base_case):
    dataset, stats, requirements = base_case
    stats[5] = dict(stats[5], backend="approx", extra={**stats[5]["extra"], "approximate": True})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-002")["passed"] is None
    assert report["stats"]["num_tokens"] == 5000
    assert report["verdict"] == "UNMEASURED"


def test_de_rdy_002_is_null_when_num_tokens_unknown(base_case):
    dataset, stats, requirements = base_case
    dataset = dict(dataset, num_tokens=None)
    stats = [s for s in stats if s["name"] != "tokenize"]
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-002")["passed"] is None
    # without tokenize, truncation is unmeasured too
    assert _check(report, "DE-RDY-007")["passed"] is None
    assert report["verdict"] == "UNMEASURED"


def test_de_rdy_003_fires_on_too_few_records(base_case):
    dataset, stats, requirements = base_case
    requirements = dict(requirements, min_records=100)
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-003")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_004_fires_when_dedup_required_but_absent(base_case):
    dataset, stats, requirements = base_case
    stats = [s for s in stats if s["name"] != "dedup"]
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-004")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_004_ok_when_dedup_not_required(base_case):
    dataset, stats, requirements = base_case
    stats = [s for s in stats if s["name"] != "dedup"]
    requirements = dict(requirements, require_dedup=False)
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-004")["passed"] is True
    assert report["stats"]["dedup_rate"] is None


def test_de_rdy_005_is_null_when_benchmarks_unmeasured(base_case):
    dataset, stats, requirements = base_case
    stats[3] = dict(stats[3], extra={**stats[3]["extra"], "unmeasured_benchmarks": ["gsm8k"]})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-005")["passed"] is None
    assert report["verdict"] == "UNMEASURED"


def test_de_rdy_005_fires_when_hits_remain_flagged(base_case):
    dataset, stats, requirements = base_case
    stats[3] = dict(stats[3], extra={"hits": 2, "action": "flag"})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-005")["passed"] is False
    assert report["stats"]["decontam_hits"] == 2
    assert report["verdict"] == "RED"


def test_de_rdy_005_ok_when_hits_removed(base_case):
    dataset, stats, requirements = base_case
    stats[3] = dict(stats[3], extra={"hits": 4, "action": "remove"})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-005")["passed"] is True


def test_de_rdy_006_is_null_without_clean_op(base_case):
    dataset, stats, requirements = base_case
    stats = [s for s in stats if s["name"] != "clean"]
    report = build_readiness(dataset, stats, requirements)
    check = _check(report, "DE-RDY-006")
    assert check["passed"] is None
    assert "unmeasured" in check["detail"].lower()
    assert report["verdict"] == "UNMEASURED"


def test_de_rdy_006_fires_when_pii_above_max(base_case):
    dataset, stats, requirements = base_case
    stats[1] = dict(stats[1], extra={"pii_remaining": 3})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-006")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_007_fires_on_high_truncation(base_case):
    dataset, stats, requirements = base_case
    stats[5] = dict(stats[5], extra={**stats[5]["extra"], "truncation_rate": 0.5})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-007")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_008_fires_on_template_fallback(base_case):
    dataset, stats, requirements = base_case
    stats[4] = dict(stats[4], extra={**stats[4]["extra"], "template_fallback": True})
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-008")["passed"] is False
    assert report["verdict"] == "RED"


def test_de_rdy_008_fires_on_family_mismatch_and_null_when_unrecorded(base_case):
    dataset, stats, requirements = base_case
    report = build_readiness(dict(dataset, chat_template_family="qwen3.5"), stats, requirements)
    assert _check(report, "DE-RDY-008")["passed"] is False
    report2 = build_readiness(dict(dataset, chat_template_family=None), stats, requirements)
    assert _check(report2, "DE-RDY-008")["passed"] is None


def test_de_rdy_009_fires_when_first_record_lacks_fs_columns(base_case, tmp_path):
    dataset, stats, requirements = base_case
    bad_shard = tmp_path / "shard-bad.jsonl"
    bad_shard.write_text(json.dumps({"id": "x", "text": "no messages here"}) + "\n", encoding="utf-8")
    dataset = dict(dataset, shards=[{"path": str(bad_shard), "sha256": "0" * 64, "records": 1}])
    report = build_readiness(dataset, stats, requirements)
    check = _check(report, "DE-RDY-009")
    assert check["passed"] is False
    assert "messages" in check["detail"]
    assert report["verdict"] == "RED"


def test_de_rdy_009_is_null_when_shard_unreadable(base_case):
    dataset, stats, requirements = base_case
    dataset = dict(dataset, shards=[{"path": "/nonexistent/shard-zzz.jsonl", "sha256": "0" * 64, "records": 2}])
    report = build_readiness(dataset, stats, requirements)
    assert _check(report, "DE-RDY-009")["passed"] is None
    assert report["verdict"] == "UNMEASURED"


def test_de_rdy_009_checks_rl_gold_column(base_case, tmp_path):
    dataset, stats, requirements = base_case
    rl_shard = tmp_path / "shard-rl.jsonl"
    rl_record = {"id": "r", "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}], "meta": {}}
    rl_shard.write_text(json.dumps(rl_record) + "\n", encoding="utf-8")
    rl_dataset = dict(
        dataset,
        format="rl",
        shards=[{"path": str(rl_shard), "sha256": "0" * 64, "records": 1}],
        fs_columns={"text_column": None, "image_column": None, "gold_key": "answer"},
        chat_template_family=None,
    )
    rl_requirements = dict(requirements, target_format="rl", chat_template_family=None)
    report = build_readiness(rl_dataset, stats, rl_requirements)
    # gold_key "answer" declared but absent from the record -> schema failure
    assert _check(report, "DE-RDY-009")["passed"] is False


def test_de_rdy_010_is_present_informational_disclosure(base_case):
    dataset, stats, requirements = base_case
    report = build_readiness(dataset, stats, requirements)
    check = _check(report, "DE-RDY-010")
    assert check["passed"] is True
    assert "full" in check["detail"].lower()
    assert "masking" in check["detail"].lower()


def test_de_rdy_010_absent_for_non_conversational(tmp_path, base_case):
    dataset, stats, requirements = base_case
    pre_shard = tmp_path / "shard-pre.jsonl"
    pre_shard.write_text(json.dumps({"id": "p", "text": "t", "meta": {}}) + "\n", encoding="utf-8")
    pre_dataset = dict(
        dataset,
        format="pretrain",
        shards=[{"path": str(pre_shard), "sha256": "0" * 64, "records": 1}],
        chat_template_family=None,
    )
    pre_requirements = dict(requirements, target_format="pretrain", chat_template_family=None)
    report = build_readiness(pre_dataset, stats, pre_requirements)
    assert "DE-RDY-010" not in {c["rule_id"] for c in report["checks"]}
    # DE-RDY-008 has no applicable template -> unmeasured by doctrine
    assert _check(report, "DE-RDY-008")["passed"] is None
    assert report["verdict"] == "UNMEASURED"


def test_markdown_renders_verdict_rules_and_statuses(base_case):
    dataset, stats, requirements = base_case
    requirements = dict(requirements, min_tokens=999999)  # make DE-RDY-002 fail
    report = build_readiness(dataset, stats, requirements)
    md = render_markdown(report)
    assert "**Verdict:** RED" in md
    assert "| DE-RDY-001 | PASS |" in md
    assert "| DE-RDY-002 | FAIL |" in md
    assert md.startswith("# Data readiness")


def test_verdict_precedence_red_over_unmeasured(base_case):
    dataset, stats, requirements = base_case
    requirements = dict(requirements, min_tokens=999999)  # DE-RDY-002 -> False
    stats = [s for s in stats if s["name"] != "clean"]  # DE-RDY-006 -> None
    report = build_readiness(dataset, stats, requirements)
    assert report["verdict"] == "RED"
