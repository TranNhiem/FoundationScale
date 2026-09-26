
"""Tests for run_pipeline: isolation with fake ops, plus a real end-to-end run."""
from __future__ import annotations

import json

import pytest

import foundationskills.skills.data_engine.ops  # noqa: F401 - register real ops
from foundationskills.core.provenance import sha256_file
from foundationskills.skills.data_engine.ops import base as ops_base
from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, counted
from foundationskills.skills.data_engine.phase2 import Phase2NotImplemented
from foundationskills.skills.data_engine.pipeline import PipelineError, run_pipeline
from foundationskills.skills.data_engine.recommend import recommend_pipeline


def _fake_ingest(records, cfg, stats):
    for i, text in enumerate(cfg.get("texts", [])):
        stats.records_out += 1
        yield {"id": str(i), "text": text, "meta": {"source": "fake", "domain": cfg.get("domain")}}


def _fake_format(records, cfg, stats):
    for rec in counted(records, stats):
        stats.records_out += 1
        out = dict(rec)
        out.pop("messages", None)
        yield out


class TestPipelineWithIsolatedOps:
    def _spec(self, ops):
        return {
            "target_format": "cpt",
            "ops": ops,
            "seed": 0,
            "tokenizer": None,
            "rationale": ["test"],
        }

    @pytest.fixture
    def fake_ops(self, monkeypatch):
        registry = {
            "ingest": FunctionOp("ingest", _fake_ingest),
            "format": FunctionOp("format", _fake_format),
        }
        monkeypatch.setattr(ops_base, "OPS", registry)
        return registry

    def test_unknown_op_names_known_ops(self, fake_ops, tmp_path):
        spec = self._spec([
            {"op": "ingest", "config": {"texts": ["hello"]}},
            {"op": "no_such_op", "config": {}},
        ])
        with pytest.raises(PipelineError) as err:
            run_pipeline(spec, tmp_path)
        assert "no_such_op" in str(err.value)
        assert "ingest" in str(err.value)  # known ops are listed

    def test_phase2_op_refuses(self, fake_ops, tmp_path):
        for name in ("semantic_dedup", "synthesize", "toolcall_format", "video_ingest"):
            spec = self._spec([
                {"op": "ingest", "config": {"texts": ["x"]}},
                {"op": name, "config": {}},
            ])
            with pytest.raises(Phase2NotImplemented, match=f"phase-2: {name}"):
                run_pipeline(spec, tmp_path)

    def test_first_op_must_be_ingest(self, fake_ops, tmp_path):
        spec = self._spec([{"op": "format", "config": {}}])
        with pytest.raises(PipelineError, match="ingest"):
            run_pipeline(spec, tmp_path)

    def test_shards_and_stats_written(self, fake_ops, tmp_path):
        spec = self._spec([
            {"op": "ingest", "config": {"texts": [f"doc {i}" for i in range(7)], "domain": "fakedom"}},
            {"op": "format", "config": {}},
        ])
        result = run_pipeline(spec, tmp_path, shard_records=3)
        assert result.dataset["num_records"] == 7
        assert len(result.shards) == 3  # 3+3+1
        assert result.dataset["num_tokens"] is None  # no tokenize op
        assert "num_tokens_approx" not in result.dataset
        for shard_row in result.dataset["shards"]:
            assert sha256_file(shard_row["path"]) == shard_row["sha256"]
        stats_file = tmp_path / "stats.json"
        data = json.loads(stats_file.read_text(encoding="utf-8"))
        names = [s["name"] for s in data["stats"]]
        assert names[:2] == ["ingest", "format"]
        pipeline_entry = next(s for s in data["stats"] if s["name"] == "pipeline")
        assert pipeline_entry["extra"]["domain_counts"] == {"fakedom": 7}

    def test_zero_records_returns_invalid_dataset_payload_not_written(self, fake_ops, tmp_path):
        spec = self._spec([{"op": "ingest", "config": {"texts": []}}, {"op": "format", "config": {}}])
        result = run_pipeline(spec, tmp_path)
        assert result.dataset["num_records"] == 0
        assert result.shards == []


class TestPipelineEndToEndRealOps:
    def test_cpt_corpus(self, tmp_path):
        corpus = tmp_path / "corpus"
        jsonl_dir = corpus / "jsonl"
        jsonl_dir.mkdir(parents=True)
        (jsonl_dir / "part.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"j{i}",
                        "text": "Industrial torque specification " + ("fasteners tightened evenly " * (3 + i)),
                        "meta": {"source": "jsonl", "domain": "manufacturing"},
                    }
                )
                + "\n"
                for i in range(1, 5)
            ),
            encoding="utf-8",
        )
        docs = corpus / "docs"
        docs.mkdir()
        (docs / "procedure.txt").write_text(
            "Maintenance procedure: inspect the bearing housing every 500 hours. "
            "Replace seals when leakage exceeds tolerance. " * 4,
            encoding="utf-8",
        )
        (docs / "safety.html").write_text(
            "<html><head><title>Safety</title></head><body><p>"
            "Lockout tagout procedure applies to the hydraulic press line. "
            "Verify zero energy before opening the guard. " * 3
            + "</p></body></html>",
            encoding="utf-8",
        )

        sources = [
            {"uri": str(jsonl_dir), "kind": "local_dir"},
            {"uri": str(docs / "procedure.txt"), "kind": "documents"},
            {"uri": str(docs / "safety.html"), "kind": "documents"},
        ]
        spec = recommend_pipeline(
            target_format="cpt",
            sources=sources,
            goal="domain_expert",
            algorithm=None,
            tokenizer=None,
            chat_template_family=None,
            domain="manufacturing",
            benchmarks=[],
        )
        out_dir = tmp_path / "out"
        result = run_pipeline(spec, out_dir, shard_records=2)

        dataset = result.dataset
        assert dataset["format"] == "cpt"
        assert dataset["num_records"] > 0
        assert dataset["fs_columns"] == {"text_column": "text", "image_column": None, "gold_key": None}
        assert dataset["shards"] and len(dataset["shards"]) == len(result.shards)
        for shard_row in dataset["shards"]:
            assert sha256_file(shard_row["path"]) == shard_row["sha256"]
            assert shard_row["records"] in {1, 2}
        assert "text" in dataset["schema"]["columns"]
        records_total = sum(s["records"] for s in dataset["shards"])
        assert records_total == dataset["num_records"]
        stats_data = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
        op_names = [s["name"] for s in stats_data["stats"]]
        assert op_names[0] == "ingest"
        assert "format" in op_names and "tokenize" in op_names
        pipeline_entry = next(s for s in stats_data["stats"] if s["name"] == "pipeline")
        assert pipeline_entry["extra"]["num_shards"] == len(result.shards)
