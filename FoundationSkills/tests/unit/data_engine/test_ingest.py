
"""Tests for the ``ingest`` op (stdlib paths plus faked optional backends)."""
from __future__ import annotations

import json
import sys
import types

import pytest

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops import ingest as ingest_mod
from foundationskills.skills.data_engine.ops.ingest import IngestError


def run_ingest(cfg: dict):
    stats = OpStats(name="ingest")
    out = list(OPS["ingest"](iter([]), cfg, stats))
    return out, stats


def src(uri, kind, **options):
    s = {"uri": str(uri), "kind": kind}
    if options:
        s["options"] = options
    return s


def test_jsonl_basic_and_bad_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"id": "a", "text": "hello"}\nnot json\n"just a string"\n{"text": "world"}\n',
        encoding="utf-8",
    )
    out, stats = run_ingest({"sources": [src(p, "jsonl")]})
    assert [r["text"] for r in out] == ["hello", "world"]
    assert out[0]["id"] == "a"
    assert stats.dropped["parse_error"] == 1
    assert stats.dropped["non_object_record"] == 1
    assert stats.records_out == 2


def test_json_list_and_data_wrapper(tmp_path):
    p1 = tmp_path / "list.json"
    p1.write_text(json.dumps([{"text": "one"}, {"text": "two"}]), encoding="utf-8")
    p2 = tmp_path / "wrapped.json"
    p2.write_text(json.dumps({"data": [{"text": "three"}, 42]}), encoding="utf-8")
    out, stats = run_ingest({"sources": [src(p1, "json"), src(p2, "json")]})
    assert [r["text"] for r in out] == ["one", "two", "three"]
    assert stats.dropped["non_object_record"] == 1


def test_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text('text\n"comma, separated"\nplain\n', encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "csv")]})
    assert [r["text"] for r in out] == ["comma, separated", "plain"]


def test_local_dir_dispatch(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"text": "from jsonl"}\n', encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("from txt", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    out, stats = run_ingest({"sources": [src(tmp_path, "local_dir")]})
    texts = sorted(r["text"] for r in out)
    assert texts == ["from jsonl", "from txt"]
    assert stats.dropped["unsupported_ext:.bin"] == 1
    # meta.source is the individual file path
    jsonl_rec = next(r for r in out if r["text"] == "from jsonl")
    assert jsonl_rec["meta"]["source"].endswith("a.jsonl")


def test_local_dir_missing_refuses(tmp_path):
    with pytest.raises(IngestError, match="missing input"):
        run_ingest({"sources": [src(tmp_path / "nope", "local_dir")]})


def test_documents_txt_md_html(tmp_path):
    (tmp_path / "a.txt").write_text("plain text doc", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nmarkdown body", encoding="utf-8")
    (tmp_path / "c.html").write_text(
        "<html><head><title>t</title><style>body{}</style></head>"
        "<body><script>var x=1;</script><p>Hello <b>World</b></p></body></html>",
        encoding="utf-8",
    )
    out, _ = run_ingest({"sources": [src(tmp_path, "documents")]})
    texts = {r["text"] for r in out}
    assert "plain text doc" in texts
    html_text = next(t for t in texts if "Hello" in t)
    assert "Hello" in html_text and "World" in html_text
    assert "var x" not in html_text
    assert "<p>" not in html_text


def test_documents_pdf_only_raises_naming_docling(tmp_path, monkeypatch):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(IngestError, match="docling"):
        run_ingest({"sources": [src(tmp_path, "documents")]})


def test_documents_mixed_counts_unsupported_without_raising(tmp_path, monkeypatch):
    (tmp_path / "keep.txt").write_text("kept", encoding="utf-8")
    (tmp_path / "doc.docx").write_bytes(b"PK fake")
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    out, stats = run_ingest({"sources": [src(tmp_path, "documents")]})
    assert [r["text"] for r in out] == ["kept"]
    assert stats.dropped["unsupported_doc:.docx"] == 1


def test_parquet_missing_dep_refuses(tmp_path, monkeypatch):
    p = tmp_path / "data.parquet"
    p.write_bytes(b"PAR1 fake")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(IngestError, match="pyarrow"):
        run_ingest({"sources": [src(p, "parquet")]})


def test_parquet_with_fake_pyarrow(tmp_path, monkeypatch):
    p = tmp_path / "data.parquet"
    p.write_bytes(b"PAR1 fake")

    class _FakeTable:
        def to_pylist(self):
            return [{"id": "p1", "text": "parquet row"}, {"text": "second"}]

    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.read_table = lambda path: _FakeTable()
    fake_pa = types.ModuleType("pyarrow")
    fake_pa.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    out, _ = run_ingest({"sources": [src(p, "parquet")]})
    assert [r["text"] for r in out] == ["parquet row", "second"]


def test_hf_dataset_missing_dep_refuses(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(IngestError, match="datasets"):
        run_ingest({"sources": [src("org/corpus", "hf_dataset")]})


def test_hf_dataset_with_fake_datasets(monkeypatch):
    calls = {}

    def load_dataset(uri, split=None, streaming=False):
        calls.update(uri=uri, split=split, streaming=streaming)
        return iter([{"text": "hf one"}, {"text": "hf two"}])

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    out, _ = run_ingest({"sources": [src("org/corpus", "hf_dataset", split="validation", streaming=True)]})
    assert [r["text"] for r in out] == ["hf one", "hf two"]
    assert calls == {"uri": "org/corpus", "split": "validation", "streaming": True}
    assert all(r["meta"]["source"] == "org/corpus" for r in out)


def test_sharegpt_mapping(tmp_path):
    p = tmp_path / "sg.jsonl"
    conv = [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]
    p.write_text(json.dumps({"conversations": conv}) + "\n", encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "jsonl")]})
    assert out[0]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_alpaca_mapping(tmp_path):
    p = tmp_path / "alp.jsonl"
    row = {"instruction": "Summarize", "input": "the passage", "output": "a summary"}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "jsonl")]})
    assert out[0]["messages"][0] == {"role": "user", "content": "Summarize\nthe passage"}
    assert out[0]["messages"][1] == {"role": "assistant", "content": "a summary"}


def test_preference_mapping(tmp_path):
    p = tmp_path / "pref.jsonl"
    row = {"prompt": "pick one", "chosen": "good", "rejected": "bad"}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "jsonl")]})
    assert out[0]["prompt"] == "pick one"
    assert out[0]["chosen"] == "good"
    assert out[0]["rejected"] == "bad"


def test_question_answer_becomes_prompt_answer(tmp_path):
    p = tmp_path / "qa.jsonl"
    p.write_text(json.dumps({"question": "2+2?", "answer": "4"}) + "\n", encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "jsonl")]})
    assert out[0]["prompt"] == "2+2?" and out[0]["answer"] == "4"


def test_field_map_and_license(tmp_path):
    p = tmp_path / "fm.jsonl"
    p.write_text(json.dumps({"body_text": "mapped", "license_x": 1}) + "\n", encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "jsonl", field_map={"text": "body_text"}, license="cc-by-4.0")]})
    assert out[0]["text"] == "mapped"
    assert out[0]["meta"]["license"] == "cc-by-4.0"


def test_image_text_jsonl(tmp_path):
    p = tmp_path / "mm.jsonl"
    row = {"img": "/data/1.png", "caption": "a cat on a mat"}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out, _ = run_ingest({"sources": [src(p, "image_text", image_key="img", text_key="caption")]})
    assert out[0]["images"] == ["/data/1.png"]
    assert out[0]["text"] == "a cat on a mat"


def test_deterministic_ids_and_max_records(tmp_path):
    p = tmp_path / "ids.jsonl"
    p.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(5)), encoding="utf-8")
    out1, _ = run_ingest({"sources": [src(p, "jsonl")]})
    out2, _ = run_ingest({"sources": [src(p, "jsonl")]})
    assert [r["id"] for r in out1] == [r["id"] for r in out2]
    out3, stats3 = run_ingest({"sources": [src(p, "jsonl")], "max_records": 2})
    assert len(out3) == 2
    assert stats3.extra["truncated_at_max_records"] == 2


def test_missing_sources_refuses():
    with pytest.raises(IngestError, match="sources"):
        run_ingest({})
