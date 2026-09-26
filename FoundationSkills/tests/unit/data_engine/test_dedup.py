
"""Tests for the ``dedup`` op (builtin MinHash/LSH plus a faked datasketch)."""
from __future__ import annotations

import hashlib
import sys
import types

import pytest

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops import dedup as dedup_mod
from foundationskills.skills.data_engine.ops.dedup import jaccard_estimate, minhash_signature


def run_dedup(records, cfg):
    stats = OpStats(name="dedup")
    out = list(OPS["dedup"](iter(records), cfg, stats))
    return out, stats


WORDS = [f"tok{i:03d}" for i in range(80)]
BASE_TEXT = " ".join(WORDS)
PARAPHRASE = " ".join(["tok999" if i == 40 else w for i, w in enumerate(WORDS)])
UNRELATED = " ".join(f"zzz{i:03d}" for i in range(80))


def test_exact_duplicates_normalized():
    # rv8: punctuation-stripping merges distinct records ("x = 1" vs "x 1"), so it
    # is the opt-in "aggressive" mode; the light default keeps them apart.
    recs = [{"id": "1", "text": "Hello,   World!"}, {"id": "2", "text": "hello world"}]
    out, _ = run_dedup(recs, {"exact": True, "near": {"enabled": False}})
    assert [r["id"] for r in out] == ["1", "2"]
    recs = [{"id": "1", "text": "Hello,   World!"}, {"id": "2", "text": "HELLO, world!"}]
    out, _ = run_dedup(recs, {"exact": True, "near": {"enabled": False}})
    assert [r["id"] for r in out] == ["1"]  # case + whitespace still normalized
    recs = [{"id": "1", "text": "Hello,   World!"}, {"id": "2", "text": "hello world"}]
    out, stats = run_dedup(recs, {"exact": True, "exact_normalize": "aggressive", "near": {"enabled": False}})
    assert [r["id"] for r in out] == ["1"]
    assert stats.dropped["exact_dup"] == 1
    assert stats.extra["exact_dupes"] == 1
    assert stats.extra["near_dupes"] == 0
    assert stats.extra["dedup_rate"] == 0.5


def test_near_duplicate_one_word_change_caught():
    recs = [
        {"id": "1", "text": BASE_TEXT},
        {"id": "2", "text": PARAPHRASE},
    ]
    out, stats = run_dedup(recs, {"exact": True, "near": {"enabled": True, "threshold": 0.7, "ngram": 5}})
    assert [r["id"] for r in out] == ["1"]
    assert stats.dropped["near_dup"] == 1
    assert stats.extra["dedup_rate"] == 0.5


def test_near_unrelated_kept():
    recs = [{"id": "1", "text": BASE_TEXT}, {"id": "2", "text": UNRELATED}]
    out, stats = run_dedup(recs, {"near": {"enabled": True, "threshold": 0.7, "ngram": 5}})
    assert [r["id"] for r in out] == ["1", "2"]
    assert stats.dropped.get("near_dup", 0) == 0


def test_minhash_exports_consistent():
    sig_a = minhash_signature(BASE_TEXT, 128, 5)
    sig_b = minhash_signature(PARAPHRASE, 128, 5)
    sig_c = minhash_signature(UNRELATED, 128, 5)
    assert len(sig_a) == 128
    assert jaccard_estimate(sig_a, sig_a) == 1.0
    assert jaccard_estimate(sig_a, sig_b) > 0.7
    assert jaccard_estimate(sig_a, sig_c) < 0.3


def test_sample_scope_paragraph_dedup_across_records():
    shared = "this paragraph is shared across records verbatim"
    rec1 = {"id": "1", "text": f"first unique opener.\n\n{shared}.\n\nfirst unique closing."}
    rec2 = {"id": "2", "text": f"second unique opener entirely.\n\n{shared}."}
    out, stats = run_dedup([rec1, rec2], {"exact": False, "near": {"enabled": False}, "scope": "sample"})
    assert len(out) == 2  # records kept; paragraphs deduped
    assert shared in out[0]["text"]
    assert shared not in out[1]["text"]
    assert stats.modified["dup_paragraphs_removed"] == 1


def test_datasketch_requested_but_absent_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasketch", None)
    recs = [{"id": "1", "text": BASE_TEXT}, {"id": "2", "text": BASE_TEXT}]
    out, stats = run_dedup(recs, {"near": {"enabled": True, "backend": "datasketch", "threshold": 0.8}})
    assert stats.backend == "builtin"
    assert "datasketch_fallback" in stats.extra
    assert len(out) == 1
    assert stats.dropped["exact_dup"] == 1


def test_datasketch_backend_with_fake_module(monkeypatch):
    from foundationskills.skills.data_engine.ops.dedup import minhash_signature as _ms

    class _FakeMinHash:
        def __init__(self, num_perm=128):
            self.num_perm = num_perm
            self._shingles: list[str] = []

        def update(self, data: bytes) -> None:
            self._shingles.append(data.decode("utf-8"))

        @property
        def hashvalues(self):
            return list(_ms(" ".join(self._shingles), self.num_perm, 1))

    class _FakeLSH:
        def __init__(self, threshold=0.8, num_perm=128):
            self._items: dict[str, _FakeMinHash] = {}

        def query(self, m):
            return list(self._items)

        def insert(self, key, m):
            self._items[key] = m

    fake = types.ModuleType("datasketch")
    fake.MinHash = _FakeMinHash
    fake.MinHashLSH = _FakeLSH
    monkeypatch.setitem(sys.modules, "datasketch", fake)

    recs = [
        {"id": "1", "text": BASE_TEXT},
        {"id": "2", "text": PARAPHRASE},
        {"id": "3", "text": UNRELATED},
    ]
    out, stats = run_dedup(
        recs,
        {"exact": False, "near": {"enabled": True, "backend": "datasketch", "threshold": 0.7, "ngram": 5}},
    )
    assert stats.backend == "datasketch"
    assert [r["id"] for r in out] == ["1", "3"]
    assert stats.dropped["near_dup"] == 1
