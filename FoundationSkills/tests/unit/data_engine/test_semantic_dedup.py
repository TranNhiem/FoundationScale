"""Tests for the ``semantic_dedup`` op: injected fake encoder (no models, no downloads) plus one gpu smoke.

``base.OpUnavailable`` arrives with this op's dependency contract; the tiny shim below keeps this file
runnable against a base.py from the same lane and is a no-op once the class is declared there.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import foundationskills.skills.data_engine.ops.base as _base

if not hasattr(_base, "OpUnavailable"):  # pragma: no cover - no-op once ops/base.py declares it
    class OpUnavailable(RuntimeError):
        pass

    _base.OpUnavailable = OpUnavailable

from foundationskills.core.schema import validate
from foundationskills.skills.data_engine.ops import semantic_dedup as sd_mod
from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops.semantic_dedup import CONFIG_SCHEMA, preflight


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def check_invariant(stats: OpStats) -> None:
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def unit(*values: float) -> tuple[float, ...]:
    vec = np.asarray(values, dtype=np.float32)
    scaled = vec / float(np.linalg.norm(vec))
    return tuple(float(x) for x in scaled)


class FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


class FakeEncoder:
    """SentenceTransformer look-alike: one fixed unit vector per exact text, no downloads."""

    def __init__(self, vectors, *, max_seq_length=512, tokenizer=None, device="cpu", revision="fake-rev"):
        self._vectors = {text: np.asarray(unit(*vec), dtype=np.float32) for text, vec in vectors.items()}
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.device = device
        self.revision = revision

    def encode(self, texts, **_kwargs):
        unknown = [t for t in texts if t not in self._vectors]
        if unknown:
            raise KeyError(f"no fixed vector for {unknown!r}")
        return np.stack([self._vectors[t] for t in texts]).astype(np.float32)


def use_fake(monkeypatch, vectors, **enc_kwargs):
    def factory(_cfg: dict) -> FakeEncoder:
        return FakeEncoder(vectors, **enc_kwargs)

    monkeypatch.setattr(sd_mod, "_ENCODER_FACTORY", factory)


def run_op(records, cfg):
    stats = OpStats(name="semantic_dedup")
    out = list(OPS["semantic_dedup"](iter(records), cfg, stats))
    return out, stats


def chain_vectors() -> dict[str, tuple[float, ...]]:
    """Unit vectors with cos(alpha,beta)=0.95, cos(beta,gamma)=0.95 but cos(alpha,gamma)=0.85."""
    alpha = (1.0, 0.0, 0.0)
    side = math.sqrt(1.0 - 0.95**2)
    beta = (0.95, side, 0.0)
    x = 0.85
    y = (0.95 - 0.95 * x) / side  # forces cos(beta, gamma) to 0.95 exactly
    z = math.sqrt(max(0.0, 1.0 - x * x - y * y))
    gamma = (x, y, z)
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def test_exact_keep_first_no_chain_collapse(monkeypatch):
    # rv8: beta is dropped as a near-duplicate of alpha, so gamma is only compared to the KEPT alpha
    # (0.85 < 0.90) and survives -- comparing against the dropped beta too would chain-collapse it away.
    use_fake(monkeypatch, chain_vectors())
    recs = [{"id": "alpha", "text": "alpha"}, {"id": "beta", "text": "beta"}, {"id": "gamma", "text": "gamma"}]
    out, stats = run_op(recs, {"threshold": 0.9, "mode": "exact"})
    assert [r["id"] for r in out] == ["alpha", "gamma"]
    assert [r["text"] for r in out] == ["alpha", "gamma"]  # kept records are untouched
    assert dict(stats.dropped) == {"semantic_duplicate": 1}
    assert stats.extra["examples"] == [{"id": "beta", "kept_id": "alpha", "sim": 0.95}]
    assert stats.extra["mode"] == "exact"
    assert stats.extra["approximate"] is False
    assert stats.extra["measurement_scope"] == "global"
    assert stats.extra["model_revision"] == "fake-rev"
    assert stats.extra["device"] == "cpu"
    assert stats.backend == f"sentence-transformers:{DEFAULT_MODEL}@fake-rev"
    assert not stats.modified  # nothing is edited, only kept or dropped
    check_invariant(stats)


def test_output_order_matches_input_and_matches_are_earlier_kept(monkeypatch):
    use_fake(monkeypatch, {"r0": (1, 0, 0), "r1": (1, 0, 0), "r2": (0, 1, 0), "r3": (0, 1, 0), "r4": (0, 0, 1)})
    recs = [{"id": rid, "text": rid} for rid in ("r0", "r1", "r2", "r3", "r4")]
    out, stats = run_op(recs, {"threshold": 0.9})
    assert [r["id"] for r in out] == ["r0", "r2", "r4"]  # input order, first occurrence wins
    assert stats.extra["examples"] == [
        {"id": "r1", "kept_id": "r0", "sim": 1.0},
        {"id": "r3", "kept_id": "r2", "sim": 1.0},
    ]  # each dropped record points at the earlier kept record that matched it
    check_invariant(stats)


def test_accounting_invariant_counts_every_record(monkeypatch):
    use_fake(monkeypatch, {"keep me": (1, 0, 0), "drop me": (1, 0, 0), "and me": (1, 0, 0), "safe": (0, 1, 0)})
    recs = [
        {"id": "1", "text": "keep me"},
        {"id": "2", "text": "drop me"},
        {"id": "3", "text": "and me"},
        {"id": "4", "text": "safe"},
        {"id": "5"},
        {"id": "6"},
    ]
    out, stats = run_op(recs, {"threshold": 0.9})
    assert [r["id"] for r in out] == ["1", "4", "5", "6"]
    assert stats.records_in == 6
    assert stats.records_out == 4
    assert dict(stats.dropped) == {"semantic_duplicate": 2}
    check_invariant(stats)


def test_similarity_histogram_sums_to_n_minus_1(monkeypatch):
    # rv8: bins over [0, 1]; 0.87 -> bin 17 (17.4) and 0.24 -> bin 4 (4.8), both far from an edge.
    side_close = math.sqrt(1 - 0.87**2)
    side_far = math.sqrt(1 - 0.24**2)
    use_fake(monkeypatch, {"base": (1, 0, 0), "close": (0.87, side_close, 0.0), "far": (0.24, 0.0, side_far)})
    recs = [{"id": name, "text": name} for name in ("base", "close", "far")]
    out, stats = run_op(recs, {"threshold": 0.9})
    assert [r["id"] for r in out] == ["base", "close", "far"]
    warm, kept_rows, _ignored = _ = stats.records_in, stats.records_out, stats.extra["n_embedded"]
    hist = stats.extra["similarity_histogram"]
    assert len(hist) == 20
    assert sum(hist) == stats.extra["n_embedded"] - 1  # the first record has no earlier kept record
    assert hist[17] == 1  # close: 0.87 to base
    assert hist[4] == 1  # far: 0.24 to base (closer to base than to close)
    assert stats.extra["near_threshold"] == 1  # only "close" sits in [0.85, 0.9)
    assert (warm, kept_rows) == (3, 3)
    check_invariant(stats)


def test_empty_key_records_are_kept_not_dropped(monkeypatch):
    use_fake(monkeypatch, {"only": (1.0, 0.0, 0.0)})
    recs = [
        {"id": "n1", "meta": {}},
        {"id": "n2", "meta": {}},  # identical to n1, but there is nothing to compare
        {"id": "ws", "text": "   "},  # whitespace is not key text either
        {"id": "k1", "text": "only"},
    ]
    out, stats = run_op(recs, {"threshold": 0.9})
    assert [r["id"] for r in out] == ["n1", "n2", "ws", "k1"]
    assert stats.extra["empty_key_kept"] == 3
    assert stats.extra["n_embedded"] == 1
    assert stats.extra["truncated"] is None  # no tokenizer -> unknown, not 0
    assert not stats.dropped
    check_invariant(stats)


def test_truncated_counts_texts_over_max_seq_length(monkeypatch):
    use_fake(
        monkeypatch,
        {"a b c d e": (1, 0, 0), "x y": (0, 1, 0), "p q r s": (0, 0, 1)},
        tokenizer=FakeTokenizer(),
        max_seq_length=3,
    )
    recs = [{"id": "1", "text": "a b c d e"}, {"id": "2", "text": "x y"}, {"id": "3", "text": "p q r s"}]
    _out, stats = run_op(recs, {"threshold": 0.9})
    assert stats.extra["truncated"] == 2  # 5 and 4 whitespace tokens over max_seq_length=3
    check_invariant(stats)


def test_kmeans_mode_is_deterministic_across_runs(monkeypatch):
    try:
        import sklearn.cluster  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment without sklearn
        pytest.skip(f"semantic_dedup kmeans mode needs sklearn.cluster ({exc})")
    vectors = {
        "a0": (1.0, 0.0, 0.0),
        "a1": (1.0, 0.03, 0.0),
        "a2": (1.0, -0.03, 0.0),
        "b0": (0.0, 1.0, 0.0),
        "b1": (0.03, 1.0, 0.0),
        "b2": (-0.03, 1.0, 0.0),
    }
    use_fake(monkeypatch, vectors)
    cfg = {"threshold": 0.9, "mode": "kmeans", "n_clusters": 2, "seed": 7}
    recs = [{"id": name, "text": name} for name in vectors]
    out1, stats1 = run_op(recs, dict(cfg))
    out2, stats2 = run_op(recs, dict(cfg))
    assert [r["id"] for r in out1] == [r["id"] for r in out2]
    assert stats1.to_dict() == stats2.to_dict()
    assert stats1.extra["mode"] == "kmeans"
    assert stats1.extra["approximate"] is True
    assert stats1.extra["measurement_scope"] == "cluster"
    check_invariant(stats1)


def test_max_records_refusal_before_any_backend_work():
    stats = OpStats(name="semantic_dedup")
    recs = [{"id": str(i), "text": f"t{i}"} for i in range(3)]
    with pytest.raises(ValueError, match="semantic_dedup: 3 records exceed max_records=2"):
        list(OPS["semantic_dedup"](iter(recs), {"max_records": 2}, stats))  # raises before touching the encoder


def test_config_schema_rejects_unknown_key_and_out_of_range_values():
    full = {
        "key": "auto",
        "model": DEFAULT_MODEL,
        "device": "auto",
        "batch_size": 256,
        "max_seq_length": 256,
        "local_files_only": True,
        "threshold": 0.9,
        "mode": "auto",
        "exact_max": 1000,
        "n_clusters": 8,
        "seed": 0,
        "max_records": 10,
    }
    assert validate(full, CONFIG_SCHEMA) == []
    assert validate({"threshold": 0.9}, CONFIG_SCHEMA) == []
    assert validate({"thresold": 0.9}, CONFIG_SCHEMA) != []  # a typo is refused, never silently ignored
    assert validate({"threshold": 0.0}, CONFIG_SCHEMA) != []  # the admitted range is (0, 1]
    assert validate({"threshold": 1.5}, CONFIG_SCHEMA) != []
    assert validate({"device": "tpu"}, CONFIG_SCHEMA) != []
    assert validate({"max_records": 0}, CONFIG_SCHEMA) != []
    assert CONFIG_SCHEMA["additionalProperties"] is False


def test_preflight_reports_missing_model_dir(tmp_path):
    model = tmp_path / "nope"
    problems = preflight({"model": str(model), "local_files_only": True, "mode": "exact"})
    assert problems
    assert any(str(model) in p for p in problems)


def test_preflight_accepts_local_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    problems = preflight({"model": str(model_dir), "local_files_only": True, "mode": "exact"})
    assert not any(str(model_dir) in p for p in problems)  # imports are checked separately from the model lookup


@pytest.mark.gpu
def test_real_encoder_smoke():
    """Real sentence-transformers backend on 4 short texts (two paraphrases); never downloads."""
    cfg = {"model": DEFAULT_MODEL, "threshold": 0.9, "mode": "exact", "local_files_only": True}
    problems = preflight(cfg)
    if problems:
        pytest.skip("semantic_dedup preflight problems: " + "; ".join(problems))
    recs = [
        {"id": "p1", "text": "A cat is sleeping on the sofa."},
        {"id": "p2", "text": "The cat sleeps on a sofa."},
        {"id": "q1", "text": "Photosynthesis turns daylight into sugar."},
        {"id": "q2", "text": "The peace treaty was signed in 1919."},
    ]
    out, stats = run_op(recs, dict(cfg))
    kept = [r["id"] for r in out]
    assert stats.records_in == 4
    check_invariant(stats)
    assert stats.backend.startswith(f"sentence-transformers:{DEFAULT_MODEL}@")
    assert stats.extra["n_embedded"] == 4
    assert "q1" in kept and "q2" in kept  # the unrelated short texts never collapse into each other
    assert isinstance(stats.extra["similarity_histogram"], list)
