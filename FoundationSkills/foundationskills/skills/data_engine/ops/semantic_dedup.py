"""``semantic_dedup`` op: embedding near-duplicate removal (SemDeDup-style greedy keep-first).

- Key (``key: "auto"``): the same key text as :mod:`foundationskills.skills.data_engine.ops.dedup`
  (``text``, else the concatenated message contents, else ``prompt`` + ``chosen``), reusing its
  ``_record_key`` so both ops agree on what "the same record" means. Records without key text are
  passed through and counted in ``stats.extra["empty_key_kept"]`` (nothing to compare is no
  evidence of duplication, so missing data never costs a record).
- Encodings come from ``_ENCODER_FACTORY(cfg)`` (a ``sentence_transformers.SentenceTransformer`` by
  default) so tests inject fixed vectors without models, downloads or a GPU. Embeddings are
  L2-normalised float32, so cosine similarity is a plain dot product and the result is stable
  across BLAS paths. ``stats.backend`` names the implementation that ran as
  ``sentence-transformers:<model>@<revision>`` with the revision taken from the resolved snapshot
  path ("unknown" when it cannot be read without network access -- never guessed).
- Algorithm: two passes over a full in-memory buffer. ``max_records`` caps that buffer (a
  2,000,000 x 1024 float32 kept matrix is ~8 GB, which is exactly why the cap is explicit and
  exceeding it raises instead of thrashing). A record is a duplicate iff its cosine similarity to
  any EARLIER KEPT record is >= ``threshold``. Comparing against kept records only is what makes
  this greedy keep-first: no chain collapse (a~b and b~c do not make c a duplicate of a when a~c
  falls below the threshold) and kept records leave in input order, unchanged.
- ``mode: "exact"`` (auto up to ``exact_max`` records, or forced) compares record blocks against
  the kept matrix with one matmul per block (4096 rows) plus the rows accepted inside the block --
  deterministic, complete, order sensitive.
- ``mode: "kmeans"`` (auto above ``exact_max``, or forced) clusters with ``sklearn``'s
  ``MiniBatchKMeans`` (``random_state=seed``: reproducible) and applies the same greedy rule inside
  each cluster only. Cross-cluster near-duplicates survive by construction, which is why the run is
  flagged ``approximate=True`` -- a false positive over-deletes, so the fast path errs on keeping.
- Drops: slug ``semantic_duplicate``. Everything else is a measurement, never a removal.

``stats.extra``: ``model``, ``model_revision``, ``device``, ``threshold``, ``mode``,
``approximate``, ``n_embedded``, ``empty_key_kept``, ``truncated`` (texts longer than the encoder's
``max_seq_length`` tokens per its tokenizer; ``None`` when the tokenizer is unavailable -- unknown
is not 0), ``similarity_histogram`` (20 bins over [0, 1] of every record's max-sim-to-earlier-kept;
records with no earlier kept candidate -- the first record, and in kmeans mode the first record of
each cluster -- have no measurable similarity and are excluded, so counts sum to the number of
comparisons made), ``near_threshold`` (records with similarity in [threshold-0.05, threshold): the
population a threshold regression would flip) and ``examples`` (up to 20 dropped pairs).
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from foundationskills.skills.data_engine.ops.base import (
    FunctionOp,
    OpStats,
    OpUnavailable,
    counted,
    register_op,
)
from foundationskills.skills.data_engine.ops.dedup import _record_key


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key changes the measurement
    "properties": {
        "key": {"enum": ["auto", "text"]},
        "model": {"type": "string", "minLength": 1},
        "device": {"enum": ["auto", "cpu", "cuda"]},
        "batch_size": {"type": "integer", "minimum": 1},
        "max_seq_length": {"type": "integer", "minimum": 1},
        "local_files_only": {"type": "boolean"},
        "threshold": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        "mode": {"enum": ["auto", "exact", "kmeans"]},
        "exact_max": {"type": "integer", "minimum": 0},
        "n_clusters": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer"},
        "max_records": {"type": "integer", "minimum": 1},
    },
}

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_BLOCK = 4096  # matmul block: cheap against the kept matrix, and the in-block rows are still exact
_HIST_BINS = 20  # similarity histogram over [0, 1] in 0.05-wide bins
_MAX_EXAMPLES = 20


def _snapshot_revision(path: str) -> str:
    """Commit hash segment of an HF ``.../snapshots/<hash>`` path, else "unknown"."""
    parts = Path(path).resolve().parts
    if "snapshots" in parts:
        index = len(parts) - 1 - parts[::-1].index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def _resolve_revision(model: str) -> str:
    """Revision the model resolves to locally; "unknown" when no snapshot can be read offline."""
    if os.path.isdir(model):
        return _snapshot_revision(model)
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model, local_files_only=True)  # cache lookup only: no network
    except Exception:
        return "unknown"
    return _snapshot_revision(snapshot)


def _model_revision(encoder: Any, model: str) -> str:
    """Revision of the encoder that ran (``.revision`` lets injected backends declare theirs)."""
    override = getattr(encoder, "revision", None)
    return str(override) if isinstance(override, str) and override else _resolve_revision(model)


def _auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_encoder(cfg: dict) -> Any:
    """Real backend (overridden by tests through :data:`_ENCODER_FACTORY`): SentenceTransformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise OpUnavailable(f"semantic_dedup: sentence-transformers is not importable ({exc})") from exc
    model = str(cfg.get("model", _DEFAULT_MODEL))
    device = str(cfg.get("device", "auto"))
    if device == "auto":
        device = _auto_device()
    local = bool(cfg.get("local_files_only", True))
    try:
        return SentenceTransformer(model, device=device, local_files_only=local)
    except Exception as exc:
        raise OpUnavailable(f"semantic_dedup: model {model!r} is unavailable ({exc})") from exc


_ENCODER_FACTORY: Callable[[dict], Any] = _load_encoder


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # zero vectors compare as 0 similarity instead of dividing by zero
    return matrix / norms


def _encode(encoder: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Embed texts as L2-normalised float32 rows (cosine similarity becomes a dot product)."""
    raw = encoder.encode(
        list(texts), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise OpUnavailable(f"semantic_dedup: encoder returned shape {matrix.shape} for {len(texts)} texts")
    return _l2_normalize(matrix)


def _count_truncated(encoder: Any, texts: Sequence[str]) -> int | None:
    """Texts the encoder will truncate (tokenizer tokens > ``max_seq_length``); None without tokenizer."""
    tokenizer = getattr(encoder, "tokenizer", None)
    max_len = getattr(encoder, "max_seq_length", None)
    if tokenizer is None or not isinstance(max_len, int):
        return None
    count = 0
    for text in texts:
        try:
            tokens = tokenizer.tokenize(text)
        except Exception:
            return None  # tokenizer unusable: "unknown", never a fake 0
        if len(tokens) > max_len:
            count += 1
    return count


def _hist_bin(sim: float) -> int:
    """Bin a similarity into ``_HIST_BINS`` equal bins over [0, 1] (cosines outside are clamped)."""
    clamped = 0.0 if sim <= 0.0 else (1.0 if sim >= 1.0 else float(sim))
    return min(_HIST_BINS - 1, int(clamped * _HIST_BINS))


def _rec_id(rec: dict, index: int) -> Any:
    """Record id for reporting: ``rec["id"]`` when usable/JSON-serialisable, else the input index."""
    value = rec.get("id")
    return value if isinstance(value, (str, int)) and not isinstance(value, bool) else index


def _greedy_keep_first(
    vectors: np.ndarray, threshold: float, block: int = _BLOCK
) -> tuple[list[bool], list[float | None], list[int | None]]:
    """SemDeDup rule over rows in order: keep row i iff max cosine to an EARLIER KEPT row < threshold.

    Kept rows only: that is what makes this keep-first greedy and chain-collapse free. ``block``
    bounds the matmul against the kept matrix; rows accepted inside a block are compared
    sequentially against each other. Returns ``(keeps, max_sims, nearest_rows)``; a row with no
    earlier kept row gets ``None`` -- it has nothing to be measured against.
    """
    rows_total, dim = vectors.shape
    keeps = [True] * rows_total
    max_sims: list[float | None] = [None] * rows_total
    nearest: list[int | None] = [None] * rows_total
    kept_matrix = np.zeros((0, dim), dtype=np.float32)
    kept_rows: list[int] = []
    start = 0
    while start < rows_total:
        end = min(rows_total, start + block)
        chunk = vectors[start:end]
        block_sims = chunk @ kept_matrix.T if kept_rows else None
        accepted = np.zeros((end - start, dim), dtype=np.float32)
        accepted_count = 0
        accepted_rows: list[int] = []
        for offset in range(end - start):
            row = start + offset
            best: float | None = None
            best_row: int | None = None
            if block_sims is not None:
                column = int(np.argmax(block_sims[offset]))
                best = float(block_sims[offset, column])
                best_row = kept_rows[column]
            if accepted_count:
                inner = accepted[:accepted_count] @ chunk[offset]
                column = int(np.argmax(inner))
                value = float(inner[column])
                if best is None or value > best:
                    best, best_row = value, accepted_rows[column]
            max_sims[row] = best
            nearest[row] = best_row
            if best is not None and best >= threshold:
                keeps[row] = False
            else:
                accepted[accepted_count] = chunk[offset]
                accepted_rows.append(row)
                accepted_count += 1
        if accepted_count:
            kept_matrix = np.concatenate([kept_matrix, accepted[:accepted_count]], axis=0)
            kept_rows.extend(accepted_rows)
        start = end
    return keeps, max_sims, nearest


def _cluster_labels(vectors: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Deterministic MiniBatchKMeans labels; sklearn is a runtime dependency of the kmeans mode."""
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:
        raise OpUnavailable(f"semantic_dedup: sklearn.cluster is required for mode=kmeans ({exc})") from exc
    clusters = max(1, min(int(n_clusters), vectors.shape[0]))
    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3, batch_size=4096)
    return np.asarray(kmeans.fit_predict(vectors), dtype=np.int64)


def _semantic_dedup_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    key_cfg = str(cfg.get("key", "auto"))
    model = str(cfg.get("model", _DEFAULT_MODEL))
    batch_size = int(cfg.get("batch_size", 256))
    threshold = float(cfg.get("threshold", 0.90))
    max_records = int(cfg.get("max_records", 2_000_000))
    exact_max = int(cfg.get("exact_max", 100_000))
    mode_cfg = str(cfg.get("mode", "auto"))
    seed = int(cfg.get("seed", 0))
    if mode_cfg not in ("auto", "exact", "kmeans"):
        raise ValueError(f"semantic_dedup: mode must be 'auto', 'exact' or 'kmeans', got {mode_cfg!r}")

    # pass 1: buffer records and key texts, refusing to grow past max_records
    buffered: list[dict] = []
    keys: list[str] = []
    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("non_dict_record")
            continue
        buffered.append(rec)
        key_text = _record_key(rec, key_cfg)
        keys.append(key_text if isinstance(key_text, str) else "")
        if len(buffered) > max_records:
            raise ValueError(
                f"semantic_dedup: {len(buffered)} records exceed max_records={max_records}; raise max_records or "
                "shard the corpus (the greedy rule is corpus-wide, so it buffers every record)"
            )

    entries = [(index, text) for index, text in enumerate(keys) if text.strip()]
    row_of: list[int | None] = [None] * len(keys)
    for row, (index, _text) in enumerate(entries):
        row_of[index] = row
    mode = "exact" if (mode_cfg == "auto" and len(buffered) <= exact_max) else ("kmeans" if mode_cfg == "auto" else mode_cfg)

    # pass 2: embed only what has key text (no key text -> no backend work at all)
    texts = [text for _index, text in entries]
    vectors = np.zeros((0, 1), dtype=np.float32)
    device_cfg = str(cfg.get("device", "auto"))
    truncated: int | None = None
    if texts:
        encoder = _ENCODER_FACTORY(cfg)
        if cfg.get("max_seq_length") is not None:
            encoder.max_seq_length = int(cfg["max_seq_length"])
        revision = _model_revision(encoder, model)
        device_cfg = str(getattr(encoder, "device", device_cfg))
        chunks = []
        for start in range(0, len(texts), batch_size):
            chunks.append(_encode(encoder, texts[start:start + batch_size], batch_size))
        vectors = np.concatenate(chunks, axis=0) if chunks else vectors
        truncated = _count_truncated(encoder, texts)
    else:
        revision = _resolve_revision(model)

    n_embedded = int(vectors.shape[0])
    keeps = [True] * n_embedded
    sims: list[float | None] = [None] * n_embedded
    matches: list[int | None] = [None] * n_embedded
    if n_embedded:
        if mode == "kmeans" and n_embedded > 1:
            n_clusters = (
                int(cfg["n_clusters"]) if cfg.get("n_clusters") is not None else max(1, round(math.sqrt(n_embedded)))
            )
            labels = _cluster_labels(vectors, n_clusters, seed)
            for label in sorted({int(x) for x in labels}):  # ascending labels, rows still in input order
                rows = [i for i in range(n_embedded) if int(labels[i]) == label]
                local_keeps, local_sims, local_matches = _greedy_keep_first(vectors[rows], threshold)
                for position, row in enumerate(rows):
                    keeps[row] = local_keeps[position]
                    sims[row] = local_sims[position]
                    match = local_matches[position]
                    matches[row] = rows[match] if match is not None else None
        else:
            keeps, sims, matches = _greedy_keep_first(vectors, threshold)

    # measurements over every record that had an earlier kept record to be measured against
    histogram = [0] * _HIST_BINS
    near_count = 0
    examples: list[dict[str, Any]] = []
    for row, (record_index, _text) in enumerate(entries):
        sim = sims[row]
        if sim is None:
            continue
        histogram[_hist_bin(sim)] += 1
        if threshold - 0.05 <= sim < threshold:
            near_count += 1
        match = matches[row]
        if keeps[row] or match is None or len(examples) >= _MAX_EXAMPLES:
            continue
        matched_index = entries[match][0]
        examples.append(
            {
                "id": _rec_id(buffered[record_index], record_index),
                "kept_id": _rec_id(buffered[matched_index], matched_index),
                "sim": round(float(sim), 4),
            }
        )

    stats.backend = f"sentence-transformers:{model}@{revision}"
    stats.extra.update(
        {
            "model": model,
            "model_revision": revision,
            "device": device_cfg,
            "threshold": threshold,
            "mode": mode,
            "approximate": mode == "kmeans",
            # kmeans measures max-sim inside each cluster only (global sims would undo the approximation):
            # the histogram then sums to n_embedded - clusters, not n_embedded - 1
            "measurement_scope": "cluster" if mode == "kmeans" else "global",
            "n_embedded": n_embedded,
            "empty_key_kept": len(keys) - len(entries),
            "truncated": truncated,
            "similarity_histogram": histogram,
            "near_threshold": near_count,
            "examples": examples,
        }
    )

    for record_index, rec in enumerate(buffered):
        row = row_of[record_index]
        if row is not None and not keeps[row]:
            stats.drop("semantic_duplicate")
            continue
        stats.records_out += 1
        yield rec


def _spec_problem(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return f"{name} is not importable ({exc})"
    return None if spec is not None else f"{name} is not installed"


def preflight(cfg: dict) -> list[str]:
    """Cheap, offline, deterministic checks for this cfg: imports and local model resolution only."""
    problems: list[str] = []
    model = str(cfg.get("model", _DEFAULT_MODEL))
    missing = _spec_problem("sentence_transformers")
    if missing:
        problems.append(missing)
    if str(cfg.get("mode", "auto")) != "exact":
        missing = _spec_problem("sklearn.cluster")
        if missing:
            problems.append(f"{missing} (required for mode=kmeans)")
    if bool(cfg.get("local_files_only", True)) and not os.path.isdir(model):
        resolved = False
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model, local_files_only=True)  # local cache lookup only, never downloads
            resolved = True
        except Exception:
            resolved = False
        if not resolved:
            problems.append(
                f"{model}: not a directory and absent from the local HF cache (preflight never downloads)"
            )
    return problems


register_op(FunctionOp("semantic_dedup", _semantic_dedup_op, CONFIG_SCHEMA))
