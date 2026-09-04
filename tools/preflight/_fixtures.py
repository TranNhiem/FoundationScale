"""Fixture world for --self-test and for the launch-time red team."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import struct
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._artifacts import (
    _canonical_sample_sha256,
    _manifest_hash_for,
    _sha256_and_lines,
)
from ._base import (
    Coverage,
    Verdict,
)
from ._core import (
    _REGISTRY_ORDER,
    REGISTRY,
    CheckResult,
    _Check,
    _execute,
    _Lane,
)

# ---------------------------------------------------------------------------
# Fixture world for --self-test and for the launch-time red team (item 4)
# ---------------------------------------------------------------------------


@dataclass
class _World:
    """A fully materialized, KNOWN-HEALTHY miniature of every artifact class the
    ten checks read, under one temp dir. cfg is a complete, schema-valid config
    with pins MEASURED from the files actually written (pins are computed, not
    asserted — a fixture whose pins disagree with its files proves nothing).
    Lanes then corrupt one artifact class at a time."""

    root: Path
    cfg: dict[str, Any]
    env: dict[str, str]
    corpus: list[Path]
    model: list[Path]
    logs: list[Path]
    merged: list[Path]
    recipe: Path
    probe_path: Path
    registry_override: Sequence[_Check] | None = None

    # -- lane helpers ---------------------------------------------------------
    def write(self, rel: str, blob: bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        return p

    @staticmethod
    def append_bytes(p: Path, blob: bytes) -> None:
        with p.open("ab") as fh:
            fh.write(blob)

    def probe_sick(self, sick: bool) -> None:
        argv = self.cfg["template"]["probe_command"]
        if sick and "--sick" not in argv:
            argv.append("--sick")
        elif not sick and "--sick" in argv:
            argv.remove("--sick")

    def coverage_map_drop_one(self) -> None:
        data = json.loads(
            (self.root / "artifacts" / "coverage_map.json").read_text(encoding="utf-8")
        )
        # Drop exactly one CONVERTED entry: the resulting uncovered header
        # tensor is the classic silent-coverage gap the map exists to catch.
        data["tensors"] = [
            e
            for e in data["tensors"]
            if not (e.get("coverage") == "converted" and e.get("name") == "encoder.layers.0.w")
        ]
        (self.root / "artifacts" / "coverage_map.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def rewrite_conv_metrics(self, loss: float) -> None:
        rows = [{"iteration": 0, "param_count": 96}, {"iteration": 1, "loss": loss, "lr": 1e-4}]
        (self.root / "artifacts" / "conv_metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    def lora_log_strip(self, cls: str) -> None:
        log = self.root / "artifacts" / "lora_run.log"
        log.write_text(
            "\n".join(ln for ln in log.read_text(encoding="utf-8").splitlines() if cls not in ln)
            + "\n",
            encoding="utf-8",
        )

    def dynamics_patch(self, iteration: int, **fields: Any) -> None:
        path = self.root / "artifacts" / "dynamics.jsonl"
        rows = [
            json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        for r in rows:
            if r.get("iteration") == iteration:
                for k, v in fields.items():
                    if v is None:
                        r.pop(k, None)
                    else:
                        r[k] = v
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _safetensors_blob(tensors: Mapping[str, tuple[int, ...]]) -> bytes:
    header: dict[str, Any] = {}
    data = bytearray()
    for name, shape in tensors.items():
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [len(data), len(data) + nbytes],
        }
        data += b"\x00" * nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + bytes(data)


_PROBE_SOURCE = '''\
"""Fixture stand-in for the FoxBrain template probe (INPUT CONTRACT item 2).

Emits rows_per_file JSON rows for one file. With --sick, cot_span escapes
masked_span on every row: the MUST_FIRE articulation of the CoT-containment
defect the real audit exists to catch."""
import json
import sys


def main() -> int:
    path, rows = sys.argv[1], int(sys.argv[2])
    sick = "--sick" in sys.argv
    for i in range(rows):
        row = {
            "row": i,
            "file": path,
            "tokens_stock": 120 + i,
            "tokens_patched": 132 + i,   # stock-vs-patched diff present
            "cot_span": [4, 12] if sick else [40, 80],
            "masked_span": [32, 96],
        }
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _fresh_world() -> _WorldCtx:
    return _WorldCtx()


class _WorldCtx:
    def __enter__(self) -> _World:
        self._tmp = tempfile.TemporaryDirectory(prefix="preflight-selftest-")
        return _build_world(Path(self._tmp.name))

    def __exit__(self, *exc: Any) -> None:
        self._tmp.cleanup()


def _build_world(root: Path) -> _World:
    art = root / "artifacts"
    art.mkdir(parents=True)
    corpus = []
    corpus_pins = []
    for i in range(4):
        p = art / f"corpus-{i}.jsonl"
        rows = [
            {"idx": i * 100 + j, "text": f"trace {i}-{j}", "cot": f"reasoning {i}-{j}"}
            for j in range(3)
        ]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        sha, lines = _sha256_and_lines(p)
        corpus.append(p)
        corpus_pins.append({"path": str(p), "sha256": sha, "lines": lines})

    model = [
        _fresh := art / "shard-00001.safetensors",
        art / "shard-00002.safetensors",
    ]
    model[0].write_bytes(
        _safetensors_blob({"encoder.layers.0.w": (4, 8), "encoder.layers.1.w": (4, 8)})
    )
    model[1].write_bytes(_safetensors_blob({"encoder.layers.2.w": (8, 4)}))
    total_bytes = sum(p.stat().st_size for p in model)

    run_config = art / "run_config.json"
    run_config.write_text('{"recipe": "e4b-fixture"}\n', encoding="utf-8")

    chat = art / "chat_template.jinja"
    chat.write_text("{{ bos }}{% for m in messages %}...{% endfor %}\n", encoding="utf-8")

    probe_path = art / "fixture_template_probe.py"
    probe_path.write_text(_PROBE_SOURCE, encoding="utf-8")

    recipe = root / "recipe" / "train.py"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(
        "def main():\n    jsonls = _env_jsonls('FOXBRAIN_SFT_JSONLS')\n    return load(jsonls)\n",
        encoding="utf-8",
    )

    coverage_map = art / "coverage_map.json"
    coverage_map.write_text(
        json.dumps(
            {
                "tensors": [
                    {"name": "encoder.layers.0.w", "bytes": 128, "coverage": "converted"},
                    {"name": "encoder.layers.1.w", "bytes": 128, "coverage": "converted"},
                    {"name": "encoder.layers.2.w", "bytes": 128, "coverage": "converted"},
                    # lm_head shares storage with the embedding table: allow-listed, and
                    # the ground (tie_word_embeddings==true) is checked against the HF config.
                    {"name": "lm_head.weight", "coverage": "allowlist", "rule": "tied"},
                ]
            }
        ),
        encoding="utf-8",
    )

    hf_cfg = art / "hf_config.json"
    hf_cfg.write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_experts": 4,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )

    conv_metrics = art / "conv_metrics.jsonl"
    conv_metrics.write_text(
        json.dumps({"iteration": 0, "param_count": 96})
        + "\n"
        + json.dumps({"iteration": 1, "loss": 2.15, "lr": 1e-4})
        + "\n",
        encoding="utf-8",
    )

    lora_log = art / "lora_run.log"
    lora_log.write_text(
        "Adding lora to q_proj of layer 0\n"
        "Adding lora to kv_proj of layer 0\n"
        "trainable params: 1,234,567 || all params: 8,000,000,000 || trainable%: 0.0154\n",
        encoding="utf-8",
    )

    lora_metrics = art / "lora_probe.jsonl"
    lora_rows = [{"iteration": i} for i in range(1, 21)]
    lora_rows[-1]["lora_b_norm"] = {"q_proj": 0.53, "kv_proj": 0.41}
    lora_metrics.write_text("\n".join(json.dumps(r) for r in lora_rows) + "\n", encoding="utf-8")

    delta = art / "delta_audit.json"
    delta.write_text(
        json.dumps(
            {
                "q_proj": {"delta_l2": 1.7e-3, "tensors_checked": 4},
                "kv_proj": {"delta_l2": 9.2e-4, "tensors_checked": 4},
            }
        ),
        encoding="utf-8",
    )

    merged_dir = root / "merged"
    merged_dir.mkdir()
    merged = [merged_dir / "model-00001.safetensors", merged_dir / "model-00002.safetensors"]
    merged[0].write_bytes(b"\x00" * 1000)
    merged[1].write_bytes(b"\x00" * 500)
    # A self-index that LIES about the total — planted in every fixture world
    # so a regression to self-index parity turns red everywhere at once.
    (merged_dir / "model.safetensors.index.json").write_text(
        '{"metadata": {"total_size": 999999999}, "weight_map": {}}', encoding="utf-8"
    )

    # The external pin is MEASURED HERE, at world-build time, by walking the
    # same bytes-on-disk quantity that lora_probe independently walks at check
    # time. Legitimate here and forbidden inside the check, for mirror-image
    # reasons: _build_world is the operator's stand-in — an authority EXTERNAL
    # to the check under test, whose pins are measurements of what it wrote,
    # never the artifact's own claims — whereas a pin derived inside the check
    # from the same walk would be self-referential (pin == observed trivially,
    # both control halves green on a detector that verifies nothing: exactly
    # the hole the planted lying index exists to expose). The walk prices the
    # planted index like any other regular file, per the check's contract, so
    # this pin is deliberately NOT 1500 (= 1000 + 500 shards-only arithmetic,
    # which forgot the planted file's real bytes).
    pinned_merged_total_bytes = sum(p.stat().st_size for p in merged_dir.rglob("*") if p.is_file())

    now = time.time()
    dyn_rows = []
    for i in range(1, 111):
        dyn_rows.append(
            {
                "iteration": i,
                "loss": max(0.6, 3.0 * (0.97**i)),
                "lr": 1e-4 * (0.999**i),
                "iter_time_s": 2.5,
                "elapsed_s": i * 2.5,
            }
        )
    dyn_path = art / "dynamics.jsonl"
    dyn_path.write_text("\n".join(json.dumps(r) for r in dyn_rows) + "\n", encoding="utf-8")

    log_dir = root / "logs"
    log_dir.mkdir()
    logs = []
    for r in range(4):
        p = log_dir / f"job.rank{r}.out"
        p.write_text(f"step=110 rank={r} max_memory_mib={4096.5 - r * 10}\n", encoding="utf-8")
        logs.append(p)

    guard = root / "recipe" / "checkpointing.py"
    guard.write_text(
        "def resume(ckpt, manifest_hash):\n"
        "    if ckpt.manifest_hash != manifest_hash:\n"
        "        raise ManifestMismatchError('checkpoint names a different preflight')\n",
        encoding="utf-8",
    )

    cfg: dict[str, Any] = {
        "run_name": "fixture-e4b",
        "world_size": 4,
        "frozen": {
            "model": {
                "files": [str(p) for p in model],
                "tensor_count": 3,
                "total_bytes": total_bytes,
            },
            "corpus": {"files": corpus_pins},
            "run_config": {
                "path": str(run_config),
                "sha256": hashlib.sha256(run_config.read_bytes()).hexdigest(),
            },
        },
        "template": {
            "probe_command": [sys.executable, str(probe_path), "{file}", "{rows}"],
            "rows_per_file": 2,
            "files": [str(corpus[0]), str(corpus[1])],
            "keep_cot_env": "FOXBRAIN_GEMMA4_KEEP_COT",
            "chat_template_path": str(chat),
            "chat_template_md5": hashlib.md5(chat.read_bytes()).hexdigest(),
        },
        "corpus_wiring": {
            "env_var": "FOXBRAIN_SFT_JSONLS",
            "recipe_files": [str(recipe)],
            "attestation_path": str(art / "attestation.json"),
        },
        "conversion": {
            "hf_config_json": str(hf_cfg),
            "coverage_map_json": str(coverage_map),
            "iter_metrics_jsonl": str(conv_metrics),
            "iter1_loss_band": [1.0, 3.0],
            "expected_param_count": 96,
            "divisibility": [
                {"field": "num_attention_heads", "divisible_by": 4},
                {"field": "num_key_value_heads", "divisible_by": 4},
                {"field": "num_experts", "divisible_by": 4},
                {"field": "tie_word_embeddings", "equals": True},
            ],
            "tied_grounding": "tie_word_embeddings",
            "shared_kv_grounding": None,
        },
        "lora": {
            "run_log": str(lora_log),
            "target_classes": ["q_proj", "kv_proj"],
            "trainable_band": [1_000_000, 2_000_000],
            "probe_metrics_jsonl": str(lora_metrics),
            "delta_audit_json": str(delta),
            "merged_dir": str(merged_dir),
            "pinned_merged_total_bytes": pinned_merged_total_bytes,
            "expected_iters": 20,
        },
        "schedule": {
            "train_iters": 1000,
            "lr_decay_iters": 1000,
            "save_interval": 250,
            "explicit_final_save": False,
            "smoke": False,
        },
        "evidence": {
            "log_glob": str(log_dir / "*.out"),
            "mem_regex": r"max_memory_mib=([0-9.]+)",
            "max_log_age_s": 3600,
            "slurm_job_id": None,
            "allow_no_slurm": True,
            "slurm_absent_reason": "fixture host has no Slurm; declared, per the opt-out contract",
        },
        "dynamics": {
            "metrics_jsonl": str(dyn_path),
            "bands": [
                {"iteration": 1, "lo": 1.0, "hi": 3.5},
                {"iteration": 50, "lo": 0.5, "hi": 2.0},
                {"iteration": 100, "lo": 0.5, "hi": 1.2},
            ],
            "hard_floor": 0.1,
        },
        "provenance": {
            "checkpoint_dirs": [str(root / "ckpt-probe")],
            "resume_guard_files": [str(guard)],
            "min_walltime_s": 1,
            "job_window_utc": [
                _dt.datetime.fromtimestamp(now - 300, _dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                _dt.datetime.fromtimestamp(now + 86400, _dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            ],
            "mtime_slack_s": 5,
            "artifacts": [
                str(root / "ckpt-probe" / "provenance.json"),
                str(merged[0]),
                str(logs[0]),
            ],
            "walltime_jsonl": str(dyn_path),
        },
    }

    env = {
        "FOXBRAIN_SFT_JSONLS": os.pathsep.join(str(p) for p in corpus),
        "FOXBRAIN_GEMMA4_KEEP_COT": "1",
    }

    world = _World(
        root=root,
        cfg=cfg,
        env=env,
        corpus=corpus,
        model=model,
        logs=logs,
        merged=merged,
        recipe=recipe,
        probe_path=probe_path,
    )

    # Dependent artifacts, computed with the SAME helper the check uses —
    # pins measured, never asserted:
    cfg_path = root / "preflight.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    manifest_sha, _payload = _manifest_hash_for(cfg["frozen"], cfg_sha)
    ckpt = root / "ckpt-probe"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "provenance.json").write_text(
        json.dumps({"manifest_hash": manifest_sha}), encoding="utf-8"
    )
    (art / "attestation.json").write_text(
        json.dumps(
            {
                "reader": "fixture-human",
                "sample_sha256": _canonical_sample_sha256(corpus[0]),
                "note": "batch-0 sample read; this fixture stands in for the human act",
            }
        ),
        encoding="utf-8",
    )
    world._cfg_sha = cfg_sha  # type: ignore[attr-defined]
    return world


def _doctored_registry_no_lanes() -> list[_Check]:
    """A registry copy in which the FIRST peer ships no fire lane: check 4's own
    positive control. Only ever synthesized; never installed."""
    out = []
    for c in _REGISTRY_ORDER:
        if c.id == "verdict_schema":
            continue
        out.append(_Check(c.id, c.title, c.section, c.fn, ()))
        break
    return out


def _run_lane_against(peer: _Check, lane: _Lane) -> CheckResult:
    """Fresh healthy world -> inject the lane's defect -> run ONLY the target.
    The shared manifest state is established by a real frozen_manifest run on
    the healthy world FIRST: a lane whose healthy precondition fails is
    inconclusive-by-ERROR, never silently counted."""
    with _fresh_world() as world:
        shared: dict[str, Any] = {"_config_sha256": world._cfg_sha}  # type: ignore[attr-defined]
        baseline = _execute(REGISTRY["frozen_manifest"], world.cfg, world.env, shared)
        if baseline.verdict is not Verdict.PASS:
            return CheckResult(
                peer.id,
                peer.title,
                Verdict.ERROR,
                Coverage.none("units"),
                f"red-team precondition failed: frozen_manifest did not pass on the "
                f"healthy fixture world ({baseline.detail}) — the lane is inconclusive, "
                f"which is not a proven firing",
            )
        lane.apply(world)
        peers = world.registry_override if world.registry_override is not None else None
        return _execute(peer, world.cfg, world.env, shared, registry=peers)
