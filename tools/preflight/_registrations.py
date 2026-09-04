"""Check and lane registrations. ORDER MATTERS -- see the banner comment."""

from __future__ import annotations

import os
from pathlib import Path

from ._core import (
    _Lane,
    _register,
)
from ._fixtures import (
    _doctored_registry_no_lanes,
)
from .items.conversion_coverage import (
    _check_conversion_coverage,
)
from .items.corpus_wiring import (
    _check_corpus_wiring,
)
from .items.evidence import (
    _check_evidence_completeness,
)
from .items.frozen_manifest import (
    _check_frozen_manifest,
)
from .items.launch_provenance import (
    _check_launch_provenance,
)
from .items.lora_probe import (
    _check_lora_probe,
)
from .items.schedule import (
    _check_schedule,
)
from .items.template_audit import (
    _check_template_audit,
)
from .items.training_dynamics import (
    _check_training_dynamics,
)
from .items.verdict_schema import (
    _check_verdict_schema,
)

# ---------------------------------------------------------------------------
# Registrations (order matters: frozen_manifest MUST run before its consumers)
# ---------------------------------------------------------------------------


def _mk(check_id, title, section, fn, lanes):
    return _register(check_id, title, section, lanes=lanes)(fn)


_mk(
    "frozen_manifest",
    "Frozen manifest (design item 1)",
    "frozen",
    _check_frozen_manifest,
    lanes=[
        _Lane(
            "corpus-bytes-tampered",
            "a corpus file's bytes no longer match the pinned sha256",
            lambda w: w.append_bytes(w.corpus[0], b"# tampered\n"),
        ),
        _Lane(
            "model-file-missing", "a declared model shard is absent", lambda w: w.model[0].unlink()
        ),
    ],
)

_mk(
    "template_audit",
    "Template audit / CoT containment (item 2)",
    "template",
    _check_template_audit,
    lanes=[
        _Lane(
            "cot-escapes-mask",
            "probe reports cot_span outside masked_span on every row",
            lambda w: w.probe_sick(True),
        ),
        _Lane(
            "keep-cot-zero",
            "FOXBRAIN_GEMMA4_KEEP_COT=0 in the launch environment",
            lambda w: w.env.__setitem__("FOXBRAIN_GEMMA4_KEEP_COT", "0"),
        ),
    ],
)

_mk(
    "corpus_wiring",
    "Corpus wiring / banner-manifest equality (item 3)",
    "corpus_wiring",
    _check_corpus_wiring,
    lanes=[
        _Lane(
            "recipe-loses-env-call",
            "no declared recipe file names _env_jsonls(",
            lambda w: w.recipe.write_text(
                "# corpus loading was refactored; env reader removed\n", encoding="utf-8"
            ),
        ),
        _Lane(
            "resolved-env-drifts",
            "FOXBRAIN_SFT_JSONLS names a file the frozen manifest does not",
            lambda w: w.env.__setitem__(
                "FOXBRAIN_SFT_JSONLS",
                os.pathsep.join(str(p) for p in w.corpus + [w.root / "phantom.jsonl"]),
            ),
        ),
    ],
)

_mk(
    "verdict_schema",
    "Verdict schema / launch-time red team (item 4)",
    None,
    _check_verdict_schema,
    lanes=[
        # check 4's own positive control: a registry that contains a peer with NO
        # fire lane must fail it by name. Exercised by the self-test harness with
        # a doctored registry; declared here as a lane so the runtime red team
        # covers meta-failure too.
        _Lane(
            "peer-ships-no-fire-lane",
            "a peer check declares zero MUST_FIRE lanes",
            lambda w: setattr(w, "registry_override", _doctored_registry_no_lanes()),
        ),
    ],
)

_mk(
    "conversion_coverage",
    "Conversion coverage map (item 5)",
    "conversion",
    _check_conversion_coverage,
    lanes=[
        _Lane(
            "map-drops-tensor",
            "the coverage map silently omits one header tensor",
            lambda w: w.coverage_map_drop_one(),
        ),
        _Lane(
            "iter1-loss-out-of-band",
            "iter-1 loss lands outside the pinned band",
            lambda w: w.rewrite_conv_metrics(loss=9.9),
        ),
    ],
)

_mk(
    "lora_probe",
    "LoRA probe 20-iter (item 6)",
    "lora",
    _check_lora_probe,
    lanes=[
        _Lane(
            "intended-class-silent",
            "an intended target class has zero 'Adding lora to' lines",
            lambda w: w.lora_log_strip("kv_proj"),
        ),
        _Lane(
            "merged-bytes-mismatch",
            "merged HF export bytes no longer match the external pin",
            lambda w: w.append_bytes(w.merged[0], b"\x00"),
        ),
    ],
)

_mk(
    "schedule_consistency",
    "Schedule banner (item 7)",
    "schedule",
    _check_schedule,
    lanes=[
        _Lane(
            "lr-decay-mismatch",
            "lr_decay_iters != train_iters",
            lambda w: w.cfg["schedule"].__setitem__(
                "lr_decay_iters", w.cfg["schedule"]["train_iters"] + 7
            ),
        ),
    ],
)

_mk(
    "evidence_completeness",
    "Evidence completeness (item 8)",
    "evidence",
    _check_evidence_completeness,
    lanes=[
        _Lane(
            "rank-log-missing", "fewer per-rank logs than world size", lambda w: w.logs[0].unlink()
        ),
        _Lane(
            "log-stale",
            "a per-rank log violates .out mtime liveness",
            lambda w: os.utime(w.logs[1], (946684800, 946684800)),
        ),
    ],
)

_mk(
    "training_dynamics",
    "Training dynamics (item 9)",
    "dynamics",
    _check_training_dynamics,
    lanes=[
        _Lane(
            "loss-floor-breach",
            "a loss below the hard floor appears mid-run",
            lambda w: w.dynamics_patch(42, loss=0.05),
        ),
        _Lane(
            "lr-row-missing",
            "an evidence row carries no lr",
            lambda w: w.dynamics_patch(7, lr=None),
        ),
    ],
)

_mk(
    "launch_provenance",
    "Launch provenance (item 10)",
    "provenance",
    _check_launch_provenance,
    lanes=[
        _Lane(
            "checkpoint-hash-mismatch",
            "a checkpoint's embedded manifest hash names a different preflight",
            lambda w: (
                Path(w.cfg["provenance"]["checkpoint_dirs"][0]) / "provenance.json"
            ).write_text('{"manifest_hash": "' + "0" * 64 + '"}', encoding="utf-8"),
        ),
        _Lane(
            "artifact-outside-window",
            "a declared artifact's mtime lies outside the job window",
            lambda w: os.utime(w.merged[0], (946684800, 946684800)),
        ),
    ],
)
