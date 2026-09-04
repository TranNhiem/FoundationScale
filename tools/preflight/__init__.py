#!/usr/bin/env python3
"""Executable pre-flight blocklist for the FoundationScale-gated E4B launch.

Why this file exists
--------------------
The launch it gates is real: Gemma-4-E4B, GB200 tray (4 GPUs), LoRA and full
fine-tuning, under Megatron-Bridge. The design that this file operationalizes
(``docs/risk-review §4``, "What CLEAR must mean before the re-run") is ten
items of prose; prose cannot block a launch. Every item is implemented here
as a check that runs on the LOGIN NODE with no GPU and no torch import —
the only tensor facts required (names, shapes, dtypes) live in safetensors
headers, which are 8 length bytes plus JSON, readable with stdlib alone.

If a future check genuinely needs torch: import it INSIDE that check, and
treat ``ImportError`` as a BLOCK that names the check, never as a skip.
There is no such check today, and that absence is deliberate.

The clearance algebra (design item 4, the load-bearing one)
-----------------------------------------------------------
Reuse ``foundationscale.gates.core.Verdict`` and ``Coverage`` unchanged —
their members fit every state a pre-flight check can land in. What this file
does NOT reuse is the gate REPORT's tolerance for SKIP next to real passes.
Design item 4 states SKIP/VACUOUS/INAPPLICABLE ==> NOT-VERIFIED, so the
clearance predicate here is, verbatim:

    bool(results) and all(r.verdict is Verdict.PASS and r.coverage.checked > 0)

That is strictly stronger than ``Verdict.blocking``, on purpose: this tool's
only output that anyone acts on is the word CLEAR, and the framework's
founding incident is the word "pass" emitted over an empty examination.
Every non-PASS line renders with a (NOT-VERIFIED) tag so the design's
vocabulary survives contact with stdout.

Exit codes: 0 = CLEAR, 1 = BLOCKED, 2 = the tool itself could not run
(which is also not a clearance).
"""

from __future__ import annotations

from ._artifacts import (
    _canonical_sample_sha256,
    _manifest_hash_for,
    _parse_iso,
    _read_safetensors_header,
    _sha256_and_lines,
)
from ._base import (
    _CHUNK,
    _HERE,
    _MISSING,
    _SAFETENSORS_DTYPE_BYTES,
    _SRC,
    EXIT_BLOCKED,
    EXIT_CLEAR,
    EXIT_TOOL_ERROR,
    TOOL_VERSION,
    Coverage,
    Verdict,
)
from ._cli import (
    main,
)
from ._config import (
    _KIND_CHECKS,
    _SCHEMA,
    K,
    _load_config,
    _post_validate,
    _walk_spec,
)
from ._core import (
    _REGISTRY_ORDER,
    REGISTRY,
    CheckResult,
    _Check,
    _discipline,
    _execute,
    _finalize,
    _is_clear,
    _Lane,
    _register,
    _shared_or_error,
    _stub_fn,
)
from ._errors import (
    ArtifactError,
    ConfigError,
    ToolError,
)
from ._fixtures import (
    _PROBE_SOURCE,
    _build_world,
    _doctored_registry_no_lanes,
    _fresh_world,
    _run_lane_against,
    _safetensors_blob,
    _World,
    _WorldCtx,
)
from ._registrations import (
    _mk,
)
from ._report import (
    _render_report,
    _write_json_record,
)
from ._selftest import (
    run_self_test,
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

__all__ = (
    "ArtifactError",
    "CheckResult",
    "ConfigError",
    "Coverage",
    "EXIT_BLOCKED",
    "EXIT_CLEAR",
    "EXIT_TOOL_ERROR",
    "K",
    "REGISTRY",
    "TOOL_VERSION",
    "ToolError",
    "Verdict",
    "_CHUNK",
    "_Check",
    "_HERE",
    "_KIND_CHECKS",
    "_Lane",
    "_MISSING",
    "_PROBE_SOURCE",
    "_REGISTRY_ORDER",
    "_SAFETENSORS_DTYPE_BYTES",
    "_SCHEMA",
    "_SRC",
    "_World",
    "_WorldCtx",
    "_build_world",
    "_canonical_sample_sha256",
    "_check_conversion_coverage",
    "_check_corpus_wiring",
    "_check_evidence_completeness",
    "_check_frozen_manifest",
    "_check_launch_provenance",
    "_check_lora_probe",
    "_check_schedule",
    "_check_template_audit",
    "_check_training_dynamics",
    "_check_verdict_schema",
    "_discipline",
    "_doctored_registry_no_lanes",
    "_execute",
    "_finalize",
    "_fresh_world",
    "_is_clear",
    "_load_config",
    "_manifest_hash_for",
    "_mk",
    "_parse_iso",
    "_post_validate",
    "_read_safetensors_header",
    "_register",
    "_render_report",
    "_run_lane_against",
    "_safetensors_blob",
    "_sha256_and_lines",
    "_shared_or_error",
    "_stub_fn",
    "_walk_spec",
    "_write_json_record",
    "main",
    "run_self_test",
)
