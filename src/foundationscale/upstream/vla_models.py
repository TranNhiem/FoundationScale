"""The VLA model registry: upstream VLA policies FoundationScale supports, and on what evidence.

Same contract as the speech registry in :mod:`foundationscale.upstream.models` (shared
:class:`ModelEntry` / :class:`ReferenceResult` types, the same :func:`registry_problems`
checks, the same :func:`foundationscale.upstream.levels.level3_plan` and ``Verdict``), kept in its
own table so the speech transcription stays pinned exactly as its owners wrote it.

A VLA entry is ``SUPPORTED`` only when FoundationScale reproduced the upstream's PUBLISHED result
on the same benchmark and protocol, on our hardware; that run is then the regression test every
upgrade must pass (``level3_plan(VLA_MODELS)``). Metric: LIBERO success rate in percentage
points. Tolerance is fixed from the published rate and the two trial counts -- a two-sample 95%
binomial margin, 1.96 * sqrt(p(1-p)/n_published + p(1-p)/n_measured) -- not from our measured
gap: see :func:`binomial_tolerance` and tests/upstream/test_vla_models.py.
"""

from __future__ import annotations

import math

from foundationscale.upstream.models import (
    Backend,
    ModelEntry,
    ModelKind,
    ReferenceResult,
    SupportStatus,
)

__all__ = ["VLA_MODELS", "binomial_tolerance"]


def binomial_tolerance(rate: float, n_published: int, n_measured: int, z: float = 1.96) -> float:
    """The two-sample binomial margin, in percentage points, for a rate measured twice."""
    if not 0.0 < rate < 1.0 or n_published < 1 or n_measured < 1:
        raise ValueError(
            f"rate {rate!r} must be in (0, 1) and trial counts {n_published!r}, {n_measured!r} "
            ">= 1 -- a margin over no trials or a degenerate rate is not a tolerance"
        )
    variance = rate * (1.0 - rate)
    return 100.0 * z * math.sqrt(variance / n_published + variance / n_measured)


VLA_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        id="gr00t-n1.7-libero-spatial",
        backend=Backend.GR00T,
        upstream_ref="nvidia/GR00T-N1.7-LIBERO",
        kind=ModelKind.VLA,
        license_weights="nvidia-open-model-license",
        attribution="NVIDIA Isaac GR00T N1.7 (libero_spatial subfolder); backbone "
        "nvidia/Cosmos-Reason2-2B (gated)",
        requires=(("torch", "==2.9.0"), ("transformers", "==4.57.3")),
        reference=ReferenceResult(
            task="libero_spatial",
            metric="success_rate_pct",
            # Isaac-GR00T examples/LIBERO/README.md: 195/200. The README labels it 97.65%;
            # 195/200 is 97.5%, and the counts are what is transcribed.
            upstream_value=97.5,
            source="Isaac-GR00T examples/LIBERO/README.md (spatial 195/200)",
            # binomial_tolerance(0.975, 200, 606) = 2.49 -> 2.5
            tolerance=2.5,
            measured=97.03,
            measured_note="588/606 rollouts on GB200 aarch64 (SDPA, real Cosmos backbone), GR00T's "
            "own server + rollout client, 3 passes of 10 tasks x 20 episodes, 2026-10-10",
        ),
        status=SupportStatus.SUPPORTED,
        notes="flash-attn has no aarch64 build: use_flash_attention=False (SDPA) is the only "
        "runtime change",
    ),
    ModelEntry(
        id="openpi-pi05-libero",
        backend=Backend.OPENPI,
        upstream_ref="gs://openpi-assets/checkpoints/pi05_libero",
        kind=ModelKind.VLA,
        license_weights="gemma",
        attribution="Physical Intelligence openpi pi0.5 (PaliGemma-derived; Gemma terms)",
        requires=(("jax", "==0.5.3"), ("transformers", "==4.53.2")),
        reference=ReferenceResult(
            task="libero_spatial",
            metric="success_rate_pct",
            upstream_value=98.8,
            source="openpi examples/libero/README.md (pi0.5 @ 30k, 50 trials x 10 tasks)",
            # binomial_tolerance(0.988, 500, 500) = 1.35 -> 1.4
            tolerance=1.4,
            measured=98.4,
            measured_note="492/500 rollouts on GB200 aarch64, openpi's own serve_policy.py (JAX) "
            "+ examples/libero client, seed 7, 2026-10-10",
        ),
        status=SupportStatus.SUPPORTED,
    ),
    ModelEntry(
        id="gr00t-n1.7-libero-goal",
        backend=Backend.GR00T,
        upstream_ref="nvidia/GR00T-N1.7-LIBERO",
        kind=ModelKind.VLA,
        license_weights="nvidia-open-model-license",
        attribution="NVIDIA Isaac GR00T N1.7 (libero_goal subfolder); backbone "
        "nvidia/Cosmos-Reason2-2B (gated)",
        requires=(("torch", "==2.9.0"), ("transformers", "==4.57.3")),
        reference=ReferenceResult(
            task="libero_goal",
            metric="success_rate_pct",
            # Isaac-GR00T examples/LIBERO/README.md: 195/200 (README label 97.5%);
            # the counts are what is transcribed.
            upstream_value=97.5,
            source="Isaac-GR00T examples/LIBERO/README.md (goal 195/200)",
            # binomial_tolerance(0.975, 200, 200) = 3.06 -> 3.1
            tolerance=3.1,
            measured=97.5,
            measured_note="195/200 on GB200 aarch64 through the FS harness "
            "(python -m foundationscale.vla.eval --backend gr00t: seeded resets, 720 env steps), "
            "0 episode errors, 2026-10-12",
        ),
        status=SupportStatus.SUPPORTED,
        notes="flash-attn has no aarch64 build: use_flash_attention=False (SDPA) is the only "
        "runtime change",
    ),
    ModelEntry(
        id="gr00t-n1.7-libero-object",
        backend=Backend.GR00T,
        upstream_ref="nvidia/GR00T-N1.7-LIBERO",
        kind=ModelKind.VLA,
        license_weights="nvidia-open-model-license",
        attribution="NVIDIA Isaac GR00T N1.7 (libero_object subfolder); backbone "
        "nvidia/Cosmos-Reason2-2B (gated)",
        requires=(("torch", "==2.9.0"), ("transformers", "==4.57.3")),
        reference=ReferenceResult(
            task="libero_object",
            metric="success_rate_pct",
            # Isaac-GR00T examples/LIBERO/README.md: 197/200 (README label 98.45%);
            # the counts are what is transcribed.
            upstream_value=98.5,
            source="Isaac-GR00T examples/LIBERO/README.md (object 197/200)",
            # binomial_tolerance(0.985, 200, 200) = 2.38 -> 2.4
            tolerance=2.4,
            measured=99.5,
            measured_note="199/200 on GB200 aarch64 through the FS harness "
            "(python -m foundationscale.vla.eval --backend gr00t: seeded resets, 720 env steps), "
            "0 episode errors, 2026-10-12",
        ),
        status=SupportStatus.SUPPORTED,
        notes="flash-attn has no aarch64 build: use_flash_attention=False (SDPA) is the only "
        "runtime change",
    ),
    ModelEntry(
        id="gr00t-n1.7-libero-10",
        backend=Backend.GR00T,
        upstream_ref="nvidia/GR00T-N1.7-LIBERO",
        kind=ModelKind.VLA,
        license_weights="nvidia-open-model-license",
        attribution="NVIDIA Isaac GR00T N1.7 (libero_10 subfolder); backbone "
        "nvidia/Cosmos-Reason2-2B (gated)",
        requires=(("torch", "==2.9.0"), ("transformers", "==4.57.3")),
        reference=ReferenceResult(
            task="libero_10",
            metric="success_rate_pct",
            # Isaac-GR00T examples/LIBERO/README.md: 189/200 (README label 94.35%);
            # the counts are what is transcribed.
            upstream_value=94.5,
            source="Isaac-GR00T examples/LIBERO/README.md (10 189/200)",
            # binomial_tolerance(0.945, 200, 200) = 4.47 -> 4.5
            tolerance=4.5,
            measured=90.5,
            measured_note="181/200 on GB200 aarch64 through the FS harness "
            "(python -m foundationscale.vla.eval --backend gr00t: seeded resets, 720 env steps), "
            "0 episode errors, 2026-10-12",
        ),
        status=SupportStatus.SUPPORTED,
        notes="flash-attn has no aarch64 build: use_flash_attention=False (SDPA) is the only "
        "runtime change",
    ),
    ModelEntry(
        id="openpi-pi05-libero-goal",
        backend=Backend.OPENPI,
        upstream_ref="gs://openpi-assets/checkpoints/pi05_libero",
        kind=ModelKind.VLA,
        license_weights="gemma",
        attribution="Physical Intelligence openpi pi0.5 (PaliGemma-derived; Gemma terms)",
        requires=(("jax", "==0.5.3"), ("transformers", "==4.53.2")),
        reference=ReferenceResult(
            task="libero_goal",
            metric="success_rate_pct",
            upstream_value=98.0,
            source="openpi examples/libero/README.md (pi0.5 @ 30k, 50 trials x 10 tasks)",
            # binomial_tolerance(0.98, 500, 500) = 1.74 -> 1.8
            tolerance=1.8,
            measured=97.2,
            measured_note="486/500 on GB200 aarch64 through the FS harness "
            "(python -m foundationscale.vla.eval --backend openpi: suite init states, seed 7), "
            "0 episode errors, 2026-10-12",
        ),
        status=SupportStatus.SUPPORTED,
    ),
    ModelEntry(
        id="openpi-pi05-libero-object",
        backend=Backend.OPENPI,
        upstream_ref="gs://openpi-assets/checkpoints/pi05_libero",
        kind=ModelKind.VLA,
        license_weights="gemma",
        attribution="Physical Intelligence openpi pi0.5 (PaliGemma-derived; Gemma terms)",
        requires=(("jax", "==0.5.3"), ("transformers", "==4.53.2")),
        reference=ReferenceResult(
            task="libero_object",
            metric="success_rate_pct",
            upstream_value=98.2,
            source="openpi examples/libero/README.md (pi0.5 @ 30k, 50 trials x 10 tasks)",
            # binomial_tolerance(0.982, 500, 500) = 1.65 -> 1.7
            tolerance=1.7,
            measured=99.2,
            measured_note="496/500 on GB200 aarch64 through the FS harness "
            "(python -m foundationscale.vla.eval --backend openpi: suite init states, seed 7), "
            "0 episode errors, 2026-10-12",
        ),
        status=SupportStatus.SUPPORTED,
    ),
    ModelEntry(
        id="openpi-pi05-libero-10",
        backend=Backend.OPENPI,
        upstream_ref="gs://openpi-assets/checkpoints/pi05_libero",
        kind=ModelKind.VLA,
        license_weights="gemma",
        attribution="Physical Intelligence openpi pi0.5 (PaliGemma-derived; Gemma terms)",
        requires=(("jax", "==0.5.3"), ("transformers", "==4.53.2")),
        reference=ReferenceResult(
            task="libero_10",
            metric="success_rate_pct",
            upstream_value=92.4,
            source="openpi examples/libero/README.md (pi0.5 @ 30k, 50 trials x 10 tasks)",
            # binomial_tolerance(0.924, 500, 500) = 3.28 -> 3.3
            tolerance=3.3,
            measured=92.4,
            measured_note="462/500 on GB200 aarch64 through the FS harness "
            "(python -m foundationscale.vla.eval --backend openpi: suite init states, seed 7), "
            "0 episode errors, 2026-10-12",
        ),
        status=SupportStatus.SUPPORTED,
    ),
)
