"""The Level 3 check plan: every `supported` registry model, reproduced against its reference.

Step "Regress" of the upgrade process (docs/research/upstream_integration.md): before a candidate
upstream profile is promoted, every model the registry calls `supported` must still reproduce its
reference result on the full test split. This module turns the registry into that plan, so adding or
promoting a model adds it to the regression run with no other edit. It is pure (no upstream
imports); the tray runner (validation_campaigns/speech_repro/run_levels.sh) executes each step in
its backend's own container and judges the result with :func:`judge`.
"""

from __future__ import annotations

from dataclasses import dataclass

from foundationscale.upstream.models import (
    MODELS,
    Backend,
    ModelEntry,
    ModelKind,
    SupportStatus,
)

__all__ = ["DecodeStep", "Verdict", "judge", "level3_plan"]

# Reference task -> manifest file name under the cluster's manifests directory.
_TASK_MANIFESTS = {
    "librispeech_test_clean": "librispeech_test_clean.jsonl",
    "librispeech_test_other": "librispeech_test_other.jsonl",
    # VLA: the "manifest" is the LIBERO suite the FS harness (foundationscale.vla.eval) rolls out.
    "libero_spatial": "libero_spatial",
}


@dataclass(frozen=True)
class DecodeStep:
    model_id: str
    upstream_ref: str  # HF repo id or NeMo pretrained name (HF steps resolve it to local weights)
    backend: str  # "hf" | "nemo" | "gr00t" | "openpi": which environment runs the step
    entry_point: str  # python -m module (nemo) or campaign script path (hf)
    args: tuple[str, ...]  # with {model}, {manifest}, {out} placeholders for the runner
    manifest: str
    card_value: float
    tolerance: float
    profile: str  # upstream profile whose environment runs the step


# The profile a step runs in when its entry does not name one.
_DEFAULT_PROFILES = {Backend.HF: "hf-26.04", Backend.NEMO: "nemo-26.08"}

_HF_DECODE_ARGS = ("--model", "{model}", "--manifest", "{manifest}", "--out", "{out}")


def _entry_point(entry: ModelEntry) -> tuple[str, tuple[str, ...]]:
    if entry.backend is Backend.HF and entry.eval_script:
        return entry.eval_script, _HF_DECODE_ARGS
    if entry.backend is Backend.NEMO and entry.kind is ModelKind.AED:
        return (
            "foundationscale.upstream.nemo.decode",
            ("--model", "{model}", "--pnc", "no", "--manifest", "{manifest}", "--out", "{out}"),
        )
    if entry.backend is Backend.NEMO and entry.kind is ModelKind.SPEECH_LLM:
        return (
            "foundationscale.upstream.nemo.salm_decode",
            (
                "--model",
                "{model}",
                "--manifest",
                "{manifest}",
                "--out",
                "{out}",
                "--max-new-tokens",
                "256",
            ),
        )
    if entry.backend is Backend.HF:
        return (
            "validation_campaigns/speech_p4/eval_wer.py",
            (
                "--model",
                "{model}",
                "--processor",
                "{model}",
                "--manifest",
                "{manifest}",
                "--out",
                "{out}",
            ),
        )
    if entry.kind is ModelKind.VLA and entry.backend in (Backend.GR00T, Backend.OPENPI):
        # The FS-owned LIBERO harness drives the backend's own policy server; it rolls out the
        # upstream's published protocol for this backend and reports the success rate in points.
        return (
            "foundationscale.vla.eval",
            (
                "--backend",
                entry.backend.value,
                "--model",
                "{model}",
                "--suite",
                "{manifest}",
                "--out",
                "{out}",
            ),
        )
    raise ValueError(
        f"no decode entry point for {entry.id!r} ({entry.backend.value}/{entry.kind.value})"
    )


def level3_plan(models: tuple[ModelEntry, ...] = MODELS) -> list[DecodeStep]:
    """One decode step per `supported` model, against its reference task. Refuses gaps."""
    steps: list[DecodeStep] = []
    for entry in models:
        if entry.status is not SupportStatus.SUPPORTED:
            continue
        ref = entry.reference
        if ref is None or ref.upstream_value is None:
            raise ValueError(
                f"{entry.id!r} is supported but has no reference value to regress against"
            )
        manifest = _TASK_MANIFESTS.get(ref.task)
        if manifest is None:
            raise ValueError(f"{entry.id!r}: no manifest known for reference task {ref.task!r}")
        module, args = _entry_point(entry)
        steps.append(
            DecodeStep(
                model_id=entry.id,
                upstream_ref=entry.upstream_ref,
                backend=entry.backend.value,
                entry_point=module,
                args=args,
                manifest=manifest,
                card_value=ref.upstream_value,
                tolerance=ref.tolerance,
                profile=entry.profile or _DEFAULT_PROFILES[entry.backend],
            )
        )
    return steps


@dataclass(frozen=True)
class Verdict:
    model_id: str
    measured: float
    card_value: float
    tolerance: float

    @property
    def gap(self) -> float:
        return self.measured - self.card_value

    @property
    def passed(self) -> bool:
        # inclusive boundary; the epsilon absorbs float error (1.61 + 0.3 == 1.9100000000000001)
        return abs(self.gap) <= self.tolerance + 1e-9

    def line(self) -> str:
        status = "PASS" if self.passed else "REGRESSION"
        return (
            f"[{status}] {self.model_id}: {self.measured:.3f} vs card {self.card_value:.3f} "
            f"(gap {self.gap:+.3f}, tolerance {self.tolerance})"
        )


def judge(step: DecodeStep, measured_wer_whisper_normalizer: float) -> Verdict:
    """The Level 3 verdict for one step, from its metric in the card's units (in points).

    For speech steps that is WER under the card's normalizer; for VLA steps it is the success
    rate in percentage points that ``foundationscale.vla.eval`` writes to its report.
    """
    return Verdict(step.model_id, measured_wer_whisper_normalizer, step.card_value, step.tolerance)
