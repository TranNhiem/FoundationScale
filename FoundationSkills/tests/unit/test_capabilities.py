
from __future__ import annotations

import pytest

from foundationskills.interfaces.fs.capabilities import FSCapabilities, probe


def base_caps(**overrides: object) -> FSCapabilities:
    kwargs = dict(
        available=True,
        fs_version="0-test",
        train_flags=frozenset({"--model", "--objective"}),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("grpo", "ppo"),
        families={"qwen": ("llm",)},
        backends=("ddp", "fsdp"),
        notes=(),
        errors=(),
    )
    kwargs.update(overrides)
    return FSCapabilities(**kwargs)  # type: ignore[arg-type]


def test_probe_importable_or_records_error() -> None:
    caps = probe(deep=False)
    if not caps.available:
        assert caps.errors, "unavailable FS must record an error, never silently skip"
        assert caps.fs_version is None
        return
    assert caps.errors == () or isinstance(caps.errors, tuple)
    assert "--model" in caps.train_flags
    assert "sft" in caps.train_objectives
    assert "fsdp" in caps.sharding_strategies
    if "megatron" in caps.backends:
        assert any("megatron" == s or "megatron" in s for s in (caps.backends + caps.notes)) or "megatron" in caps.backends


@pytest.mark.slow
def test_probe_deep_is_a_measurement() -> None:
    caps = probe(deep=True)
    assert isinstance(caps, FSCapabilities)
    if not caps.available:
        assert caps.errors and not caps.axes_measured
        return
    # Every axis classified, and each refusal carries FS's own reason naming the axis.
    assert caps.axes_measured, caps.notes
    assert set(caps.executed_axes) | set(caps.refused_axes) == {"tp", "pp", "ep", "cp"}
    for axis in caps.refused_axes:
        assert any(n.startswith(f"axis {axis} refused by FS:") and f"{axis}=" in n for n in caps.notes)


@pytest.mark.slow
def test_probe_axes_must_fire_on_broken_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUST_FIRE: a probe command line that fails for an unrelated reason (here,
    no --dataset) must yield UNMEASURED axes, never "all refused"."""
    from foundationskills.interfaces.fs import capabilities as mod

    broken = tuple(a for a in mod._PROBE_BASE if a not in ("--dataset", "probe.jsonl"))
    monkeypatch.setattr(mod, "_PROBE_BASE", broken)
    notes: list[str] = []
    executed, refused, measured = mod._probe_axes(True, None, notes)
    assert (executed, refused, measured) == ((), (), False)
    assert any("control failed" in n for n in notes)


def test_check_stage_and_algorithm_semantics() -> None:
    caps = base_caps()
    assert caps.check("pretrain") is None
    assert caps.check("cpt") is None
    assert caps.check("sft") is None
    assert caps.check("rl", algorithm="grpo") is None
    missing_alg = caps.check("preference", algorithm="dpo-unknown")
    assert missing_alg and "missing: RL algorithm dpo-unknown" in missing_alg
    no_sft = base_caps(train_objectives=("lm",))
    assert no_sft.check("sft") and "sft objective" in (no_sft.check("sft") or "")


def test_check_backend_axes_and_multi_gpu_rl() -> None:
    caps = base_caps()
    assert caps.check("sft", backend="megatron") == "missing: megatron backend (installed FS has: ddp, fsdp)"
    assert caps.check("rl", algorithm="ppo", multi_gpu_rl=True) == "missing: multi-GPU RL (FS RLTrainer is single-device)"
    pp_refused = caps.check("sft", pp=2)
    assert pp_refused and "pp>1 is REFUSED" in pp_refused
    unmeasured = base_caps(axes_measured=False, refused_axes=())
    assert "pp unmeasured; run probe(deep=True)" in (unmeasured.check("sft", pp=2) or "")
    assert unmeasured.check("sft", tp=2) is None


def test_capabilities_to_dict_round_shape() -> None:
    caps = base_caps(notes=("objective vocabulary unmeasured",))
    data = caps.to_dict()
    assert data["train_flags"] == ["--model", "--objective"]
    assert data["families"] == {"qwen": ["llm"]}
    assert set(data["refused_axes"]) == {"pp", "ep"}
