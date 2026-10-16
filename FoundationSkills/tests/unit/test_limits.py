"""A5.3: README limits are generated from a probe record, and the committed README matches it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from foundationskills.cli import main
from foundationskills.interfaces.fs import limits

ROOT = Path(__file__).resolve().parents[2]


def caps(**over: object) -> dict:
    base = {
        "fs_version": "0-test",
        "rl_reward_kinds": ["mcq_letter"],
        "rl_adapter_support": True,
        "pref_saves_checkpoint": False,
        "axes_measured": True,
        "refused_axes": ["pp", "ep"],
        "backends": ["ddp", "fsdp"],
        "probe_keys": {
            "sft_assistant_label_mask": {
                "status": "measured",
                "value": {"assistant_only": True},
                "reason": "",
            },
            "megatron_bridge": {
                "status": "measured",
                "reason": "not importable here: megatron_core",
                "value": {"fs_lane": True, "megatron_core": False, "megatron_bridge": False},
            },
            "speech": {
                "status": "measured",
                "value": {"module": True, "audio_decode": True},
                "reason": "",
            },
            "video_frame_budget": {
                "status": "measured",
                "reason": "",
                "value": {"decode_backends": ["av"], "samplings": ["uniform"]},
            },
        },
    }
    base.update(over)
    return base


def test_rows_follow_the_measured_fields() -> None:
    text = "\n".join(limits.limit_rows(caps()))
    assert "mcq_letter" in text and "RL trains LoRA adapters" in text
    assert "saves no checkpoint" in text and "assistant tokens only" in text
    assert "refuses PP/EP" in text and "cannot run it" in text and "decode backends av" in text
    assert "speech" not in text  # audio decode works: no limit to report


def test_unmeasured_fields_say_so_instead_of_guessing() -> None:
    rows = "\n".join(
        limits.limit_rows(caps(rl_adapter_support=None, axes_measured=False, probe_keys={}))
    )
    assert "RL LoRA support: UNMEASURED" in rows
    assert "parallel axes (TP/PP/EP/CP): UNMEASURED" in rows
    assert "SFT loss masking: UNMEASURED" in rows and "Megatron: UNMEASURED" in rows


def test_flipped_facts_flip_the_rows() -> None:
    mask = {"status": "measured", "value": {"assistant_only": False}, "reason": "differ"}
    speech = {
        "status": "measured",
        "value": {"module": True, "audio_decode": False},
        "reason": "no soundfile",
    }
    keys = dict(caps()["probe_keys"], sft_assistant_label_mask=mask, speech=speech)
    rows = "\n".join(
        limits.limit_rows(
            caps(rl_adapter_support=False, backends=["ddp", "megatron"], probe_keys=keys)
        )
    )
    assert "RL has no LoRA support" in rows and "assistant-only loss is NOT verified" in rows
    assert "a Megatron backend is usable" in rows and "audio decode unavailable here" in rows


def test_replace_section_requires_markers() -> None:
    with pytest.raises(ValueError, match="markers not found"):
        limits.replace_section("no markers here", "x")


def test_cli_write_then_check_and_stale_detection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "probe.json"
    record.write_text(json.dumps(caps()))
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{limits.BEGIN}\nold\n{limits.END}\noutro\n")
    assert main(["limits", "--probe-json", str(record), "--write", str(readme)]) == 0
    assert main(["limits", "--probe-json", str(record), "--check", str(readme)]) == 0
    record.write_text(json.dumps(caps(rl_adapter_support=False)))
    assert main(["limits", "--probe-json", str(record), "--check", str(readme)]) == 5
    assert readme.read_text().startswith("intro\n") and readme.read_text().endswith("outro\n")
    capsys.readouterr()


def test_committed_readme_matches_the_committed_probe_record() -> None:
    record = json.loads((ROOT / "docs/probe_record.json").read_text())
    text = (ROOT / "README.md").read_text()
    fresh = limits.replace_section(text, limits.render(record, "docs/probe_record.json"))
    assert fresh == text, (
        "README limits are stale: run fskills limits --probe-json docs/probe_record.json "
        "--write README.md"
    )
