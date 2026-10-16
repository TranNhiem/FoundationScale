"""A5.1: each former blind spot gets a probe key, measured or unmeasured (never inferred)."""
from __future__ import annotations

import sys

import pytest
from foundationskills.interfaces.fs import capabilities as cap


def _assert_shape(key: dict) -> None:
    assert key["status"] in ("measured", "unmeasured")
    assert isinstance(key["reason"], str)
    if key["status"] == "unmeasured":
        assert key["value"] is None and key["reason"]


def test_probe_reports_every_key_with_a_valid_shape() -> None:
    keys = cap.probe().probe_keys
    assert tuple(keys) == cap.PROBE_KEYS
    for key in keys.values():
        _assert_shape(key)


def test_label_mask_is_measured_behaviourally() -> None:
    pytest.importorskip("torch")
    key = cap._probe_label_mask_key()
    assert key == {"status": "measured", "value": {"assistant_only": True}, "reason": ""}


def test_label_mask_reports_a_wrong_mask_as_measured_false(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    import foundationscale.train.conversation as conv

    def all_masked(ids, _attn, _markers):
        return torch.full_like(ids, -100)

    monkeypatch.setattr(conv, "assistant_label_mask", all_masked)
    key = cap._probe_label_mask_key()
    assert key["status"] == "measured" and key["value"] == {"assistant_only": False}
    assert "differ" in key["reason"]


def test_a_probe_that_cannot_measure_is_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    import foundationscale.video as video

    def broken() -> list[str]:
        raise OSError("no decoder could be loaded")

    monkeypatch.setattr(video, "available_backends", broken)
    key = cap._probe_video_key()
    _assert_shape(key)
    assert key["status"] == "unmeasured" and "no decoder could be loaded" in key["reason"]


def test_megatron_backend_needs_the_lane_and_megatron_core(monkeypatch: pytest.MonkeyPatch) -> None:
    present = {"foundationscale.rl.megatron": True, "megatron.core": False,
               "megatron.bridge": False}
    monkeypatch.setattr(cap, "_has_module", lambda name: present[name])
    key = cap._probe_megatron_key()
    assert key["value"] == {"fs_lane": True, "megatron_core": False, "megatron_bridge": False}
    assert "megatron_core" in key["reason"]
    # a lane module alone is not evidence
    assert "megatron" not in cap._probe_backends(("ddp", "fsdp"), key)

    present.update({"megatron.core": True, "megatron.bridge": True})
    key = cap._probe_megatron_key()
    assert key["reason"] == "" and cap._probe_backends(("ddp",), key) == ("ddp", "megatron")


def test_a_crashing_key_probe_is_recorded_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dict:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(cap, "_probe_speech_key", boom)
    errors: list[str] = []
    keys = cap._probe_keys(errors)
    assert keys["speech"]["status"] == "unmeasured"
    assert "probe exploded" in keys["speech"]["reason"]
    assert errors and "speech probe failed" in errors[0]


def test_fs_missing_marks_every_key_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "foundationscale", None)
    caps = cap.probe()
    assert caps.available is False and tuple(caps.probe_keys) == cap.PROBE_KEYS
    assert all(k["status"] == "unmeasured" for k in caps.probe_keys.values())
    assert "probe_keys" in caps.to_dict()
