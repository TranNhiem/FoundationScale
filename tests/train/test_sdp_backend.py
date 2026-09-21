"""Tests for train/sdp_backend.py -- torch-free by construction (#526/#527).

The production module never imports torch; it receives the
torch.backends.cuda object as a plain argument, so these tests hand it a
recording fake and inspect the CALL RECORD. No torch, no CUDA, and no
skip/xfail: if a gate cannot be made to inspect something in this
environment, the gate -- not the test -- is defective.

Coverage: the pin enables exactly one backend and disables the other
three; the unpinned path announces instead of defaulting silent; the
manifest pair is "measured" when pinned and "unmeasured"-with-a-reason
(not None) when unpinned; a missing toggle yields a refusal reason that
names backend, toggle, and torch version.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from foundationscale.train.sdp_backend import (
    SDP_BACKEND_TOGGLES,
    apply_sdp_pin,
    sdp_backend_telemetry_pair,
    sdp_pin_refusal_reason,
    sdp_pinned_announcement,
    sdp_unpinned_announcement,
)


class FakeCudaBackends:
    """Stand-in for torch.backends.cuda recording every toggle invocation.

    WHY a hand-rolled recording fake rather than a mock framework: the
    claim under test -- "exactly one backend enabled, the other three
    disabled" -- is a property of the call record, and the record must be
    inspectable afterwards without coupling to a mocking library's call
    algebra. `drop` simulates a torch build that predates a toggle, which
    is the REFUSE-96 precondition.
    """

    def __init__(self, drop: str | None = None) -> None:
        self.calls: list[tuple[str, bool]] = []
        for name, toggle in SDP_BACKEND_TOGGLES.items():
            if name == drop:
                continue  # this "torch build" simply lacks the toggle
            setattr(self, toggle, self._recorder(toggle))

    def _recorder(self, toggle: str) -> Callable[[bool], None]:
        def record(enabled: bool) -> None:
            self.calls.append((toggle, enabled))

        return record


def test_pin_enables_declared_backend_and_disables_the_other_three() -> None:
    for backend in SDP_BACKEND_TOGGLES:
        fake = FakeCudaBackends()
        apply_sdp_pin(backend, fake)
        # Every toggle was touched -- a mask half-set is not a pin.
        assert len(fake.calls) == len(SDP_BACKEND_TOGGLES) == 4
        assert dict(fake.calls) == {
            toggle: name == backend for name, toggle in SDP_BACKEND_TOGGLES.items()
        }


def test_present_toggle_yields_no_refusal() -> None:
    fake = FakeCudaBackends()
    for backend in SDP_BACKEND_TOGGLES:
        assert sdp_pin_refusal_reason(backend, fake, "2.9.1") is None


def test_refusal_reason_names_backend_toggle_and_torch_version() -> None:
    fake = FakeCudaBackends(drop="flash")
    reason = sdp_pin_refusal_reason("flash", fake, "2.9.1+cu130")
    assert reason is not None
    assert "'flash'" in reason
    assert "enable_flash_sdp" in reason
    assert "2.9.1+cu130" in reason
    assert "96" in reason  # the refusal names its exit, not just its cause


def test_missing_sibling_toggle_fails_loudly_not_silently() -> None:
    # The declared backend's toggle exists but a sibling's does not: the
    # disable pass must be LOUD (AttributeError). A mask the process only
    # half-set must never quietly pass for a pin.
    fake = FakeCudaBackends(drop="math")
    with pytest.raises(AttributeError):
        apply_sdp_pin("flash", fake)


def test_unpinned_path_announces_instead_of_defaulting_silent() -> None:
    msg = sdp_unpinned_announcement()
    assert msg.strip()  # the announcement EXISTS; silence is the defect
    assert "UNPINNED" in msg
    assert "per shape" in msg
    assert "NOT" in msg and "reproducible" in msg
    assert "94%" in msg  # the measurement rides along, or it is boilerplate
    # An unpinned run is legitimate -- the announcement must not read as
    # a refusal.
    assert "proceeds" in msg


def test_pinned_announcement_names_backend_and_disabled_others() -> None:
    msg = sdp_pinned_announcement("math")
    assert "enable_math_sdp" in msg
    for other in ("flash", "mem_efficient", "cudnn"):
        assert other in msg  # the disabled three are part of the mask claim


def test_manifest_entry_is_measured_when_pinned() -> None:
    assert sdp_backend_telemetry_pair("cudnn") == ("cudnn", "measured")
    assert sdp_backend_telemetry_pair("math") == ("math", "measured")


def test_manifest_entry_is_unmeasured_with_a_reason_when_unpinned() -> None:
    value, source = sdp_backend_telemetry_pair(None)
    assert source == "unmeasured"
    # The house contract: for source == "unmeasured", `value` carries the
    # REASON STRING -- never None, never an opaque sentinel.
    assert isinstance(value, str) and value.strip()
    assert "torch selects" in value
    assert "per shape" in value
    assert "not" in value and "recorded" in value
    assert "#526/#527" in value


def test_no_cli_help_string_leaks_argparse_params() -> None:
    """If deleted: an unescaped ``%`` in any --flag help silently corrupts
    ``--help``.

    Measured while wiring --sdp-backend: the help quoted the finding's "up to
    94%", argparse %-formatted it against its own params dict, and the rendered
    help read ``up to 94{'option_strings': ['--sdp-backend'], 'dest': ...}``.
    Nothing failed -- not a test, not a lint, not the parse -- because argparse
    formats help lazily and only when it renders. This asserts over the WHOLE
    parser rather than one flag, since the trap belongs to every help string
    that ever quotes a percentage.
    """
    from foundationscale.train.cli import build_parser

    rendered = build_parser().format_help()

    assert "option_strings" not in rendered, (
        "argparse substituted its params dict into a help string -- some help= "
        "text contains an unescaped '%'; double it to '%%'"
    )
    assert "'dest':" not in rendered
    # Positive control: the parser really did render help text, so the two
    # assertions above cannot pass by inspecting an empty string.
    assert "--sdp-backend" in rendered
