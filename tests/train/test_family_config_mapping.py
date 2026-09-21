"""Contract tests for ``_family_config_mapping`` (foundationscale.train.loop).

The helper exists so that config introspection for the manifest NEVER raises:
whatever shape ``config`` arrives in -- absent, exploding, already a mapping,
or something else entirely -- the answer is a Mapping, with every failure
degrading to ``{}`` ("unknown"), never an abort. Each test below pins one
branch of loop.py 291-302 and asserts the RETURNED VALUE, because a stub that
always returns ``{}`` must fail the mapping-identity case and a stub that
echoes its input must fail the exploding-to_dict case.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from foundationscale.train.loop import _family_config_mapping


def test_absent_config_attribute_degrades_to_empty_mapping() -> None:
    """Pins the ``config is None`` guard: when the caller's object carries no
    ``config`` attribute the helper receives None and must answer ``{}``, so
    manifest assembly treats the config as UNKNOWN rather than crashing.

    If deleted: a None config dereference (``getattr(None, "to_dict")`` is
    fine, but any branch assuming attributes) can raise into the manifest
    writer, turning an unknown config into a failed run.
    """
    result = _family_config_mapping(None)
    assert result == {}
    assert isinstance(result, Mapping)


def test_exploding_to_dict_degrades_to_empty_mapping() -> None:
    """Pins loop.py lines 297-298: ANY exception from ``config.to_dict()``
    means "unknown", returned as ``{}`` -- config failure is not an abort.

    If deleted: a config class whose serialisation backend is broken
    propagates its RuntimeError into loop bookkeeping, and the whole
    NEVER-raises contract (the reason this helper exists) is gone.
    """

    class _ExplodingConfig:
        def to_dict(self) -> dict[str, str]:
            raise RuntimeError("config serialisation backend exploded")

    result = _family_config_mapping(_ExplodingConfig())
    assert result == {}
    assert isinstance(result, Mapping)


def test_plain_mapping_config_is_returned_unchanged() -> None:
    """Pins loop.py line 302: a config that is ALREADY a Mapping (and carries
    no ``to_dict``) must be returned as that same object -- identity, not a
    rebuilt copy and certainly not ``{}``.

    If deleted: a ``return {}`` stub passes the two degradation tests above,
    and the caller's actual config values (model_type, sizes) vanish from the
    manifest with no error anywhere.
    """
    # The argument is the MODEL, not the config: the function reaches through
    # `.config` itself, so handing it a bare mapping tests the no-`config`
    # degradation path instead of the one this test names.
    source: dict[str, Any] = {"model_type": "gemma4", "hidden_size": 42}
    result = _family_config_mapping(SimpleNamespace(config=source))
    assert result is source
    assert result["model_type"] == "gemma4"
    assert result["hidden_size"] == 42


def test_unrecognised_config_object_degrades_to_empty_mapping() -> None:
    """Pins the terminal ``return {}``: an object that is neither a Mapping
    nor carries a callable ``to_dict`` has no honest mapping, so the answer
    is ``{}`` -- and the helper still must not raise.

    If deleted: the fallthrough branch raises (AttributeError/TypeError) on a
    stray config type, and manifest writing inherits a crash class the exit
    contract has no code for.
    """

    class _Opaque:
        __slots__ = ()

    result = _family_config_mapping(_Opaque())
    assert result == {}
    assert isinstance(result, Mapping)
