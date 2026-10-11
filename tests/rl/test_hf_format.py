"""``can_save_original_format``: a dequantizing load conversion cannot be reversed on save."""

from __future__ import annotations

from types import SimpleNamespace

from foundationscale.rl.hf_format import can_save_original_format, save_format_kwargs


class Mxfp4Dequantize:
    """Name-alike of transformers' MXFP4 dequantize operation."""


class Concatenate:
    """A reversible operation (stacking experts) -- original format stays writable."""


def _model(*operations: object) -> SimpleNamespace:
    return SimpleNamespace(_weight_conversions=[SimpleNamespace(operations=list(operations))])


def test_dequantized_load_cannot_be_saved_in_original_format() -> None:
    assert can_save_original_format(_model(Concatenate(), Mxfp4Dequantize())) is False


def test_reversible_conversions_keep_the_original_format() -> None:
    assert can_save_original_format(_model(Concatenate())) is True


def test_models_without_recorded_conversions_keep_the_original_format() -> None:
    assert can_save_original_format(SimpleNamespace()) is True
    assert can_save_original_format(SimpleNamespace(_weight_conversions=None)) is True
    assert (
        can_save_original_format(SimpleNamespace(_weight_conversions=[SimpleNamespace()])) is True
    )


def test_save_kwargs_add_the_flag_only_when_required() -> None:
    assert save_format_kwargs(_model(Mxfp4Dequantize())) == {"save_original_format": False}
    assert save_format_kwargs(_model(Concatenate())) == {}
