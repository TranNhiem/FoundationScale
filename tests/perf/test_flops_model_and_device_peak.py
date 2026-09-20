"""Pinned validation arms for ``FlopsModel`` and ``DevicePeak``.

The lines under test are the refusal gates of the two declarations that anchor
every FLOP/s and MFU figure the perf plane publishes: ``FlopsModel`` fixes the
formula, ``DevicePeak`` fixes the denominator, and both refuse -- by returning
None or by raising a ValueError that names the offending field -- rather than
mint a number from a missing input. Each positive test is paired with the
negative control that isolates the one input whose absence flips the outcome,
so the pivot of every gate is visible in the test body, not inferred.

No torch, by construction and by intent: ``from_model`` identifies embeddings
by class NAME so that the module under test stays torch-free, and the model
doubles below honour that contract literally -- plain classes exposing exactly
the three attributes the walk reads (``config``, ``named_parameters``,
``named_modules``), including one class actually named ``Embedding`` built with
``type()`` where the name itself is the thing under test. Environment access is
monkeypatched per test so the ambient shell cannot smuggle in a declaration.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from foundationscale.perf.telemetry import DevicePeak, FlopsModel


class _FakeParameter:
    """A parameter double exposing only what the walk reads: ``numel()``."""

    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class Embedding:
    """Module double whose class NAME is the recognition signal the walk uses.

    ``parameters`` honours ``recurse`` the way nn.Module does -- own tensors
    when False, own plus descendants when True -- so the suite can prove the
    walk asks for own parameters only.
    """

    def __init__(
        self,
        own: list[_FakeParameter],
        descendants: list[_FakeParameter] | None = None,
    ) -> None:
        self._own = list(own)
        self._descendants = list(descendants or [])

    def parameters(self, recurse: bool = True):
        owned = list(self._own)
        if recurse:
            owned.extend(self._descendants)
        return iter(owned)


class VocabEmbedding:
    """An embedding in every respect except its NAME -- the walk must miss it."""

    def __init__(self, own: list[_FakeParameter]) -> None:
        self._own = list(own)

    def parameters(self, recurse: bool = True):
        return iter(self._own)


class _DenseModule:
    """A module the walk skips by name; it needs no parameters accessor at all."""

    def __init__(self, params: list[_FakeParameter] | None = None) -> None:
        self._params = list(params or [])


# An ``Embedding`` by name and nothing else: no ``parameters`` accessor, so the
# walk's ``getattr(module, "parameters", None)`` lands on its None branch.
BARE_EMBEDDING = type("Embedding", (), {})


class _FakeModel:
    """A model double carrying exactly the three attributes ``from_model`` reads."""

    def __init__(
        self,
        config: object,
        named: tuple[tuple[str, object], ...] = (),
        modules: tuple[tuple[str, object], ...] = (),
    ) -> None:
        self.config = config
        self._named = list(named)
        self._modules = list(modules)

    def named_parameters(self):
        return iter(list(self._named))

    def named_modules(self):
        return iter(list(self._modules))


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Delete all three FS_DEVICE_PEAK_* variables for the duration of a test.

    from_env reads the process environment directly; without this fixture an
    operator-exported FS_DEVICE_PEAK_TFLOPS on the test host would turn every
    "undeclared" assertion into a measurement of the shell, not of the code.
    """
    for name in ("FS_DEVICE_PEAK_TFLOPS", "FS_DEVICE_PEAK_SOURCE", "FS_DEVICE_PEAK_NAME"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# FlopsModel: the flops_per_token arithmetic
# ---------------------------------------------------------------------------


def test_flops_per_token_matches_the_stated_formula_by_hand() -> None:
    """The per-token count is 6N + 12*L*h*s, checked against hand arithmetic.

    With N=1000, L=2, h=16, s=8: 6*1000 = 6000 and 12*2*16*8 = 3072, so the
    expected total is 9072 -- stated as a literal so a re-implemented formula
    cannot agree with itself.
    """
    model = FlopsModel(
        parameters=999,
        non_embedding_parameters=1000,
        layers=2,
        hidden_size=16,
        sequence_length=8,
    )
    assert model.flops_per_token == 9072


def test_flops_per_token_ignores_the_total_parameter_count() -> None:
    """The total parameter count never enters the formula.

    Two declarations differing ONLY in ``parameters`` must produce the same
    per-token figure; if the total leaked into the arithmetic, the second
    number would move.
    """
    base = dict(non_embedding_parameters=1000, layers=2, hidden_size=16, sequence_length=8)
    small_total = FlopsModel(parameters=999, **base)
    huge_total = FlopsModel(parameters=10_000_000, **base)
    assert small_total.flops_per_token == huge_total.flops_per_token == 9072


def test_flops_per_token_carries_the_attention_sequence_length_term() -> None:
    """The 12*L*h*s term is present: doubling the sequence length adds 12*L*h*Δs.

    From s=8 to s=16 with L=2, h=16 the increment is 12*2*16*8 = 3072, taking
    the total from 9072 to 12144. A formula that dropped the attention term
    would return 6000 at every sequence length.
    """
    short = FlopsModel(
        parameters=999,
        non_embedding_parameters=1000,
        layers=2,
        hidden_size=16,
        sequence_length=8,
    )
    long = FlopsModel(
        parameters=999,
        non_embedding_parameters=1000,
        layers=2,
        hidden_size=16,
        sequence_length=16,
    )
    assert short.flops_per_token == 9072
    assert long.flops_per_token == 12144


def test_a_declared_flops_model_is_immutable() -> None:
    """A stated formula cannot be edited after the fact."""
    model = FlopsModel(
        parameters=999,
        non_embedding_parameters=1000,
        layers=2,
        hidden_size=16,
        sequence_length=8,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.layers = 7


# ---------------------------------------------------------------------------
# FlopsModel.from_hf_config: the field gates
# ---------------------------------------------------------------------------


def test_from_hf_config_builds_from_a_flat_config_and_passes_counts_through() -> None:
    """A complete flat config yields a declaration with every field verbatim."""
    config = SimpleNamespace(num_hidden_layers=2, hidden_size=16)
    result = FlopsModel.from_hf_config(
        config, sequence_length=8, parameters=1500, non_embedding_parameters=1000
    )
    assert result is not None
    assert result.layers == 2
    assert result.hidden_size == 16
    assert result.parameters == 1500
    assert result.non_embedding_parameters == 1000
    assert result.sequence_length == 8


def test_from_hf_config_returns_none_when_num_hidden_layers_is_absent() -> None:
    """A config without a layer count is refused, and the layer count is the pivot.

    The two configs differ ONLY in ``num_hidden_layers``: the complete one is
    accepted, the one missing that single field is refused -- naming, by
    construction, the field responsible for the None.
    """
    complete = SimpleNamespace(num_hidden_layers=2, hidden_size=16)
    missing_layers = SimpleNamespace(hidden_size=16)
    counts = dict(sequence_length=8, parameters=1500, non_embedding_parameters=1000)
    assert FlopsModel.from_hf_config(complete, **counts) is not None
    assert FlopsModel.from_hf_config(missing_layers, **counts) is None


def test_from_hf_config_returns_none_when_hidden_size_is_absent() -> None:
    """A config without a hidden size is refused, and the hidden size is the pivot.

    The mirror of the layer-count control: identical configs except for
    ``hidden_size``, so its absence alone produces the None.
    """
    complete = SimpleNamespace(num_hidden_layers=2, hidden_size=16)
    missing_hidden = SimpleNamespace(num_hidden_layers=2)
    counts = dict(sequence_length=8, parameters=1500, non_embedding_parameters=1000)
    assert FlopsModel.from_hf_config(complete, **counts) is not None
    assert FlopsModel.from_hf_config(missing_hidden, **counts) is None


def test_from_hf_config_refuses_boolean_fields() -> None:
    """A boolean is a flag, not a count -- True is refused for either field."""
    counts = dict(sequence_length=8, parameters=1500, non_embedding_parameters=1000)
    bool_layers = SimpleNamespace(num_hidden_layers=True, hidden_size=16)
    bool_hidden = SimpleNamespace(num_hidden_layers=2, hidden_size=True)
    assert FlopsModel.from_hf_config(bool_layers, **counts) is None
    assert FlopsModel.from_hf_config(bool_hidden, **counts) is None


def test_from_hf_config_refuses_non_integer_fields() -> None:
    """A float layer count or a string hidden size is refused, not coerced."""
    counts = dict(sequence_length=8, parameters=1500, non_embedding_parameters=1000)
    float_layers = SimpleNamespace(num_hidden_layers=2.0, hidden_size=16)
    str_hidden = SimpleNamespace(num_hidden_layers=2, hidden_size="512")
    assert FlopsModel.from_hf_config(float_layers, **counts) is None
    assert FlopsModel.from_hf_config(str_hidden, **counts) is None


def test_from_hf_config_refuses_nonpositive_fields() -> None:
    """A zero or negative count is refused for either field."""
    counts = dict(sequence_length=8, parameters=1500, non_embedding_parameters=1000)
    for layers, hidden in ((0, 16), (-3, 16), (2, 0), (2, -64)):
        config = SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden)
        assert FlopsModel.from_hf_config(config, **counts) is None


def test_from_hf_config_reads_a_composite_config_from_its_text_config() -> None:
    """A VLM's fields come from text_config FIRST, and the flat values do not leak.

    The flat attributes carry plausibly-wrong VLM top-level values (80 layers,
    4096 hidden); the accepted declaration must read 4 and 64 from the text
    sub-config, proving precedence rather than merely compatibility.
    """
    config = SimpleNamespace(
        num_hidden_layers=80,
        hidden_size=4096,
        text_config=SimpleNamespace(num_hidden_layers=4, hidden_size=64),
    )
    result = FlopsModel.from_hf_config(
        config, sequence_length=8, parameters=1500, non_embedding_parameters=1000
    )
    assert result is not None
    assert result.layers == 4
    assert result.hidden_size == 64


def test_from_hf_config_does_not_fall_back_from_text_config_to_flat_fields() -> None:
    """An incomplete text_config yields None even when the flat config is complete.

    The deliberate no-fallback rule: on a VLM the flat attributes belong to a
    different sub-model, so a wrong value is worse than None.
    """
    config = SimpleNamespace(
        num_hidden_layers=2,  # flat is complete...
        hidden_size=16,
        text_config=SimpleNamespace(hidden_size=64),  # ...but text_config lacks layers
    )
    result = FlopsModel.from_hf_config(
        config, sequence_length=8, parameters=1500, non_embedding_parameters=1000
    )
    assert result is None


def test_from_hf_config_ignores_a_none_text_config() -> None:
    """A text_config attribute that is None means "no composite": read the flat fields."""
    config = SimpleNamespace(num_hidden_layers=3, hidden_size=32, text_config=None)
    result = FlopsModel.from_hf_config(
        config, sequence_length=8, parameters=1500, non_embedding_parameters=1000
    )
    assert result is not None
    assert result.layers == 3
    assert result.hidden_size == 32


# ---------------------------------------------------------------------------
# FlopsModel.from_model: the parameter walk
# ---------------------------------------------------------------------------


def test_from_model_counts_parameters_and_subtracts_embedding_parameters() -> None:
    """Total is the sum over named tensors; non-embedding subtracts what an Embedding owns.

    Hand count: 80 + 120 + 10 = 210 total; the Embedding owns the 80, leaving
    120 + 10 = 130 as non-embedding.
    """
    embed_w = _FakeParameter(80)
    linear_w = _FakeParameter(120)
    bias = _FakeParameter(10)
    model = _FakeModel(
        SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named=(
            ("embed.weight", embed_w),
            ("dense.weight", linear_w),
            ("dense.bias", bias),
        ),
        modules=(
            ("embed", Embedding([embed_w])),
            ("dense", _DenseModule([linear_w, bias])),
        ),
    )
    result = FlopsModel.from_model(model, sequence_length=4)
    assert result is not None
    assert result.parameters == 210
    assert result.non_embedding_parameters == 130
    assert result.layers == 2
    assert result.hidden_size == 8
    assert result.sequence_length == 4


def test_from_model_counts_a_tied_embedding_table_once() -> None:
    """A tied embedding/head parameter is counted ONCE, not once per name.

    The walk dedupes parameter objects by identity: the shared table appears
    under both ``embed.weight`` and ``lm_head.weight``, but 80 must enter the
    total once -- 200, not 280 -- and be subtracted once, leaving 120, not 40.
    """
    shared_table = _FakeParameter(80)
    linear_w = _FakeParameter(120)
    model = _FakeModel(
        SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named=(
            ("embed.weight", shared_table),
            ("lm_head.weight", shared_table),
            ("dense.weight", linear_w),
        ),
        modules=(
            ("embed", Embedding([shared_table])),
            ("lm_head", _DenseModule([shared_table])),
        ),
    )
    result = FlopsModel.from_model(model, sequence_length=4)
    assert result is not None
    assert result.parameters == 200
    assert result.non_embedding_parameters == 120


def test_from_model_subtracts_only_the_embedding_module_own_parameters() -> None:
    """The embedding walk passes recurse=False: a descendant tensor stays non-embedding.

    The double's ``parameters(recurse=True)`` would also yield the child
    tensor; the walk must use own-only access. Hand count: total 80+50+120 =
    250, own embedding table 80 subtracted, leaving 50+120 = 170. A
    recurse=True walk would subtract 130 and report 120.
    """
    embed_w = _FakeParameter(80)
    child_w = _FakeParameter(50)
    linear_w = _FakeParameter(120)
    model = _FakeModel(
        SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named=(
            ("embed.weight", embed_w),
            ("embed.child.weight", child_w),
            ("dense.weight", linear_w),
        ),
        modules=(
            ("embed", Embedding([embed_w], descendants=[child_w])),
            ("dense", _DenseModule([linear_w])),
        ),
    )
    result = FlopsModel.from_model(model, sequence_length=4)
    assert result is not None
    assert result.parameters == 250
    assert result.non_embedding_parameters == 170


def test_from_model_treats_a_differently_named_embedding_as_non_embedding() -> None:
    """A VocabEmbedding is counted as non-embedding: overstatement, by design.

    Recognition is by class NAME, so only a class named exactly ``Embedding``
    is subtracted. Hand count: 80 + 120 = 200 total, and the misnamed table
    stays in the non-embedding side, so non-embedding is also 200.
    """
    table_w = _FakeParameter(80)
    linear_w = _FakeParameter(120)
    model = _FakeModel(
        SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named=(("table.weight", table_w), ("dense.weight", linear_w)),
        modules=(("table", VocabEmbedding([table_w])), ("dense", _DenseModule([linear_w]))),
    )
    result = FlopsModel.from_model(model, sequence_length=4)
    assert result is not None
    assert result.parameters == 200
    assert result.non_embedding_parameters == 200


def test_from_model_subtracts_nothing_when_the_embedding_module_exposes_no_accessor() -> None:
    """The bare class name is not enough: an Embedding without ``parameters`` subtracts nothing.

    BARE_EMBEDDING is named ``Embedding`` but defines no ``parameters``
    accessor, so the walk's ``getattr(..., None)`` branch skips it. The table
    remains in the non-embedding count: 80 + 120 = 200 on both sides.
    """
    embed_w = _FakeParameter(80)
    linear_w = _FakeParameter(120)
    model = _FakeModel(
        SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named=(("embed.weight", embed_w), ("dense.weight", linear_w)),
        modules=(("embed", BARE_EMBEDDING()),),
    )
    result = FlopsModel.from_model(model, sequence_length=4)
    assert result is not None
    assert result.parameters == 200
    assert result.non_embedding_parameters == 200


def test_from_model_returns_none_without_a_config() -> None:
    """A model with no config offers nothing to derive the formula from.

    Both absence shapes are covered: no ``config`` attribute at all, and a
    ``config`` attribute explicitly set to None.
    """
    configless = SimpleNamespace(
        named_parameters=lambda: iter([("w", _FakeParameter(5))]),
        named_modules=lambda: iter([]),
    )
    assert FlopsModel.from_model(configless, sequence_length=4) is None
    null_config = _FakeModel(None, named=(("w", _FakeParameter(5)),))
    assert FlopsModel.from_model(null_config, sequence_length=4) is None


def test_from_model_returns_none_without_named_parameters() -> None:
    """A model that cannot enumerate its parameters cannot have them counted."""
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named_modules=lambda: iter([]),
    )
    assert FlopsModel.from_model(model, sequence_length=4) is None


def test_from_model_returns_none_without_named_modules() -> None:
    """A model that cannot enumerate its modules cannot classify its parameters."""
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named_parameters=lambda: iter([("w", _FakeParameter(5))]),
    )
    assert FlopsModel.from_model(model, sequence_length=4) is None


def test_from_model_returns_none_when_a_parameter_has_no_numel() -> None:
    """A single parameter without ``numel`` refuses the whole declaration.

    The control walks the identical shape with a numel-bearing tensor and is
    accepted, so ``numel`` alone is the pivot between None and a number.
    """
    config = SimpleNamespace(num_hidden_layers=2, hidden_size=8)
    broken = _FakeModel(config, named=(("w", object()),))
    working = _FakeModel(config, named=(("w", _FakeParameter(5)),))
    assert FlopsModel.from_model(broken, sequence_length=4) is None
    assert FlopsModel.from_model(working, sequence_length=4) is not None


def test_from_model_returns_none_with_zero_total_parameters() -> None:
    """An empty parameter list is a zero, and a zero is refused, not published."""
    model = _FakeModel(SimpleNamespace(num_hidden_layers=2, hidden_size=8))
    assert FlopsModel.from_model(model, sequence_length=4) is None


def test_from_model_returns_none_with_zero_non_embedding_parameters() -> None:
    """A model whose every tensor is an embedding has nothing the formula can bill.

    Total is 80 (accepted: nonzero), but subtracting the Embedding-owned table
    leaves 0, and that zero is refused.
    """
    embed_w = _FakeParameter(80)
    model = _FakeModel(
        SimpleNamespace(num_hidden_layers=2, hidden_size=8),
        named=(("embed.weight", embed_w),),
        modules=(("embed", Embedding([embed_w])),),
    )
    assert FlopsModel.from_model(model, sequence_length=4) is None


def test_from_model_returns_none_when_valid_counts_cannot_rescue_a_missing_field() -> None:
    """Good parameter counts never substitute for a missing layer count.

    The config here lacks ``num_hidden_layers``; the walk's totals are sound,
    but delegation to from_hf_config still returns None -- it never guesses.
    """
    model = _FakeModel(
        SimpleNamespace(hidden_size=8),
        named=(("dense.weight", _FakeParameter(120)),),
    )
    assert FlopsModel.from_model(model, sequence_length=4) is None


# ---------------------------------------------------------------------------
# DevicePeak: the declaration gate
# ---------------------------------------------------------------------------


def test_device_peak_carries_a_sourced_declaration_verbatim() -> None:
    """A positive, sourced declaration round-trips with every field intact."""
    peak = DevicePeak(
        name="h100-sxm5",
        bf16_dense_tflops=989.5,
        source="vendor datasheet, dense bf16, no sparsity",
    )
    assert peak.name == "h100-sxm5"
    assert peak.bf16_dense_tflops == 989.5
    assert peak.source == "vendor datasheet, dense bf16, no sparsity"


def test_a_declared_device_peak_is_immutable() -> None:
    """A stated peak cannot be edited after the fact."""
    peak = DevicePeak(name="h100", bf16_dense_tflops=989.5, source="datasheet")
    with pytest.raises(dataclasses.FrozenInstanceError):
        peak.bf16_dense_tflops = 1.0


def test_device_peak_refuses_an_empty_or_blank_source_by_name() -> None:
    """A peak that cannot name its origin is refused, and the refusal names the field."""
    for source in ("", "   "):
        with pytest.raises(ValueError) as excinfo:
            DevicePeak(name="h100", bf16_dense_tflops=989.5, source=source)
        assert "DevicePeak.source" in str(excinfo.value)


def test_device_peak_refuses_a_nonpositive_peak_by_name() -> None:
    """Zero and negative peaks are refused, and the refusal names the field and value."""
    for value, shown in ((0.0, "0.0"), (-3.5, "-3.5")):
        with pytest.raises(ValueError) as excinfo:
            DevicePeak(name="h100", bf16_dense_tflops=value, source="datasheet")
        message = str(excinfo.value)
        assert "DevicePeak.bf16_dense_tflops" in message
        assert shown in message


def test_device_peak_refuses_a_non_finite_peak_by_name() -> None:
    """Infinity and NaN are refused -- a peak must be finite to divide by."""
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError) as excinfo:
            DevicePeak(name="h100", bf16_dense_tflops=value, source="datasheet")
        assert "DevicePeak.bf16_dense_tflops" in str(excinfo.value)


# ---------------------------------------------------------------------------
# DevicePeak.from_env: declared, undeclared and malformed shapes
# ---------------------------------------------------------------------------


def test_from_env_returns_none_when_nothing_is_declared(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """An absent declaration means undeclared -- None, so MFU stays unmeasured."""
    assert DevicePeak.from_env() is None


def test_from_env_returns_none_when_only_the_value_is_declared(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A value without a source is treated as undeclared, not as malformed.

    Half a declaration carries no provenance to audit; the module demotes it
    to the unmeasured path rather than raising, because absence -- even
    partial absence -- means not declared.
    """
    clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", "100")
    assert DevicePeak.from_env() is None


def test_from_env_returns_none_when_only_the_source_is_declared(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A source without a value is likewise undeclared, not an error."""
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "operator GEMM sweep")
    assert DevicePeak.from_env() is None


def test_from_env_refuses_a_non_numeric_value_by_name(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A declared-but-unparseable peak raises, naming the env variable and the raw text.

    This is different from absence: the operator DID declare, and the
    declaration is unusable, so it is refused rather than silently demoted.
    """
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "operator GEMM sweep")
    for raw in ("loud", ""):
        clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", raw)
        with pytest.raises(ValueError) as excinfo:
            DevicePeak.from_env()
        message = str(excinfo.value)
        assert "FS_DEVICE_PEAK_TFLOPS" in message
        assert repr(raw) in message


def test_from_env_refuses_a_zero_value_by_name(clean_env: pytest.MonkeyPatch) -> None:
    """A declared zero reaches the dataclass gate and is refused by field name."""
    clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", "0")
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "operator GEMM sweep")
    with pytest.raises(ValueError) as excinfo:
        DevicePeak.from_env()
    message = str(excinfo.value)
    assert "DevicePeak.bf16_dense_tflops" in message
    assert "0.0" in message


def test_from_env_refuses_a_negative_value_by_name(clean_env: pytest.MonkeyPatch) -> None:
    """A declared negative reaches the dataclass gate and is refused by field name."""
    clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", "-12.5")
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "operator GEMM sweep")
    with pytest.raises(ValueError) as excinfo:
        DevicePeak.from_env()
    message = str(excinfo.value)
    assert "DevicePeak.bf16_dense_tflops" in message
    assert "-12.5" in message


def test_from_env_refuses_an_empty_source_by_name(clean_env: pytest.MonkeyPatch) -> None:
    """A declared-but-empty source raises rather than being demoted to unmeasured.

    An empty string is not absence: the variable WAS set, the declaration is
    unusable, and the refusal names the field the operator must correct.
    """
    clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", "700.5")
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "")
    with pytest.raises(ValueError) as excinfo:
        DevicePeak.from_env()
    assert "DevicePeak.source" in str(excinfo.value)


def test_from_env_round_trips_a_declared_peak_with_its_source_string_intact(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """Value, source and optional name arrive verbatim -- the audit trail is preserved."""
    clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", "700.5")
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "Operator GEMM sweep, 2024-11 runbook section 3")
    clean_env.setenv("FS_DEVICE_PEAK_NAME", "a100-sxm4-80gb")
    peak = DevicePeak.from_env()
    assert peak is not None
    assert peak.bf16_dense_tflops == 700.5
    assert peak.source == "Operator GEMM sweep, 2024-11 runbook section 3"
    assert peak.name == "a100-sxm4-80gb"


def test_from_env_supplies_a_default_name_when_none_is_declared(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """A nameless declaration is named after the mechanism that declared it."""
    clean_env.setenv("FS_DEVICE_PEAK_TFLOPS", "312")
    clean_env.setenv("FS_DEVICE_PEAK_SOURCE", "vendor datasheet")
    peak = DevicePeak.from_env()
    assert peak is not None
    assert peak.name == "device declared through FS_DEVICE_PEAK_TFLOPS"
    assert peak.bf16_dense_tflops == 312.0
