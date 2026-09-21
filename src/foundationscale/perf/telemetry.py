"""Steady-state performance telemetry: tokens, FLOPs and MFU, each with provenance.

The training loop records that a model trained -- runtime, samples/s, steps/s,
loss, peak memory -- and counts no tokens and no FLOPs. ``total_flos`` is
deliberately absent from its units table because no producer existed, and a
units table wider than the emitted key set is exactly the drift the telemetry
unit control exists to catch. This module is the producer that was missing. It
answers the question the loop cannot: not WHETHER the model trains but how
EFFICIENTLY it trains.

THE PROVENANCE CONTRACT -- the load-bearing rule of this module
---------------------------------------------------------------
Every metric is a 2-tuple ``(value, provenance)`` where provenance is one of
the literal strings ``"measured"``, ``"derived"`` or ``"unmeasured"``:

* ``"measured"``   -- read from a real counter this run produced.
* ``"derived"``    -- computed from measured values by a stated formula. The
  formula is stated in the summary() docstring next to the key, because a
  derived number whose formula is folklore is a measured number with the
  audit trail removed.
* ``"unmeasured"`` -- the value is None and the SECOND element is instead a
  reason string naming the specific missing input and what the operator must
  declare to obtain the metric.

An unmeasured metric is NOT an error and NOT a zero. Nothing here invents a
number. In particular MFU is unmeasured unless the device peak throughput was
DECLARED: there is no default peak, because guessing one would turn a hardware
assumption into a published efficiency claim -- the exact failure this
contract exists to stop.

WHAT IS CLAIMED
---------------
* Step, between-step and optimizer-bracket durations, timed with
  ``time.perf_counter()`` and split into a warmup bucket and a steady-state
  bucket, because a benchmark that averages in the first step measures the
  CUDA allocator and the autotuner, not the model.
* Token throughput, when the trainer counted tokens.
* Model FLOP/s and MFU, when -- and only when -- a FlopsModel and a DevicePeak
  were respectively declared, each carrying the provenance of its inputs.

WHAT IS NOT CLAIMED
-------------------
* That the between-step gap PROVES dataloader starvation. It is the gap
  between steps; dataloader fetch, collation, H2D copy and callback overhead
  all live inside it, and this module names it ``between_steps`` rather than
  ``dataloader_stall`` for that reason.
* That the optimizer bracket is communication time. Under DDP the gradient
  all-reduce overlaps backward; the bracket captures only the synchronisation
  TAIL.
* That the FLOP count is exact. It is the PaLM / Megatron-LM estimate, chosen
  so the numbers are comparable to published MFU -- the only reason to prefer
  a known-imperfect formula over a bespoke one.
* Any hardware fact the operator did not declare. There is no built-in device
  table; see DevicePeak for why a table would be the same guess with extra
  steps.

This module imports without transformers installed (the gates import the
package at collection time) and never touches torch at module scope, so it is
import-safe and CPU-safe by construction.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Pin the callback base to ``object`` so the typecheck does not depend on
    # whether the optional [train] extra is installed. loop.py documents the
    # failure this avoids: with the base resolved from the real import, the
    # identical source is clean in one environment and one error in another,
    # and a silencer whose own necessity depends on the environment is the
    # same defect written in one comment. Runtime is untouched -- with
    # transformers present the class below subclasses the real
    # TrainerCallback; without it, object, and the hooks are simply never
    # called by anyone.
    _CallbackBase = object
else:
    try:
        from transformers import TrainerCallback as _CallbackBase
    except ImportError:
        _CallbackBase = object

__all__ = (
    "PERF_TELEMETRY_UNITS",
    "DevicePeak",
    "FlopsModel",
    "StepTelemetry",
)


_NO_PEAK_REASON = (
    "UNMEASURED: no device peak throughput was declared -- pass "
    "device_peak=DevicePeak(...) or set FS_DEVICE_PEAK_TFLOPS and "
    "FS_DEVICE_PEAK_SOURCE. There is NO default peak: vendor peaks differ by "
    "SKU, clock and whether sparsity is counted, and guessing one would turn "
    "a hardware assumption into a published MFU claim, which is the exact "
    "failure the provenance contract exists to stop"
)

_TOKENS_UNMEASURED_REASON = (
    "UNMEASURED: state.num_input_tokens_seen was absent, None, or still zero "
    "at step end -- TrainingArguments(include_num_input_tokens_seen=True) "
    "was not set, so the trainer counted no tokens this run. The zero case "
    "is the unread-counter signature: transformers defaults the counter to 0 "
    "and only feeds it under that flag, and a completed step cannot have "
    "consumed zero tokens, so recording the zero as 'measured' would mint a "
    "number the trainer never produced"
)


def _is_routed_expert_parameter(qualified_name: str) -> bool:
    """True for a parameter owned by a ROUTED expert, judged by qualified name.

    Name-based because this module must import without torch, the same
    constraint that makes the embedding walk match on class name.

    BOTH STORAGE LAYOUTS COUNT, and the measured checkpoint is why. An earlier
    version of this function required an INDEXED ``experts.<i>`` segment, which
    is how ``mtp`` stores its experts. The language model does not: on
    Qwen3.5-35B-A3B every one of its 256 experts per layer is FUSED into two
    stacked tensors, ``mlp.experts.gate_up_proj`` and ``mlp.experts.down_proj``,
    carrying no index at all. Requiring an index therefore found 0 routed
    parameters in the language model of a model that is 90% experts by weight --
    768 indexed tensors under ``mtp.`` and none under ``model.``. Matching the
    ``experts`` path SEGMENT catches both layouts.

    A SHARED expert is deliberately excluded: Qwen3.5 declares
    ``shared_expert_intermediate_size`` and runs that expert for EVERY token, so
    scaling it by the routing fraction would understate the work. On the
    measured checkpoint no tensor matches both patterns, so this exclusion is
    belt-and-braces rather than load-bearing -- it is kept because it makes the
    "routed" contract true by construction rather than by luck of naming.

    An unrecognised spelling leaves the parameter counted as always-active,
    which OVERSTATES FLOPs and therefore UNDERSTATES MFU. That direction is
    chosen on purpose: the failure mode worth designing against is a
    performance number that flatters the framework.
    """
    if "shared_expert" in qualified_name:
        return False
    return "experts" in qualified_name.split(".")


def _positive_int(source: object, names: tuple[str, ...]) -> int | None:
    """First of ``names`` on ``source`` whose VALUE is a positive int, else None.

    Keyed on the value, never on key presence, because the measured
    gemma-4-31B config -- a DENSE model -- states ``num_experts: null``. A
    ``hasattr``/``in`` test therefore classifies it as a Mixture-of-Experts
    model with an unknown expert count and destroys its MFU, which is the same
    failure shape as reading a composite config flatly, one level further in.

    ``bool`` is excluded because it is an ``int`` subclass and a flag is not a
    count -- the rule this module already applies to layer and hidden sizes.
    """
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value
    return None


@dataclass(frozen=True)
class FlopsModel:
    """Declares how FLOPs per token are counted, so the formula is auditable.

    WHAT IS CLAIMED: a per-token FLOP figure computed by the stated formula
    ``6 * non_embedding_parameters + 12 * layers * hidden_size *
    sequence_length`` from the declared fields.

    WHAT IS NOT CLAIMED: that this is the true FLOP count. The estimate
    ignores layernorm, softmax, activation and optimizer FLOPs. It is the
    same estimate used by the PaLM and Megatron-LM papers, and that is the
    only reason to prefer it over a bespoke formula: MFU computed this way is
    comparable to published MFU, and a more accurate but privately-defined
    count would make every comparison against a published number a category
    error. The fields are data, not code, so the declaration travels with the
    number and the formula can be re-derived by any reader.
    """

    parameters: int
    non_embedding_parameters: int
    layers: int
    hidden_size: int
    sequence_length: int
    routed_expert_parameters: int = 0
    """Parameters living in ROUTED experts -- the ones a router selects per token.

    Zero on a dense model, and zero is what keeps the dense estimate bit-identical
    to the pre-#529 formula. A SHARED expert (Qwen3.5 declares
    ``shared_expert_intermediate_size``) runs for every token and therefore does
    NOT belong here: scaling it by the routing fraction would understate."""

    experts_total: int | None = None
    """Routed experts the layer owns. ``None`` means dense, not unknown."""

    experts_active: int | None = None
    """Routed experts that run per token (``num_experts_per_tok``)."""

    unmeasured_reason: str | None = None
    """Set when a FLOP count cannot honestly be produced for this model.

    Callers must consult this BEFORE :attr:`flops_per_token`. It exists because
    the alternative -- returning the dense number for a model whose active
    expert count is unknown -- overstates MFU by the total/active ratio, about
    6x on gemma-4-26B-A4B, while looking entirely plausible (#529)."""

    @property
    def flops_per_token(self) -> int:
        """The standard transformer estimate: ``6N_active + 12 * L * h * s``.

        The ``6 *`` term is forward+backward matmul work per non-embedding
        parameter (2 FLOPs forward, 4 backward). The ``12 * L * h * s`` term
        is the attention score/context matmuls, which scale with sequence
        length and are NOT captured by any parameter count -- dropping it
        would understate long-sequence runs precisely where attention
        dominates. Both terms are estimates; see the class docstring for what
        is not claimed.

        ACTIVE, NOT TOTAL (#529). On a Mixture-of-Experts model only
        ``experts_active`` of ``experts_total`` routed experts run per token, so
        charging every expert overstates the work by the total/active ratio --
        measured on this estate as roughly 6x for gemma-4-26B-A4B and 12x for
        Qwen3.5-122B-A10B. The reference implementations (Megatron-LM,
        torchtitan) charge active parameters, so charging total would also break
        the comparability that is the entire reason for using this formula.

        THE DENSE PATH IS BIT-IDENTICAL, not merely close: when
        ``routed_expert_parameters`` is zero the expression below returns before
        any division happens, so no float ever enters the arithmetic and the
        integer result is the same one the pre-#529 formula produced. That
        matters because a refactor that quietly moved every dense MFU number by
        a rounding step would invalidate comparisons against runs already
        published.
        """
        dense = (
            6 * self.non_embedding_parameters
            + 12 * self.layers * self.hidden_size * self.sequence_length
        )
        if self.routed_expert_parameters <= 0:
            return dense
        if not self.experts_total or not self.experts_active:
            # Unreachable through the constructors, which set unmeasured_reason
            # instead. Charging the dense figure here would be the silent
            # overstatement this property exists to prevent, so charge the
            # routed experts in full: wrong in the SAFE direction, because an
            # overstated FLOP count understates MFU.
            return dense
        inactive = self.routed_expert_parameters * (self.experts_total - self.experts_active)
        return dense - 6 * (inactive // self.experts_total)

    @classmethod
    def from_hf_config(
        cls,
        config: object,
        sequence_length: int,
        parameters: int,
        non_embedding_parameters: int,
        routed_expert_parameters: int = 0,
    ) -> FlopsModel | None:
        """Build from an HF config, or return None -- never a guess.

        ``num_hidden_layers`` and ``hidden_size`` are read from
        ``config.text_config`` FIRST when that attribute is present, and from
        the flat config only when it is not. A vision-language model states
        these fields on the text sub-config, and a flat getattr on the
        composite config silently returns the wrong value or None -- the
        producer/consumer defect class where each side is internally
        consistent and the two are never introduced. When text_config exists
        there is deliberately NO fallback to the flat config: on a VLM the
        flat attributes belong to a different sub-model, and a wrong value is
        worse than None because None is honest. Any absent field returns
        None, because a partial FlopsModel is a guessed formula.
        """
        text_config = getattr(config, "text_config", None)
        source = text_config if text_config is not None else config
        layers = getattr(source, "num_hidden_layers", None)
        hidden = getattr(source, "hidden_size", None)
        # bool is an int subclass; a True here is a flag, not a layer count,
        # and the codebase excludes it wherever numbers are read.
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
            return None
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            return None
        experts_total = _positive_int(source, ("num_experts", "num_local_experts"))
        experts_active = _positive_int(
            source, ("num_experts_per_tok", "num_experts_per_token", "top_k")
        )
        reason: str | None = None
        if experts_total is not None:
            if experts_active is None:
                reason = (
                    f"UNMEASURED: the config declares {experts_total} experts but states "
                    "no per-token expert count under any of num_experts_per_tok, "
                    "num_experts_per_token or top_k, so how many experts run per token is "
                    "unknown. Charging all of them would overstate MFU by the total/active "
                    "ratio -- measured as about 6x on gemma-4-26B-A4B, which is exactly "
                    "this case. Declare the per-token count to measure it (#529)"
                )
            elif routed_expert_parameters <= 0:
                reason = (
                    f"UNMEASURED: the config declares {experts_total} experts but no "
                    "routed-expert parameter count was supplied, so the active fraction "
                    "cannot be applied to anything. Pass routed_expert_parameters, or use "
                    "from_model which counts them off the live module tree (#529)"
                )
        return cls(
            parameters=parameters,
            non_embedding_parameters=non_embedding_parameters,
            layers=layers,
            hidden_size=hidden,
            sequence_length=sequence_length,
            routed_expert_parameters=max(0, routed_expert_parameters),
            experts_total=experts_total,
            experts_active=experts_active,
            unmeasured_reason=reason,
        )

    @classmethod
    def from_model(cls, model: object, sequence_length: int) -> FlopsModel | None:
        """Build from a live model, counting its parameters, or return None.

        The non-embedding count is total parameters MINUS every parameter
        owned by an embedding module. Embeddings are identified by their
        module class NAME rather than by ``isinstance(m, nn.Embedding)``, so
        this file keeps its no-torch-import property; the cost is that a
        custom embedding class named something else is counted as
        non-embedding, which OVERSTATES the FLOPs estimate rather than
        understating it. That direction is deliberate: an overstated
        denominator understates MFU, and a performance number that flatters
        the framework is the one failure mode worth designing against.

        Tied embeddings are counted ONCE, because the walk is over unique
        parameter objects by identity -- a tied lm_head shares the embedding
        tensor, and counting it twice would subtract the same 671M
        parameters twice on a model like gemma-4.

        Returns None when the model exposes no config or the config is
        missing the fields ``from_hf_config`` requires. It never guesses: a
        FlopsModel built on a guessed layer count would produce a TFLOP/s
        number with no measurement behind it.
        """
        config = getattr(model, "config", None)
        if config is None:
            return None
        named_parameters = getattr(model, "named_parameters", None)
        named_modules = getattr(model, "named_modules", None)
        if named_parameters is None or named_modules is None:
            return None
        by_id: dict[int, int] = {}
        for _, parameter in named_parameters():
            numel = getattr(parameter, "numel", None)
            if numel is None:
                return None
            by_id[id(parameter)] = int(numel())
        embedding_ids: set[int] = set()
        for _, module in named_modules():
            if type(module).__name__ != "Embedding":
                continue
            own = getattr(module, "parameters", None)
            if own is None:
                continue
            for parameter in own(recurse=False):
                embedding_ids.add(id(parameter))
        parameters = sum(by_id.values())
        non_embedding = sum(n for pid, n in by_id.items() if pid not in embedding_ids)
        if parameters < 1 or non_embedding < 1:
            return None
        routed = 0
        for name, parameter in named_parameters():
            if _is_routed_expert_parameter(name):
                numel = getattr(parameter, "numel", None)
                if numel is not None:
                    routed += int(numel())
        return cls.from_hf_config(
            config,
            sequence_length=sequence_length,
            parameters=parameters,
            non_embedding_parameters=non_embedding,
            routed_expert_parameters=routed,
        )


@dataclass(frozen=True)
class DevicePeak:
    """A declared device peak throughput, with mandatory provenance.

    WHAT IS CLAIMED: that the operator stated a peak bf16 dense TFLOP/s
    figure and stated WHERE it came from -- a vendor datasheet line, a
    measured GEMM sweep. ``source`` is mandatory free text because a peak
    with no provenance cannot be audited, and an unauditable peak makes the
    MFU derived from it unauditable.

    WHAT IS NOT CLAIMED: that the figure was measured by this run, or that it
    is correct. The declaration is the operator's statement; this dataclass
    only refuses to carry one that cannot name its origin.

    There is deliberately NO built-in device table. A table would be the same
    guess with extra steps: vendor peaks differ by SKU, by clock, and by
    whether sparsity is counted, and an operator who cannot state their own
    peak cannot defend the MFU number derived from it. Refusing to ship the
    table is what keeps MFU unmeasured -- rather than wrong -- on hardware
    the operator has not characterised.
    """

    name: str
    bf16_dense_tflops: float
    source: str

    def __post_init__(self) -> None:
        # Refused at statement time, not at MFU-computation time: an
        # out-of-range declaration is a config error with a named field,
        # mirroring how TrainConfig range-checks its axes in __post_init__.
        if not self.source.strip():
            raise ValueError(
                "DevicePeak.source must name where the number came from (a "
                "vendor datasheet line, a measured GEMM sweep); a peak with "
                "no provenance cannot be audited, so it is refused rather "
                "than carried"
            )
        if not math.isfinite(self.bf16_dense_tflops) or self.bf16_dense_tflops <= 0:
            raise ValueError(
                f"DevicePeak.bf16_dense_tflops must be a positive finite "
                f"number, got {self.bf16_dense_tflops!r}; a non-positive "
                "peak would make MFU a division by zero or a negative "
                "efficiency, both of which are fabricated numbers"
            )

    @classmethod
    def from_env(cls) -> DevicePeak | None:
        """Read a declared peak from FS_DEVICE_PEAK_TFLOPS / FS_DEVICE_PEAK_SOURCE.

        Returns None when either variable is ABSENT -- absence means not
        declared, and not declared means MFU stays unmeasured. A malformed
        value is a different state from an absent one: the operator DID
        declare, and declared something unusable, so a non-numeric
        FS_DEVICE_PEAK_TFLOPS or an empty FS_DEVICE_PEAK_SOURCE raises
        ValueError with the field named rather than being silently demoted to
        "unmeasured", which would hide the operator's mistake inside an
        honest-looking gap.
        """
        tflops_raw = os.environ.get("FS_DEVICE_PEAK_TFLOPS")
        source = os.environ.get("FS_DEVICE_PEAK_SOURCE")
        if tflops_raw is None or source is None:
            return None
        try:
            tflops = float(tflops_raw)
        except ValueError as exc:
            raise ValueError(
                f"FS_DEVICE_PEAK_TFLOPS={tflops_raw!r} is not a number; the "
                "operator declared a peak, so the operator corrects it -- "
                "this is refused rather than coerced or ignored"
            ) from exc
        name = os.environ.get(
            "FS_DEVICE_PEAK_NAME", "device declared through FS_DEVICE_PEAK_TFLOPS"
        )
        return cls(name=name, bf16_dense_tflops=tflops, source=source)


def _world_size() -> int:
    """Read the distributed world size, guarded so this never explodes on CPU.

    A process with no initialised process group IS a world of one -- that is
    a measurement of the topology, not a default -- so the fallback is 1 and
    the metric stays "measured". torch is imported lazily because this module
    must import and run in environments where torch is absent entirely (the
    gates import the package at collection time).
    """
    try:
        import torch
    except ImportError:
        return 1
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except Exception:  # noqa: BLE001 -- telemetry must never turn a run RED
        return 1
    return 1


def _read_cuda_memory_bytes() -> tuple[int, int, int] | str:
    """Read (peak allocated, peak reserved, device total) bytes, or a reason.

    Returns either the three counters or a reason STRING -- one return type
    per outcome, so the caller cannot accidentally pair a None value with a
    None reason. Every torch access is guarded: on a CPU-only run there is no
    counter to read, and that is an unmeasured metric with a named cause, not
    an exception.
    """
    try:
        import torch
    except ImportError:
        return (
            "UNMEASURED: torch is not importable in this environment, so no "
            "CUDA memory counter exists to read"
        )
    try:
        if not torch.cuda.is_available():
            return (
                "UNMEASURED: torch.cuda.is_available() is False -- this run "
                "has no CUDA device, so there is no GPU memory counter to read"
            )
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        total = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    except Exception as exc:  # noqa: BLE001 -- telemetry must never turn a run RED
        return f"UNMEASURED: reading torch.cuda memory counters raised {type(exc).__name__}: {exc}"
    return (allocated, reserved, total)


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, stated so the convention is auditable.

    The rank is ``(n - 1) * q`` with linear interpolation between neighbours
    -- the convention numpy calls "linear". It is stated here because p50 of
    an even-count list is the midpoint of the two central values under this
    convention and the upper of the two under nearest-rank, and a percentile
    whose convention is unstated is a number two readers can disagree about
    while both are right.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


_NO_REASON_REASON = (
    "UNMEASURED: this entry carries neither a value nor a reason, which is a "
    "defect in the perf plane itself rather than a property of the run -- "
    "report it as a FoundationScale bug"
)


def _entry(value: object, reason: str | None, source: str = "derived") -> tuple[object, str]:
    """Pair a value with its source, or None with the reason it is missing.

    Every metric below is built the same way: compute a value, or set a reason
    string saying which input was absent. This function is the one place that
    turns that pair into the WORKING shape ``(value_or_None, source_or_reason)``
    used everywhere inside this module, and it exists for two reasons beyond
    tidiness.

    First, it makes the package rule -- an unmeasured number is None paired
    with a reason that names the missing input, never 0.0 and never absent --
    a single enforced code path instead of a convention repeated at a dozen
    sites, where the thirteenth would eventually differ.

    Second, it closes the shape that means nothing. ``(None, None)`` says a
    number is missing and declines to say why, which is strictly worse than
    either a number or an explanation; it is also unreachable by construction,
    since every caller sets a reason on the branch where the value is None.
    Unreachable-by-construction is exactly the invariant that decays under
    later edits, so it is handled rather than asserted: the entry degrades to
    a reason naming the perf plane as the culprit. It does NOT raise --
    ``summary()`` runs after training has already succeeded, and an instrument
    that can turn a finished run into a traceback is not an instrument.

    The working shape is NOT the shape the manifest stores. ``summary()``
    converts every entry through :func:`_as_manifest_entries` on the way out;
    see that function for why the two differ and why the conversion is a
    boundary rather than a rule each site is asked to remember.
    """
    if value is not None:
        return (value, source)
    return (None, reason or _NO_REASON_REASON)


#: The one token :class:`~foundationscale.provenance.manifest.TelemetryEntry`
#: accepts for an absent measurement. Spelled once so a typo is an ImportError
#: here rather than a ValueError thrown at the end of somebody's training run.
_UNMEASURED_SOURCE = "unmeasured"


def _as_manifest_entries(entries: dict[str, tuple[object, str]]) -> dict[str, tuple[object, str]]:
    """Convert the working shape into the shape ``TelemetryEntry`` accepts.

    #518, and it is worth stating plainly because it cost a RED run: this
    module's internal pairs and the manifest's pairs read identically and mean
    opposite things. Inside here an absent metric is ``(None, "UNMEASURED: the
    counter was never fed")`` -- value first, explanation second, which is the
    natural way to write it at a computation site. ``TelemetryEntry`` stores an
    absent metric as ``("UNMEASURED: the counter was never fed", "unmeasured")``
    -- the REASON occupies the value slot, because ``source`` is a closed
    vocabulary of exactly three tokens and ``value=None`` under ``measured`` is
    itself a refused shape (#427). Handing the working pair straight to the
    manifest therefore puts a sentence where a token belongs, and
    ``TelemetryEntry.__post_init__`` raises ``ValueError``.

    That raise does not surface as a tidy validation message. It escapes
    ``train()`` after the model is trained and the checkpoint is written, is
    caught by the thin path's last-resort handler and adjudicated **RED (5)** --
    a completed, correct training run reported as a failure by its own
    instrument. MEASURED: it fires on any run where even one perf metric is
    unmeasured, which includes every run that does not declare
    ``FS_DEVICE_PEAK_TFLOPS`` (``perf_mfu`` and ``perf_device_peak_tflops`` are
    both unmeasured there) and every run short enough to leave the steady-state
    window empty. The default path, in other words.

    The conversion lives here, at the single exit, rather than at the twenty-odd
    sites that build entries. Those sites already have the harder job of getting
    the reason right, and a shape rule they must each remember is a rule the
    twenty-first site will break -- which is exactly how this defect arrived.
    A boundary is total over the dict by construction; a convention is not.
    """
    shaped: dict[str, tuple[object, str]] = {}
    for key, (value, source) in entries.items():
        if value is None:
            # ``source`` is carrying the reason. Empty is not reachable from
            # _entry, which substitutes _NO_REASON_REASON, but a direct tuple
            # site could produce it -- and an unmeasured entry with an empty
            # reason is refused by the manifest just as hard as a None value.
            shaped[key] = (source or _NO_REASON_REASON, _UNMEASURED_SOURCE)
        else:
            shaped[key] = (value, source)
    return shaped


class StepTelemetry(_CallbackBase):
    """A TrainerCallback that times steps and assembles the perf summary.

    WHAT IS CLAIMED
    ---------------
    * Wall-clock durations, taken with ``time.perf_counter()`` at the
      callback boundaries transformers provides, split into a warmup bucket
      (step index < warmup_steps) and a steady-state bucket. Warmup steps are
      TIMED but excluded from every steady-state aggregate, and both counts
      are reported: a benchmark that averages in the first step measures the
      CUDA allocator and the autotuner, not the model.
    * ``between_steps_s``: the gap between one step's end and the next step's
      begin. This is the DATA-STARVATION window -- dataloader fetch,
      collation, H2D copy and callback overhead all live inside it -- but it
      is named honestly: it is the gap between steps, not a proof that the
      dataloader was the cause.
    * ``optimizer_step_s``: the on_pre_optimizer_step -> on_optimizer_step
      bracket. Under DDP the gradient all-reduce is OVERLAPPED with backward,
      so this bracket captures only the synchronisation TAIL, not total
      communication -- calling it "communication time" would overstate what
      was measured.
    * Token counts from ``state.num_input_tokens_seen`` when the trainer
      counted tokens; see _TOKENS_UNMEASURED_REASON for why a zero counter
      is treated as unread rather than measured.

    WHAT IS NOT CLAIMED
    -------------------
    * Any number whose inputs were absent. Every unmeasured key carries a
      reason string naming the specific missing input; None is never
      coerced to 0.0, because an unmeasured metric is not an error and not
      a zero.
    * That the hooks fire on every transformers release. If
      on_pre_optimizer_step never fires, the optimizer metric says so
      instead of reporting a stale or partial bracket.

    The hooks perform no I/O and no torch access, so they cannot turn a run
    RED; all torch access lives in summary(), guarded behind helpers that
    return reasons instead of raising.
    """

    def __init__(
        self,
        *,
        warmup_steps: int = 3,
        flops_model: FlopsModel | None = None,
        device_peak: DevicePeak | None = None,
    ) -> None:
        super().__init__()
        if warmup_steps < 0:
            # Refused at statement time: a negative warmup is a config the
            # operator did not mean, and it would silently classify every
            # step as steady-state.
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self._warmup_steps = warmup_steps
        self._flops_model = flops_model
        self._device_peak = device_peak
        self._reset_state()

    def _reset_state(self) -> None:
        """Drop all accumulated state so a reused callback cannot leak runs.

        on_train_begin calls this: a callback object attached to two Trainers
        in sequence must not average the first run's steps into the second
        run's summary -- the producer/consumer defect class again, where each
        run is internally consistent and the summary silently mixes them.
        """
        self._t0: float | None = None
        self._step_begin: float | None = None
        self._last_step_end: float | None = None
        self._pre_optimizer_t: float | None = None
        self._pre_optimizer_seen = False
        self._current_between_s: float | None = None
        self._current_optimizer_s = 0.0
        self._steps_ended = 0
        self._warmup_step_times: list[float] = []
        self._steady_step_times: list[float] = []
        self._warmup_between_s: list[float] = []
        self._steady_between_s: list[float] = []
        self._warmup_optimizer_s: list[float] = []
        self._steady_optimizer_s: list[float] = []
        self._per_device_batch: int | None = None
        self._grad_accum: int | None = None
        self._batch_reason = (
            "UNMEASURED: on_train_begin never ran, so the batch geometry was "
            "never read from TrainingArguments"
        )
        self._tokens_last: int | None = None
        self._tokens_last_t: float | None = None
        self._steady_tokens_start: int | None = None
        self._steady_tokens_start_t: float | None = None
        self._tokens_reason: str | None = None

    def _capture_batch_geometry(self, args: object) -> None:
        """Read per-device batch size and accumulation from TrainingArguments.

        These two are the only route from steps/s to samples/s. Read once at
        train begin, with the bool guard the codebase applies wherever
        numbers are read; a missing or non-positive field leaves the samples
        axis unmeasured with the field named, never defaulted to 1 -- a
        defaulted batch size would launder an absent statement into a
        measured throughput.
        """
        per_device = getattr(args, "per_device_train_batch_size", None)
        if isinstance(per_device, bool) or not isinstance(per_device, int) or per_device < 1:
            self._batch_reason = (
                "UNMEASURED: TrainingArguments.per_device_train_batch_size "
                "was absent or not a positive integer at train begin, so "
                "samples per step cannot be derived"
            )
            return
        accum = getattr(args, "gradient_accumulation_steps", None)
        if isinstance(accum, bool) or not isinstance(accum, int) or accum < 1:
            self._batch_reason = (
                "UNMEASURED: TrainingArguments.gradient_accumulation_steps "
                "was absent or not a positive integer at train begin, so "
                "samples per step cannot be derived"
            )
            return
        self._per_device_batch = per_device
        self._grad_accum = accum

    def on_train_begin(
        self,
        args: object,  # noqa: ARG002 -- TrainerCallback's signature, not ours
        state: object,  # noqa: ARG002
        control: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        self._reset_state()
        self._t0 = time.perf_counter()
        self._capture_batch_geometry(args)

    def on_step_begin(
        self,
        args: object,  # noqa: ARG002 -- TrainerCallback's signature, not ours
        state: object,  # noqa: ARG002
        control: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        now = time.perf_counter()
        # The gap since the previous step's end is the data-starvation
        # window. The FIRST step of a run has no predecessor and therefore no
        # gap -- recording 0.0 there would fabricate a "no stall" observation
        # from an interval that does not exist.
        if self._last_step_end is not None:
            self._current_between_s = now - self._last_step_end
        else:
            self._current_between_s = None
        self._current_optimizer_s = 0.0
        self._step_begin = now

    def on_pre_optimizer_step(
        self,
        args: object,  # noqa: ARG002 -- TrainerCallback's signature, not ours
        state: object,  # noqa: ARG002
        control: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        self._pre_optimizer_seen = True
        self._pre_optimizer_t = time.perf_counter()

    def on_optimizer_step(
        self,
        args: object,  # noqa: ARG002 -- TrainerCallback's signature, not ours
        state: object,  # noqa: ARG002
        control: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        now = time.perf_counter()
        # Only close a bracket that was opened: if on_pre_optimizer_step did
        # not fire on this transformers release there is no interval to time,
        # and pairing this hook with the previous step's open bracket would
        # manufacture a duration from two unrelated events.
        if self._pre_optimizer_t is not None:
            self._current_optimizer_s += now - self._pre_optimizer_t
            self._pre_optimizer_t = None

    def on_step_end(
        self,
        args: object,  # noqa: ARG002 -- TrainerCallback's signature, not ours
        state: object,
        control: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        now = time.perf_counter()
        if self._step_begin is None:
            # A step end with no begin has never been observed; ignoring it
            # is honest, timing it against stale state would not be.
            return
        step_s = now - self._step_begin
        steady = self._steps_ended >= self._warmup_steps
        if steady:
            self._steady_step_times.append(step_s)
            if self._current_between_s is not None:
                self._steady_between_s.append(self._current_between_s)
            if self._pre_optimizer_seen:
                self._steady_optimizer_s.append(self._current_optimizer_s)
        else:
            self._warmup_step_times.append(step_s)
            if self._current_between_s is not None:
                self._warmup_between_s.append(self._current_between_s)
            if self._pre_optimizer_seen:
                self._warmup_optimizer_s.append(self._current_optimizer_s)
        tokens = getattr(state, "num_input_tokens_seen", None)
        # bool excluded as everywhere numbers are read. Zero is excluded for
        # the reason _TOKENS_UNMEASURED_REASON states: it is the signature of
        # a counter the trainer never fed, not a measurement.
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or tokens <= 0:
            self._tokens_reason = _TOKENS_UNMEASURED_REASON
        else:
            tokens_int = int(tokens)
            self._tokens_last = tokens_int
            self._tokens_last_t = now
            if steady and self._steady_tokens_start is None:
                # First steady-state reading: the rate is later computed as
                # the DIFFERENCE between this and the last reading over the
                # wall time between them, so warmup tokens never enter the
                # steady-state rate.
                self._steady_tokens_start = tokens_int
                self._steady_tokens_start_t = now
        self._steps_ended += 1
        self._last_step_end = now
        self._step_begin = None

    def summary(self) -> dict[str, tuple[object, str]]:
        """Assemble the perf telemetry dict under the provenance contract.

        Every key is always present; every value is a ``(value, source)``
        2-tuple in the shape
        :class:`~foundationscale.provenance.manifest.TelemetryEntry` stores,
        which is the shape loop.py merges straight into the manifest. A
        measured or derived metric is ``(number, "measured"|"derived")``. An
        unmeasured one is ``(reason, "unmeasured")`` -- the reason naming the
        specific missing input occupies the VALUE slot, never None and never
        0.0, because the manifest's ``source`` is a closed three-token
        vocabulary and absence has to be legible as text (#518; the shape
        used inside this method is the other way round, see
        :func:`_as_manifest_entries`). The stated formulas (the "derived"
        contract):

        * perf_step_time_mean_s / p50 / p90 -- mean and linear-interpolation
          percentiles (see _percentile) over the steady-state step times.
        * perf_between_steps_mean_s -- mean of the steady-state between-step
          gaps.
        * perf_dataloader_stall_fraction -- between_steps / (between_steps +
          step_time), summed over the steady-state window. A value near zero
          means the GPUs were not waiting on input; this is the metric to
          read BEFORE blaming the model for low utilisation.
        * perf_optimizer_sync_mean_s -- mean of the steady-state optimizer
          brackets (the synchronisation tail, NOT total communication).
        * perf_tokens_per_second -- (last token count - first steady-state
          token count) / wall time between those two readings. A rate is a
          difference between two readings; one reading is a level, not a
          rate, so a single steady-state step leaves this unmeasured.
        * perf_tokens_per_second_per_device -- the above / world size.
        * perf_samples_per_second -- (per_device_batch * grad_accum *
          world_size) / mean steady-state step time.
        * perf_model_tflops_per_second -- tokens/s * flops_per_token / 1e12.
        * perf_model_tflops_per_second_per_device -- the above / world size.
        * perf_mfu -- per-device model TFLOP/s / declared device peak.
          UNMEASURED unless a DevicePeak was declared; there is no default
          peak.
        * perf_memory_utilisation_fraction -- peak allocated / device total.
        """
        world = _world_size()
        warmup_count = len(self._warmup_step_times)
        steady_count = len(self._steady_step_times)
        observed = warmup_count + steady_count
        no_steady_reason = (
            f"UNMEASURED: the steady-state window is empty -- {observed} "
            f"step(s) ended and warmup_steps={self._warmup_steps} held "
            f"{warmup_count} of them in the warmup bucket; an average over "
            "an empty window is a fabricated number, not a low one"
        )

        mean_step_value: float | None = None
        if steady_count:
            mean_step_value = sum(self._steady_step_times) / steady_count
            step_time_mean: tuple[object, str] = (mean_step_value, "derived")
            step_time_p50: tuple[object, str] = (
                _percentile(self._steady_step_times, 0.50),
                "derived",
            )
            step_time_p90: tuple[object, str] = (
                _percentile(self._steady_step_times, 0.90),
                "derived",
            )
        else:
            step_time_mean = (None, no_steady_reason)
            step_time_p50 = (None, no_steady_reason)
            step_time_p90 = (None, no_steady_reason)

        no_gap_reason = (
            "UNMEASURED: no between-step gap was recorded in the "
            "steady-state window -- the first step of a run has no "
            "predecessor, so a steady window of a single step has no gap to "
            "measure, and a stall fraction without a measured gap would be "
            "a fabricated zero"
        )
        if self._steady_between_s:
            between_mean: tuple[object, str] = (
                sum(self._steady_between_s) / len(self._steady_between_s),
                "derived",
            )
        elif steady_count:
            between_mean = (None, no_gap_reason)
        else:
            between_mean = (None, no_steady_reason)

        if steady_count and self._steady_between_s:
            step_sum = sum(self._steady_step_times)
            between_sum = sum(self._steady_between_s)
            stall_fraction: tuple[object, str] = (
                between_sum / (between_sum + step_sum),
                "derived",
            )
        elif steady_count:
            stall_fraction = (None, no_gap_reason)
        else:
            stall_fraction = (None, no_steady_reason)

        if not self._pre_optimizer_seen:
            optimizer_sync_mean: tuple[object, str] = (
                None,
                (
                    "UNMEASURED: on_pre_optimizer_step never fired -- this "
                    "transformers release does not bracket the optimizer "
                    "step, so there is no synchronisation-tail interval to "
                    "time"
                ),
            )
        elif self._steady_optimizer_s:
            optimizer_sync_mean = (
                sum(self._steady_optimizer_s) / len(self._steady_optimizer_s),
                "derived",
            )
        else:
            optimizer_sync_mean = (None, no_steady_reason)

        token_axis_reason: str | None = None
        if self._tokens_last is not None:
            tokens_total: tuple[object, str] = (self._tokens_last, "measured")
        else:
            token_axis_reason = self._tokens_reason or _TOKENS_UNMEASURED_REASON
            tokens_total = (None, token_axis_reason)

        tps: float | None = None
        tps_reason: str | None = None
        if token_axis_reason is not None:
            tps_reason = token_axis_reason
        elif self._tokens_last is None or self._tokens_last_t is None:
            tps_reason = _TOKENS_UNMEASURED_REASON
        elif self._steady_tokens_start is None or self._steady_tokens_start_t is None:
            tps_reason = (
                "UNMEASURED: no steady-state step carried a token count -- "
                "a steady-state rate is the difference between two readings "
                "inside the window, and there is not even a first one"
            )
        else:
            window_s = self._tokens_last_t - self._steady_tokens_start_t
            window_tokens = self._tokens_last - self._steady_tokens_start
            if window_s > 0:
                tps = window_tokens / window_s
            else:
                tps_reason = (
                    "UNMEASURED: exactly one steady-state step carried a "
                    "token count -- one reading is a level, not a rate, and "
                    "a second reading is required before a tokens/s figure "
                    "exists"
                )
        tps_entry = _entry(tps, tps_reason)
        tps_device_entry = _entry(tps / world if tps is not None else None, tps_reason)

        if self._per_device_batch is None or self._grad_accum is None:
            sps_entry: tuple[object, str] = (None, self._batch_reason)
        elif mean_step_value is None:
            sps_entry = (None, no_steady_reason)
        else:
            sps_entry = (
                (self._per_device_batch * self._grad_accum * world) / mean_step_value,
                "derived",
            )

        model_tflops: float | None = None
        model_tflops_reason: str | None = None
        if self._flops_model is None:
            model_tflops_reason = (
                "UNMEASURED: no FlopsModel was declared -- pass "
                "flops_model=FlopsModel(...) or "
                "FlopsModel.from_hf_config(...) so that FLOPs per token is "
                "an auditable formula rather than folklore"
            )
        elif self._flops_model.unmeasured_reason is not None:
            # Consulted BEFORE the arithmetic and before tps, because this says
            # the FORMULA cannot be evaluated honestly for this model -- a fact
            # about the declaration, not about whether this run happened to
            # count tokens. Reporting a dense number here is the #529 defect.
            model_tflops_reason = self._flops_model.unmeasured_reason
        elif tps is None:
            model_tflops_reason = tps_reason
        else:
            model_tflops = tps * self._flops_model.flops_per_token / 1e12
        model_tflops_entry = _entry(model_tflops, model_tflops_reason)
        model_tflops_device: float | None = (
            model_tflops / world if model_tflops is not None else None
        )
        model_tflops_device_entry = _entry(model_tflops_device, model_tflops_reason)

        if self._device_peak is None:
            mfu_entry: tuple[object, str] = (None, _NO_PEAK_REASON)
            peak_entry: tuple[object, str] = (None, _NO_PEAK_REASON)
        else:
            mfu_entry = _entry(
                (
                    model_tflops_device / self._device_peak.bf16_dense_tflops
                    if model_tflops_device is not None
                    else None
                ),
                model_tflops_reason,
            )
            # The tri-state contract has no "declared" label, and adding a
            # fourth was rejected to keep this dict shape-identical to the
            # loop's telemetry. The peak is reported "measured" because the
            # operator's sourced declaration is the instrument of record for
            # this metric -- the mandatory DevicePeak.source string is the
            # audit trail, and reporting a declared, auditable number as
            # unmeasured would throw away real information. MFU derived from
            # it carries "derived", which is where the declaration's
            # epistemic weight honestly sits.
            peak_entry = (self._device_peak.bf16_dense_tflops, "measured")

        memory = _read_cuda_memory_bytes()
        if isinstance(memory, str):
            allocated_entry: tuple[object, str] = (None, memory)
            reserved_entry: tuple[object, str] = (None, memory)
            total_entry: tuple[object, str] = (None, memory)
            util_entry: tuple[object, str] = (None, memory)
        else:
            allocated_bytes, reserved_bytes, total_bytes = memory
            allocated_entry = (allocated_bytes, "measured")
            reserved_entry = (reserved_bytes, "measured")
            total_entry = (total_bytes, "measured")
            if total_bytes > 0:
                util_entry = (allocated_bytes / total_bytes, "derived")
            else:
                util_entry = (
                    None,
                    (
                        "UNMEASURED: torch.cuda.get_device_properties "
                        "reported 0 total bytes, so no utilisation fraction "
                        "can be formed"
                    ),
                )

        # #518: the dict below is in this module's WORKING shape; the caller is
        # handed the manifest's. Every exit from summary() goes through this one
        # call, which is the whole point of it being here and not at each site.
        return _as_manifest_entries(
            {
                "perf_steps_observed": (observed, "measured"),
                "perf_steps_warmup": (warmup_count, "measured"),
                "perf_steps_steady": (steady_count, "measured"),
                "perf_step_time_mean_s": step_time_mean,
                "perf_step_time_p50_s": step_time_p50,
                "perf_step_time_p90_s": step_time_p90,
                "perf_between_steps_mean_s": between_mean,
                "perf_dataloader_stall_fraction": stall_fraction,
                "perf_optimizer_sync_mean_s": optimizer_sync_mean,
                "perf_tokens_total": tokens_total,
                "perf_tokens_per_second": tps_entry,
                "perf_tokens_per_second_per_device": tps_device_entry,
                "perf_samples_per_second": sps_entry,
                "perf_model_tflops_per_second": model_tflops_entry,
                "perf_model_tflops_per_second_per_device": model_tflops_device_entry,
                "perf_mfu": mfu_entry,
                "perf_device_peak_tflops": peak_entry,
                "perf_world_size": (world, "measured"),
                "perf_peak_memory_allocated_bytes": allocated_entry,
                "perf_peak_memory_reserved_bytes": reserved_entry,
                "perf_device_memory_total_bytes": total_entry,
                "perf_memory_utilisation_fraction": util_entry,
            }
        )


# Units of the perf telemetry keys above, keyed by metric name -- the same
# convention as loop.py's _TELEMETRY_UNITS: the unit describes the METRIC, so
# an unmeasured entry keeps the unit of the thing it failed to measure.
# perf_dataloader_stall_fraction, perf_mfu and perf_memory_utilisation_fraction
# are deliberately ABSENT: they are unitless ratios, and the convention is
# that a key absent from this table is unitless. This comment exists because
# an omitted unitless key is otherwise indistinguishable from an oversight --
# the same failure the total_flos row taught in loop.py, read from the other
# side: there a row existed with no producer, here producers exist with no
# unit, and only one of those is drift.
PERF_TELEMETRY_UNITS: dict[str, str] = {
    "perf_steps_observed": "steps",
    "perf_steps_warmup": "steps",
    "perf_steps_steady": "steps",
    "perf_step_time_mean_s": "s",
    "perf_step_time_p50_s": "s",
    "perf_step_time_p90_s": "s",
    "perf_between_steps_mean_s": "s",
    "perf_optimizer_sync_mean_s": "s",
    "perf_tokens_total": "tokens",
    "perf_tokens_per_second": "tokens/s",
    "perf_tokens_per_second_per_device": "tokens/s/device",
    "perf_samples_per_second": "samples/s",
    "perf_model_tflops_per_second": "TFLOP/s",
    "perf_model_tflops_per_second_per_device": "TFLOP/s/device",
    "perf_device_peak_tflops": "TFLOP/s",
    "perf_world_size": "ranks",
    "perf_peak_memory_allocated_bytes": "bytes",
    "perf_peak_memory_reserved_bytes": "bytes",
    "perf_device_memory_total_bytes": "bytes",
}
