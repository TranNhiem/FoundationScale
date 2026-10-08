"""Music reward: rule-scored MIDI human-likeness for ABC-notation agentic RL.

WHAT IS CLAIMED HERE: the FoundationScale public adapter,
:class:`MusicReward` / :class:`MusicScore`, wrapping the ported stdlib scorer
(:mod:`.core`, :mod:`.feats`, :mod:`.score`, :mod:`.pipeline`, :mod:`.baseline`
-- each carries its own upstream attribution header; see THIRD_PARTY_NOTICES.md)
and the re-exported scorer internals for callers that want them directly.

``MusicReward.score`` follows the FoundationScale None-abstention rule used
throughout this plane (see ``contracts.Trajectory``'s ``reward`` /
``abstention_reason`` biconditional): a MODEL-attributable outcome -- no ABC
in the response, a rejected render, an unscoreable feature extraction -- is a
MEASURED value of ``0.0`` with a reason naming why. An INFRASTRUCTURE outcome
-- the ``abc2midi`` binary is missing, or it failed/timed out for a reason
that is not the policy's fault -- is an ABSTENTION: ``value=None`` with an
``"infra:"``-prefixed reason, never a score. Upstream's ``compute_score``
flattened the infra cases into ``0.0`` too; this adapter does not, and
:mod:`.pipeline`'s module docstring documents the one new exception type
(``Abc2MidiError``) that makes the distinction possible. Any OTHER exception
(a bug in this adapter, or in ``abc2midi``'s own I/O, that is neither of the
above) is left to propagate -- swallowing it behind ``traceback.print_exc()``,
as upstream's ``compute_score`` does for its own callers, would misreport an
unknown failure as a measured score of ``0.0``, which this adapter never does.

WHAT IS NOT CLAIMED: any sandboxing of the ``abc2midi`` subprocess (the
caller names the binary and the timeout; this module runs it bare), and no
behaviour for any reward domain other than music -- the sibling
``foundationscale.agentic_rl.rewards`` package re-exports this adapter only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .baseline import REF_FULL4K
from .core import parse_midi
from .feats import analyze
from .pipeline import Abc2MidiError, Abc2MidiMissing, compute_score, do, extract_abc
from .score import SPEC, W, score

__all__ = (
    "Abc2MidiError",
    "Abc2MidiMissing",
    "MusicReward",
    "MusicRewardRefusal",
    "MusicScore",
    "REF_FULL4K",
    "SPEC",
    "W",
    "analyze",
    "compute_score",
    "do",
    "extract_abc",
    "parse_midi",
    "score",
)


class MusicRewardRefusal(ValueError):
    """A config error constructing :class:`MusicReward` -- not a scoring or infra fault.

    Every message names the failed field and why an invented default would be
    worse than refusing: a reward adapter that guessed a binary path or a
    timeout would hide a misconfiguration behind a result that LOOKS measured.
    """


def _described(value: object) -> str:
    return f"{value!r} (type {type(value).__name__})"


@dataclass(frozen=True)
class MusicScore:
    """The outcome of scoring one response's embedded ABC notation.

    ``value`` and ``reason`` are NOT the strict biconditional
    ``contracts.Trajectory`` uses for ``reward``/``abstention_reason``: a
    model-attributable ``0.0`` carries both a real value AND a reason
    explaining why it is zero (``"no_abc"``, ``"rejected"``,
    ``"unscored:<skip>"``), so a caller reading only ``value`` still gets an
    honest score and a caller reading ``reason`` can tell a zero that was EARNED
    from a zero that was never computed. ``reason`` is ``None`` only when
    ``value`` is a genuine ``clamp(total / 100, 0, 1)`` score. ``value`` is
    ``None`` -- an abstention -- exactly when ``reason`` starts with
    ``"infra:"``: the one case this adapter refuses to report as a score at all.

    ``details`` carries the record :func:`~foundationscale.agentic_rl.rewards.music.pipeline.do`
    returned (feature summary, reject flags, pitch stats, ...) -- empty when
    ``do`` was never reached (no ABC was extracted).
    """

    value: float | None
    reason: str | None
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        where = "MusicScore"
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise MusicRewardRefusal(
                    f"{where}: field 'value' is {_described(self.value)}: a score is None "
                    f"(abstained) or a real finite number -- a bool is never read as a score"
                )
            if not math.isfinite(float(self.value)):
                raise MusicRewardRefusal(
                    f"{where}: field 'value' is {self.value!r}: a score is finite -- nan or "
                    f"infinite is not a measurement"
                )
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise MusicRewardRefusal(
                f"{where}: field 'reason' is {_described(self.reason)}: reason is None (a "
                f"genuine score needs no explanation) or a non-empty str naming why the score "
                f"is zero or why value is absent -- an empty string asserts neither"
            )
        if self.value is None and (self.reason is None or not self.reason.startswith("infra:")):
            raise MusicRewardRefusal(
                f"{where}: field 'value' is None together with field 'reason' "
                f"{self.reason!r}: value is None exactly for an infrastructure abstention, "
                f"whose reason always starts with 'infra:' -- every model-attributable outcome "
                f"(no_abc, rejected, unscored:<skip>) is a MEASURED value, never a None"
            )
        if not isinstance(self.details, Mapping):
            raise MusicRewardRefusal(
                f"{where}: field 'details' is {_described(self.details)}: details is a mapping "
                f"of the record fields do() returned -- never a sequence or scalar"
            )


@dataclass(frozen=True)
class MusicReward:
    """Rule-scored MIDI human-likeness for ABC-notation agentic RL.

    ``abc2midi_bin`` and ``timeout_s`` are declared here and NEVER read from
    the environment: upstream read ``ABC2MIDI_BIN`` at import time, which is
    exactly the kind of implicit, driver-only configuration that silently
    fails to reach a remote worker (the upstream docstring for
    ``Abc2MidiMissing`` names this failure explicitly). A caller that wants
    PATH lookup still names the binary (``MusicReward(abc2midi_bin="abc2midi")``).
    """

    abc2midi_bin: str
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        where = "MusicReward"
        if not isinstance(self.abc2midi_bin, str) or not self.abc2midi_bin:
            raise MusicRewardRefusal(
                f"{where}: field 'abc2midi_bin' is {_described(self.abc2midi_bin)}: the "
                f"abc2midi binary path is a non-empty str -- FoundationScale never reads it "
                f"from the environment, so a caller that means 'use PATH lookup' must still "
                f"name the binary explicitly (e.g. 'abc2midi')"
            )
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)):
            raise MusicRewardRefusal(
                f"{where}: field 'timeout_s' is {_described(self.timeout_s)}: the subprocess "
                f"timeout is a finite positive number of seconds -- a bool is never read as one"
            )
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise MusicRewardRefusal(
                f"{where}: field 'timeout_s' is {self.timeout_s!r}: the subprocess timeout "
                f"must be finite and positive -- zero, negative, nan or infinite would never "
                f"let abc2midi run or never time out"
            )

    def score(self, response: str) -> MusicScore:
        """Score one model response's embedded ABC notation.

        Never raises for a model-attributable failure or a known abc2midi
        infrastructure fault -- both are reported IN the returned
        :class:`MusicScore`. Any other exception propagates: see the module
        docstring for why this adapter does not swallow the unexpected.
        """
        abc = extract_abc(response)
        if abc is None:
            return MusicScore(0.0, "no_abc", {})

        rec: dict[str, Any] = {
            "key": "agentic_rl_music",
            "id": 0,
            "rep": 0,
            "abc": abc,
            "tag": None,
            "lang": None,
            "abc_len": len(abc),
            "nvoice": 0,
            "latency": None,
        }
        try:
            r = do(rec, self.abc2midi_bin, self.timeout_s)
        except Abc2MidiMissing:
            return MusicScore(None, "infra:abc2midi_missing", {})
        except Abc2MidiError as e:
            return MusicScore(None, f"infra:abc2midi_error:{e.cause_name}", {})

        if r.get("skip"):
            return MusicScore(0.0, f"unscored:{r['skip']}", r)
        if r.get("reject"):
            return MusicScore(0.0, "rejected", r)
        if "total" not in r:
            return MusicScore(0.0, f"unscored:{r.get('scorer_skip', 'no_total')}", r)
        value = max(0.0, min(1.0, float(r["total"]) / 100.0))
        return MusicScore(value, None, r)
