# SPDX-License-Identifier: Apache-2.0
"""Controls for #496: an UNMEASURED RL step must never leave silently.

`_one_step` returns None from four places, each meaning "this step produced no
gradient". Two of them announced themselves on stderr and two did not, so a run
in which every step abstained returned an empty list, printed nothing, and was
indistinguishable from a run that trained. That is the failure this framework
exists to refuse, so the property is pinned structurally rather than by
exercising one instance of it: a fifth silent path added later must fail here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from foundationscale.rl import trainer as trainer_module
from foundationscale.rl.trainer import TrainerRefusal, _refuse_vacuous_run


def _silent_unmeasured_exits(source: str, function: str) -> tuple[int, list[int]]:
    """Return (total bare `return None`, line numbers of the SILENT ones).

    A bare `return None` is announced when the statement immediately preceding
    it in the same suite is a call to `print`. Anything else is silent.
    """
    total = 0
    silent: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for parent in ast.walk(node):
            for attribute in ("body", "orelse", "finalbody"):
                suite = getattr(parent, attribute, None)
                if not isinstance(suite, list):
                    continue
                for index, statement in enumerate(suite):
                    if not isinstance(statement, ast.Return):
                        continue
                    value = statement.value
                    if not (isinstance(value, ast.Constant) and value.value is None):
                        continue
                    total += 1
                    previous = suite[index - 1] if index else None
                    announced = (
                        isinstance(previous, ast.Expr)
                        and isinstance(previous.value, ast.Call)
                        and getattr(previous.value.func, "id", "") == "print"
                    )
                    if not announced:
                        silent.append(statement.lineno)
    return total, silent


def _trainer_source() -> str:
    return Path(inspect.getfile(trainer_module)).read_text(encoding="utf-8")


def test_every_unmeasured_exit_from_one_step_is_announced() -> None:
    total, silent = _silent_unmeasured_exits(_trainer_source(), "_one_step")
    # The count is asserted so this cannot pass by matching nothing: if the
    # function is renamed or its early exits are restructured away, the control
    # fails rather than reporting a clean zero over an empty set.
    assert total >= 4, f"expected at least 4 bare `return None` in _one_step, found {total}"
    assert silent == [], f"UNMEASURED steps leave silently at line(s) {silent}"


def test_the_silence_detector_fires_on_a_planted_silent_exit() -> None:
    """MUST-FIRE: the matcher above must reject a suite it should reject."""
    planted = (
        "def _one_step(self):\n"
        "    if not rows:\n"
        "        print('loud', file=sys.stderr)\n"
        "        return None\n"
        "    if not kept:\n"
        "        return None\n"
        "    return report\n"
    )
    total, silent = _silent_unmeasured_exits(planted, "_one_step")
    assert total == 2
    assert silent == [6], f"detector missed the planted silent exit: {silent}"


def test_a_run_that_lands_no_step_is_refused_as_vacuous() -> None:
    with pytest.raises(TrainerRefusal) as excinfo:
        _refuse_vacuous_run(attempted=20, measured=0)
    message = str(excinfo.value)
    assert "20 step(s) attempted" in message
    assert "vacuous" in message


def test_a_run_that_lands_one_step_is_not_refused() -> None:
    # A partially measured run is reported, not refused: `used < offered` is
    # the estimator's own account of what it dropped, and dropping some rows is
    # a measurement, not the absence of one.
    _refuse_vacuous_run(attempted=20, measured=1)


def test_the_guard_is_called_after_the_step_loop_not_before_it() -> None:
    source = _trainer_source()
    run = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    loops = [node for node in ast.walk(run) if isinstance(node, ast.For)]
    calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_refuse_vacuous_run"
    ]
    assert calls, "run() never consults the vacuous-run guard, so the helper is dead code"
    assert loops, "run() no longer contains a step loop; this control is out of date"
    last_loop_end = max(loop.end_lineno or loop.lineno for loop in loops)
    assert min(call.lineno for call in calls) > last_loop_end, (
        "the vacuous-run guard runs before the step loop finishes, so it cannot "
        "see how many steps were actually measured"
    )
