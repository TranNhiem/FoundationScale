"""Regressions found by the M1 live smoke (2026-10-05): emitter string fields and the eval interpreter."""
from __future__ import annotations

import sys

from foundationskills.interfaces.fs.emit_trial import emit_trial

SBATCH = "#!/bin/bash\n#SBATCH --time=10-00:00:00\n"


def _trial(**eval_overrides):
    return {
        "trial": "t-live", "role": "baseline", "kind": "eval_only", "nodes": 1, "gpus_per_node": 1,
        "eval_request": {"benchmarks": ["arc_easy"], **eval_overrides},
    }


def _emitter(result, calls):
    def fake(request, **kwargs):
        calls.append((dict(request), kwargs))
        return dict(result)
    return fake


def test_joined_missing_string_is_split_into_items_not_characters():
    calls: list = []
    spec = {"executable": False, "missing": "EV-IN-004: missing input: lm_eval is not installed; EV-IN-001: x",
            "sbatch": SBATCH}
    fact = emit_trial({"trial_spec": _trial()}, emit_eval_fn=_emitter(spec, calls))
    assert fact["missing"] == ["EV-IN-004: missing input: lm_eval is not installed", "EV-IN-001: x"]
    assert fact["executable"] is False


def test_string_drops_and_notes_are_not_exploded():
    calls: list = []
    spec = {"executable": True, "drops": "dropped_device", "notes": "one note", "sbatch": SBATCH}
    fact = emit_trial({"trial_spec": _trial(python="/env/bin/python")}, emit_eval_fn=_emitter(spec, calls))
    assert fact["drops"] == ["dropped_device"]
    assert fact["notes"] == ["one note"]


def test_unnamed_interpreter_defaults_to_this_process_and_is_noted():
    calls: list = []
    fact = emit_trial({"trial_spec": _trial()}, emit_eval_fn=_emitter({"executable": True, "sbatch": SBATCH}, calls))
    request, kwargs = calls[0]
    assert kwargs["python"] == sys.executable
    assert "python" not in request
    assert f"python defaulted to {sys.executable}" in fact["notes"]


def test_named_interpreter_is_forwarded_and_not_passed_as_a_request_key():
    calls: list = []
    fact = emit_trial(
        {"trial_spec": _trial(python="/env/bin/python")},
        emit_eval_fn=_emitter({"executable": True, "sbatch": SBATCH}, calls),
    )
    request, kwargs = calls[0]
    assert kwargs["python"] == "/env/bin/python"
    assert "python" not in request
    assert not any(n.startswith("python defaulted") for n in fact["notes"])
