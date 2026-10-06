"""Test-only: review regressions for claim linking, chain tolerance and concurrency/queue gates (pure, CPU-only)."""
from __future__ import annotations

from typing import Any, Callable

from foundationskills.interfaces.fs import squeue
from foundationskills.skills.auto_research import claims, concurrency

SEEDS = [101, 102, 103]


def _decision() -> dict[str, Any]:
    return {"verdict": "accepted_gain", "mean_delta": 0.5, "tau": 0.1, "n_pairs": 3}


def _bare(trial: str, reference: str) -> dict[str, Any]:
    return claims.claim_body(trial, reference, None, SEEDS, _decision(), "fp1")


def _claim(trial: str, reference: str, prev: str | None) -> dict[str, Any]:
    body = claims.claim_body(trial, reference, prev, SEEDS, _decision(), "fp1")
    return {**body, "claim_id": claims.claim_id(body)}


def _result(trial: str, seed: int, role: str = "candidate") -> dict[str, Any]:
    return {
        "trial": trial,
        "role": role,
        "seed": seed,
        "status": "ok",
        "limited": False,
        "metrics": {"reward": {"value": 1.0, "se": 0.1}},
    }


def _evidence() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for trial, role in (("baseline", "baseline"), ("t1", "confirm"), ("t2", "confirm")):
        for seed in SEEDS:
            out[(trial, seed)] = _result(trial, seed, role=role)
    return out


def _body_of(entry: dict[str, Any]) -> dict[str, Any]:
    """The hashed body behind a produced entry (everything but ``claim_id``)."""
    return {key: value for key, value in entry.items() if key != "claim_id"}


def _outcome(fn: Callable[..., Any], *args: Any) -> tuple[str, Any]:
    """What one call does (the result, or the exception class name) -- the behavioural measurement."""
    try:
        return ("returns", fn(*args))
    except Exception as exc:
        return ("raises", type(exc).__name__)


class _Reply:
    """One fake runner reply: rc 0 over fixed stdout (a successful empty query drops finished jobs)."""

    def __init__(self, stdout: str) -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


class _Runner:
    """Fake squeue runner recording every command it is given."""

    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout
        self.calls: list[Any] = []

    def __call__(self, command: Any, capture_output: bool = False, text: bool = False, timeout: float = 0) -> _Reply:
        self.calls.append(command)
        return _Reply(self.stdout)


def test_claim_entries_hash_the_final_linked_body() -> None:
    bare1, bare2 = _bare("t1", "baseline"), _bare("t2", "t1")
    entries = claims.claim_entries([bare2, bare1])
    assert [entry["prev"] for entry in entries] == ["baseline", entries[0]["claim_id"]]
    for entry in entries:
        assert entry["claim_id"] == claims.claim_id(_body_of(entry))  # the id covers the FINAL body
    assert entries[0]["claim_id"] != claims.claim_id(bare1)  # not the id of the unlinked body
    assert entries[1]["claim_id"] != claims.claim_id(bare2)


def test_claim_entries_servicing_prev_first_rederives_a_stale_id() -> None:
    stale = _bare("t1", "baseline")
    body2 = _bare("t2", "t1")
    entries = claims.claim_entries([{**stale, "claim_id": claims.claim_id(stale)}, body2])
    for entry in entries:
        assert entry["claim_id"] == claims.claim_id(_body_of(entry))
    assert entries[0]["claim_id"] != claims.claim_id(stale)  # the link is serviced FIRST
    assert entries[1]["prev"] == entries[0]["claim_id"]


def test_derive_chain_skips_non_dict_claims() -> None:
    c1 = _claim("t1", "baseline", "baseline")
    c2 = _claim("t2", "t1", c1["claim_id"])
    chain, drops = claims.derive_chain([c1, "junk", 42, None, c2], _evidence())
    assert [(row["claim_id"], row["status"]) for row in chain] == [
        (c1["claim_id"], "accepted"),
        (c2["claim_id"], "accepted"),
    ]
    assert drops == []


def test_champion_and_reference_rows_skip_non_dict_rows() -> None:
    rows = [_result("baseline", 101, role="baseline"), _result("t1", 101), "junk", 5, None]
    accepted = {"claim_id": "c1", "trial": "t1", "status": "accepted"}
    assert claims.champion([accepted, None, "junk"]) == {"trial": "t1", "claim_id": "c1"}
    assert claims.champion([None, "junk"]) == {"trial": None, "claim_id": "baseline"}
    assert claims.reference_rows([None, "junk"], rows) == [rows[0]]
    assert claims.reference_rows([accepted], rows) == [rows[1]]


def test_state_index_scalar_cancelled_ids_read_as_one_element() -> None:
    entries = [
        {"job_id": "39", "trial": "t1", "seed": 101},
        {"job_id": "40", "trial": "t1", "seed": 102},
        {"job_id": "41", "trial": "t1", "seed": 103},
    ]
    for cancelled in (39, "39", ["39"]):
        index = concurrency.state_index(entries, None, cancelled)
        assert index["terminal"] == ["39"]
        assert index["in_flight"] == ["40", "41"]
        assert index["count"] == 2
        assert index["drops"] == []


def test_state_index_ignores_bools_as_cancelled_job_ids() -> None:
    entries = [
        {"job_id": "39", "trial": "t1", "seed": 101},
        {"job_id": "40", "trial": "t1", "seed": 102},
    ]
    for cancelled in (True, False, [True], (True, False)):
        index = concurrency.state_index(entries, None, cancelled)  # a bool is never a job id
        assert index["terminal"] == []
        assert index["in_flight"] == ["39", "40"]
    mixed = concurrency.state_index(entries, None, [True, "40"])
    assert mixed["terminal"] == ["40"]  # only the real job id is cancelled
    assert mixed["in_flight"] == ["39"]


def test_non_dict_spec_cluster_budget_read_as_empty_dicts() -> None:
    launches = [{"trial": "t1", "seed": 101}]
    assert _outcome(concurrency.resolve_max_in_flight, "junk") == _outcome(concurrency.resolve_max_in_flight, {})
    assert _outcome(concurrency.resolve_max_in_flight, {"cluster": 7}) == _outcome(
        concurrency.resolve_max_in_flight, {"cluster": {}}
    )
    assert _outcome(concurrency.reserve_state, "junk", launches) == _outcome(concurrency.reserve_state, {}, launches)
    assert _outcome(concurrency.reserve_state, {"budget": 7}, launches) == _outcome(
        concurrency.reserve_state, {"budget": {}}, launches
    )
    assert _outcome(concurrency.reserve_check, "junk", "confirm", launches) == _outcome(
        concurrency.reserve_check, {}, "confirm", launches
    )
    assert _outcome(concurrency.reserve_check, {"budget": "junk"}, "baseline", launches) == _outcome(
        concurrency.reserve_check, {"budget": {}}, "baseline", launches
    )
    assert _outcome(concurrency.concurrency_check, "junk", []) == _outcome(concurrency.concurrency_check, {}, [])


def test_job_states_treats_scalar_int_ids_and_keeps_terminal_when_absent() -> None:
    runner = _Runner("")  # a successful empty query: squeue drops finished jobs
    assert squeue.job_states(12345, runner=runner) == {"12345": "terminal"}  # KEEP: absent reads 'terminal'
    assert "12345" in " ".join(runner.calls[0])  # the scalar is queried as [str(id)]
    assert squeue.job_states("12345", runner=_Runner("")) == {"12345": "terminal"}
    assert squeue.job_states(0, runner=_Runner("")) == {"0": "terminal"}


def test_job_states_never_maps_a_bool_and_never_runs_the_runner() -> None:
    runner = _Runner("")
    assert squeue.job_states(True, runner=runner) == {}  # maps nothing
    assert squeue.job_states([False], runner=runner) == {}
    assert runner.calls == []  # never reaches the runner
    mixed = squeue.job_states([True, 12345], runner=_Runner(""))
    assert set(mixed) == {"12345"}
    assert mixed["12345"] == "terminal"


def test_job_states_documents_the_deliberate_terminal_rule() -> None:
    assert "deliberate" in (squeue.job_states.__doc__ or "").lower()
