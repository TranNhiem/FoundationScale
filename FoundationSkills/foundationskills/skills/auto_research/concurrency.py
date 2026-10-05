"""In-flight run slots and the confirm run-slot reserve (AR-LN-008, AR-LN-009).

Doctrine: job state is measured fail-closed. A submitted run holds a slot until
the ledger settles it -- a ``trial_result`` for its ``(trial, seed)`` or its
``job_id`` among ``cancelled_ids`` -- so PENDING jobs count as in-flight (jobs
pend for hours here). An injected ``state_fn`` (e.g. the ``squeue`` connector)
refines live state only: "pending"/"running" hold a slot, "terminal" settles,
and "unknown"/None (or any unusable answer) holds a slot, names
``state_unmeasured:<id>`` and leaves the count ``None`` (never 0) while
unmeasured.

Reserve (amendment A3: run slots ONLY): ``ceil(reserve_frac * max_runs)`` of the
run slots are held back for ``confirm``-phase repeats; a non-``confirm`` submit
that would touch them is refused ``reserve_locked_for_confirm:runs``
(AR-LN-009). The GPU-hours reserve stays the role-based AR-LN-002 gate in
``campaign.check_launch`` -- no ``:gpu_hours`` variant lives here.
"""
from __future__ import annotations

import math
from typing import Any, Callable

StateFn = Callable[[str], "str | None"]

DEFAULT_RESERVE_FRAC = 0.3


def _positive_int(value: Any) -> int | None:
    """A valid ``int >= 1`` (bools excluded); else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _dict(value: Any) -> dict:
    """A mapping as-is; a non-dict spec/cluster/budget is read as ``{}`` (never a lookup crash)."""
    return value if isinstance(value, dict) else {}


def resolve_max_in_flight(spec: Any) -> int:
    """``cluster.max_in_flight`` when a valid int >= 1, else ``int(cluster.max_nodes)``."""
    cluster = _dict(_dict(spec).get("cluster"))
    cap = _positive_int(cluster.get("max_in_flight"))
    if cap is not None:
        return cap
    nodes = cluster.get("max_nodes")
    if isinstance(nodes, bool) or not isinstance(nodes, (int, float)):
        raise ValueError("cluster.max_nodes is not a run count: " + repr(nodes))
    return int(nodes)


def _payload(item: Any) -> dict:
    """One payload dict: unwraps full ledger entries (``op`` + ``payload``)."""
    if not isinstance(item, dict):
        return {}
    nested = item.get("payload")
    if isinstance(nested, dict) and "op" in item:
        return nested
    return item


def _pair(payload: Any) -> tuple[str, str] | None:
    """``(trial, seed)`` run identity; ``None`` when either side is missing."""
    if not isinstance(payload, dict):
        return None
    trial = payload.get("trial")
    seed = payload.get("seed")
    if trial is None or seed is None:
        return None
    return (str(trial).strip(), str(seed).strip())


def _pairs(items: Any) -> set[tuple[str, str]]:
    """Run identities of a results collection (payload list or ``(trial, seed)`` map)."""
    if items is None:
        return set()
    if isinstance(items, dict):
        if "trial" in items or "seed" in items or "job_id" in items:
            items = [items]
        else:
            merged: list[dict] = []
            for key, value in items.items():
                payload = _payload(value)
                if _pair(payload) is None and isinstance(key, tuple) and len(key) == 2:
                    payload = {"trial": key[0], "seed": key[1], **payload}
                merged.append(payload)
            items = merged
    return {pair for pair in (_pair(_payload(item)) for item in items) if pair is not None}


def _cancelled_id_set(cancelled_ids: Any) -> set[str]:
    """Cancelled job ids from plain ids, ``job_cancelled`` payloads, or id lists.

    A scalar id (``str``/``int``) is a one-element collection; a ``bool`` is ignored
    (it is never a job id).
    """
    out: set[str] = set()
    if cancelled_ids is None or isinstance(cancelled_ids, bool):
        return out  # a bool is never a job id
    if isinstance(cancelled_ids, (str, bytes, dict, int)):
        items = [cancelled_ids]
    else:
        try:
            items = list(cancelled_ids)
        except TypeError:
            items = [cancelled_ids]  # an unreadable scalar reads as its one element
    for item in items:
        if isinstance(item, bool):
            continue  # a bool is never a job id
        if isinstance(item, dict):
            if item.get("job_id") is not None:
                out.add(str(item["job_id"]).strip())
            out.update(str(job_id).strip() for job_id in (item.get("job_ids") or []))
        else:
            out.add(str(item).strip())
    return out


def state_index(
    job_entries: Any,
    results: Any = None,
    cancelled_ids: Any = None,
    state_fn: StateFn | None = None,
) -> dict[str, Any]:
    """Classify every submitted run as in-flight or terminal (fail-closed).

    Facts beat probes: a ``trial_result`` for the job's ``(trial, seed)`` or its
    ``job_id`` among ``cancelled_ids`` settles the job. Otherwise
    ``state_fn(job_id)`` refines what is live ("pending"/"running" hold a slot,
    "terminal" settles); an "unknown"/None answer -- and an entry with no settle
    path at all -- holds a slot, names ``state_unmeasured:<id>`` and leaves the
    count ``None``.

    Returns ``{"in_flight", "terminal", "count", "drops"}`` with labels in input
    order (``entry:<index>`` for an entry with no ``job_id``); ``count`` is the
    number of held slots or ``None`` (never 0) while any in-flight candidate is
    unmeasurable and unsettled.
    """
    settled_pairs = _pairs(results)
    cancelled = _cancelled_id_set(cancelled_ids)
    in_flight: list[str] = []
    terminal: list[str] = []
    drops: list[str] = []
    seen: set[str] = set()
    unmeasured = 0
    for index, entry in enumerate(job_entries or []):
        payload = _payload(entry)
        job_id = str(payload.get("job_id", "") or "").strip()
        label = job_id or f"entry:{index}"
        if label in seen:
            drops.append(f"duplicate_job:{label}")
            continue
        seen.add(label)
        pair = _pair(payload)
        has_path = job_id != "" or pair is not None
        settled = (pair is not None and pair in settled_pairs) or (job_id != "" and job_id in cancelled)
        if settled:
            terminal.append(label)
            continue
        probe = state_fn(job_id) if (state_fn is not None and job_id != "") else None
        if probe in ("pending", "running"):
            in_flight.append(label)
            continue
        if probe == "terminal":
            terminal.append(label)
            continue
        # "unknown"/None probe, an unprobeable entry, or the probe-less default
        if state_fn is not None or not has_path:
            drops.append(f"state_unmeasured:{label}")
            in_flight.append(label)
            unmeasured += 1
            continue
        in_flight.append(label)  # ledger-derived: a settle path is a measurable held slot
    return {
        "in_flight": in_flight,
        "terminal": terminal,
        "count": None if unmeasured else len(in_flight),
        "drops": drops,
    }


def concurrency_check(
    spec: Any,
    job_entries: Any,
    results: Any = None,
    cancelled_ids: Any = None,
    state_fn: StateFn | None = None,
) -> list[tuple[str, str]]:
    """AR-LN-008 findings for one prospective submit: ``[]`` while a slot is free.

    Fails closed: ``in_flight_cap:<count>/<cap>`` at or above
    ``cluster.max_in_flight`` (COUNTED in-flight runs hold slots),
    ``in_flight_unmeasured:<n>`` when ``n`` held slots are unmeasured (the count
    is then ``None``, never 0, so an unknown queue NEVER admits).
    """
    cap = resolve_max_in_flight(spec)
    index = state_index(job_entries, results, cancelled_ids, state_fn)
    count = index["count"]
    if count is None:
        unmeasured = sum(1 for drop in index["drops"] if drop.startswith("state_unmeasured:"))
        return [("AR-LN-008", f"in_flight_unmeasured:{unmeasured}")]
    if count >= cap:
        return [("AR-LN-008", f"in_flight_cap:{count}/{cap}")]
    return []


def _reserve_frac(budget: Any) -> float:
    """Share of run slots held back for confirm: spec value when usable, else 0.3."""
    value = _dict(budget).get("reserve_frac", DEFAULT_RESERVE_FRAC)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_RESERVE_FRAC
    frac = float(value)
    if not math.isfinite(frac) or not 0.0 <= frac <= 1.0:
        return DEFAULT_RESERVE_FRAC
    return frac


def _max_runs(budget: Any) -> int:
    """``budget.max_runs`` as a non-negative int run count; else ``ValueError``."""
    value = _dict(budget).get("max_runs")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("budget.max_runs is not a run count: " + repr(value))
    number = int(value)
    if number < 0:
        raise ValueError("budget.max_runs is negative: " + repr(value))
    return number


def reserve_state(spec: Any, launches: Any) -> dict[str, int]:
    """Run-slot reserve: ``{reserved_runs, max_runs, used_runs, free_runs}``.

    ``reserved_runs = ceil(reserve_frac * max_runs)`` (``reserve_frac`` default
    0.3; non-finite or unusable reads 0.3). Every launch row is one used
    ``(trial, seed)`` run. ``free_runs`` is what a non-``confirm`` submit may
    still take (negative once confirm repeats have spent past the reserve); the
    GPU-hours reserve is AR-LN-002's, not this module's (amendment A3).
    """
    budget = _dict(_dict(spec).get("budget"))
    max_runs = _max_runs(budget)
    reserved_runs = math.ceil(_reserve_frac(budget) * max_runs)
    used_runs = len(launches or [])
    return {
        "reserved_runs": reserved_runs,
        "max_runs": max_runs,
        "used_runs": used_runs,
        "free_runs": max_runs - reserved_runs - used_runs,
    }


def reserve_check(spec: Any, phase: Any, launches: Any) -> list[tuple[str, str]]:
    """AR-LN-009: a non-``confirm`` submit may not touch the confirm run reserve.

    One prospective run is charged (``used_runs + 1``) against
    ``max_runs - reserved_runs``; the named refusal is
    ``reserve_locked_for_confirm:runs``. ``confirm`` may spend the reserve;
    GPU-hour reserve breaches are AR-LN-002 findings (no ``:gpu_hours`` variant
    exists here).
    """
    state = reserve_state(spec, launches)
    if phase != "confirm" and state["used_runs"] + 1 > state["max_runs"] - state["reserved_runs"]:
        return [("AR-LN-009", "reserve_locked_for_confirm:runs")]
    return []
