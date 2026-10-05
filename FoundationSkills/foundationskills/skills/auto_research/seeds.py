'''Pure seed and phase bookkeeping for auto_research (no I/O).

There is NO trial_spec.phase field (amendment A2): the phase is derived from the M1 role
vocabulary (PHASE_OF_ROLE - a candidate job runs the screening phase) and every job_submitted
payload proves the phase and the seed it ran. A confirm set is complete only when every
requested seed carries measured evidence for the trial AND its reference.
'''
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from foundationskills.skills.auto_research.accept import _point

PHASES = ('baseline', 'screening', 'confirm')
'''Phase vocabulary in order; set_status takes one of these as its phase keyword.'''

PHASE_OF_ROLE: dict[str, str] = {'baseline': 'baseline', 'candidate': 'screening', 'confirm': 'confirm'}
'''A2 binding mapping: the phase derives from the role and is never a trial_spec field.'''

JOB_OP = 'job_submitted'

DEFAULT_BASELINE_REPEATS = 3
DEFAULT_CONFIRM_REPEATS = 3
DEFAULT_SCREENING_REPEATS = 1

_REPEAT_FIELD = {'baseline': 'baseline_repeats', 'screening': 'screening_repeats', 'confirm': 'confirm_repeats'}


# ---- tiny pure helpers ----------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _norm_trial(value: Any) -> str:
    return '' if value is None else str(value)


def _seed_key(value: Any) -> Any:
    '''Canonical seed key: 101, 101.0 and the string 101 name one and the same seed.'''
    if value is None or isinstance(value, bool):
        return '' if value is None else str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text and text.lstrip('+-').isdigit():
        try:
            return int(text)
        except ValueError:
            pass
    return text


def _payload(row: Any) -> dict[str, Any]:
    '''The payload of a ledger entry ({op, payload}) - or the row itself when it already is one.'''
    if not isinstance(row, Mapping):
        return {}
    inner = row.get('payload')
    if isinstance(inner, Mapping) and any(key in row for key in ('op', 'seq', 'hash', 'prev')):
        return dict(inner)
    return dict(row)


def _job_rows(job_entries: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    '''The job_submitted payloads of a bare/ledger-shaped entry stream (other ops are ignored).'''
    rows: list[dict[str, Any]] = []
    for entry in (job_entries or ()):
        if not isinstance(entry, Mapping):  # junk entries are ignored, never a job record
            continue
        outer = entry
        if str(outer.get('op') or '') not in ('', JOB_OP):
            continue
        row = _payload(entry)
        if str(row.get('op') or '') not in ('', JOB_OP):
            continue
        rows.append(row)
    return rows


def _as_phase(value: Any) -> str:
    '''The phase named by a role or a phase spelling (candidate -> screening); empty when unknown.'''
    name = str(value or '')
    phase = PHASE_OF_ROLE.get(name)
    return phase if phase is not None else (name if name in PHASES else '')


def _resolve_phase(trial_spec: Any) -> tuple[str | None, str]:
    '''(derived phase, raw role name) of a trial spec (A2): the role is the only phase source.'''
    spec = _dict(trial_spec)
    name = str(spec.get('role') or '')
    return phase_of(name), name


def _trial_seeds(trial_spec: Any) -> list[Any]:
    '''The seeds a trial spec asks for (a seeds list, else its single seed).'''
    spec = _dict(trial_spec)
    raw = spec.get('seeds')
    if isinstance(raw, (list, tuple)):
        return [seed for seed in raw if seed is not None]
    seed = spec.get('seed')
    return [] if seed is None else [seed]


def _numeric(value: Any) -> bool:
    '''True for a finite int/float number (bools, strings, NaN and +-Inf never measure anything).'''
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _has_evidence(row: Mapping[str, Any], metric: str, fingerprint: str) -> bool:
    '''True when the row measures the objective metric under the expected eval_policy fingerprint.

    Malformed input is never a measurement and never a crash: a truthy non-Mapping metrics block, a
    non-Mapping metric point or a non-finite/non-numeric metric value means 'not measured'.
    '''
    if row.get('status') != 'ok' or row.get('limited'):
        return False
    if str(row.get('eval_policy_fingerprint') or '') != fingerprint:
        return False
    metrics = row.get('metrics')
    if isinstance(metrics, Mapping):
        point = metrics.get(metric)
        if not isinstance(point, Mapping) or not _numeric(point.get('value')):
            return False
        if 'se' in point and not _numeric(point.get('se')):
            return False
    elif metrics:
        return False
    try:
        return _point(dict(row), metric) is not None
    except Exception:  # noqa: BLE001 - a malformed shape is 'not measured', never a crash
        return False


# ---- phase bookkeeping (A2) -----------------------------------------------

def phase_of(role: Any) -> str | None:
    '''The phase derived from a role (A2): candidate runs the screening phase, else None.'''
    return PHASE_OF_ROLE.get(str(role or ''))


def trial_phase(trial_spec: Any) -> str | None:
    '''The derived phase of a trial spec (None when its role names no phase; A2 role-only).'''
    return _resolve_phase(trial_spec)[0]


# ---- seed plan --------------------------------------------------------------

def seed_plan(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    '''The requested seed plan: the seed list plus the repeat counts per phase (junk-safe defaults).'''
    seeds = _dict(_dict(spec).get('seeds'))
    raw = seeds.get('seed_list')
    items = list(raw) if isinstance(raw, (list, tuple)) else []
    seed_list: list[int] = []
    for item in items:
        if _is_int(item) and item not in seed_list:
            seed_list.append(item)
    plan: dict[str, Any] = {'seed_list': seed_list}
    for field, default in (
        ('baseline_repeats', DEFAULT_BASELINE_REPEATS),
        ('confirm_repeats', DEFAULT_CONFIRM_REPEATS),
        ('screening_repeats', DEFAULT_SCREENING_REPEATS),
    ):
        value = seeds.get(field)
        plan[field] = int(value) if _is_int(value) else default
    return plan


def phase_problems(trial_spec: Mapping[str, Any] | None, spec: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    '''AR-LN-001 problems (A2): an unknown/missing derived phase, or confirm seeds outside seed_list.'''
    phase, name = _resolve_phase(trial_spec)
    if phase is None:
        return [('AR-LN-001', f'phase_unknown:{name}')]
    if phase != 'confirm':
        return []
    known = {_seed_key(seed) for seed in seed_plan(spec)['seed_list']}
    return [
        ('AR-LN-001', f'seed_not_in_seed_list:{seed}')
        for seed in _trial_seeds(trial_spec)
        if _seed_key(seed) not in known
    ]


# ---- job provenance -------------------------------------------------------

class PhaseIndex(dict):
    '''(trial, seed) -> phase map with type-tolerant keys (101 and the string 101 are one seed).'''

    def _lookup(self, key: Any) -> Any:
        try:
            value = dict.get(self, key)
        except TypeError:
            return None
        if value is not None:
            return value
        if isinstance(key, tuple) and len(key) == 2:
            return dict.get(self, (_norm_trial(key[0]), _seed_key(key[1])))
        return None

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        value = self._lookup(key)
        return default if value is None else value

    def __contains__(self, key: Any) -> bool:  # type: ignore[override]
        return self._lookup(key) is not None

    def __getitem__(self, key: Any) -> Any:
        value = self._lookup(key)
        if value is None:
            raise KeyError(key)
        return value


def job_phase_index(job_entries: Iterable[Mapping[str, Any]] | None) -> PhaseIndex:
    '''(trial, seed) -> phase provenance from job_submitted payloads that ledgered both fields.'''
    index = PhaseIndex()
    for row in _job_rows(job_entries):
        trial, seed, phase = row.get('trial'), row.get('seed'), row.get('phase')
        if trial is None or seed is None or phase is None or str(phase) == '':
            continue
        key = (_norm_trial(trial), _seed_key(seed))
        if key not in index:  # first entry wins: provenance is never rewritten
            index[key] = str(phase)
    return index


# ---- confirm-set status ---------------------------------------------------

def set_status(
    spec: Mapping[str, Any] | None,
    trial: str,
    results: Iterable[Mapping[str, Any]] | None,
    job_entries: Iterable[Mapping[str, Any]] | None,
    reference_rows: Iterable[Mapping[str, Any]] | None,
    *,
    phase: str,
) -> dict[str, Any]:
    '''Requested/measured/paired seeds of one phase plus the named drops that explain every gap.

    measured means the trial result is ok, not limited, carries spec.eval_policy.fingerprint and has
    a usable accept._point for the objective metric; a pair needs that evidence for the trial AND its
    reference (baseline repeats pair with nothing). The requested set is seed_list[:confirm_repeats]
    for the confirm phase, [:baseline_repeats] for baseline, [:screening_repeats] for screening.
    Drops are sorted, de-duplicated and rendered as <drop>:<trial>:<seed>:

    seed_missing (no job, no result) | seed_pending (job, no result) | seed_crashed | seed_limited |
    seed_unpaired (the pair is not measured - also a result without usable evidence) |
    phase_unprovenanced (a measured seed without a job_submitted entry proving this phase value;
    demanded whenever the trial ledgered jobs and always for the confirm phase).
    '''
    tname = str(trial)
    spec_d = _dict(spec)
    metric = str(_dict(spec_d.get('objective')).get('metric') or '')
    fingerprint = str(_dict(spec_d.get('eval_policy')).get('fingerprint') or '')
    plan = seed_plan(spec_d)

    ph = _as_phase(phase)
    repeats = int(plan[_REPEAT_FIELD.get(ph, 'confirm_repeats')])
    pair_reference = ph != 'baseline'

    requested: list[Any] = []
    for seed in plan['seed_list']:
        if seed not in requested:
            requested.append(seed)
    requested = requested[: max(0, repeats)]

    def _evidence(row: Mapping[str, Any]) -> bool:
        return _has_evidence(row, metric, fingerprint)

    def _rank(row: Mapping[str, Any]) -> int:
        if _evidence(row):
            return 3
        if row.get('status') == 'crash':
            return 0
        return 1 if row.get('limited') else 2

    def _best(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        out: dict[Any, dict[str, Any]] = {}
        ranks: dict[Any, int] = {}
        for row in rows:
            key = _seed_key(row.get('seed'))
            if key not in out or _rank(row) > ranks[key]:
                out[key], ranks[key] = row, _rank(row)
        return out

    mine = [
        row
        for row in (_payload(item) for item in (results or ()))
        if row.get('seed') is not None and ('trial' not in row or _norm_trial(row.get('trial')) == tname)
    ]
    refs = [
        row
        for row in (_payload(item) for item in (reference_rows or ()))
        if row.get('seed') is not None and _norm_trial(row.get('trial')) != tname
    ]
    best_mine, best_ref = _best(mine), _best(refs)

    mine_jobs = [row for row in _job_rows(job_entries) if _norm_trial(row.get('trial')) == tname]
    job_pairs = {
        (_norm_trial(row.get('trial')), _seed_key(row.get('seed')))
        for row in mine_jobs
        if row.get('seed') is not None
    }
    provenance = job_phase_index(job_entries)

    drops: list[str] = []
    measured: list[Any] = []
    paired: list[Any] = []
    for seed in requested:
        key = _seed_key(seed)
        tag = f'{tname}:{seed}'
        row = best_mine.get(key)
        if row is None:
            name = 'seed_pending' if (tname, key) in job_pairs else 'seed_missing'
            drops.append(f'{name}:{tag}')
            continue
        if row.get('status') == 'crash':
            drops.append(f'seed_crashed:{tag}')
            continue
        if row.get('limited'):
            drops.append(f'seed_limited:{tag}')
            continue
        if not _evidence(row):
            drops.append(f'seed_unpaired:{tag}')
            continue
        measured.append(seed)
        ref = best_ref.get(key)
        if not pair_reference:
            paired.append(seed)
        elif ref is not None and _evidence(ref):
            paired.append(seed)
        else:
            drops.append(f'seed_unpaired:{tag}')

    # phase provenance (findings 1/3): a measured (trial, seed) counts toward phase <ph> only when a
    # job_submitted entry for it ledgered that very phase value. The confirm phase always demands it;
    # baseline and screening only when the trial ledgered any job (M0 ledgers have none).
    if ph == 'confirm' or mine_jobs:
        for seed in measured:
            if provenance.get((tname, _seed_key(seed))) != ph:
                drops.append(f'phase_unprovenanced:{tname}:{seed}')

    deduped = sorted(set(drops))
    complete = bool(requested) and not deduped and paired == requested
    return {
        'requested': requested,
        'measured': measured,
        'paired': paired,
        'complete': complete,
        'drops': deduped,
    }
