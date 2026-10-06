'''Unit tests for the auto_research seed/phase bookkeeping (pure, CPU-only).'''
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / 'foundationskills').is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from foundationskills.skills.auto_research.seeds import (  # noqa: E402
    PHASES,
    PHASE_OF_ROLE,
    job_phase_index,
    phase_of,
    phase_problems,
    seed_plan,
    set_status,
    trial_phase,
)

FP = 'sha256:' + 'a' * 64
OTHER_FP = 'sha256:' + 'b' * 64
SEEDS = [101, 102, 103]


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        'objective': {'metric': 'score', 'direction': 'max'},
        'eval_policy': {'fingerprint': FP, 'metrics': ['score']},
        'seeds': {'seed_list': list(SEEDS), 'baseline_repeats': 3, 'confirm_repeats': 3},
    }
    spec.update(overrides)
    return spec


def _result(trial: str, seed: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        'trial': trial,
        'role': 'confirm',
        'seed': seed,
        'status': 'ok',
        'limited': False,
        'eval_policy_fingerprint': FP,
        'metrics': {'score': {'value': 1.0, 'se': 0.1}},
    }
    row.update(overrides)
    return row


def _job(job_id: int, trial: str, seed: int, phase: str | None = 'confirm') -> dict[str, Any]:
    row: dict[str, Any] = {'job_id': job_id, 'trial': trial, 'seed': seed}
    if phase is not None:
        row['phase'] = phase
    return row


def _jobs_for(trial: str, seeds: list[int], phase: str | None = 'confirm') -> list[dict[str, Any]]:
    return [_job(index, trial, seed, phase) for index, seed in enumerate(seeds, start=100)]


# ---- phase derivation (A2) -------------------------------------------------

def test_phase_of_maps_the_role_vocabulary() -> None:
    assert PHASES == ('baseline', 'screening', 'confirm')
    assert PHASE_OF_ROLE == {'baseline': 'baseline', 'candidate': 'screening', 'confirm': 'confirm'}
    assert phase_of('baseline') == 'baseline'
    assert phase_of('candidate') == 'screening'
    assert phase_of('confirm') == 'confirm'
    assert phase_of('screening') is None  # a phase is not a role
    assert phase_of(None) is None
    assert phase_of('') is None


def test_trial_phase_is_role_derived() -> None:
    assert trial_phase({'role': 'candidate'}) == 'screening'
    assert trial_phase({'role': 'confirm'}) == 'confirm'
    assert trial_phase({'role': 'baseline'}) == 'baseline'
    assert trial_phase({'phase': 'confirm'}) is None  # never a trial_spec field (A2: role only)
    assert trial_phase({'role': 'candidate', 'phase': 'confirm'}) == 'screening'  # the role is authoritative
    assert trial_phase({}) is None
    assert trial_phase({'role': 'bogus'}) is None


def test_phase_problems_flags_an_unknown_or_missing_phase() -> None:
    spec = _spec()
    assert phase_problems({'role': 'bogus'}, spec) == [('AR-LN-001', 'phase_unknown:bogus')]
    assert phase_problems({'role': 'bogus', 'seed': 101}, spec) == [('AR-LN-001', 'phase_unknown:bogus')]
    assert phase_problems({}, spec) == [('AR-LN-001', 'phase_unknown:')]
    assert phase_problems({'role': 'candidate'}, spec) == []
    assert phase_problems({'role': 'confirm', 'seed': 103}, spec) == []


def test_phase_problems_checks_only_confirm_seeds() -> None:
    spec = _spec()
    assert phase_problems({'role': 'candidate', 'seed': 999}, spec) == []
    assert phase_problems({'role': 'baseline', 'seed': 999}, spec) == []
    assert phase_problems({'role': 'confirm', 'seed': 999}, spec) == [('AR-LN-001', 'seed_not_in_seed_list:999')]
    assert phase_problems({'role': 'confirm', 'seeds': [101, 999, 1000]}, spec) == [
        ('AR-LN-001', 'seed_not_in_seed_list:999'),
        ('AR-LN-001', 'seed_not_in_seed_list:1000'),
    ]


# ---- seed plan -------------------------------------------------------------

def test_seed_plan_defaults() -> None:
    assert seed_plan(_spec()) == {
        'seed_list': [101, 102, 103],
        'baseline_repeats': 3,
        'confirm_repeats': 3,
        'screening_repeats': 1,
    }
    assert seed_plan({}) == {
        'seed_list': [],
        'baseline_repeats': 3,
        'confirm_repeats': 3,
        'screening_repeats': 1,
    }


def test_seed_plan_reads_explicit_repeats_and_drops_junk() -> None:
    spec = _spec(
        seeds={
            'seed_list': [101, 101, 'x', 103],
            'baseline_repeats': 2,
            'confirm_repeats': 1,
            'screening_repeats': 3,
        }
    )
    assert seed_plan(spec) == {
        'seed_list': [101, 103],
        'baseline_repeats': 2,
        'confirm_repeats': 1,
        'screening_repeats': 3,
    }


# ---- job_phase_index -------------------------------------------------------

def test_job_phase_index_reads_job_submitted_payloads() -> None:
    entries = [
        {'op': 'job_submitted', 'payload': _job(111, 't1', 101, 'confirm')},
        _job(112, 't1', 102, 'screening'),
        _job(113, 't1', 103, None),  # a job without a ledgered phase proves nothing
        {'op': 'trial_result', 'payload': _result('t1', 101)},
    ]
    index = job_phase_index(entries)
    assert index == {('t1', 101): 'confirm', ('t1', 102): 'screening'}


def test_job_phase_index_is_seed_type_tolerant() -> None:
    index = job_phase_index([_job(111, 't1', 101, 'confirm')])
    assert index.get(('t1', 101)) == 'confirm'
    assert index.get(('t1', '101')) == 'confirm'
    assert ('t1', 101) in index
    assert index.get(('t2', 101)) is None


def test_job_phase_index_first_entry_wins() -> None:
    index = job_phase_index([_job(1, 't1', 101, 'confirm'), _job(2, 't1', 101, 'screening')])
    assert index == {('t1', 101): 'confirm'}


# ---- set_status ------------------------------------------------------------

def test_set_status_is_complete_when_every_seed_is_measured_and_paired() -> None:
    jobs = _jobs_for('t1', SEEDS)
    results = [_result('t1', seed) for seed in SEEDS]
    refs = [_result('ref', seed) for seed in SEEDS]
    status = set_status(_spec(), 't1', results, jobs, refs, phase='confirm')
    assert status == {
        'requested': [101, 102, 103],
        'measured': [101, 102, 103],
        'paired': [101, 102, 103],
        'complete': True,
        'drops': [],
    }


def test_set_status_is_incomplete_when_the_reference_seed_is_missing() -> None:
    jobs = _jobs_for('t1', SEEDS)
    results = [_result('t1', seed) for seed in SEEDS]
    refs = [_result('ref', seed) for seed in (101, 102)]
    status = set_status(_spec(), 't1', results, jobs, refs, phase='confirm')
    assert status['complete'] is False
    assert status['measured'] == [101, 102, 103]
    assert status['paired'] == [101, 102]
    assert status['drops'] == ['seed_unpaired:t1:103']


def test_set_status_slices_the_requested_set_to_the_phase_repeats() -> None:
    spec = _spec(seeds={'seed_list': list(SEEDS), 'baseline_repeats': 3, 'confirm_repeats': 2})
    status = set_status(
        spec,
        't1',
        [_result('t1', seed) for seed in SEEDS],
        _jobs_for('t1', SEEDS),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['requested'] == [101, 102]
    assert status['complete'] is True


def test_set_status_baseline_repeats_pair_with_nothing() -> None:
    status = set_status(
        _spec(),
        'base',
        [_result('base', seed) for seed in SEEDS],
        _jobs_for('base', SEEDS, 'baseline'),
        [],
        phase='baseline',
    )
    assert status['paired'] == [101, 102, 103]
    assert status['drops'] == []
    assert status['complete'] is True


def test_set_status_screening_repeats_limit_the_requested_set() -> None:
    status = set_status(
        _spec(),
        't1',
        [_result('t1', 101), _result('t1', 102, status='crash'), _result('t1', 103, limited=True)],
        _jobs_for('t1', SEEDS, 'screening'),
        [_result('ref', seed) for seed in SEEDS],
        phase='screening',
    )
    assert status['requested'] == [101]  # seeds.screening_repeats defaults to 1
    assert status['drops'] == []  # 102/103 were never requested
    assert status['complete'] is True


def test_set_status_names_a_missing_seed() -> None:
    # 103 never ran at all: no job and no result
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in (101, 102)],
        _jobs_for('t1', [101, 102]),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == ['seed_missing:t1:103']
    assert status['complete'] is False


def test_set_status_names_a_pending_seed() -> None:
    # 103 was submitted but nothing is back yet
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in (101, 102)],
        _jobs_for('t1', SEEDS),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == ['seed_pending:t1:103']
    assert status['complete'] is False


def test_set_status_names_a_crashed_seed() -> None:
    results = [_result('t1', seed) for seed in (101, 102)] + [_result('t1', 103, status='crash')]
    status = set_status(
        _spec(),
        't1',
        results,
        _jobs_for('t1', SEEDS),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == ['seed_crashed:t1:103']
    assert status['complete'] is False


def test_set_status_names_a_limited_seed() -> None:
    results = [_result('t1', seed) for seed in (101, 102)] + [_result('t1', 103, limited=True)]
    status = set_status(
        _spec(),
        't1',
        results,
        _jobs_for('t1', SEEDS),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == ['seed_limited:t1:103']
    assert status['complete'] is False


def test_set_status_names_an_unpaired_seed() -> None:
    # the trial measured 103 but the reference crashed there
    refs = [_result('ref', 101), _result('ref', 102), _result('ref', 103, status='crash')]
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in SEEDS],
        _jobs_for('t1', SEEDS),
        refs,
        phase='confirm',
    )
    assert status['drops'] == ['seed_unpaired:t1:103']
    assert status['measured'] == [101, 102, 103]
    assert status['complete'] is False


def test_set_status_names_an_unprovenanced_result() -> None:
    # 103 ran and paired but no job_submitted ever proved its phase
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in SEEDS],
        _jobs_for('t1', [101, 102]),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == ['phase_unprovenanced:t1:103']
    assert status['paired'] == [101, 102, 103]
    assert status['complete'] is False


def test_set_status_requires_a_ledgered_job_phase() -> None:
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in SEEDS],
        _jobs_for('t1', SEEDS, None),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == [
        'phase_unprovenanced:t1:101',
        'phase_unprovenanced:t1:102',
        'phase_unprovenanced:t1:103',
    ]
    assert status['complete'] is False


def test_set_status_without_jobs_demands_confirm_provenance() -> None:
    # phase 'confirm' always demands the job_submitted provenance of every measured seed
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in SEEDS],
        [],
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == [
        'phase_unprovenanced:t1:101',
        'phase_unprovenanced:t1:102',
        'phase_unprovenanced:t1:103',
    ]
    assert status['complete'] is False


def test_set_status_an_unmeasurable_result_is_never_measured() -> None:
    jobs = _jobs_for('t1', SEEDS)
    refs = [_result('ref', seed) for seed in SEEDS]
    good = [_result('t1', seed) for seed in (101, 102)]
    variants = [
        good + [_result('t1', 103, eval_policy_fingerprint=OTHER_FP)],
        good + [_result('t1', 103, metrics={'other': {'value': 1.0, 'se': 0.1}})],
    ]
    for results in variants:
        status = set_status(_spec(), 't1', results, jobs, refs, phase='confirm')
        assert status['measured'] == [101, 102]
        assert status['paired'] == [101, 102]
        assert status['drops'] == ['seed_unpaired:t1:103']
        assert status['complete'] is False


def test_set_status_drops_are_sorted() -> None:
    # 102 is unpaired and 103 crashed: the names sort, they are not emitted in request order
    status = set_status(
        _spec(),
        't1',
        [_result('t1', 101), _result('t1', 102), _result('t1', 103, status='crash')],
        _jobs_for('t1', SEEDS),
        [_result('ref', 101), _result('ref', 102, status='crash')],
        phase='confirm',
    )
    assert status['drops'] == ['seed_crashed:t1:103', 'seed_unpaired:t1:102']


def test_set_status_drops_are_deduplicated() -> None:
    # two unprovenanced rows at (t1, 102) repeat one and the same named drop
    results = [
        _result('t1', 101),
        _result('t1', 102, metrics={'score': {'value': 0.5, 'se': 0.1}}),
        _result('t1', 102, metrics={'score': {'value': 0.7, 'se': 0.1}}),
    ]
    status = set_status(
        _spec(),
        't1',
        results,
        _jobs_for('t1', [101]),
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['drops'] == ['phase_unprovenanced:t1:102', 'seed_missing:t1:103']
    assert status['drops'] == sorted(set(status['drops']))
