'''Review-finding regressions for the auto_research seed/phase bookkeeping (pure, CPU-only).'''
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / 'foundationskills').is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from foundationskills.skills.auto_research import seeds as seeds_mod  # noqa: E402
from foundationskills.skills.auto_research.seeds import (  # noqa: E402
    phase_problems,
    seed_plan,
    set_status,
    trial_phase,
)

FP = 'sha256:' + 'a' * 64
SEEDS = [101, 102, 103]


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        'objective': {'metric': 'score', 'direction': 'max'},
        'eval_policy': {'fingerprint': FP, 'metrics': ['score']},
        'seeds': {'seed_list': list(SEEDS), 'baseline_repeats': 3, 'confirm_repeats': 3},
    }
    spec.update(overrides)
    return spec


def _result(trial: str, seed: Any, **overrides: Any) -> dict[str, Any]:
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


def _job(job_id: int, trial: str, seed: Any, phase: Any = 'confirm') -> dict[str, Any]:
    row: dict[str, Any] = {'job_id': job_id, 'trial': trial, 'seed': seed}
    if phase is not None:
        row['phase'] = phase
    return row


def _drops(trial: str, text: str) -> list[str]:
    return [f'{text}:{trial}:{seed}' for seed in SEEDS]


# ---- finding 1: provenance compares the ledgered phase VALUE ----------------

def test_a_job_with_another_phase_does_not_prove_a_confirm_set() -> None:
    status = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in SEEDS],
        [_job(index, 't1', seed, 'screening') for index, seed in enumerate(SEEDS, 1)],
        [_result('ref', seed) for seed in SEEDS],
        phase='confirm',
    )
    assert status['measured'] == [101, 102, 103]
    assert status['drops'] == _drops('t1', 'phase_unprovenanced')
    assert status['complete'] is False


def test_a_job_with_another_phase_does_not_prove_a_baseline_or_screening_set() -> None:
    baseline = set_status(
        _spec(),
        'base',
        [_result('base', seed) for seed in SEEDS],
        [_job(index, 'base', seed, 'confirm') for index, seed in enumerate(SEEDS, 1)],
        [],
        phase='baseline',
    )
    assert baseline['drops'] == _drops('base', 'phase_unprovenanced')
    assert baseline['complete'] is False
    screening = set_status(
        _spec(),
        't1',
        [_result('t1', seed) for seed in SEEDS],
        [_job(index, 't1', seed, 'baseline') for index, seed in enumerate(SEEDS, 1)],
        [_result('ref', seed) for seed in SEEDS],
        phase='screening',
    )
    assert screening['drops'] == ['phase_unprovenanced:t1:101']
    assert screening['complete'] is False


# ---- finding 2: the phase derives from the ROLE only (A2) -------------------

def test_the_phase_never_falls_back_to_a_trial_spec_phase_key() -> None:
    assert trial_phase({'phase': 'confirm'}) is None
    assert trial_phase({'phase': 'confirm', 'seed': 101}) is None
    assert trial_phase({'phase': 'candidate', 'role': ''}) is None
    assert trial_phase({'role': 'confirm', 'phase': 'candidate'}) == 'confirm'
    spec = _spec()
    assert phase_problems({'phase': 'confirm', 'seed': 999}, spec) == [('AR-LN-001', 'phase_unknown:')]
    assert phase_problems({'phase': 'confirm', 'seeds': [101, 999]}, spec) == [('AR-LN-001', 'phase_unknown:')]
    assert phase_problems({'phase': 'screening', 'role': 'bogus'}, spec) == [('AR-LN-001', 'phase_unknown:bogus')]


# ---- finding 3: confirm always demands the phase provenance -----------------

def test_a_jobless_confirm_set_is_always_unprovenanced() -> None:
    results = [_result('t1', seed) for seed in SEEDS]
    refs = [_result('ref', seed) for seed in SEEDS]
    confirm = set_status(_spec(), 't1', results, [], refs, phase='confirm')
    assert confirm['measured'] == [101, 102, 103]
    assert confirm['paired'] == [101, 102, 103]
    assert confirm['drops'] == _drops('t1', 'phase_unprovenanced')
    assert confirm['complete'] is False
    # baseline and screening keep the M0 rule: without job rows nothing is demanded
    baseline = set_status(_spec(), 'base', [_result('base', seed) for seed in SEEDS], [], [], phase='baseline')
    assert baseline['drops'] == []
    assert baseline['complete'] is True
    screening = set_status(_spec(), 't1', results, [], refs, phase='screening')
    assert screening['drops'] == []
    assert screening['complete'] is True


# ---- finding 4: malformed input is never a crash and never a measurement ----

def test_malformed_rows_and_metrics_are_never_measured() -> None:
    jobs = [_job(1, 't1', 101, 'confirm'), _job(2, 't1', 102, 'confirm')]
    refs = [_result('ref', seed) for seed in SEEDS]
    variants: list[Any] = [
        _result('t1', 101, metrics='oops'),
        _result('t1', 101, metrics=['oops']),
        _result('t1', 101, metrics={'score': 'oops'}),
        _result('t1', 101, metrics={'score': ['oops']}),
        _result('t1', 101, metrics={'score': {}}),
        _result('t1', 101, metrics={'score': {'value': 'oops', 'se': 0.1}}),
        _result('t1', 101, metrics={'score': {'value': None, 'se': 0.1}}),
        _result('t1', 101, metrics={'score': {'value': True, 'se': 0.1}}),
        _result('t1', 101, metrics={'score': {'value': float('nan'), 'se': 0.1}}),
        _result('t1', 101, metrics={'score': {'value': float('inf'), 'se': 0.1}}),
        _result('t1', 101, metrics={'score': {'value': 1.0, 'se': float('nan')}}),
    ]
    for row in variants:
        status = set_status(_spec(), 't1', [row, 'junk', None], jobs, refs, phase='confirm')
        assert status['measured'] == [], row
        assert status['paired'] == [], row
        assert status['complete'] is False, row


def test_junk_stream_entries_are_never_jobs_or_measurements() -> None:
    refs = [_result('ref', seed) for seed in SEEDS]
    empty = set_status(_spec(), 't1', ['junk', None, 3, ['row']], ['junk', None], refs, phase='confirm')
    assert empty['measured'] == []
    assert empty['drops'] == [
        'seed_missing:t1:101',
        'seed_missing:t1:102',
        'seed_missing:t1:103',
    ]
    assert empty['complete'] is False
    # junk job rows prove nothing: a job-less confirm result is unprovenanced
    status = set_status(
        _spec(),
        't1',
        [_result('t1', 101)],
        ['junk', None],
        [_result('ref', 101)],
        phase='confirm',
    )
    assert status['measured'] == [101]
    assert status['drops'] == [
        'phase_unprovenanced:t1:101',
        'seed_missing:t1:102',
        'seed_missing:t1:103',
    ]
    assert status['complete'] is False


# ---- finding 5: drop names render the canonical requested seed --------------

def test_drops_render_the_canonical_requested_seed() -> None:
    spec = _spec(seeds={'seed_list': list(SEEDS), 'baseline_repeats': 2, 'confirm_repeats': 2})
    status = set_status(
        spec,
        't1',
        [_result('t1', 101.0), _result('t1', '102')],
        [],
        [_result('ref', 101), _result('ref', 102)],
        phase='confirm',
    )
    assert status['measured'] == [101, 102]
    assert status['paired'] == [101, 102]
    assert status['drops'] == ['phase_unprovenanced:t1:101', 'phase_unprovenanced:t1:102']
    crash = set_status(
        _spec(seeds={'seed_list': list(SEEDS), 'baseline_repeats': 1, 'confirm_repeats': 1}),
        't1',
        [_result('t1', 101.0, status='crash')],
        [_job(1, 't1', 101.0, 'confirm')],
        [_result('ref', 101)],
        phase='confirm',
    )
    assert crash['drops'] == ['seed_crashed:t1:101']


# ---- finding 6: check_seed_spec is gone (campaign.check_spec owns AR-IN-007) 

def test_check_seed_spec_is_gone_and_the_seed_plan_stays() -> None:
    assert not hasattr(seeds_mod, 'check_seed_spec')
    assert 'check_seed_spec' not in list(vars(seeds_mod))
    assert seed_plan(_spec()) == {
        'seed_list': [101, 102, 103],
        'baseline_repeats': 3,
        'confirm_repeats': 3,
        'screening_repeats': 1,
    }
