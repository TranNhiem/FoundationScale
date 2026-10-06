# Regression tests for the auto_research M2 review findings (locks allowlist + robustness,
# campaign non-dict cluster / reserve_frac range). Each test FAILS on the pre-fix code.
from __future__ import annotations

import json

from foundationskills.skills.auto_research.campaign import _check_max_in_flight, check_spec
from foundationskills.skills.auto_research.ledger import ledger_files
from foundationskills.skills.auto_research.locks import closed_campaign, closing_check

CAMPAIGN = 'c1'
FINDING = [('AR-LG-002', 'campaign_closed')]


def _entries(events):
    chain = ledger_files(events)['ledger/chain.jsonl']
    return [json.loads(line) for line in chain.splitlines() if line.strip()]


def _closed():
    return _entries(
        [
            ('campaign_approved', CAMPAIGN, '', {'campaign_hash': 'sha256:' + '0' * 64}),
            ('campaign_closed', CAMPAIGN, '', {'reason': 'budget exhausted'}),
        ]
    )


def _base_spec():
    return {
        'objective': {'metric': 'eval/perplexity', 'direction': 'min'},
        'eval_policy': {'metrics': ['eval/perplexity'], 'fingerprint': 'sha256:' + 'a' * 64},
        'base': {'model': 'm1', 'fingerprint': 'sha256:' + 'b' * 64},
        'confirm': {'guardrail_directions': {}},
        'budget': {'gpu_hours_total': 100.0, 'max_runs': 10, 'per_run_timeout_h': 24.0},
        'axes': [],
        'cluster': {
            'time': '10-00:00:00',
            'exclude': ['r01dgx02'],
            'max_nodes': 4,
            'gpus_per_node': 8,
            'partition': 'p1',
            'max_in_flight': 4,
        },
    }


def test_locks_allowlist_refuses_everything_but_check_and_close_on_closed_campaign():
    entries = _closed()
    for junk in (None, '', 'verify', 'query', 'Launch', 'claim ', 'mystery', 'CHECK', 42):
        problems = closing_check(entries, CAMPAIGN, junk)
        assert problems == FINDING, (junk, problems)
    for keep in ('check', 'close'):
        problems = closing_check(entries, CAMPAIGN, keep)
        assert problems == [], (keep, problems)


def test_locks_non_dict_entries_are_skipped_never_crash():
    for_junk_first = ['str-junk', {'op': 'other', 'campaign': CAMPAIGN}, ['list-junk'], None, 42]
    assert closed_campaign(for_junk_first, CAMPAIGN) is False
    assert closing_check(for_junk_first, CAMPAIGN, 'claim') == []
    hidden = ['str-junk', {'op': 'campaign_closed', 'campaign': CAMPAIGN}, None]
    assert closed_campaign(hidden, CAMPAIGN) is True
    assert closing_check(hidden, CAMPAIGN, 'claim') == FINDING


def test_locks_non_list_entries_is_treated_as_no_entries():
    for entries in (None, 42, 3.14, 'nonsense', {'a': 1}, ('tup',), True, b'bytes'):
        assert closed_campaign(entries, CAMPAIGN) is False, entries
        assert closing_check(entries, CAMPAIGN, 'claim') == [], entries


def test_campaign_check_max_in_flight_non_dict_cluster_is_treated_as_empty():
    for bad in ('not-a-dict', ['no', 'pair', 'here'], 42, True, ['also-junk', 'later']):
        spec = {'cluster': bad, 'budget': {'max_runs': 5, 'gpu_hours_total': 100.0}}
        problems = _check_max_in_flight(spec, 5)
        assert problems == [], (bad, problems)


def test_campaign_reserve_frac_out_of_range_is_ar_in_003():
    def _has(problems):
        return any(p[0] == 'AR-IN-003' and 'reserve_frac_out_of_range' in p[1] for p in problems)

    for bad in (1.5, -0.1, 1.0, True, 'abc', float('inf'), float('nan')):
        spec = _base_spec()
        spec['budget']['reserve_frac'] = bad
        problems = check_spec(spec)
        assert _has(problems), (bad, problems)
    for good in (0.0, 0.3, 0.5, 0.99):
        spec = _base_spec()
        spec['budget']['reserve_frac'] = good
        problems = check_spec(spec)
        assert not _has(problems), (good, problems)
    problems = check_spec(_base_spec())
    assert not _has(problems), problems
