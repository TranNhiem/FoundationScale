"""M3 Optuna proposer tests: a fake optuna module over told rows, cards, retries and the missing extra."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from foundationskills.skills.auto_research import proposers
from foundationskills.skills.auto_research.proposers import eligible, get_proposer, replay, select
from foundationskills.skills.auto_research.proposers_optuna import OptunaProposer

METRIC = 'val_accuracy'
FPR = 'sha256:' + 'a1' * 32

SPEC: dict[str, Any] = {
    'id': 'c1',
    'objective': {'metric': METRIC, 'direction': 'max', 'benchmarks': []},
    'eval_policy': {'fingerprint': FPR, 'metrics': [METRIC]},
    'seeds': {'baseline_repeats': 3, 'confirm_repeats': 3, 'seed_list': [1, 2, 3]},
    'axes': [
        {'key': 'optim.lr', 'type': 'log_float', 'min': 1e-6, 'max': 1e-3},
        {'key': 'optim.warmup_ratio', 'type': 'float', 'min': 0.0, 'max': 0.1},
        {'key': 'lora.rank', 'type': 'int', 'min': 4, 'max': 64},
        {'key': 'train.method', 'type': 'categorical', 'values': ['full', 'lora']},
    ],
}
CURRENT: dict[str, Any] = {'optim.lr': 1e-3, 'optim.warmup_ratio': 0.02, 'lora.rank': 24, 'train.method': 'lora'}
SYMPTOMS: list[str] = ['loss diverging']
AXIS_KEYS = {str(axis['key']) for axis in SPEC['axes']}

DRAW_A: dict[str, Any] = {'optim.lr': 5e-5, 'optim.warmup_ratio': 0.03, 'lora.rank': 12, 'train.method': 'lora'}
DRAW_B: dict[str, Any] = {'optim.lr': 2e-4, 'optim.warmup_ratio': 0.06, 'lora.rank': 32, 'train.method': 'full'}
DRAW_C: dict[str, Any] = {'optim.lr': 1e-3, 'optim.warmup_ratio': 0.01, 'lora.rank': 64, 'train.method': 'full'}
DRAW_BAD: dict[str, Any] = {'optim.lr': 5e-5, 'optim.warmup_ratio': 9.0, 'lora.rank': 12, 'train.method': 'lora'}

BASELINES: list[dict[str, Any]] = [
    {
        'trial': 'baseline',
        'role': 'baseline',
        'seed': i,
        'status': 'ok',
        'limited': False,
        'delta': {'optim.lr': 9e-4},  # the noise floor carries deltas and is still never told
        'metrics': {METRIC: {'value': 0.5, 'se': 0.001}},
    }
    for i in (1, 2, 3)
]


def trial_row(
    trial: str,
    value: float | None,
    *,
    seed: int = 1,
    role: str = 'candidate',
    status: str = 'ok',
    limited: bool = False,
    delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One trial_result row shaped like a ledger payload."""
    row: dict[str, Any] = {
        'trial': trial,
        'role': role,
        'seed': seed,
        'status': status,
        'limited': limited,
        'eval_policy_fingerprint': FPR,
        'metrics': {METRIC: {'value': value, 'se': 0.0}},
    }
    if delta is not None:
        row['delta'] = delta
    return row


def model_rows_ledger(n: int) -> list[dict[str, Any]]:
    """n measured non-baseline trials (t00..) on top of the calibrated baseline noise floor."""
    rows: list[dict[str, Any]] = list(BASELINES)
    for i in range(n):
        rows.append(trial_row(f't{i:02d}', 0.2 + 0.05 * i, delta={'optim.lr': (i + 1) * 5e-5}))
    return rows


def spec_with(proposer_block: dict[str, Any]) -> dict[str, Any]:
    """SPEC carrying a proposer block."""
    return {**SPEC, 'proposer': dict(proposer_block)}


def spec_with_direction(direction: str | None) -> dict[str, Any]:
    """SPEC with objective.direction set, or dropped (the default path)."""
    objective = {**SPEC['objective']}
    if direction is None:
        objective.pop('direction')
    else:
        objective['direction'] = direction
    return {**SPEC, 'objective': objective}


# ---- the injected fake optuna module ---------------------------------------


@dataclass
class FakeFloatDistribution:
    low: float
    high: float
    log: bool = False


@dataclass
class FakeIntDistribution:
    low: int
    high: int


@dataclass
class FakeCategoricalDistribution:
    choices: tuple


@dataclass
class FakeSampler:
    seed: int = 0


@dataclass
class FakeToldTrial:
    params: dict[str, Any]
    distributions: dict[str, Any]
    value: float


def _mid(dist: Any) -> Any:
    """A deterministic in-range draw per distribution (the fake unscripted fallback)."""
    if isinstance(dist, FakeCategoricalDistribution):
        return list(dist.choices)[0]
    if isinstance(dist, FakeIntDistribution):
        return (int(dist.low) + int(dist.high)) // 2
    return (float(dist.low) + float(dist.high)) / 2


class FakeStudy:
    """Records add_trial calls and serves the scripted draws one ask at a time."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.add_trial_calls: list[FakeToldTrial] = []
        self.ask_calls: list[dict[str, Any]] = []
        self._script = [dict(item) for item in script]
        self._cursor = 0

    def add_trial(self, trial: FakeToldTrial) -> None:
        self.add_trial_calls.append(trial)

    def ask(self, dists: dict[str, Any]) -> SimpleNamespace:
        self.ask_calls.append(dict(dists))
        if self._cursor < len(self._script):
            params = dict(self._script[self._cursor])
            self._cursor += 1
        else:
            params = {key: _mid(dist) for key, dist in dists.items()}
        return SimpleNamespace(params=params)


def fake_optuna(*, script: tuple[dict[str, Any], ...] = (), version: str = '0.0-fake') -> SimpleNamespace:
    """An injected fake optuna module: distributions, sampler, create_study/ask and create_trial."""
    studies: list[FakeStudy] = []
    calls: list[dict[str, Any]] = []
    log_levels: list[int] = []

    def create_trial(*, params: Any, distributions: Any, value: Any) -> FakeToldTrial:
        return FakeToldTrial(params=dict(params or {}), distributions=dict(distributions or {}), value=float(value))

    def create_study(*, direction: str = 'maximize', sampler: Any = None) -> FakeStudy:
        calls.append({'direction': direction, 'sampler': sampler})
        study = FakeStudy(list(script))
        studies.append(study)
        return study

    return SimpleNamespace(
        __version__=version,
        distributions=SimpleNamespace(
            FloatDistribution=FakeFloatDistribution,
            IntDistribution=FakeIntDistribution,
            CategoricalDistribution=FakeCategoricalDistribution,
        ),
        samplers=SimpleNamespace(TPESampler=FakeSampler),
        trial=SimpleNamespace(create_trial=create_trial),
        create_study=create_study,
        logging=SimpleNamespace(WARNING=30, set_verbosity=log_levels.append),
        studies=studies,
        study_calls=calls,
        log_calls=log_levels,
    )


# ---- tests -----------------------------------------------------------------


class TestAddTrial:
    def test_add_trial_tells_model_rows_only(self) -> None:
        """One add_trial per model row: measured, non-baseline, params recovered and kept to the axes."""
        fake = fake_optuna(script=(DRAW_A,))
        results = [
            *BASELINES,                                                             # noise floor: never told
            trial_row('noise-2', 0.5, role='baseline', delta={'optim.lr': 7e-4}),    # baseline role: never told
            trial_row('tcrash', 0.9, status='crash', delta={'optim.lr': 3e-4}),      # crash row: never told
            trial_row('tlimit', 0.9, limited=True, delta={'optim.lr': 3e-4}),        # limited row: never told
            trial_row('tnull', None, delta={'optim.lr': 3e-4}),                       # unmeasured: never told as 0
            trial_row('ta', 0.2, delta={'optim.lr': 2e-4, 'parallel.tp': 2}),         # off-axis keys are dropped
            trial_row('tb', 0.25, seed=1, delta={'lora.rank': 8}),
            trial_row('tb', 0.5, seed=2, role='confirm', delta={'lora.rank': 8}),     # value = trial mean
            trial_row('td', 0.3, delta={'train.method': 'lora'}),
            trial_row('naked', 0.3),                                                  # measured, no params anywhere
        ]
        proposer = OptunaProposer(seed=11, optuna_module=fake)
        cards, drops = proposer.propose(SPEC, results, [], CURRENT, SYMPTOMS, k=1)
        dists = proposer.distributions(SPEC)
        started = fake.studies[0]
        told = started.add_trial_calls
        assert len(told) == 3
        assert [(t.params, t.value) for t in told] == [
            ({'optim.lr': 2e-4}, 0.2),
            ({'lora.rank': 8}, 0.375),
            ({'train.method': 'lora'}, 0.3),
        ]
        assert all(t.distributions == {key: dists[key] for key in t.params} for t in told)
        told_params = [t.params for t in told]
        assert {'optim.lr': 9e-4} not in told_params           # baseline noise rows never told
        assert {'optim.lr': 7e-4} not in told_params           # baseline ROLE rows never told
        assert {'optim.lr': 3e-4} not in told_params           # crash/limited/unmeasured rows never told
        assert all(t.value is not None for t in told)
        assert 0.0 not in [t.value for t in told]              # never told as 0
        assert drops == ['model_row_no_params:naked']
        assert [c['idea'] for c in cards] == ['optuna:0'] and cards[0]['delta'] == DRAW_A
        assert cards[0]['reason'] == 'tpe seed=11 told 3 row(s)'

    def test_row_params_outside_the_distributions_are_a_named_drop(self) -> None:
        """A row kept with no in-model parameter is dropped model_row_no_params:<trial>, never told."""

        class EmptyDists(OptunaProposer):
            def distributions(self, spec: dict[str, Any]) -> dict[str, Any]:
                return {}

        fake = fake_optuna()
        rows = [
            trial_row('t1', 0.5, delta={'optim.lr': 2e-4}),
            trial_row('t2', 0.5, delta={'lora.rank': 16}),
        ]
        cards, drops = EmptyDists(seed=0, optuna_module=fake).propose(SPEC, rows, [], CURRENT, SYMPTOMS, k=0)
        assert cards == []
        assert drops == ['model_row_no_params:t1', 'model_row_no_params:t2']
        assert fake.studies[0].add_trial_calls == [] and fake.studies[0].ask_calls == []


class TestCards:
    def test_k_cards_are_full_in_axes_assignments(self) -> None:
        """Exactly k cards, optuna:0..k-1, every delta exactly the spec axes keys, score None."""
        fake = fake_optuna(script=(DRAW_A, DRAW_B, DRAW_C))
        cards, drops = OptunaProposer(seed=0, optuna_module=fake).propose(
            SPEC, model_rows_ledger(3), [], CURRENT, SYMPTOMS, k=3
        )
        assert [c['idea'] for c in cards] == ['optuna:0', 'optuna:1', 'optuna:2']
        assert [c['delta'] for c in cards] == [DRAW_A, DRAW_B, DRAW_C]
        assert all(set(c['delta']) == AXIS_KEYS for c in cards)
        assert all(c['score'] is None for c in cards)
        assert all(c['reason'] == 'tpe seed=0 told 3 row(s)' for c in cards)
        assert drops == []


class TestDirectionAndSampler:
    def test_direction_from_objective_and_seeded_tpe(self) -> None:
        """min -> minimize, anything else -> maximize, and TPESampler always gets self.seed."""
        fake = fake_optuna(script=(DRAW_A, DRAW_B, DRAW_C))
        rows = model_rows_ledger(1)
        OptunaProposer(seed=7, optuna_module=fake).propose(spec_with_direction('min'), rows, [], CURRENT, SYMPTOMS, k=1)
        OptunaProposer(seed=3, optuna_module=fake).propose(spec_with_direction('max'), rows, [], CURRENT, SYMPTOMS, k=1)
        OptunaProposer(seed=5, optuna_module=fake).propose(spec_with_direction(None), rows, [], CURRENT, SYMPTOMS, k=1)
        calls = fake.study_calls
        assert [call['direction'] for call in calls] == ['minimize', 'maximize', 'maximize']
        assert all(isinstance(call['sampler'], FakeSampler) for call in calls)
        assert [call['sampler'].seed for call in calls] == [7, 3, 5]


class TestDistributions:
    def test_axes_in_order_log_float_and_categorical(self) -> None:
        """log_float -> FloatDistribution(log=True); categorical -> the values tuple; unknown types skipped."""
        fake = fake_optuna()
        spec: dict[str, Any] = {**SPEC, 'axes': [*SPEC['axes'], {'key': 'data.mix', 'type': 'str', 'min': 0, 'max': 1}]}
        dists = OptunaProposer(seed=0, optuna_module=fake).distributions(spec)
        assert list(dists) == ['optim.lr', 'optim.warmup_ratio', 'lora.rank', 'train.method']
        assert dists['optim.lr'] == FakeFloatDistribution(1e-6, 1e-3, log=True)
        assert dists['optim.warmup_ratio'] == FakeFloatDistribution(0.0, 0.1)
        assert dists['lora.rank'] == FakeIntDistribution(4, 64)
        assert dists['train.method'] == FakeCategoricalDistribution(('full', 'lora'))
        assert dists['train.method'].choices == ('full', 'lora')


class TestRetries:
    def test_out_of_axes_draw_is_reasked_once_and_the_retry_is_kept(self) -> None:
        """One bad draw costs exactly one extra ask (k + 1) and the in-bounds retry becomes the card."""
        fake = fake_optuna(script=(DRAW_BAD, DRAW_A, DRAW_B))
        cards, drops = OptunaProposer(seed=0, optuna_module=fake).propose(
            SPEC, model_rows_ledger(1), [], CURRENT, SYMPTOMS, k=2
        )
        assert len(fake.studies[0].ask_calls) == 3            # k + 1
        assert [c['delta'] for c in cards] == [DRAW_A, DRAW_B]
        assert [c['idea'] for c in cards] == ['optuna:0', 'optuna:1']
        assert drops == []

    def test_two_bad_draws_drop_the_card_and_never_clamp(self) -> None:
        """A second bad draw drops model_out_of_axes:optuna:<i>; the card is absent and nothing is clamped."""
        fake = fake_optuna(script=(DRAW_BAD, DRAW_BAD, DRAW_B))
        cards, drops = OptunaProposer(seed=0, optuna_module=fake).propose(
            SPEC, model_rows_ledger(1), [], CURRENT, SYMPTOMS, k=2
        )
        assert len(fake.studies[0].ask_calls) == 3
        assert [c['idea'] for c in cards] == ['optuna:1']
        assert cards[0]['delta'] == DRAW_B
        assert all(c['delta'] != DRAW_BAD for c in cards)     # never clamped back into the axes
        assert drops == ['model_out_of_axes:optuna:0']


class TestMissingExtra:
    def test_unimportable_optuna_is_a_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without optuna installed the proposer is absent: ImportError, None for get_proposer, gated."""
        monkeypatch.setitem(sys.modules, 'optuna', None)  # the optional extra is not installed
        with pytest.raises(ImportError):
            OptunaProposer(seed=0)
        assert get_proposer('optuna') is None
        ten_rows = [{'trial': f't{i:02d}', 'params': {'optim.lr': 2e-4}, 'value': 0.1} for i in range(10)]
        assert eligible('optuna', spec_with({'name': 'optuna'}), ten_rows) == (False, 'extra_missing:optuna')


class TestVersionAndLogging:
    def test_version_recorded_and_info_logging_quieted(self) -> None:
        """version is the module version and set_verbosity(WARNING) is best effort for bare fakes."""
        fake = fake_optuna(version='9.9.9-fake')
        proposer = OptunaProposer(seed=0, optuna_module=fake)
        assert proposer.name == 'optuna' and proposer.version == '9.9.9-fake'
        assert fake.log_calls == [30]
        bare = OptunaProposer(seed=0, optuna_module=SimpleNamespace())   # no __version__, no logging
        assert bare.version == 'unknown'


class TestSelectEndToEnd:
    def test_fake_optuna_speaks_and_replays_through_select(self) -> None:
        """select over 10 model rows: optuna provenance, no fallback, cards exactly the fake draws."""
        fake = fake_optuna(script=(DRAW_A, DRAW_B, DRAW_C))
        registry = {
            'catalog': proposers.CatalogProposer,
            'optuna': lambda seed: OptunaProposer(seed=seed, optuna_module=fake),
        }
        spec = spec_with({'name': 'optuna', 'min_rows': 10})
        results = model_rows_ledger(10)
        cards, drops, provenance = select(spec, results, [], CURRENT, SYMPTOMS, registry=registry)
        reason = 'tpe seed=0 told 10 row(s)'
        assert drops == []
        assert provenance['requested'] == 'optuna' and provenance['fallback'] is None
        assert provenance['proposer'] == {'name': 'optuna', 'version': '0.0-fake'}
        assert provenance['seed'] == 0 and provenance['k'] == 3
        assert provenance['rows_digest'] == proposers.rows_digest(spec, results, [], CURRENT, SYMPTOMS)
        assert cards == [
            {'idea': 'optuna:0', 'delta': DRAW_A, 'score': None, 'reason': reason},
            {'idea': 'optuna:1', 'delta': DRAW_B, 'score': None, 'reason': reason},
            {'idea': 'optuna:2', 'delta': DRAW_C, 'score': None, 'reason': reason},
        ]
        assert all(len(study.add_trial_calls) == 10 for study in fake.studies)  # 10 model rows told per study
        assert replay(spec, results, [], CURRENT, SYMPTOMS, provenance, cards, registry) is True
