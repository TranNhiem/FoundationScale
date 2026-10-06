"""M3 Optuna proposer for auto_research (optional extra): TPE cards over the told model rows.

``optuna`` is an optional extra: the import lives in ``OptunaProposer.__init__`` (never at module
level) and its ImportError propagates to ``proposers.get_proposer`` as a missing extra. Every told
row is a B6 recovered model row; every card is a B5 FULL sampled assignment over ``spec.axes``; an
out-of-axes draw is asked once more and then dropped as ``model_out_of_axes:optuna:<i>`` - never
clamped. A fresh seeded study per call (B10) keeps ``proposers.replay`` byte-verifiable.
"""
from __future__ import annotations

from typing import Any

from . import proposers
from .campaign import axis_value_fits


def _axes(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """spec axis by key in declared order (junk axis entries are ignored, never crash)."""
    return {str(axis.get('key')): dict(axis) for axis in (spec.get('axes') or []) if isinstance(axis, dict)}


def _fits_all(axes: dict[str, dict[str, Any]], delta: dict[str, Any]) -> bool:
    """True for a non-empty draw whose every value fits its axis (bad draws are re-asked, never clamped)."""
    if not delta:
        return False
    for key, value in delta.items():
        axis = axes.get(key)
        if axis is None or not axis_value_fits(axis, value):
            return False
    return True


class OptunaProposer:
    """TPE proposer: one fresh in-memory study per call over told rows (seeded, byte-replayable)."""

    name = 'optuna'

    def __init__(self, seed: int = 0, optuna_module: Any = None) -> None:
        module = self._load() if optuna_module is None else optuna_module
        self.seed = seed
        self._optuna = module
        self.version = str(getattr(module, '__version__', None) or 'unknown')
        self._quiet_logs()

    @staticmethod
    def _load() -> Any:
        """``import optuna``: the optional extra; its ImportError is a missing extra, never a crash."""
        import optuna as module

        return module

    def _quiet_logs(self) -> None:
        """Best effort: silence optuna INFO logs (fakes need not provide ``logging``)."""
        logging_module = getattr(self._optuna, 'logging', None)
        if logging_module is None or not hasattr(logging_module, 'set_verbosity'):
            return
        try:
            logging_module.set_verbosity(getattr(logging_module, 'WARNING', 30))
        except Exception:
            pass  # logging is cosmetic: never a crash

    def distributions(self, spec: dict[str, Any]) -> dict[str, Any]:
        """``spec.axes`` in declared order: axis key -> distribution; unknown axis types are skipped."""
        built: dict[str, Any] = {}
        for key, axis in _axes(spec).items():
            dist = self._distribution(axis)
            if dist is not None:
                built[key] = dist
        return built

    def _distribution(self, axis: dict[str, Any]) -> Any:
        distributions = self._optuna.distributions
        kind = str(axis.get('type') or '')
        try:
            if kind == 'log_float':
                return distributions.FloatDistribution(float(axis['min']), float(axis['max']), log=True)
            if kind == 'float':
                return distributions.FloatDistribution(float(axis['min']), float(axis['max']))
            if kind == 'int':
                return distributions.IntDistribution(int(axis['min']), int(axis['max']))
            if kind == 'categorical':
                return distributions.CategoricalDistribution(tuple(axis.get('values') or ()))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None  # a broken axis is a skipped axis: never a crash
        return None

    @staticmethod
    def _direction(spec: dict[str, Any]) -> str:
        """The study direction: only an exact min maps to minimize, everything else maximizes."""
        objective = spec.get('objective') if isinstance(spec.get('objective'), dict) else {}
        return 'minimize' if objective.get('direction') == 'min' else 'maximize'

    def _ask(self, study: Any, dists: dict[str, Any]) -> dict[str, Any]:
        """One sampled assignment over ``dists`` (B5: the FULL assignment, attributed as such)."""
        trial = study.ask(dists)
        params = getattr(trial, 'params', None)
        if not isinstance(params, dict):
            return {}
        return {key: params[key] for key in dists if key in params}

    def propose(
        self,
        spec: dict[str, Any],
        results: list[dict[str, Any]],
        launches: list[dict[str, Any]],
        current: dict[str, Any],
        symptoms: list[str],
        *,
        k: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """(cards, drops): TPE cards over the told model rows; out-of-axes draws re-asked, then dropped."""
        rows, row_drops = proposers.model_rows(spec, results, launches)
        drops = {str(drop) for drop in row_drops}
        dists = self.distributions(spec)
        axes = _axes(spec)
        study = self._optuna.create_study(
            direction=self._direction(spec),
            sampler=self._optuna.samplers.TPESampler(seed=self.seed),
        )
        told = 0
        for row in rows:
            params = {key: value for key, value in row['params'].items() if key in dists}
            if not params:
                trial_name = str(row.get('trial'))
                drops.add(f'model_row_no_params:{trial_name}')  # no parameter recovered: no model row (B6)
                continue
            study.add_trial(
                self._optuna.trial.create_trial(
                    params=params,
                    distributions={key: dists[key] for key in params},
                    value=float(row['value']),
                )
            )
            told += 1
        cards: list[dict[str, Any]] = []
        for i in range(k):
            kept: dict[str, Any] | None = None
            for _attempt in (0, 1):  # ask once more on an out-of-axes draw: never a clamp (B5)
                draw = self._ask(study, dists)
                if _fits_all(axes, draw):
                    kept = draw
                    break
            if kept is None:
                drops.add(f'model_out_of_axes:optuna:{i}')
                continue
            cards.append({
                'idea': f'optuna:{i}',
                'delta': dict(kept),
                'score': None,
                'reason': f'tpe seed={self.seed} told {told} row(s)',
            })
        return cards, sorted(drops)
