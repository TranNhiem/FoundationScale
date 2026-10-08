"""M3 proposers for auto_research (pure): the catalog baseline and pluggable model adapters.

The ideas catalog (``propose.py``, untouched) is the deterministic baseline and the
regression oracle; model proposers are lazy optional extras. Every model failure is a
named drop falling back to the catalog - never a crash, never a silent switch - and the
returned ``provenance`` pins who actually spoke (name/version/seed/rows_digest).
"""
from __future__ import annotations

import copy
import statistics
from typing import Any

from foundationskills.skills.auto_research import propose as _catalog
from foundationskills.skills.auto_research.accept import _point
from foundationskills.skills.auto_research.campaign import axis_value_fits
from foundationskills.skills.auto_research.ledger import canonical, sha256_hex

DEFAULT_PROPOSER = "catalog"
DEFAULT_MIN_ROWS = 10
DEFAULT_K = 3
KNOWN_PROPOSERS = ("catalog", "optuna", "optuna-cma")


def proposer_config(spec: dict[str, Any]) -> dict[str, Any]:
    """The ``proposer`` block with defaults applied (shape only - AR-IN-008 validates in campaign.py)."""
    block = spec.get("proposer") if isinstance(spec.get("proposer"), dict) else {}
    return {
        "name": block.get("name") if block.get("name") is not None else DEFAULT_PROPOSER,
        "min_rows": block.get("min_rows") if block.get("min_rows") is not None else DEFAULT_MIN_ROWS,
        "require_model": block.get("require_model") if block.get("require_model") is not None else False,
        "seed": block.get("seed") if block.get("seed") is not None else 0,
    }


class CatalogProposer:
    """The ideas catalog: deterministic baseline and byte-identity regression oracle."""

    name = "catalog"
    version = "1"
    package_version = "builtin"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

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
        """EXACTLY ``propose.propose`` (same call, byte-identical cards) + evaluate()'s pruned ideas as named drops."""
        cards = _catalog.propose(spec, results, current, symptoms, k=k, launches=launches)
        _live, dropped = _catalog.evaluate(spec, results, current, symptoms, launches)
        return cards, [f"catalog_{card['reason']}:{card['idea']}" for card in dropped]


def _optuna_factory(seed: int):  # pragma: no cover - one-line lazy hop to the optional extra
    """Lazy Optuna adapter (``proposers_optuna`` and ``optuna`` are optional extras, never a crash)."""
    from foundationskills.skills.auto_research.proposers_optuna import OptunaProposer

    return OptunaProposer(seed=seed)


def _optuna_cma_factory(seed: int):  # pragma: no cover - one-line lazy hop to the optional extra
    """Lazy Optuna CMA-ES adapter (``proposers_optuna`` with its ``cmaes`` sampler)."""
    from foundationskills.skills.auto_research.proposers_optuna import OptunaProposer

    return OptunaProposer(seed=seed, sampler="cmaes")


REGISTRY: dict[str, Any] = {"catalog": CatalogProposer, "optuna": _optuna_factory, "optuna-cma": _optuna_cma_factory}


def get_proposer(name: str, seed: int = 0, registry: dict[str, Any] | None = None):
    """The proposer for ``name``; None for an unknown name or a factory raising ImportError (a missing extra)."""
    if not isinstance(name, str):
        return None
    factory = dict(REGISTRY if registry is None else registry).get(name)
    if factory is None:
        return None
    try:
        return factory(seed)
    except ImportError:
        return None


def package_version(proposer: Any) -> str | None:
    """The recorded package version of a built proposer (C4); None when it records none."""
    return getattr(proposer, "package_version", None)


# ---- parameter recovery (B6) ------------------------------------------------


def _axes(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """spec axis by its declared key (junk axis entries are ignored, never crash)."""
    return {str(a.get("key")): dict(a) for a in (spec.get("axes") or []) if isinstance(a, dict)}


def _keep_axes(axes: dict[str, dict[str, Any]], delta: Any) -> dict[str, Any]:
    """Only spec axis keys whose value fits the axis (non-dict deltas are ignored, never crash)."""
    kept: dict[str, Any] = {}
    if not isinstance(delta, dict):
        return kept
    for key, value in delta.items():
        axis = axes.get(str(key))
        if axis is not None and axis_value_fits(axis, value):
            kept[str(key)] = value
    return kept


def _launch_delta(payload: dict[str, Any], launch_spec: dict[str, Any]) -> Any:
    """One launch payload's delta chain (B6): top-level, else launch_spec, else launch_spec.trial_spec."""
    specs = [s for s in (launch_spec.get("trial_spec"), payload.get("trial_spec")) if isinstance(s, dict)]
    for candidate in (payload.get("delta"), launch_spec.get("delta"), *(s.get("delta") for s in specs)):
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def trial_params(spec: dict[str, Any], results: Any, launches: Any) -> dict[str, dict[str, Any]]:
    """trial -> recovered axis params (B6): the latest result row's delta, else the latest launch payload chain."""
    axes = _axes(spec)
    winners: dict[str, dict[str, Any]] = {}
    from_result: set[str] = set()
    for row in results or []:
        if not isinstance(row, dict):
            continue
        kept = _keep_axes(axes, row.get("delta"))
        if kept:  # the latest result delta that recovers an axis param wins over any launch payload (B6)
            trial = str(row.get("trial"))
            from_result.add(trial)
            winners[trial] = kept
    latest: dict[str, dict[str, Any]] = {}
    for payload in launches or []:
        if not isinstance(payload, dict):
            continue
        raw_spec = payload.get("launch_spec")
        launch_spec = dict(raw_spec) if isinstance(raw_spec, dict) else {}
        trial = payload.get("trial")
        if trial is None:
            trial = launch_spec.get("trial")
        if trial is None:
            continue
        kept = _keep_axes(axes, _launch_delta(payload, launch_spec))
        if kept:
            latest[str(trial)] = kept
    for trial, params in latest.items():
        if trial not in from_result:
            winners[trial] = params
    return {trial: params for trial, params in winners.items() if params}


def model_rows(spec: dict[str, Any], results: Any, launches: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """(rows, drops) per B2: one row per non-baseline trial with a measured objective point and axis params."""
    metric = str(dict(spec.get("objective") or {}).get("metric") or "")
    params = trial_params(spec, results, launches)
    measured: dict[str, list[float]] = {}
    for row in results or []:
        if not isinstance(row, dict) or row.get("role") == "baseline":
            continue
        trial = str(row.get("trial"))
        if trial == "baseline":
            continue
        point = _point(row, metric)
        if point is None:
            continue  # crash/limited/None metric: unmeasured, never 0
        measured.setdefault(trial, []).append(point["value"])
    rows: list[dict[str, Any]] = []
    drops: set[str] = set()
    for trial, values in measured.items():
        if not params.get(trial):
            drops.add(f"model_row_no_params:{trial}")
            continue
        rows.append({"trial": trial, "params": dict(params[trial]), "value": statistics.fmean(values)})
    rows.sort(key=lambda row: row["trial"])
    return rows, sorted(drops)


# ---- gating, provenance ------------------------------------------------------


def rows_digest(spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any) -> str:
    """"sha256:<hex>" pinning the exact proposer input (spec fields + rows + recorded request hints)."""
    body = {
        "spec": spec,
        "catalog": sha256_hex(canonical(_catalog.load_catalog())),
        "proposer": proposer_config(spec),
        "results": results,
        "launches": launches,
        "current": current,
        "symptoms": symptoms,
    }
    return "sha256:" + sha256_hex(canonical(body))


def eligible(name: str, spec: dict[str, Any], rows: list[dict[str, Any]], registry: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """(ok, reason) gate; the catalog is always eligible. Reason strings are the stable drop names."""
    if name == DEFAULT_PROPOSER:
        return True, None
    if not isinstance(name, str):
        return False, f"unknown_proposer:{name}"
    if name not in dict(REGISTRY if registry is None else registry):
        return False, f"unknown_proposer:{name}"
    cfg = proposer_config(spec)
    try:
        min_rows = max(1, int(cfg["min_rows"]))
    except (TypeError, ValueError):
        min_rows = DEFAULT_MIN_ROWS
    if len(rows) < min_rows:
        return False, f"below_min_rows:{len(rows)}/{min_rows}"
    try:
        available = get_proposer(name, cfg["seed"], registry) is not None
    except Exception:  # a broken (not missing) extra: counted, never a crash
        return False, "model_error"
    if not available:
        return False, f"extra_missing:{name}"
    if not (spec.get("axes") or []):
        return False, "no_axes"
    return True, None


def _card_in_axes(spec: dict[str, Any], card: Any) -> bool:
    """True when a card's delta is a non-empty axis-respecting dict."""
    if not isinstance(card, dict):
        return False
    delta = card.get("delta")
    if not isinstance(delta, dict) or not delta:
        return False
    axes = _axes(spec)
    for key, value in delta.items():
        axis = axes.get(str(key))
        if axis is None or not axis_value_fits(axis, value):
            return False
    return True


def _catalog_cards(
    spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any, k: int
) -> tuple[list[dict[str, Any]], list[str]]:
    return CatalogProposer().propose(spec, results, launches, current, symptoms, k=k)


def _reproduces(
    name: str, seed: Any, spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any,
    k: int, raw: Any, registry: dict[str, Any] | None,
) -> bool:
    """A FRESH proposer over the same inputs reproduces the raw cards byte-identically (AR-PR-001 evidence)."""
    try:
        fresh = get_proposer(name, seed, registry)
        if fresh is None:
            return False
        again, _ = fresh.propose(spec, results, launches, current, symptoms, k=k)
        return canonical(again) == canonical(raw)
    except Exception:
        return False


def _digest_or_none(spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any) -> str | None:
    """rows_digest, or None when the inputs are not canonicalisable (e.g. NaN in a request hint): unmeasured, never faked."""
    try:
        return rows_digest(spec, results, launches, current, symptoms)
    except (TypeError, ValueError):
        return None


def select(
    spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any,
    *, k: int = DEFAULT_K, registry: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """(cards, drops, provenance): the named proposer speaks, or the catalog does with a counted fallback drop."""
    cfg = proposer_config(spec)
    try:
        k = max(0, int(k))
    except (TypeError, ValueError):
        k = DEFAULT_K
    provenance: dict[str, Any] = {
        "requested": cfg["name"],
        "proposer": {"name": CatalogProposer.name, "version": CatalogProposer.version},
        "package_version": CatalogProposer.package_version,
        "seed": cfg["seed"],
        "rows_digest": _digest_or_none(spec, results, launches, current, symptoms),
        "k": k,
        "fallback": None,
    }
    name = cfg["name"]
    if name == DEFAULT_PROPOSER:
        cards, catalog_drops = _catalog_cards(spec, results, launches, current, symptoms, k)
        return cards, catalog_drops, provenance

    rows, row_drops = model_rows(spec, results, launches)
    drops: list[str] = list(row_drops)
    live: list[Any] = []
    who = {"name": CatalogProposer.name, "version": CatalogProposer.version}
    ok, reason = eligible(name, spec, rows, registry)
    if ok:
        try:
            proposer = get_proposer(name, cfg["seed"], registry)
        except Exception:
            proposer = None
        if proposer is None:
            reason = f"extra_missing:{name}"
        else:
            who = {
                "name": str(getattr(proposer, "name", name)),
                "version": str(getattr(proposer, "version", "0")),
            }
            raw: Any = []
            raw_drops: Any = []
            try:
                raw, raw_drops = proposer.propose(spec, results, launches, current, symptoms, k=k)
            except Exception:
                reason, raw = "model_error", []
            drops.extend(str(drop) for drop in (raw_drops or []))
            kept: list[Any] = []
            for card in (raw or []):
                if _card_in_axes(spec, card):
                    kept.append(card)
                else:
                    idea = card.get("idea") if isinstance(card, dict) else None
                    drops.append(f"model_out_of_axes:{idea}")
            if reason is None:
                if not kept:
                    reason = "no_model_cards"
                elif _reproduces(name, cfg["seed"], spec, results, launches, current, symptoms, k, raw, registry):
                    live = kept
                else:
                    reason = "unverified"

    drops = list(dict.fromkeys(drops))
    if reason is None and live:
        provenance["proposer"] = dict(who)
        provenance["package_version"] = package_version(proposer)
        return list(live), drops, provenance
    reason = reason or "no_model_cards"
    drops.append(f"proposer_fallback_catalog:{reason}")
    provenance["fallback"] = reason
    cards, catalog_drops = _catalog_cards(spec, results, launches, current, symptoms, k)
    return cards, list(dict.fromkeys([*drops, *catalog_drops])), provenance


def replay(
    spec: dict[str, Any], results: Any, launches: Any, current: Any, symptoms: Any,
    provenance: Any, cards: Any, registry: dict[str, Any] | None = None,
) -> bool:
    """True when the recorded provenance re-proposes canonical-identical cards over the recorded inputs."""
    if not isinstance(provenance, dict):
        return False
    try:
        if rows_digest(spec, results, launches, current, symptoms) != provenance.get("rows_digest"):
            return False
    except Exception:
        return False
    who = provenance.get("proposer")
    name = who.get("name") if isinstance(who, dict) else None
    if not isinstance(name, str) or not name:
        return False
    try:
        proposer = get_proposer(name, provenance.get("seed", 0), registry)
    except Exception:
        return False
    if proposer is None:
        return False
    try:
        k = max(0, int(provenance.get("k", DEFAULT_K)))
        fresh, _ = proposer.propose(spec, results, launches, current, symptoms, k=k)
        if name != DEFAULT_PROPOSER:
            fresh = [card for card in (fresh or []) if _card_in_axes(spec, card)]
        return canonical(fresh) == canonical(cards)
    except Exception:
        return False


def reverify(
    spec: dict[str, Any], proposal: Any, results: Any, launches: Any, *, registry: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Close-time replay re-check (C6) over one recorded ``proposal``: (status, reason) - never raises."""
    try:
        return _reverify(spec, proposal, results, launches, registry)
    except Exception:
        return "unmeasured", "legacy_proposal"  # junk is a declined claim, never a crash


def _reverify(
    spec: dict[str, Any], proposal: Any, results: Any, launches: Any, registry: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """The C6 checks in record order: legacy -> recorded -> prefix -> rebuild -> re-propose."""
    if not isinstance(proposal, dict):
        return "unmeasured", "legacy_proposal"                       # 1. not a record at all
    who = proposal.get("proposer")
    if isinstance(who, dict) and who.get("name") == "llm":          # M5b: generation is unreplayable - re-parse only
        from .proposers_llm import reverify_record
        return reverify_record(spec, proposal)
    inputs = proposal.get("replay_inputs")
    if not isinstance(inputs, dict):
        return "unmeasured", "legacy_proposal"                       # 1. missing/None replay_inputs: an M3 proposal
    if proposal.get("replay_status") != "byte_identical":
        return "unmeasured", "recorded_unmeasured"                   # 2. the record itself was never measured
    for raw_count in (inputs.get("results_count"), inputs.get("launches_count")):
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
            return "unmeasured", "legacy_proposal"
    results_count, launches_count = inputs["results_count"], inputs["launches_count"]
    results_prefix = list(results or ())
    launches_prefix = list(launches or ())
    if len(results_prefix) < results_count or len(launches_prefix) < launches_count:
        return "unmeasured", "ledger_prefix_missing"                 # 3. the append-only ledger lost its prefix
    who = proposal.get("proposer")
    name = who.get("name") if isinstance(who, dict) else None
    seed = proposal.get("seed")
    if not isinstance(name, str) or "k" not in proposal or "cards" not in proposal:
        return "unmeasured", "legacy_proposal"
    if not isinstance(seed, int) or isinstance(seed, bool):
        return "unmeasured", "legacy_proposal"                       # a seedless record cannot pin the draw: never default it
    k = proposal["k"]
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        return "unmeasured", "legacy_proposal"                       # select() records a sanitised int k: anything else is junk
    try:
        rebuilt = get_proposer(name, seed, registry)                 # a missing extra (ImportError) is None
    except Exception:
        return "unmeasured", "model_error"                           # a present but broken extra is not a missing one
    if rebuilt is None:
        return "unmeasured", f"extra_missing:{name}"                 # 5. the proposer itself is gone
    if package_version(rebuilt) != proposal.get("package_version"):
        return "unmeasured", f"version_changed:{name}"               # 5. a rebuild would be a different build
    try:  # a broken extra is an unmeasured claim (C6 step 6), never a crash
        fresh, _rebuilt_drops = rebuilt.propose(
            spec,
            results_prefix[:results_count],
            launches_prefix[:launches_count],
            copy.deepcopy(inputs.get("current")),                   # the ledger record is never handed out by reference
            copy.deepcopy(inputs.get("symptoms")),
            k=k,
        )
    except Exception:
        return "unmeasured", "model_error"                           # 6. the rebuild cannot speak any more
    if name == DEFAULT_PROPOSER:
        cards = fresh or []                                          # the catalog: compare the raw output
    else:
        cards = [card for card in (fresh or []) if _card_in_axes(spec, card)]  # only the live cards were recorded
    try:
        same = canonical(cards) == canonical(proposal["cards"])
    except (TypeError, ValueError):
        return "unmeasured", "model_error"                           # 6. uncomparable fresh cards are not evidence
    return ("byte_identical", None) if same else ("drifted", None)
