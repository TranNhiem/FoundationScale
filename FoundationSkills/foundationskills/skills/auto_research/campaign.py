"""Campaign spec checks, the approval hash, derived launch tokens and the launch gates."""
from __future__ import annotations

import math
import re
from typing import Any

from foundationskills.skills.auto_research.ledger import canonical, sha256_hex

CLUSTER_TIME = "10-00:00:00"
CLUSTER_TIME_LIMIT_H = 240.0  # 10 days expressed in hours
EXCLUDED_NODES = ("r01dgx02",)
LOCKED_FIELDS = ("base", "model")
RESULT_ROLES = ("baseline", "candidate", "confirm")
GUARD_DIRECTIONS = ("max", "min")

SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")

# Knobs FoundationSkills can actually run today; kind is float | int | cat | str.
AXIS_PATHS: dict[str, str] = {
    "optim.lr": "float",
    "optim.warmup_ratio": "float",
    "optim.weight_decay": "float",
    "train.method": "cat",
    "lora.rank": "int",
    "lora.alpha": "int",
    "train.seq_len": "int",
    "train.global_batch": "int",
    "train.micro_batch": "int",
    "train.max_steps": "int",
    "train.epochs": "float",
    "rl.algorithm": "cat",
    "rl.group_size": "int",
    "rl.temperature": "float",
    "rl.kl_coef": "float",
    "data.mix": "str",
}

# Alternatives for the "cat" kinds (the "cat full|lora" / "cat dr_grpo|gspo|dapo" knobs).
AXIS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "train.method": ("full", "lora"),
    "rl.algorithm": ("dr_grpo", "gspo", "dapo"),
}

# M3 proposer block (AR-IN-008): which proposer may suggest axes and its floor rows.
PROPOSER_NAMES = ("catalog", "optuna", "optuna-cma", "llm")
_PROPOSER_KEYS = {"name", "min_rows", "require_model", "seed", "llm"}
_LLM_KEYS = ("pool_key", "model", "max_cards", "max_calls",
             "max_evidence_chars", "parse_spec_version", "sampling")

UNSUPPORTED_AXES: dict[str, str] = {
    "parallel.tp": "tensor parallelism is planned, not a tunable axis in the measured FS",
    "parallel.pp": "pipeline parallelism is planned, not a tunable axis in the measured FS",
    "parallel.ep": "expert parallelism is planned, not a tunable axis in the measured FS",
    "train.async": "async actor/learner plumbing is planned, not a tunable axis in the measured FS",
}

_FORBIDDEN_COMMANDS = (
    re.compile(r"\bpkill\s+(-\w+\s+)*-u\b"),
    re.compile(r"\bscancel\b"),
    re.compile(r"\bkillall\b"),
)


def campaign_hash(spec: dict[str, Any]) -> str:
    """The sha256 of the canonical campaign spec: exactly what a human approves once."""
    return sha256_hex(canonical(spec))


def launch_token(confirm: str, launch_spec: dict[str, Any]) -> str:
    """Per-launch token derived from the human confirm hash and the launch spec."""
    return sha256_hex(b"fs-ar-launch-v1|" + confirm.encode("utf-8") + b"|" + canonical(launch_spec))


def _block(spec: dict[str, Any], key: str) -> dict[str, Any]:
    """A spec block as a dict; a junk (non-mapping) block reads as empty so the checks refuse instead of crash."""
    value = spec.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        float(value)  # an int too large for float() is not a usable number
    except OverflowError:
        return False
    return True


def _finite(value: Any, default: float = 0.0) -> float:
    """``float(value)`` when it is a finite number, otherwise ``default`` (never raises)."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _count(value: Any, default: int = 0) -> int:
    """``value`` when it is a real int, otherwise ``default`` (never raises)."""
    return value if _is_int(value) else default


# ---- spec validation ------------------------------------------------------

def _check_axis(axis: dict[str, Any]) -> list[tuple[str, str]]:
    key = str(axis.get("key") or "")
    if key in UNSUPPORTED_AXES:
        return [("AR-IN-005", f"axis {key!r} is unsupported: {UNSUPPORTED_AXES[key]}")]
    if key not in AXIS_PATHS:
        return [("AR-IN-005", f"axis {key!r} is not an AXIS_PATHS knob")]
    kind, declared = AXIS_PATHS[key], str(axis.get("type") or "")
    problems: list[tuple[str, str]] = []
    if kind == "float":
        if declared not in {"float", "log_float"}:
            problems.append(("AR-IN-005", f"axis {key!r} is a float knob but declares type={declared!r}"))
        if not _is_num(axis.get("min")) or not _is_num(axis.get("max")):
            problems.append(("AR-IN-005", f"axis {key!r} needs numeric min/max"))
        elif not float(axis["min"]) < float(axis["max"]) or (declared == "log_float" and float(axis["min"]) <= 0.0):
            problems.append(("AR-IN-005", f"axis {key!r} range {axis.get('min')!r}..{axis.get('max')!r} is invalid"))
    elif kind == "int":
        if declared != "int":
            problems.append(("AR-IN-005", f"axis {key!r} is an int knob but declares type={declared!r}"))
        if not _is_int(axis.get("min")) or not _is_int(axis.get("max")) or int(axis["min"]) >= int(axis["max"]):
            problems.append(
                ("AR-IN-005", f"axis {key!r} needs integer min < max (got {axis.get('min')!r}, {axis.get('max')!r})")
            )
    else:  # cat | str
        if declared != "categorical":
            problems.append(("AR-IN-005", f"axis {key!r} is a {kind} knob but declares type={declared!r}"))
        values = list(axis.get("values") or [])
        if not values or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
            problems.append(("AR-IN-005", f"axis {key!r} needs a unique non-empty string `values` list"))
        elif kind == "cat":
            allowed = AXIS_CATEGORIES.get(key, ())
            bad = [v for v in values if v not in allowed]
            if bad:
                problems.append(("AR-IN-005", f"axis {key!r} values {bad} outside {list(allowed)}"))
    return problems


# ---- M2 spec shims (AR-IN-007) -------------------------------------------
#
# Inline here on purpose: campaign.check_spec owns AR-IN-007 (seeds.py has no spec validator).
# Inspected: M1's check_spec below covered NO seed fields at all - AR-IN-002 is base,
# AR-IN-003 is budget, AR-IN-004 is the eval fingerprint, AR-IN-005 is axes - so no existing rule
# already covers these cases and nothing is left double-reported. Requiredness of seeds.seed_list /
# seeds.confirm_repeats stays on the AR-IN required-field rule ("required, already").


def _check_max_in_flight(spec: dict[str, Any], max_runs: Any) -> list[tuple[str, str]]:
    """AR-IN-007: ``cluster.max_in_flight`` is optional, but binds as an int >= 1 <= ``budget.max_runs``."""
    raw_cluster = spec.get("cluster")
    cluster = dict(raw_cluster) if isinstance(raw_cluster, dict) else {}
    value = cluster.get("max_in_flight")
    if value is None:
        return []  # absent (or None): the cap resolves to cluster.max_nodes downstream
    if not _is_int(value) or int(value) < 1:
        return [("AR-IN-007", f"max_in_flight_invalid: cluster.max_in_flight must be an int >= 1 (got {value!r})")]
    if _is_int(max_runs) and int(value) > int(max_runs):
        return [
            ("AR-IN-007", f"max_in_flight_over_max_runs: cluster.max_in_flight {value} > budget.max_runs {max_runs}")
        ]
    return []


def _check_seeds(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """AR-IN-007 seed-plan shape: unique-int ``seed_list`` + both repeat counters (values only, see above)."""
    seeds = spec.get("seeds")
    if seeds is None:
        return []  # no seeds block: the AR-IN required-field rule owns the absence
    if not isinstance(seeds, dict):
        return [("AR-IN-007", f"seed_plan_invalid: seeds must be an object (got {seeds!r})")]
    problems: list[tuple[str, str]] = []
    raw_list = seeds.get("seed_list")
    if raw_list is not None and not isinstance(raw_list, (list, tuple)):
        problems.append(
            ("AR-IN-007", f"seed_list_invalid: seeds.seed_list must be a list of unique ints (got {raw_list!r})")
        )
        raw_list = None
    values = list(raw_list or [])
    if raw_list is not None and not values:
        problems.append(("AR-IN-007", "seed_list_empty: seeds.seed_list must be a non-empty list"))
    junk = [s for s in values if not _is_int(s)]
    if junk:
        problems.append(("AR-IN-007", f"seed_list_nonint: seeds.seed_list must hold ints, not bools (got {junk!r})"))
    ints = [s for s in values if _is_int(s)]
    dupes = sorted({s for s in ints if ints.count(s) > 1})
    if dupes:
        problems.append(("AR-IN-007", f"seed_list_duplicate: seeds.seed_list repeats {dupes}"))
    confirm = seeds.get("confirm_repeats")
    if confirm is not None:
        if not _is_int(confirm) or int(confirm) < 1:
            problems.append(
                ("AR-IN-007", f"confirm_repeats_invalid: seeds.confirm_repeats must be an int >= 1 (got {confirm!r})")
            )
        elif values and int(confirm) > len(values):
            problems.append(
                ("AR-IN-007",
                 f"confirm_repeats_over_seed_list: seeds.confirm_repeats {confirm} > len(seed_list) {len(values)}")
            )
    screening = seeds.get("screening_repeats")
    if screening is not None and (not _is_int(screening) or int(screening) < 1):
        problems.append(
            ("AR-IN-007", f"screening_repeats_invalid: seeds.screening_repeats must be an int >= 1 (got {screening!r})")
        )
    return problems


# ---- M3 proposer config (AR-IN-008) ---------------------------------------


def check_proposer_spec(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """AR-IN-008: optional proposer block shape (values only); absent/None -> no findings."""
    raw = spec.get("proposer")
    if raw is None:
        return []  # absent (or None): every key takes its downstream default
    if not isinstance(raw, dict):
        return [("AR-IN-008", f"proposer must be a mapping (got {type(raw).__name__})")]
    problems: list[tuple[str, str]] = []
    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or name not in PROPOSER_NAMES):
        problems.append(("AR-IN-008", f"proposer.name {name!r} not in {list(PROPOSER_NAMES)}"))
    min_rows = raw.get("min_rows")
    if name == "llm":  # the llm proposer is the <=10-row proposer: min_rows may be 0
        if min_rows is not None and (not _is_int(min_rows) or int(min_rows) < 0):
            problems.append(("AR-IN-008", f"proposer.min_rows must be an int >= 0 (got {min_rows!r})"))
    elif min_rows is not None and (not _is_int(min_rows) or int(min_rows) < 1):
        problems.append(("AR-IN-008", f"proposer.min_rows must be an int >= 1 (got {min_rows!r})"))
    require_model = raw.get("require_model")
    if require_model is not None and not isinstance(require_model, bool):
        problems.append(("AR-IN-008", f"proposer.require_model must be a bool (got {require_model!r})"))
    seed = raw.get("seed")
    if seed is not None and not _is_int(seed):
        problems.append(("AR-IN-008", f"proposer.seed must be an int (got {seed!r})"))
    extra = {str(k) for k in set(raw) - _PROPOSER_KEYS}
    if extra:
        problems.append(("AR-IN-008", f"proposer has unknown key(s) {sorted(extra)}"))
    if name != "llm" and "llm" in raw:
        problems.append(("AR-IN-008", "proposer.llm is only valid with name 'llm'"))
    if name == "optuna-cma":  # AR-PR-002 (C3): CMA-ES is numeric-only - a categorical axis REFUSES, never drops
        for axis in spec.get("axes") or []:
            if isinstance(axis, dict) and axis.get("type") == "categorical":
                problems.append(("AR-PR-002", f"proposer_axis_unsupported:optuna-cma:{axis.get('key')}"))
    return problems


def check_llm_spec(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """AR-IN-010: llm proposer config block; findings only for proposer.name == 'llm'."""
    raw = spec.get("proposer")
    if not isinstance(raw, dict) or raw.get("name") != "llm":
        return []
    llm = raw.get("llm")
    if not isinstance(llm, dict):  # missing or not a mapping: stop here
        return [("AR-IN-010", "llm_config:block")]
    problems: list[tuple[str, str]] = []
    pool_key = llm.get("pool_key")
    if not isinstance(pool_key, str) or not pool_key:
        problems.append(("AR-IN-010", "llm_config:pool_key"))
    model = llm.get("model")
    if not isinstance(model, str) or not model:
        problems.append(("AR-IN-010", "llm_config:model"))
    budget = _block(spec, "budget")
    max_runs = budget.get("max_runs")
    if "max_cards" in llm:
        max_cards = llm["max_cards"]
        bad = not _is_int(max_cards) or int(max_cards) < 1
        if not bad and _is_int(max_runs) and int(max_cards) > int(max_runs):  # bools are not ints
            bad = True
        if bad:
            problems.append(("AR-IN-010", "llm_config:max_cards"))
    for key in ("max_calls", "max_evidence_chars", "parse_spec_version"):
        if key in llm and (not _is_int(llm[key]) or int(llm[key]) < 1):
            problems.append(("AR-IN-010", f"llm_config:{key}"))
    if "sampling" in llm and not isinstance(llm["sampling"], dict):
        problems.append(("AR-IN-010", "llm_config:sampling"))
    unknown = sorted({str(key) for key in llm} - set(_LLM_KEYS))
    if unknown:
        problems.append(("AR-IN-010", f"llm_config:unknown:{unknown}"))
    for field in ("pool_key", "model"):
        value = llm.get(field)
        if isinstance(value, str) and "://" in value:  # a URL never belongs in the spec
            problems.append(("AR-IN-010", f"llm_config:url_shaped:{field}"))
    return problems


def check_objectives_spec(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """AR-IN-009: optional multi-objectives block (M5a); absent/None -> no findings (M4 path untouched)."""
    raw = spec.get("objectives")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [("AR-IN-009", f"objectives_count:{type(raw).__name__}")]
    if not 2 <= len(raw) <= 4:
        return [("AR-IN-009", f"objectives_count:{len(raw)}")]
    problems: list[tuple[str, str]] = []
    good: list[int] = []
    for index, entry in enumerate(raw):
        shape_ok = (
            isinstance(entry, dict)
            and set(entry) == {"metric", "direction"}
            and isinstance(entry.get("metric"), str)
            and bool(entry.get("metric"))
            and entry.get("direction") in GUARD_DIRECTIONS
        )
        if shape_ok:
            good.append(index)
        else:
            problems.append(("AR-IN-009", f"objective_shape:{index}"))
    if 0 in good:  # objectives[0] must agree with objective (direction defaults to "max")
        first = raw[0]
        primary = _block(spec, "objective")
        if (first.get("metric"), first.get("direction")) != (
            primary.get("metric"), primary.get("direction", "max")
        ):
            problems.append(("AR-IN-009", f"objectives0_mismatch:{first.get('metric')}"))
    seen: set[str] = set()
    dupes: set[str] = set()
    distinct: list[str] = []
    for index in good:
        metric = raw[index]["metric"]
        if metric in seen:
            if metric not in dupes:
                problems.append(("AR-IN-009", f"objective_metric_duplicate:{metric}"))
                dupes.add(metric)
        else:
            seen.add(metric)
            distinct.append(metric)
    metrics = [str(m) for m in (_block(spec, "eval_policy").get("metrics") or [])]
    for metric in distinct:
        if metric not in metrics:
            problems.append(("AR-IN-009", f"objective_not_in_eval_policy:{metric}"))
    guards = _block(spec, "confirm").get("guardrails") or {}
    names = {str(g) for g in guards}
    for metric in distinct:
        if metric in names:
            problems.append(("AR-IN-009", f"guardrail_is_objective:{metric}"))
    return problems


def check_spec(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """(rule_id, message) problems in the campaign spec; [] means the spec is launchable."""
    problems: list[tuple[str, str]] = []
    objective = _block(spec, "objective")
    eval_policy = _block(spec, "eval_policy")
    base = _block(spec, "base")
    confirm = _block(spec, "confirm")
    budget = _block(spec, "budget")
    metrics = [str(m) for m in (eval_policy.get("metrics") or [])]
    metric = objective.get("metric")
    if not isinstance(metric, str) or not metric or metric not in metrics:
        problems.append(("AR-IN-001", f"objective.metric {metric!r} missing or not in eval_policy.metrics {metrics}"))
    direction = objective.get("direction")
    if "direction" in objective and direction not in GUARD_DIRECTIONS:
        problems.append(
            ("AR-IN-001", f"objective.direction must be one of "
                          f"{list(GUARD_DIRECTIONS)} (got {direction!r})")
        )
    guard_directions = confirm.get("guardrail_directions") or {}
    for name, mode in dict(guard_directions).items():
        if str(mode) not in GUARD_DIRECTIONS:
            problems.append(
                ("AR-IN-001", f"confirm.guardrail_directions[{name!r}] must be one of "
                              f"{list(GUARD_DIRECTIONS)} (got {mode!r})")
            )
    if not isinstance(base.get("model"), str) or not base.get("model"):
        problems.append(("AR-IN-002", f"base.model missing or empty: {base.get('model')!r}"))
    if not SHA256_REF.match(str(base.get("fingerprint") or "")):
        problems.append(("AR-IN-002", f"base.fingerprint is not 'sha256:<64 hex>': {base.get('fingerprint')!r}"))
    hours, runs = budget.get("gpu_hours_total"), budget.get("max_runs")
    if not _is_num(hours) or float(hours) <= 0 or not _is_int(runs) or int(runs) <= 0:
        problems.append(("AR-IN-003", f"budget.gpu_hours_total/max_runs must be > 0 (got {hours!r}, {runs!r})"))
    reserve_frac_raw = budget.get("reserve_frac")
    if reserve_frac_raw is not None:
        # Present: must be a finite non-bool number strictly in [0, 1). Absent is fine (default 0.3 downstream).
        frac_ok = (
            _is_num(reserve_frac_raw)
            and math.isfinite(float(reserve_frac_raw))
            and 0.0 <= float(reserve_frac_raw) < 1.0
        )
        if not frac_ok:
            problems.append((
                "AR-IN-003",
                f"reserve_frac_out_of_range: budget.reserve_frac must be in [0, 1) (got {reserve_frac_raw!r})",
            ))
    if not SHA256_REF.match(str(eval_policy.get("fingerprint") or "")):
        problems.append(("AR-IN-004", f"eval_policy.fingerprint malformed: {eval_policy.get('fingerprint')!r}"))
    for axis in (spec.get("axes") or []):
        problems.extend(_check_axis(dict(axis or {})))
    timeout = budget.get("per_run_timeout_h")
    if not _is_num(timeout) or float(timeout) <= 0 or float(timeout) > CLUSTER_TIME_LIMIT_H:
        problems.append(
            ("AR-LN-005", f"budget.per_run_timeout_h must be in (0, {CLUSTER_TIME_LIMIT_H}] (got {timeout!r})")
        )
    cluster = _block(spec, "cluster")
    if cluster.get("time") != CLUSTER_TIME:
        problems.append(("AR-LN-004", f"cluster.time must be exactly {CLUSTER_TIME!r} (got {cluster.get('time')!r})"))
    excluded = [str(x) for x in (cluster.get("exclude") or [])]
    missing = [node for node in EXCLUDED_NODES if node not in excluded]
    if missing:
        problems.append(("AR-LN-004", f"cluster.exclude must contain quarantined node(s) {missing}"))
    # AR-IN-007 (M2): run-cap and seed-plan shape, in the same (rule_id, message) finding shape.
    problems.extend(_check_max_in_flight(spec, runs))
    problems.extend(_check_seeds(spec))
    problems.extend(check_proposer_spec(spec))
    # AR-IN-009 (M5a): the optional objectives block validates only when present (M4 path is byte-identical).
    problems.extend(check_objectives_spec(spec))
    problems.extend(check_llm_spec(spec))
    return problems


# ---- launch validation ----------------------------------------------------

def axis_value_fits(axis: dict[str, Any], value: Any) -> bool:
    """True when ``value`` is inside the axis's declared range/values for its kind."""
    declared = str(axis.get("type") or "")
    try:
        if declared == "categorical":
            return value in list(axis.get("values") or [])
        if declared == "int":
            return _is_int(value) and int(axis["min"]) <= int(value) <= int(axis["max"])
        if declared in {"float", "log_float"}:
            return _is_num(value) and float(axis["min"]) <= float(value) <= float(axis["max"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return False


def _node_names(named: Any) -> str:
    return " ".join(str(x) for x in named) if isinstance(named, list) else str(named or "")


def check_launch(
    spec: dict[str, Any], launch_spec: dict[str, Any], ledger_entries_payloads: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Launch gates against the campaign spec and the already-authorised launch payloads."""
    problems: list[tuple[str, str]] = []
    axes = {str(a.get("key") or ""): dict(a) for a in (spec.get("axes") or [])}
    cluster = _block(spec, "cluster")
    budget = _block(spec, "budget")
    prior = list(ledger_entries_payloads or [])

    for key, value in (launch_spec.get("delta") or {}).items():
        if key not in axes:
            problems.append(("AR-LN-001", f"delta key {key!r} is outside spec.axes {sorted(axes)}"))
        elif not axis_value_fits(axes[key], value):
            low, high = axes[key].get("min"), axes[key].get("max")
            problems.append(("AR-LN-001", f"delta {key}={value!r} is outside {low!r}..{high!r}"))
    nodes, gpus = launch_spec.get("nodes"), launch_spec.get("gpus_per_node")
    if not _is_int(nodes) or not 1 <= int(nodes) <= _count(cluster.get("max_nodes")):
        problems.append(("AR-LN-001", f"nodes {nodes!r} outside 1..{cluster.get('max_nodes')}"))
    if not _is_int(gpus) or not 1 <= int(gpus) <= _count(cluster.get("gpus_per_node")):
        problems.append(("AR-LN-001", f"gpus_per_node {gpus!r} outside 1..{cluster.get('gpus_per_node')}"))
    if str(launch_spec.get("partition") or "") != str(cluster.get("partition") or ""):
        problems.append(
            ("AR-LN-001", f"partition {launch_spec.get('partition')!r} differs from {cluster.get('partition')!r}")
        )
    role = str(launch_spec.get("role") or "")
    if role not in RESULT_ROLES:
        problems.append(("AR-LN-001", f"role {role!r} must be one of {list(RESULT_ROLES)}"))
    for locked in LOCKED_FIELDS:
        if locked in launch_spec:
            problems.append(("AR-LN-001", f"launch_spec carries locked field {locked!r}; the base model is sealed"))

    raw_est = launch_spec.get("gpu_hours_est")
    est = _finite(raw_est)
    est_ok = _is_num(raw_est) and est > 0.0
    if not est_ok:
        problems.append(("AR-LN-001", f"gpu_hours_est must be a positive number (got {raw_est!r})"))
    used = sum(_finite(p.get("gpu_hours_est")) for p in prior)  # junk in already-ledgered rows counts as 0
    total = _finite(budget.get("gpu_hours_total"))
    reserve_frac = _finite(budget.get("reserve_frac", 0.3), 0.3)
    max_runs = _count(budget.get("max_runs"))
    if max_runs > 0 and len(prior) >= max_runs:
        problems.append(("AR-LN-002", f"budget.max_runs reached ({len(prior)} authorised)"))
    if est_ok:  # budget and wall-time math only over a real estimate; junk never raises here
        if used + est > total:
            problems.append(("AR-LN-002", f"gpu hour budget exceeded: used {used} + est {est} > total {total}"))
        if role != "confirm" and used + est > (1.0 - reserve_frac) * total:
            problems.append(
                ("AR-LN-002", f"a {role!r} launch would dip into the confirm reserve "
                              f"({used + est} > {(1.0 - reserve_frac) * total})")
            )

    if str(launch_spec.get("time") or "") != CLUSTER_TIME:
        problems.append(("AR-LN-004", f"launch time must be exactly {CLUSTER_TIME!r} (got {launch_spec.get('time')!r})"))
    commands = [str(c) for c in (launch_spec.get("commands") or [])]
    for command in commands:
        for pattern in _FORBIDDEN_COMMANDS:
            if pattern.search(command):
                problems.append(("AR-LN-004", f"forbidden command {command!r} matches {pattern.pattern}"))
    named = _node_names(launch_spec.get("nodelist")) + " " + " ".join(commands)
    hits = [node for node in EXCLUDED_NODES if node in named]
    if hits:
        problems.append(("AR-LN-004", f"node list/command names quarantined node(s) {hits}"))

    timeout = _finite(budget.get("per_run_timeout_h"))
    slots = _count(nodes) * _count(gpus)
    if est_ok and timeout > 0.0 and slots > 0 and est / slots > timeout:
        problems.append(
            ("AR-LN-005", f"gpu_hours_est {est} implies {est / slots}h wall time > per_run_timeout_h {timeout}")
        )
    return problems


# ---- rendered/command scan (G2: gate what actually runs) ------------------

_FOLD = re.compile(r"[^\w+\-]+")  # json punctuation and whitespace all read as one separator
_EXCLUDE_OPTION = re.compile(r"[\w-]*exclude[\w-]*\s*[=:]?\s*\S*", re.IGNORECASE)


def _flatten_text(value: Any, out: list[str]) -> None:
    """Every string leaf of a nested payload (argv lines, sbatch blocks, opaque requests), in order."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten_text(item, out)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _flatten_text(item, out)
    elif value is not None:
        out.append(str(value))


def scan_command_text(text: str) -> list[tuple[str, str]]:
    """AR-LN-004 hits over free command text: an argv line, an sbatch body or ``json.dumps(train_request)``.

    The text is separator/JSON-punctuation folded first so ``["pkill","-u","x"]``, ``["pkill -u x"]`` and
    ``"pkill -u x"`` all read the same to the SAME forbidden-command patterns ``check_launch`` uses (a
    command and its JSON serialisation can never disagree about what is refused). Pure: it parses and
    executes nothing.
    """
    folded = _FOLD.sub(" ", str(text or ""))
    return [
        ("AR-LN-004", f"AR-LN-004: forbidden command {folded.strip()!r} matches {pattern.pattern}")
        for pattern in _FORBIDDEN_COMMANDS
        if pattern.search(folded)
    ]


def scan_rendered(spec: Any, text: str) -> list[tuple[str, str]]:
    """(rule_id, message) hits over a RENDERED fs_launch_spec (``,``.join(argv) + sbatch): what runs.

    ``spec`` is the rendered fs_launch_spec (its string leaves - the ``" ".join(argv)`` argv included - are
    scanned) and ``text`` the rendered sbatch body. The render, not the trial_spec, carries what actually
    runs, so its strings are scanned with the SAME AR-LN-004 forbidden-command patterns and the SAME
    AR-LN-005 quarantined/excluded node names ``check_launch`` refuses. An ``exclude`` option NAMES the node
    to skip and is therefore never a hit.
    """
    parts: list[str] = []
    _flatten_text(spec, parts)
    _flatten_text(text, parts)
    rendered = _FOLD.sub(" ", " ".join(parts))
    problems = scan_command_text(rendered)
    hits = [node for node in EXCLUDED_NODES if node in _EXCLUDE_OPTION.sub(" ", rendered)]
    if hits:
        problems.append(("AR-LN-005", f"AR-LN-005: rendered sbatch names quarantined node(s) {hits}"))
    return problems
