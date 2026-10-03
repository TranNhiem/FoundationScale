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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any, default: float = 0.0) -> float:
    """``float(value)`` when it is a finite number, otherwise ``default`` (never raises)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
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


def check_spec(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """(rule_id, message) problems in the campaign spec; [] means the spec is launchable."""
    problems: list[tuple[str, str]] = []
    objective = dict(spec.get("objective") or {})
    eval_policy = dict(spec.get("eval_policy") or {})
    base = dict(spec.get("base") or {})
    confirm = dict(spec.get("confirm") or {})
    budget = dict(spec.get("budget") or {})
    metrics = [str(m) for m in (eval_policy.get("metrics") or [])]
    metric = objective.get("metric")
    if not isinstance(metric, str) or not metric or metric not in metrics:
        problems.append(("AR-IN-001", f"objective.metric {metric!r} missing or not in eval_policy.metrics {metrics}"))
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
    if not SHA256_REF.match(str(eval_policy.get("fingerprint") or "")):
        problems.append(("AR-IN-004", f"eval_policy.fingerprint malformed: {eval_policy.get('fingerprint')!r}"))
    for axis in (spec.get("axes") or []):
        problems.extend(_check_axis(dict(axis or {})))
    timeout = budget.get("per_run_timeout_h")
    if not _is_num(timeout) or float(timeout) <= 0 or float(timeout) > CLUSTER_TIME_LIMIT_H:
        problems.append(
            ("AR-LN-005", f"budget.per_run_timeout_h must be in (0, {CLUSTER_TIME_LIMIT_H}] (got {timeout!r})")
        )
    cluster = dict(spec.get("cluster") or {})
    if cluster.get("time") != CLUSTER_TIME:
        problems.append(("AR-LN-004", f"cluster.time must be exactly {CLUSTER_TIME!r} (got {cluster.get('time')!r})"))
    excluded = [str(x) for x in (cluster.get("exclude") or [])]
    missing = [node for node in EXCLUDED_NODES if node not in excluded]
    if missing:
        problems.append(("AR-LN-004", f"cluster.exclude must contain quarantined node(s) {missing}"))
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
    except (KeyError, TypeError, ValueError):
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
    cluster = dict(spec.get("cluster") or {})
    budget = dict(spec.get("budget") or {})
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
