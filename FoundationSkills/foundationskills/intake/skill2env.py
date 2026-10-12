"""Import a skill-2-environment corpus and its licence manifest (B1): ``fskills.skill2env_import/1``.

This module is the B1 importer: the boundary between a third-party skill corpus and the
FoundationSkills acceptance gate. It reads a corpus of task directories (``task.toml`` +
instruction + environment + solution + tests), joins every task against the bundle's
licence manifest and -- given the report inputs -- records the licence verdict, the
Dockerfile preflight and the two acceptance gates (oracle and no-op) per task as
``fskills.skill2env_import/1``, the schema the import report is written as.

Inputs
------
A corpus root is a directory of task directories, each containing ``task.toml``,
``instruction.md``, ``environment/Dockerfile``, ``solution/solve.sh`` and ``tests/test.sh``.
``task.toml`` names the task, the ``source_skill`` (``metadata.source_skill``, optionally
``<prefix>/<skill_id>``), the bundle digest and the ``metadata.base_image_pins`` map
(image reference -> ``sha256:...``) along with the network modes and timeouts of the
verifier, the agent and the environment build. The licence manifest is a CSV with header
``skill_group,skill_id,license,source_link,local_path``; the join key is ``source_skill``
with one leading ``<prefix>/`` stripped, equal to the CSV's ``skill_id``.

Verdicts and the doctrine of absence
------------------------------------
Statuses are exactly ``PASS``, ``RED``, ``UNMEASURED`` and ``REFUSED`` (exit codes
``0``, ``5``, ``95``, ``96``). The rule throughout is that absence of evidence is never
``PASS``: a task without a licence row is ``UNMEASURED``, a builder that cannot be imported
is ``UNMEASURED``, an exception anywhere in a check is ``REFUSED`` for that task and never
``PASS``, and a gate that was not run is ``UNMEASURED``. A task's verdict is the worst of
its inputs with order ``REFUSED > RED > UNMEASURED > PASS`` (:func:`task_verdict`), and
every non-PASS reason carries a rule id (``SE-SCAN``, ``SE-LIC``, ``SE-PRE``, ``SE-GATE``).

Licence
-------
An SPDX expression is split on ``AND`` and each trimmed component is checked against the
permissive allowlist :data:`ALLOWED_LICENSES`. Any ``EPL-*`` component is ``RED`` (weak
copyleft excluded); a component outside the allowlist is ``RED``; no row is
``UNMEASURED``; otherwise ``PASS``. ``CC-BY-4.0`` and ``OFL-1.1`` additionally mark the
skill :func:`needs_attribution <needs_attribution>` and are listed in
``attribution_required`` and rendered by :func:`render_notices`.

Dockerfile preflight
--------------------
:func:`dockerfile_preflight` enforces two rules: every ``FROM <ref>`` (``AS`` alias and
``scratch`` ignored) must be a key of ``base_image_pins`` (``SE-PRE-001``), and the
FoundationSkills builder must accept the file. The validator parameter is injectable for
tests; the default imports the FS sandbox builder and runs its parse + validate steps.
A builder ``SandboxRefusal`` is ``RED`` (``SE-PRE-002``); the builder being unimportable is
``UNMEASURED``; any other exception is ``REFUSED`` ("preflight crashed").

Acceptance gates
----------------
:func:`run_gate` shells out to Harbor (argv list, never a shell string) to run one trial
with an ``oracle`` or ``nop`` agent and then reads the single trial ``result.json`` under
the jobs directory (:func:`parse_harbor_trial`). The oracle gate is ``PASS`` only when every
numeric leaf of ``verifier_result.rewards`` equals 1 (any leaf below 1 is ``RED``: the
oracle did not reach full reward); the no-op gate is ``PASS`` only when the mean leaf is 0
(anything above 0 is ``RED``: the verifier is not discriminating). Missing rewards or a
recorded exception is ``UNMEASURED`` (an aborted trial is not a result).

The report
----------
:func:`build_import_report` renders the deterministic (sorted, timestamp-free) report dict
and :func:`render_notices` its third-party notices section. :func:`write_json` is atomic
(tmp + replace). This module is stdlib + ``tomllib`` only (Python 3.11) and never touches a
network.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import tomllib

EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSED = 96

PASS = "PASS"
RED = "RED"
UNMEASURED = "UNMEASURED"
REFUSED = "REFUSED"

_REFUSAL_RANK: dict[str, int] = {REFUSED: 3, RED: 2, UNMEASURED: 1, PASS: 0}

REPORT_SCHEMA = "fskills.skill2env_import/1"

#: Permissive components we accept without further review.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "0BSD",
        "CC0-1.0",
        "Zlib",
        "CC-BY-4.0",
        "OFL-1.1",
        "Apache-1.1",
    }
)

#: Components that additionally require a THIRD-PARTY_NOTICES attribution entry.
ATTRIBUTION_LICENSES: frozenset[str] = frozenset({"CC-BY-4.0", "OFL-1.1"})

JOIN_PREFIXES: tuple[str, ...] = ("local/",)

REQUIRED_FILES: tuple[str, ...] = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
)

RULE_SCAN = "SE-SCAN-001"
RULE_LICENSE = "SE-LIC-001"
RULE_FROM_PIN = "SE-PRE-001"
RULE_BUILDER = "SE-PRE-002"
RULE_ORACLE = "SE-GATE-001"
RULE_NOP = "SE-GATE-002"


@dataclass(frozen=True)
class TaskRecord:
    """Everything the importer needs to know about one task directory.

    ``bundle_digest`` is the 64-hex ``source_bundle_digest`` from ``task.toml``,
    ``base_image_pins`` the ``(image_ref, sha256:...)`` pins with ``<ref> = <digest>`` from
    the same file in task-dirname order, and ``network_modes`` the triple
    ``(verifier, agent, environment)``.
    """

    dirname: str
    name: str
    source_skill: str
    skill_id: str
    bundle_digest: str
    base_image_pins: tuple[tuple[str, str], ...]
    agent_timeout_s: float | None
    verifier_timeout_s: float | None
    network_modes: tuple[str, str, str]
    has_solution: bool
    has_tests: bool
    instruction_sha256: str


@dataclass(frozen=True)
class GateResult:
    """The outcome of one acceptance-gate trial (an ``oracle`` or ``nop`` run).

    ``rewards`` is the trial's ``verifier_result.rewards`` leaf map (or ``None`` when the
    trial recorded none), ``exception`` the ``exception_info.exception_type`` of an aborted
    trial (or ``None``) and ``status`` one of ``PASS``/``RED``/``UNMEASURED``/``REFUSED``.
    """

    agent: str
    rewards: dict[str, float] | None
    exception: str | None
    status: str


def _worst(statuses: Sequence[str]) -> str:
    """Return the highest-ranked status with ``REFUSED > RED > UNMEASURED > PASS``."""
    if not statuses:
        return UNMEASURED
    return max(statuses, key=lambda s: _REFUSAL_RANK[s])


def scan_skill_id(source_skill: str) -> str:
    """Strip one leading ``<prefix>/`` from ``metadata.source_skill`` to get the join key.

    ``"local/firebase-app-hosting-basics"`` becomes ``"firebase-app-hosting-basics"`` and
    a bare id passes through unchanged, so it matches the licence CSV's ``skill_id``.
    """
    for prefix in JOIN_PREFIXES:
        if source_skill.startswith(prefix):
            return source_skill[len(prefix) :]
    return source_skill


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_failure(task: str, reason: str) -> dict[str, str]:
    return {"task": task, "verdict": REFUSED, "reason": reason}


def scan_corpus(root: Path) -> tuple[list[TaskRecord], list[dict]]:
    """Scan ``root`` for task directories and return ``(records, scan_failures)``.

    Task directories are visited in sorted ``dirname`` order (deterministic). A directory
    missing any required file, with unparsable ``task.toml`` or with a required key gone,
    contributes a scan failure (``SE-SCAN-001`` later) instead of a record: a crash in this
    scan is a refusal for the task and never a record with a passing default.
    """
    records: list[TaskRecord] = []
    failures: list[dict] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        missing = [rel for rel in REQUIRED_FILES if not (entry / rel).is_file()]
        if missing:
            failures.append(
                _scan_failure(entry.name, f"missing required file(s): {', '.join(missing)}")
            )
            continue
        try:
            data = tomllib.loads((entry / "task.toml").read_text(encoding="utf-8"))
            task = data["task"]
            metadata = data["metadata"]
            name = str(task["name"])
            source_skill = str(metadata["source_skill"])
            digest_raw = metadata["source_bundle_digest"]
            if not isinstance(digest_raw, str) or len(digest_raw) != 64 or any(
                c not in "0123456789abcdef" for c in digest_raw.lower()
            ):
                failures.append(
                    _scan_failure(
                        entry.name,
                        "metadata.source_bundle_digest must be 64 hex characters",
                    )
                )
                continue
            pins_raw = metadata.get("base_image_pins", {})
            base_image_pins = tuple(
                (str(k), str(v)) for k, v in sorted(dict(pins_raw).items())
            )
            agent_timeout_s = _opt_float(data.get("agent", {}).get("timeout_sec"))
            verifier_timeout_s = _opt_float(data.get("verifier", {}).get("timeout_sec"))
            network_modes = (
                str(data.get("verifier", {}).get("network_mode", "")),
                str(data.get("agent", {}).get("network_mode", "")),
                str(data.get("environment", {}).get("network_mode", "")),
            )
            instruction_sha256 = _sha256_file(entry / "instruction.md")
        except (tomllib.TOMLDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            failures.append(
                _scan_failure(entry.name, f"task.toml unreadable or incomplete: {exc!r}")
            )
            continue
        records.append(
            TaskRecord(
                dirname=entry.name,
                name=name,
                source_skill=source_skill,
                skill_id=scan_skill_id(source_skill),
                bundle_digest=digest_raw,
                base_image_pins=base_image_pins,
                agent_timeout_s=agent_timeout_s,
                verifier_timeout_s=verifier_timeout_s,
                network_modes=network_modes,
                has_solution=True,
                has_tests=True,
                instruction_sha256=instruction_sha256,
            )
        )
    return records, failures


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def load_licenses(csv_path: Path) -> dict[str, str]:
    """Read the licence manifest CSV and return ``skill_id -> SPDX expression``.

    Rows without a ``skill_id`` or ``license`` cell are skipped (their skill then joins as
    "missing"). Duplicates last-write-wins in file order.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        mapping: dict[str, str] = {}
        for row in reader:
            skill_id = (row.get("skill_id") or "").strip()
            expr = (row.get("license") or "").strip()
            if skill_id and expr:
                mapping[skill_id] = expr
    return mapping


def load_licenses_full(csv_path: Path) -> dict[str, tuple[str, str]]:
    """Like :func:`load_licenses` but returns ``skill_id -> (license, source_link)``.

    This is what :func:`render_notices` needs to cite a source link per attribution.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        mapping: dict[str, tuple[str, str]] = {}
        for row in reader:
            skill_id = (row.get("skill_id") or "").strip()
            expr = (row.get("license") or "").strip()
            link = (row.get("source_link") or "").strip()
            if skill_id and expr:
                mapping[skill_id] = (expr, link)
    return mapping


def _split_components(expr: str) -> list[str]:
    raw = expr.replace("(", "").replace(")", "")
    return [part.strip() for part in raw.split(" AND ") if part.strip()]


def license_verdict(expr: str | None) -> tuple[str, str]:
    """Judge an SPDX expression and return ``(status, reason)``.

    The expression is split on ``AND`` (parentheses ignored). Any ``EPL-*`` component is
    ``RED`` ("weak copyleft EPL excluded"); any other component outside
    :data:`ALLOWED_LICENSES` is ``RED`` naming the offending component. A missing row /
    expression is ``UNMEASURED`` ("no license row for skill") -- never ``PASS``. An empty
    component list (``None`` truthiness aside) is also ``UNMEASURED``.
    """
    if expr is None or not str(expr).strip():
        return (UNMEASURED, "no license row for skill")
    components = _split_components(str(expr))
    if not components:
        return (UNMEASURED, "no license row for skill")
    for comp in components:
        if comp.startswith("EPL-"):
            return (RED, "weak copyleft EPL excluded")
    for comp in components:
        if comp not in ALLOWED_LICENSES:
            return (RED, f"license component {comp} not on the allowlist")
    return (PASS, f"all components on the allowlist: {', '.join(components)}")


def needs_attribution(expr: str | None) -> bool:
    """True when the expression contains ``CC-BY-4.0`` or ``OFL-1.1`` (a notices entry)."""
    if expr is None:
        return False
    return any(comp in ATTRIBUTION_LICENSES for comp in _split_components(str(expr)))


def _dockerfile_from_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("FROM"):
            continue
        tokens = [t for t in stripped.split()[1:] if not t.startswith("--")]  # skip --platform=... flags
        if not tokens:
            continue
        ref = tokens[0]
        if "<" in ref or ref.lower() == "scratch":
            continue
        refs.append(ref)
    return refs


def _default_dockerfile_validator(
    text: str, context: Path
) -> None:
    """Use FoundationSkills' own sandbox builder to parse + validate the Dockerfile.

    Imports ``foundationscale.agentic_rl.sandbox.build`` lazily so this module stays
    deployable without it; a ``SandboxRefusal`` surfaces as ``RED``, the builder being
    absent as ``UNMEASURED`` and anything else as ``REFUSED``.
    """
    from foundationscale.agentic_rl.sandbox.build import (  # type: ignore[import-not-found]
        _validate_steps,
        parse_dockerfile,
    )

    steps = parse_dockerfile(text)
    _validate_steps(steps, {}, None, context)


def dockerfile_preflight(
    dockerfile: Path,
    pins: Mapping[str, str],
    validator: Callable[[str, Path], None] | None = None,
) -> tuple[str, str]:
    """Preflight one ``Dockerfile`` and return ``(status, reason)``.

    Rule one (``SE-PRE-001``): every ``from <ref>`` (alias clause ``AS name`` and ``scratch``
    ignored) must be a key of ``pins`` -- a reference without a digest pin is ``RED``.
    Rule two (``SE-PRE-002``): the FS builder accepts the file. The builder call is
    ``validator(text, dockerfile.parent)``; a sed-raised ``SandboxRefusal`` is ``RED``
    ("FS builder refuses: <msg>"), an unimportable builder is ``UNMEASURED`` ("FS builder
    unavailable"), any other exception is ``REFUSED`` ("preflight crashed: <repr>") and
    silence is ``PASS`` ("FS builder accepts").
    """
    text = dockerfile.read_text(encoding="utf-8")
    for ref in _dockerfile_from_refs(text):
        # real Skill2Env Dockerfiles write `FROM image:tag@sha256:<d>` while task.toml pins `image:tag` -> d:
        # pinned when the inline digest equals the task.toml pin (or the bare ref is pinned); a disagreeing
        # digest is RED, never silently accepted
        bare, _, inline = ref.partition("@")
        pinned = pins.get(bare) or pins.get(ref)
        if inline and pinned and inline != pinned:
            return (RED, f"FROM {ref} digest disagrees with the task.toml pin {pinned}")
        if not pinned and not (inline and inline in set(pins.values())):
            return (RED, f"FROM {ref} not digest-pinned in task.toml")
    call = validator if validator is not None else _default_dockerfile_validator
    try:
        call(text, dockerfile.parent)
    except BaseException as exc:  # noqa: BLE001 - a crash is a refusal, never a pass
        name = type(exc).__name__
        if name == "SandboxRefusal":
            return (RED, f"FS builder refuses: {exc}")
        mod = type(exc).__module__
        if isinstance(exc, ModuleNotFoundError) or name == "ModuleNotFoundError":
            return (UNMEASURED, "FS builder unavailable")
        if isinstance(exc, ImportError) and "foundationscale" in (getattr(exc, "name", "") or ""):
            return (UNMEASURED, "FS builder unavailable")
        if "foundationscale" in mod and name == "SandboxRefusal":
            return (RED, f"FS builder refuses: {exc}")
        return (REFUSED, f"preflight crashed: {exc!r}")
    return (PASS, "FS builder accepts")


def _numeric_leaves(rewards: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    for key, value in rewards.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            leaves.update(_numeric_leaves(value, prefix=f"{path}."))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            leaves[path] = float(value)
    return leaves


def parse_harbor_trial(result_json: Path, agent: str) -> GateResult:
    """Reduce one Harbor trial ``result.json`` to a :class:`GateResult`.

    The reward leaf map is ``verifier_result.rewards`` reduced to numeric leaves (a nested
    dict is flattened, a numeric leaf kept; the leaf reduction is the mean). The oracle
    gates PASS only at full reward (all leaves == 1), the no-op gates PASS only at zero
    mean; anything else is RED and names the rule. No rewards map leaves or a recorded
    ``exception_info.exception_type`` is UNMEASURED -- a trial that crashed or graded
    nothing is not evidence. An unreadable or malformed ``result.json`` is UNMEASURED with
    the parse problem named (a crash here would otherwise be a silent PASS).
    """
    try:
        raw = result_json.read_text(encoding="utf-8")
        data = __import__("json").loads(raw)
    except (OSError, ValueError) as exc:
        return GateResult(agent, None, None, UNMEASURED)
    extractor = data if isinstance(data, Mapping) else {}
    verifier_result = extractor.get("verifier_result")
    rewards_raw = (
        verifier_result.get("rewards")
        if isinstance(verifier_result, Mapping)
        else None
    )
    exception_info = extractor.get("exception_info")
    exception = (
        str(exception_info.get("exception_type"))
        if isinstance(exception_info, Mapping) and exception_info.get("exception_type")
        else None
    )
    rewards: dict[str, float] | None = None
    if isinstance(rewards_raw, Mapping):
        rewards = _numeric_leaves(rewards_raw)
        if not rewards:
            rewards = None
    if exception is not None:
        return GateResult(agent, rewards, exception, UNMEASURED)
    if rewards is None:
        return GateResult(
            agent, None, None, UNMEASURED,
        )
    if agent == "oracle":
        if all(value == 1.0 for value in rewards.values()):
            return GateResult(agent, rewards, None, PASS)
        return GateResult(agent, rewards, None, RED)
    if agent == "nop":
        mean = sum(rewards.values()) / len(rewards)
        if mean == 0:
            return GateResult(agent, rewards, None, PASS)
        return GateResult(agent, rewards, None, RED)
    return GateResult(agent, rewards, None, UNMEASURED)


def run_gate(
    task_dir: Path,
    *,
    harbor_bin: str,
    env_import_path: str,
    env_kwargs: Mapping[str, str],
    jobs_dir: Path,
    agent: str,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = 300.0,
) -> GateResult:
    """Run one acceptance gate (``oracle`` or ``nop``) and parse its single trial result.

    The argv list (never a shell string) is ``[harbor_bin, "run", "--path", <task_dir>,
    "--agent", <agent>, "--environment-import-path", <env_import_path>]`` followed by one
    ``--ek k=v`` per ``env_kwargs`` item and ``--jobs-dir <jobs_dir>``. The jobs directory
    must then contain exactly one ``*__*/result.json`` trial result; more than one is
    REFUSED (the run is ambiguous), none is UNMEASURED. A timeout or a process-level
    failure is UNMEASURED -- the gate did not run, and that is never evidence.
    """
    argv: list[str] = [
        harbor_bin,
        "run",
        "--path",
        str(task_dir),
        "--agent",
        agent,
        "--environment-import-path",
        env_import_path,
    ]
    for key in sorted(env_kwargs):
        argv.extend(["--ek", f"{key}={env_kwargs[key]}"])
    argv.extend(["--jobs-dir", str(jobs_dir)])
    try:
        run(argv, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return GateResult(agent, None, "TimeoutExpired", UNMEASURED)
    except OSError as exc:
        return GateResult(agent, None, type(exc).__name__, UNMEASURED)
    # Harbor nests a trial as <jobs_dir>/<timestamp>/<task>__<id>/result.json, and the timestamp dir
    # ("2026-10-12__13-00-11") also contains "__": identify the TRIAL result by structure (it sits beside the
    # trial's agent/ or verifier/ dir), never by name -- the job-level result.json carries no verifier_result
    results = sorted(p for p in jobs_dir.rglob("result.json")
                     if (p.parent / "agent").is_dir() or (p.parent / "verifier").is_dir())
    if len(results) > 1:
        return GateResult(
            agent,
            None,
            "multiple trials recorded",
            REFUSED,
        )
    if not results:
        return GateResult(agent, None, "no trial result recorded", UNMEASURED)
    return parse_harbor_trial(results[0], agent)


def task_verdict(
    rec: TaskRecord,
    license_status: str | tuple[str, str],
    preflight_status: str | tuple[str, str],
    gate: Mapping[str, GateResult] | None,
) -> tuple[str, list[str]]:
    """Reduce one task to ``(verdict, reasons)`` and attach rule ids to every reason.

    The verdict is the worst of licence, preflight, oracle and no-op. ``gate is None``
    counts as UNMEASURED ("acceptance gate not run") -- gates that were not run can never
    downgrade a task to PASS. Reasons are prefixed ``SE-LIC-00x`` (licence),
    ``SE-PRE-00x`` (pinning / builder), ``SE-GATE-00x`` (gates) and ``SE-SCAN-001`` (a scan
    failure), so every RED/UNMEASURED verdict can be traced to the rule that produced it.
    """
    reasons: list[str] = []
    # accept the (status, reason) pairs license_verdict / dockerfile_preflight return, or bare statuses
    license_reason = preflight_reason = ""
    if isinstance(license_status, tuple):
        license_status, license_reason = license_status
    if isinstance(preflight_status, tuple):
        preflight_status, preflight_reason = preflight_status
    if license_status != PASS:
        reasons.append(f"{RULE_LICENSE} {license_status}: {license_reason}".rstrip(": "))
    if preflight_status != PASS:
        rule = RULE_FROM_PIN if "digest-pinned" in preflight_reason else RULE_BUILDER
        reasons.append(f"{rule} {preflight_status}: {preflight_reason}".rstrip(": "))
    statuses: list[str] = [license_status, preflight_status]
    oracle = (gate or {}).get("oracle")
    nop = (gate or {}).get("nop")
    if gate is None:
        statuses.append(UNMEASURED)
        reasons.append(f"{RULE_ORACLE} acceptance gate not run")
        reasons.append(f"{RULE_NOP} acceptance gate not run")
    else:
        if oracle is None:
            statuses.append(UNMEASURED)
            reasons.append(f"{RULE_ORACLE} acceptance gate not run (oracle)")
        else:
            statuses.append(oracle.status)
            if oracle.status != PASS:
                reasons.append(
                    f"{RULE_ORACLE} {oracle.status}: "
                    + (oracle.exception or _gate_reason_oracle(oracle))
                )
        if nop is None:
            statuses.append(UNMEASURED)
            reasons.append(f"{RULE_NOP} acceptance gate not run (nop)")
        else:
            statuses.append(nop.status)
            if nop.status != PASS:
                reasons.append(
                    f"{RULE_NOP} {nop.status}: "
                    + (nop.exception or _gate_reason_nop(nop))
                )
    verdict = _worst(statuses)
    if license_status != PASS and not any(r.startswith(RULE_LICENSE) for r in reasons):
        reasons.append(f"{RULE_LICENSE} {license_status}")
    return (verdict, reasons)


def _gate_reason_oracle(result: GateResult) -> str:
    if result.status == RED:
        return "oracle did not reach full reward"
    return "oracle trial yielded no rewards"


def _gate_reason_nop(result: GateResult) -> str:
    if result.status == RED:
        return "no-op scored > 0: verifier is not discriminating"
    return "no-op trial yielded no rewards"


def build_import_report(
    records: Sequence[TaskRecord],
    scan_failures: Sequence[dict],
    licenses: Mapping[str, str],
    preflight: Mapping[str, tuple[str, str]],
    gates: Mapping[str, Mapping[str, GateResult]] | None,
    *,
    dataset: str,
    revision: str,
    operator_decision: str,
) -> dict:
    """Build the ``fskills.skill2env_import/1`` import report (deterministically ordered).

    Tasks are emitted in dirname order; counts and exclusion tallies are ``{status: n}`` /
    ``{"epl", "unlisted_license", "missing_license"}`` tallies over the licence verdicts
    and ``join_coverage`` is the fraction of tasks that found a licence row. The report
    carries no timestamps (rebuild the same inputs and get the same JSON).
    """
    tasks: list[dict] = []
    counts: dict[str, int] = {PASS: 0, RED: 0, UNMEASURED: 0, REFUSED: 0}
    license_counts: dict[str, int] = {}
    excluded = {"epl": 0, "unlisted_license": 0, "missing_license": 0}
    joined = 0
    attribution_required: list[str] = []
    preflight_report: dict[str, dict] = {}
    for rec in sorted(records, key=lambda r: r.dirname):
        expr = licenses.get(rec.skill_id)
        if expr is None:
            joined += 0
        else:
            joined += 1
        license_status, license_reason = license_verdict(expr)
        license_counts[expr if expr is not None else "<missing>"] = (
            license_counts.get(expr if expr is not None else "<missing>", 0) + 1
        )
        if license_status == RED:
            if "EPL" in license_reason:
                excluded["epl"] += 1
            else:
                excluded["unlisted_license"] += 1
        elif license_status == UNMEASURED:
            excluded["missing_license"] += 1
        pre_status, pre_reason = preflight.get(rec.dirname, (UNMEASURED, "preflight not run"))
        preflight_report[rec.dirname] = {"status": pre_status, "reason": pre_reason}
        gate_map = (gates or {}).get(rec.dirname)
        verdict, reasons = task_verdict(rec, (license_status, license_reason), (pre_status, pre_reason), gate_map)
        if needs_attribution(expr) and rec.skill_id not in attribution_required:
            attribution_required.append(rec.skill_id)
        counts[verdict] = counts.get(verdict, 0) + 1
        gate_entry: dict[str, dict] = {}
        if gate_map is not None:
            for agent in sorted(gate_map):
                result = gate_map[agent]
                gate_entry[agent] = {
                    "status": result.status,
                    "rewards": result.rewards,
                    "exception": result.exception,
                }
        tasks.append(
            {
                "dirname": rec.dirname,
                "source_skill": rec.source_skill,
                "bundle_digest": rec.bundle_digest,
                "base_image_pins": {ref: digest for ref, digest in rec.base_image_pins},
                "license": expr,
                "license_status": license_status,
                "preflight": {"status": pre_status, "reason": pre_reason},
                "gate": gate_entry,
                "verdict": verdict,
                "reasons": reasons,
            }
        )
    # a task that could not even be scanned is REFUSED (never silently dropped, never PASS)
    for failure in sorted(scan_failures, key=lambda f: str(f.get("task", ""))):
        counts[REFUSED] = counts.get(REFUSED, 0) + 1
        tasks.append({"dirname": str(failure.get("task", "")), "source_skill": None, "bundle_digest": None,
                      "base_image_pins": {}, "license": None, "license_status": UNMEASURED,
                      "preflight": {"status": UNMEASURED, "reason": "not scanned"}, "gate": {},
                      "verdict": REFUSED, "reasons": [f"{RULE_SCAN} {failure.get('reason', 'scan failure')}"]})
    tasks.sort(key=lambda t: t["dirname"])
    return {
        "schema": REPORT_SCHEMA,
        "dataset": dataset,
        "revision": revision,
        "operator_decision": operator_decision,
        "counts": counts,
        "license_counts": dict(sorted(license_counts.items())),
        "excluded": excluded,
        "join_coverage": (joined / len(tasks)) if tasks else 0.0,
        "attribution_required": sorted(attribution_required),
        "tasks": tasks,
    }


def render_notices(
    report: Mapping[str, Any],
    licenses_full: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Render the THIRD_PARTY_NOTICES markdown section for attribution-required skills.

    ``licenses_full`` maps ``skill_id -> (license, source_link)`` (see
    :func:`load_licenses_full`). Each attribution-required skill gets one bullet naming the
    bundle, its SPDX expression and its source link. A skill with no source link still
    appears (attribution is owed whether or not a link was recorded).
    """
    lines = ["## THIRD_PARTY_NOTICES", ""]
    required = list(report.get("attribution_required", []) or [])
    if not required:
        lines.append("_No third-party components require attribution._")
        lines.append("")
        return "\n".join(lines)
    for skill_id in required:
        full = (licenses_full or {}).get(skill_id)
        expr = full[0] if full and len(full) > 0 and full[0] else "<unknown>"
        link = full[1] if full and len(full) > 1 and full[1] else None
        entry = f"- **{skill_id}** — licence: `{expr}`"
        if link:
            entry += f" — source: {link}"
        lines.append(entry)
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` to ``path`` atomically (tmp file + ``replace``) with sorted keys."""
    import json

    payload = json.dumps(obj, sort_keys=True, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
