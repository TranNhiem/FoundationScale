"""
Unit tests for ``foundationskills.intake.skill2env`` (B1 importer core).

Contract decisions encoded here (see the B1 importer contract):

* the status vocabulary is exactly ``PASS`` / ``RED`` / ``UNMEASURED`` / ``REFUSED``;
* absence of evidence is never ``PASS`` and a crashing check is ``REFUSED`` for that
  task, never a silent pass;
* ``task_verdict`` receives ``license_status`` and ``preflight_status`` as the
  ``(status, reason)`` tuples produced by ``license_verdict`` and
  ``dockerfile_preflight`` respectively.

Everything (corpus, license CSV, Harbor trial outputs) is synthesised under
``tmp_path``; the suite never touches the network or the ``foundationscale`` package.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import pytest

import foundationskills.intake.skill2env as skill2env


# --------------------------------------------------------------------------
# Synthetic corpus constants
# --------------------------------------------------------------------------

BUNDLE_DIGEST = "a1" * 32  # 64 hex chars, as mandated for source_bundle_digest
IMG_REF = "gcr.io/foundationskills/base:1.0"
UNPINNED_REF = "gcr.io/foundationskills/other:2.0"
PIN_DIGEST = "sha256:" + "b" * 64

ALLOWED_LICENSES = (
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
)


def _toml_key(ref: str) -> str:
    return '"' + ref.replace("\\", "\\\\").replace('"', '\\"') + '"'


def task_toml(*, name, source_skill, bundle_digest, pins) -> str:
    """Render a ``task.toml`` in the real corpus format."""
    body = [
        "[task]",
        f'name = "{name}"',
        "",
        "[metadata]",
        f'source_skill = "{source_skill}"',
        f'source_bundle_digest = "{bundle_digest}"',
        "",
        "[metadata.base_image_pins]",
    ]
    for ref, digest in pins.items():
        body.append(f"{_toml_key(ref)} = \"{digest}\"")
    body += [
        "",
        "[verifier]",
        'network_mode = "bridge"',
        "timeout_sec = 600",
        "",
        "[agent]",
        'network_mode = "none"',
        "timeout_sec = 900",
        "",
        "[environment]",
        'network_mode = "bridge"',
        "build_timeout_sec = 300",
        "",
    ]
    return "\n".join(body) + "\n"


def make_task(
    root,
    dirname,
    *,
    name=None,
    source_skill=None,
    bundle_digest=BUNDLE_DIGEST,
    pins=None,
    toml=None,
    skip=(),
    instruction="Do the thing.\n",
    dockerfile=None,
) -> Path:
    """Materialise one synthetic task dir in the real corpus layout."""
    skill = source_skill if source_skill is not None else f"local/{dirname}"
    pins = dict(pins) if pins is not None else {IMG_REF: PIN_DIGEST}
    task_dir = Path(root) / dirname
    files = {
        "task.toml": toml
        if toml is not None
        else task_toml(
            name=name or dirname,
            source_skill=skill,
            bundle_digest=bundle_digest,
            pins=pins,
        ),
        "instruction.md": instruction,
        "environment/Dockerfile": dockerfile
        if dockerfile is not None
        else f"FROM {IMG_REF}\nRUN true\n",
        "solution/solve.sh": "#!/bin/sh\n",
        "tests/test.sh": "#!/bin/sh\n",
        "tests/verify.py": "print('verify')\n",
        "tests/rubric.md": "# Rubric\n",
    }
    for rel, text in files.items():
        if rel in skip:
            continue
        path = task_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return task_dir


def corpus_root(tmp_path: Path) -> Path:
    root = Path(tmp_path) / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_license_csv(path, rows) -> Path:
    """rows: (skill_id, license_expression, source_link) triples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["skill_group", "skill_id", "license", "source_link", "local_path"])
        for skill_id, expr, link in rows:
            writer.writerow([f"group/{skill_id}", skill_id, expr, link, f"vendor/{skill_id}"])
    return path


def write_result_json(path, doc) -> Path:
    """A trial result laid out as Harbor writes it: result.json beside the trial's agent/ dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "agent").mkdir(exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def oracle_result_doc(rewards, exception_type=None):
    return {
        "verifier_result": {} if rewards is None else {"rewards": rewards},
        "exception_info": {"exception_type": exception_type},
    }


def make_gate(status, *, agent="oracle", rewards=None, exception=None):
    return skill2env.GateResult(
        agent=agent, rewards=rewards, exception=exception, status=status
    )


def sample_record(**overrides):
    kwargs = dict(
        dirname="task-sample",
        name="task-sample",
        source_skill="local/skill-sample",
        skill_id="skill-sample",
        bundle_digest=BUNDLE_DIGEST,
        base_image_pins=((IMG_REF, PIN_DIGEST),),
        agent_timeout_s=900,
        verifier_timeout_s=600,
        network_modes=("bridge", "none", "bridge"),
        has_solution=True,
        has_tests=True,
        instruction_sha256=hashlib.sha256(b"Do the thing.\n").hexdigest(),
    )
    kwargs.update(overrides)
    return skill2env.TaskRecord(**kwargs)


def clear_gates():
    return {
        "oracle": make_gate("PASS", rewards={"tests/test.sh": 1.0}),
        "nop": make_gate("PASS", agent="nop", rewards={"tests/test.sh": 0.0}),
    }


def preflight_fields(entry):
    """Accept either shape of a serialised preflight result."""
    if isinstance(entry, dict):
        return entry["status"], entry.get("reason", "")
    return entry[0], entry[1]


# --------------------------------------------------------------------------
# scan_corpus
# --------------------------------------------------------------------------


def test_scan_corpus_reads_every_record_field(tmp_path):
    root = corpus_root(tmp_path)
    instruction = "Host an app on Firebase App Hosting.\n"
    make_task(
        root,
        "task-a",
        name="Firebase App Hosting Basics",
        source_skill="local/firebase-app-hosting-basics",
        instruction=instruction,
    )

    records, failures = skill2env.scan_corpus(root)

    assert failures == []
    assert len(records) == 1
    rec = records[0]
    assert rec.dirname == "task-a"
    assert rec.name == "Firebase App Hosting Basics"
    assert rec.source_skill == "local/firebase-app-hosting-basics"
    assert rec.skill_id == "firebase-app-hosting-basics"
    assert rec.bundle_digest == BUNDLE_DIGEST
    assert dict(rec.base_image_pins) == {IMG_REF: PIN_DIGEST}
    assert rec.agent_timeout_s == 900
    assert rec.verifier_timeout_s == 600
    assert sorted(rec.network_modes) == ["bridge", "bridge", "none"]
    assert rec.has_solution is True
    assert rec.has_tests is True
    assert rec.instruction_sha256 == hashlib.sha256(instruction.encode()).hexdigest()


def test_scan_corpus_missing_required_files_are_refused(tmp_path):
    root = corpus_root(tmp_path)
    make_task(root, "task-good")
    make_task(root, "task-no-tests", skip=("tests/test.sh",))
    make_task(root, "task-no-solve", skip=("solution/solve.sh",))
    make_task(root, "task-no-dockerfile", skip=("environment/Dockerfile",))

    records, failures = skill2env.scan_corpus(root)

    assert [r.dirname for r in records] == ["task-good"]
    by_task = {f["task"]: f for f in failures}
    assert set(by_task) == {"task-no-tests", "task-no-solve", "task-no-dockerfile"}
    for dirname, missing in [
        ("task-no-tests", "test.sh"),
        ("task-no-solve", "solve.sh"),
        ("task-no-dockerfile", "Dockerfile"),
    ]:
        failure = by_task[dirname]
        assert failure["verdict"] == "REFUSED"
        assert {"task", "verdict", "reason"} <= set(failure)
        assert missing in failure["reason"]


def test_scan_corpus_unparsable_toml_is_refused(tmp_path):
    root = corpus_root(tmp_path)
    make_task(root, "task-broken", toml="[task\nname = 'x'\n")

    records, failures = skill2env.scan_corpus(root)

    assert records == []
    assert len(failures) == 1
    failure = failures[0]
    assert failure["task"] == "task-broken"
    assert failure["verdict"] == "REFUSED"
    assert failure["reason"]


def test_scan_corpus_missing_required_metadata_is_refused(tmp_path):
    # Absence of evidence is never PASS: an incomplete task.toml is a scan failure.
    root = corpus_root(tmp_path)
    make_task(root, "task-no-metadata", toml='[task]\nname = "x"\n')

    records, failures = skill2env.scan_corpus(root)

    assert records == []
    assert [f["task"] for f in failures] == ["task-no-metadata"]
    assert failures[0]["verdict"] == "REFUSED"
    assert failures[0]["reason"]


def test_scan_corpus_is_sorted_and_deterministic(tmp_path):
    root = corpus_root(tmp_path)
    for dirname in ["task-c", "task-b", "task-a", "zz-broken"]:
        make_task(root, dirname, toml="not toml at all" if dirname == "zz-broken" else None)

    first = skill2env.scan_corpus(root)
    second = skill2env.scan_corpus(root)

    assert [r.dirname for r in first[0]] == ["task-a", "task-b", "task-c"]
    assert [f["task"] for f in first[1]] == ["zz-broken"]
    assert first == second


# --------------------------------------------------------------------------
# join key (source_skill vs. csv skill_id)
# --------------------------------------------------------------------------


def test_join_key_strips_the_source_skill_prefix(tmp_path):
    root = corpus_root(tmp_path)
    make_task(root, "t1", source_skill="local/firebase-app-hosting-basics")
    make_task(root, "t2", source_skill="plain-skill")
    records = {r.dirname: r for r in skill2env.scan_corpus(root)[0]}

    assert records["t1"].source_skill == "local/firebase-app-hosting-basics"
    assert records["t1"].skill_id == "firebase-app-hosting-basics"
    assert records["t2"].source_skill == "plain-skill"
    assert records["t2"].skill_id == "plain-skill"

    csv_path = write_license_csv(
        tmp_path / "licenses.csv",
        [
            ("firebase-app-hosting-basics", "MIT", "https://example.org/fb"),
            ("plain-skill", "ISC", "https://example.org/plain"),
        ],
    )
    licenses = skill2env.load_licenses(csv_path)

    assert licenses[records["t1"].skill_id] == "MIT"
    assert licenses[records["t2"].skill_id] == "ISC"


def test_load_licenses_parses_the_csv(tmp_path):
    csv_path = write_license_csv(
        tmp_path / "licenses.csv",
        [
            ("skill-a", "MIT", "https://example.org/a"),
            ("skill-b", "Apache-2.0 AND CC-BY-4.0", "https://example.org/b"),
        ],
    )
    assert skill2env.load_licenses(csv_path) == {
        "skill-a": "MIT",
        "skill-b": "Apache-2.0 AND CC-BY-4.0",
    }
    assert skill2env.load_licenses(csv_path) == skill2env.load_licenses(csv_path)


# --------------------------------------------------------------------------
# license_verdict / needs_attribution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    list(ALLOWED_LICENSES) + ["MIT AND Apache-2.0", "Apache-1.1 AND (MIT AND Zlib)"],
)
def test_license_verdict_allowlisted_expressions_pass(expr):
    status, reason = skill2env.license_verdict(expr)
    assert status == "PASS"
    assert reason


def test_license_verdict_ccby_passes_but_needs_attribution():
    status, reason = skill2env.license_verdict("Apache-2.0 AND CC-BY-4.0")
    assert status == "PASS"
    assert reason
    assert skill2env.needs_attribution("Apache-2.0 AND CC-BY-4.0") is True


def test_license_verdict_weak_copyleft_epl_is_red():
    status, reason = skill2env.license_verdict("EPL-2.0")
    assert status == "RED"
    assert "EPL" in reason
    assert "copyleft" in reason


def test_license_verdict_epl_component_trumps_a_clean_sibling():
    status, reason = skill2env.license_verdict("MIT AND EPL-2.0")
    assert status == "RED"
    assert "EPL" in reason


@pytest.mark.parametrize("expr,component", [("FunPL-1.0", "FunPL-1.0"), ("MIT AND SecretPL-9", "SecretPL-9")])
def test_license_verdict_unlisted_component_is_red(expr, component):
    status, reason = skill2env.license_verdict(expr)
    assert status == "RED"
    assert component in reason
    assert "allowlist" in reason


@pytest.mark.parametrize("expr", [None, ""])
def test_license_verdict_missing_row_is_unmeasured(expr):
    status, reason = skill2env.license_verdict(expr)
    assert status == "UNMEASURED"
    assert "license" in reason


@pytest.mark.parametrize(
    "expr,expected",
    [
        (None, False),
        ("MIT", False),
        ("Apache-2.0 AND CC-BY-4.0", True),
        ("OFL-1.1", True),
        ("OFL-1.1 AND CC-BY-4.0", True),
        ("EPL-2.0", False),
    ],
)
def test_needs_attribution(expr, expected):
    assert skill2env.needs_attribution(expr) == expected


# --------------------------------------------------------------------------
# dockerfile_preflight
# --------------------------------------------------------------------------


class SandboxRefusal(Exception):
    """Local stand-in for ``foundationscale.agentic_rl.sandbox.build.SandboxRefusal``."""


def builder_refusal(message: str) -> Exception:
    """Return a ``SandboxRefusal``-shaped exception (canonical name and type)."""
    remote = getattr(skill2env, "SandboxRefusal", None)
    if isinstance(remote, type) and issubclass(remote, Exception):
        try:
            hybrid = type("SandboxRefusal", (SandboxRefusal, remote), {})
        except TypeError:
            hybrid = remote
        return hybrid(message)
    return SandboxRefusal(message)


def accepting_validator():
    def validator(*args, **kwargs):
        return None

    return validator


def test_preflight_passes_when_from_is_pinned_and_the_builder_accepts(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {IMG_REF} AS build\nRUN true\n")
    calls = []

    def validator(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    status, reason = skill2env.dockerfile_preflight(dockerfile, {IMG_REF: PIN_DIGEST}, validator=validator)

    assert status == "PASS"
    assert "builder" in reason
    assert calls, "the module must run the FS builder (or the injected validator)"


def test_preflight_ignores_scratch_and_stage_aliases(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM scratch\nFROM {IMG_REF} AS build\nRUN true\n")
    status, reason = skill2env.dockerfile_preflight(dockerfile, {IMG_REF: PIN_DIGEST}, validator=accepting_validator())
    assert status == "PASS"
    assert reason

    scratch_only = tmp_path / "Dockerfile.scratch"
    scratch_only.write_text("FROM scratch\n")
    status, reason = skill2env.dockerfile_preflight(scratch_only, {}, validator=accepting_validator())
    assert status == "PASS"


def test_preflight_red_when_any_from_is_not_digest_pinned(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {IMG_REF}\nRUN true\nFROM {UNPINNED_REF}\n")

    status, reason = skill2env.dockerfile_preflight(dockerfile, {IMG_REF: PIN_DIGEST}, validator=accepting_validator())

    assert status == "RED"
    assert UNPINNED_REF in reason
    assert "digest-pinned" in reason


def test_preflight_pin_rule_outranks_a_broken_builder(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {UNPINNED_REF}\n")

    def validator(*args, **kwargs):
        raise ZeroDivisionError("boom")

    status, reason = skill2env.dockerfile_preflight(dockerfile, {}, validator=validator)
    assert status == "RED"
    assert "digest-pinned" in reason


def test_preflight_red_when_the_fs_builder_refuses(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {IMG_REF}\nRUN true\n")

    def validator(*args, **kwargs):
        raise builder_refusal("step 3: bind mounts are disallowed")

    status, reason = skill2env.dockerfile_preflight(dockerfile, {IMG_REF: PIN_DIGEST}, validator=validator)
    assert status == "RED"
    assert reason.startswith("FS builder refuses")
    assert "step 3: bind mounts are disallowed" in reason


def test_preflight_refused_when_preflight_crashes(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM {IMG_REF}\nRUN true\n")

    def validator(*args, **kwargs):
        raise ZeroDivisionError("boom")

    status, reason = skill2env.dockerfile_preflight(dockerfile, {IMG_REF: PIN_DIGEST}, validator=validator)
    assert status == "REFUSED"
    assert "preflight crashed" in reason
    assert "boom" in reason


# --------------------------------------------------------------------------
# parse_harbor_trial
# --------------------------------------------------------------------------


def _trial(tmp_path: Path, doc) -> Path:
    return write_result_json(Path(tmp_path) / "trial-0__" / "result.json", doc)


def test_parse_harbor_trial_oracle_full_reward_is_pass(tmp_path):
    path = _trial(tmp_path, oracle_result_doc({"tests/test.sh": 1.0, "verify.py": 1.0}))
    gate = skill2env.parse_harbor_trial(path, "oracle")
    assert gate.agent == "oracle"
    assert gate.status == "PASS"
    assert gate.rewards == {"tests/test.sh": 1.0, "verify.py": 1.0}
    assert gate.exception is None


def test_parse_harbor_trial_oracle_partial_reward_is_red(tmp_path):
    path = _trial(tmp_path, oracle_result_doc({"tests/test.sh": 1.0, "verify.py": 0.5}))
    gate = skill2env.parse_harbor_trial(path, "oracle")
    assert gate.status == "RED"


def test_parse_harbor_trial_oracle_zero_reward_is_red(tmp_path):
    path = _trial(tmp_path, oracle_result_doc({"tests/test.sh": 0.0}))
    gate = skill2env.parse_harbor_trial(path, "oracle")
    assert gate.status == "RED"


def test_parse_harbor_trial_nop_zero_reward_is_pass(tmp_path):
    path = _trial(tmp_path, oracle_result_doc({"tests/test.sh": 0.0, "verify.py": 0.0}))
    gate = skill2env.parse_harbor_trial(path, "nop")
    assert gate.agent == "nop"
    assert gate.status == "PASS"
    assert gate.exception is None


@pytest.mark.parametrize(
    "rewards",
    [{"tests/test.sh": 1.0}, {"tests/test.sh": 0.0, "verify.py": 0.5}],
)
def test_parse_harbor_trial_nop_positive_mean_is_red(tmp_path, rewards):
    path = _trial(tmp_path, oracle_result_doc(rewards))
    gate = skill2env.parse_harbor_trial(path, "nop")
    assert gate.status == "RED"


def test_parse_harbor_trial_exception_is_unmeasured(tmp_path):
    path = _trial(tmp_path, oracle_result_doc(None, exception_type="SandboxTimeout"))
    gate = skill2env.parse_harbor_trial(path, "oracle")
    assert gate.status == "UNMEASURED"
    assert gate.exception == "SandboxTimeout"
    path = _trial(tmp_path, oracle_result_doc(None, exception_type="SandboxTimeout"))
    assert skill2env.parse_harbor_trial(path, "nop").status == "UNMEASURED"


def test_parse_harbor_trial_exception_voids_otherwise_perfect_rewards(tmp_path):
    # Absence of evidence is never PASS: a crashed trial proves nothing.
    path = _trial(
        tmp_path,
        oracle_result_doc({"tests/test.sh": 1.0}, exception_type="RuntimeError"),
    )
    gate = skill2env.parse_harbor_trial(path, "oracle")
    assert gate.status == "UNMEASURED"
    assert gate.exception == "RuntimeError"


@pytest.mark.parametrize("agent", ["oracle", "nop"])
@pytest.mark.parametrize("doc", [{}, {"verifier_result": {}}, {"verifier_result": {"rewards": {}}}])
def test_parse_harbor_trial_without_rewards_is_unmeasured(tmp_path, agent, doc):
    doc = dict(doc)
    doc["exception_info"] = {"exception_type": None}
    path = _trial(tmp_path, doc)
    gate = skill2env.parse_harbor_trial(path, agent)
    assert gate.status == "UNMEASURED"
    assert gate.agent == agent


# --------------------------------------------------------------------------
# run_gate
# --------------------------------------------------------------------------


def _gate_kwargs(task_dir, jobs_dir, **overrides):
    kwargs = dict(
        harbor_bin="harbor",
        env_import_path="foundationskills.corpus_env:make_env",
        env_kwargs={"A": "1"},
        jobs_dir=jobs_dir,
        agent="oracle",
        timeout_s=60,
    )
    kwargs.update(overrides)
    return (task_dir, kwargs)


def test_run_gate_builds_the_argv_list_and_parses_the_oracle_trial(tmp_path):
    task_dir = make_task(tmp_path, "task-a")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    calls = []

    def fake_run(*f_args, **f_kwargs):
        argv = f_kwargs.get("args") if f_kwargs.get("args") is not None else f_args[0]
        assert isinstance(argv, list), "run_gate must hand the runner an argv list, never a shell string"
        assert all(isinstance(item, str) for item in argv)
        assert not f_kwargs.get("shell")
        calls.append(list(argv))
        write_result_json(
            jobs_dir / "trial-0__" / "result.json",
            oracle_result_doc({"tests/test.sh": 1.0, "verify.py": 1.0}),
        )
        return subprocess.CompletedProcess(args=list(argv), returncode=0, stdout="", stderr="")

    positionals, kwargs = _gate_kwargs(task_dir, jobs_dir, run=fake_run)
    gate = skill2env.run_gate(positionals, **kwargs)

    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "harbor"
    assert argv[1] == "run"
    assert argv[argv.index("--path") + 1] == str(task_dir)
    assert argv[argv.index("--agent") + 1] == "oracle"
    assert argv[argv.index("--environment-import-path") + 1] == "foundationskills.corpus_env:make_env"
    assert argv[argv.index("--ek") + 1] == "A=1"
    assert argv[argv.index("--jobs-dir") + 1] == str(jobs_dir)
    assert gate.agent == "oracle"
    assert gate.status == "PASS"
    assert gate.rewards == {"tests/test.sh": 1.0, "verify.py": 1.0}
    assert gate.exception is None


def test_run_gate_forwards_every_env_kwarg_and_parses_the_nop_trial(tmp_path):
    task_dir = make_task(tmp_path, "task-a")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    calls = []

    def fake_run(*f_args, **f_kwargs):
        argv = f_kwargs.get("args") if f_kwargs.get("args") is not None else f_args[0]
        calls.append(list(argv))
        write_result_json(
            jobs_dir / "trial-0__" / "result.json",
            oracle_result_doc({"tests/test.sh": 0.0}),
        )
        return subprocess.CompletedProcess(args=list(argv), returncode=0, stdout="", stderr="")

    positionals, kwargs = _gate_kwargs(
        task_dir,
        jobs_dir,
        env_kwargs={"A": "1", "B": "2"},
        agent="nop",
        run=fake_run,
    )
    gate = skill2env.run_gate(positionals, **kwargs)

    argv = calls[0]
    assert argv.count("--ek") == 2
    assert "A=1" in argv
    assert "B=2" in argv
    assert argv[argv.index("--agent") + 1] == "nop"
    assert gate.agent == "nop"
    assert gate.status == "PASS"


def test_run_gate_timeout_is_unmeasured(tmp_path):
    task_dir = make_task(tmp_path, "task-a")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    def fake_run(*f_args, **f_kwargs):
        raise subprocess.TimeoutExpired(cmd=["harbor"], timeout=1)

    positionals, kwargs = _gate_kwargs(task_dir, jobs_dir, run=fake_run, timeout_s=1)
    gate = skill2env.run_gate(positionals, **kwargs)
    assert gate.status == "UNMEASURED"


def test_run_gate_without_a_result_json_is_unmeasured(tmp_path):
    task_dir = make_task(tmp_path, "task-a")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    def fake_run(*f_args, **f_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    positionals, kwargs = _gate_kwargs(task_dir, jobs_dir, run=fake_run)
    gate = skill2env.run_gate(positionals, **kwargs)
    assert gate.status == "UNMEASURED"


# --------------------------------------------------------------------------
# task_verdict
# --------------------------------------------------------------------------


LICENSE_OK = ("PASS", "MIT on the allowlist")
PREFLIGHT_OK = ("PASS", "FS builder accepts")


def test_task_verdict_all_clear_is_pass_without_reasons():
    verdict, reasons = skill2env.task_verdict(sample_record(), LICENSE_OK, PREFLIGHT_OK, clear_gates())
    assert verdict == "PASS"
    assert reasons == []


def test_task_verdict_license_rule_id():
    verdict, reasons = skill2env.task_verdict(
        sample_record(),
        ("RED", "license component EPL-1.0 not on the allowlist"),
        PREFLIGHT_OK,
        clear_gates(),
    )
    assert verdict == "RED"
    joined = " ".join(reasons)
    assert "SE-LIC-001" in joined
    assert "EPL-1.0" in joined


def test_task_verdict_preflight_from_pin_rule_id():
    verdict, reasons = skill2env.task_verdict(
        sample_record(),
        LICENSE_OK,
        ("RED", f"FROM {UNPINNED_REF} not digest-pinned in task.toml"),
        clear_gates(),
    )
    assert verdict == "RED"
    joined = " ".join(reasons)
    assert "SE-PRE-001" in joined
    assert "digest-pinned" in joined


def test_task_verdict_builder_refusal_rule_id():
    verdict, reasons = skill2env.task_verdict(
        sample_record(),
        LICENSE_OK,
        ("RED", "FS builder refuses: bind mounts are disallowed"),
        clear_gates(),
    )
    assert verdict == "RED"
    joined = " ".join(reasons)
    assert "SE-PRE-002" in joined
    assert "bind mounts are disallowed" in joined


def test_task_verdict_gate_rule_ids():
    gates = {
        "oracle": make_gate("RED", rewards={"tests/test.sh": 0.5}, agent="oracle"),
        "nop": make_gate("RED", rewards={"tests/test.sh": 1.0}, agent="nop"),
    }
    verdict, reasons = skill2env.task_verdict(sample_record(), LICENSE_OK, PREFLIGHT_OK, gates)
    assert verdict == "RED"
    joined = " ".join(reasons)
    assert "SE-GATE-001" in joined
    assert "oracle did not reach full reward" in joined
    assert "SE-GATE-002" in joined
    assert "no-op" in joined


def test_task_verdict_gate_not_run_is_unmeasured():
    verdict, reasons = skill2env.task_verdict(sample_record(), LICENSE_OK, PREFLIGHT_OK, None)
    assert verdict == "UNMEASURED"
    joined = " ".join(reasons)
    assert "acceptance gate not run" in joined
    assert any(rule in joined for rule in ("SE-GATE-001", "SE-GATE-002"))


def test_task_verdict_takes_the_worst_status_of_every_input():
    ranks = {"PASS": 0, "UNMEASURED": 1, "RED": 2, "REFUSED": 3}
    for license_s, preflight_s, oracle_s, nop_s in itertools.product(ranks, repeat=4):
        gates = {
            "oracle": make_gate(oracle_s, rewards={"tests/test.sh": 1.0}),
            "nop": make_gate(nop_s, agent="nop", rewards={"tests/test.sh": 0.0}),
        }
        verdict, reasons = skill2env.task_verdict(
            sample_record(),
            (license_s, "license reason"),
            (preflight_s, "preflight reason"),
            gates,
        )
        worst = max((license_s, preflight_s, oracle_s, nop_s), key=lambda s: ranks[s])
        assert verdict == worst
        if verdict == "PASS":
            assert reasons == []
        else:
            assert reasons


# --------------------------------------------------------------------------
# build_import_report / render_notices / write_json
# --------------------------------------------------------------------------

DATASET = "fsh/foundation-skills"
REVISION = "git:4f0c1e2"
OPERATOR_DECISION = "import into fs-401"

# (dirname, skill_id, license expression, source_link); None == no license row at all
DATASET_SPEC = [
    ("task-01", "skill-ofl", "OFL-1.1", "https://example.org/skill-ofl"),
    ("task-02", "skill-epl", "EPL-2.0", "https://example.org/skill-epl"),
    ("task-03", "skill-ccby", "Apache-2.0 AND CC-BY-4.0", "https://example.org/skill-ccby"),
    ("task-04", "skill-mit", "MIT", "https://example.org/skill-mit"),
    ("task-05", "skill-nolic", None, None),
    ("task-06", "skill-weird", "Mega-PL-1.0", "https://example.org/skill-weird"),
]


def build_report(tmp_path):
    tmp_path = Path(tmp_path)
    root = tmp_path / "corpus"
    for dirname, skill, _expr, _link in DATASET_SPEC:
        make_task(root, dirname, source_skill=f"local/{skill}")
    csv_path = write_license_csv(
        tmp_path / "licenses.csv",
        [(skill, expr, link) for _dn, skill, expr, link in DATASET_SPEC if expr is not None],
    )
    records, scan_failures = skill2env.scan_corpus(root)
    licenses = skill2env.load_licenses(csv_path)
    preflight = {r.dirname: ("PASS", "FS builder accepts") for r in records}
    gates = {r.dirname: clear_gates() for r in records}
    return skill2env.build_import_report(
        records,
        scan_failures,
        licenses,
        preflight,
        gates,
        dataset=DATASET,
        revision=REVISION,
        operator_decision=OPERATOR_DECISION,
    )


def test_build_import_report_counts_exclusions_and_task_entries(tmp_path):
    report = build_report(tmp_path)

    assert report["schema"] == "fskills.skill2env_import/1"
    assert set(report) == {
        "schema",
        "dataset",
        "revision",
        "operator_decision",
        "counts",
        "license_counts",
        "excluded",
        "join_coverage",
        "attribution_required",
        "tasks",
    }
    assert report["dataset"] == DATASET
    assert report["revision"] == REVISION
    assert report["operator_decision"] == OPERATOR_DECISION

    for status, expected in {"PASS": 3, "RED": 2, "UNMEASURED": 1, "REFUSED": 0}.items():
        assert report["counts"].get(status, 0) == expected
    assert sum(report["counts"].values()) == len(DATASET_SPEC)
    assert report["excluded"] == {"epl": 1, "unlisted_license": 1, "missing_license": 1}
    assert report["join_coverage"] == pytest.approx(5 / 6)
    assert report["attribution_required"] == ["skill-ccby", "skill-ofl"]
    for expr in ["MIT", "OFL-1.1", "EPL-2.0", "Mega-PL-1.0", "Apache-2.0 AND CC-BY-4.0"]:
        assert report["license_counts"][expr] == 1

    by_dir = {t["dirname"]: t for t in report["tasks"]}
    assert set(by_dir) == {dn for dn, _skill, _expr, _link in DATASET_SPEC}
    assert [t["dirname"] for t in report["tasks"]] == sorted(by_dir)

    entry = by_dir["task-03"]
    assert set(entry) == {
        "dirname",
        "source_skill",
        "bundle_digest",
        "base_image_pins",
        "license",
        "license_status",
        "preflight",
        "gate",
        "verdict",
        "reasons",
    }
    assert entry["source_skill"] == "local/skill-ccby"
    assert entry["bundle_digest"] == BUNDLE_DIGEST
    assert dict(entry["base_image_pins"]) == {IMG_REF: PIN_DIGEST}
    assert entry["license"] == "Apache-2.0 AND CC-BY-4.0"
    assert entry["license_status"] == "PASS"
    assert preflight_fields(entry["preflight"])[0] == "PASS"
    assert set(entry["gate"]) == {"oracle", "nop"}
    assert {"status", "rewards", "exception"} <= set(entry["gate"]["oracle"])
    assert entry["gate"]["oracle"]["status"] == "PASS"
    assert entry["gate"]["oracle"]["rewards"] == {"tests/test.sh": 1.0}
    assert entry["gate"]["oracle"]["exception"] is None
    assert entry["verdict"] == "PASS"
    assert entry["reasons"] == []

    epl_entry = by_dir["task-02"]
    assert epl_entry["verdict"] == "RED"
    assert "SE-LIC-001" in " ".join(epl_entry["reasons"])
    assert "EPL" in " ".join(epl_entry["reasons"])

    weird_entry = by_dir["task-06"]
    assert weird_entry["verdict"] == "RED"

    nolic_entry = by_dir["task-05"]
    assert nolic_entry["license"] is None
    assert nolic_entry["license_status"] == "UNMEASURED"
    assert nolic_entry["verdict"] == "UNMEASURED"
    assert "SE-LIC-001" in " ".join(nolic_entry["reasons"])


def test_build_import_report_is_deterministic(tmp_path):
    one = build_report(Path(tmp_path) / "one")
    two = build_report(Path(tmp_path) / "two")
    again = build_report(Path(tmp_path) / "one")

    blob = json.dumps(one, sort_keys=True)
    assert blob == json.dumps(two, sort_keys=True)
    assert blob == json.dumps(again, sort_keys=True)  # no timestamps, no cache side effects
    assert str(tmp_path) not in blob  # no absolute paths leak into the artefact


def test_build_import_report_scan_failures_are_refused_never_pass(tmp_path):
    tmp_path = Path(tmp_path)
    root = tmp_path / "corpus"
    make_task(root, "task-good", source_skill="local/skill-mit")
    make_task(root, "task-bad", toml="[task\nname = 'x'\n")

    records, scan_failures = skill2env.scan_corpus(root)
    assert len(records) == 1
    assert len(scan_failures) == 1

    report = skill2env.build_import_report(
        records,
        scan_failures,
        {"skill-mit": "MIT"},
        {r.dirname: ("PASS", "FS builder accepts") for r in records},
        {r.dirname: clear_gates() for r in records},
        dataset=DATASET,
        revision=REVISION,
        operator_decision=OPERATOR_DECISION,
    )

    blob = json.dumps(report, sort_keys=True)
    assert "SE-SCAN-001" in blob
    assert report["counts"].get("REFUSED", 0) == 1
    assert sum(report["counts"].values()) == 2
    for entry in report["tasks"]:
        if entry.get("dirname") == "task-bad":
            assert entry["verdict"] != "PASS"
    good_entries = [t for t in report["tasks"] if t.get("dirname") == "task-good"]
    assert good_entries and good_entries[0]["verdict"] == "PASS"


def test_render_notices_lists_attribution_required_skills(tmp_path):
    report = build_report(tmp_path)
    licenses_full = {
        skill: (expr, link) for _dn, skill, expr, link in DATASET_SPEC if expr is not None
    }
    licenses_full["skill-nolic"] = (None, "")

    out = skill2env.render_notices(report, licenses_full=licenses_full)

    assert any(line.lstrip().startswith("#") and "THIRD_PARTY_NOTICES" in line for line in out.splitlines())
    assert "skill-ccby" in out
    assert "Apache-2.0 AND CC-BY-4.0" in out
    assert "https://example.org/skill-ccby" in out
    assert "skill-ofl" in out
    assert "OFL-1.1" in out
    assert "https://example.org/skill-ofl" in out
    # skills that need no attribution must not show up in the notices file
    assert "skill-mit" not in out
    assert "skill-epl" not in out
    assert "skill-weird" not in out
    assert "skill-nolic" not in out

    # the optional mapping is optional
    bare = skill2env.render_notices(report)
    assert "THIRD_PARTY_NOTICES" in bare
    assert "skill-ccby" in bare
    assert "skill-ofl" in bare


def test_write_json_is_atomic_and_key_sorted(tmp_path):
    path = tmp_path / "out" / "report.json"
    path.parent.mkdir(parents=True)
    obj = {"b": 1, "a": {"z": 1, "y": 2}, "c": [{"k": 1}, {"j": 2}]}

    skill2env.write_json(path, obj)

    raw = path.read_text(encoding="utf-8")
    assert json.loads(raw) == json.loads(json.dumps(obj))
    assert raw.index('"a"') < raw.index('"b"') < raw.index('"c"')
    assert raw.index('"y"') < raw.index('"z"')
    assert sorted(child.name for child in path.parent.iterdir()) == [path.name]

    # overwrite leaves no stray temp files behind either (tmp + os.replace)
    skill2env.write_json(path, {"x": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}
    assert sorted(child.name for child in path.parent.iterdir()) == [path.name]


def test_statuses_are_exactly_the_contract_vocabulary():
    values = set()
    for expr in [None, "MIT", "Mega-PL-1.0", "EPL-2.0"]:
        values.add(skill2env.license_verdict(expr)[0])
    dockerfile = Path(__file__)
    values.add(skill2env.dockerfile_preflight(dockerfile, {IMG_REF: PIN_DIGEST}, validator=accepting_validator())[0])
    assert values == {"PASS", "RED", "UNMEASURED"} | values-{"PASS", "RED", "UNMEASURED"}
    assert values <= {"PASS", "RED", "UNMEASURED", "REFUSED"}


def test_from_pin_accepts_inline_digest_matching_task_toml_and_skips_platform_flags(tmp_path):
    """Real-corpus finding (B1.2): 7,492/7,496 Dockerfiles write `FROM image:tag@sha256:d` while task.toml pins
    `image:tag` -> d; the first rule version read every one as unpinned."""
    d = "sha256:" + "a" * 64
    df = Path(tmp_path) / "Dockerfile"
    ok = lambda text, ctx: None  # noqa: E731 - the builder leg is not under test here
    df.write_text(f"FROM --platform=linux/amd64 python:3.12-slim@{d}\nRUN true\n")
    assert skill2env.dockerfile_preflight(df, {"python:3.12-slim": d}, validator=ok)[0] == "PASS"
    df.write_text(f"FROM python:3.12-slim@sha256:{'b' * 64}\n")
    status, reason = skill2env.dockerfile_preflight(df, {"python:3.12-slim": d}, validator=ok)
    assert status == "RED" and "disagrees" in reason
    df.write_text("FROM python:3.12-slim\n")
    assert skill2env.dockerfile_preflight(df, {}, validator=ok)[0] == "RED"


def test_run_gate_reads_the_trial_result_not_the_job_summary(tmp_path):
    """Real-run finding (B2.1 R0): Harbor writes <jobs>/<timestamp>/<task>__<id>/result.json, and the timestamp
    dir also contains "__"; the job-level summary (no verifier_result) was read and every task looked UNMEASURED."""
    jobs = Path(tmp_path) / "jobs"

    def fake_run(argv, **kwargs):
        job = jobs / "2026-10-12__13-00-11"
        trial = job / "task_x__abc123"
        (trial / "agent").mkdir(parents=True)
        (job / "result.json").write_text(json.dumps({"stats": {}}))  # job summary: no verifier_result
        (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"a": 1.0, "b": 1.0}},
                                                       "exception_info": None}))
        return None

    g = skill2env.run_gate(Path(tmp_path) / "task", harbor_bin="harbor", env_import_path="m:E", env_kwargs={},
                           jobs_dir=jobs, agent="oracle", run=fake_run)
    assert g.status == "PASS" and g.rewards == {"a": 1.0, "b": 1.0}
