"""M6 evidence-bound trial results: every derivation branch of evidence.py."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from foundationskills.skills.auto_research.evidence import (
    EVIDENCE_FIELDS,
    EvidenceError,
    derive_result,
    rehash_problems,
    sha256_file,
)

_FPRINT = "sha256:" + "ab" * 32
_ABSENT = object()
_METRICS = ("arc_easy_acc", "rl_measured_fraction")
_BENCH = [{"name": "arc_easy", "score": 0.5, "stderr": 0.01}]


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _eval(path, *, verdict="PASS", checkpoint=_ABSENT, base=_ABSENT, limited=_ABSENT,
          fingerprint=_FPRINT, benchmarks=_ABSENT):
    report = {"verdict": verdict, "policy": {"fingerprint": fingerprint} if fingerprint is not _ABSENT else {}}
    for key, value in (("checkpoint", checkpoint), ("base", base), ("limited", limited),
                       ("benchmarks", benchmarks)):
        if value is not _ABSENT:
            report[key] = value
    return _write(path, report)


def _rl(path, *, status="PASS", checkpoint=_ABSENT, raw_checkpoint=_ABSENT,
        measured=0.7, min_floor=0.5, steps=12):
    payload = {"status": status, "steps_measured": steps}
    if measured is not _ABSENT:
        payload["measured_fraction"] = measured
    if min_floor is not _ABSENT:
        payload["min_measured_fraction"] = min_floor
    if checkpoint is not _ABSENT:
        payload["checkpoint"] = f"saved: {checkpoint}"
    if raw_checkpoint is not _ABSENT:
        payload["checkpoint"] = raw_checkpoint
    return _write(path, payload)


def _sft(tmp_path, output_dir, *, logs=("[fs:train:done]          PASS",),
         telemetry=_ABSENT, config=_ABSENT):
    payload = {"artifact_paths": {"output_dir": str(output_dir)}}
    if telemetry is not _ABSENT:
        payload["telemetry"] = telemetry
    if config is not _ABSENT:
        payload["config"] = config
    man = _write(Path(tmp_path) / "run_manifest.json", payload)
    if logs is not None:
        (Path(tmp_path) / "fskills_launch.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    return man


def _identity(**extra):
    ident = {"trial": "t1", "role": "train", "seed": 7}
    ident.update(extra)
    return ident


def _evi(manifests, report):
    return {"run_manifests": manifests, "eval_report": report}


def _error(identity, evidence, *, metric_names=(), rl_floor=None):
    with pytest.raises(EvidenceError) as excinfo:
        derive_result(identity, evidence, metric_names=metric_names, rl_floor=rl_floor)
    return excinfo.value


def test_api_surface():
    assert EVIDENCE_FIELDS == ("evidence", "evidence_class", "metric_sources", "derived")
    err = EvidenceError("AR-IN-011", "boom", {"k": "v"})
    assert isinstance(err, ValueError)
    assert (err.rule_id, err.message, err.detail) == ("AR-IN-011", "boom", {"k": "v"})


def test_sha256_file_format(tmp_path):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"abc")
    digest = sha256_file(str(blob))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert digest == "sha256:" + hashlib.sha256(b"abc").hexdigest()


def test_rl_pass(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(_identity(), _evi([man], rep), metric_names=_METRICS, rl_floor=0.5)
    assert (res["status"], res["limited"], res["steps"], res["eval_policy_fingerprint"]) == (
        "ok", False, 12, _FPRINT)
    assert res["evidence_class"] == "bound"
    assert res["metrics"]["arc_easy_acc"] == {"value": 0.5, "se": 0.01}
    assert res["metrics"]["rl_measured_fraction"] == {"value": 0.7, "se": 0.0}
    assert res["metric_sources"] == {"arc_easy_acc": "eval_report", "rl_measured_fraction": "run_manifest"}
    assert res["derived"] == {"measured_fraction": 0.7, "min_measured_fraction": 0.5, "checkpoint": str(final)}
    record = res["evidence"]["run_manifests"][0]
    assert (record["kind"], record["status"], record["path"]) == ("rl", "ok", man)
    assert record["sha256"] == sha256_file(man)
    report = res["evidence"]["eval_report"]
    assert (report["verdict"], report["checkpoint"], report["sha256"]) == (
        "PASS", str(final), sha256_file(rep))
    assert res["evidence"]["launch_logs"] == []


def test_rl_unmeasured_below_floor_keeps_metrics(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    measured = 21 / 128
    man = _rl(tmp_path / "fskills_rl_manifest.json", status="UNMEASURED: 21/128 below floor 0.5",
              checkpoint=str(final), measured=measured)
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(_identity(), _evi([man], rep), metric_names=_METRICS, rl_floor=0.5)
    assert res["status"] == "unmeasured"  # D2: metrics stay visible even though unmeasured
    assert res["metrics"]["arc_easy_acc"] == {"value": 0.5, "se": 0.01}
    assert res["metrics"]["rl_measured_fraction"]["value"] == pytest.approx(measured)
    assert res["derived"]["measured_fraction"] == pytest.approx(measured)
    assert res["derived"]["checkpoint"] == str(final)  # checkpoint still saved
    assert res["evidence"]["run_manifests"][0]["status"] == "unmeasured"
    assert rehash_problems(res) == []


def test_rl_checkpoint_not_saved_record_degrades(tmp_path):
    man = _rl(tmp_path / "fskills_rl_manifest.json", raw_checkpoint=str(tmp_path / "final"))
    rep = _eval(tmp_path / "report.json", checkpoint="/any/where")
    res = derive_result(_identity(), _evi([man], rep), metric_names=())
    assert res["status"] == "unmeasured"
    assert res["steps"] == 12
    assert res["derived"]["checkpoint"] is None
    assert res["evidence"]["run_manifests"][0]["status"] == "unmeasured"



def test_preference_manifest_row(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _write(tmp_path / "fskills_preference_manifest.json",
                 {"status": "PASS", "steps_measured": 4, "checkpoint": f"saved: {final}",
                  "measured_fraction": 0.5})
    rep = _eval(tmp_path / "report.json", checkpoint=str(final))
    res = derive_result(_identity(), _evi([man], rep), metric_names=_METRICS)
    assert res["status"] == "ok" and res["steps"] == 4
    assert res["evidence"]["run_manifests"][0]["kind"] == "preference"
    assert res["derived"]["measured_fraction"] is None
    assert res["derived"]["min_measured_fraction"] is None
    assert res["metrics"]["rl_measured_fraction"] == {"value": None, "se": None}
    assert res["metric_sources"]["rl_measured_fraction"] == "absent"


def test_sft_manifest_with_launch_log(tmp_path):
    out = tmp_path / "out"
    (out / "final").mkdir(parents=True)
    man = _sft(tmp_path, out, telemetry={"perf_steps_observed": 5}, config={"max_steps": 3},
               logs=("[fs:train:done]             UNMEASURED: retry", "[fs:train:done]          PASS"))
    log = tmp_path / "fskills_launch.log"
    rep = _eval(tmp_path / "report.json", checkpoint=str(out / "final"))
    res = derive_result(_identity(), _evi([man], rep), metric_names=("tl_acc",))
    assert res["status"] == "ok" and res["steps"] == 5  # last done line wins; telemetry steps first
    assert res["derived"]["checkpoint"] == str(out / "final")
    record = res["evidence"]["run_manifests"][0]
    assert (record["kind"], record["status"], record["sha256"]) == ("sft", "ok", sha256_file(man))
    assert res["evidence"]["launch_logs"] == [{"path": str(log), "sha256": sha256_file(log)}]
    assert res["derived"]["measured_fraction"] is None


@pytest.mark.parametrize("telemetry,config,expected", [
    ({"perf_steps_observed": 5}, {"max_steps": 3}, 5),
    ({"perf_steps_observed": "5"}, {"max_steps": 3}, 3),
    (_ABSENT, {"max_steps": 3}, 3),
    (_ABSENT, _ABSENT, 0),
])
def test_sft_steps_fallback(tmp_path, telemetry, config, expected):
    out = tmp_path / "out"
    (out / "final").mkdir(parents=True)
    man = _sft(tmp_path, out, telemetry=telemetry, config=config)
    rep = _eval(tmp_path / "report.json", checkpoint=str(out / "final"))
    res = derive_result(_identity(), _evi([man], rep), metric_names=())
    assert res["steps"] == expected


def test_sft_final_dir_missing_degrades(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    man = _sft(tmp_path, out, telemetry={"perf_steps_observed": 5})
    rep = _eval(tmp_path / "report.json", checkpoint="/elsewhere")
    res = derive_result(_identity(), _evi([man], rep), metric_names=())
    assert res["status"] == "unmeasured"
    assert res["derived"]["checkpoint"] is None


def test_base_eval_without_manifests(tmp_path):
    base = str(tmp_path / "base_model")
    rep = _eval(tmp_path / "report.json", checkpoint=base, base=base, benchmarks=_BENCH)
    res = derive_result(_identity(), _evi([], rep), metric_names=_METRICS)
    assert res["status"] == "ok" and res["steps"] == 0
    assert res["derived"] == {"measured_fraction": None, "min_measured_fraction": None, "checkpoint": None}
    assert res["metrics"]["arc_easy_acc"] == {"value": 0.5, "se": 0.01}
    assert res["metric_sources"]["rl_measured_fraction"] == "absent"
    assert res["evidence"]["run_manifests"] == [] and res["evidence"]["launch_logs"] == []


def test_eval_unmeasured_keeps_metrics(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", verdict="UNMEASURED: n<30", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(_identity(), _evi([man], rep), metric_names=_METRICS)
    assert res["status"] == "unmeasured"
    assert res["evidence"]["eval_report"]["verdict"] == "UNMEASURED: n<30"
    assert res["metrics"]["arc_easy_acc"] == {"value": 0.5, "se": 0.01}


def test_eval_bound_to_other_checkpoint_refused(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(tmp_path / "elsewhere"))
    err = _error(_identity(), _evi([man], rep))
    assert err.rule_id == "AR-IN-012" and "eval_not_bound_to_trained_checkpoint" in err.message


def test_trained_checkpoint_eval_without_manifest_refused(tmp_path):
    base = str(tmp_path / "base_model")
    rep = _eval(tmp_path / "report.json", checkpoint=str(tmp_path / "trained"), base=base)
    err = _error(_identity(), _evi([], rep))
    assert err.rule_id == "AR-IN-012" and "eval_of_trained_checkpoint_without_manifest" in err.message


@pytest.mark.parametrize("limited,expected", [(_ABSENT, False), (True, True), (False, False)])
def test_eval_limited_marker(tmp_path, limited, expected):
    base = str(tmp_path / "base")
    kw = {} if limited is _ABSENT else {"limited": limited}
    rep = _eval(tmp_path / "report.json", checkpoint=base, base=base, **kw)
    res = derive_result(_identity(), _evi([], rep), metric_names=())
    assert res["limited"] is expected


def test_eval_limited_non_bool_refused(tmp_path):
    base = str(tmp_path / "base")
    rep = _eval(tmp_path / "report.json", checkpoint=base, base=base, limited="yes")
    err = _error(_identity(), _evi([], rep))
    assert err.rule_id == "AR-IN-011" and "eval_limited_not_bool:yes" in err.message


@pytest.mark.parametrize("raw,expected", [
    ("deadbeef", "sha256:deadbeef"),
    ("sha256:deadbeef", "sha256:deadbeef"),
])
def test_eval_fingerprint_padding(tmp_path, raw, expected):
    base = str(tmp_path / "base")
    rep = _eval(tmp_path / "report.json", checkpoint=base, base=base, fingerprint=raw)
    res = derive_result(_identity(), _evi([], rep), metric_names=())
    assert res["eval_policy_fingerprint"] == expected


def test_eval_fingerprint_missing_refused(tmp_path):
    base = str(tmp_path / "base")
    rep = _eval(tmp_path / "report.json", checkpoint=base, base=base, fingerprint=_ABSENT)
    err = _error(_identity(), _evi([], rep))
    assert err.rule_id == "AR-IN-011" and "eval_policy_fingerprint_missing" in err.message


def test_metrics_and_metric_sources(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    low = 21 / 128
    man1 = _rl(tmp_path / "a" / "fskills_rl_manifest.json", checkpoint=str(final), measured=low)
    man2 = _rl(tmp_path / "b" / "fskills_rl_manifest.json", checkpoint=str(final), measured=0.7)
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(_identity(), _evi([man1, man2], rep), metric_names=_METRICS, rl_floor=0.5)
    assert res["metrics"]["arc_easy_acc"] == {"value": 0.5, "se": 0.01}
    entry = res["metrics"]["rl_measured_fraction"]
    assert entry["se"] == 0.0 and entry["value"] == pytest.approx(low)  # min over rl manifests
    assert res["metric_sources"] == {"arc_easy_acc": "eval_report", "rl_measured_fraction": "run_manifest"}
    assert res["derived"]["measured_fraction"] == pytest.approx(low)
    assert res["derived"]["min_measured_fraction"] == 0.5
    assert res["steps"] == 24 and res["derived"]["checkpoint"] == str(final)


def test_asserted_and_unknown_metrics(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(_identity(metrics={"loss": {"value": 0.3, "se": None}}),
                        _evi([man], rep), metric_names=("arc_easy_acc", "mystery"))
    assert res["metrics"]["arc_easy_acc"] == {"value": 0.5, "se": 0.01}
    assert res["metrics"]["mystery"] == {"value": None, "se": None}
    assert res["metrics"]["loss"] == {"value": 0.3, "se": None}  # kept verbatim (asserted)
    assert res["metric_sources"] == {"arc_easy_acc": "eval_report", "mystery": "absent", "loss": "asserted"}


def test_matching_claims_accepted(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(
        _identity(status="ok", limited=False, steps=12, eval_policy_fingerprint=_FPRINT,
                  metrics={"arc_easy_acc": {"value": 0.5, "se": 0.01}}),
        _evi([man], rep), metric_names=("arc_easy_acc",))
    assert res["status"] == "ok"


@pytest.mark.parametrize("field,value,needle", [
    ("status", "unmeasured", "assertion_contradicts_evidence:status"),
    ("limited", True, "assertion_contradicts_evidence:limited"),
    ("steps", 13, "assertion_contradicts_evidence:steps"),
    ("eval_policy_fingerprint", "sha256:" + "cd" * 32,
     "assertion_contradicts_evidence:eval_policy_fingerprint"),
])
def test_scalar_contradictions_refused(tmp_path, field, value, needle):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    err = _error(_identity(**{field: value}), _evi([man], rep), metric_names=_METRICS)
    assert err.rule_id == "AR-IN-012" and needle in err.message


@pytest.mark.parametrize("name,claim,needle", [
    ("arc_easy_acc", {"value": 0.4, "se": 0.01}, "assertion_contradicts_evidence:metrics:arc_easy_acc"),
    ("arc_easy_acc", {"value": 0.5, "se": 0.2}, "assertion_contradicts_evidence:metrics:arc_easy_acc"),
    ("arc_easy_acc", "not-a-mapping", "assertion_contradicts_evidence:metrics:arc_easy_acc"),
    ("rl_measured_fraction", {"value": 0.3, "se": 0.0},
     "assertion_contradicts_evidence:metrics:rl_measured_fraction"),
])
def test_metric_contradictions_refused(tmp_path, name, claim, needle):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    err = _error(_identity(metrics={name: claim}), _evi([man], rep), metric_names=_METRICS)
    assert err.rule_id == "AR-IN-012" and needle in err.message


@pytest.mark.parametrize("min_floor,needle", [
    (0.25, "trained_under_different_floor:0.25"),
    (_ABSENT, "trained_under_different_floor:None"),
])
def test_rl_floor_mismatch_refused(tmp_path, min_floor, needle):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final), min_floor=min_floor)
    rep = _eval(tmp_path / "report.json", checkpoint=str(final))
    err = _error(_identity(), _evi([man], rep), rl_floor=0.5)
    assert err.rule_id == "AR-IN-012" and needle in err.message


def _broken(tmp_path, case):
    out = tmp_path / "out"
    (out / "final").mkdir(parents=True, exist_ok=True)
    final = str(out / "final")
    good_eval = _eval(tmp_path / "report.json", checkpoint=final, base=final)
    if case == "manifest_missing":
        return _evi([str(tmp_path / "sub" / "fskills_rl_manifest.json")], good_eval)
    if case == "manifest_not_json":
        bad = tmp_path / "fskills_rl_manifest.json"
        bad.write_text("[1, 2]", encoding="utf-8")
        return _evi([str(bad)], good_eval)
    if case == "manifest_unrecognised":
        return _evi([_write(tmp_path / "notes.json", {"status": "PASS"})], good_eval)
    if case == "train_verdict":
        return _evi([_rl(tmp_path / "fskills_rl_manifest.json", status="FAIL", checkpoint=final)], good_eval)
    if case == "bad_steps":
        return _evi([_rl(tmp_path / "fskills_rl_manifest.json", steps="lots", checkpoint=final)], good_eval)
    if case == "eval_missing":
        return _evi([], str(tmp_path / "absent.json"))
    if case == "eval_not_object":
        bad = tmp_path / "raw.json"
        bad.write_text('"hello"', encoding="utf-8")
        return _evi([], str(bad))
    if case == "eval_verdict":
        return _evi([], _eval(tmp_path / "report2.json", verdict="FAIL", checkpoint=final, base=final))
    if case in ("log_missing", "log_no_done"):
        logs = None if case == "log_missing" else ("[fs:train:step] 3 ok",)
        man = _sft(tmp_path, out, logs=logs)
        return _evi([man], _eval(tmp_path / "report3.json", checkpoint=final))
    raise AssertionError(case)


@pytest.mark.parametrize("case,needle", [
    ("manifest_missing", "evidence_file_unreadable"),
    ("manifest_not_json", "evidence_file_unreadable"),
    ("manifest_unrecognised", "evidence_manifest_unrecognised:notes.json"),
    ("train_verdict", "training_verdict_not_recordable:FAIL"),
    ("bad_steps", "evidence_malformed:steps_measured:lots"),
    ("eval_missing", "evidence_file_unreadable"),
    ("eval_not_object", "evidence_file_unreadable"),
    ("eval_verdict", "eval_verdict_not_recordable:FAIL"),
    ("log_missing", "evidence_file_unreadable"),
    ("log_no_done", "training_verdict_not_recordable:fs:train:done_line_missing"),
])
def test_ar_in_011_evidence_failures(tmp_path, case, needle):
    err = _error(_identity(), _broken(tmp_path, case))
    assert err.rule_id == "AR-IN-011" and needle in err.message


@pytest.mark.parametrize("identity,needle", [
    ({"role": "train", "seed": 1}, "identity_missing_trial"),
    ({"trial": "t1", "seed": 1}, "identity_missing_role"),
    ({"trial": "t1", "role": "train", "seed": 1, "metrics": [1]}, "metrics_not_dict"),
])
def test_ar_in_011_identity_malformed(tmp_path, identity, needle):
    base = str(tmp_path / "base")
    rep = _eval(tmp_path / "report.json", checkpoint=base, base=base)
    err = _error(identity, _evi([], rep))
    assert err.rule_id == "AR-IN-011" and needle in err.message


def test_ar_in_011_identity_not_dict():
    err = _error([], None)
    assert err.rule_id == "AR-IN-011" and "identity_not_dict" in err.message


@pytest.mark.parametrize("evidence,needle", [
    (None, "evidence_malformed:evidence_not_dict"),
    ({}, "evidence_malformed:eval_report_absent"),
    ({"run_manifests": "x", "eval_report": "y"}, "evidence_malformed:run_manifests_not_list_of_str"),
    ({"run_manifests": ["ok", 3], "eval_report": "y"}, "evidence_malformed:run_manifests_not_list_of_str"),
])
def test_ar_in_011_evidence_malformed(evidence, needle):
    err = _error(_identity(), evidence)
    assert err.rule_id == "AR-IN-011" and needle in err.message


def test_rehash_problems_unchanged_and_non_bound(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    man = _rl(tmp_path / "fskills_rl_manifest.json", checkpoint=str(final))
    rep = _eval(tmp_path / "report.json", checkpoint=str(final), benchmarks=_BENCH)
    res = derive_result(_identity(), _evi([man], rep), metric_names=_METRICS)
    assert rehash_problems(res) == []
    assert rehash_problems({"evidence_class": "asserted", "evidence": {}}) == []
    assert rehash_problems({}) == []


def test_rehash_problems_missing_changed(tmp_path):
    out = tmp_path / "out"
    (out / "final").mkdir(parents=True)
    man = _sft(tmp_path, out, logs=("[fs:train:done]          PASS",))
    log = Path(tmp_path) / "fskills_launch.log"
    rep = _eval(tmp_path / "report.json", checkpoint=str(out / "final"))
    res = derive_result(_identity(), _evi([man], rep), metric_names=())
    man_sha = res["evidence"]["run_manifests"][0]["sha256"]
    log_sha = res["evidence"]["launch_logs"][0]["sha256"]
    rep_sha = res["evidence"]["eval_report"]["sha256"]
    Path(man).unlink()
    Path(rep).write_text(Path(rep).read_text(encoding="utf-8") + " ", encoding="utf-8")
    log.write_text(log.read_text(encoding="utf-8") + "\n[fs:train:done]  PASS\n", encoding="utf-8")
    problems = rehash_problems(res)
    assert problems == [f"evidence_file_missing:{man_sha.split(':', 1)[-1][:12]}",
                        f"evidence_file_changed:{rep_sha.split(':', 1)[-1][:12]}",
                        f"evidence_file_changed:{log_sha.split(':', 1)[-1][:12]}"]


# --- review fixes (pool review of the M6a diff) -------------------------------------------------


def _rl_review(tmp_path, ckpt, **fields):
    payload = {"status": "PASS", "steps_measured": 10, "checkpoint": f"saved: {ckpt}",
               "measured_fraction": 0.8, "min_measured_fraction": 0.5, **fields}
    path = tmp_path / "fskills_rl_manifest.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _report_review(tmp_path, ckpt, score=0.6):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"verdict": "PASS", "checkpoint": str(ckpt), "policy": {"fingerprint": "sha256:" + "a" * 64},
                                "benchmarks": [{"name": "m", "score": score, "stderr": 0.01}]}))
    return str(path)


def _derive_review(tmp_path, ckpt, score=0.6, **fields):
    evidence = {"run_manifests": [_rl_review(tmp_path, ckpt, **fields)], "eval_report": _report_review(tmp_path, ckpt, score)}
    return derive_result({"trial": "t", "role": "confirm", "seed": 1}, evidence, metric_names=["m_acc"])


def test_pass_manifest_below_its_floor_is_unmeasured(tmp_path):
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    assert _derive_review(tmp_path, ckpt)["status"] == "ok"
    assert _derive_review(tmp_path, ckpt, measured_fraction=0.1)["status"] == "unmeasured"


@pytest.mark.parametrize("steps", [None, True, "12", 1.5])
def test_steps_measured_must_be_declared_int(tmp_path, steps):
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    fields = {"steps_measured": steps} if steps is not None else {"steps_measured": None}
    with pytest.raises(EvidenceError) as err:
        _derive_review(tmp_path, ckpt, **fields)
    assert err.value.rule_id == "AR-IN-011"


def test_rl_checkpoint_absent_on_disk_is_unmeasured(tmp_path):
    assert _derive_review(tmp_path, tmp_path / "deleted")["status"] == "unmeasured"


def test_non_numeric_benchmark_score_refused(tmp_path):
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    with pytest.raises(EvidenceError) as err:
        _derive_review(tmp_path, ckpt, score="97%")
    assert err.value.rule_id == "AR-IN-011"
