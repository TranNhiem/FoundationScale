"""Fail-closed contract tests for tools/preflight.py.

The module is loaded by path (tools/ is not a package), which also proves the
login-node bootstrap of `foundationscale` works from a bare checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "tools" / "preflight.py")
preflight = importlib.util.module_from_spec(spec)
# Register BEFORE exec_module. @dataclass resolves string annotations through
# sys.modules[cls.__module__], so a module executed while absent from sys.modules
# raises AttributeError on the first frozen dataclass — the tool imports fine as
# __main__ from the CLI and only breaks under a by-path load, which is exactly
# the shape that would have reached the login node untested.
sys.modules[spec.name] = preflight
spec.loader.exec_module(preflight)  # fail-before: file absent -> every test reds here


def world(tmp_path):
    return preflight._build_world(tmp_path)


def write_cfg(w, tmp_path):
    """Write w.cfg to preflight.json AND re-derive everything keyed to its bytes.

    `_build_world` ends by computing the config sha over the exact bytes it wrote
    (`indent=1`) and stamping the derived manifest hash into ckpt-probe's
    provenance.json. Every test that mutates `w.cfg` and rewrites the file with
    different bytes -- a compact dump, a flipped `schedule.smoke` -- moves the
    config sha, hence the manifest hash, and leaves that stamp behind pointing at
    a manifest that no longer exists. launch_provenance then reports a
    checkpoint/manifest tie failure that is an artifact of the fixture rather
    than of anything under test.

    That mattered in both directions, which is why this is centralized rather
    than patched at the one loud site: the healthy-launch test failed on it
    visibly, but the BLOCKED-expecting tests were at risk of passing off a stale
    hash instead of the defect each one claims to pin -- a control that fires for
    the wrong reason is not a control. Re-deriving with the tool's own helper
    keeps the pin measured rather than asserted, exactly as the world builder
    does it.
    """
    p = tmp_path / "preflight.json"
    p.write_text(json.dumps(w.cfg), encoding="utf-8")
    cfg_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_sha, _payload = preflight._manifest_hash_for(w.cfg["frozen"], cfg_sha)
    (Path(w.root) / "ckpt-probe" / "provenance.json").write_text(
        json.dumps({"manifest_hash": manifest_sha}), encoding="utf-8"
    )
    w._cfg_sha = cfg_sha
    return p


def run_check(w, check_id, registry=None):
    shared = {"_config_sha256": w._cfg_sha}
    if check_id == "frozen_manifest":
        # There is no healthy-precondition baseline to prove FOR the baseline
        # itself: a lane whose defect lands on an artifact frozen_manifest
        # reads made the helper's insistence on a PASSing baseline fire before
        # the assertion under test ever ran — test 8's measured red was this
        # helper assert, not the check's verdict. Mirrors run_self_test's own
        # special case (`if chk.id == "frozen_manifest": res = baseline`).
        return preflight._execute(
            preflight.REGISTRY[check_id], w.cfg, w.env, shared, registry=registry
        )
    base = preflight._execute(preflight.REGISTRY["frozen_manifest"], w.cfg, w.env, shared)
    assert base.verdict is preflight.Verdict.PASS, base.detail
    return preflight._execute(preflight.REGISTRY[check_id], w.cfg, w.env, shared, registry=registry)


# 1. Pins: the registry ships exactly the ten design checks, in design order.
#    Red line: delete any one `_mk(...)` registration.
def test_registry_is_the_ten_design_checks():
    assert list(preflight.REGISTRY) == [
        "frozen_manifest",
        "template_audit",
        "corpus_wiring",
        "verdict_schema",
        "conversion_coverage",
        "lora_probe",
        "schedule_consistency",
        "evidence_completeness",
        "training_dynamics",
        "launch_provenance",
    ]


# 2. Pins: clearance requires AT LEAST ONE result and ALL PASS with checked>0;
#    SKIP does not clear (design item 4 verbatim).
#    Red line: change `all(...)` to `any(...)` in _is_clear, or replace the
#    predicate with `not r.verdict.blocking` (revives SKIP-passes).
def test_clearance_algebra():
    def mk(v, n):
        return preflight.CheckResult("x", "x", v, preflight.Coverage(n, "units"), "", {"n": n})

    assert not preflight._is_clear([])
    assert preflight._is_clear([mk(preflight.Verdict.PASS, 3)])
    assert not preflight._is_clear([mk(preflight.Verdict.PASS, 3), mk(preflight.Verdict.SKIP, 1)])
    assert not preflight._is_clear([mk(preflight.Verdict.PASS, 0)])


# 3. Pins: the PASS downgrade ladder (vacuous/short/over) and the empty-evidence
#    guard. Red line: delete the `coverage.is_vacuous` branch in _finalize, or
#    the `not res.evidence` guard in _discipline.
def test_pass_downgrade_ladder_and_evidence_guard():
    r = preflight._finalize(
        "c", "t", preflight.Verdict.PASS, preflight.Coverage(0, "units"), "d", {"a": 1}
    )
    assert r.verdict is preflight.Verdict.VACUOUS and "0 units" in r.detail
    r = preflight._finalize(
        "c", "t", preflight.Verdict.PASS, preflight.Coverage(3, "units", expected=9), "d", {"a": 1}
    )
    assert r.verdict is preflight.Verdict.UNDERCOVERED
    r = preflight._finalize(
        "c", "t", preflight.Verdict.PASS, preflight.Coverage(9, "units", expected=3), "d", {"a": 1}
    )
    assert r.verdict is preflight.Verdict.OVERCOVERED
    bare = preflight._finalize(
        "c", "t", preflight.Verdict.PASS, preflight.Coverage(1, "u"), "d", {}
    )
    assert preflight._discipline(bare).verdict is preflight.Verdict.ERROR


# 4. Pins: a healthy world clears END-TO-END — exit 0, banner embeds the
#    manifest hash, the JSON record carries it, and the denominator line shows
#    10/10. Red line: drop manifest_sha256 from the banner f-string, or let
#    _render_report skip the summary line.
def test_healthy_launch_clears_with_manifest_banner(tmp_path, capsys):
    w = world(tmp_path)
    w.cfg["schedule"]["smoke"] = False
    write_cfg(w, tmp_path)
    old = os.environ.copy()
    os.environ.update(w.env)
    try:
        rc = preflight.main(
            ["--config", str(tmp_path / "preflight.json"), "--json", str(tmp_path / "record.json")]
        )
    finally:
        os.environ.clear()
        os.environ.update(old)
    out = capsys.readouterr().out
    assert rc == 0
    assert "10/10 checks PASS" in out
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["overall"] == "CLEAR"
    assert rec["manifest_sha256"] in out  # the banner ties the clearance to the frozen manifest
    sha, _ = preflight._manifest_hash_for(w.cfg["frozen"], rec["config_sha256"])
    assert rec["manifest_sha256"] == sha


# 5. Pins: zero-check refusal names 0. Red line: delete the `if not selected:`
#    guard so main proceeds to `all([])`-shaped clarity.
def test_zero_selection_blocks_naming_zero(tmp_path, capsys):
    w = world(tmp_path)
    write_cfg(w, tmp_path)
    rc = preflight.main(
        [
            "--config",
            str(tmp_path / "preflight.json"),
            "--only",
            "frozen_manifest",
            "--exclude",
            "frozen_manifest",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1 and "0 of 10" in out


# 6. Pins: unknown selection ids block and name the ids. Red line: drop the
#    `unknown` computation (a typo would then vanish into a partial sweep).
def test_unknown_check_id_blocks_named(tmp_path, capsys):
    w = world(tmp_path)
    write_cfg(w, tmp_path)
    rc = preflight.main(
        ["--config", str(tmp_path / "preflight.json"), "--only", "frozzen_manifest"]
    )
    out = capsys.readouterr().out
    assert rc == 1 and "frozzen_manifest" in out and "0 checks examined" in out


# 7. Pins: missing key -> exit 1 naming the dotted key; unknown key -> same;
#    unparseable JSON -> exit 2. Red lines: remove the key from _SCHEMA /
#    delete the unknown-keys loop / route ToolError to EXIT_BLOCKED.
def test_config_fail_closed(tmp_path, capsys):
    w = world(tmp_path)
    cfg = json.loads(json.dumps(w.cfg))
    del cfg["dynamics"]["bands"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    assert preflight.main(["--config", str(p)]) == 1
    assert "dynamics.bands" in capsys.readouterr().out

    cfg = json.loads(json.dumps(w.cfg))
    cfg["schedule"]["train_iters_typoed"] = 1
    p.write_text(json.dumps(cfg))
    assert preflight.main(["--config", str(p)]) == 1
    assert "unknown config key" in capsys.readouterr().out

    p.write_text("{ not json")
    assert preflight.main(["--config", str(p)]) == 2


# 8. Pins (integration MUST_FIRE direct): frozen_manifest flips on tampered
#    corpus bytes and on a missing shard. Red line: comment out
#    `if sha != entry["sha256"]: mismatches.append(...)`.
def test_frozen_manifest_fire_lanes(tmp_path):
    w = world(tmp_path)
    preflight.REGISTRY["frozen_manifest"].lanes[0].apply(w)
    r = run_check(w, "frozen_manifest")
    assert r.verdict is preflight.Verdict.FAIL
    assert r.evidence["corpus"]["files"][0]["sha_matches_pin"] is False


# 9. Pins: template containment + KEEP_COT pin. Red line: delete the
#    `not (m_lo <= c_lo and c_hi <= m_hi)` comparison, or invert the keep=="0" test.
def test_template_audit_lanes(tmp_path):
    w = world(tmp_path)
    w.probe_sick(True)
    r = run_check(w, "template_audit")
    assert r.verdict is preflight.Verdict.FAIL and "escapes masked_span" in r.detail
    w2 = world(tmp_path / "w2")
    w2.env["FOXBRAIN_GEMMA4_KEEP_COT"] = "0"
    r2 = run_check(w2, "template_audit")
    assert r2.verdict is preflight.Verdict.FAIL and "KEEP_COT" in r2.detail


# 10. Pins: corpus wiring machine-verifies the batch-0 attestation and the
#     recipe grep; env drift FAILS with the phantom named. Red lines: replace
#     the `claimed != actual` comparison with `False`, or drop the
#     `recipe_hits` any()-test.
def test_corpus_wiring_attestation_and_recipe(tmp_path):
    w = world(tmp_path)
    ok = run_check(w, "corpus_wiring")
    assert ok.verdict is preflight.Verdict.PASS
    assert (
        ok.evidence["attestation"]["recomputed_sample_sha256"]
        == ok.evidence["attestation"]["claimed_sample_sha256"]
    )
    preflight.REGISTRY["corpus_wiring"].lanes[0].apply(w)  # recipe loses _env_jsonls(
    r = run_check(w, "corpus_wiring")
    assert r.verdict is preflight.Verdict.FAIL and "_env_jsonls(" in r.detail
    w2 = world(tmp_path / "w2")
    preflight.REGISTRY["corpus_wiring"].lanes[1].apply(w2)
    r2 = run_check(w2, "corpus_wiring")
    assert r2.verdict is preflight.Verdict.FAIL and "phantom" in r2.detail


# 11. Pins: conversion coverage catches a silently dropped tensor and an
#     out-of-band iter-1 loss; allow-list grounding passes when true. Red
#     lines: skip the `uncovered` accumulation loop, or drop the band check.
def test_conversion_lanes(tmp_path):
    w = world(tmp_path)
    ok = run_check(w, "conversion_coverage")
    assert ok.verdict is preflight.Verdict.PASS
    assert ok.coverage.checked == ok.coverage.expected == 3
    preflight.REGISTRY["conversion_coverage"].lanes[0].apply(w)
    r = run_check(w, "conversion_coverage")
    assert r.verdict is preflight.Verdict.FAIL and "neither converted nor allow-listed" in r.detail
    w2 = world(tmp_path / "w2")
    w2.rewrite_conv_metrics(loss=9.9)
    assert run_check(w2, "conversion_coverage").verdict is preflight.Verdict.FAIL


# 12. Pins: merged-HF parity compares against the EXTERNAL PIN (never the
#     self-index): the fixture plants a lying index.json and still passes;
#     appending one byte flips it. Red line: implement parity by reading
#     index.json total_size, or `pinned = observed` (self-referential compare
#     would green both halves).
def test_lora_parity_is_external_pin_never_self_index(tmp_path):
    w = world(tmp_path)
    ok = run_check(w, "lora_probe")
    assert ok.verdict is preflight.Verdict.PASS
    # `== 1500` pinned the author's hand arithmetic over the two shards and
    # forgot the lying index planted in the SAME directory — 57 bytes the
    # check's contract prices as bytes-on-disk while never parsing it. The
    # intent survives, strengthened: observed must equal the EXTERNAL pin the
    # world-builder measured from the bytes it wrote, the pin must provably
    # differ from the self-index's lie, and the walk must have priced all
    # three regular files.
    pinned = w.cfg["lora"]["pinned_merged_total_bytes"]
    observed = ok.evidence["merged"]["observed_bytes"]
    assert observed == pinned != 999999999
    assert ok.evidence["merged"]["files"] == 3  # two shards + the lying index, priced as bytes
    preflight.REGISTRY["lora_probe"].lanes[1].apply(w)
    assert run_check(w, "lora_probe").verdict is preflight.Verdict.FAIL
    w2 = world(tmp_path / "w2")
    w2.lora_log_strip("kv_proj")
    r = run_check(w2, "lora_probe")
    assert r.verdict is preflight.Verdict.FAIL and "kv_proj" in r.detail


# 13. Pins: dynamics hard floor + lr-on-every-row. Red lines: delete the
#     floor sweep or the no_lr guard.
def test_dynamics_lanes(tmp_path):
    w = world(tmp_path)
    w.dynamics_patch(42, loss=0.05)
    r = run_check(w, "training_dynamics")
    assert r.verdict is preflight.Verdict.FAIL and "42" in r.detail
    w2 = world(tmp_path / "w2")
    w2.dynamics_patch(7, lr=None)
    r2 = run_check(w2, "training_dynamics")
    assert r2.verdict is preflight.Verdict.FAIL and "lr" in r2.detail


# 14. Pins: evidence completeness needs exactly world_size live rank logs.
#     Red line: `!=` -> `<` in the log count comparison.
def test_evidence_rank_log_lanes(tmp_path):
    w = world(tmp_path)
    w.logs[0].unlink()
    r = run_check(w, "evidence_completeness")
    assert r.verdict is preflight.Verdict.FAIL and "3 logs for world size 4" in r.detail
    w2 = world(tmp_path / "w2")
    os.utime(w2.logs[1], (946684800, 946684800))
    assert run_check(w2, "evidence_completeness").verdict is preflight.Verdict.FAIL


# 15. Pins: provenance tie + window + static resume guard. A guard file that
#     NAMES the hash but never refuses FAILS. Red line: drop the
#     `blocks` conjunct in guard evaluation, or the embedded-hash comparison.
def test_launch_provenance_lanes(tmp_path):
    w = world(tmp_path)
    ok = run_check(w, "launch_provenance")
    assert ok.verdict is preflight.Verdict.PASS
    preflight.REGISTRY["launch_provenance"].lanes[0].apply(w)
    assert run_check(w, "launch_provenance").verdict is preflight.Verdict.FAIL
    w2 = world(tmp_path / "w2")
    os.utime(w2.merged[0], (946684800, 946684800))
    r2 = run_check(w2, "launch_provenance")
    assert r2.verdict is preflight.Verdict.FAIL and "outside" in r2.detail
    w3 = world(tmp_path / "w3")
    (tmp_path / "w3" / "recipe" / "checkpointing.py").write_text(
        "# mentions manifest_hash, refuses nothing"
    )
    r3 = run_check(w3, "launch_provenance")
    assert r3.verdict is preflight.Verdict.FAIL and "has_refusal=False" in r3.detail


# 16. Pins: --self-test exits 0 and reports BOTH denominators, fully proven.
#     Red line: skip the MUST_PASS loop (`if False:`) — pass_proven collapses
#     while the exit math stays green unless denominators are asserted.
def test_self_test_proves_both_halves_with_denominators(capsys):
    code, report = preflight.run_self_test(out=lambda s: None)
    assert code == 0
    assert report["checks_total"] == 10
    assert report["must_pass_proven"] == report["must_pass_total"] == 10
    assert report["must_fire_proven"] == report["must_fire_total"] > 0
    assert report["failures"] == []


# 17. Pins: a check shipping no MUST_FIRE lane fails the self-test BY NAME.
#     Red line: delete the `if not chk.lanes:` failure branch.
def test_self_test_flags_check_without_fire_lanes(monkeypatch, capsys):
    victim = preflight.REGISTRY["schedule_consistency"]
    monkeypatch.setattr(victim, "lanes", ())
    code, report = preflight.run_self_test(out=lambda s: None)
    assert code == 1
    assert any("schedule_consistency" in f and "NO MUST_FIRE" in f for f in report["failures"])


# 18. Pins: the launch-time red team (check 4) flips PASS on the real world and
#     FAILS on a registry whose peer ships no lanes. Red line: treat
#     verdict ERROR as 'flipped' (a dying detector would certify the launch).
def test_verdict_schema_runtime_red_team(tmp_path):
    w = world(tmp_path)
    ok = run_check(w, "verdict_schema", registry=None)
    assert ok.verdict is preflight.Verdict.PASS
    assert ok.coverage.checked == ok.coverage.expected
    w2 = world(tmp_path / "w2")
    w2.registry_override = preflight._doctored_registry_no_lanes()
    shared = {"_config_sha256": w2._cfg_sha}
    base = preflight._execute(preflight.REGISTRY["frozen_manifest"], w2.cfg, w2.env, shared)
    assert base.verdict is preflight.Verdict.PASS
    r = preflight._execute(
        preflight.REGISTRY["verdict_schema"], w2.cfg, w2.env, shared, registry=w2.registry_override
    )
    assert r.verdict is preflight.Verdict.FAIL and "NO MUST_FIRE lane" in r.detail


# 19. Pins: SMOKE runs clear ONLY with the qualifier embedded in the banner
#     (design item 7, second sentence). Red line: drop the smoke branch in the
#     banner block.
def test_smoke_banner_carries_no_correctness_claim(tmp_path, capsys):
    w = world(tmp_path)
    w.cfg["schedule"]["smoke"] = True
    write_cfg(w, tmp_path)
    old = os.environ.copy()
    os.environ.update(w.env)
    try:
        rc = preflight.main(["--config", str(tmp_path / "preflight.json")])
    finally:
        os.environ.clear()
        os.environ.update(old)
    out = capsys.readouterr().out
    assert rc == 0
    assert "CLEAR (SMOKE — this banner makes NO training-correctness claim)" in out


# 20. Pins (defect-1 vocabulary, directly): a declared shard ABSENT from disk
#     is a defect of the ARTIFACT — frozen_manifest must FAIL and NAME it
#     ("absent"), with honest coverage (the absent shard was not examined, and
#     the denominator did not shrink to hide that fact); a shard present but
#     unparseable as safetensors is likewise a NAMED artifact defect
#     ("corrupt"); only an OS-level read refusal (EACCES & friends, injected
#     through the helper's documented __cause__ chaining so the result does not
#     depend on running as non-root) is a defect of the CHECK — ERROR, fail
#     closed. The two classes must be distinguishable from the detail string
#     alone, because the operator response is opposite in the two cases.
#     Red lines on the current tree: both artifact-defect halves return ERROR
#     with an "unreadable" detail — the FAIL and naming assertions below all
#     fail pre-patch and pass post-patch.
def test_absent_or_corrupt_shard_is_named_fail__environmental_refusal_is_error(
    tmp_path, monkeypatch
):
    w = world(tmp_path)
    victim = w.model[0]
    victim.unlink()
    r = run_check(w, "frozen_manifest")
    assert r.verdict is preflight.Verdict.FAIL
    assert "absent" in r.detail and str(victim) in r.detail
    # 1 parsed shard + 4 corpus + 1 run-config examined, out of 2 + 4 + 1
    # declared: the absent shard is not counted, and expected did not move.
    assert (r.coverage.checked, r.coverage.expected) == (6, 7)
    assert {"path": str(victim), "state": "absent"} in r.evidence["model"]["unexamined"]

    w3 = world(tmp_path / "w3")
    w3.model[0].write_bytes(b"junk")  # present on disk, not remotely safetensors
    r3 = run_check(w3, "frozen_manifest")
    assert r3.verdict is preflight.Verdict.FAIL
    assert "corrupt or unparseable" in r3.detail
    assert (r3.coverage.checked, r3.coverage.expected) == (6, 7)

    w2 = world(tmp_path / "w2")

    def refuse(path):
        # The helper's documented OS-refusal shape, injected: ArtifactError
        # with the originating OSError chained as __cause__. chmod(0) would
        # sail past root, so the refusal is simulated, never faked.
        try:
            raise PermissionError(13, "Permission denied", str(path))
        except PermissionError as exc:
            raise preflight.ArtifactError(f"{path}: unreadable: {exc}") from exc

    monkeypatch.setattr(preflight, "_read_safetensors_header", refuse)
    r2 = run_check(w2, "frozen_manifest")
    assert r2.verdict is preflight.Verdict.ERROR
    # Judge the VOCABULARY, not the paths. Every path in this detail lives under
    # the pytest tmp dir, whose name is derived from this test's own name and so
    # contains the literal substring "absent" -- an unredacted `"absent" not in
    # detail` reds on a correct tool, and would have been "fixed" by deleting the
    # very assertion that distinguishes the FAIL vocabulary from the ERROR one.
    vocab = r2.detail.replace(str(w2.root), "<root>")
    assert "unreadable" in vocab and "absent" not in vocab, vocab
