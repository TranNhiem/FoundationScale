"""Exit codes 0, 1 and 3 as exact integers, because the retry policy reads them.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    DENSE_CFG,
    _census_file,
    _dense_base_with_ckpt,
    _dense_full_tensors,
    _healthy_lora,
    _make_base,
    _materialize_artifact,
    _write_cfg,
    json,
    lsg_cli,
)


class TestExitCodes:
    def test_cli_clear_is_exactly_zero_and_report_roundtrips(self, tmp_path, capsys):
        """[PASSES-BEFORE] Red if: main() returns d.exit_code is replaced by
        a truthiness collapse (`return 0 if d.ok else 1` would still pass here
        but break nothing else -- the REAL red-maker: assert instead paired
        with test_cli_missing_* pinning 3 distinctly, which that collapse
        breaks). Report carries denominators on the wire.
        fix45: the CLI lora path now demands --adapter-modules (#78); the
        argv below carries the honest census -- the 12 artifact-namespace
        module stems implied by the run's declared structure (6 dense
        layers x 2 targets), written outside the judged tree. This also
        keeps the new flag's own plumbing pinned: drop the pass-through in
        main() and this test dies on the refusal exit, green-laundering
        impossible. Assertions unchanged."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        out = tmp_path / "report.json"
        census = _census_file(
            tmp_path,
            [f"layers.{i}.self_attn.{w}" for i in range(6) for w in ("q_proj", "v_proj")],
        )
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "lora",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(cfg),
                "--adapter-prefix",
                "",
                "--adapter-modules",
                str(census),
                "--json",
                str(out),
            ]
        )
        assert code == 0
        rpt = json.loads(out.read_text(encoding="utf-8"))
        assert rpt["exit_code"] == 0
        # #80: report inventory = physical artifact (31 real: 24 adapter +
        # 6 optimizer.* + 1 rng_state); the judged adapter population
        # stays 24; exit-code discrimination (0 CLEAR / 1 blocked /
        # 3 unmeasured) is untouched.
        assert rpt["inventory"]["real_tensors"] == 31
        assert rpt["inventory"]["base_tensors"] == 12
        assert len(rpt["controls"]) == 3
        assert rpt["declared_basis"]["run_kind"]

    def test_cli_blocked_is_exactly_one(self, tmp_path):
        """[PASSES-BEFORE] Red if: blocking-reason assembly is bypassed
        (`exit_code = EXIT_CLEAR if not blocking else ...` style rewrite that
        forgets cross-check reasons)."""
        base, _ = _dense_base_with_ckpt(tmp_path)
        truncated = dict(list(_dense_full_tensors().items())[:4])
        ckpt = _materialize_artifact(tmp_path, truncated, name="trunc")
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "full",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(_write_cfg(tmp_path, {})),
            ]
        )
        assert code == 1

    def test_cli_unmeasured_is_exactly_three_for_missing_checkpoint(self, tmp_path):
        """[PASSES-BEFORE] 3, not 1: the launcher retries measurement
        failures, it does not treat them as verdicts. Red if: the
        `except GateUnmeasured` mapping to EXIT_UNMEASURED is changed to 1."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        code = lsg_cli.main(
            [
                str(tmp_path / "no-such-ckpt"),
                "--run-kind",
                "full",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(_write_cfg(tmp_path, {})),
            ]
        )
        assert code == 3

    def test_cli_unmeasured_three_for_missing_base(self, tmp_path):
        """[PASSES-BEFORE] Independent source A absent = cannot measure. Red
        if: BaseModel.load's missing-dir raise is weakened to an empty model."""
        ckpt = _materialize_artifact(tmp_path, _dense_full_tensors())
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "full",
                "--base-model-dir",
                str(tmp_path / "no-base"),
                "--train-config",
                str(_write_cfg(tmp_path, {})),
            ]
        )
        assert code == 3

    def test_cli_unmeasured_three_for_missing_train_config_path(self, tmp_path):
        """[PASSES-BEFORE] The asymmetry S2 names: ABSENT flag is tolerated,
        a SUPPLIED path that does not exist is a measurement failure. Red if:
        _load_train_config's is_file() guard is deleted."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "full",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(tmp_path / "ghost.json"),
            ]
        )
        assert code == 3

    def test_cli_unmeasured_three_for_unparseable_rank(self, tmp_path):
        """[PASSES-BEFORE] A rank value the tool cannot read must not be
        coerced to a default. Red if: the int() coercion grows a try/except-
        pass (i.e., the existing GateUnmeasured raise is what saves it)."""
        base, ckpt, _cfg = _healthy_lora(tmp_path)
        bad = _write_cfg(
            tmp_path, {"peft_scheme": "lora", "lora_rank": "eight"}, name="bad-rank.json"
        )
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "lora",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(bad),
            ]
        )
        assert code == 3

    def test_cli_unmeasured_three_for_unreadable_artifact(self, tmp_path):
        """[PASSES-BEFORE] Garbage where the checkpoint should be is a
        measurement failure, never a verdict. Red if: _measure's
        GateUnmeasured conversion is narrowed to swallow and return None."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        junk = tmp_path / "junk-ckpt"
        junk.mkdir()
        (junk / "model.safetensors").write_bytes(b"not a safetensors file")
        code = lsg_cli.main(
            [
                str(junk),
                "--run-kind",
                "full",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(_write_cfg(tmp_path, {})),
            ]
        )
        assert code == 3

    def test_cli_tool_bug_is_three_not_a_verdict(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] An unexpected exception inside adjudication is
        exit 3 with the 'a tool bug is not a checkpoint verdict' framing. Red
        if: the broad `except Exception` mapping is deleted (the traceback
        would then escape main and the process would exit 1 via the
        interpreter -- a verdict-shaped accident)."""
        monkeypatch.setattr(
            lsg_cli,
            "adjudicate_checkpoint",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        base, ckpt, cfg = _healthy_lora(tmp_path)
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "lora",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(cfg),
            ]
        )
        assert code == 3
