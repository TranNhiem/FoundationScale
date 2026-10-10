"""Tests for :mod:`foundationscale.rl.megatron.onboard` verdicts, checks and CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers

from foundationscale.rl.megatron import onboard

_RESUMED_LINE = "RESUMED from /x/state/step_000003 at step 4 param_hash=0ec9fbbc49419560"
_LOGITS = torch.tensor(
    [
        [0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 0.25, -0.25],
        [1.0, 0.0, 0.5, -1.0, -2.0, 2.0, -0.25, 0.25],
        [-1.0, 1.5, 0.0, 2.0, 0.5, -0.5, 1.0, -1.0],
        [0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 2.0, -2.0],
    ]
)


class _FixedLogits(torch.nn.Module):
    """Tiny stand-in for a causal LM: fixed logits, no transformers weights involved."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(logits.clone())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        x = args[0]
        return SimpleNamespace(logits=self.weights[: x.shape[1]].unsqueeze(0))


def _picks(ids: list[int]) -> list[float]:
    log_probs = torch.log_softmax(_LOGITS, dim=-1)
    return [float(log_probs[i, token]) for i, token in enumerate(ids[1:])]


def _step(step: int = 1, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "step": step,
        "ratio_mean": 1.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.5,
        "update_ok": True,
        "refit_written": 10,
        "refit_unwritten": 0,
        "param_hash": "abc123",
    }
    record.update(overrides)
    return record


def _save(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "save": "step_000003",
        "step": 3,
        "path": "/x",
        "gates": {"save_complete": "PASS"},
        "ok": True,
    }
    record.update(overrides)
    return record


def test_compare_row_skips_index_zero_and_excludes_pads() -> None:
    ids = [5, 1, 0, 2]
    row = onboard.compare_row([100.0, 0.1, 0.2, 0.3], [0.0, 0.25, 0.5], ids, pad_id=0)
    assert row["n"] == 2
    assert row["max_abs"] == pytest.approx(0.2)
    assert row["mean_abs"] == pytest.approx(0.15)


def test_compare_row_pad_none_keeps_every_token() -> None:
    ids = [5, 1, 0, 2]
    row = onboard.compare_row([100.0, 0.1, 0.2, 0.3], [0.0, 0.25, 0.5], ids, pad_id=None)
    assert row["n"] == 3
    assert row["mean_abs"] == pytest.approx(0.35 / 3)


def test_compare_row_all_pad_row_is_empty() -> None:
    row = onboard.compare_row([0.0, 1.0, 2.0], [0.5, 0.5], [5, 0, 0], pad_id=0)
    assert row["n"] == 0
    assert math.isnan(row["mean_abs"])
    assert math.isnan(row["max_abs"])


def test_hf_token_logprobs_matches_log_softmax_picks() -> None:
    ids = [5, 1, 2, 0]
    got = onboard.hf_token_logprobs(_FixedLogits(_LOGITS), ids)
    assert len(got) == len(ids) - 1
    assert got == pytest.approx(_picks(ids))


def test_check_parity_passes_under_tolerance() -> None:
    rows = [{"n": 3, "mean_abs": 0.01, "max_abs": 0.2}]
    check = onboard.check_parity(rows, mean_tol=0.05, max_tol=1.0)
    assert (check.name, check.verdict) == ("parity", onboard.PASS)


def test_check_parity_fails_over_mean_tolerance() -> None:
    rows = [{"n": 2, "mean_abs": 0.5, "max_abs": 0.5}]
    assert onboard.check_parity(rows, mean_tol=0.05, max_tol=1.0).verdict == onboard.FAIL


def test_check_parity_fails_over_max_tolerance() -> None:
    rows = [{"n": 2, "mean_abs": 0.01, "max_abs": 2.0}]
    assert onboard.check_parity(rows, mean_tol=0.05, max_tol=1.0).verdict == onboard.FAIL


def test_check_parity_unmeasured_without_compared_rows() -> None:
    assert onboard.check_parity([]).verdict == onboard.UNMEASURED
    empty = [{"n": 0, "mean_abs": math.nan, "max_abs": math.nan}]
    assert onboard.check_parity(empty).verdict == onboard.UNMEASURED


def test_check_parity_ignores_rows_with_no_tokens() -> None:
    rows = [
        {"n": 0, "mean_abs": math.nan, "max_abs": math.nan},
        {"n": 2, "mean_abs": 0.01, "max_abs": 0.1},
    ]
    check = onboard.check_parity(rows)
    assert check.verdict == onboard.PASS
    assert "1 rows" in check.detail


def test_check_refit_passes_when_every_step_writes() -> None:
    metrics = [_step(1, refit_written=10), _step(2, refit_written=3)]
    assert onboard.check_refit(metrics).verdict == onboard.PASS


def test_check_refit_fails_when_tensors_stay_unwritten() -> None:
    assert onboard.check_refit([_step(1, refit_unwritten=2)]).verdict == onboard.FAIL


def test_check_refit_fails_when_nothing_is_written() -> None:
    assert onboard.check_refit([_step(1, refit_written=0)]).verdict == onboard.FAIL


def test_check_refit_unmeasured_without_refit_counts() -> None:
    record = _step(1)
    del record["refit_unwritten"]
    assert onboard.check_refit([record]).verdict == onboard.UNMEASURED


def test_check_refit_ignores_save_records() -> None:
    metrics = [_step(1), _save(refit_written=0, refit_unwritten=5)]
    assert onboard.check_refit(metrics).verdict == onboard.PASS


def test_check_step1_passes_on_onpolicy_first_step() -> None:
    first = _step(1, ratio_mean=1.0, clip_fraction=0.0, grad_norm=0.5)
    check = onboard.check_step1([first])
    assert (check.name, check.verdict) == ("step1", onboard.PASS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ratio_mean": 1.5},
        {"clip_fraction": 0.5},
        {"grad_norm": math.nan},
        {"grad_norm": 0.0},
        {"update_ok": False},
    ],
)
def test_check_step1_fails(overrides: dict[str, Any]) -> None:
    assert onboard.check_step1([_step(1, **overrides)]).verdict == onboard.FAIL


def test_check_step1_uses_lowest_step_even_when_unsorted() -> None:
    assert onboard.check_step1([_step(7), _step(2, ratio_mean=9.0)]).verdict == onboard.FAIL
    assert onboard.check_step1([_step(2), _step(7, ratio_mean=9.0)]).verdict == onboard.PASS


def test_check_step1_fails_without_ratio_mean() -> None:
    record = _step(1)
    del record["ratio_mean"]
    assert onboard.check_step1([record]).verdict == onboard.FAIL


def test_check_step1_unmeasured_without_steps() -> None:
    assert onboard.check_step1([]).verdict == onboard.UNMEASURED


def test_check_save_passes_when_ok_and_gates_pass() -> None:
    metrics = [_save(), _save(gates={"save_complete": "PASS", "weights": "PASS"})]
    assert onboard.check_save(metrics).verdict == onboard.PASS


def test_check_save_fails_when_not_ok() -> None:
    assert onboard.check_save([_save(ok=False)]).verdict == onboard.FAIL


def test_check_save_fails_on_weak_gate() -> None:
    gates = {"save_complete": "PASS", "weights": "FAIL"}
    assert onboard.check_save([_save(gates=gates)]).verdict == onboard.FAIL


def test_check_save_unmeasured_without_records() -> None:
    assert onboard.check_save([]).verdict == onboard.UNMEASURED


def test_check_resume_passes_on_matching_hash() -> None:
    metrics = [_step(3, param_hash="0ec9fbbc49419560")]
    assert onboard.check_resume(metrics, _RESUMED_LINE).verdict == onboard.PASS


def test_check_resume_passes_on_prefix_match_either_way() -> None:
    long_hash = "0ec9fbbc49419560abcdef00"
    check = onboard.check_resume([_step(3, param_hash=long_hash)], _RESUMED_LINE)
    assert check.verdict == onboard.PASS
    long_line = _RESUMED_LINE + "abcdef00"
    check = onboard.check_resume([_step(3, param_hash="0ec9fbbc49419560")], long_line)
    assert check.verdict == onboard.PASS


def test_check_resume_fails_on_hash_mismatch() -> None:
    metrics = [_step(3, param_hash="ffffffffffffffff")]
    assert onboard.check_resume(metrics, _RESUMED_LINE).verdict == onboard.FAIL


def test_check_resume_unmeasured_without_resumed_line() -> None:
    metrics = [_step(3, param_hash="0ec9fbbc49419560")]
    assert onboard.check_resume(metrics, "training finished").verdict == onboard.UNMEASURED


def test_check_resume_unmeasured_when_original_lacks_step() -> None:
    metrics = [_step(1, param_hash="0ec9fbbc49419560")]
    assert onboard.check_resume(metrics, _RESUMED_LINE).verdict == onboard.UNMEASURED


def test_verdict_exit_codes() -> None:
    passed = onboard.Check("a", onboard.PASS, "")
    failed = onboard.Check("b", onboard.FAIL, "")
    unmeasured = onboard.Check("c", onboard.UNMEASURED, "")
    assert onboard.verdict_exit([passed, passed]) == 0
    assert onboard.verdict_exit([passed, failed, unmeasured]) == 5
    assert onboard.verdict_exit([passed, unmeasured]) == 95
    assert onboard.verdict_exit([]) == 95


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    metrics = tmp_path / "metrics.jsonl"
    records = [_step(1), _step(3, param_hash="0ec9fbbc49419560"), _save()]
    metrics.write_text("".join(json.dumps(r) + "\n" for r in records))
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps([{"n": 3, "mean_abs": 0.001, "max_abs": 0.01}]))
    resumed = tmp_path / "resumed.log"
    resumed.write_text(_RESUMED_LINE + "\n")
    return metrics, parity, resumed


def test_main_all_pass_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    metrics, parity, resumed = _inputs(tmp_path)
    argv = [
        "--parity",
        str(parity),
        "--metrics",
        str(metrics),
        "--resumed-log",
        str(resumed),
    ]
    assert onboard.main(argv) == 0
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 5
    assert all("PASS" in line for line in lines)
    assert "ONBOARD_VERDICT rc=0" in captured.err


def test_main_without_inputs_is_unmeasured() -> None:
    assert onboard.main([]) == 95


def test_main_fails_on_wide_parity(tmp_path: Path) -> None:
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps([{"n": 2, "mean_abs": 0.5, "max_abs": 0.5}]))
    assert onboard.main(["--parity", str(parity)]) == 5


def test_main_parity_dump_delegates_to_parity_from_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps({"input_ids": [], "logprobs": []}))
    calls: list[tuple[str, str]] = []
    rows: list[dict[str, float]] = [{"n": 2, "mean_abs": 0.0, "max_abs": 0.0}]

    def fake(dump_path: str, hf_model: str) -> list[dict[str, float]]:
        calls.append((dump_path, hf_model))
        return rows

    monkeypatch.setattr(onboard, "parity_from_dump", fake)
    assert onboard.main(["--parity-dump", str(dump), "--hf-model", "tiny-hf"]) == 95
    assert calls == [(str(dump), "tiny-hf")]


def test_parity_from_dump_excludes_pads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ids = [5, 1, 2, 0]
    picks = _picks(ids)
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps({"input_ids": [ids], "logprobs": [[0.0, *picks]]}))
    model = _FixedLogits(_LOGITS)

    def fake_tokenizer(_name: str) -> SimpleNamespace:
        return SimpleNamespace(pad_token_id=None, eos_token_id=0)

    def fake_model(_name: str, **_kwargs: Any) -> _FixedLogits:
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)
    rows = onboard.parity_from_dump(str(dump), "tiny-hf")
    assert len(rows) == 1
    assert rows[0]["n"] == 2  # ids[2] == 0 is the pad/eos token
    assert rows[0]["max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["mean_abs"] == pytest.approx(0.0, abs=1e-6)
