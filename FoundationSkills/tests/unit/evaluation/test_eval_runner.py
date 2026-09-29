from __future__ import annotations

import json
import sys

import pytest

from foundationskills.skills.evaluation.runner import (
    HarnessRequest,
    SubprocessLmEvalRunner,
    build_argv,
    extract_metric,
    offline_env,
)


def _req(tmp_path, **kw):
    base = dict(role="checkpoint", model_dir="/m/base", task="mmlu", output_path=str(tmp_path / "o"),
                eval_cache=str(tmp_path / "cache"))
    base.update(kw)
    return HarnessRequest(**base)


def test_offline_env_forces_offline_and_drops_cache_overrides():
    env = offline_env("/c", {"HF_DATASETS_CACHE": "/elsewhere", "HF_HUB_CACHE": "/x", "PATH": "/bin"})
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HOME"] == "/c" and env["TOKENIZERS_PARALLELISM"] == "false"
    assert "HF_DATASETS_CACHE" not in env and "HF_HUB_CACHE" not in env and env["PATH"] == "/bin"


def test_argv_for_adapter_with_chat_template(tmp_path):
    argv = build_argv(_req(tmp_path, adapter_dir="/runs/final", num_fewshot=5, chat_template=True,
                           fewshot_as_multiturn=True, gen_kwargs="temperature=0", limit=8))
    assert argv[:4] == ["lm-eval", "run", "--model", "hf"]
    assert argv[argv.index("--model_args") + 1] == "pretrained=/m/base,peft=/runs/final,dtype=bfloat16"
    for flag in ("--apply_chat_template", "--fewshot_as_multiturn", "--log_samples"):
        assert flag in argv
    assert argv[argv.index("--num_fewshot") + 1] == "5"
    assert argv[argv.index("--limit") + 1] == "8"


def test_argv_without_chat_template_never_adds_multiturn(tmp_path):
    argv = build_argv(_req(tmp_path, fewshot_as_multiturn=True))
    assert "--apply_chat_template" not in argv and "--fewshot_as_multiturn" not in argv
    assert "--num_fewshot" not in argv and "--limit" not in argv


def test_paths_that_break_model_args_are_refused(tmp_path):
    with pytest.raises(ValueError, match="precondition failed"):
        build_argv(_req(tmp_path, model_dir="/m/a,b"))


RESULTS = {
    "results": {"mmlu": {"acc,none": 0.61, "acc_stderr,none": 0.004},
                "gsm8k": {"exact_match,strict-match": 0.3, "exact_match_stderr,strict-match": "N/A"}},
    "n-samples": {"mmlu": {"original": 14042, "effective": 14042}},
}


def test_extract_metric_with_filter_stderr_and_n():
    assert extract_metric(RESULTS, "mmlu", "acc,none") == {"score": 0.61, "stderr": 0.004, "n": 14042}


def test_extract_metric_non_numeric_stderr_is_none():
    assert extract_metric(RESULTS, "gsm8k", "exact_match,strict-match")["stderr"] is None


def test_extract_metric_absent_task_or_metric_is_none():
    assert extract_metric(RESULTS, "hellaswag", "acc,none") is None
    assert extract_metric(RESULTS, "mmlu", "acc_norm,none") is None
    assert extract_metric(None, "mmlu", "acc,none") is None


def _fake_harness(tmp_path, body: str):
    script = tmp_path / "fake_lm_eval"
    script.write_text(f"#!{sys.executable}\nimport json, os, sys\nargv = sys.argv\n{body}\n")
    script.chmod(0o755)
    return str(script)


def test_subprocess_runner_reads_newest_results_and_is_offline(tmp_path):
    body = (
        "out = argv[argv.index('--output_path') + 1]\n"
        "d = os.path.join(out, 'model'); os.makedirs(d, exist_ok=True)\n"
        "json.dump({'results': {'mmlu': {'acc,none': 0.5}}, 'env': os.environ['HF_HUB_OFFLINE']},"
        " open(os.path.join(d, 'results_2026.json'), 'w'))\n"
        "print('done')\n"
    )
    outcome = SubprocessLmEvalRunner(_fake_harness(tmp_path, body)).run(_req(tmp_path))
    assert outcome.returncode == 0 and outcome.error is None
    assert outcome.results["env"] == "1"
    assert extract_metric(outcome.results, "mmlu", "acc,none")["score"] == 0.5


def test_subprocess_runner_failure_carries_the_log_tail(tmp_path):
    body = "print(\"DatasetNotFoundError: cais/mmlu not in cache\"); sys.exit(1)"
    outcome = SubprocessLmEvalRunner(_fake_harness(tmp_path, body)).run(_req(tmp_path))
    assert outcome.returncode == 1 and outcome.results is None
    assert "cais/mmlu" in outcome.error


def test_subprocess_runner_exit_zero_without_results_is_an_error(tmp_path):
    outcome = SubprocessLmEvalRunner(_fake_harness(tmp_path, "pass")).run(_req(tmp_path))
    assert outcome.results is None and "no results_*.json" in outcome.error


def test_subprocess_runner_missing_executable(tmp_path):
    outcome = SubprocessLmEvalRunner(str(tmp_path / "nope")).run(_req(tmp_path))
    assert outcome.returncode == 127 and "not found" in outcome.error
