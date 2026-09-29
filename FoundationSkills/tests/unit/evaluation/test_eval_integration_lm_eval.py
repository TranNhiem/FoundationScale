"""Real lm-eval, end to end, on CPU and offline.

A tiny random Llama with a word-level tokenizer is built in tmp, and a custom
multiple-choice task reads a local JSONL through ``--include_path``, so no hub
dataset or network is needed. Skipped (not passed) when lm_eval/torch are
absent: a skip here is not evidence the harness works.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status

lm_eval = pytest.importorskip("lm_eval")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from foundationskills.skills.evaluation.runner import PINNED_LM_EVAL, SubprocessLmEvalRunner, harness_version  # noqa: E402
from foundationskills.skills.evaluation.skill import EvalSkill  # noqa: E402

DOCS = [
    {"question": "red is a", "choices": ["color", "number"], "label": 0},
    {"question": "two is a", "choices": ["color", "number"], "label": 1},
    {"question": "blue is a", "choices": ["color", "number"], "label": 0},
    {"question": "five is a", "choices": ["color", "number"], "label": 1},
]


def _tiny_model(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({w for d in DOCS for w in (d["question"].split() + d["choices"])})
    vocab = {"[UNK]": 0, "[PAD]": 1, "<s>": 2, "</s>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                                bos_token="<s>", eos_token="</s>")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                                   bos_token_id=2, eos_token_id=3, pad_token_id=1)
    transformers.LlamaForCausalLM(cfg).save_pretrained(path)
    fast.save_pretrained(path)
    return path


def _task_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    data = root / "data.jsonl"
    data.write_text("".join(json.dumps(d) + "\n" for d in DOCS))
    (root / "fs_tiny_mcq.yaml").write_text(
        "task: fs_tiny_mcq\n"
        "dataset_path: json\n"
        f"dataset_kwargs:\n  data_files:\n    test: {data}\n"
        "test_split: test\n"
        "output_type: multiple_choice\n"
        "doc_to_text: '{{question}}'\n"
        "doc_to_choice: '{{choices}}'\n"
        "doc_to_target: '{{label}}'\n"
        "metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
    )
    return root


@pytest.mark.slow
def test_real_lm_eval_base_vs_base_passes_then_reuses_the_cached_baseline(tmp_path):
    assert harness_version() == PINNED_LM_EVAL
    exe = Path(sys.executable).with_name("lm-eval")
    if not exe.exists():
        pytest.skip(f"lm-eval executable not next to {sys.executable}")
    model = _tiny_model(tmp_path / "tiny")
    tasks = _task_dir(tmp_path / "tasks")
    cache = tmp_path / "hf_home"
    cache.mkdir()
    policy = tmp_path / "policy.yaml"
    policy.write_text("policy_version: 1\ntasks:\n  fs_tiny_mcq: {metric: 'acc,none', num_fewshot: 0, abs_epsilon: 0.0}\n")
    req = {"checkpoint": str(model), "base": str(model), "benchmarks": ["fs_tiny_mcq"], "policy": str(policy),
           "eval_cache": str(cache), "out": str(tmp_path / "eval" / "eval_report.json"),
           "baseline_cache": str(tmp_path / "baselines"), "include_path": str(tasks), "device": "cpu",
           "dtype": "float32", "batch_size": "1"}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # inherited by the harness subprocess; never touch shared GPUs
    try:
        skill = EvalSkill(runner=SubprocessLmEvalRunner(str(exe), timeout_s=900))
        first = skill.execute(req, SkillContext(workdir=tmp_path))
        report = json.loads(Path(req["out"]).read_text())
        row = report["benchmarks"][0]
        assert first.status is Status.PASS, (first.findings, row)
        assert row["score"] == row["baseline"] and row["n"] == len(DOCS)
        assert row["baseline_provenance"] == "measured" and row["task_hash"]
        assert "--device" in row["argv"] and "--apply_chat_template" not in row["argv"]

        second = skill.execute(req, SkillContext(workdir=tmp_path))
        assert second.status is Status.PASS
        assert json.loads(Path(req["out"]).read_text())["benchmarks"][0]["baseline_provenance"] == "cached"
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
