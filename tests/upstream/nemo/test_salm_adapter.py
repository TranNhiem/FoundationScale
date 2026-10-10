"""CI for the SALM worker adapters (PHASE 1.2b of docs/research/upstream_integration.md).

The PURE half of ``foundationscale.upstream.nemo.{salm_finetune,salm_adjudicate,salm_decode}`` runs
here with no NeMo in sight: the loader/optimiser dicts a run used, the CLI, the freeze-contract
verdict lines and the decode prompt/pairing are all plain transforms the no-container CI path
depends on. Every module must also import with ``nemo`` absent -- that is what keeps the pure half
loadable outside a NeMo container at all.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from foundationscale.upstream.nemo.salm_adjudicate import frozen_line, lora_filter, names_line
from foundationscale.upstream.nemo.salm_decode import chat_prompt, pair_rows, prompts_for
from foundationscale.upstream.nemo.salm_finetune import (
    data_config,
    lr_scheduler_config,
    optimizer_config,
    parse_args,
)

_MODULES = (
    "foundationscale.upstream.nemo.salm_finetune",
    "foundationscale.upstream.nemo.salm_adjudicate",
    "foundationscale.upstream.nemo.salm_decode",
)
_AUDIO_TAG = "<|audio|>"


def test_data_config_seed_none_is_todays_dict() -> None:
    """Regression: --seed unset must reproduce today's data cfg dict exactly -- in particular
    ``"seed": 42`` and ``"shard_seed": "randomized"``, the loader of every run before this phase.
    Anything else would make the seeded line no longer sit next to the earlier ones."""
    assert data_config("train_manifest.json", _AUDIO_TAG, "<|en|>", 8, None) == {
        "train_ds": {
            "sample_rate": 16000,
            "prompt_format": "<|en|>",
            "input_cfg": [
                {
                    "type": "lhotse_as_conversation",
                    "manifest_filepath": "train_manifest.json",
                    "audio_locator_tag": _AUDIO_TAG,
                    "tags": {"context": "Transcribe the following:"},
                }
            ],
            "token_equivalent_duration": 0.08,
            "batch_size": 8,
            "shuffle": True,
            "use_bucketing": False,
            "shard_seed": "randomized",
            "num_workers": 4,
            "seed": 42,
            "fault_tolerant_audio_loading": False,
        }
    }


def test_data_config_seed_pins_seed_and_shard_seed() -> None:
    """Regression: one seed must pin BOTH fields and change nothing else -- lhotse's shard shuffle
    was otherwise not reproducible (measured on the AED lane, 2026-10-10), so a run given --seed 7
    had to be given --seed 7 to be re-runable."""
    unseeded = data_config("train_manifest.json", _AUDIO_TAG, "<|en|>", 8, None)
    assert data_config("train_manifest.json", _AUDIO_TAG, "<|en|>", 8, 7) == {
        "train_ds": {
            "sample_rate": 16000,
            "prompt_format": "<|en|>",
            "input_cfg": [
                {
                    "type": "lhotse_as_conversation",
                    "manifest_filepath": "train_manifest.json",
                    "audio_locator_tag": _AUDIO_TAG,
                    "tags": {"context": "Transcribe the following:"},
                }
            ],
            "token_equivalent_duration": 0.08,
            "batch_size": 8,
            "shuffle": True,
            "use_bucketing": False,
            "shard_seed": 7,
            "num_workers": 4,
            "seed": 7,
            "fault_tolerant_audio_loading": False,
        }
    }
    assert list(data_config("train_manifest.json", _AUDIO_TAG, "<|en|>", 8, 7)["train_ds"]) == list(
        unseeded["train_ds"]
    )


def test_optimizer_config_is_the_campaign_dict() -> None:
    """Regression: configure_optimizers() only builds the released AdamW (betas 0.9/0.98,
    weight_decay 1e-3, foreach) from this dict -- a drifted value silently re-tunes the model."""
    assert optimizer_config(1e-4) == {
        "_target_": "torch.optim.AdamW",
        "lr": 1e-4,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "foreach": True,
    }


def test_lr_scheduler_config_is_the_campaign_dict() -> None:
    """Regression: the warmup-Cosine schedule with a 0.1*lr floor is the released recipe; the two
    stringify differently (``nemo.core.optim.lr_scheduler.CosineAnnealing`` vs ``CosineAnnealing``)
    and the wrong name would silently fall back to no schedule at all."""
    # lr 1.0 keeps the floor a plain literal (1.0 * 0.1 is exactly 0.1 in binary float), so the dict
    # comparison below compares values rather than scaled roundoff.
    assert lr_scheduler_config(1.0, 50, 500) == {
        "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
        "warmup_steps": 50,
        "min_lr": 0.1,
        "max_steps": 500,
    }
    assert lr_scheduler_config(1e-4, 50, 500)["min_lr"] == 1e-4 * 0.1


def test_parse_args_keeps_the_campaign_flags_and_adds_seed() -> None:
    """Regression: the CLI surface is the campaign's (same names, same defaults) plus --seed; a
    renamed flag would break every recorded run command, and --seed unset must stay None so the
    loader keeps its earlier defaults."""
    args = parse_args(
        [
            "--model",
            "nvidia/canary-qwen-2.5b",
            "--train",
            "train.jsonl",
            "--out-dir",
            "out",
            "--seed",
            "7",
        ]
    )
    assert (args.model, args.train, args.out_dir) == (
        "nvidia/canary-qwen-2.5b",
        "train.jsonl",
        "out",
    )
    assert (args.max_steps, args.batch_size, args.lr, args.warmup) == (500, 8, 1e-4, 50)
    assert args.seed == 7
    unseeded = parse_args(["--model", "m", "--train", "t", "--out-dir", "o"])
    assert unseeded.seed is None
    assert (unseeded.max_steps, unseeded.batch_size, unseeded.lr, unseeded.warmup) == (
        500,
        8,
        1e-4,
        50,
    )


def test_frozen_line_passes_on_bit_identical_weights() -> None:
    """PASS shape of the freeze contract: the line must read the campaign's byte for byte, or the
    ADJ history stops being comparable line-for-line."""
    line, blocking = frozen_line(
        {"llm.w": "h1", "embed_tokens.w": "h2"},
        {"llm.w": "h1", "embed_tokens.w": "h2"},
    )
    assert line == "[PASS] salm.frozen_llm_unchanged: 2/2 params -- 0 changed 0 new/missing"
    assert blocking is False


def test_frozen_line_reds_on_changed_tensor() -> None:
    """A changed digest is the silent-thaw case: it must block, with the changed names in the
    detail (up to 5)."""
    line, blocking = frozen_line(
        {"llm.w": "h1", "embed_tokens.w": "h2"},
        {"llm.w": "hX", "embed_tokens.w": "h2"},
    )
    assert line == (
        "[RED] salm.frozen_llm_unchanged: 2/2 params -- 1 changed 0 new/missing; "
        "changed up to 5 ['llm.w']"
    )
    assert blocking is True


def test_frozen_line_reds_on_renamed_tensor() -> None:
    """A name present in one model and not the other blocks too: a rename would hide movement from
    every digest comparison."""
    line, blocking = frozen_line({"llm.w": "h1"}, {"llm.w2": "h1"})
    assert line == (
        "[RED] salm.frozen_llm_unchanged: 0/2 params -- 0 changed 2 new/missing; "
        "new/missing up to 5 ['llm.w', 'llm.w2']"
    )
    assert blocking is True


def test_frozen_line_reds_with_zero_names() -> None:
    """Nothing checked proves nothing: an empty dict comparison must be RED, never vacuously PASS
    (the LoRA filter stripping every frozen weight would otherwise sail through)."""
    assert frozen_line({}, {}) == (
        "[RED] salm.frozen_llm_unchanged: 0/0 params -- 0 changed 0 new/missing",
        True,
    )


def test_names_line_pass_and_red_on_renamed_and_empty() -> None:
    """``salm.param_names_unchanged`` is the rename veto: identical names PASS as "identical keys",
    a symmetric difference REDs with the offending names, and zero names REDs (checked must be
    > 0)."""
    assert names_line({"a", "b"}, {"a", "b"}) == (
        "[PASS] salm.param_names_unchanged: 2/2 params -- identical keys",
        False,
    )
    assert names_line({"a", "b"}, {"a", "c"}) == (
        "[RED] salm.param_names_unchanged: 1/3 params -- new/missing up to 5 ['b', 'c']",
        True,
    )
    assert names_line(set(), set()) == (
        "[RED] salm.param_names_unchanged: 0/0 params -- new/missing up to 5 []",
        True,
    )


def test_lora_filter_selects_peft_tensors_by_name() -> None:
    """Regression: ``.lora_`` is the PEFT marker. The freeze verdict must never see a lora tensor
    (it is supposed to move) and the "lora" movement line must see only those -- a leak either way
    is a silent PASS or a spurious RED."""
    items = [
        ("llm.w", "base"),
        ("llm.w.lora_A", "peft"),
        ("llm.w.lora_B", "peft2"),
        ("embed_tokens.w", "base2"),
    ]
    assert lora_filter(items, "any") == items
    assert lora_filter(items, "only") == [("llm.w.lora_A", "peft"), ("llm.w.lora_B", "peft2")]
    assert lora_filter(items, "none") == [("llm.w", "base"), ("embed_tokens.w", "base2")]
    # A name merely containing "lora" without the ".lora_" marker is not a PEFT tensor.
    assert lora_filter([("llm.flora_gate", "base")], "none") == [("llm.flora_gate", "base")]


def test_chat_prompt_is_the_instruction_plus_audio_locator() -> None:
    """Regression: the prompt must stay "Transcribe the following: " + the model's audio locator tag
    -- the same instruction DATA_CFG feeds and every earlier WER was measured with. A drift here
    silently re-decodes the corpus as another task and the WER stops being a continuation."""
    assert chat_prompt({"audio": "a.wav", "answer": "text"}, _AUDIO_TAG) == [
        {
            "role": "user",
            "content": f"Transcribe the following: {_AUDIO_TAG}",
            "audio": ["a.wav"],
        }
    ]


def test_prompts_for_builds_one_prompt_per_row_in_order() -> None:
    """Generation order is row order, so the pairing below relies on this order staying put."""
    rows = [{"audio": "a.wav"}, {"audio": "b.wav"}]
    assert prompts_for(rows, _AUDIO_TAG) == [
        chat_prompt(rows[0], _AUDIO_TAG),
        chat_prompt(rows[1], _AUDIO_TAG),
    ]


def test_pair_rows_is_the_eval_all_shape_and_refuses_to_zip() -> None:
    """The ``all`` records are the eval's regression index (id + duration locate a bad clip):
    four keys in this order, every row present. A missing or extra hypothesis must raise rather
    than being silently zipped away."""
    rows = [
        {"id": "u1", "duration": 2.5, "answer": "ref one"},
        {"id": "u2", "answer": "ref two"},
    ]
    assert pair_rows(rows, ["hyp one", "hyp two"]) == [
        {"id": "u1", "duration": 2.5, "reference": "ref one", "hypothesis": "hyp one"},
        {"id": "u2", "duration": None, "reference": "ref two", "hypothesis": "hyp two"},
    ]
    with pytest.raises(ValueError, match="strict pairing required"):
        pair_rows(rows, ["hyp one"])


def test_modules_import_with_nemo_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: an ``import nemo`` at module scope would take the pure half down with it. These
    modules must import while ``nemo`` is absent -- that is the no-container CI path that calls
    the config builders and the verdict helpers above."""
    monkeypatch.setitem(sys.modules, "nemo", None)  # `import nemo` now raises ImportError
    for name in _MODULES:
        sys.modules.pop(name, None)
        module = importlib.import_module(name)
        assert callable(module.main)
        assert callable(module.parse_args)
