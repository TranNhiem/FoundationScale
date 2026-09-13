"""REFUSE-vs-RED adjudication tests for the three environment-fault arms in train().

``foundationscale.train.loop`` classifies every exception that escapes its three
guarded sites -- model/dataset construction, ``Trainer.train()``, and the final
``save_model`` -- through ``_environment_failure_reason`` before deciding the
verdict: a fault that is a property of the machine (a mapped OSError anywhere in
the ``__cause__``/``__context__`` chain) is REFUSE (96, unscored), and anything
else is RED (5, a genuine defect of the training plane).

Sites pinned (each gets a PAIR):

* construction guard (the ``except`` after the tokenization map, ~line 2157):
  REFUSE arm with a RuntimeError wrapping ``OSError(EMFILE)``; RED control with
  a plain RuntimeError.
* Trainer.train() guard (~line 2542): REFUSE arm with a wrapped EDQUOT; RED
  control with ``OSError(EINVAL)`` -- an OSError is a SUPERCLASS of every
  environment fault, so this control is the one that proves REFUSE is a
  selection over the mapped-errno set, not a default for the type.
* final-save guard (~line 2686): REFUSE arm with a wrapped EROFS; RED control
  with a plain ValueError.

The pair is the point: a lone "OSError -> 96" test also passes against an
implementation that returns 96 unconditionally. Every REFUSE test below fails
against such an implementation only when its control is also green, so the two
must be read together.

The wrapped form is deliberate, not stylistic: the classifier's whole reason
for walking the chain is that libraries wrap an OSError rather than let it
through. Note that the wrapped errnos here are drawn from the set lines
100-108 actually MAP -- a house example elsewhere names ENOLCK, but the shipped
mapping does not contain it, and a fixture raised with an unmapped errno would
(correctly) adjudicate RED.

No skips anywhere: this module runs under FS_FORBID_SKIPS=1 on CPU-only boxes;
the autouse fixture pins CUDA/MPS probes to doubles that are not narrower than
the originals (see ``_false_probe_like``).
"""

from __future__ import annotations

import errno
import functools
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Trainer

from foundationscale.train import loop
from foundationscale.train.loop import (
    EXIT_RED,
    EXIT_REFUSE,
    TrainConfig,
    train,
)

# Same shape of profile fixture as tests/test_train_entry.py: a single-node
# reference profile the declared 1x1 topology satisfies exactly, so the
# prologue emits no blocking findings and execution reaches the guarded sites.
PROFILE_DATA: dict = {
    "name": "reference-single-node",
    "scheduler": "slurm",
    "partitions": ["batch"],
    "node_pattern": r"compute-0[1-8]",
    "gpus_per_node": 1,
    "nccl_socket_ifname": "eth0",
    "ib_hca_pattern": "mlx5_*",
    "mnnvl_available": False,
    "container_runtime": "none",
    "container_image": "",
    "filesystem_roots": ["/tmp"],
    "max_nodes": 8,
}

# vocab_size=256 below, so the tokenizer must emit ids strictly below 256 or a
# fixture defect would masquerade as a training defect. 3 + 253 == 256 exactly.
_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(253))


# Whatever the autouse fixture displaced, recorded at patch time: a test body
# cannot read the original itself because the fixture has already run by then.
_ORIGINAL_CUDA_PROBES: dict[str, object] = {}
_ORIGINAL_MPS_PROBES: dict[str, Any] = {}


def _false_probe_like(original: Any) -> Any:
    """A zero-arg ``False`` stub that is not NARROWER than the probe it replaces.

    torch introspects these probes -- ``torch/_dynamo/variables/torch.py`` reads
    ``__wrapped__`` while building its constant-fold table -- so a bare lambda
    is narrower than the cached original and the failure then surfaces, or does
    not, according to when the lazy dynamo import happens to land. Wrapping the
    stub in ``lru_cache`` when the original is cached keeps every attribute
    reachable through ``__wrapped__`` answering False.
    """
    stub: Any = lambda: False  # noqa: E731 -- must be a plain function to match shape
    if hasattr(original, "cache_info"):
        stub = functools.lru_cache(maxsize=None)(stub)
    return stub


@pytest.fixture(scope="module")
def _offline_module() -> None:
    """Offline flags for the module-scoped model build, which predates any
    function-scoped fixture's monkeypatching.

    ``from_config`` touches no network, but pinning the flags here too means
    the fact is guarded twice -- an env fixture is cheap, an accidental hub
    call in CI is a hang measured in minutes.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setenv("TRANSFORMERS_OFFLINE", "1")
    mp.setenv("HF_DATASETS_OFFLINE", "1")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _offline_and_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every hardware/network reach to a double; the real arms are wired by raise.

    Nothing here may touch torch.cuda for real: this suite's contract is to
    pass on a CPU-only box. The environment-fault arms do not fake hardware --
    they raise a wrapped OSError at the call site under test, which is the
    shape a wrapped fault takes in the libraries this plane calls.
    """
    _ORIGINAL_CUDA_PROBES["is_available"] = torch.cuda.is_available
    monkeypatch.setattr(torch.cuda, "is_available", _false_probe_like(torch.cuda.is_available))
    for name in ("is_available", "is_built"):
        original = getattr(torch.backends.mps, name)
        _ORIGINAL_MPS_PROBES[name] = original
        monkeypatch.setattr(torch.backends.mps, name, _false_probe_like(original))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory, _offline_module: None) -> Path:
    """A real, loadable causal LM + tokenizer on disk, built once per module.

    Constructed from a config OBJECT, never a hub id: AutoModelForCausalLM.
    from_config allocates random weights for a 2-layer Llama whose embeddings
    are 256 x 32, small enough that one training step on CPU takes well under
    a second. The tokenizer is a WordLevel model over the same 256-token vocab,
    wrapped in PreTrainedTokenizerFast so ``save_pretrained`` produces a plain
    tokenizer.json that AutoTokenizer.from_pretrained reads offline. Both live
    in the SAME directory because that is the layout train() assumes when it
    resolves cfg.model.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

    model_dir = tmp_path_factory.mktemp("tiny-model")

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(str(model_dir))

    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _WORD_TOKENS)}
    wordlevel = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    wordlevel.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=wordlevel,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    fast.save_pretrained(str(model_dir))

    return model_dir


@pytest.fixture()
def profile_path(tmp_path: Path) -> Path:
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(PROFILE_DATA), encoding="utf-8")
    return path


@pytest.fixture()
def text_dataset(tmp_path: Path) -> Path:
    """16 rows with a "text" column of whitespace-joined in-vocab words.

    WordLevel with a Whitespace pre-tokenizer splits on spaces, so the text
    must be written pre-segmented -- feeding natural sentences would map every
    word to <unk> and the fixture would silently stop measuring the model it
    claims to train.
    """
    path = tmp_path / "train.jsonl"
    rows = [
        {"text": f"w{(i * 7) % 253} w{(i * 7 + 1) % 253} w{(i * 7 + 2) % 253} w{(i * 7 + 3) % 253}"}
        for i in range(16)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def cfg(
    tmp_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
    profile_path: Path,
) -> TrainConfig:
    """Field set mirrors the config fixture of tests/test_train_execution.py.

    Only fields TrainConfig actually declares: the cluster profile arrives as
    ``profile_path`` (a Path to the JSON file the ``profile_path`` fixture
    writes, not as a dict), the batch-size field is ``per_device_batch_size``,
    the save cadence is ``save_interval``, and the required topology fields
    ``nodes``/``gpus_per_node`` are given explicitly. ``objective`` is declared
    because the final-save arm lets the REAL Trainer.train() run, and the
    objective gate refuses an undeclared run at its first observed step --
    omitting it would make that arm red for a reason that has nothing to do
    with what the arm is pinning.
    """
    return TrainConfig(
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        output_dir=tmp_path / "output",
        nodes=1,
        gpus_per_node=1,
        profile_path=profile_path,
        objective="sft",
        precision="fp32",
        attn_implementation="eager",
        max_steps=2,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_interval=50,
        dry_run=False,
    )


def _wrapped_environment_fault(error_number: int) -> BaseException:
    """An OSError a top-frame-only check would miss: it sits one frame down.

    This is the shape the classifier's docstring says is the common case -- the
    library's own exception type on top, the OSError as ``__cause__``. A test
    that raised the bare OSError would pass against a classifier that never
    walks the chain, so the wrapping is part of what is being measured.
    """
    inner = OSError(error_number, os.strerror(error_number))
    outer = RuntimeError(f"the library call failed around errno {error_number}")
    outer.__cause__ = inner
    return outer


def _raiser(exc: BaseException) -> Callable[..., Any]:
    """A stand-in call that raises ``exc`` regardless of how it is invoked."""

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return _raise


def test_chain_walk_classifies_a_wrapped_mapped_errno_as_environment() -> None:
    """Fixture sanity: the wrapped fault is not an OSError at the top frame."""
    fault = _wrapped_environment_fault(errno.EMFILE)
    assert not isinstance(fault, OSError), (
        f"wanted the fixture to hide the OSError under a wrapper, but the top "
        f"frame itself is an OSError: {fault!r}"
    )
    reason = loop._environment_failure_reason(fault)
    assert reason is not None, (
        f"wanted the chain walk to reach the mapped errno {errno.EMFILE} under "
        f"the RuntimeError wrapper and name it as an environment fault; got "
        f"None, which would adjudicate this fixture RED"
    )


def test_chain_walk_leaves_a_genuine_defect_unclassified() -> None:
    """Fixture sanity: a plain RuntimeError must come back None (a genuine RED)."""
    fault = RuntimeError("loss went NaN on the first step")
    reason = loop._environment_failure_reason(fault)
    assert reason is None, (
        f"wanted None for a genuine defect so the run is scored RED; got "
        f"{reason!r}, which would adjudicate it REFUSE and take the run off "
        f"the board"
    )


def test_environment_fault_in_construction_refuses(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construction guard, REFUSE arm: a wrapped EMFILE unscores the run (96)."""
    monkeypatch.setattr(
        loop,
        "_load_raw_dataset",
        _raiser(_wrapped_environment_fault(errno.EMFILE)),
    )
    verdict = train(cfg)
    assert verdict == EXIT_REFUSE, (
        f"wanted train() to adjudicate a wrapped per-process open-file "
        f"exhaustion (errno {errno.EMFILE}) during model/dataset construction "
        f"as REFUSE (exit {EXIT_REFUSE}): construction never began, so there "
        f"is no run to score; got exit {verdict}"
    )


def test_genuine_fault_in_construction_is_red(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construction guard, RED control: same site, no environment fault -> 5."""
    monkeypatch.setattr(
        loop,
        "_load_raw_dataset",
        _raiser(RuntimeError("the dataset spec is not a mapping")),
    )
    verdict = train(cfg)
    assert verdict == EXIT_RED, (
        f"wanted train() to adjudicate a plain RuntimeError escaping model/"
        f"dataset construction as RED (exit {EXIT_RED}): no mapped OSError "
        f"anywhere in the chain means this is a defect of the training plane; "
        f"got exit {verdict}"
    )


def test_environment_fault_during_training_refuses(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trainer.train() guard, REFUSE arm: a wrapped EDQUOT unscores the run (96)."""
    monkeypatch.setattr(
        Trainer,
        "train",
        _raiser(_wrapped_environment_fault(errno.EDQUOT)),
    )
    verdict = train(cfg)
    assert verdict == EXIT_REFUSE, (
        f"wanted train() to adjudicate a wrapped disk-quota exhaustion (errno "
        f"{errno.EDQUOT}) escaping Trainer.train() as REFUSE (exit "
        f"{EXIT_REFUSE}): the fault is a property of the machine, so the run "
        f"is unscored rather than failed; got exit {verdict}"
    )


def test_unmapped_os_error_during_training_is_red(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trainer.train() guard, RED control: an OSError is not automatically REFUSE.

    EINVAL is an OSError the classifier does NOT name. OSError is a superclass
    of every environment fault, so this arm is the load-bearing one: an
    implementation that REFUSEd every OSError, or every exception, would pass
    the REFUSE test above and must fail here.
    """
    monkeypatch.setattr(
        Trainer,
        "train",
        _raiser(OSError(errno.EINVAL, os.strerror(errno.EINVAL))),
    )
    verdict = train(cfg)
    assert verdict == EXIT_RED, (
        f"wanted train() to adjudicate an unmapped OSError (errno "
        f"{errno.EINVAL}, not in the environment set) escaping Trainer.train() "
        f"as RED (exit {EXIT_RED}): REFUSE is a selection over the mapped "
        f"errno set, not a default for the OSError type; got exit {verdict}"
    )


def test_environment_fault_at_final_save_refuses(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final-save guard, REFUSE arm: a wrapped EROFS unscores the run (96)."""
    monkeypatch.setattr(
        Trainer,
        "save_model",
        _raiser(_wrapped_environment_fault(errno.EROFS)),
    )
    verdict = train(cfg)
    assert verdict == EXIT_REFUSE, (
        f"wanted train() to adjudicate a wrapped read-only-filesystem fault "
        f"(errno {errno.EROFS}) at the final save as REFUSE (exit "
        f"{EXIT_REFUSE}): the weights trained and the filesystem refused "
        f"them, so the run is unscored rather than failed; got exit {verdict}"
    )


def test_genuine_fault_at_final_save_is_red(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final-save guard, RED control: same site, no environment fault -> 5."""
    monkeypatch.setattr(
        Trainer,
        "save_model",
        _raiser(ValueError("the state dict contains a meta tensor")),
    )
    verdict = train(cfg)
    assert verdict == EXIT_RED, (
        f"wanted train() to adjudicate a plain ValueError at the final save as "
        f"RED (exit {EXIT_RED}): a serialization defect is the training "
        f"plane's own failure, however late in the run it lands; got exit "
        f"{verdict}"
    )
