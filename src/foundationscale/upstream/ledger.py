"""Every copy, workaround and private-API use of an upstream framework, as typed data.

Rule 2 of the upstream design (docs/research/upstream_integration.md) is "wrap, don't copy": when a
copy or a workaround is unavoidable, it is recorded here with why it exists, where it lives, the
upstream version it was observed on, and the observable condition under which it can be deleted.

The record cannot go stale silently. Each entry names a repo-relative ``path`` and a literal
``anchor`` that must occur in that file; :func:`ledger_problems` checks both against the tree, and
a test runs it on every CI run. Moving, renaming or deleting the code without updating its entry is
therefore a failing check, not a forgotten note. The ``expiry_check`` is the sentence a person (or
the Level 2 tray run, once it exists) evaluates after an upstream upgrade: when it holds, the entry
and its code are deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = ["LEDGER", "LedgerEntry", "LedgerKind", "entries_for", "ledger_problems"]


class LedgerKind(str, Enum):
    COPY = "copy"  # upstream logic reproduced in our tree
    WORKAROUND = "workaround"  # code that exists only to step around an upstream defect
    PRIVATE_API = "private_api"  # a call into an undocumented / underscore upstream symbol


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    kind: LedgerKind
    framework: str
    path: str  # repo-relative file that holds the code
    anchor: str  # literal substring that must occur in ``path``
    why: str
    upstream_version: str  # where the behaviour was observed
    expiry_check: str  # the observable condition under which the entry can be deleted
    issue: str = ""  # upstream issue or URL, when one is known


# PHASE 1.2a (docs/research/upstream_integration.md): the Canary AED lane moved from
# ``validation_campaigns/speech_canary/nemo_finetune.py`` to the worker-side adapter at
# ``src/foundationscale/upstream/nemo/finetune.py``. The two Canary entries below point at
# the new path with the SAME anchor strings (they are preserved verbatim in the move).
_CANARY_AED_FINETUNE = "src/foundationscale/upstream/nemo/finetune.py"
# PHASE 1.2b (docs/research/upstream_integration.md): the NeMo speechlm2 SALM lane moved from
# ``validation_campaigns/speech_canary/salm_finetune.py`` to the worker-side adapter at
# ``src/foundationscale/upstream/nemo/salm_finetune.py``. The three SALM entries below point at
# the new path with the SAME anchor strings (they are preserved verbatim in the move).
_SALM_FINETUNE = "src/foundationscale/upstream/nemo/salm_finetune.py"
_NEMO_SEEN = "nemo 3.1 (container nemo-26.08)"
_HF_SEEN = "transformers 5.5"

LEDGER: tuple[LedgerEntry, ...] = (
    LedgerEntry(
        id="nemo-salm-strict-loading",
        kind=LedgerKind.WORKAROUND,
        framework="nemo",
        path=_SALM_FINETUNE,
        anchor="class _StrictSALMDataset(SALMDataset):",
        why=(
            "NeMo's strict audio loading sets AudioSamples(fault_tolerant=False), which returns 2 "
            "values while collate_conversation_audio_fault_tolerant unpacks 3, so strict loading "
            "crashes. The subclass keeps the 3-value loader with strict_audio_loading=True, so a "
            "dropped or reordered row raises instead of being skipped."
        ),
        upstream_version=_NEMO_SEEN,
        expiry_check=(
            "SALMDataset.with_fault_tolerant_audio_loading(False) followed by one collate of a "
            "2-row batch no longer raises ValueError"
        ),
    ),
    LedgerEntry(
        id="nemo-salm-cudnn-sdpa-off",
        kind=LedgerKind.WORKAROUND,
        framework="torch/cudnn (via nemo)",
        path=_SALM_FINETUNE,
        anchor="torch.backends.cuda.enable_cudnn_sdp(False)",
        why=(
            "cuDNN fused attention fails in backward on the GB200 stack (cuDNN Frontend reshape "
            "error) during SALM fine-tuning; flash/efficient SDPA compute the same attention."
        ),
        upstream_version=_NEMO_SEEN,
        expiry_check=(
            "one SALM training step with cuDNN SDPA enabled completes without the cuDNN "
            "Frontend error"
        ),
    ),
    LedgerEntry(
        id="nemo-datamodule-null-validation",
        kind=LedgerKind.WORKAROUND,
        framework="nemo",
        path=_SALM_FINETUNE,
        anchor="No validation_ds key at all",
        why=(
            "speechlm2 DataModule touches any present validation_ds key, so a null value "
            "crashes; the key is omitted."
        ),
        upstream_version=_NEMO_SEEN,
        expiry_check="DataModule(cfg with validation_ds: null) constructs",
    ),
    LedgerEntry(
        id="nemo-canary-train-cfg-strip",
        kind=LedgerKind.WORKAROUND,
        framework="nemo",
        path=_CANARY_AED_FINETUNE,
        anchor='"bucket_duration_bins",',
        why=(
            "The released Canary train_ds config carries tarred/bucketing keys for NVIDIA's own "
            "data; they are stripped so a plain manifest loads."
        ),
        upstream_version=_NEMO_SEEN,
        expiry_check=(
            "setup_training_data accepts the released train_ds with only manifest_filepath and "
            "batch_size overridden"
        ),
    ),
    LedgerEntry(
        id="nemo-canary-text-field",
        kind=LedgerKind.WORKAROUND,
        framework="nemo",
        path=_CANARY_AED_FINETUNE,
        anchor='train_cfg.text_field = "text"',
        why=(
            "The released train_ds names the transcript field 'answer' while our NeMo manifests "
            "write 'text'; left inherited, the targets would be silently empty."
        ),
        upstream_version=_NEMO_SEEN,
        expiry_check=(
            "never expires through an upstream change: it encodes our manifest contract; remove "
            "only if the NeMo manifest writer changes the field name"
        ),
    ),
    LedgerEntry(
        id="hf-qwen2-audio-token-formula",
        kind=LedgerKind.COPY,
        framework="transformers",
        path="src/foundationscale/train/audio.py",
        anchor=(
            '_MASK_LENGTH_TOKEN_FORMULAS: dict[str, Any] = {"Qwen2AudioProcessor": '
            "_qwen2_audio_tokens}"
        ),
        why=(
            "Qwen2AudioProcessor exposes no public per-row audio-token count, so its formula "
            "((L-1)//2+1, then (x-2)//2+1) is copied to verify placeholders per row."
        ),
        upstream_version=_HF_SEEN,
        expiry_check="Qwen2AudioProcessor exposes a public per-row audio token count",
    ),
    LedgerEntry(
        id="hf-qwen3-asr-token-formula",
        kind=LedgerKind.COPY,
        framework="transformers",
        path="src/foundationscale/train/audio.py",
        anchor='_MASK_LENGTH_TOKEN_FORMULAS["Qwen3ASRProcessor"] = _qwen3_asr_tokens',
        why=(
            "Qwen3ASRProcessor's per-clip count is the private _get_audio_token_length, so its "
            "chunked formula (13 tokens per 100-frame chunk plus three stride-2 convolutions "
            "over the remainder) is copied to verify placeholders per row."
        ),
        upstream_version="transformers 5.19.0",
        expiry_check="Qwen3ASRProcessor exposes a public per-row audio token count",
    ),
    LedgerEntry(
        id="hf-gemma-private-audio-token-count",
        kind=LedgerKind.PRIVATE_API,
        framework="transformers",
        path="src/foundationscale/train/audio.py",
        anchor='getattr(processor, "_compute_audio_num_tokens", None)',
        why=(
            "The per-row placeholder check calls the processor's private "
            "_compute_audio_num_tokens (Gemma-4); there is no public equivalent."
        ),
        upstream_version=_HF_SEEN,
        expiry_check="a public method for the per-row audio token count exists",
    ),
    LedgerEntry(
        id="hf-whisper-prefix-tokens",
        kind=LedgerKind.PRIVATE_API,
        framework="transformers",
        path="src/foundationscale/train/speech_kinds.py",
        anchor="tokenizer.prefix_tokens[1:]",
        why=(
            "The declared-language check reads WhisperTokenizer.prefix_tokens (undocumented) "
            "after set_prefix_tokens to verify the labels' language/task prefix."
        ),
        upstream_version=_HF_SEEN,
        expiry_check="a documented API returns the decoder prompt ids for a declared language/task",
    ),
)


def ledger_problems(repo_root: Path, entries: tuple[LedgerEntry, ...] = LEDGER) -> list[str]:
    """Every way the ledger disagrees with the tree; ``[]`` when it is consistent.

    Codes: ``duplicate_id:<id>``, ``empty_field:<id>:<field>``, ``path_missing:<id>``,
    ``anchor_missing:<id>``.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            problems.append(f"duplicate_id:{entry.id}")
        seen.add(entry.id)
        for field in (
            "id",
            "framework",
            "path",
            "anchor",
            "why",
            "upstream_version",
            "expiry_check",
        ):
            if not str(getattr(entry, field)).strip():
                problems.append(f"empty_field:{entry.id}:{field}")
        target = repo_root / entry.path
        if not target.is_file():
            problems.append(f"path_missing:{entry.id}")
        elif entry.anchor and entry.anchor not in target.read_text(encoding="utf-8"):
            problems.append(f"anchor_missing:{entry.id}")
    return problems


def entries_for(
    framework: str, entries: tuple[LedgerEntry, ...] = LEDGER
) -> tuple[LedgerEntry, ...]:
    """The entries recorded against one framework, in ledger order."""
    return tuple(e for e in entries if e.framework == framework)
