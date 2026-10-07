"""Checkpoint saves for the Megatron RL online lane: HF weights, run manifest, save gates.

The online lane already keeps a Hugging Face copy of the policy on the writer rank
and refits it from the Megatron shards before every rollout
(:func:`foundationscale.rl.megatron.online.refit_hf_policy`). A save therefore
needs no second export path: right after a refit the HF copy IS the trained
policy, and ``save_pretrained`` writes it as a servable safetensors directory.

What a bare ``save_pretrained`` cannot give is a checkpoint the save gates can
adjudicate. ``checkpoint.save_complete`` compares the tensors on disk against a
DECLARED set, and with no run manifest beside the checkpoint it has no
denominator and goes VACUOUS -- which is exactly where the Megatron lane sat. The
declared set here is the names the refit received from ``export_hf_weights``:
the Megatron side's statement of what the policy is, measured independently of
the file that ``save_pretrained`` writes, so a save that dropped a tensor shows
up as a missing name instead of as "what is there matches what is there".

The module imports without torch; the gate and manifest imports are lazy.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_NAME",
    "run_policy_save_gates",
    "save_policy_checkpoint",
]

# The reserved name ``foundationscale.checkpoint.load_manifest`` searches first.
MANIFEST_NAME = "run_manifest.json"

_TOPOLOGY_KEYS = (
    "nodes",
    "gpus_per_node",
    "tensor_parallel",
    "pipeline_parallel",
    "data_parallel",
    "expert_parallel",
    "context_parallel",
)


def save_policy_checkpoint(
    hf_model: Any,
    tokenizer: Any,
    out_dir: str | os.PathLike[str],
    *,
    declared_names: Iterable[str],
    run_id: str,
    topology: Mapping[str, int],
    config: Mapping[str, object],
    defaults: Mapping[str, object] | None = None,
) -> Path:
    """Write the HF policy, its tokenizer and a run manifest under ``out_dir``.

    Writer-rank only: the caller has already run the (collective) refit, so this
    function touches no process group. ``declared_names`` is the denominator the
    completeness gate adjudicates; an empty one is refused before anything is
    written, because "all 0 declared tensors present" is the ``all([])`` trap and
    would turn the gate's VACUOUS into a clean-looking save.

    ``topology`` maps the seven provenance topology fields (``nodes``,
    ``gpus_per_node``, ``tensor_parallel``, ``pipeline_parallel``,
    ``data_parallel``, ``expert_parallel``, ``context_parallel``) explicitly --
    the two ``Topology`` classes in this repo share field names for different
    quantities, so nothing is splatted from one into the other.

    ``config`` is recorded key by key; a value equal to its entry in ``defaults``
    is attributed ``source="default"``, anything else ``"cli"``, so the manifest
    does not present an untouched default as an operator's choice. ``None``
    values are recorded too (as ``"None"``): an unset knob is a fact about the run.

    A directory that already holds a run manifest is refused: overwriting it
    would replace one run's provenance with another's under the same path.
    """
    import dataclasses

    from foundationscale.provenance.manifest import (
        EffectiveValue,
        RunManifest,
        Topology,
        capture_code_provenance,
        capture_environment,
    )
    from foundationscale.train.loop import _declare_checkpoint

    names = sorted(set(declared_names))
    if not names:
        raise ValueError(
            "save_policy_checkpoint: declared_names is empty; with no declared tensor "
            "set the completeness gate has nothing to compare the checkpoint against"
        )
    missing_keys = [key for key in _TOPOLOGY_KEYS if key not in topology]
    if missing_keys:
        raise ValueError(f"save_policy_checkpoint: topology lacks {missing_keys}")
    # The expert/dense basis (num_experts, and the positive dense declaration) comes
    # from the SFT plane's canonical declarer over the live HF copy; only the tensor
    # set is swapped for the export's names, the source independent of this file.
    expert_basis, _notes = _declare_checkpoint(hf_model)
    declared = dataclasses.replace(expert_basis, declared_fqns=tuple(names))
    out = Path(out_dir)
    if (out / MANIFEST_NAME).exists():
        raise FileExistsError(
            f"save_policy_checkpoint: {out / MANIFEST_NAME} already exists; refusing to "
            "overwrite another save's weights and provenance"
        )
    out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out, safe_serialization=True)
    if callable(getattr(tokenizer, "save_pretrained", None)):
        tokenizer.save_pretrained(out)

    manifest = RunManifest(
        run_id=run_id,
        attempt=1,
        code=capture_code_provenance(Path.cwd(), entrypoint=sys.argv[0]),
        config={
            str(key): EffectiveValue(
                key=str(key),
                value=str(value),
                source="default"
                if defaults is not None and key in defaults and defaults[key] == value
                else "cli",
            )
            for key, value in config.items()
        },
        environment=capture_environment(),
        topology=Topology(**{key: int(topology[key]) for key in _TOPOLOGY_KEYS}),
        artifact_paths={"checkpoint": str(out)},
        declared=declared,
    )
    target = out / MANIFEST_NAME
    tmp = target.with_name(f".{MANIFEST_NAME}.tmp")
    tmp.write_text(manifest.to_json() + "\n", encoding="utf-8")
    tmp.replace(target)
    return out


def run_policy_save_gates(out_dir: str | os.PathLike[str], *, first: bool) -> Any:
    """Run every registered save gate over the checkpoint at ``out_dir``.

    ``first`` selects the ``FIRST_SAVE`` event (which adds the one-shot
    first-save checks) over ``SAVE``. Typed dispatch with
    ``missing_ctx="report-skip"`` mirrors the SFT loop's ``_run_save_gates``: a
    gate from another context family is reported unwired, not handed a context
    it cannot read. Returns the :class:`~foundationscale.gates.core.GateReport`;
    its ``ok`` is the verdict.
    """
    from foundationscale.gates.checkpoint_gates import CheckpointGateContext
    from foundationscale.gates.core import REGISTRY, Lifecycle, run_event

    event = Lifecycle.FIRST_SAVE if first else Lifecycle.SAVE
    ctx = CheckpointGateContext.from_path(Path(out_dir))
    return run_event(REGISTRY, event, ctx, missing_ctx="report-skip")
