"""Fix: LoRA adapter saved EMPTY under FSDP version 1 (peft + FSDP1 only).

MEASURED (gemma-4-12B-it, 20 steps, LoRA r16, FSDP1 per-layer auto-wrap):
``adapter_model.safetensors`` landed at 40 bytes -- 0 of the 656 LoRA tensors
the run actually trained -- and FoundationScale's own save gate correctly
adjudicated that checkpoint RED. Root cause, verified with a standalone
repro at ``scratch/fsdp_peft_save/repro.py`` on the same hardware:

  1. ``Trainer.save_model`` (transformers 5.18.0) gathers a CORRECT, full,
     un-sharded state dict via the collective ``accelerator.get_state_dict
     (self.model)`` -- 656 LoRA tensors present and non-zero -- then calls
     ``self._save(output_dir, state_dict=state_dict)`` on the writing rank.
  2. ``Trainer._save`` passes that correct dict straight to
     ``self.model.save_pretrained(output_dir, state_dict=state_dict)``, and
     ``self.model`` at that point is the LIVE, FSDP-wrapped ``PeftModel``.
  3. ``PeftModel.save_pretrained`` calls ``peft.get_peft_model_state_dict``,
     which derives the adapter key PREFIXES it is looking for by walking
     ``self.named_modules()`` of whatever object it was given
     (``peft/tuners/tuners_utils.py``, ``_get_tuner_state_dict_key_prefixes``)
     and strips only a LEADING ``"_fsdp_wrapped_module."`` segment. FSDP
     version 1's per-layer auto-wrap -- required here to avoid OOM on a tied
     48 GiB embedding (see loop.py's own comment beside
     ``fsdp_config["version"] = 1``) -- plus peft's own
     ``fsdp_auto_wrap_policy`` (which separately wraps each LoRA leaf) means
     the live tree's module names carry ``_fsdp_wrapped_module.`` segments in
     the MIDDLE of the path, not only at the front. The derived prefixes
     therefore never match the (already correctly unsharded) keys in
     ``state_dict``, and 0 are selected.
  4. ``accelerator.unwrap_model()`` does not fix this either: it only strips
     the OUTERMOST wrapper, not the nested per-layer/per-leaf wrapping already
     baked into the live module tree.

FSDP version 2 (the untied Qwen3.6 arm) is UNAFFECTED: its composable
``fully_shard`` does not rename submodules the way FSDP1's class-wrapping
does, so the unmodified ``save_model`` path already writes a correct
adapter. This module changes nothing for FSDP2, DDP, or a non-peft model --
see :func:`is_fsdp1_peft_save`.

THE FIX (verified working in ``scratch/fsdp_peft_save/repro.py``'s
``FIX_SAVE`` path): never ask the live, wrapped tree for its own module
names. Build a FRESH ``PeftModel`` skeleton on the meta device (same base
model class and config, same ``LoraConfig``(s)) that was NEVER touched by
FSDP, and hand it the ALREADY-CORRECT ``state_dict`` collective-gathered by
``save_model``. A meta-device skeleton costs no real memory or compute, and
its ``named_modules()`` is clean because it was never wrapped at all.

WHY THIS HOOK: ``Trainer._save`` is the one method both intermediate
``save_steps`` checkpoints (via ``_save_checkpoint`` -> ``save_model`` ->
``_save``) and the final save (``trainer.save_model(final_dir)`` -> the same
``_save``) funnel through, so overriding it (rather than duplicating the
final-save call site, or monkeypatching peft globally) covers every
checkpoint write with one change and leaves every other code path --
FSDP2, DDP, no adapter, the ``fsdp_state_dict="sharded"`` DCP path (which
never calls ``_save`` at all; see loop.py's ``_save_final_fsdp_sharded``) --
byte-identical, because :meth:`FsdpPeftSaveTrainer._save` always calls
``super()._save(...)`` first and only OVERWRITES the adapter file afterward,
and only when :func:`is_fsdp1_peft_save` says this save needs it.

The dispatch condition (:func:`is_fsdp1_peft_save`) is a pure function of
three booleans so it is unit-testable with no torch/transformers/peft
installed; the skeleton construction and the Trainer subclass need those
libraries and import them function-locally, matching this package's
torch-free-import contract (see train/loop.py's module docstring).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def is_fsdp1_peft_save(
    *,
    is_fsdp_enabled: bool,
    fsdp_version: int | None,
    has_peft_skeleton_factory: bool,
) -> bool:
    """Does THIS save need the FSDP1+peft empty-adapter fix?

    True only for the exact combination the root cause requires: FSDP is the
    active distribution strategy, it is specifically version 1 (version 2's
    composable ``fully_shard`` does not corrupt ``named_modules()`` the way
    version 1's class-wrapping does), and the model being saved is a peft
    model (``has_peft_skeleton_factory`` is true exactly when the caller built
    a skeleton factory for it -- see :func:`build_peft_fsdp1_skeleton_factory`
    and its call site in loop.py, which only builds one when ``cfg.adapter``
    is declared). Every other combination -- FSDP2, DDP, no adapter -- must
    return False so the caller's save path stays byte-identical.
    """
    return bool(is_fsdp_enabled and has_peft_skeleton_factory and fsdp_version == 1)


def build_peft_fsdp1_skeleton_factory(peft_model: Any) -> Callable[[], Any]:
    """Capture what a save-time skeleton needs, BEFORE FSDP wraps anything.

    Call this in loop.py right after ``get_peft_model(...)`` succeeds, while
    ``model`` is still a plain (not yet FSDP-wrapped) ``PeftModel`` -- FSDP
    wrapping only happens later, inside ``accelerator.prepare`` as part of
    ``trainer.train()``. Reading the base model class, its config, and the
    adapter config(s) HERE, from the clean pre-wrap object, means the save-time
    fix never has to reflect through the live (and, for FSDP1, structurally
    renamed) module tree at all -- it builds a wholly independent object
    instead, exactly like the verified ``FIX_SAVE`` path in
    ``scratch/fsdp_peft_save/repro.py``.

    Returns a zero-argument factory rather than the skeleton itself: building
    the meta-device model now would mean constructing and carrying a second
    module tree for the run's entire duration for no reason, since the only
    rank that ever needs it is the one writing a checkpoint, at the moment it
    writes one. Meta-device construction is ~free (no real storage is
    allocated), so paying that cost at every save is the cheaper trade.

    Raises ValueError if ``peft_model`` carries no adapter config at all --
    that would mean this was called on something other than a just-wrapped
    PeftModel, a caller defect this function refuses to paper over.
    """
    base_model = peft_model.get_base_model()
    base_cls = type(base_model)
    base_config = base_model.config
    peft_configs: Mapping[str, Any] = dict(peft_model.peft_config)
    if not peft_configs:
        raise ValueError(
            "build_peft_fsdp1_skeleton_factory requires a peft-wrapped model "
            "with at least one adapter config; peft_model.peft_config is empty"
        )
    adapter_names = list(peft_configs)

    def factory() -> Any:
        import torch  # noqa: PLC0415
        from peft import get_peft_model  # noqa: PLC0415

        # Direct construction, not Auto*.from_config(): PreTrainedModel (and
        # this concrete class) has no `from_config` classmethod at all --
        # only the Auto* dispatcher classes do (measured: AttributeError on
        # the installed transformers when this called
        # Gemma4UnifiedForConditionalGeneration.from_config). dtype is
        # irrelevant here and deliberately not set: this skeleton never has
        # its own parameters read -- save_pretrained(state_dict=...) below
        # pulls every real tensor from the COLLECTIVE-gathered state_dict by
        # key name, and the skeleton exists only so get_peft_model_state_dict
        # can derive clean (never-FSDP-wrapped) key prefixes from its
        # named_modules().
        with torch.device("meta"):
            skeleton = base_cls(base_config)
        first_name = adapter_names[0]
        skeleton = get_peft_model(skeleton, peft_configs[first_name], adapter_name=first_name)
        for extra_name in adapter_names[1:]:
            skeleton.add_adapter(extra_name, peft_configs[extra_name])
        return skeleton

    return factory


def _fsdp_version(trainer: Any) -> int | None:
    """``accelerator.state.fsdp_plugin.fsdp_version``, or None if unreadable.

    Reads what accelerate actually BUILT (the same "measured, not declared"
    rule loop.py applies elsewhere, e.g. its ``fsdp_plugin_args`` cpu_offload
    check), not what loop.py's own ``fsdp_config`` dict asked for. Every
    ``getattr`` falls back to None rather than raising, so a bare test double
    or a non-FSDP run reads as "no FSDP version" instead of crashing.
    """
    state = getattr(getattr(trainer, "accelerator", None), "state", None)
    plugin = getattr(state, "fsdp_plugin", None)
    return getattr(plugin, "fsdp_version", None)


def _atomically_overwrite_adapter(skeleton: Any, resolved_dir: str, state_dict: Any) -> None:
    """``skeleton.save_pretrained(resolved_dir, ...)``, but atomically.

    ``PeftModel.save_pretrained`` writes directly into ``resolved_dir``,
    which already holds ``super()._save``'s broken (0-tensor) adapter from
    one line up -- a crash or a concurrent reader (the save gate, running
    the moment this write finishes) between the first byte and the last
    would see a TRUNCATED ``adapter_model.safetensors``, not the broken-but-
    complete one it replaced. Writing into a fresh temp directory on the
    SAME filesystem (``tempfile.mkdtemp(dir=resolved_dir)``, so the later
    ``os.replace`` is a same-device rename, the only case POSIX/NTFS make
    atomic) and then replacing each file one at a time means every reader
    sees either the old file or the complete new one, never a partial
    write. Writer-rank-only: this function is only ever reached from
    ``_save`` on the rank ``_save`` itself only runs on (see its own
    docstring) -- no cross-rank coordination is needed or attempted here.
    """
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    tmp_dir = Path(tempfile.mkdtemp(dir=resolved_dir, prefix=".fs_fsdp1_peft_save_tmp_"))
    try:
        skeleton.save_pretrained(str(tmp_dir), state_dict=state_dict)
        for entry in tmp_dir.iterdir():
            entry.replace(Path(resolved_dir) / entry.name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def fsdp1_peft_save_trainer_class(base_trainer_cls: type) -> type:
    """Build a ``base_trainer_cls`` subclass that fixes the FSDP1+peft save.

    ``base_trainer_cls`` is passed in (rather than imported here) so this
    module never imports ``transformers`` itself, matching the train package's
    torch-free-import contract; loop.py already has ``Trainer`` in scope by
    the time it calls this.

    The override always calls ``super()._save(output_dir, state_dict)``
    FIRST -- exactly the stock call, with the stock (possibly version-
    specific) tail behaviour for saving the processing class and training
    args -- and only OVERWRITES ``adapter_model.safetensors`` /
    ``adapter_config.json`` afterward, and only when
    :func:`is_fsdp1_peft_save` says this save needs it. That means:

      * every non-FSDP1-peft save (FSDP2, DDP, no adapter) is byte-identical
        to calling ``base_trainer_cls._save`` directly -- this method adds
        nothing on those paths beyond the one dispatch check;
      * this plane never has to replicate transformers' own
        version-sensitive tail logic (tokenizer/processor save, training
        args pickle), which a from-scratch rewrite would have to track
        across the ``transformers>=4.40`` range this package declares.

    ``_fs_peft_skeleton_factory`` is a plain instance attribute (default None
    on the class), set by loop.py AFTER construction
    (``trainer._fs_peft_skeleton_factory = factory_or_none``) rather than
    threaded through ``__init__`` -- the smaller surface of the two, since it
    leaves ``__init__`` itself untouched.
    """

    class FsdpPeftSaveTrainer(base_trainer_cls):
        _fs_peft_skeleton_factory: Callable[[], Any] | None = None

        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            factory = self._fs_peft_skeleton_factory
            needs_fix = False
            if factory is not None:
                # Attribute access is guarded behind "a factory exists" so a
                # bare test double representing a non-peft/non-FSDP Trainer
                # (which may answer ANY attribute lookup, e.g. via a catch-all
                # __getattr__) is never consulted at all on the overwhelmingly
                # common path where there is nothing to fix.
                needs_fix = is_fsdp1_peft_save(
                    is_fsdp_enabled=bool(getattr(self, "is_fsdp_enabled", False)),
                    fsdp_version=_fsdp_version(self),
                    has_peft_skeleton_factory=True,
                )
            super()._save(output_dir, state_dict)
            if not needs_fix:
                return
            if state_dict is None:
                # save_model()'s FSDP branch is what is expected to have
                # already gathered state_dict via the COLLECTIVE
                # accelerator.get_state_dict(self.model) before calling
                # _save -- see this module's docstring, step 1. Re-deriving
                # it HERE would be a single-rank collective call (this method
                # only runs on the writing rank) and would hang every other
                # rank waiting on it. Refusing loudly is the doctrine-correct
                # answer: never guess a replacement for a missing collective.
                raise RuntimeError(
                    "FsdpPeftSaveTrainer._save: FSDP1+peft fix engaged but "
                    "state_dict is None. The caller was expected to pass the "
                    "already-gathered FULL_STATE_DICT (see module docstring); "
                    "re-gathering it from inside a writing-rank-only method "
                    "would be a single-rank collective and hang the other "
                    "ranks, so this refuses instead of guessing"
                )
            resolved_dir = output_dir if output_dir is not None else self.args.output_dir
            # needs_fix is only ever True inside the `if factory is not None`
            # branch above, but that narrowing does not persist across the
            # intervening `super()._save(...)` call for the type checker --
            # this assert is the same fact, stated where mypy can use it.
            assert factory is not None
            skeleton = factory()
            _atomically_overwrite_adapter(skeleton, resolved_dir, state_dict)

    return FsdpPeftSaveTrainer
