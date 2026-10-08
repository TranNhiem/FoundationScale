"""``--fused-loss liger``: fused linear cross-entropy, required to reach 32K context.

MEASURED: gemma-4-12B-it, LoRA, text-only, at ``--max-sequence-length 32768``
OOMs allocating exactly 32 GiB = 32768 (sequence) x 262144 (vocab) x 4 bytes
(fp32) -- the materialized logits tensor for ONE micro-batch, before the loss
is even computed. At 16384 the measured peak was 90 GB/GPU. Fused linear
cross-entropy (liger_kernel) never materializes that tensor: it fuses the
final linear projection and the cross-entropy reduction into one kernel that
streams over vocabulary chunks, so the win is specifically the (seq x vocab)
intermediate, not a general speedup.

This module patches the MODEL INSTANCE -- never TrainingArguments, and never
a class-wide monkeypatch -- so the change is scoped to one run's model
object, applied once, before peft and FSDP wrap it (both wrappers need to see
the real forward they will call; patching after either would either be
invisible to the wrapper or require unwrapping first).

Coverage (liger_kernel 0.8.4, installed): ``_apply_liger_kernel_to_instance``
dispatches on ``model.config.model_type`` through
``MODEL_TYPE_TO_APPLY_LIGER_FN``, which this plane's installed version maps
for ``"gemma4"``, ``"qwen3_5"`` and ``"qwen3_5_moe"`` -- covering
gemma-4-26B-A4B-it, gemma-4-31B-it, Qwen3.6-27B and Qwen3.6-35B-A3B. It does
NOT cover ``"gemma4_unified"`` (gemma-4-12B-it, class
``Gemma4UnifiedForConditionalGeneration``): liger's own "gemma4" patch targets
``transformers.models.gemma4.modeling_gemma4``, a different module path from
``transformers.models.gemma4_unified.modeling_gemma4_unified``, which is what
gemma-4-12B-it actually loads (MEASURED: its ``config.json`` carries
``"model_type": "gemma4_unified"``, architecture
``Gemma4UnifiedForConditionalGeneration``). For that one family this module
owns an FS-written forward patch (:func:`_fs_gemma4_unified_lce_forward`),
adapted from ``liger_kernel.transformers.model.gemma4.multimodal_forward``
(which patches the sibling, non-unified class) to the actual stock forward at
``transformers/models/gemma4_unified/modeling_gemma4_unified.py``
(``Gemma4UnifiedForConditionalGeneration.forward``, read at patch-design time
against the installed transformers 5.18.0), preserving
``final_logit_softcapping`` exactly as that stock forward applies it.

Any other ``model_type`` is REFUSED (96) at the call site in loop.py, naming
the model_type and the supported list -- never silently trained unfused under
a declared label.

NUMERICS: the generic liger path explicitly turns every OTHER kernel swap
back OFF (``rope=False, cross_entropy=False, rms_norm=False, geglu=False,
swiglu=False, layer_norm=False``) even though liger's own defaults enable
rms_norm and ge/swiglu by default -- this flag's contract is "only the loss
changes", so the non-loss kernels stay stock. The custom gemma4_unified patch
only replaces the final lm_head-plus-loss computation and otherwise calls the
exact same ``self.model(...)`` the stock forward calls, for the same reason.
See tests/train/test_fused_loss_parity.py for the on-GPU numeric check.
"""

from __future__ import annotations

from typing import Any

#: model_type values the installed liger_kernel (0.8.4) patches generically,
#: through ``_apply_liger_kernel_to_instance`` / ``MODEL_TYPE_TO_APPLY_LIGER_FN``.
#: "qwen3_5_text"/"qwen3_5_moe_text" are included alongside their un-suffixed
#: siblings (liger maps both to the SAME apply function) because MEASURED:
#: ``AutoModelForCausalLM.from_pretrained`` on Qwen3.6-27B/35B-A3B does not
#: return the multimodal ``Qwen3_5For*ConditionalGeneration`` class at all --
#: it resolves to the text-only ``Qwen3_5For*CausalLM`` class, whose
#: ``config.model_type`` is the "_text" variant. Gemma4 does not have this
#: quirk (gemma-4-26B-A4B-it/-31B-it both measured as plain "gemma4" through
#: the identical loader), so "gemma4_text" is deliberately NOT added here --
#: it is untested surface no model on this estate ever actually produces.
LIGER_GENERIC_MODEL_TYPES: tuple[str, ...] = (
    "gemma4",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
)

#: model_type values this module owns an FS-written forward patch for, because
#: liger_kernel 0.8.4 ships none that reaches them (see module docstring).
LIGER_CUSTOM_MODEL_TYPES: tuple[str, ...] = ("gemma4_unified",)

#: The union -- what ``--fused-loss liger`` can actually be applied to on this
#: plane's installed liger_kernel. Named explicitly (not computed in the
#: refusal message) so a refusal can quote the accepted set verbatim.
FUSED_LOSS_SUPPORTED_MODEL_TYPES: tuple[str, ...] = (
    LIGER_GENERIC_MODEL_TYPES + LIGER_CUSTOM_MODEL_TYPES
)

FUSED_LOSS_BACKENDS: tuple[str, ...] = ("liger",)


def fused_loss_refusal_reason(backend: str, model_type: str | None) -> str | None:
    """None if `backend` can be applied to `model_type`; else the refusal message.

    Pure and import-free (no liger_kernel/transformers needed) so the
    REFUSE-before-touching-anything decision can be made, and tested, without
    the optional dependency installed.
    """
    if backend not in FUSED_LOSS_BACKENDS:
        # Unreachable via cli.py's choices=("liger",); kept for programmatic
        # TrainConfig callers that bypass the CLI parser.
        return f"--fused-loss={backend!r} is not one of {FUSED_LOSS_BACKENDS}"
    if model_type is None:
        return (
            "--fused-loss=liger is declared, but the loaded model exposes no "
            "config.model_type to dispatch a fused-loss patch on"
        )
    if model_type not in FUSED_LOSS_SUPPORTED_MODEL_TYPES:
        return (
            f"--fused-loss=liger is declared, but model_type={model_type!r} is "
            f"not one of {FUSED_LOSS_SUPPORTED_MODEL_TYPES}, the model types this "
            "plane's installed liger_kernel (plus its FS-owned gemma4_unified "
            "patch) can fuse. Train without --fused-loss, or on a supported "
            "family -- never silently unfused under a declared label"
        )
    return None


def _ensure_liger_flce_is_dtensor_safe() -> None:
    """Idempotently wrap liger's fused-linear-cross-entropy to accept DTensor args.

    MEASURED (Qwen3.6-27B, FSDP version 2, 2 ranks): liger_kernel 0.8.4's own
    ``fused_linear_cross_entropy`` reads ``self.lm_head.weight`` directly and
    feeds it straight into a raw ``aten.mm``-based kernel. Under FSDP2 that
    weight is a ``torch.distributed.tensor.DTensor`` -- FSDP2 keeps parameters
    DTensor-wrapped even once "unsharded" for the active forward (a
    Replicate-placement DTensor, not a plain ``torch.Tensor``) -- while the
    hidden states feeding the SAME matmul are a plain Tensor, which raised
    ``aten.mm.out got mixed torch.Tensor and DTensor, need to convert all
    torch.Tensor to DTensor before calling distributed operators!`` the moment
    training reached the first loss computation. Grepping the installed
    package confirms liger ships DTensor handling for SEVERAL other kernels
    (rms_norm, layer_norm, softmax, swiglu, jsd) but NONE on the
    fused-linear-cross-entropy path -- an upstream gap, not a usage error.

    Every covered model_type's loss -- both liger's own generic
    qwen3_5/qwen3_5_moe/gemma4 forwards (applied via
    ``_apply_liger_kernel_to_instance``) and this module's FS-owned
    gemma4_unified forward -- funnels through exactly ONE function,
    ``liger_kernel.transformers.model.loss_utils`` imports
    ``liger_kernel.transformers.functional`` as a MODULE (``import ... as F``),
    not via ``from ... import``, so patching the attribute on the module
    object is visible to every caller regardless of when it imported --
    making this the single, narrow choke point to fix rather than
    reimplementing liger's own per-model forwards. FSDP1/DDP/no-distribution
    pass plain ``torch.Tensor`` arguments straight through unchanged (the
    DTensor check is a no-op for them), so this is inert off FSDP2.

    ``DTensor.full_tensor()`` is autograd-aware (correct backward: redistributes
    the gradient back to the sharded local shard), and is a cheap local
    materialize with no new collective when the parameter is already
    Replicate-placement -- which it is here, since this runs while the root
    FSDP unit's forward hook has already unsharded it for the current call.
    """
    import liger_kernel.transformers.functional as _liger_functional

    original = _liger_functional.liger_fused_linear_cross_entropy
    if getattr(original, "_fs_dtensor_safe", False):
        return

    def _materialize(value: Any) -> Any:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return value
        return value.full_tensor() if isinstance(value, DTensor) else value

    def _dtensor_safe(
        input: Any, weight: Any, target: Any, bias: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        return original(
            _materialize(input),
            _materialize(weight),
            _materialize(target),
            _materialize(bias),
            *args,
            **kwargs,
        )

    _dtensor_safe._fs_dtensor_safe = True  # type: ignore[attr-defined]
    _dtensor_safe.__name__ = original.__name__
    _dtensor_safe.__doc__ = original.__doc__
    _liger_functional.liger_fused_linear_cross_entropy = _dtensor_safe


def apply_fused_loss(model: Any, model_type: str) -> str:
    """Patch `model` IN PLACE for backend='liger'. Returns a one-line log/manifest message.

    Callers MUST check :func:`fused_loss_refusal_reason` first and refuse on a
    non-None result; reaching this function with an unsupported model_type is
    a caller defect (ValueError), not an operator-facing refusal.

    May raise ImportError if liger_kernel (or, for the generic path, its
    gemma4/qwen3_5 submodules) is not installed -- callers should translate
    that into the same REFUSE (96) + EXTRA_HINT shape the rest of loop.py uses
    for every other optional-dependency gap, never let it escape as a
    traceback.
    """
    # The caller-defect check comes before any liger import, so an unsupported
    # model_type is reported as what it is even where liger is not installed.
    if model_type not in LIGER_GENERIC_MODEL_TYPES and model_type != "gemma4_unified":
        raise ValueError(
            f"apply_fused_loss called with unsupported model_type={model_type!r}; "
            "callers must call fused_loss_refusal_reason first and refuse on a "
            "non-None result rather than reaching here"
        )
    _ensure_liger_flce_is_dtensor_safe()
    if model_type in LIGER_GENERIC_MODEL_TYPES:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance  # noqa: PLC0415

        _apply_liger_kernel_to_instance(
            model=model,
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False,
            geglu=False,
            swiglu=False,
            layer_norm=False,
        )
        return (
            f"fused_loss=liger: _apply_liger_kernel_to_instance(model_type="
            f"{model_type!r}) with fused_linear_cross_entropy=True and every "
            "other liger kernel swap explicitly off (rope, cross_entropy, "
            "rms_norm, geglu, swiglu, layer_norm) -- only the loss computation "
            "changes, nothing else"
        )
    if model_type == "gemma4_unified":
        import types  # noqa: PLC0415

        model.forward = types.MethodType(_fs_gemma4_unified_lce_forward, model)
        return (
            "fused_loss=liger: FS-owned forward patch for model_type="
            "'gemma4_unified' (liger_kernel 0.8.4 ships no patch that reaches "
            "this class -- its own gemma4 patch targets a different module, "
            "transformers.models.gemma4; see this module's docstring). "
            "fused_linear_cross_entropy only; final_logit_softcapping is "
            "applied exactly as the stock forward applies it"
        )
    raise ValueError(
        f"apply_fused_loss called with unsupported model_type={model_type!r}; "
        "callers must call fused_loss_refusal_reason first and refuse on a "
        "non-None result rather than reaching here"
    )


def _fs_gemma4_unified_lce_forward(
    self: Any,
    input_ids: Any = None,
    pixel_values: Any = None,
    pixel_values_videos: Any = None,
    input_features: Any = None,
    attention_mask: Any = None,
    input_features_mask: Any = None,
    position_ids: Any = None,
    image_position_ids: Any = None,
    video_position_ids: Any = None,
    past_key_values: Any = None,
    mm_token_type_ids: Any = None,
    inputs_embeds: Any = None,
    labels: Any = None,
    use_cache: Any = None,
    logits_to_keep: Any = 0,
    mm_encoder_outputs: Any = None,
    **kwargs: Any,
) -> Any:
    """Fused-linear-cross-entropy forward for ``Gemma4UnifiedForConditionalGeneration``.

    Mirrors the stock ``forward`` (transformers 5.18.0,
    ``transformers/models/gemma4_unified/modeling_gemma4_unified.py``)
    argument-for-argument -- same call into ``self.model(...)``, same
    ``logits_to_keep`` slicing, same ``final_logit_softcapping`` application,
    same ``return_dict`` tuple-or-object contract -- and takes the IDENTICAL
    non-fused branch the stock forward runs whenever ``labels`` is absent
    (generation, or an eval that asks for logits), so only the
    labelled-loss path differs.

    The fused path engages whenever ``labels is not None``, regardless of
    ``self.training``: a plain ``self.training`` gate (which liger's OWN
    generic gemma4/qwen3_5/qwen3_5_moe forwards -- the ``_apply_liger_kernel_
    to_instance`` path above -- DO use, measured by reading
    ``liger_kernel.transformers.model.gemma4.causal_forward``'s
    ``skip_logits = self.training and (labels is not None or shift_labels is
    not None)``) would still materialize the full ``(batch, seq,
    vocab=262144)`` fp32 logits tensor on an EVAL pass at long context, the
    exact 32 GiB-at-32768-tokens shape this flag exists to avoid (see module
    docstring) -- eval is not exempt from that allocation just because no
    gradient follows it. This is a DELIBERATE divergence from liger's own
    families, made because this is the one family this plane owns the
    forward for and can fix outright; the liger-generic families keep
    liger's training-only gate (an upstream limitation, not something this
    plane patches), so an eval-with-labels pass on THOSE families still
    materializes full logits today. ``logits`` is ``None`` in the fused
    branch's returned output, matching liger's own convention for the
    skip-logits case; nothing downstream in this plane's SFT loop reads
    ``.logits``, only ``.loss``.
    """
    import torch  # noqa: PLC0415
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss  # noqa: PLC0415
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (  # noqa: PLC0415
        Gemma4UnifiedCausalLMOutputWithPast,
    )

    # Popped, never forwarded: the stock forward has no `return_dict`
    # parameter of its own (its `@can_return_tuple` decorator consumes it),
    # and `self.model(..., return_dict=True, **kwargs)` below ALREADY passes
    # return_dict=True explicitly -- a caller-supplied return_dict left in
    # kwargs would collide with that and raise "multiple values for keyword
    # argument 'return_dict'" (measured) rather than simply being honoured.
    return_dict = kwargs.pop("return_dict", True)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        mm_encoder_outputs=mm_encoder_outputs,
        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )
    kept_hidden_states = hidden_states[:, slice_indices, :]
    text_config = self.config.get_text_config()

    logits = None
    loss = None
    if labels is not None:
        loss = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            hidden_size=text_config.hidden_size,
            final_logit_softcapping=getattr(text_config, "final_logit_softcapping", None),
            **kwargs,
        )
    else:
        logits = self.lm_head(kept_hidden_states)
        final_logit_softcapping = text_config.final_logit_softcapping
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

    result = Gemma4UnifiedCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        shared_kv_states=outputs.shared_kv_states,
    )
    return result if return_dict else result.to_tuple()
