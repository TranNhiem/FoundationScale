import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

__all__ = [
    "PromptSurface",
    "resolve_prompt_surface",
    "encode_prompts",
    "chat_template_or_refuse",
]


def _refuse_exit_96(message: str) -> NoReturn:
    """Loud refusal, exit 96. Never 1, never a silent fallback.

    The defect being fixed (trainer.py #371) was pixels dropped between
    loader and model with no error while the run looked healthy. Every
    refusal here names the sample / path / modality so the failure is
    actionable, then exits 96 rather than degrading quietly.
    """
    print(f"REFUSAL (exit 96): {message}", file=sys.stderr)
    raise SystemExit(96)


@dataclass(frozen=True)
class PromptSurface:
    """How prompts become model inputs for this run.

    kind: 'processor' when multimodal routing is required, 'tokenizer' for
    text-only corpora. surface: the loaded AutoProcessor/AutoTokenizer.
    reason: printable justification -- every choice is announced, because a
    quiet surface choice was how vision records once trained on text alone.
    """

    kind: str  # 'processor' | 'tokenizer'
    surface: Any
    reason: str
    supports_images: bool


def resolve_prompt_surface(model_id: str, needs_images: bool) -> PromptSurface:
    """Pick and load the prompt surface.

    If needs_images, AutoProcessor is REQUIRED. Falling back to a tokenizer
    here would reproduce the silent-drop defect (#371) at the resolution
    layer: the record keeps its images, the pixels still never reach the
    model. So a processor that fails to load is a refusal (96), never a
    downgrade. If not needs_images, AutoTokenizer is sufficient -- a text
    corpus pays no multimodal cost (measured: bare prompt is 10 tokens; ONE
    image costs EXACTLY 258.0 more, linear at n=1,2,4,8).

    transformers is imported FUNCTION-LOCALLY: this module must import on a
    torch-free host.
    """
    if needs_images:
        try:
            from transformers import AutoProcessor
        except ImportError:
            _refuse_exit_96(
                "corpus carries images but transformers is absent; "
                "AutoProcessor cannot be imported and NO tokenizer fallback "
                "is permitted -- that fallback is the silent-drop defect "
                "being fixed (pixels discarded, run looks healthy)"
            )
        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception as exc:
            _refuse_exit_96(
                f"corpus carries images but AutoProcessor failed to load for "
                f"{model_id!r}: {exc!r}. Refusing rather than downgrading to "
                "a tokenizer; on gemma-4-E4B the processor is what emits "
                "pixel_values/image_position_ids and spends the 258.0 "
                "tokens/image the model expects -- a tokenizer would emit "
                "text-only keys (input_ids, attention_mask, "
                "mm_token_type_ids) and silently drop the pixels"
            )
        reason = (
            f"processor path: corpus carries images; AutoProcessor loaded "
            f"for {model_id!r}. Measured on gemma-4-E4B: with-image keys add "
            "pixel_values and image_position_ids; cost is EXACTLY 258.0 "
            "tokens per image (linear, zero variance at n=1,2,4,8); context "
            "window is 131072 (text_config.max_position_embeddings)."
        )
        print(reason)
        return PromptSurface(
            kind="processor",
            surface=processor,
            reason=reason,
            supports_images=True,
        )

    try:
        from transformers import AutoTokenizer
    except ImportError:
        _refuse_exit_96(
            "text-only corpus but transformers is absent; AutoTokenizer "
            "cannot be imported and no pure-python fallback exists for "
            "chat-template tokenisation"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        _refuse_exit_96(
            f"AutoTokenizer failed to load for {model_id!r}: {exc!r}; "
            "cannot build prompts without it, refusing rather than guessing "
            "a tokeniser"
        )
    reason = (
        f"tokenizer path: corpus is text-only; AutoTokenizer suffices for "
        f"{model_id!r}. No images means no 258.0-token/image spend (prompt "
        "width measured at 10 tokens bare), so the processor's extra "
        "surface would buy nothing."
    )
    print(reason)
    return PromptSurface(
        kind="tokenizer",
        surface=tokenizer,
        reason=reason,
        supports_images=False,
    )


def chat_template_or_refuse(surface: PromptSurface) -> str:
    """Return the surface's chat_template, or refuse (96) if it has none.

    Mirrors the trainer's existing behaviour: generating without the
    template would silently change the prompt distribution the checkpoint
    was aligned to.
    """
    template = getattr(surface.surface, "chat_template", None)
    if not template:
        _refuse_exit_96(
            f"{surface.kind} surface for this run has no chat_template; "
            "refusing rather than hand-rolling a prompt format the "
            "checkpoint was never trained on"
        )
    return str(template)


def _load_image_or_refuse(sample_id: str, path: str) -> Any:
    """Load one image path with PIL; refuse (96) on a missing/unreadable file.

    A missing image silently becoming a text-only prompt is the #371 defect
    one layer down, so this names the sample AND the path. PIL is imported
    function-locally (no module-scope heavy deps).
    """
    if not Path(path).exists():
        _refuse_exit_96(
            f"sample {sample_id!r} references image {path!r}, which does "
            "not exist on disk. Dropping it would recreate the silent-drop "
            "defect one layer down; fix the corpus or the path."
        )
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        _refuse_exit_96(
            f"sample {sample_id!r} has images but PIL is absent; cannot "
            "load pixels and no fallback is permitted"
        )
    try:
        image = Image.open(path)
        image.load()  # decode NOW: a lazy handle failing mid-batch is quieter
    except Exception as exc:
        _refuse_exit_96(
            f"sample {sample_id!r} image {path!r} could not be decoded: "
            f"{exc!r}; refusing rather than substituting an empty frame"
        )
    return image


def encode_prompts(
    surface: PromptSurface,
    samples: Sequence[Any],
    device: str,
) -> Any:
    """Encode one chunk of samples into a batch mapping on ``device``.

    With images present, the chat template is fed content BLOCKS --
    [{'type': 'image'} for each image, then {'type': 'text', ...}] -- and
    the processor receives the matched PIL images in order, so pixel_values
    line up with the 258.0-token-per-image expansion measured on
    gemma-4-E4B (10 bare -> 268 with ONE image). Text-only samples go
    through as plain strings.

    Refusals (96): a tokenizer surface handed images (silent drop), an
    image path that does not exist (silent drop one layer down), no chat
    template. If NO sample in the chunk carries images but the surface is a
    processor, the tokenizer-style text path is used -- there are no pixels
    to drop, so this is not the defect.
    """
    chat_template_or_refuse(surface)

    any_images = any(getattr(s, "images", ()) for s in samples)
    if any_images and not surface.supports_images:
        carriers = [s.sample_id for s in samples if getattr(s, "images", ())]
        _refuse_exit_96(
            f"{len(carriers)} sample(s) carry images but the surface is a "
            f"tokenizer ({carriers[:5]}): encoding them would drop the "
            "pixels -- the exact defect this module exists to refuse"
        )

    prompts: list[Any] = []
    flat_images: list[Any] = []  # processor order: sample-major, then per-sample order
    for sample in samples:
        images = getattr(sample, "images", ()) or ()
        if images:
            loaded = [_load_image_or_refuse(sample.sample_id, p) for p in images]
            flat_images.extend(loaded)
            conversation = [
                {
                    "role": role,
                    "content": (
                        [{"type": "image"} for _ in loaded] + [{"type": "text", "text": text}]
                    ),
                }
                for role, text in sample.prompt_turns
            ]
        else:
            conversation = [{"role": role, "content": text} for role, text in sample.prompt_turns]
        prompts.append(conversation)

    if any_images:
        # One apply_chat_template per conversation: block content is a
        # per-conversation structure, and the processor then consumes the
        # flattened image list in the same order the blocks appeared.
        texts = [
            surface.surface.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in prompts
        ]
        encoded = surface.surface(
            text=texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    else:
        texts = surface.surface.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )
        # `text=` is a KEYWORD on both surfaces, deliberately. A processor's
        # first POSITIONAL parameter is `images`, not `text`, so a positional
        # call here would hand the vision tower a list of prompt strings; a
        # tokenizer names its first parameter `text` too, so one keyword form
        # is correct for both and the ambiguity simply stops existing. This
        # branch is reached on a MIXED corpus: the run resolves a processor
        # because some sample carries images, then a chunk arrives with none.
        #
        # add_special_tokens=False on BOTH paths and in the image branch
        # above: apply_chat_template has already emitted BOS and the turn
        # markers, so letting the tokenizer add them again yields a second
        # BOS. That would make an image-carrying corpus train against a
        # different prompt distribution than a text-only one -- silent, and
        # exactly the instrument-mismatch class this plane keeps hitting.
        encoded = surface.surface(
            text=texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

    # BatchEncoding supports .to(device) natively; a plain mapping of tensors
    # needs the explicit per-value move. torch stays function-local.
    if hasattr(encoded, "to"):
        return encoded.to(device)
    import torch  # noqa: F401 -- presence check; .to calls below are on tensors

    return {key: value.to(device) for key, value in encoded.items()}
