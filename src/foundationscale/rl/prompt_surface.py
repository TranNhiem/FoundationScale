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
    "train_image_collator_or_refuse",
    "refuse_if_pixel_column_dropped",
    "PIXEL_KEY",
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
        from PIL import Image
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
    # ONE sub-list per sample, EMPTY when that sample carries no images -- never
    # a flat list. MEASURED on two real processor families, same two rows:
    #     gemma-4-E4B-it  flat    -> ValueError: Received inconsistently sized
    #                                batches of images (1) and text (2)
    #     gemma-4-E4B-it  nested  -> ok, input_ids [2, 282]
    #     qwen2.5-vl      flat    -> ok, input_ids [2, 2362]
    #     qwen2.5-vl      nested  -> ok, input_ids [2, 2362]  (byte-identical)
    # A family that normalises a flat list collapses it to ONE image-batch and
    # then compares N images against M texts. Nesting fixes that family and is
    # a strict no-regression for the one that already worked.
    nested_images: list[list[Any]] = []
    for sample in samples:
        images = getattr(sample, "images", ()) or ()
        loaded = [_load_image_or_refuse(sample.sample_id, p) for p in images]
        nested_images.append(loaded)
        if loaded:
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
        # per-conversation structure, and the processor then consumes that
        # conversation's own image sub-list, positionally, in the same order
        # the blocks appeared.
        texts = [
            surface.surface.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in prompts
        ]
        encoded = surface.surface(
            text=texts,
            images=nested_images,
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
        # above. MEASURED against a real Idefics3Processor, not argued:
        #     with the kwarg   : width 3038, first ids [128000, 1502, 25]
        #     without it       : width 3039, first ids [128000, 128000, ...]
        # BOS (128000) is emitted TWICE without it, because
        # apply_chat_template already wrote it. The same run confirmed a
        # real processor ACCEPTS the kwarg rather than raising TypeError.
        # Left unset, an image-carrying corpus would train against a
        # different prompt distribution than a text-only one -- silent,
        # and exactly the instrument-mismatch class this plane keeps
        # hitting.
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


PIXEL_KEY = "pixel_values"
# The key the vision tower consumes. Declared ONCE here because the train
# plane's survival probe, the collator and the verification adjudicator must
# agree on the spelling, and the agreement must be imported, not re-typed.


def refuse_if_pixel_column_dropped(
    batch_keys: Any,
    image_column: str,
    pixel_key: str = PIXEL_KEY,
) -> None:
    """REFUSE (96) when a collator's output batch lost the pixel column."""
    keys = {str(k) for k in batch_keys}
    if pixel_key not in keys:
        _refuse_exit_96(
            f"image column {image_column!r} is DECLARED, but the batch the "
            f"model would actually receive has keys {sorted(keys)}: the pixel "
            f"column {pixel_key!r} was DROPPED between the dataset and the "
            "forward. That is the silent-drop defect (#371/#410, and #422's "
            "class -- a declared axis that is not executed is a refusal, not "
            "a pass). The run is refused; it never trains text-only under a "
            "multimodal label."
        )


# transformers writes int(1e30) into model_max_length to mean "this family
# never declared one". MEASURED: qwen2.5-vl reports a real 131072, while
# gemma-4-E4B-it reports exactly 1000000000000000019884624838656. Anything at
# or above this threshold is the sentinel, not a bound.
_UNSET_MODEL_MAX_LENGTH = 10**15


def _refuse_if_image_batch_exceeds_declared_window(
    surface: PromptSurface, batch: Any, image_column: str
) -> None:
    """Refuse 96 when an untruncated image batch overruns the model's own window.

    The image path deliberately does not truncate, because truncating a batch
    that carries image placeholders drops pixels on the floor -- the silent
    defect this module exists to refuse. That leaves one honest failure mode:
    the encoded batch is simply wider than the model can accept. This refuses
    it, naming both numbers, rather than handing the forward a batch that will
    fail somewhere less legible.

    The bound is read from the family's OWN tokenizer, never hardcoded. When
    the family declares no window, the width is UNMEASURABLE against a bound
    that does not exist, so this returns without refusing -- it does not invent
    a threshold, and it does not pretend the check ran.
    """
    tokenizer = getattr(surface.surface, "tokenizer", None)
    declared = getattr(tokenizer, "model_max_length", None)
    if not isinstance(declared, int) or declared >= _UNSET_MODEL_MAX_LENGTH:
        return
    input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
    if input_ids is None:
        return
    width = int(input_ids.shape[-1])
    if width > declared:
        _refuse_exit_96(
            f"image column {image_column!r} encoded to {width} tokens per row, "
            f"wider than the {declared} this model declares. The image path "
            "does not truncate on purpose -- truncating a batch that carries "
            "image placeholders drops pixels between the dataset and the "
            "forward, which is the silent-drop defect. Reduce the images per "
            "row or the text length; this refuses rather than corrupts."
        )


def train_image_collator_or_refuse(
    surface: PromptSurface,
    *,
    image_column: str,
    max_length: int,
    text_column: str = "text",
) -> Any:
    """WIDENED FOR #410: the surface now also emits the TRAIN plane's collator."""
    if surface.kind != "processor" or not surface.supports_images:
        _refuse_exit_96(
            f"a train-time image collator was requested for column "
            f"{image_column!r} but the resolved surface is {surface.kind!r}: "
            "encoding images through a tokenizer is the silent-drop defect "
            "one layer up, so this refuses rather than collates"
        )

    def collate(features: Sequence[Any]) -> Any:
        # Images nest ONE SUB-LIST PER FEATURE, including an EMPTY sub-list for
        # a row that carries none. MEASURED, same two rows, both families:
        #     shape                         gemma-4-E4B-it     qwen2.5-vl
        #     flat  [img]     / 2 texts     ValueError 1 vs 2  ok
        #     nested [[img]]  / 2 texts     ValueError 1 vs 2  ok
        #     nested [[img],[]] / 2 texts   ok  [2, 277]       ok  [2, 2358]
        # Dropping the empty sub-list reproduces the original defect: the
        # processor normalises the short outer list to one image-batch and then
        # reads N images against M texts. The empty sub-list is what keeps the
        # images list and the texts list the same length.
        texts: list[str] = []
        nested_images: list[list[Any]] = []
        any_images = False
        for i, feature in enumerate(features):
            row = feature if isinstance(feature, dict) else vars(feature)
            paths = row.get(image_column) or []
            if isinstance(paths, (str, Path)):
                paths = [paths]
            loaded = [_load_image_or_refuse(f"train-row[{i}]", str(p)) for p in paths]
            nested_images.append(loaded)
            any_images = any_images or bool(loaded)
            # The placeholder comes from the processor's OWN chat template, so
            # it is whatever token that family uses and nothing is hardcoded.
            # add_generation_prompt=False: this is the TRAIN plane and labels
            # are the input_ids, so an assistant-turn opener would be trained on.
            conversation = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"} for _ in loaded]
                        + [{"type": "text", "text": str(row.get(text_column, ""))}]
                    ),
                }
            ]
            texts.append(
                surface.surface.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )

        if any_images:
            # NO truncation on the image path. truncation=True at any constant
            # max_length desynchronises the placeholders from the pixels, and no
            # constant can be right: MEASURED, the same two corpus images expand
            # to 2337 and 1026 tokens through one family's processor and ~258
            # through another's. Raising 128 to 1024 fixes one and breaks the
            # other. The bound is per-image, per-family AND per-row, so it
            # cannot be a literal; an overrun is refused below, never truncated.
            batch = surface.surface(
                text=texts,
                images=nested_images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            _refuse_if_image_batch_exceeds_declared_window(surface, batch, image_column)
        else:
            # Zero pixels across every row, so there are no placeholders to
            # desynchronise and truncation is safe. max_length keeps its
            # original meaning on exactly this path.
            batch = surface.surface(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch

    return collate
