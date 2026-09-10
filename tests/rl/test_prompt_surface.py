"""prompt_surface: the modality routing layer, and its four refusals (#371).

WHY THIS MODULE EXISTS. corpus.py has always parsed `image` and `video` into
Sample.images / Sample.video, and its docstring advertised multi-image and
video support. trainer.py referenced neither field and held an AutoTokenizer,
never an AutoProcessor, so a vision record loaded cleanly, validated, and
trained on the TEXT ALONE with the pixels dropped between loader and model --
no error, no UNMEASURED line, every emitted signal healthy.

MEASURED on gemma-4-E4B, which is why this layer is worth having: the model is
fully tri-modal and one processor serves all of it.
    text bare        : 10 tokens   (input_ids, attention_mask, mm_token_type_ids)
    + one image      : 268 tokens  (+ pixel_values, image_position_ids)
                       EXACTLY 258.0 tok/image, linear at n = 1, 2, 4, 8
    + one sec audio  :  38 tokens  (+ input_features, input_features_mask)
    video            : native -- the processor has .video_processor
    context window   : 131072

EVERY LEG HERE IS A REFUSAL LEG. That is deliberate: the defect being fixed was
a silent fallback, so the thing worth pinning is that each degradation path is
loud. A test that only checked the happy path would pass on the broken code.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl.corpus import Sample
from foundationscale.rl.prompt_surface import (
    PromptSurface,
    chat_template_or_refuse,
    encode_prompts,
    resolve_prompt_surface,
)


def _sample(sample_id: str = "s0", images: tuple[str, ...] = ()) -> Sample:
    return Sample(
        sample_id=sample_id,
        prompt_turns=(("user", "what is this?"),),
        response="a cat",
        gold="A",
        images=images,
    )


def test_text_only_batch_resolves_a_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    # No images => a tokenizer is sufficient, and asking for a processor would
    # impose a dependency the batch does not need.
    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(x=1)),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(y=2)),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    surface = resolve_prompt_surface("any/model", needs_images=False)
    assert surface.kind == "tokenizer"
    assert surface.supports_images is False
    assert surface.reason  # the choice must be justified, not silent


def test_image_batch_resolves_a_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(x=1)),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(y=2)),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    surface = resolve_prompt_surface("any/model", needs_images=True)
    assert surface.kind == "processor"
    assert surface.supports_images is True


def test_a_processor_that_will_not_load_is_REFUSED_not_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # THE LOAD-BEARING LEG. If AutoProcessor cannot load for an image batch,
    # falling back to the tokenizer would silently discard the pixels that
    # corpus.py already parsed -- reinstating the exact defect this module was
    # written to remove. A refusal is the only correct outcome, and exit 96 is
    # the REFUSE code. The failing input: any environment where AutoProcessor
    # raises for a batch that carries images.
    def boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("no processor config in this checkpoint")

    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(x=1)),
        AutoProcessor=SimpleNamespace(from_pretrained=boom),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(SystemExit) as excinfo:
        resolve_prompt_surface("any/model", needs_images=True)
    assert excinfo.value.code == 96


def test_a_surface_without_a_chat_template_is_refused() -> None:
    # Mirrors the trainer's own rule: concatenating strings by hand would
    # silently change the prompt distribution the objective is priced against.
    surface = SimpleNamespace(kind="tokenizer", surface=SimpleNamespace(chat_template=None))
    with pytest.raises(SystemExit) as excinfo:
        chat_template_or_refuse(surface)
    assert excinfo.value.code == 96


def test_a_present_chat_template_is_returned_not_refused() -> None:
    # The complementary leg. Without it the guard above could refuse
    # EVERYTHING and still pass -- the vacuous-gate shape this repo keeps
    # finding, most recently in a control that measured whether the machine
    # had a GPU rather than whether a fixture pinned anything.
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    surface = SimpleNamespace(kind="tokenizer", surface=SimpleNamespace(chat_template=template))
    assert chat_template_or_refuse(surface) == template


class _FakeTokens(dict[str, Any]):
    """A BatchEncoding-shaped mapping: a dict that also exposes .to(device).

    encode_prompts only needs the .to protocol; giving it this fake keeps the
    test torch-free while pinning that the returned batch lands on the device.
    """

    def __init__(self) -> None:
        super().__init__(input_ids="ids", attention_mask="mask")
        self.moved_to: str | None = None

    def to(self, device: str) -> _FakeTokens:
        self.moved_to = device
        return self


class _FakeTensor:
    # Only what encode_prompts's plain-mapping branch touches: .to(device).
    def __init__(self, label: str) -> None:
        self.label = label
        self.device: str | None = None

    def to(self, device: str) -> _FakeTensor:
        self.device = device
        return self


class _LoadedFakeImage:
    # PIL.Image has .load(); the loader also relies on the object surviving
    # past Image.open returning, so keep the path it came from for ordering
    # assertions.
    def __init__(self, path: str) -> None:
        self.path = path
        self.decoded = False

    def load(self) -> None:
        self.decoded = True


class _RecordingSurface:
    """Processor/tokenizer-shaped fake covering BOTH call styles in
    encode_prompts: a single conversation (image path) and a batch of
    conversations (text path) through apply_chat_template, and the surface
    itself being callable.
    """

    def __init__(self, encoded: Any = None) -> None:
        self.chat_template = "{{ tpl }}"  # must exist or the guard fires first
        self.encoded = encoded if encoded is not None else _FakeTokens()
        self.templated: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: Any,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        assert tokenize is False
        assert add_generation_prompt is True
        self.templated.append(conversation)
        if isinstance(conversation, list) and conversation and isinstance(conversation[0], list):
            return [f"PROMPT-{i}" for i in range(len(conversation))]
        return "PROMPT"

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.encoded


def _surface(
    kind: str, supports_images: bool, encoded: Any = None
) -> tuple[PromptSurface, _RecordingSurface]:
    rec = _RecordingSurface(encoded)
    return (
        PromptSurface(
            kind=kind, surface=rec, reason="test fixture", supports_images=supports_images
        ),
        rec,
    )


def _absent_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    # sys.modules['transformers'] = None makes `from transformers import ...`
    # raise ImportError without touching the installed environment.
    monkeypatch.setitem(sys.modules, "transformers", None)


def _fake_pil(monkeypatch: pytest.MonkeyPatch, loader: type) -> None:
    # A PIL-module-shaped double: `from PIL import Image` needs exactly one
    # attribute, and the real module's Image exposes .open.
    class _PILModule:
        Image = loader

    monkeypatch.setitem(sys.modules, "PIL", _PILModule)


def test_absent_transformers_is_refused_for_an_image_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FAILING INPUT: a vision corpus run on a host with no transformers
    # installed. On the broken shape of this code the run would fall through
    # to whatever surface happened to exist and the pixels would vanish with
    # no error; the refusal must name why (AutoProcessor cannot be imported).
    _absent_transformers(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        resolve_prompt_surface("any/model", needs_images=True)
    assert excinfo.value.code == 96


def test_absent_transformers_is_refused_for_a_text_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FAILING INPUT: a text-only corpus on a host with no transformers. There
    # is no pure-python chat-template tokeniser to fall back to, so the only
    # honest outcome is 96 -- otherwise prompts would be hand-assembled and the
    # prompt distribution the checkpoint was priced against would silently shift.
    _absent_transformers(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        resolve_prompt_surface("any/model", needs_images=False)
    assert excinfo.value.code == 96


def test_a_tokenizer_that_will_not_load_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # FAILING INPUT: a checkpoint directory whose tokenizer files are corrupt.
    # Guessing a tokeniser here (the alternative to refusing) would tokenise
    # with the wrong vocabulary and train garbage while every metric still
    # ticks; the refusal must name the model id so the operator can act.
    def boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("tokenizer.json is truncated")

    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=boom),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(SystemExit) as excinfo:
        resolve_prompt_surface("gemma/broken-checkpoint", needs_images=False)
    assert excinfo.value.code == 96
    assert "gemma/broken-checkpoint" in capsys.readouterr().err


def test_a_tokenizer_surface_handed_images_refuses_and_names_carriers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: encode_prompts called with a tokenizer PromptSurface over
    # a chunk where at least one Sample carries images. This is #371 verbatim
    # -- pixels parsed by corpus.py, never reaching the model, run looking
    # healthy -- so exit 96 and the refusal must name the carrying sample ids.
    surface, _rec = _surface(kind="tokenizer", supports_images=False)
    samples = [_sample(sample_id="vis-7", images=("/tmp/x.png",)), _sample(sample_id="txt-1")]
    with pytest.raises(SystemExit) as excinfo:
        encode_prompts(surface, samples, device="cpu")
    assert excinfo.value.code == 96
    assert "vis-7" in capsys.readouterr().err


def test_a_missing_image_path_is_refused_with_sample_and_path_named(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: a corpus row pointing at an image that was moved or
    # deleted after curation. Silently encoding the text alone would recreate
    # the drop defect one layer down, so the refusal must quadrangulate: the
    # sample id AND the offending path, so the corpus can actually be fixed.
    surface, _rec = _surface(kind="processor", supports_images=True)
    samples = [_sample(sample_id="s-missing", images=("/no/such/dir/ghost.png",))]
    with pytest.raises(SystemExit) as excinfo:
        encode_prompts(surface, samples, device="cpu")
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "s-missing" in err
    assert "/no/such/dir/ghost.png" in err


def test_pil_absent_for_an_existing_image_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # FAILING INPUT: a host where the image file exists but PIL was never
    # installed. A drop-to-text fallback here is identical to the original
    # defect (run healthy, pixels gone), so the only correct outcome is 96.
    img = tmp_path / "real.bin"
    img.write_bytes(b"\x00")
    monkeypatch.setitem(sys.modules, "PIL", None)  # `from PIL import ...` -> ImportError
    surface, _rec = _surface(kind="processor", supports_images=True)
    samples = [_sample(sample_id="s-nopil", images=(str(img),))]
    with pytest.raises(SystemExit) as excinfo:
        encode_prompts(surface, samples, device="cpu")
    assert excinfo.value.code == 96


def test_an_undecodable_image_is_refused_not_substituted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # FAILING INPUT: a file that exists but is not a decodable image (zero
    # bytes, truncated transfer). Substituting an empty/black frame would feed
    # the model a stimulus the label was never produced under; refuse instead.
    img = tmp_path / "truncated.png"
    img.write_bytes(b"\x89PNG")

    class _BoomLoader:
        @staticmethod
        def open(*_a: Any, **_k: Any) -> _LoadedFakeImage:
            raise OSError("cannot identify image file")

    _fake_pil(monkeypatch, _BoomLoader)
    surface, _rec = _surface(kind="processor", supports_images=True)
    samples = [_sample(sample_id="s-bad", images=(str(img),))]
    with pytest.raises(SystemExit) as excinfo:
        encode_prompts(surface, samples, device="cpu")
    assert excinfo.value.code == 96


def test_images_are_loaded_sample_major_and_matched_to_content_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # FAILING INPUT: a multi-sample, multi-image chunk. If images were
    # flattened in any order other than sample-major-then-per-sample, the
    # processor would pair sample B's pixels with sample A's <image> blocks --
    # wrong supervision with zero error signal. This is the happy path that
    # makes the orderings above legible: blocks per conversation must match
    # the flat image list the processor receives, pairwise.
    paths = [str(tmp_path / f"{n}.png") for n in "abc"]
    for p in paths:
        (tmp_path / p.rsplit("/", 1)[-1]).write_bytes(b"\x00")
    opened: list[str] = []

    class _Loader:
        @staticmethod
        def open(path: str, *_a: Any, **_k: Any) -> _LoadedFakeImage:
            opened.append(path)
            return _LoadedFakeImage(path)

    _fake_pil(monkeypatch, _Loader)
    surface, rec = _surface(kind="processor", supports_images=True)
    samples = [
        _sample(sample_id="s0", images=(paths[0], paths[1])),
        _sample(sample_id="s1", images=(paths[2],)),
        _sample(sample_id="s2"),  # text-only middle of an image chunk: plain content
    ]
    batch = encode_prompts(surface, samples, device="cpu")

    assert opened == paths  # load order is the pairing contract
    flat = rec.calls[0]["images"]
    assert [img.path for img in flat] == paths  # sample-major, then per-sample order
    body = rec.templated[0][0]["content"]  # s0, user turn
    assert body == [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": "what is this?"},
    ]
    assert rec.templated[1][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "what is this?"},
    ]
    assert rec.templated[2][0]["content"] == "what is this?"  # no blocks without images
    assert rec.calls[0]["text"] == ["PROMPT"] * 3  # one template application per conversation
    assert batch.moved_to == "cpu"


def test_a_text_chunk_on_a_processor_uses_the_tokenizer_style_text_path() -> None:
    # FAILING INPUT: a processor-surface run whose chunk happens to carry no
    # images. Sending this down the image path would call the surface with an
    # empty images list and per-conversation templates, which on real
    # processors mis-shapes the batch; the text path must be chosen, exactly
    # one apply_chat_template for the whole chunk, and no add_special_tokens
    # duplication (the template already emitted them).
    surface, rec = _surface(kind="processor", supports_images=True)
    samples = [_sample(sample_id="t0"), _sample(sample_id="t1")]
    batch = encode_prompts(surface, samples, device="meta")

    assert len(rec.templated) == 1
    assert len(rec.templated[0]) == 2  # batch, not per-conversation
    assert rec.calls[0]["text"] == ["PROMPT-0", "PROMPT-1"]
    assert "images" not in rec.calls[0]
    assert rec.calls[0]["add_special_tokens"] is False
    assert rec.calls[0]["padding"] is True
    assert batch.moved_to == "meta"


def test_a_plain_mapping_batch_is_moved_value_by_value_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FAILING INPUT: a surface that returns a plain dict (no .to) instead of a
    # BatchEncoding. Dropping the per-value move would leave the batch on CPU
    # while the model sits on the accelerator -- a device-mismatch crash, or
    # worse a silent host-side copy every step. torch itself is only
    # presence-checked, so a stub module keeps this host torch-free.
    fake_torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    encoded = {"input_ids": _FakeTensor("ids"), "attention_mask": _FakeTensor("mask")}
    surface, _rec = _surface(kind="tokenizer", supports_images=False, encoded=encoded)
    batch = encode_prompts(surface, [_sample()], device="meta")

    assert batch["input_ids"].device == "meta"
    assert batch["attention_mask"].device == "meta"


def test_the_train_extra_DECLARES_the_vision_backends() -> None:
    # FAILING INPUT: a clean `pip install -e ".[train]"` followed by a corpus
    # that carries one image. Measured, not reasoned: transformers cannot
    # construct ANY image processor without Pillow, and on transformers 5.x it
    # resolves to a backend PAIR (pil / torchvision) and needs one of them
    # importable. With neither, the loader raises
    #     ValueError("Could not load any image processor class for <model>")
    # which reads as a broken checkpoint, not a missing dependency.
    #
    # Both were absent from every extra while VLM training is the headline
    # capability -- #226's shape a third time ("the extra was written by
    # reading imports, not by running the path"). prompt_surface imports PIL
    # FUNCTION-LOCALLY, so no import-time signal exists and only running the
    # path reveals it.
    #
    # Parsed by text rather than tomllib because the package supports 3.10,
    # where tomllib does not exist; a skip here would be forbidden by
    # FS_FORBID_SKIPS and would silently retire the check on the one
    # interpreter CI still runs.
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^train = \[(.*?)^\]", text, re.S | re.M)
    assert match, "the [train] extra must exist -- #224 is what happens when it does not"
    block = match.group(1)
    declared = set(re.findall(r'^\s*"([A-Za-z0-9_.\-]+)', block, re.M))
    for package in ("pillow", "torchvision"):
        assert package in declared, (
            f"{package} is not declared in the [train] extra, so the advertised "
            f"image path cannot run on a clean install. Declared: {sorted(declared)}"
        )
