"""Text-only tokenizer tests for ``_template_messages``.

FAST tier -- no torch, no network, no model files. MEASURED with
Qwen2.5-7B-Instruct on transformers 5.18.0, 2x A100, 2026-10-12: a plain
tokenizer's chat template concatenates ``message['content']`` as a str, so the
content-block lists the conversation path builds make its template raise
``TypeError: can only concatenate str (not "list") to str``. The fakes here pin
the helper's three decisions: flatten for a plain tokenizer, leave a multimodal
processor alone, and leave mixed-media contents and the input untouched.
"""

from __future__ import annotations

from typing import Any

import pytest

from foundationscale.train.conversation import _template_messages


class _PlainTokenizer:
    """Fake text-only tokenizer whose template concatenates content as a str."""

    def convert_tokens_to_ids(self, tokens: Any) -> Any:
        return tokens

    def apply_chat_template(self, messages: Any, tokenize: bool = False, **kwargs: Any) -> Any:
        # Qwen2.5's real template concatenates role + content as plain strings.
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("can only concatenate str (not 'list') to str")
            _ = "" + message["role"] + "\n" + content
        return [0] if tokenize else "rendered"


class _MultimodalProcessor:
    """Fake multimodal processor -- it carries a ``.tokenizer``, like ProcessorMixin."""

    def __init__(self) -> None:
        self.tokenizer = _PlainTokenizer()

    def apply_chat_template(self, messages: Any, tokenize: bool = False, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)


class _BareProcessor:
    """Generic test fake: neither ``.tokenizer`` nor ``convert_tokens_to_ids``."""

    def apply_chat_template(self, messages: Any, tokenize: bool = False, **kwargs: Any) -> Any:
        return messages


def _blocks(*texts: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text} for text in texts]


def test_plain_tokenizer_text_blocks_flatten_and_render() -> None:
    tokenizer = _PlainTokenizer()
    messages = [
        {"role": "user", "content": _blocks("hello ")},
        {"role": "assistant", "content": _blocks("wor", "ld")},
    ]
    with pytest.raises(TypeError):
        tokenizer.apply_chat_template(messages)

    rendered = _template_messages(tokenizer, messages)

    assert [message["content"] for message in rendered] == ["hello ", "world"]
    assert tokenizer.apply_chat_template(rendered) == "rendered"


def test_multimodal_processor_messages_are_returned_unchanged() -> None:
    processor = _MultimodalProcessor()
    messages = [{"role": "user", "content": _blocks("hello")}]

    assert _template_messages(processor, messages) is messages


def test_fake_processor_without_tokenizer_is_returned_unchanged() -> None:
    messages = [{"role": "user", "content": _blocks("hello")}]

    assert _template_messages(_BareProcessor(), messages) is messages


def test_mixed_text_and_media_blocks_stay_lists() -> None:
    tokenizer = _PlainTokenizer()
    messages = [
        {"role": "user", "content": _blocks("look: ") + [{"type": "image"}]},
        {"role": "assistant", "content": "plain str already"},
    ]

    rendered = _template_messages(tokenizer, messages)

    assert rendered[0]["content"] == [{"type": "text", "text": "look: "}, {"type": "image"}]
    assert isinstance(rendered[0]["content"], list)
    assert rendered[1]["content"] == "plain str already"


def test_input_messages_are_not_mutated() -> None:
    tokenizer = _PlainTokenizer()
    content = _blocks("hello")
    messages = [{"role": "user", "content": content}]

    rendered = _template_messages(tokenizer, messages)

    assert rendered is not messages
    assert rendered[0] is not messages[0]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert messages[0]["content"] is content


def test_reasoning_content_is_preserved() -> None:
    tokenizer = _PlainTokenizer()
    messages = [
        {"role": "user", "content": _blocks("question")},
        {
            "role": "assistant",
            "content": _blocks("answer"),
            "reasoning_content": "thinking out loud",
        },
    ]

    rendered = _template_messages(tokenizer, messages)

    assert rendered[0]["content"] == "question"
    assert rendered[1]["content"] == "answer"
    assert rendered[1]["reasoning_content"] == "thinking out loud"


def test_is_plain_tokenizer_classifies_tokenizer_processor_and_fake() -> None:
    from foundationscale.train.conversation import _is_plain_tokenizer

    assert _is_plain_tokenizer(_PlainTokenizer())
    assert not _is_plain_tokenizer(_MultimodalProcessor())
    assert not _is_plain_tokenizer(object())


def test_text_tokenizer_is_self_for_tokenizer_and_attribute_for_processor() -> None:
    from foundationscale.train.conversation import _text_tokenizer

    tok = _PlainTokenizer()
    proc = _MultimodalProcessor()
    assert _text_tokenizer(tok) is tok
    assert _text_tokenizer(proc) is proc.tokenizer
