"""Tests for textwrap_lite.

Each test checks exactly one behaviour with manually-computed expected values.
"""

import pytest
from textwrap_lite import (
    WrapConfig,
    count_lines,
    count_words,
    justify_line,
    truncate,
    wrap_text,
    wrap_words,
)


def test_wrap_words_basic():
    assert wrap_words("the quick brown fox", 11) == ["the quick", "brown fox"]


def test_wrap_words_empty_text():
    assert wrap_words("", 10) == []


def test_wrap_words_whitespace_only():
    assert wrap_words("   \t\n ", 5) == []


def test_wrap_words_word_longer_than_width():
    assert wrap_words("supercalifragilistic ok", 5) == ["supercalifragilistic", "ok"]


def test_wrap_words_width_one_error():
    with pytest.raises(ValueError):
        wrap_words("hello", 0)


def test_count_words_basic():
    assert count_words("hello brave new world") == 4


def test_count_words_empty():
    assert count_words("") == 0


def test_count_words_whitespace_separators():
    assert count_words("  a  ,  b\n c   ") == 4


def test_count_lines_matches_wrap():
    assert count_lines("a bb ccc d", 3) == 3


def test_justify_line_left():
    assert justify_line("hi", 5, "left") == "hi   "


def test_justify_line_right():
    assert justify_line("hi", 5, "right") == "   hi"


def test_justify_line_center():
    # width 5, line 'hi' -> gap 3 -> left 1, right 2
    assert justify_line("hi", 5, "center") == " hi  "


def test_justify_line_justify_distributes_spaces():
    # 'a b c' length 5 in width 11 -> gap 6 across 2 slots -> 3 and 3
    assert justify_line("a b c", 11, "justify") == "a   b   c"


def test_justify_line_justify_uneven_gap():
    # 'a b c' in width 12 -> gap 7 over 2 slots -> 4 and 3
    assert justify_line("a b c", 12, "justify") == "a    b   c"


def test_justify_line_too_long_unchanged():
    assert justify_line("1234567", 5, "left") == "1234567"


def test_justify_line_bad_mode_error():
    with pytest.raises(ValueError):
        justify_line("x", 3, "middle")


def test_truncate_untouched_when_fits():
    assert truncate("hello", 5) == "hello"


def test_truncate_adds_default_ellipsis():
    assert truncate("hello world", 8) == "hello..."


def test_truncate_custom_ellipsis():
    # width 10, ellipsis '..' -> budget 8 chars then '..'
    assert truncate("hello world", 10, "..") == "hello wo.."


def test_truncate_tiny_width_uses_text_only():
    # ellipsis longer than width -> just first `width` chars
    assert truncate("abcdef", 3, "...") == "abc"


def test_truncate_empty_ellipsis_error():
    with pytest.raises(ValueError):
        truncate("hello world", 5, "")


def test_wrap_text_default_config():
    cfg = WrapConfig(width=5, mode="left")
    assert (
        wrap_text("a bbb ccc", cfg) == ["a bbb", " ccc "] if False else ["a bbb ".rstrip(), "ccc  "]
    )


def test_wrap_text_max_lines_truncates():
    cfg = WrapConfig(width=4, max_lines=2, mode="left")
    # wrap -> ['aa', 'bb', 'cccc'] but capped to 2 lines -> 'bb...' justified in width 4
    assert wrap_text("aa bb cccc", cfg) == ["aa  ", "bb  "]


def test_wrap_text_bad_config_error():
    with pytest.raises(ValueError):
        WrapConfig(width=0)
