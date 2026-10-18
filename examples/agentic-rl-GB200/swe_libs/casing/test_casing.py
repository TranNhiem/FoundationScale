import pytest
from casing import (
    convert,
    detect,
    split_words,
    to_camel,
    to_kebab,
    to_pascal,
    to_snake,
)


def test_split_words_normalizes_mixed_separators():
    assert split_words("User_Name-spam") == ["user", "name", "spam"]


def test_split_words_handles_consecutive_capitals():
    assert split_words("HTTPStatusCode") == ["http", "status", "code"]


def test_split_words_raises_on_empty_string():
    with pytest.raises(ValueError):
        split_words("")


def test_split_words_raises_on_separators_only():
    with pytest.raises(ValueError):
        split_words("__--__")


def test_to_snake_from_pascal():
    assert to_snake("UserName") == "user_name"


def test_to_snake_from_camel_with_numbers():
    assert to_snake("user2Name") == "user2_name"


def test_to_snake_single_word():
    assert to_snake("user") == "user"


def test_to_kebab_from_pascal():
    assert to_kebab("UserName") == "user-name"


def test_to_kebab_from_snake():
    assert to_kebab("user_name") == "user-name"


def test_to_pascal_from_snake():
    assert to_pascal("user_name") == "UserName"


def test_to_pascal_from_acronym():
    assert to_pascal("http_status_code") == "HttpStatusCode"


def test_to_camel_from_pascal():
    assert to_camel("UserName") == "userName"


def test_to_camel_single_word():
    assert to_camel("User") == "user"


def test_to_camel_from_snake_three_words():
    assert to_camel("a_b_c") == "aBC"


def test_convert_to_snake_with_explicit_source():
    assert convert("user-name", "snake", "kebab") == "user_name"


def test_convert_infers_source_automatically():
    assert convert("user-name", "pascal") == "UserName"


def test_convert_to_camel_from_kebab():
    assert convert("first-name", "camel") == "firstName"


def test_convert_garbage_target_throws():
    with pytest.raises(ValueError):
        convert("user", "flat")


def test_convert_empty_identifier_throws():
    with pytest.raises(ValueError):
        convert("   ", "snake")


def test_convert_bad_source_throws():
    with pytest.raises(ValueError):
        convert("user", "snake", "shouty")


def test_detect_kebab_case():
    assert detect("user-name") == "kebab"


def test_detect_snake_case():
    assert detect("user_name") == "snake"


def test_detect_pascal_case():
    assert detect("UserName") == "pascal"


def test_detect_camel_case():
    assert detect("userName") == "camel"
