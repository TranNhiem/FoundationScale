import pytest
from rle import (
    RLEError,
    delta_decode,
    delta_encode,
    rle_decode,
    rle_decode_pairs,
    rle_encode,
    rle_encode_pairs,
)


def test_encode_pairs_simple() -> None:
    assert rle_encode_pairs("aaabbc") == [("a", 3), ("b", 2), ("c", 1)]


def test_encode_pairs_empty() -> None:
    assert rle_encode_pairs("") == []


def test_encode_pairs_all_single_chars() -> None:
    assert rle_encode_pairs("abcd") == [("a", 1), ("b", 1), ("c", 1), ("d", 1)]


def test_encode_pairs_long_run() -> None:
    assert rle_encode_pairs("z" * 12) == [("z", 12)]


def test_decode_pairs_roundtrip() -> None:
    assert rle_decode_pairs([("x", 1), ("y", 4), ("z", 2)]) == "xyyyyzz"


def test_decode_pairs_rejects_bad_count() -> None:
    with pytest.raises(RLEError):
        rle_decode_pairs([("a", 0)])


def test_decode_pairs_rejects_bad_char() -> None:
    with pytest.raises(RLEError):
        rle_decode_pairs([("ab", 3)])


def test_rle_encode_basic() -> None:
    assert rle_encode("aaabcccc") == "3ab4c"


def test_rle_encode_single_characters_stay_bare() -> None:
    assert rle_encode("abcd") == "abcd"


def test_rle_encode_empty() -> None:
    assert rle_encode("") == ""


def test_rle_encode_long_run_uses_x_multipler() -> None:
    assert rle_encode("a" * 12) == "x34a"  # 12 re-encoded via 3*4 -> see module


def test_rle_decode_basic() -> None:
    assert rle_decode("3ab4c") == "aaabcccc"


def test_rle_decode_bare_chars() -> None:
    assert rle_decode("xy") == "xy"


def test_rle_decode_rejects_count_without_char() -> None:
    with pytest.raises(RLEError):
        rle_decode("3a4")


def test_rle_decode_rejects_one_as_explicit_count() -> None:
    with pytest.raises(RLEError):
        rle_decode("1z")


def test_rle_roundtrip_with_digits_in_string() -> None:
    # digits adjacent to counts are ambiguous but decoded strictly
    with pytest.raises(RLEError):
        rle_decode(rle_encode("222"))


def test_rle_roundtrip_letters() -> None:
    s = "aaabbccccdde"
    assert rle_decode(rle_encode(s)) == s


def test_delta_encode_basic() -> None:
    assert delta_encode([5, 7, 6, 10]) == [5, 2, -1, 4]


def test_delta_encode_empty() -> None:
    assert delta_encode([]) == []


def test_delta_encode_single() -> None:
    assert delta_encode([-3]) == [-3]


def test_delta_decode_basic() -> None:
    assert delta_decode([5, 2, -1, 4]) == [5, 7, 6, 10]


def test_delta_decode_empty() -> None:
    assert delta_decode([]) == []


def test_delta_roundtrip_negative_and_zero() -> None:
    vals = [0, -1, -1, 3, 3, -5]
    assert delta_decode(delta_encode(vals)) == vals


def test_delta_encode_rejects_non_list() -> None:
    with pytest.raises(TypeError):
        delta_encode((1, 2, 3))


def test_delta_decode_rejects_bools() -> None:
    with pytest.raises(TypeError):
        delta_decode([1, True, 2])
