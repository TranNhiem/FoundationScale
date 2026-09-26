
"""Tests for the ``clean`` op (builtin backends only)."""
from __future__ import annotations

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops import clean as clean_mod
from foundationskills.skills.data_engine.ops.clean import builtin_langid, html_to_text, pii_scan


def run_clean(records, cfg):
    stats = OpStats(name="clean")
    out = list(OPS["clean"](iter(records), cfg, stats))
    return out, stats


BASE_CFG = {
    "normalize_unicode": False,
    "fix_mojibake": False,
    "strip_html": False,
    "langid": {"allowed": None, "min_confidence": 0.5},
    "pii": {"redact": False},
}


def cfg(**overrides):
    merged = dict(BASE_CFG)
    merged.update(overrides)
    return merged


def test_unicode_nfkc(tmp_path):
    rec = {"id": "1", "text": "ﬁle café"}  # ligature fi + combining acute
    out, stats = run_clean([rec], cfg(normalize_unicode=True))
    assert out[0]["text"] == "file cafe\u0301".replace("e\u0301", "\u00e9")
    assert stats.modified["unicode_normalized"] == 1


def test_mojibake_repair():
    rec = {"id": "1", "text": "cafÃ© au lait"}
    out, stats = run_clean([rec], cfg(fix_mojibake=True))
    assert out[0]["text"] == "caf\u00e9 au lait"
    assert stats.modified["mojibake_fixed"] == 1


def test_mojibake_ignored_when_repair_does_not_help():
    rec = {"id": "1", "text": "café normal"}
    out, stats = run_clean([rec], cfg(fix_mojibake=True))
    assert out[0]["text"] == "café normal"
    assert stats.modified.get("mojibake_fixed", 0) == 0


def test_strip_html_auto():
    rec = {"id": "1", "text": "<p>Hello <b>World</b></p>"}
    out, stats = run_clean([rec], cfg(strip_html="auto"))
    assert out[0]["text"] == "Hello World"
    assert stats.modified["html_stripped"] == 1


def test_html_to_text_drops_script_and_style():
    raw = "<html><head><style>x{}</style></head><body><script>bad();</script><p>Keep me</p></body></html>"
    text = html_to_text(raw)
    assert "Keep me" in text
    assert "bad()" not in text and "x{}" not in text


def _pii_cfg(types=None):
    return cfg(pii={"redact": True, "types": types or ["email", "phone", "ipv4", "credit_card", "national_id", "ssn"]})


def test_pii_email():
    out, stats = run_clean([{"text": "mail me at jane.smith+ai@example.com please"}], _pii_cfg())
    assert "<PII_EMAIL>" in out[0]["text"]
    assert "jane.smith+ai@example.com" not in out[0]["text"]
    assert stats.modified["pii:email"] == 1


def test_pii_phone_international():
    for raw in ("+886 912 345 678", "+84 912 345 678", "+1 415 555 0132"):
        out, stats = run_clean([{"text": f"call {raw} now"}], _pii_cfg())
        assert "<PII_PHONE>" in out[0]["text"], raw
        assert stats.modified["pii:phone"] == 1


def test_pii_ipv4():
    out, stats = run_clean([{"text": "server 192.168.0.1 down"}], _pii_cfg())
    assert "<PII_IPV4>" in out[0]["text"]
    assert stats.modified["pii:ipv4"] == 1


def test_pii_credit_card_luhn_positive_and_negative():
    ok, stats = run_clean([{"text": "card 4111 1111 1111 1111 exp 01/29"}], _pii_cfg())
    assert "<PII_CREDIT_CARD>" in ok[0]["text"]
    assert stats.modified["pii:credit_card"] == 1
    # one digit off: Luhn fails -> must NOT be redacted
    bad, stats2 = run_clean([{"text": "code 4111 1111 1111 1112 here"}], _pii_cfg())
    assert "<PII_CREDIT_CARD>" not in bad[0]["text"]
    assert stats2.modified.get("pii:credit_card", 0) == 0


def test_pii_national_id_and_ssn():
    out, stats = run_clean([{"text": "id A123456789 and ssn 123-45-6789"}], _pii_cfg())
    assert "<PII_NATIONAL_ID>" in out[0]["text"]
    assert "<PII_SSN>" in out[0]["text"]
    assert stats.modified["pii:national_id"] == 1
    assert stats.modified["pii:ssn"] == 1
    # SSN must not be double-counted as a phone number
    assert stats.modified.get("pii:phone", 0) == 0


def test_pii_scan_counts_and_zeros():
    counts = pii_scan("a@b.com 192.168.0.1 123-45-6789 A123456789")
    assert counts["email"] == 1
    assert counts["ipv4"] == 1
    assert counts["ssn"] == 1
    assert counts["national_id"] == 1
    assert counts["credit_card"] == 0
    assert counts["phone"] == 0


def test_pii_applies_to_messages_and_prompt_fields(tmp_path):
    rec = {
        "messages": [{"role": "user", "content": "email me a@b.com"}, {"role": "assistant", "content": "ok"}],
        "prompt": "ping 10.0.0.1",
        "chosen": "sure",
    }
    out, stats = run_clean([rec], _pii_cfg())
    assert "<PII_EMAIL>" in out[0]["messages"][0]["content"]
    assert "<PII_IPV4>" in out[0]["prompt"]


def test_langid_fallback_en_zh_vi():
    en = {"text": "This is an ordinary English sentence about programming and data."}
    zh = {"text": "这是一段用于测试语言识别的中文文本，包含足够多的汉字。"}
    vi = {"text": "Hôm nay tôi đi học ở thành phố Hồ Chí Minh và đọc sách."}
    out, stats = run_clean([en, zh, vi], cfg(langid={"allowed": ["en", "vi"], "min_confidence": 0.5}))
    assert stats.extra["approximate_langid"] is True
    assert len(out) == 2  # zh dropped
    assert stats.dropped["lang"] == 1
    kept_langs = [r["meta"]["lang"] for r in out]
    assert kept_langs == ["en", "vi"]


def test_builtin_langid_labels():
    assert builtin_langid("A plain English sentence with enough words.")[0] == "en"
    assert builtin_langid("这是一句纯粹的中文话，用来检测。")[0] == "zh"
    assert builtin_langid("Xin chào, đây là tiếng Việt.")[0] == "vi"


def _drop_cfg(**thresholds):
    return cfg(min_chars=thresholds.pop("min_chars", None),
               max_chars=thresholds.pop("max_chars", None),
               max_non_alnum_ratio=thresholds.pop("max_non_alnum_ratio", None),
               max_line_repeat_ratio=thresholds.pop("max_line_repeat_ratio", None),
               max_char_repeat_ratio=thresholds.pop("max_char_repeat_ratio", None))


def test_drop_empty_and_too_short_and_too_long():
    out, stats = run_clean(
        [{"text": "   "}, {"text": "hi"}, {"text": "x" * 5000}],
        _drop_cfg(min_chars=10, max_chars=4000),
    )
    assert out == []
    assert stats.dropped["empty"] == 1
    assert stats.dropped["too_short"] == 1
    assert stats.dropped["too_long"] == 1


def test_drop_non_alnum():
    out, stats = run_clean([{"text": "### $$$ %%% &&& ***"}], _drop_cfg(max_non_alnum_ratio=0.5))
    assert out == []
    assert stats.dropped["non_alnum"] == 1


def test_drop_line_repeat():
    text = "\n".join(["the same repeated line here"] * 9 + ["a different closing line"])
    out, stats = run_clean([{"text": text}], _drop_cfg(max_line_repeat_ratio=0.5))
    assert out == []
    assert stats.dropped["line_repeat"] == 1


def test_drop_char_repeat():
    text = "some normal prefix then " + "x" * 400
    out, stats = run_clean([{"text": text}], _drop_cfg(max_char_repeat_ratio=0.5))
    assert out == []
    assert stats.dropped["char_repeat"] == 1


def test_clean_text_kept_and_counts():
    out, stats = run_clean([{"text": "A perfectly ordinary English sentence about science."}], cfg())
    assert len(out) == 1
    assert stats.records_out == 1
    assert out[0]["meta"]["lang"] == "en"
