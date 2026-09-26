
"""Tests for the ``quality`` op heuristics, SFT checks and coverage_report."""
from __future__ import annotations

from foundationskills.skills.data_engine.ops.base import OPS, OpStats
from foundationskills.skills.data_engine.ops import quality as quality_mod
from foundationskills.skills.data_engine.ops.quality import coverage_report


def run_quality(records, cfg):
    stats = OpStats(name="quality")
    out = list(OPS["quality"](iter(records), cfg, stats))
    return out, stats


GOOD = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is a normal piece of prose about data and its quality. "
    "Another line of text ends with a proper period."
)


def _failed(rec):
    return rec["meta"]["quality"]["failed"]


def test_gopher_good_text_scores_perfectly():
    out, _ = run_quality([{"text": GOOD}], {"heuristics": "gopher"})
    assert _failed(out[0]) == []
    assert out[0]["meta"]["quality"]["score"] == 1.0


def test_gopher_mean_word_length():
    out, _ = run_quality([{"text": "anticonstitutionnellement disproportionnément inintellectualisable"}], {"heuristics": "gopher"})
    assert "gopher:mean_word_length" in _failed(out[0])


def test_gopher_symbol_ratio():
    out, _ = run_quality([{"text": "data #### ++++ $$$$ #### ++++ $$$$ ####"}], {"heuristics": "gopher"})
    assert "gopher:symbol_ratio" in _failed(out[0])


def test_gopher_alpha_words():
    out, _ = run_quality([{"text": "1234 5678 9012 3456 7890 1111 one two"}], {"heuristics": "gopher"})
    assert "gopher:alpha_words" in _failed(out[0])


def test_gopher_stopwords():
    out, _ = run_quality([{"text": "Quixotic zephyrs jinx fjords glyphs vex waltz nymph quick."}], {"heuristics": "gopher"})
    assert "gopher:stopwords" in _failed(out[0])


def test_gopher_bullet_and_ellipsis_ratios():
    bullets = "\n".join("- item line number one" for _ in range(6))
    out, _ = run_quality([{"text": bullets}], {"heuristics": "gopher"})
    assert "gopher:bullet_ratio" in _failed(out[0])
    ellip = "\n".join("a sentence that trails off..." for _ in range(4))
    out2, _ = run_quality([{"text": ellip}], {"heuristics": "gopher"})
    assert "gopher:ellipsis_ratio" in _failed(out2[0])


def test_gopher_cjk_skips_word_rules():
    out, stats = run_quality(
        [{"text": "这是一段质量不错的中文文本，用来验证基于单词的规则会被跳过。它应该得到满分。"}],
        {"heuristics": "gopher"},
    )
    assert stats.extra["cjk_skipped_word_rules"] is True
    assert not any(tag.startswith("gopher:mean") or tag.startswith("gopher:stop") for tag in _failed(out[0]))


def test_c4_lorem_and_javascript_and_braces():
    out, _ = run_quality([{"text": "Lorem ipsum dolor sit amet, consectetur adipiscing."}], {"heuristics": "c4"})
    assert "c4:lorem_ipsum" in _failed(out[0])
    out2, _ = run_quality([{"text": "Please enable JavaScript to view this page."}], {"heuristics": "c4"})
    assert "c4:javascript_boilerplate" in _failed(out2[0])
    out3, _ = run_quality([{"text": "The config uses {value} interpolation here."}], {"heuristics": "c4"})
    assert "c4:curly_braces" in _failed(out3[0])
    code = {"text": "The config uses {value} interpolation here.", "meta": {"domain": "code"}}
    out4, _ = run_quality([code], {"heuristics": "c4"})
    assert "c4:curly_braces" not in _failed(out4[0])


def test_fineweb_short_and_duplicate_lines():
    short = "ab\ncd\nef\ngh\nij"
    out, _ = run_quality([{"text": short}], {"heuristics": "fineweb"})
    assert "fineweb:short_line_ratio" in _failed(out[0])
    dup = "the one and only repeated line\n" * 5
    out2, _ = run_quality([{"text": dup.strip()}], {"heuristics": "fineweb"})
    assert "fineweb:dup_lines" in _failed(out2[0])


SFT = {"heuristics": "none", "sft_checks": True}


def test_sft_empty_answer():
    out, _ = run_quality([{"prompt": "What is 2+2?", "answer": "   "}], SFT)
    assert "sft:empty_answer" in _failed(out[0])


def test_sft_echo():
    out, _ = run_quality([{"prompt": "Repeat after me?", "answer": "Repeat after me?"}], SFT)
    assert "sft:echo" in _failed(out[0])


def test_sft_truncated_answer():
    out, _ = run_quality([{"prompt": "write prose", "answer": "word " * 100}], SFT)
    assert "sft:truncated" in _failed(out[0])
    good = {"prompt": "write prose", "answer": ("word " * 100).strip() + "."}
    out2, _ = run_quality([good], SFT)
    assert "sft:truncated" not in _failed(out2[0])


def test_sft_refusal_boilerplate():
    out, _ = run_quality([{"prompt": "help", "answer": "As an AI language model, I cannot help with that."}], SFT)
    assert "sft:refusal_boilerplate" in _failed(out[0])


def test_sft_short_answer_relative_to_question():
    question = "Explain in careful detail " * 10
    out, _ = run_quality([{"prompt": question, "answer": "yes"}], SFT)
    assert "sft:short_answer" in _failed(out[0])


def test_sft_missing_assistant():
    rec = {"messages": [{"role": "user", "content": "hello there"}]}
    out, _ = run_quality([rec], SFT)
    assert "sft:missing_assistant" in _failed(out[0])


def test_sft_bad_role_order():
    rec = {"messages": [{"role": "assistant", "content": "I speak first."}, {"role": "user", "content": "then me"}]}
    out, _ = run_quality([rec], SFT)
    assert "sft:bad_role_order" in _failed(out[0])
    good = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "an answer."}]}
    out2, _ = run_quality([good], SFT)
    assert "sft:bad_role_order" not in _failed(out2[0])


def test_min_quality_score_drops_record():
    recs = [{"id": "1", "text": GOOD}, {"id": "2", "text": "data #### ++++ $$$$ #### ++++ $$$$ ####"}]
    out, stats = run_quality(recs, {"heuristics": "gopher", "min_quality_score": 0.99})
    assert [r["id"] for r in out] == ["1"]
    assert stats.dropped["low_quality"] == 1


def test_coverage_report_basic():
    records = [
        {"text": "apples and oranges are tasty fruit", "meta": {"domain": "food", "lang": "en"}},
        {"text": "oranges and lemons make citrus juice", "meta": {"domain": "food", "lang": "en"}},
        {"text": "神经网络训练需要大量数据和计算资源。", "meta": {"domain": "tech", "lang": "zh"}},
    ]
    report = coverage_report(records)
    assert report["records"] == 3
    assert report["domains"] == {"food": 2, "tech": 1}
    assert report["langs"] == {"en": 2, "zh": 1}
    assert len(report["top_3grams"]) <= 50
    assert any(gram == "and oranges are" for gram, _ in report["top_3grams"])
    assert 0.0 < report["distinct_1"] <= 1.0
    assert 0.0 < report["distinct_2"] <= 1.0
    assert report["clusters"]["count"] >= 2
    assert report["clusters"]["size_entropy"] > 0.0
