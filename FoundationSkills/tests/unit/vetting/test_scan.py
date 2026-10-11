"""Unit tests for :mod:`foundationskills.vetting.scan` (VG-04..VG-09 and VG-14)."""
from __future__ import annotations

import pytest

from foundationskills.vetting import lock, scan
from foundationskills.vetting.errors import VettingRefused


def md(text: str, name: str = "README.md") -> dict[str, bytes]:
    return {name: text.encode("utf-8")}


def reviewed(kind, decision, files, paths, reviewer="ann", date="2024-01-01"):
    return {
        "kind": kind,
        "decision": decision,
        "reviewer": reviewer,
        "date": date,
        "files_sha256": lock.subset_digest(files, list(paths)),
    }


def only_hit(result):
    assert len(result.hits) == 1, result.to_dict()
    return result.hits[0]


# VG-04 network primitives ----------------------------------------------------------

def test_network_hit_names_path_pattern_and_1_based_line():
    result = scan.check_network({}, md("intro\ncurl -s https://example.com > out\n"))
    assert (result.id, result.title, result.status) == ("VG-04", scan.CHECK_TITLES["VG-04"], "RED")
    hit = only_hit(result)
    assert (hit.path, hit.line, hit.pattern) == ("README.md", 2, r"\bcurl\b")
    assert hit.excerpt == "curl -s https://example.com > out"
    assert result.to_dict()["hits"] == [
        {"path": "README.md", "line": 2, "pattern": r"\bcurl\b", "excerpt": "curl -s https://example.com > out"}
    ]
    assert result.note == "1 network primitive hit(s)"


def test_network_near_miss_prose_and_markdown_link_never_fire():
    for line in (
        "This helper fetches the data from the local cache.",
        "the request is retried after a timeout.",
        "ncurses rendering draws the table.",
        "see [the guide](https://example.com/guide) for more",
    ):
        result = scan.check_network({}, md(line))
        assert (result.status, result.hits) == ("PASS", ()), line
        assert result.note == "no network primitive found"


def test_network_approved_review_excuses_only_the_reviewed_path():
    files = {"a.sh": b"curl a\n", "b.sh": b"wget b\n"}
    entry = {"reviews": [reviewed("network", "approve", files, ["a.sh"])]}
    result = scan.check_network(entry, files)
    assert result.status == "RED"
    assert [hit.path for hit in result.hits] == ["b.sh"]
    assert result.note == "approved network review excuses a.sh; 1 network primitive hit(s)"


def test_network_reject_review_refuses_the_check():
    files = {"a.sh": b"curl a\n"}
    entry = {"reviews": [reviewed("network", "reject", files, ["a.sh"], "bo", "2024-01-02")]}
    result = scan.check_network(entry, files)
    assert (result.status, [hit.path for hit in result.hits]) == ("REFUSED", ["a.sh"])
    assert result.note == "network review refused a.sh; 1 network primitive hit(s)"


# VG-05 credential paths and env tokens ----------------------------------------------

def test_credential_hit_names_path_pattern_and_1_based_line():
    result = scan.check_credentials(md("one\ntwo\nread id_rsa off the disk\n", name="docs/env.md"))
    assert (result.id, result.status) == ("VG-05", "RED")
    hit = only_hit(result)
    assert (hit.path, hit.line, hit.pattern) == ("docs/env.md", 3, r"\bid_rsa\b")
    assert hit.excerpt == "read id_rsa off the disk"
    assert result.note == "1 credential path or env token hit(s)"


def test_credential_near_miss_documented_env_prose_never_fires():
    result = scan.check_credentials(md("environment variables are documented in the guide\n"))
    assert (result.status, result.hits) == ("PASS", ())
    assert result.note == "no credential path or env token found"


# VG-06 dynamic execution -----------------------------------------------------------

def test_dynamic_exec_hit_names_path_pattern_and_1_based_line():
    result = scan.check_dynamic_exec(md("x = 1\neval(payload)\n", name="run.py"))
    assert (result.id, result.status) == ("VG-06", "RED")
    hit = only_hit(result)
    assert (hit.path, hit.line, hit.pattern) == ("run.py", 2, r"\beval\s*\(")
    assert hit.excerpt == "eval(payload)"
    assert result.note == "1 dynamic execution hit(s)"


def test_dynamic_exec_near_miss_evaluation_and_executive_never_fire():
    result = scan.check_dynamic_exec(md("An executive summary of the evaluation; nothing is run.\n"))
    assert (result.status, result.hits, result.note) == ("PASS", (), "no dynamic execution found")


# VG-07 hidden HTML comments --------------------------------------------------------

def test_hidden_comment_hit_names_path_pattern_line_and_excerpt():
    result = scan.check_html_comments(md("# Title\n<!-- hidden plan -->\nbody\n"))
    assert (result.id, result.status) == ("VG-07", "RED")
    hit = only_hit(result)
    assert (hit.path, hit.line, hit.pattern) == ("README.md", 2, "<!--")
    assert hit.excerpt == "<!-- hidden plan -->"


def test_hidden_comment_near_miss_markdown_link_and_python_never_fire():
    link = scan.check_html_comments(md("see [the guide](https://example.com/guide) for more\n"))
    assert (link.status, link.hits, link.note) == ("PASS", (), "no hidden HTML comment found")
    code = scan.check_hidden if False else scan.check_html_comments({"tool.py": b"markup = '<!--'\n"})
    assert (code.status, code.hits) == ("PASS", ())


# VG-08 invisible Unicode -----------------------------------------------------------

def test_invisible_unicode_hit_names_path_pattern_and_1_based_line():
    result = scan.check_invisible_unicode(md("plain\nx\u202ey\n"))
    assert (result.id, result.status) == ("VG-08", "RED")
    hit = only_hit(result)
    assert (hit.path, hit.line) == ("README.md", 2)
    assert hit.pattern == "\\u202e"
    assert hit.excerpt == "x\\u202ey"


def test_invisible_tag_character_escapes_with_uppercase_u():
    hit = only_hit(scan.check_invisible_unicode(md("a\U000e0043b\n")))
    assert hit.pattern == "\\U000e0043"
    assert hit.excerpt == "a\\U000e0043b"


def test_invisible_unicode_near_miss_ascii_only_never_fires():
    result = scan.check_invisible_unicode(md("plain ASCII only: see [a](b) for more\n"))
    assert (result.status, result.hits, result.note) == ("PASS", (), "no invisible character found")


# VG-09 allowed-tools ---------------------------------------------------------------

def test_allowed_tools_matrix():
    cases = (
        ({}, {"allowed_tools_need": ["Read"]}, "PASS", [], "no allowed-tools declaration to vet"),
        ({"allowed-tools": ""}, {"allowed_tools_need": []}, "PASS", [], "allowed-tools declares no tool at all"),
        ({"allowed-tools": "Read Grep"}, {"allowed_tools_need": ["Read", "Grep"]}, "PASS", [],
         "every token is in the declared need"),
        ({"allowed-tools": "Read, Write"}, {"allowed_tools_need": ["Read"]}, "RED", ["Write"],
         "1 token(s) outside allowed_tools_need"),
        ({"allowed-tools": ["Bash"]}, {"allowed_tools_need": ["Bash(apply_patch)"]}, "RED", ["Bash"],
         "wildcard or unscoped tool request: Bash"),
        ({"allowed-tools": "Bash"}, {"allowed_tools_need": ["Bash"]}, "PASS", [],
         "every token is in the declared need"),
        ({"allowed-tools": "*"}, {"allowed_tools_need": ["Read", "Bash"]}, "RED", ["*"],
         "wildcard or unscoped tool request: *"),
    )
    for front, entry, status, patterns, fragment in cases:
        result = scan.check_allowed_tools(entry, front)
        assert (result.id, result.status) == ("VG-09", status), front
        assert [hit.pattern for hit in result.hits] == patterns, front
        assert fragment in result.note, front
        for hit in result.hits:
            assert (hit.path, hit.line) == ("SKILL.md", 0), front


# VG-14 .github review --------------------------------------------------------------

def test_github_refuses_when_no_review_covers_the_policy():
    files = {".github/workflow.yml": b"on: push\n", "notes.md": b"hi\n"}
    result = scan.check_github({}, files)
    assert (result.id, result.status, result.hits) == ("VG-14", "REFUSED", ())
    assert result.note == "no approved github review covers .github/workflow.yml"


def test_github_approved_review_passes_reject_review_refuses():
    files = {".github/workflow.yml": b"on: push\n", ".github/CODEOWNERS": b"* @a\n"}
    paths = sorted(files)
    entry = {"reviews": [reviewed("github", "approve", files, paths)]}
    approved = scan.check_github(entry, files)
    assert (approved.status, approved.hits) == ("PASS", ())
    assert approved.note == "approved github review covers " + ", ".join(paths)

    entry = {"reviews": [reviewed("github", "reject", files, paths, "bo", "2024-01-02")]}
    refused = scan.check_github(entry, files)
    assert (refused.status, refused.note) == ("REFUSED", "github review refused " + ", ".join(paths))


def test_github_near_miss_plain_paths_are_never_reviewed():
    result = scan.check_github({}, {".github_notes.md": b"hi\n", "docs/.github.md": b"x\n"})
    assert (result.status, result.hits, result.note) == ("PASS", (), "no .github/ policy in the tree")


# determinism, excerpts and readers -------------------------------------------------

def test_hits_are_sorted_by_path_line_pattern_deterministically():
    files = {"b.md": b"id_rsa\n\nid_rsa\n", "a.md": b"\nid_rsa\n"}
    first = scan.check_credentials(files).to_dict()["hits"]
    second = scan.check_credentials(dict(reversed(list(files.items())))).to_dict()["hits"]
    assert first == second
    assert [(h["path"], h["line"], h["pattern"]) for h in first] == [
        ("a.md", 2, r"\bid_rsa\b"),
        ("b.md", 1, r"\bid_rsa\b"),
        ("b.md", 3, r"\bid_rsa\b"),
    ]


def test_to_dict_sorts_arbitrary_hits_by_path_line_pattern():
    result = scan.CheckResult(
        "VG-05", scan.CHECK_TITLES["VG-05"], "RED",
        (scan.Hit("b.md", 1, "p", "x"), scan.Hit("a.md", 9, "p", "x"),
         scan.Hit("a.md", 2, "q", "x"), scan.Hit("a.md", 2, "p", "y")),
        "",
    )
    assert [(h["path"], h["line"], h["pattern"]) for h in result.to_dict()["hits"]] == [
        ("a.md", 2, "p"), ("a.md", 2, "q"), ("a.md", 9, "p"), ("b.md", 1, "p")
    ]


def test_excerpt_is_capped_at_120_and_bad_status_is_refused():
    assert scan.Hit("a.md", 1, "p", "x" * 200).excerpt == "x" * 117 + "..."
    assert len(scan.Hit("a.md", 1, "p", "x" * 200).excerpt) == 120
    with pytest.raises(VettingRefused):
        scan.CheckResult("VG-05", "title", "DONE")


def test_text_files_refuses_unvettable_scannable_bytes_and_skips_binaries():
    with pytest.raises(VettingRefused):
        scan.text_files({"SKILL.md": b"\xff\xfe not utf-8"})
    with pytest.raises(VettingRefused):
        scan.text_files({"scripts/run.sh": b"\xff"})
    assert scan.text_files({"logo.png": b"\xff\x00", "a.md": b"ok\n"}) == {"a.md": "ok\n"}


def test_parse_frontmatter_ok_and_refusals():
    assert scan.parse_frontmatter(b"---\nlicense: MIT\n---\nbody\n") == {"license": "MIT"}
    for bad in (
        b"no frontmatter at all\n",
        b"---\nlicense: MIT\n",
        b"---\n[a, b\n---\n",
        b"---\n- one\n- two\n---\n",
    ):
        with pytest.raises(VettingRefused):
            scan.parse_frontmatter(bad)
