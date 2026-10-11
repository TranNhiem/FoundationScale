"""Unit tests for :mod:`foundationskills.vetting.lock` (trees, pins, digests, reviews)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from foundationskills.vetting.lock import (
    LOCK_SCHEMA,
    approved_review,
    check_license,
    check_pins,
    check_scripts_policy,
    check_source,
    file_sha256,
    load_lock,
    lock_entry,
    read_tree,
    review_decision,
    subset_digest,
    tree_digest,
)
from foundationskills.vetting.errors import VettingRefused

ALLOWLIST = {
    "sources": ["github.com/foundationskills"],
    "publishers": ["foundationskills"],
    "spdx": ["MIT", "Apache-2.0"],
}


def make_lock() -> dict:
    return {"schema": LOCK_SCHEMA, "allowlist": dict(ALLOWLIST), "skills": {}}


def review(kind, decision, digest, reviewer="ann", date="2024-01-01"):
    return {"kind": kind, "decision": decision, "files_sha256": digest, "reviewer": reviewer, "date": date}


def test_file_sha256_is_hex_sha256_of_the_bytes() -> None:
    assert file_sha256(b"x\n") == hashlib.sha256(b"x\n").hexdigest()


# load_lock / lock_entry ------------------------------------------------------------

def test_load_lock_refuses_missing_bad_json_and_wrong_schema(tmp_path: Path) -> None:
    with pytest.raises(VettingRefused) as exc:
        load_lock(tmp_path / "skills.lock")
    assert "lock file not found" in str(exc.value)
    bad = tmp_path / "bad.lock"
    bad.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(VettingRefused):
        load_lock(bad)
    wrong = tmp_path / "wrong.lock"
    wrong.write_text(json.dumps({"schema": "fskills.skills_lock/2", "allowlist": ALLOWLIST, "skills": {}}))
    with pytest.raises(VettingRefused) as exc:
        load_lock(wrong)
    assert "fskills.skills_lock/2" in str(exc.value)


def test_load_lock_refuses_missing_mappings_and_reads_the_good_case(tmp_path: Path) -> None:
    cases = (
        ([1, 2], "lock file is not a mapping"),
        ({"schema": LOCK_SCHEMA, "skills": {}}, "'allowlist'"),
        ({"schema": LOCK_SCHEMA, "allowlist": ALLOWLIST}, "'skills'"),
        ({"schema": LOCK_SCHEMA, "allowlist": {"sources": []}, "skills": {}}, "missing 'publishers'"),
    )
    for document, fragment in cases:
        path = tmp_path / "case.lock"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(VettingRefused) as exc:
            load_lock(path)
        assert fragment in str(exc.value)
    good = tmp_path / "skills.lock"
    good.write_text(json.dumps(make_lock()), encoding="utf-8")
    assert load_lock(good)["schema"] == LOCK_SCHEMA


def test_lock_entry_refuses_a_skill_the_lock_does_not_pin() -> None:
    data = make_lock()
    data["skills"] = {"alpha": {"commit": "0" * 40, "files": {}}}
    assert lock_entry(data, "alpha")["commit"] == "0" * 40
    with pytest.raises(VettingRefused) as exc:
        lock_entry(data, "beta")
    assert "not pinned" in str(exc.value)


# digests ---------------------------------------------------------------------------

def test_subset_digest_is_order_independent_stable_and_exact() -> None:
    files = {"a.md": b"a\n", "b.md": b"b\n", "c.md": b"c\nd"}
    forward = subset_digest(files, ["a.md", "b.md"])
    assert forward == subset_digest(files, ["b.md", "a.md"])
    assert forward == subset_digest(files, ["b.md", "a.md", "b.md"])
    assert forward == subset_digest(files, iter(["a.md", "b.md"]))
    expected = file_sha256(
        "".join(f"{p}\0{file_sha256(files[p])}\n" for p in ("a.md", "b.md")).encode("utf-8")
    )
    assert forward == expected
    assert forward != subset_digest(files, ["a.md"])
    assert subset_digest(files, []) == file_sha256(b"")
    assert tree_digest(files) == subset_digest(files, ["a.md", "b.md", "c.md"])
    with pytest.raises(VettingRefused) as exc:
        subset_digest(files, ["gone.md"])
    assert "gone.md" in str(exc.value)


# reviews ---------------------------------------------------------------------------

def test_review_decision_lets_reject_win_and_ignores_a_stale_digest() -> None:
    files = {"a.md": b"a\n"}
    current = subset_digest(files, ["a.md"])
    stale = subset_digest({"a.md": b"a\n!\n"}, ["a.md"])
    entry = {"reviews": [
        review("network", "approve", current, "ann", "2024-01-01"),
        review("network", "reject", current, "bo", "2024-01-02"),
        review("network", "approve", stale, "cyd", "2024-03-03"),
    ]}
    decision = review_decision(entry, "network", current)
    assert decision["decision"] == "reject" and decision["reviewer"] == "bo"
    assert approved_review(entry, "network", current)["reviewer"] == "ann"
    assert review_decision(entry, "network", stale)["reviewer"] == "cyd"
    assert review_decision(entry, "network", "0" * 64) is None
    assert approved_review({"reviews": []}, "network", current) is None


def test_review_selection_is_latest_with_lock_order_as_tiebreak() -> None:
    digest = subset_digest({"a.md": b"a"}, ["a.md"])
    entry = {"reviews": [
        review("scripts", "approve", digest, "ann", "2024-01-01"),
        review("scripts", "approve", digest, "bo", "2024-01-02"),
        review("scripts", "approve", digest, "cyd", "2024-01-02"),
    ]}
    assert approved_review(entry, "scripts", digest)["reviewer"] == "cyd"


# VG-01 source allowlist ------------------------------------------------------------

def test_check_source_allowlist_pass_and_refuse() -> None:
    data = make_lock()
    ok = check_source(data, {"source": "github.com/foundationskills", "publisher": "foundationskills"})
    assert (ok.id, ok.status, ok.hits) == ("VG-01", "PASS", ())
    assert "both on the allowlist" in ok.note
    for entry, fragment in (
        ({"source": "example.org/evil", "publisher": "foundationskills"},
         "source example.org/evil is not on allowlist.sources"),
        ({"source": "github.com/foundationskills", "publisher": "impostor"},
         "publisher impostor is not on allowlist.publishers"),
        ({}, "source <none> is not on allowlist.sources"),
    ):
        result = check_source(data, entry)
        assert (result.status, result.hits) == ("REFUSED", ())
        assert fragment in result.note


# VG-02 pins ------------------------------------------------------------------------

def test_check_pins_passes_when_commit_and_sha256_all_match() -> None:
    files = {"a.md": b"a\n", "b.md": b"b\n"}
    entry = {"commit": "1" * 40, "files": {p: file_sha256(files[p]) for p in files}}
    result = check_pins(entry, files)
    assert (result.id, result.status, result.hits) == ("VG-02", "PASS", ())
    assert result.note == "commit " + "1" * 40 + " pinned; 2 file sha256 match the tree exactly"


def test_check_pins_notes_name_the_mismatched_missing_and_extra_paths() -> None:
    files = {"a.md": b"a\n", "b.md": b"edited\n", "c.md": b"c\n"}
    entry = {
        "commit": "0" * 40,
        "files": {"a.md": file_sha256(b"a\n"), "b.md": file_sha256(b"b\n"), "d.md": file_sha256(b"d\n")},
    }
    result = check_pins(entry, files)
    assert (result.status, result.hits) == ("REFUSED", ())
    assert "3 problem(s)" in result.note
    assert "sha256 mismatch: b.md" in result.note
    assert "missing path (pinned, not in the tree): d.md" in result.note
    assert "extra path (in the tree, not pinned): c.md" in result.note


def test_check_pins_refuses_a_commit_that_is_not_40_lowercase_hex() -> None:
    files = {"a.md": b"a\n"}
    for commit in ("", "HEAD", "g" * 40, "0" * 39, "0" * 41, "A" * 40):
        result = check_pins({"commit": commit, "files": {"a.md": file_sha256(b"a\n")}}, files)
        assert (result.status, result.hits) == ("REFUSED", ()), commit
        assert "is not 40 lowercase hex" in result.note, commit
    missing = check_pins({"commit": "0" * 40, "files": None}, files)
    assert "entry.files is missing (no per-file sha256 pins)" in missing.note


# VG-03 licence ---------------------------------------------------------------------

def test_check_license_matrix() -> None:
    cases = (
        ({}, {}, "UNMEASURED",
         "no licence in the lock entry and none in the SKILL.md frontmatter"),
        ({"license": "MIT"}, {"license": "Apache-2.0"}, "REFUSED",
         "licence disagreement: lock pins 'MIT', frontmatter claims 'Apache-2.0'"),
        ({"license": "EPL-2.0"}, {"license": "EPL-2.0"}, "REFUSED", "EPL-2.0: weak copyleft excluded"),
        ({"license": "MPL-2.0"}, {"license": "MPL-2.0"}, "REFUSED", "licence MPL-2.0 is not on allowlist.spdx"),
        ({}, {"license": "Zlib"}, "REFUSED", "licence Zlib is not on allowlist.spdx"),
        ({"license": "MIT"}, {"license": "MIT"}, "PASS", "licence MIT is on the allowlist (lock and frontmatter)"),
        ({"license": "MIT"}, {}, "PASS", "licence MIT is on the allowlist (lock)"),
        ({}, {"license": "MIT"}, "PASS", "licence MIT is on the allowlist (frontmatter)"),
    )
    for entry, front, status, note in cases:
        result = check_license(make_lock(), entry, front)
        assert (result.id, result.status, result.hits) == ("VG-03", status, ()), note
        assert result.note == note


# VG-10 scripts policy --------------------------------------------------------------

def test_scripts_policy_without_scripts_passes() -> None:
    result = check_scripts_policy({}, {"SKILL.md": b"---\n---\n"})
    assert (result.id, result.status, result.hits) == ("VG-10", "PASS", ())
    assert result.note == "no scripts/ files in the tree"


def test_scripts_policy_unreviewed_scripts_are_refused() -> None:
    files = {"scripts/run.sh": b"echo hi\n", "SKILL.md": b"---\n---\n"}
    digest = subset_digest(files, ["scripts/run.sh"])
    result = check_scripts_policy({"reviews": []}, files)
    assert (result.status, result.hits) == ("REFUSED", ())
    assert f"no scripts review matches the current 1 scripts/ file(s) (files_sha256 {digest[:12]}...)" in result.note
    assert "scripts/run.sh" in result.note


def test_scripts_policy_approved_rejected_and_approved_then_edited() -> None:
    files = {"scripts/run.sh": b"echo hi\n", "scripts/two.sh": b"echo two\n"}
    digest = subset_digest(files, ["scripts/run.sh", "scripts/two.sh"])
    approved = check_scripts_policy({"reviews": [review("scripts", "approve", digest)]}, files)
    assert (approved.status, approved.hits) == ("PASS", ())
    assert approved.note == (
        f"scripts review by ann on 2024-01-01 covers all 2 scripts/ file(s) "
        f"(files_sha256 {digest[:12]}...) and is approved; never executed by the gate"
    )
    rejected = check_scripts_policy({"reviews": [review("scripts", "reject", digest, "bo", "2024-01-02")]}, files)
    assert (rejected.status, rejected.hits) == ("REFUSED", ())
    assert f"by bo on 2024-01-02 rejected 2 scripts/ file(s) (files_sha256 {digest[:12]}...)" in rejected.note
    edited = dict(files)
    edited["scripts/run.sh"] = b"echo edited\n"
    after = check_scripts_policy({"reviews": [review("scripts", "approve", digest)]}, edited)
    assert (after.status, after.hits) == ("REFUSED", ())
    assert "no scripts review matches the current 2 scripts/ file(s)" in after.note


# read_tree -------------------------------------------------------------------------

def test_read_tree_reads_regular_files_and_refuses_a_symlink_escaping_the_root(tmp_path: Path) -> None:
    root = tmp_path / "skill"
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_bytes(b"---\n---\n")
    (root / "scripts" / "run.sh").write_bytes(b"echo hi\n")
    (root / "alias.md").symlink_to(root / "SKILL.md")
    assert read_tree(root) == {
        "SKILL.md": b"---\n---\n",
        "alias.md": b"---\n---\n",
        "scripts/run.sh": b"echo hi\n",
    }
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret\n")
    (root / "escape.md").symlink_to(outside)
    with pytest.raises(VettingRefused) as exc:
        read_tree(root)
    assert "symlink escapes the skill root: escape.md" in str(exc.value)
    with pytest.raises(VettingRefused):
        read_tree(tmp_path / "missing")
