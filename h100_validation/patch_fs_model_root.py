#!/usr/bin/env python3
"""Build stage for the model-root plane: apply the shallowest-depth ambiguity fix.

The generator resolved ambiguity by counting every config in the bounded subtree. That
rule is measurably wrong on stock upstream checkpoints that ship a nested variant
directory beside one unambiguous root config, and measurably right on vendor family
directories that have no root config and several one level down. This stage patches the
resolver and its suite to separate those two measurements instead of widening either
into a guess.

Both edits are anchored literals with sentinels rather than regenerated files: the fixed
artifacts are already generated elsewhere, and rewriting them here would make this stage
a second source of truth. Anchors must occur exactly once, because a duplicated anchor
means the input drifted and a blind replace would patch the wrong copy silently.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

SENTINEL_A = "def _depth_of(config_path: str, root: str) -> int:"
SENTINEL_B = "def test_t5a_shallowest_wins_over_nested_variant"

A_HELPER_ANCHOR = "def _find_configs(root: str,"
A_HELPER_REPLACEMENT = '''def _depth_of(config_path: str, root: str) -> int:
    """Directory depth of a config below root; 0 means the config sits at the root."""
    rel = os.path.relpath(os.path.dirname(config_path), root)
    return 0 if rel == os.curdir else rel.count(os.sep) + 1


def _find_configs(root: str,'''

A_COUNT_ANCHOR = '''    candidates = _find_configs(root, config_name, max_depth)
    n = len(candidates)
    if n == 0:
        raise ModelRootError(
            f"found 0 {config_name} under {root} at depth <={max_depth}; "
            "a model root with no config is UNMEASURED, not empty")
    if n > 1:
        shown = ", ".join(repr(os.path.relpath(c, root)) for c in candidates[:5])
        raise ModelRootError(
            f"found {n} {config_name} candidates under {root} at depth "
            f"<={max_depth} (showing up to 5: {shown}); choosing among them "
            "would be a guess, not a measurement")

    config_path = candidates[0]'''

A_COUNT_REPLACEMENT = '''    candidates = _find_configs(root, config_name, max_depth)
    found = len(candidates)
    if found == 0:
        raise ModelRootError(
            f"found 0 {config_name} under {root} at depth <={max_depth}; "
            "a model root with no config is UNMEASURED, not empty")

    # Ambiguity is a property of the SHALLOWEST populated depth, not of the whole
    # subtree. MEASURED on a real estate: stock upstream checkpoints ship a nested
    # variant directory beside an unambiguous root config -- gpt-oss-20b has
    # original/config.json and sentence-transformers models have 1_Pooling/config.json,
    # both d0=1 d1=1. Counting flat calls those ambiguous and refuses a perfectly
    # well-formed model root. Genuine ambiguity measures differently: a vendor family
    # directory has NO root config and several one level down (d0=0 d1=2, d0=0 d1=7).
    # Both denominators are reported, so narrowing to the shallowest depth stays
    # visible rather than quietly discarding the rest of the subtree.
    by_depth = [(_depth_of(c, root), c) for c in candidates]
    shallowest = min(depth for depth, _ in by_depth)
    contenders = [c for depth, c in by_depth if depth == shallowest]
    n = len(contenders)
    if n > 1:
        shown = ", ".join(repr(os.path.relpath(c, root)) for c in contenders[:5])
        raise ModelRootError(
            f"found {n} {config_name} candidates at depth {shallowest} under {root} "
            f"({found} in the subtree at depth <={max_depth}) "
            f"(showing up to 5: {shown}); choosing among them "
            "would be a guess, not a measurement")

    config_path = contenders[0]'''

B_OLD_TEST = '''def test_t5_ambiguity_refuses_with_count(tmp_path: Path) -> None:
    root = tmp_path / "med-gemma"
    _touch(root / "config.json")
    _touch(root / "subdir" / "config.json")
    with pytest.raises(ModelRootError) as excinfo:
        resolve_model_root(root)
    msg = str(excinfo.value)
    assert "2" in msg
    assert "guess" in msg'''

B_NEW_TESTS = '''def test_t5a_shallowest_wins_over_nested_variant(tmp_path: Path) -> None:
    # MEASURED on the estate: stock upstream checkpoints ship a nested variant dir
    # beside an unambiguous root config (gpt-oss-20b `original/`, sentence-transformers
    # `1_Pooling/`). Refusing these is a false refusal on a legitimate checkpoint.
    root = tmp_path / "model"
    _touch(root / "config.json")
    _touch(root / "original" / "config.json")
    mr = resolve_model_root(root)
    assert mr.layout == LAYOUT_SELF_CONTAINED
    assert mr.config_dir == _real(root)
    assert mr.config_candidates == 1


def test_t5b_ambiguity_at_shallowest_depth_refuses_with_both_denominators(
        tmp_path: Path) -> None:
    # Genuine ambiguity: NO config at the root, several one level down. Both
    # denominators must appear -- the count at the chosen depth AND the subtree total --
    # so that narrowing to the shallowest depth stays visible rather than silently
    # discarding the rest of the subtree.
    root = tmp_path / "family"
    _touch(root / "a" / "config.json")
    _touch(root / "b" / "config.json")
    _touch(root / "a" / "deeper" / "config.json")
    with pytest.raises(ModelRootError) as excinfo:
        resolve_model_root(root)
    msg = str(excinfo.value)
    # Assert the full phrase, never a bare number: "2" also occurs in "depth <=2".
    assert "found 2 config.json candidates at depth 1" in msg
    assert "(3 in the subtree at depth <=2)" in msg
    assert "guess" in msg'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    module = root / "h100" / "gen" / "fs_model_root.py"
    suite = root / "h100" / "gen" / "test_fs_model_root.py"

    module0 = module.read_bytes()
    suite0 = suite.read_bytes()
    a0 = module0.decode("utf-8")
    b0 = suite0.decode("utf-8")

    have_a = SENTINEL_A in a0
    have_b = SENTINEL_B in b0
    if have_a and have_b:
        print("already applied")
        return 0

    gates: list[tuple[str, bool, str]] = []
    a1 = a0
    b1 = b0

    if not have_a:
        helper_hits = a0.count(A_HELPER_ANCHOR)
        count_hits = a0.count(A_COUNT_ANCHOR)
        gates.append(
            ("A-C1 each module anchor occurs exactly once",
             helper_hits == 1 and count_hits == 1,
             f"helper={helper_hits} count-block={count_hits}")
        )
        depth_gate_hits = a0.count("if depth > max_depth:")
        # The partition is only meaningful while discovery is depth-bounded; if that
        # bound is ever removed, "shallowest inside max_depth" stops being a measurement
        # and becomes whatever the walker happened to reach first.
        gates.append(
            ("A-C3 config discovery is still bounded by max_depth",
             depth_gate_hits == 1, f"{depth_gate_hits} bound(s)")
        )
        if gates[-2][1] and gates[-1][1]:
            a1 = a0.replace(A_HELPER_ANCHOR, A_HELPER_REPLACEMENT, 1)
            a1 = a1.replace(A_COUNT_ANCHOR, A_COUNT_REPLACEMENT, 1)

        # The old unpartitioned read is the exact false-refusal mechanism seen on
        # gpt-oss-20b and sentence-transformers layouts; leaving any candidates[0]
        # behind means a second resolver path can still count flat.
        gates.append(
            ("A-C4 old unpartitioned candidates[0] read is gone",
             "candidates[0]" not in a1, f"{a1.count('candidates[0]')} remaining")
        )
        gates.append(
            ("A-C5 _depth_of defined once and shallowest partition present",
             a1.count("def _depth_of(") == 1 and "shallowest" in a1,
             f"defs={a1.count('def _depth_of(')} shallowest={'shallowest' in a1}")
        )

    if not have_b:
        old_hits = b0.count(B_OLD_TEST)
        gates.append(("B-C1 old t5 function occurs exactly once", old_hits == 1, f"{old_hits} hit(s)"))
        # t5a asserts the L1 layout by name; if the import drifted away the new test
        # would fail at collection time with NameError, which looks like a resolver
        # regression rather than the stale suite it is.
        gates.append(
            ("B-C2 suite already imports LAYOUT_SELF_CONTAINED",
             "LAYOUT_SELF_CONTAINED" in b0, "")
        )
        if gates[-2][1] and gates[-1][1]:
            b1 = b0.replace(B_OLD_TEST, B_NEW_TESTS, 1)

        gates.append(
            ("B-C3 test function count increased by exactly one",
             b1.count("def test_") - b0.count("def test_") == 1,
             f"{b0.count('def test_')} -> {b1.count('def test_')}")
        )
        gates.append(
            ("B-C4 old t5 refusal test is gone",
             "def test_t5_ambiguity_refuses_with_count" not in b1, "")
        )

    a2 = a1 if SENTINEL_A in a1 else a1
    b2 = b1 if SENTINEL_B in b1 else b1
    gates.append(
        ("S1 re-application is a byte-exact no-op",
         a2.encode("utf-8") == a1.encode("utf-8")
         and b2.encode("utf-8") == b1.encode("utf-8"), "")
    )

    if not all(ok for _, ok, _ in gates):
        return _fail(gates)

    _atomic_write(module, a1.encode("utf-8"))
    _atomic_write(suite, b1.encode("utf-8"))

    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(module), str(suite)],
        capture_output=True, text=True,
    )
    gates.append(("S2 py_compile clean on module and suite", proc.returncode == 0,
                  proc.stderr.strip()[:120]))
    if proc.returncode != 0:
        _atomic_write(module, module0)
        _atomic_write(suite, suite0)
        _report(gates)
        print("restored the original bytes; an unverified patch is not left in place")
        return 4

    _report(gates)
    print(f"patched {module} and {suite}; py_compile: clean")
    return 0


def _atomic_write(target: pathlib.Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, target)


def _fail(gates: list[tuple[str, bool, str]]) -> int:
    _report(gates)
    return 2


def _report(gates: list[tuple[str, bool, str]]) -> None:
    print("gate table:")
    for label, ok, note in gates:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}{(' (' + note + ')') if note else ''}")


if __name__ == "__main__":
    raise SystemExit(main())
