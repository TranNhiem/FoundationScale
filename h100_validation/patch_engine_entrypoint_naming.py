#!/usr/bin/env python3
"""#182 v2: the launch plane's operator-facing text names programs the build does not ship.

MEASURED DEFECT. h100/PUBLISH_SET.txt states, in its own words, that
h100/upstream/tools__fs_train.py has "zero consumers in the published build".
Yet sites across three SHIPPED artifacts tell the operator that
tools/fs_train.py -- or one estate's recipe script -- is the thing reading
their knobs:

    h100/gen/launch_fs_h100.fixed.sh            2 sites (one comment, one printf)
    h100/gen/fs_container_backend.bound.sh      2 sites (allowlist comment + its continuation)
    h100/gen/fs_container_backend.spliced.sh    3 sites (the same two, upstream of .bound.sh,
                                                  plus the fs_launch_python helper comment)

No build stage produces that text -- it survives from the estate's base
launcher. The measured truth is that the in-container program is whatever
FS_ENGINE_LAUNCH_CMD names: the launcher resolves it at
LAUNCH_CMD="${FS_ENGINE_LAUNCH_CMD:-}" and refuses with exit 96 when it is
unset. The framework deliberately does not know which program that is -- that
is the model-agnostic seam. Naming one estate's script there is both false
and a re-introduction of the coupling the seam removes. This is the same
class as #194 (text that states a fact the build does not hold), and worse
here, because the reader is an operator debugging a probe that did not stop.

THE FIX. Seven counted anchor replacements across three files and nothing
else: the probe-phase comment and the PROBE denominator printf in the
launcher; the FS_ITERATION_BUDGET allowlist comment AND its continuation line
in BOTH backend files (.spliced.sh is the upstream of .bound.sh and both
ship; editing only the derived one leaves the defect in a published file);
and the fs_launch_python helper comment in .spliced.sh. v1 rewrote the
allowlist anchor and left its continuation line ("and fs_train.py runs
INSIDE the container") -- a dangling clause -- and then declared an
EXCEPTION for its own residue, which launders the finding. The continuation
is now repaired, not excepted: it says the engine entrypoint runs inside the
container and names no script, keeping the two-line shape and the column
alignment of the surrounding allowlist block. The helper comment named
run_recipe.py where the whole point of the helper is that it composes the
interpreter prefix in front of whatever FS_ENGINE_LAUNCH_CMD names; it now
says exactly that. EXCEPTIONS is empty: a declared state, not an absence of
one, and "0 exceptions declared" is printed explicitly. No knob is renamed,
no validation arithmetic is touched, no new knob is added.

THE GENERALIZATION (why this is a stage and not a sed). A one-off replacement
leaves the CLASS open: any future edit can name another unshipped script in
operator-facing text and nothing would notice. So the stage carries a census
with a stated denominator, run as a post-condition on the produced text of
all three files. OPERATOR-FACING TEXT is defined as: any '#' comment line
(whole-line or trailing), plus the contents of any single- or double-quoted
string passed to printf, echo or fail. In that text, every token matching
[A-Za-z0-9_./-]+\\.py is collected; each token's BASENAME is checked against
an HONEST denominator: the basenames of h100/PUBLISH_SET.txt (parsed --
comment lines and blanks skipped -- never hardcoded) UNION the basenames of
the tracked files of the enclosing published repository. v1 asked the
publish set alone; it enumerates only the h100_validation/ subtree, so 6 of
its 9 tokens were FALSE REDS (e.g. tests/test_fix32_container_env_passthrough.py,
which ships at the repository root) -- the same class of defect the census
exists to catch, turned on the census. The repository is resolved in order,
and the resolution is printed: $FS_PUBLISHED_REPO_ROOT (set and not a
directory is REFUSE 96, named), else walking up from the plane directory for
a .git (in a clone the stage sits at <repo>/h100_validation/, so this
resolves with no configuration; in the build tree it does not, which is the
honest answer there). Tracked files come from git -C <root> ls-files; git
absent or erroring is UNMEASURED for the repo half, never a silent fall-back
to the publish set as the whole world. A token in neither half with the repo
half RESOLVED is RED (5) -- the denominator can answer. A token in neither
half with the repo half UNRESOLVED is UNMEASURED (95), listed by name and
file:line: the stage cannot distinguish "this file does not exist" from
"this file is provided by a part of the repository the stage cannot see" --
absence of visibility is not evidence of absence. The denominator is printed
every run: publish-set names, repo names (or "repo half UNRESOLVED"), and
the total. Zero tokens found is UNMEASURED (95), not PASS: all([]) is True,
so a census that matched nothing cannot distinguish clean text from a broken
pattern. An exception that matches nothing is RED -- a stale exception is
how the next instance of this class hides.

EXIT CONTRACT. 0 PASS (applied, or READY in dry-run) / 5 RED / 95 UNMEASURED
/ 96 REFUSE. Post-conditions are verified on the produced text BEFORE
anything is written; any failed gate, any MUST_FIRE that does not fire, or
any MUST_PASS that fails is REFUSE 96 with NOTHING written to any of the
three files -- including the ones whose edits succeeded, because partial
application across a three-file stage produces a plane no rebuild reproduces.
Ten controls: the seven from v1 (stale-token pre/post counts now 2/2/2 -> 0;
planted-unshipped census RED, now driven with a resolved repo half so the
denominator can convict; census GREEN on the real post-image; empty-text
UNMEASURED; printf arity; seam named and resolving; byte idempotence) plus
REPO_HALF_ADMITS_A_ROOT_LEVEL_FILE (a fixture repo root, not a checkout),
UNRESOLVED_REPO_IS_UNMEASURED_NOT_RED, and DANGLING_CONTINUATION_IS_CAUGHT.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
TARGETS = {
    "launcher": ROOT / "h100" / "gen" / "launch_fs_h100.fixed.sh",
    "bound": ROOT / "h100" / "gen" / "fs_container_backend.bound.sh",
    "spliced": ROOT / "h100" / "gen" / "fs_container_backend.spliced.sh",
}
PUBLISH_SET = ROOT / "h100" / "PUBLISH_SET.txt"
MARK = "fs182"
TAG = MARK + ":"
N_CONTROLS = 10

# Needles are ASSEMBLED from fragments, never one literal: this stage counts
# them in the targets, and a source that contains its own needle is inside its
# own denominator.
STALE = "tools/" + "fs_train" + ".py"
STALE_TOKEN = "fs_train" + ".py"
RECIPE = "run_recipe" + ".py"
SEAM = "FS_ENGINE_" + "LAUNCH" + "_CMD"
A_ASSIGN = "LAUNCH" + '_CMD="${FS_ENGINE_' + "LAUNCH" + '_CMD:-}"'

A_COMMENT = ("the container boundary and are read by " + STALE +
             ", which fatals in probe phase with no")
R_COMMENT = (
    "the container boundary and are read by the engine entrypoint named in " + SEAM + ".\n"
    "  # No filename appears here because the entrypoint is operator-supplied: naming one would be a\n"
    "  # claim about the operator's engine, not about this launcher. It must fatal in probe phase with no"
)

A_PRINTF = "consumed in-container via FS_ENV_ALLOWLIST by " + STALE
R_PRINTF = ("consumed in-container via FS_ENV_ALLOWLIST by the engine entrypoint "
            "named in " + SEAM)

A_ALLOW = "must cross: L4 wires " + STALE + " to read it,"
R_ALLOW = "must cross: L4 wires the engine entrypoint (" + SEAM + ") to read it,"

# The continuation line of the same allowlist comment. Rewriting line 1 and
# leaving line 2 leaves a dangling clause that still names the estate's
# script; the repair says the engine entrypoint runs inside the container and
# names no script, keeping the two-line shape and the column alignment.
A_ALLOW_CONT = "and " + STALE_TOKEN + " runs INSIDE the container"
R_ALLOW_CONT = "and the engine entrypoint runs INSIDE the container"

# The fs_launch_python helper comment in .spliced.sh: the helper composes the
# interpreter prefix that goes in front of whatever the seam names -- that is
# the whole point of the seam -- not in front of one estate's recipe script.
A_HELPER = "front of " + RECIPE + "."
R_HELPER = "front of whatever " + SEAM + " names."

EDITS = {
    "launcher": [(A_COMMENT, R_COMMENT, 1), (A_PRINTF, R_PRINTF, 1)],
    "bound": [(A_ALLOW, R_ALLOW, 1), (A_ALLOW_CONT, R_ALLOW_CONT, 1)],
    "spliced": [(A_ALLOW, R_ALLOW, 1), (A_ALLOW_CONT, R_ALLOW_CONT, 1),
                (A_HELPER, R_HELPER, 1)],
}
# Measured pre-image counts of the TOKEN fs_train.py: the launcher carries it
# in the probe comment and the printf; each backend carries it in the
# allowlist anchor and its continuation line. Post-image must be 0 in all three.
EXPECT_PRE_STALE = {"launcher": 2, "bound": 2, "spliced": 2}

# Declared census exceptions: basename -> REASON. EMPTY is a declared state,
# not an absence of one: the stale-exception rule is kept, and an entry that
# matches nothing in the census is RED -- a stale exception is how the next
# instance of this class hides.
EXCEPTIONS = {}

_PY_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+\.py")
_CMD_RE = re.compile(r"\b(?:printf|echo|fail)\b")
_QUOTED_RE = re.compile(r"'([^']*)'|\"((?:[^\"\\]|\\.)*)\"")


class _Refuse(Exception):
    pass


def _stderr(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _split_comment(line: str):
    """Split a shell line into (code, comment). A '#' starts a comment only
    outside quotes, outside ${...}, and at a word boundary (line start or
    after whitespace) -- ${VAR#pat} and foo#bar are not comments."""
    in_s = in_d = False
    depth = 0
    i = 0
    while i < len(line):
        c = line[i]
        if in_s:
            if c == "'":
                in_s = False
        elif in_d:
            if c == "\\":
                i += 1
            elif c == '"':
                in_d = False
        elif c == "'":
            in_s = True
        elif c == '"':
            in_d = True
        elif c == "$" and line[i + 1:i + 2] == "{":
            depth += 1
            i += 1
        elif c == "}" and depth:
            depth -= 1
        elif c == "#" and depth == 0 and (i == 0 or line[i - 1] in " \t"):
            return line[:i], line[i:]
        i += 1
    return line, ""


def _operator_segments(text: str):
    """Yield (lineno, segment) of OPERATOR-FACING TEXT: every '#' comment
    (whole-line or trailing), plus the contents of every single- or
    double-quoted string on any line whose code invokes printf, echo or fail."""
    for lineno, line in enumerate(text.splitlines(), 1):
        code, comment = _split_comment(line)
        if comment:
            yield lineno, comment
        if _CMD_RE.search(code):
            for m in _QUOTED_RE.finditer(code):
                yield lineno, m.group(1) if m.group(1) is not None else m.group(2)


def _shipped_basenames(path: pathlib.Path):
    """Parse the publish set -- skip '#' comment lines and blanks, take each
    remaining line's first field's basename. Never a hardcoded list."""
    shipped = set()
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        shipped.add(line.split()[0].rstrip("/").split("/")[-1])
    return shipped


def _resolve_repo_root(plane_dir: pathlib.Path, explicit):
    """Resolve the enclosing published repository, in order: the explicit
    $FS_PUBLISHED_REPO_ROOT value (set and not a directory is a REFUSE,
    named), else walk up from the plane directory looking for a .git. In a
    clone the stage sits at <repo>/h100_validation/, so the walk resolves
    with no configuration; in the build tree it does not, which is the honest
    answer there. Returns (root_or_None, how)."""
    if explicit:
        root = pathlib.Path(explicit)
        if not root.is_dir():
            raise _Refuse(f"FS_PUBLISHED_REPO_ROOT is set but is not a directory: {explicit}")
        return root.resolve(), f"$FS_PUBLISHED_REPO_ROOT={explicit}"
    for cand in (plane_dir, *plane_dir.parents):
        if (cand / ".git").exists():
            return cand, f".git found walking up from the plane directory {plane_dir}"
    return None, (f"FS_PUBLISHED_REPO_ROOT unset and no .git walking up from the "
                  f"plane directory {plane_dir}")


def _repo_tracked_basenames(root: pathlib.Path):
    """Basenames of the repository's tracked files via git ls-files. git
    absent or erroring is UNMEASURED for this half of the denominator (None)
    -- never a silent fall-back to the publish set as the whole world."""
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files"],
                           capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    return {ln.strip().rstrip("/").split("/")[-1] for ln in r.stdout.splitlines() if ln.strip()}


def _census(files, shipped, repo):
    """The .py census over [(name, text)]. shipped: publish-set basenames.
    repo: tracked-file basenames of the enclosing repository, or None when
    that half of the denominator is UNRESOLVED. Returns (verdict, report)
    with verdict GREEN / RED / UNMEASURED. Zero tokens is UNMEASURED, not
    PASS. A token in neither half is RED only when the repo half resolved --
    otherwise UNMEASURED, because absence of visibility is not evidence of
    absence."""
    tokens = []  # (name, lineno, token)
    for name, text in files:
        for lineno, seg in _operator_segments(text):
            for tok in _PY_TOKEN_RE.findall(seg):
                tokens.append((name, lineno, tok))
    by_base = {}
    for name, lineno, tok in tokens:
        by_base.setdefault(tok.split("/")[-1], []).append((name, lineno, tok))
    denom = shipped | (repo if repo is not None else set())
    hits = {b: l for b, l in by_base.items() if b in denom}
    unseen = {b: l for b, l in by_base.items() if b not in denom}
    used_exc = {b: l for b, l in unseen.items() if b in EXCEPTIONS}
    stale_exc = [b for b in EXCEPTIONS if b not in by_base]
    bad = {b: l for b, l in unseen.items() if b not in EXCEPTIONS}
    report = [
        f"tokens: {len(tokens)} python-script token(s) in operator-facing text, "
        f"{len(by_base)} distinct basename(s): {len(hits)} in the denominator, "
        f"{len(unseen)} in neither half"
    ]
    label = "unshipped" if repo is not None else "unmeasured"
    for b in sorted(unseen):
        for name, lineno, tok in unseen[b]:
            report.append(f"{label}: {name}:{lineno}: {tok}")
    for b in sorted(used_exc):
        report.append(f"exception used: {b} ({len(used_exc[b])} token(s)) -- {EXCEPTIONS[b]}")
    for b in stale_exc:
        report.append(f"stale exception: {b} matched nothing in the census -- {EXCEPTIONS[b]}")
    if not tokens:
        report.append("UNMEASURED: census matched zero tokens; all([]) is True, so a census "
                      "that matched nothing cannot distinguish clean text from a broken pattern")
        return "UNMEASURED", report
    if bad and repo is None:
        report.append(f"UNMEASURED: {len(bad)} basename(s) in neither half of the denominator "
                      "and the enclosing repository could not be resolved: the stage cannot "
                      "distinguish \"this file does not exist\" from \"this file is provided "
                      "by a part of the repository the stage cannot see\" -- absence of "
                      "visibility is not evidence of absence")
        return "UNMEASURED", report
    if bad:
        report.append(f"RED: {len(bad)} basename(s) in neither half of a fully resolved "
                      "denominator, with no declared exception: " + ", ".join(sorted(bad)))
        return "RED", report
    if stale_exc:
        report.append(f"RED: {len(stale_exc)} stale exception(s) matched nothing: "
                      + ", ".join(stale_exc))
        return "RED", report
    report.append("GREEN: every python-script basename in operator-facing text is in the "
                  "denominator or carries a declared, non-stale exception")
    return "GREEN", report


def _transform(name: str, text: str) -> str:
    """Apply this file's counted edits. Any anchor whose count is not exactly
    the stated number is a REFUSE, never a guess."""
    for anchor, _repl, want in EDITS[name]:
        got = text.count(anchor)
        if got != want:
            raise _Refuse(f"anchor count wrong in {TARGETS[name]}: anchor {anchor!r} "
                          f"expected {want} occurrence(s), found {got}")
    for anchor, repl, _want in EDITS[name]:
        text = text.replace(anchor, repl)
    return text


def _bash_n(text: str):
    """bash -n the produced text (via a temp file; nothing is written to the
    target). Returns (ok, stderr)."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(text)
            tmp = fh.name
        r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        return r.returncode == 0, (r.stderr or "").strip()
    finally:
        if tmp:
            try:
                pathlib.Path(tmp).unlink()
            except OSError:
                pass


def _printf_arity(text: str):
    """(%s count, argument count) of the PROBE denominator printf line."""
    for line in text.splitlines():
        if "PROBE denominator:" in line and "printf" in line:
            specs = line.count("%s")
            m = re.search(r"'[^']*'", line)
            args = len(re.findall(r'"[^"]*"', line[m.end():])) if m else -1
            return specs, args
    return -1, -1


def _post_image_stale_ok(posts) -> bool:
    """The real post-condition: the stale token's post-image count is 0 in
    all three files."""
    return all(posts[n].count(STALE_TOKEN) == 0 for n in TARGETS)


def _gate_results(pre, post):
    g = []
    for name in TARGETS:
        for anchor, repl, _want in EDITS[name]:
            g.append((f"gate/{name}/replacement-present", repl in post[name],
                      f"replacement text occurs x{post[name].count(repl)} in produced {name}"))
            g.append((f"gate/{name}/anchor-consumed", anchor not in post[name],
                      f"pre-image anchor occurs x{post[name].count(anchor)} in produced {name}"))
    for name in TARGETS:
        ok, err = _bash_n(post[name])
        g.append((f"gate/{name}/bash-n", ok, err or f"bash -n clean on produced {name}"))
    return g


def _controls(pre, post, shipped, repo):
    notes = []
    ok = 0

    pre_counts = {n: pre[n].count(STALE_TOKEN) for n in TARGETS}
    post_counts = {n: post[n].count(STALE_TOKEN) for n in TARGETS}
    fired = (all(pre_counts[n] == EXPECT_PRE_STALE[n] for n in TARGETS)
             and _post_image_stale_ok(post))
    for n in TARGETS:
        notes.append(f"MUST_FIRE/STALE_NAME_IS_GONE {n}: drove the transform on the real "
                     f"pre-image; observed {STALE_TOKEN} x{pre_counts[n]} before "
                     f"(expected x{EXPECT_PRE_STALE[n]}), x{post_counts[n]} after (expected x0)")
    if any(pre_counts[n] == 0 for n in TARGETS):
        notes.append("MUST_FIRE/STALE_NAME_IS_GONE: a pre-image count is already 0 -- the stage "
                     "would be certifying an edit it did not make; this is REFUSE 96, not a "
                     "silent success")
        fired = False
    ok += int(fired)

    planted_base = "not_in_publish_set" + ".py"
    plant_line = ("# " + MARK + " control plant: the debug probe reads its knobs via tools/" +
                  planted_base)
    planted = post["launcher"] + "\n" + plant_line + "\n"
    plant_lineno = planted.splitlines().index(plant_line) + 1
    # Driven with the repo half RESOLVED (an empty fixture set): only a
    # denominator that can answer the question may convict.
    verdict2, report2 = _census([("planted_copy", planted)], shipped, set())
    named = any(f"planted_copy:{plant_lineno}" in r for r in report2)
    fired2 = verdict2 == "RED" and planted_base not in shipped and named
    notes.append(f"MUST_FIRE/CENSUS_CATCHES_A_PLANTED_UNSHIPPED_SCRIPT: planted {plant_line!r} "
                 f"at planted_copy:{plant_lineno} in a COPY of the produced launcher and ran the "
                 f"real census over it with the repo half resolved (empty fixture set); "
                 f"observed verdict {verdict2}, planted file:line named in the report: {named}")
    ok += int(fired2)

    verdict3, report3 = _census([(n, post[n]) for n in TARGETS], shipped, repo)
    notes.append(f"MUST_PASS/CENSUS_GREEN_ON_THE_REAL_POST_IMAGE: ran the real census over the "
                 f"three real produced files; observed verdict {verdict3} -- {report3[0]}")
    ok += int(verdict3 == "GREEN")

    verdict4, report4 = _census([("synthetic", "x=1\ny=2\n")], shipped, repo)
    fired4 = verdict4 == "UNMEASURED" and any("all([])" in r for r in report4)
    notes.append(f"MUST_FIRE/EMPTY_TEXT_IS_UNMEASURED: drove the census over a synthetic file "
                 f"with no operator-facing text at all; observed verdict {verdict4} "
                 f"({'the all([]) message was printed' if fired4 else '; '.join(report4)})")
    ok += int(fired4)

    pre_specs, pre_args = _printf_arity(pre["launcher"])
    post_specs, post_args = _printf_arity(post["launcher"])
    ok5 = (post_specs, post_args) == (pre_specs, pre_args) and post_specs == post_args > 0
    notes.append(f"MUST_PASS/PRINTF_ARITY_UNCHANGED: pre-image printf %s x{pre_specs} / args "
                 f"x{pre_args}; post-image %s x{post_specs} / args x{post_args} -- a printf "
                 f"whose specifier count drifts from its argument count fails silently")
    ok += int(ok5)

    named6 = (all(SEAM in r for _a, r, _w in EDITS["launcher"])
              and SEAM in R_ALLOW and SEAM in R_HELPER)
    resolves6 = A_ASSIGN in post["launcher"]
    ok6 = named6 and resolves6 and post["launcher"].count(SEAM) >= 2
    notes.append(f"MUST_PASS/SEAM_IS_NAMED_AND_RESOLVES: replacement text names {SEAM}: "
                 f"{named6}; the produced launcher really resolves it at {A_ASSIGN!r}: "
                 f"{resolves6} -- the text points at a seam the file has, not a "
                 f"plausible-sounding variable")
    ok += int(ok6)

    idem = True
    for n in TARGETS:
        again = post[n]
        for anchor, repl, _w in EDITS[n]:
            again = again.replace(anchor, repl)
        if again != post[n]:
            idem = False
            notes.append(f"MUST_PASS/BYTE_IDEMPOTENT: {n} changed under re-application")
    notes.append(f"MUST_PASS/BYTE_IDEMPOTENT: applied the whole transform to its own output on "
                 f"all three files; observed byte-identical output: {idem}")
    ok += int(idem)

    root_level = "tests/test_fix32_container_env_passthrough.py"
    base8 = root_level.split("/")[-1]
    text8 = "# " + MARK + " control: the debug probe reads its knobs via " + root_level + "\n"
    fixture = pathlib.Path(tempfile.mkdtemp(prefix=MARK + "_repo8_"))
    try:
        (fixture / "tests").mkdir(parents=True, exist_ok=True)
        (fixture / root_level).write_text("# fixture: stands in for the real shipped file\n",
                                          "utf-8")
        try:
            init = subprocess.run(["git", "init", "-q", str(fixture)],
                                  capture_output=True, text=True)
            add = subprocess.run(["git", "-C", str(fixture), "add", root_level],
                                 capture_output=True, text=True)
            git_ok = init.returncode == 0 and add.returncode == 0
        except OSError:
            git_ok = False
        repo8 = _repo_tracked_basenames(fixture) if git_ok else None
        verdict8, _report8 = _census([("repo_half_case", text8)], shipped, repo8)
        in_repo8 = repo8 is not None and base8 in repo8
        ok8 = in_repo8 and base8 not in shipped and verdict8 == "GREEN"
        notes.append(f"MUST_PASS/REPO_HALF_ADMITS_A_ROOT_LEVEL_FILE: built a fixture repo root "
                     f"at {fixture} with {root_level} tracked in it (git init + git add; no "
                     f"checkout depended on) and ran the real census over a comment naming the "
                     f"real measured case; observed verdict {verdict8}, basename in the repo "
                     f"half: {in_repo8}, basename in the publish set: {base8 in shipped}")
    finally:
        shutil.rmtree(fixture, ignore_errors=True)
    ok += int(ok8)

    plane9 = pathlib.Path(tempfile.mkdtemp(prefix=MARK + "_plane9_"))
    try:
        root9, how9 = _resolve_repo_root(plane9, None)
        verdict9, _report9 = _census([("repo_half_case", text8)], shipped, None)
        rc9 = {"GREEN": 0, "RED": 5, "UNMEASURED": 95}[verdict9]
        fired9 = root9 is None and verdict9 == "UNMEASURED" and rc9 == 95
        notes.append(f"MUST_FIRE/UNRESOLVED_REPO_IS_UNMEASURED_NOT_RED: the same token "
                     f"({root_level}) with no FS_PUBLISHED_REPO_ROOT passed and no .git above "
                     f"the fixture plane directory ({how9}); observed state {verdict9}, exit "
                     f"code {rc9} -- UNMEASURED and NOT RED")
    finally:
        shutil.rmtree(plane9, ignore_errors=True)
    ok += int(fired9)

    restored = post["bound"].replace(R_ALLOW_CONT, A_ALLOW_CONT)
    post10 = dict(post, bound=restored)
    count10 = restored.count(STALE_TOKEN)
    red10 = not _post_image_stale_ok(post10)
    fired10 = restored != post["bound"] and count10 == 1 and red10
    notes.append(f"MUST_FIRE/DANGLING_CONTINUATION_IS_CAUGHT: took a copy of the produced "
                 f"bound backend, restored ONLY the continuation line to its pre-image text "
                 f"({A_ALLOW_CONT!r}), and ran the real post-conditions over it; observed "
                 f"{STALE_TOKEN} x{count10} in the restored copy and the post-image count "
                 f"assert going red: {red10}")
    ok += int(fired10)

    return ok, notes


def main() -> int:
    # The build's stage convention is NO ARGUMENT == --apply: build_h100_plane.sh invokes every
    # stage as a bare `python3 "$s"` (:315) and expects it to have written. A stage whose bare
    # invocation is a dry-run runs GREEN inside the build and lands nothing, which is exactly the
    # orphan class of #188/#189/#190 -- a fix that was authored, certified and never applied.
    # --check is the explicit dry-run, matching patch_launch_provenance.py:662.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_engine_entrypoint_naming.py [--apply|--check]   "
                "(no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    for name, path in TARGETS.items():
        if not path.exists():
            print(f"UNMEASURED 95: target missing: {path} ({name})")
            return 95
    if not PUBLISH_SET.exists():
        print(f"UNMEASURED 95: publish set missing: {PUBLISH_SET}")
        return 95

    pre = {n: p.read_text("utf-8") for n, p in TARGETS.items()}
    shipped = _shipped_basenames(PUBLISH_SET)
    if not shipped:
        print(f"UNMEASURED 95: {PUBLISH_SET} parsed to zero shipped basenames; the census "
              f"denominator cannot be stated")
        return 95

    try:
        repo_root, how = _resolve_repo_root(ROOT, os.environ.get("FS_PUBLISHED_REPO_ROOT"))
    except _Refuse as exc:
        _stderr(f"REFUSE 96: {exc}; writing nothing to any of the three files")
        return 96
    if repo_root is None:
        repo = None
        print(f"repo root: UNRESOLVED -- {how}")
    else:
        print(f"repo root: resolved via {how}: {repo_root}")
        repo = _repo_tracked_basenames(repo_root)
        if repo is None:
            print(f"repo half UNRESOLVED: git -C {repo_root} ls-files failed or git is "
                  f"absent; the repo half of the denominator is UNMEASURED -- the publish "
                  f"set is NOT treated as the whole world")
    denom = shipped | (repo if repo is not None else set())
    repo_desc = (f"{len(repo)} basename(s) from the repository" if repo is not None
                 else "repo half UNRESOLVED")
    print(f"denominator: {len(shipped)} basename(s) from the publish set, {repo_desc}, "
          f"{len(denom)} total")
    if EXCEPTIONS:
        print(f"exceptions: {len(EXCEPTIONS)} declared: " + ", ".join(sorted(EXCEPTIONS)))
    else:
        print("exceptions: 0 exceptions declared")

    post = {}
    for n in TARGETS:
        try:
            post[n] = _transform(n, pre[n])
        except _Refuse as exc:
            _stderr(f"REFUSE 96: {exc}; the stage does not recognise this file and will not "
                    f"guess; writing nothing")
            return 96

    gates = 0
    gres = _gate_results(pre, post)
    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)

    verdict, report = _census([(n, post[n]) for n in TARGETS], shipped, repo)
    for line in report:
        print("census " + line)
    if verdict == "UNMEASURED":
        _stderr("UNMEASURED 95: the census over the produced text cannot answer on this "
                "denominator (zero tokens, or basename(s) in neither half with the "
                "repository half unresolved); absence of visibility is not evidence of "
                "absence; writing nothing")
        return 95
    if verdict == "RED":
        _stderr("RED 5: census over the produced text found python-script basename(s) in "
                "neither half of a fully resolved denominator, or a stale exception; "
                "writing nothing")
        return 5

    cok, cnotes = _controls(pre, post, shipped, repo)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls "
                f"{cok}/{N_CONTROLS}; writing nothing to any of the three files")
        return 96
    if not apply:
        print(f"verdict: READY  seven entrypoint-naming edits would be applied across three "
              f"files, {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    for n, path in TARGETS.items():
        path.write_text(post[n], "utf-8")
    print(f"{TAG} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls; "
          f"three files written")
    return 0


def _guarded() -> int:
    # Four states only (0/5/95/96): an unhandled exception is a REFUSE with a
    # named message, never a bare rc=1 that collapses the states into one.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {TAG} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())
