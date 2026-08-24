#!/usr/bin/env python3
"""f78 leg splice -- wire the two census-writer legs of
launchers/test_launcher_contracts.sh onto launchers/f78_census_writer_driver.py,
FAIL-CLOSED.

Why a splice tool instead of a blind Edit block
-----------------------------------------------
The driver is complete and in evidence; the two legs it replaces are NOT --
they sit outside every listed window of the suite, and the driver docstring's
own stale line count proves the surrounding text drifts. Doctrine 4 forbids
shipping a find-anchor against unseen bytes. So the swap ships as this tool.
Every boundary needle is a string MEASURED in tonight's red suite output:

  * header needle  -- the 'fix78: the census --out producer refuses ...'
                      section banner (kept verbatim; still accurate);
  * leg needles    -- 'census --out refusal' and 'census --out admission',
                      the two legs being replaced, proven present by their own
                      FAIL lines and proven BOUNDED by file-wide counts
                      (each needle's file-wide occurrence count must equal its
                      in-region count, or the splice refuses: no second copy
                      may be amputated);
  * end needle     -- 'fix78-realmodel', the ABSTAIN leg that follows the two
                      legs (kept verbatim; the region ends strictly before it);
  * anti-needles   -- '#78-B', 'fix41', 'fix44', 'fix45', 'fs_live': legs that
                      must live OUTSIDE the replaced region; any occurrence
                      inside means the boundary over-ran and the tool refuses;
  * machinery      -- 'F78_STAGE', evidence the region really is the old
                      inline neuter/exec family (stage=F78_STAGE=exec-failed)
                      and not already-wired text.

Driver pre-flight (ordering is enforced, not assumed)
-----------------------------------------------------
The spliced legs assert POSITIVE per-unit evidence: one grep of
'F78_FQN_OK|<fqn>' per expected fqn, because verify echoes every expected
name verbatim inside F78_FQNS_MISSING on failure and a bare-name grep would
go green exactly when the artifact is absent (doctrine 3). Those greps are
only provable against a driver that prints F78_FQN_OK|<fqn>, so BOTH 'splice'
and 'check' read the driver and refuse -- rc 15, zero writes,
stage=F78_SPLICE=driver-vocabulary-failed -- unless the marker is present,
naming the ordering duty: the driver's positive-evidence edit lands FIRST.
Without this, a spliced suite on an old driver would mint a red whose cause
is unattributable (doctrine 5).

On any mismatch: rc 15, stage=F78_SPLICE=<why>-failed, ZERO writes. On
success: the original file is first copied byte-for-byte to
SUITE.f78-presplice.bak (audit trail AND the row-schema reconciliation duty
the MUST_PASS no-branch names); the candidate is needle-verified as a pure
string, written to a sibling tmp, syntax-gated with 'bash -n' BEFORE rename,
renamed, re-read and compared. Exit vocabulary mirrors the driver: 0
spliced/verified/already-wired, 15 any refusal, 2 argv misuse.

Usage (from repo root):
  python3 launchers/f78_splice_driver_legs.py splice [SUITE]
  python3 launchers/f78_splice_driver_legs.py check  [SUITE]

What the spliced legs assert (both halves of each contract, per the driver's
own docstring integration; nothing weakened, nothing laundered):
  * MUST_FIRE (zero attachment set): F78_OUT_EXISTS=0 on a doubly-pinned,
    fresh-dir path ('unknown' can never satisfy the grep -- unmeasured is not
    pass) AND F78_VERDICT_TOKENS=.*UNMEASURED, off the REAL AST-lifted writer
    whose probe top level never executes.
  * MUST_PASS (legitimate 2-module fixture): F78_OUT_EXISTS=1,
    F78_JSON_PARSE=ok, each expected FQN covered individually as POSITIVE
    evidence 'F78_FQN_OK|<fqn>' (the bare name is also echoed verbatim inside
    F78_FQNS_MISSING on failure, so a bare-name grep false-greens exactly
    when the artifact is absent -- doctrine 3 needs the positive control),
    F78_FQNS_MISSING=none, and the count as an explicit denominator --
    'F78_FQNS_FOUND=2 of 2' always, 'F78_ARTIFACT_ROWS=2 of 2' whenever the
    artifact carries a rows list ('absent' is honest unmeasured, never a
    forged 0 of m).
  * additive, both legs: F78_EXTRACT_UNRESOLVED=none; !=none re-reads the red
    as a DRIVER extraction gap (extend the driver's lifting, never the leg's
    assertions).
"""

import contextlib
import os
import shutil
import subprocess
import sys

SSTAGE = "F78_SPLICE"
RC_INFRA = 15
RC_USAGE = 2

SUITE_DEFAULT = "launchers/test_launcher_contracts.sh"
DRIVER_PATH = "launchers/f78_census_writer_driver.py"
DRIVER_FQN_MARKER = "F78_FQN_OK|"

HEADER_NEEDLE = "fix78: the census --out producer refuses"
END_NEEDLE = "fix78-realmodel"
LEG_NEEDLES = ("census --out refusal", "census --out admission")
ANTI_NEEDLES = ("#78-B", "fix41", "fix44", "fix45", "fs_live")
MACHINERY_NEEDLE = "F78_STAGE"
DRIVER_MARKER = "f78_census_writer_driver"

# Facts the spliced file must carry; each is a literal substring of the new
# leg block (kept next to it below, so a drifted edit cannot silently pass).
POST_NEEDLES = (
    DRIVER_MARKER,
    "stage=F78_STAGE=drove rc=0",
    "stage=F78_STAGE=verified rc=0",
    "F78_EXTRACT_UNRESOLVED=none",
    "F78_OUT_EXISTS=0",
    "F78_VERDICT_TOKENS=.*UNMEASURED",
    "F78_OUT_EXISTS=1",
    "F78_JSON_PARSE=ok",
    "F78_FQN_OK|",
    "F78_FQNS_MISSING=none",
    "F78_FQNS_FOUND=2 of 2",
    "F78_ARTIFACT_ROWS=2 of 2",
)

USAGE = ("usage: f78_splice_driver_legs.py splice [SUITE]"
         " | f78_splice_driver_legs.py check [SUITE]")


def _fail(tag, detail):
    print(f"stage={SSTAGE}={tag}-failed {detail}")
    sys.exit(RC_INFRA)


def _read(path, tag):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        # doctrine 4: unreadable is not empty; a file we cannot read is an
        # infrastructure red, never a vacuous target.
        _fail(tag, f"{path} unreadable: {exc}")


def _check_driver_vocabulary(path):
    dtext = _read(path, "driver-read")
    if DRIVER_FQN_MARKER not in dtext:
        _fail("driver-vocabulary",
              f"{path} never prints {DRIVER_FQN_MARKER!r} -- the spliced legs assert POSITIVE"
              " per-unit evidence ('F78_FQN_OK|<fqn>' per expected fqn,"
              " because the bare name is echoed inside F78_FQNS_MISSING on"
              " failure) and against a driver without it that evidence would"
              " be unprovable, minting a red whose cause is unattributable"
              " (doctrines 3/5). Land the driver's positive-evidence edit"
              " FIRST, then splice.")
    return dtext


def _missing_needles(text):
    return [n for n in POST_NEEDLES if n not in text]


def _bash_syntax_gate(path, tag):
    try:
        cp = subprocess.run(["bash", "-n", path],
                            capture_output=True, text=True)
    except OSError as exc:
        # the syntax gate is load-bearing; a host without bash cannot green.
        _fail(tag, f"bash unavailable for syntax gate: {exc}")
    if cp.returncode != 0:
        detail = (cp.stderr.strip().splitlines() or ["bash -n failed"])[0]
        _fail("syntax", f"candidate failed 'bash -n': {detail[:200]} (original untouched)")


LEG_BLOCK = r'''
# -- fix78 census-writer legs, driver-wired (spliced by
# launchers/f78_splice_driver_legs.py). These two legs run the probe's REAL
# _persist_adapter_census exactly once per contract arm via
# launchers/f78_census_writer_driver.py, which AST-lifts the writer, its
# same-module closure, its literal constants and its __future__ flags and
# NEVER exec's the probe top level -- the failure mode
# (stage=F78_STAGE=exec-failed) that had made both legs controls that never
# RUN (doctrine 3). Coverage of each expected FQN in MUST_PASS is POSITIVE:
# one 'F78_FQN_OK|<fqn>' grep per unit, because verify echoes every expected
# name verbatim inside F78_FQNS_MISSING on failure -- a bare-name grep would
# go green exactly when the artifact is absent (doctrine 3/5; greps must key
# on the positive form, never the bare name). The spliced-out block (fixtures
# + inline neuter) is preserved byte-for-byte at this suite's
# .f78-presplice.bak: if the MUST_PASS no-branch fires naming a
# fixture-shape raise, reconcile the rows schema from that .bak and a read of
# _persist_adapter_census BEFORE believing a probe red -- schema is a fixture
# fact, today unmeasured, and doctrine 4 forbids guessing it silently.
f78_driver="launchers/f78_census_writer_driver.py"
f78_probe="${F39_PROBE:-launchers/lora_target_census.py}"
f78_msim=$(mktemp -d "${TMPDIR:-/tmp}/fs-f78-legs.XXXXXX" 2>/dev/null) || f78_msim=""
[ -n "$f78_msim" ] || { f78_msim="${TMPDIR:-/tmp}/fs-f78-legs.$$"; mkdir -p "$f78_msim" 2>/dev/null || f78_msim=""; }
if [ -z "$f78_msim" ]; then
  no "fix78 UNREACHABLE: no scratch dir for the census-writer fixtures -- both fix78 legs are controls that never RUN (doctrine 3), and a leg that never runs is not a control"
elif [ ! -r "$f78_driver" ] || [ ! -r "$f78_probe" ]; then
  no "fix78 UNREACHABLE: driver readable=$([ -r "$f78_driver" ] && echo 1 || echo 0) probe readable=$([ -r "$f78_probe" ] && echo 1 || echo 0) -- unreadable is not empty (doctrine 4); both census-writer legs unproven"
else
  # MUST_FIRE fixture: a genuinely ZERO attachment set (rows [] beside a
  # non-vacuous scanned population of 1556 -- the probe's own published
  # census header number). The artifact path is pinned twice (kwargs AND
  # spec.out_path) and lives in the fresh scratch dir, so F78_OUT_EXISTS is
  # measured against a path provably absent beforehand -- never 'unknown'.
  cat > "$f78_msim/fire-spec.json" <<F78SPEC
{"kwargs": {"out_path": "$f78_msim/fire-census.json", "rows": [], "targets": [], "total": 0, "population": 1556, "hf_model_path": "fix78-zero-attachment-fixture"}, "out_path": "$f78_msim/fire-census.json"}
F78SPEC
  f78_rc=0
  out=$(python3 "$f78_driver" drive "$f78_probe" "$f78_msim/fire-spec.json" 2>&1) || f78_rc=$?
  f78_funcs=$(printf '%s\n' "$out" | sed -n 's/^F78_EXTRACT_FUNCS=//p' | head -n 1)
  if ! printf '%s\n' "$out" | grep -qF 'stage=F78_STAGE=drove rc=0'; then
    no "MUST_FIRE UNREACHABLE (census --out refusal): the driver itself failed before measuring anything (rc=$f78_rc; $(printf '%s\n' "$out" | grep 'F78_STAGE=' | head -n 1)) -- infrastructure red on the driver's published vocabulary, never a probe verdict; a control that cannot RUN is not a control"
  elif ! printf '%s\n' "$out" | grep -qF 'F78_EXTRACT_UNRESOLVED=none'; then
    no "MUST_FIRE (census --out refusal) DRIVER extraction gap (funcs=$f78_funcs unresolved=$(printf '%s\n' "$out" | sed -n 's/^F78_EXTRACT_UNRESOLVED=//p' | head -n 1)) -- a lifted-closure name miss is a DRIVER red: extend the driver's lifting, never this leg's assertions"
  elif printf '%s\n' "$out" | grep -qF 'F78_OUT_EXISTS=0' && printf '%s\n' "$out" | grep -q 'F78_VERDICT_TOKENS=.*UNMEASURED'; then
    ok "MUST_FIRE (census --out refusal): a zero attachment set makes the REAL census writer (lifted funcs=$f78_funcs) refuse by the book -- NO census file afterwards (F78_OUT_EXISTS=0 on a doubly-pinned, pre-absent path; 'unknown' can never satisfy this grep) AND the verdict is UNMEASURED in the writer's own captured words"
  else
    no "MUST_FIRE (census --out refusal): the zero-attachment drive did not observe the refusal ($(printf '%s\n' "$out" | grep -E '^F78_(OUT_EXISTS|VERDICT_TOKENS|RAISED|EXIT)=' | tr '\n' ' ')) -- either the empty guard never ran under the REAL writer or the producer ships a census on an empty attachment set; both are this leg's red, and neither may be read as a pass"
  fi
  # MUST_PASS fixture: a legitimate non-empty set of two synthesized module
  # rows authored as bare fqn strings. The row schema is the one look-up owed
  # to the preserved .bak and the probe read named above; it is grounds for
  # the distinctly-routed red below, never for a green.
  f78_fqn1="module.layers.0.self_attn.linear_qkv"
  f78_fqn2="module.layers.0.self_attn.linear_proj"
  cat > "$f78_msim/pass-spec.json" <<F78SPEC
{"kwargs": {"out_path": "$f78_msim/pass-census.json", "rows": ["$f78_fqn1", "$f78_fqn2"], "targets": ["linear_qkv", "linear_proj"], "total": 2, "population": 1556, "hf_model_path": "fix78-two-module-fixture"}, "out_path": "$f78_msim/pass-census.json"}
F78SPEC
  cat > "$f78_msim/verify-spec.json" <<F78SPEC
{"artifact": "$f78_msim/pass-census.json", "expect_fqns": ["$f78_fqn1", "$f78_fqn2"], "expect_denominator": 2}
F78SPEC
  f78_rc=0
  out=$(python3 "$f78_driver" drive "$f78_probe" "$f78_msim/pass-spec.json" 2>&1) || f78_rc=$?
  f78_vrc=0
  ver=$(python3 "$f78_driver" verify "$f78_msim/verify-spec.json" 2>&1) || f78_vrc=$?
  f78_funcs=$(printf '%s\n' "$out" | sed -n 's/^F78_EXTRACT_FUNCS=//p' | head -n 1)
  f78_rows=$(printf '%s\n' "$ver" | sed -n 's/^F78_ARTIFACT_ROWS=/rows=/p' | head -n 1)
  # Positive per-unit evidence, doctrine 2/3: one F78_FQN_OK|<fqn> line per
  # expected fqn. The bare name is also echoed verbatim inside
  # F78_FQNS_MISSING on failure, so this greps the positive form ONLY -- a
  # bare-name grep would go green exactly when the artifact is absent.
  f78_all_ok=1
  for f78_want in "$f78_fqn1" "$f78_fqn2"; do
    printf '%s\n' "$ver" | grep -qF "F78_FQN_OK|$f78_want" || f78_all_ok=0
  done
  if ! printf '%s\n' "$out" | grep -qF 'stage=F78_STAGE=drove rc=0' || ! printf '%s\n' "$ver" | grep -qF 'stage=F78_STAGE=verified rc=0'; then
    no "MUST_PASS UNREACHABLE (census --out admission): the driver itself failed before measuring anything (drive rc=$f78_rc verify rc=$f78_vrc; $(printf '%s\n%s\n' "$out" "$ver" | grep 'F78_STAGE=' | head -n 1)) -- infrastructure red on the driver's published vocabulary, never a probe verdict"
  elif ! printf '%s\n' "$out" | grep -qF 'F78_EXTRACT_UNRESOLVED=none'; then
    no "MUST_PASS (census --out admission) DRIVER extraction gap (funcs=$f78_funcs unresolved=$(printf '%s\n' "$out" | sed -n 's/^F78_EXTRACT_UNRESOLVED=//p' | head -n 1)) -- a DRIVER red: extend the driver's lifting, never this leg's assertions"
  elif printf '%s\n' "$out" | grep -qF 'F78_OUT_EXISTS=1' \
    && printf '%s\n' "$ver" | grep -qF 'F78_JSON_PARSE=ok' \
    && [ "$f78_all_ok" -eq 1 ] \
    && printf '%s\n' "$ver" | grep -qF 'F78_FQNS_MISSING=none' \
    && printf '%s\n' "$ver" | grep -qF 'F78_FQNS_FOUND=2 of 2' \
    && { printf '%s\n' "$ver" | grep -qF 'F78_ARTIFACT_ROWS=2 of 2' || printf '%s\n' "$ver" | grep -qF 'F78_ARTIFACT_ROWS=absent'; }; then
    ok "MUST_PASS (census --out admission): the REAL census writer (lifted funcs=$f78_funcs) admitted a legitimate 2-module fixture -- the file EXISTS (F78_OUT_EXISTS=1) and parses, each expected fqn is proven found individually (positive F78_FQN_OK|<fqn>, never satisfiable by the F78_FQNS_MISSING echo of a failed run), F78_FQNS_MISSING=none at denominator 2, and the count is printed as an explicit denominator (F78_FQNS_FOUND=2 of 2; $f78_rows -- 'absent' is honest unmeasured per doctrine 4, never a forged 0 of 2)"
  else
    no "MUST_PASS (census --out admission): the real writer did not admit a legitimate non-empty fixture ($(printf '%s\n%s\n' "$out" "$ver" | grep -E '^F78_(OUT_EXISTS|RAISED|EXIT|JSON_PARSE|FQNS_MISSING|FQNS_FOUND|FQN_OK|ARTIFACT_ROWS|EXTRACT_UNRESOLVED)=' | tr '\n' ' ')) -- if F78_RAISED names a fixture-shape error, the rows schema is owed reconciliation against the writer's contract via the preserved .f78-presplice.bak and a probe read (a FIXTURE red, attributed); otherwise the writer refuses a legitimate non-empty set (a PROBE red); a control that cannot green on a good input cries wolf, and an unread cause is never a pass"
  fi
  [ -n "$f78_msim" ] && rm -rf "$f78_msim" || true
fi
'''


def cmd_check(path):
    _check_driver_vocabulary(DRIVER_PATH)
    text = _read(path, "check")
    if DRIVER_MARKER not in text:
        _fail("not-wired", f"suite carries no reference to {DRIVER_MARKER} -- run 'splice'")
    missing = _missing_needles(text)
    if missing:
        _fail("verify", "wired suite is missing post-splice needles: {}".format(",".join(missing)))
    _bash_syntax_gate(path, "check-syntax")
    print(f"stage={SSTAGE}=verified rc=0")
    print(f"F78_SPLICE_DRIVER_VOCAB={DRIVER_FQN_MARKER!r} present (positive per-unit evidence)")
    print("F78_SPLICE_POST_NEEDLES=%d of %d"
          % (len(POST_NEEDLES), len(POST_NEEDLES)))
    return 0


def cmd_splice(path):
    _check_driver_vocabulary(DRIVER_PATH)
    text = _read(path, "read")
    if DRIVER_MARKER in text:
        # Idempotence is a VERIFIED state, not an assumption (doctrine 3: a
        # control must be observed, and here the observation is cheap).
        missing = _missing_needles(text)
        if missing:
            _fail("verify", "suite is wired but lost post-splice needles: {}".format(",".join(missing)))
        _bash_syntax_gate(path, "check-syntax")
        print(f"stage={SSTAGE}=already-wired rc=0")
        print(f"F78_SPLICE_DRIVER_VOCAB={DRIVER_FQN_MARKER!r} present (positive per-unit"
              " evidence)")
        print("F78_SPLICE_POST_NEEDLES=%d of %d"
              % (len(POST_NEEDLES), len(POST_NEEDLES)))
        return 0

    lines = text.splitlines(keepends=True)
    hdr = [i for i, l in enumerate(lines) if HEADER_NEEDLE in l]
    if not hdr:
        _fail("header-missing", f"needle not found: {HEADER_NEEDLE!r}")
    if len(hdr) != 1:
        _fail("header-ambiguous", "needle at suite lines %s"
              % [i + 1 for i in hdr])
    h = hdr[0]
    ends = [i for i in range(h + 1, len(lines)) if END_NEEDLE in lines[i]]
    if not ends:
        _fail("boundary-missing",
              "%r not found after header at suite line %d" % (END_NEEDLE,
                                                              h + 1))
    e = ends[0]
    if e - (h + 1) < 2:
        _fail("region-empty", "only %d line(s) between header (suite line %d)"
              " and %r boundary -- nothing recognizable to replace"
              % (e - (h + 1), h + 1, END_NEEDLE))

    region = lines[h + 1:e]
    rtext = "".join(region)
    # Both legs must lie ENTIRELY inside the region: each needle's file-wide
    # count must equal its in-region count (no second copy may be amputated).
    for n in LEG_NEEDLES:
        c_all, c_reg = text.count(n), rtext.count(n)
        if c_reg == 0:
            _fail("leg-needle-missing",
                  "%r absent from region (suite lines %d-%d) -- refusing to"
                  " splice unverified bytes" % (n, h + 2, e + 1))
        if c_all != c_reg:
            _fail("leg-needle-not-unique",
                  "%r occurs %d time(s) file-wide but %d in region -- the"
                  " boundary would amputate a copy outside it"
                  % (n, c_all, c_reg))
    for n in ANTI_NEEDLES:
        if n in rtext:
            _fail("region-overrun",
                  "preserved-neighbor needle %r inside region (suite lines"
                  " %d-%d) -- boundary over-ran into a leg that must live"
                  % (n, h + 2, e + 1))
    if MACHINERY_NEEDLE not in rtext:
        _fail("machinery-absent",
              f"no {MACHINERY_NEEDLE!r} evidence in region -- the inline neuter/exec family is"
              " not recognizable here")

    block = LEG_BLOCK if LEG_BLOCK.endswith("\n") else LEG_BLOCK + "\n"
    new_text = "".join(lines[:h + 1]) + block + "".join(lines[e:])

    # Candidate validation happens on BYTES NOT YET WRITTEN: needle facts,
    # preserved-neighbor counts, then bash -n on a sibling tmp. The rename is
    # the last mutation, and the original is backed up first (doctrine 4).
    missing = _missing_needles(new_text)
    if missing:
        _fail("self-check", "embedded leg block is missing needles: {} -- the"
              " DRIVER-side splice payload drifted; fix this file, not the"
              " legs".format(",".join(missing)))
    preserved = END_NEEDLE, "#78-B", HEADER_NEEDLE
    for n in preserved:
        if text.count(n) != new_text.count(n):
            _fail("self-check", "preserved needle %r count changed (%d -> %d)"
                  % (n, text.count(n), new_text.count(n)))

    backup = path + ".f78-presplice.bak"
    tmp = path + ".f78-splice.tmp"
    try:
        shutil.copy2(path, backup)
    except OSError as exc:
        _fail("backup", f"could not preserve the original at {backup}: {exc} -- no"
              " splice without an audit trail")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new_text)
    except OSError as exc:
        _fail("write", f"candidate tmp unwritable: {exc}")
    _bash_syntax_gate(tmp, "syntax")
    try:
        os.replace(tmp, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        _fail("rename", f"os.replace failed: {exc} (original untouched; backup"
              f" {backup})")
    if _read(path, "post") != new_text:
        _fail("post", "re-read of the spliced suite differs from the"
              f" validated candidate (backup {backup})")

    print(f"stage={SSTAGE}=spliced rc=0")
    print("F78_SPLICE_REPLACED_LINES=%d" % len(region))
    print("F78_SPLICE_REGION=suite lines %d-%d (header kept at %d, %r"
          " boundary kept at %d pre-splice)" % (h + 2, e + 1, h + 1,
                                                END_NEEDLE, e + 1))
    print(f"F78_SPLICE_BAK={backup}")
    print(f"F78_SPLICE_DRIVER_VOCAB={DRIVER_FQN_MARKER!r} present (positive per-unit evidence)")
    print("F78_SPLICE_POST_NEEDLES=%d of %d"
          % (len(POST_NEEDLES), len(POST_NEEDLES)))
    # The one duty that could NOT be discharged blind (doctrine 4, stated,
    # never guessed): the writer's row schema, previously carried only by the
    # spliced-out fixture. It is preserved for reconciliation; a fixture red
    # names this line, and no amount of mismatch mints a green.
    print("F78_SPLICE_OWED=row-schema reconciliation: .bak above plus"
          " sed -n '/def _persist_adapter_census/,/^def /p'"
          " launchers/lora_target_census.py")
    return 0


def main(argv):
    path = argv[2] if len(argv) == 3 else SUITE_DEFAULT
    if len(argv) in (2, 3) and argv[1] == "splice":
        return cmd_splice(path)
    if len(argv) in (2, 3) and argv[1] == "check":
        return cmd_check(path)
    sys.stderr.write(USAGE + "\n")
    return RC_USAGE


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SystemExit:
        raise  # _fail()'s rc-15, usage rc-2 and sys.exit(main()) pass through
    except BaseException as exc:
        # off-vocabulary exits are forbidden: any unexpected splice bug is
        # INFRA red on the published vocabulary, never a bare rc-1 traceback.
        print(f"stage={SSTAGE}=splice-failed {type(exc).__name__}: {exc}")
        sys.exit(RC_INFRA)
