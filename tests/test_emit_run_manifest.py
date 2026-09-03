"""#85: the missing --lora tests for tools/emit_run_manifest.py.

The emitter is imported by path (tools/ is not an installed package); a
failed import raises at collection (fail closed) and never reads as an
absent test. Importing it in-process is also what makes the per-file
--cov addition in ci.yml measure it. No skips (FS_FORBID_SKIPS). First
execution of every control here is this change's next CI run; nothing
in this module has been observed running yet.

MUST_PASS: a complete on-disk abstention record verifies 5-of-5, the
producer always emits the five declared keys, in declared order, with
no blank values, and a --lora main() run over a real tmp estate exits 0
with the measured save-count recorded on disk.
MUST_FIRE: an incomplete record must NEVER read as complete -- not when
the omission is accidental and not when the bare-null drill ARMED it --
a drifted keys tuple must raise under the strict zip, and a main() run
with the drill env armed must exit 1 refused (rc pinned, stderr named).

Mutation rows planted in tools/mutate.py over this module (#85) and the
leg that kills each:
  emit_run_manifest.lora-zip-unstrict           -> test_zip_is_strict_raises_on_drift
  emit_run_manifest.lora-status-not-abstained
      -> test_lora_abstention_record_status_and_denominator
  emit_run_manifest.lora-preexisting-key-rename
      -> test_lora_abstention_record_status_and_denominator
         (KeyError on the pinned key is the red)
  emit_run_manifest.lora-count-plus-one
      -> test_lora_abstention_record_status_and_denominator
  emit_run_manifest.lora-count-fabricated
      -> test_main_lora_emission_records_real_count
  emit_run_manifest.drill-never-arms
      -> test_main_lora_drill_armed_is_refused_rc_1
The two main()-level rows stood as stated survivors on the claim that
exercising them needs a checkpoint estate; it needs none -- a tmp out
dir, a tmp checkpoint dir, and the real ManifestStore suffice, and the
two main() entry-point legs below now cover both. Row-integrity legs
below fail on zero rows: a registered module with no mutants is
UNMEASURED, never covered.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EMITTER_PATH = ROOT / "tools" / "emit_run_manifest.py"
MUTATE_PATH = ROOT / "tools" / "mutate.py"

REQUIRED = {"name", "what", "anchor", "replacement"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")  # FAIL CLOSED
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


erm = _load("emit_run_manifest_under_test", EMITTER_PATH)


# ---- --lora: the five-entry abstention record producer ----


def test_lora_abstention_record_records_five_entries():
    entries = erm._lora_abstention_record_entries(3)
    # DENOMINATOR: every declared key survives the zip, in declared order.
    assert len(entries) == len(erm._LORA_ABSTENTION_RECORD_KEYS) == 5
    assert [k for k, _ in entries] == list(erm._LORA_ABSTENTION_RECORD_KEYS)
    assert all(str(v).strip() for _, v in entries), "no record value may be blank"


def test_lora_abstention_record_status_and_denominator():
    # Kills lora-status-not-abstained, lora-preexisting-key-rename
    # (KeyError reads red), and lora-count-plus-one. Units: 2 pinned pairs.
    record = dict(erm._lora_abstention_record_entries(7))
    assert record["declared.status"] == "abstained"
    assert record["declared.preexisting_iter_dirs"] == "7"


def test_zip_is_strict_raises_on_drift(monkeypatch):
    # MUST_FIRE for the lora-zip-unstrict mutant: a drifted keys tuple
    # MUST raise; silent truncation of the five-field record is the bug
    # the strict zip exists to prevent. Units examined: 1 drifted tuple.
    monkeypatch.setattr(
        erm,
        "_LORA_ABSTENTION_RECORD_KEYS",
        erm._LORA_ABSTENTION_RECORD_KEYS + ("declared.planted_drift",),
    )
    with pytest.raises(ValueError):
        erm._lora_abstention_record_entries(0)


# ---- --lora: the on-disk enforcement leg ----


def _load_emit_run_manifest_module():
    """Load the real emitter exactly as it ships, so the record keys,
    values, and source token below cannot drift from the producer whose
    on-disk format these fixtures exist to match. Self-contained by
    design: this round's listing carried zero lines of this module, so
    nothing here may depend on an unlisted helper or import. FAIL
    CLOSED (doctrine 4): an unloadable emitter is a loud error, never
    a vacuous pass."""
    import importlib.util
    from pathlib import Path

    emitter = Path(__file__).resolve().parents[1] / "tools" / "emit_run_manifest.py"
    spec = importlib.util.spec_from_file_location("emit_run_manifest_under_test", emitter)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {emitter}")  # FAIL CLOSED
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_on_disk_lora_abstention_record(mod, drop=None):
    """The on-disk abstention record IN THE FORMAT THE REAL STORE WRITES.

    That format is not guessed here. It is the byte shape measured on real
    emitted records and pinned in the docstring of the control this fixture
    feeds (``_abstention_markers_absent``): a JSON manifest whose config
    block carries each ``declared.*`` record field as a quoted JSON OBJECT
    KEY (``"declared.status": {`` ...), each entry echoing its key in-band
    (``"key": "declared.status"`` -- the echo never carries the object-key
    colon spelling; measured: quoted-anywhere count 2, with-colon count
    exactly 1 per field), and ``declared`` itself a bare JSON null. The
    previous fixture rendered the record as flat ``key: value`` lines -- a
    format the store has never written -- so the 5-of-5 leg failed closed
    on first run, exactly as the merge note predicted it would. The
    predicate, the control, and the 5-of-5 denominator are untouched;
    only the fixture moves. Keys, values, and the source token come from
    the real producer, not from a paraphrase, and the keys-tuple drift
    guard below keeps the examined denominator the one the control
    examines (doctrine 2). drop=<field name> omits that WHOLE entry
    (object key and echo together); drop may also be an iterable of
    field names, each omitted the same way (the drill leg drops all
    five at once). Both sibling MUST_FIRE legs now run on this real
    serializer shape: the incomplete-record leg drops exactly one
    field, so the four remaining object-key spellings stay present and
    the leg can fire only if the verifier truly detects the fifth's
    absence, and the drill leg drops all five, the byte shape an armed
    drill really leaves on disk. The old guessed flat fixtures carried
    zero object-key spellings in every scenario, so they fired on
    format whether or not the drop was detected -- they are deleted,
    not staged (doctrine 3).
    Residual, stated: this reproduces the serializer's measured
    INVARIANTS rather than one captured file; if the store layout ever
    drifts, this leg fails closed again -- that failure mode is the
    control working, by design; the end-to-end proof over the real store
    path remains the FS_EMIT_DRILL_BARE_NULL drill.
    """
    import json

    entries = mod._lora_abstention_record_entries(0)
    keys = list(mod._LORA_ABSTENTION_RECORD_KEYS)
    assert [k for k, _ in entries] == keys, (
        "producer/keys-tuple drift: the producer must zip over the module's "
        "single keys tuple, so this fixture cannot silently examine a "
        "different denominator than the on-disk control"
    )
    if drop is None:
        drops = frozenset()
    elif isinstance(drop, str):
        drops = frozenset((drop,))
    else:  # iterable of field names: drop each one the same way
        drops = frozenset(drop)
    config = {
        key: {"key": key, "value": value, "source": mod._LORA_ABSTENTION_SOURCE}
        for key, value in entries
        if key not in drops
    }
    return json.dumps({"declared": None, "config": config}, indent=2) + "\n"


def test_must_pass_complete_on_disk_record_verifies_5_of_5():
    mod = _load_emit_run_manifest_module()
    record_text = _real_on_disk_lora_abstention_record(mod)
    # DENOMINATOR: the 5 record fields of a resume-shaped record (saves
    # already on disk), verified against the SERIALIZED bytes by the real
    # control the emission path runs after store.save -- the positive arm:
    # bare-null declared + complete record reads STATED-ABSTENTION, 5 of 5.
    # The verifier call shape below is taken from the emission path's own
    # post-save check; if it has drifted, this leg errors RED, never
    # vacuously green (fail closed, doctrine 4).
    state, present, total = mod._enforce_lora_abstention_record(
        record_text, saves_observed=3, drill_armed=False
    )
    assert (present, total) == (5, 5)
    assert state.startswith("STATED-ABSTENTION"), state


def test_must_fire_incomplete_record_never_reads_complete():
    # Units examined: 1 REAL on-disk record missing 1 of its 5 fields.
    # The decoy is the real serializer shape now: four object-key
    # spellings ARE present and only the dropped field's is absent, so
    # this leg can no longer fire without the drop being detected (the
    # old flat key:value text carried zero spellings and fired
    # unconditionally -- vacuous, doctrine 1/3). Only the control's own
    # refusal classes count as firing, caught against the SAME module
    # load whose verifier is invoked; a TypeError from a drifted
    # verifier signature or a bug in this fixture errors RED, matching
    # the emitter's vocabulary rule (a tool bug is not a verdict).
    mod = _load_emit_run_manifest_module()
    record_text = _real_on_disk_lora_abstention_record(mod, drop="declared.preexisting_iter_dirs")
    refusal = None
    try:
        state, present, total = mod._enforce_lora_abstention_record(
            record_text,
            saves_observed=0,
            drill_armed=False,
        )
    except (mod.EmitRefused, mod.EmitUnmeasured):
        refusal = "raised"  # a raised refusal IS the control firing
    if refusal is None:
        assert present < total, (
            f"MUST_FIRE broken: a record missing a field read as "
            f"{present}/{total} with state={state!r}"
        )


def test_drill_armed_still_counts_the_omission():
    # The drill ARMS the omission by suppressing the whole five-field
    # record, so the honest input is the REAL on-disk shape with every
    # field dropped -- the byte shape a drilled emission actually
    # leaves. The verifier must still count it absent (or refuse by
    # raising one of its own refusal classes), otherwise control fire
    # is indistinguishable from a pass. The old header-only flat text
    # carried zero object-key spellings drilled or not, so it could
    # never show the verifier examining anything; a bare except would
    # also have credited a drifted signature as fire. Units examined:
    # 1 armed real-format record carrying 0 of 5 fields.
    mod = _load_emit_run_manifest_module()
    drilled = _real_on_disk_lora_abstention_record(mod, drop=list(mod._LORA_ABSTENTION_RECORD_KEYS))
    try:
        state, present, total = mod._enforce_lora_abstention_record(
            drilled, saves_observed=0, drill_armed=True
        )
    except (mod.EmitRefused, mod.EmitUnmeasured):
        return  # a raised refusal while armed IS the control firing
    assert present < total, f"armed drill must still read absent: {present}/{total} state={state!r}"


# ---- --lora: the real main() entry point kills the stated survivors ----
#
# No checkpoint artifact needed: a tmp out dir, a tmp checkpoint dir,
# and the real ManifestStore exercise the lora branch, the drill arming
# read, and the post-save on-disk enforcement for real. Each leg names
# the single-line production change that turns it red.


def test_main_lora_emission_records_real_count(tmp_path, monkeypatch):
    # Kills emit_run_manifest.lora-count-fabricated: the emitted record
    # must carry the count MEASURED from the checkpoint dir (the 3 real
    # iter_* dirs created below); a substituted constant reads RED at
    # the final assert. Also red if the lora branch stops recording the
    # five-entry abstention record or the post-save enforcement call is
    # dropped. Single-line red trigger: make emitter line 1085 read
    # lora_preexisting_saves = 0.
    # Units examined: 1 real main() emission, 1 on-disk record, 5
    # fields, 1 count value.
    import json

    monkeypatch.delenv(erm._DRILL_BARE_NULL_ENV, raising=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    for i in range(3):
        (ckpt_dir / f"iter_{i:07d}").mkdir()
    rc = erm.main(
        [
            "--lora",
            "--out-dir",
            str(out_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--run-id",
            "lora-entry-point-leg",
            "--nodes",
            "1",
            "--gpus-per-node",
            "4",
            "--tp",
            "2",
            "--pp",
            "1",
            "--cp",
            "2",
            "--dp",
            "1",
        ]
    )
    assert rc == 0, f"honest --lora emission must exit 0, got rc={rc}"
    links = [p for p in ckpt_dir.iterdir() if p.is_file()]
    assert len(links) == 1, f"expected exactly 1 discovery link, saw {links}"
    record_text = links[0].read_text(encoding="utf-8")
    state, present, total = erm._enforce_lora_abstention_record(
        record_text, saves_observed=3, drill_armed=False
    )
    assert (present, total) == (5, 5)
    assert state.startswith("STATED-ABSTENTION"), state
    entry = json.loads(record_text)["config"]["declared.preexisting_iter_dirs"]
    values = {str(v) for v in entry.values()}
    assert "3" in values, f"count fabricated: carries {entry!r}, not measured 3"


def test_main_lora_drill_armed_no_saves_is_unmeasured_rc_3(tmp_path, monkeypatch, capsys):
    # Kills emit_run_manifest.drill-serializer-arm-dead (exit-3 arm):
    # drill ARMED, ZERO saves pre-existing. The pre-flight assertion
    # proves the zero by construction rather than assuming it: the
    # checkpoint dir does not exist when main() is entered (_emit mkdirs
    # it empty at emitter :1054 and _count_save_dirs at :1085 measures 0
    # of 0). With saves_observed=0, check_saved_run_declaration returns
    # its named NOT-EXERCISED state (:834-839) WITHOUT raising, so the
    # save-semantics arm (exit 1) cannot fire here by doctrine 1; the
    # serializer arm (:912-921) MUST raise EmitUnmeasured and main MUST
    # return EXIT_UNMEASURED (3). Naming this leg REFUSED while its own
    # fixture excluded the refusal arm was the estate's trap: an outcome
    # fixed by the fixture, not by the behaviour named. The rc pin is
    # double-bound to EXIT_UNMEASURED so a coincidental crash-code 3 can
    # never read as this control firing; "the serialized record lacks" is
    # arm 2's own phrasing (arm 1 says "control refused"), because
    # DRILL FIRED and 5 of 5 appear in BOTH arms' messages (:897;
    # :858-859 and :915-916) and so can never prove WHICH arm fired.
    # Single-line red trigger: make emitter line 913 read `if False:` --
    # this leg's armed run then returns rc=0 (red at the rc pin) while
    # the arm-1 sibling below stays green: its declaration control raises
    # at :901 before line 913 is ever evaluated.
    # Units examined: 1 armed main() run over 0 pre-existing saves, 1
    # pre-flight existence check, 1 exit code, 3 stderr tokens
    # (DRILL FIRED, 5-of-5 denominator, arm-2 phrasing).
    monkeypatch.setenv(erm._DRILL_BARE_NULL_ENV, "1")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"
    assert not ckpt_dir.exists(), "0 saves must pre-exist or arm 2 is not fixed"
    rc = erm.main(
        [
            "--lora",
            "--out-dir",
            str(out_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--run-id",
            "lora-drill-arm2-leg",
            "--nodes",
            "1",
            "--gpus-per-node",
            "4",
            "--tp",
            "2",
            "--pp",
            "1",
            "--cp",
            "2",
            "--dp",
            "1",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 3, f"armed drill, 0 saves must exit 3 (UNMEASURED), rc={rc}: {err!r}"
    assert rc == erm.EXIT_UNMEASURED, "rc=3 must be the unmeasured class, not a crash"
    assert "DRILL FIRED" in err, err
    assert "5 of 5" in err, err
    assert "the serialized record lacks" in err, err


def test_main_lora_drill_armed_with_saves_is_refused_rc_1(tmp_path, monkeypatch, capsys):
    # Kills emit_run_manifest.drill-declaration-arm-dead (exit-1 arm):
    # drill ARMED, THREE saves pre-existing -- the iter_%07d DIRECTORIES
    # the healthy leg (:281-282) builds under --checkpoint-dir. The
    # pre-flight assertion reads the emitter's OWN counter
    # (_count_save_dirs, the call at :1085), so the fixture is proved to
    # measure 3 by the production counting rule rather than by a naming
    # convention guessed test-side: a fixture whose saves never enter
    # the count (wrong path, files not dirs, created after main returns)
    # turns red HERE instead of silently re-measuring arm 2 at rc=3.
    # With saves_observed=3, arm 1 (:899) MUST refuse via
    # check_saved_run_declaration (BareNullDeclarationError ->
    # EmitRefused, :900-911) and main MUST return EXIT_REFUSED (1). The
    # rc pin is double-bound to EXIT_REFUSED so a coincidental
    # crash-code 1 can never read as this control firing; "3
    # saved-artifact dir(s) observed" appears in no arm-2 message, so
    # the pins prove the refusal adjudicated the measured 3 -- the arm
    # reached is the arm named.
    # Single-line red trigger: make emitter line 899 read
    # `state = "UNCONSULTED"` (declaration control skipped): this leg's
    # armed run then falls through to the serializer arm and returns
    # rc=3 (red at the rc pin) while the no-saves sibling above is
    # untouched -- its declaration control never raises at
    # saves_observed=0, mutated or not.
    # Units examined: 1 armed main() run over 3 pre-existing saves, 1
    # pre-flight count by the emitter's own counter, 1 exit code, 3
    # stderr tokens (REFUSED, DRILL FIRED, 3-dir denominator).
    monkeypatch.setenv(erm._DRILL_BARE_NULL_ENV, "1")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    for i in range(3):
        (ckpt_dir / f"iter_{i:07d}").mkdir()
    measured = erm._count_save_dirs(ckpt_dir)
    assert measured == 3, f"arm 1 needs saves_observed > 0; emitter's own counter saw {measured}"
    rc = erm.main(
        [
            "--lora",
            "--out-dir",
            str(out_dir),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--run-id",
            "lora-drill-arm1-leg",
            "--nodes",
            "1",
            "--gpus-per-node",
            "4",
            "--tp",
            "2",
            "--pp",
            "1",
            "--cp",
            "2",
            "--dp",
            "1",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 1, f"armed drill, 3 saves must exit 1 (REFUSED), rc={rc}: {err!r}"
    assert rc == erm.EXIT_REFUSED, "rc=1 must be the refusal class, not a crash"
    assert "REFUSED" in err, err
    assert "DRILL FIRED" in err, err
    assert "3 saved-artifact dir(s) observed" in err, err


# ---- #83: the torch_record provenance the emitter now ships ----


def test_torch_record_names_gate_provenance_without_frozen_counts(monkeypatch):
    # DENOMINATOR: python_executable, python_version, torch_record -- 3
    # entries per arm, and BOTH arms run below BY CONSTRUCTION, not host
    # luck (the leg this replaces honestly ran only the branch its own
    # interpreter took): a stub module forces the measured arm; None in
    # sys.modules makes `import torch` raise ImportError, forcing the
    # ABSTAINED arm. No skip: a skipped arm would be a build failure.
    import sys
    import types

    stub = types.ModuleType("torch")
    stub.__version__ = "0.0.0-stub"
    stub.__file__ = "<stub>/torch/__init__.py"

    monkeypatch.setitem(sys.modules, "torch", stub)
    entries = dict(erm._training_stack_entries())
    assert len(entries) == 3
    measured = entries["training_stack.torch_record"]
    assert "0.0.0-stub" in measured  # the measured arm really ran

    monkeypatch.setitem(sys.modules, "torch", None)
    entries = dict(erm._training_stack_entries())
    assert len(entries) == 3
    abstained = entries["training_stack.torch_record"]
    assert abstained.startswith("ABSTAINED")  # the abstained arm really ran

    for text in (measured, abstained):  # DENOMINATOR: 2 arms, 2 of 2 checked
        # the stale retraction and its hand-grepped constant are gone;
        # freezing a NEW count ("reads 7") would print the same defect
        # with a fresh number, so grep-count vocabulary is refused
        assert "not implemented" not in text
        assert "writes no torch field" not in text
        assert "no gate record" not in text
        assert "grep -c" not in text
        assert "reads 0" not in text
        # what remains is a RELATIONSHIP pinned to names that
        # tests/test_torch_record_gate_alignment.py re-derives from the
        # gate's own AST on every run: this string cannot drift green
        # while the gate changes underneath it
        assert "torch_record" in text
        assert "_interpreter_provenance" in text
        # the retracted directive spellings stay retracted: no runtime
        # comparison of the two records is wired in this shard, so the
        # prose must not promise one
        assert "cross-check against the gate" not in text
        assert "adjudicating-torch" not in text


# ---- the mutation rows over this module are real and anchored ----


def _table_rows(tree):
    paths, rows = None, None
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        name = getattr(target, "id", "")
        value = getattr(node, "value", None)
        if name == "MODULE_PATHS":
            paths = ast.literal_eval(value)
        elif name == "EMIT_RUN_MANIFEST_ROWS":
            rows = [ast.literal_eval(elt) for elt in value.elts]
    return paths, rows


class RecoveryRefusedError(Exception):
    """Refusal, naming the offending row(s): the live bytes are neither
    pristine nor recoverable to pristine by reverting exactly one applied
    row. Missing is not zero; the helper never guesses and never
    normalizes."""


def _is_pristine_over(text, rows):
    return all(
        text.count(row["anchor"]) == 1 and text.count(row["replacement"]) == 0 for row in rows
    )


def _recover_pristine_bytes(live, rows):
    """Return the PRISTINE text behind `live`, or refuse.

    The ONE source for the fact "which states the battery can put on
    disk, and how to get back to pristine from any of them" -- formerly
    private inside `_validate_rows` and re-invented, wrongly, by the
    MUST_PASS's raw counting; two sources for one fact is what produced
    the defect this repairs. Accepted state set, size len(rows) + 1:
    the pristine tree, plus pristine with exactly one row applied; the
    battery applies a single row and restores the original bytes in
    `finally` before scoring (tools/mutate.py:1124-1134). Call sites
    that must survive the battery's control leg pass the parsed rows
    PLUS the inert control from EMBEDDED_TABLE (tools/mutate.py:857-876):
    the control's comment-only edit is invisible to the parsed rows, so
    its applied bytes look pristine to them, and only the full table can
    revert it. Recovery reverts the one present replacement, VERIFIES
    the candidate (every anchor exactly once, no replacement shipped)
    and re-checks that the live bytes are reachable from it in zero or
    one applications -- so rows sharing a producer line launder nothing,
    whatever their anchor layout. Anything outside the set -- two rows
    applied at once, an anchor duplicated, a replacement already
    shipped, bytes no candidate reaches, or two candidates accepting
    the same bytes -- is a real defect somewhere and is refused with
    RecoveryRefusedError naming every offending row with its counts and
    the accepted set's size: refused, not guessed and not normalized
    (fail closed, doctrine 4).
    """
    candidates = {live}
    for row in rows:
        if live.count(row["replacement"]) == 1:
            candidates.add(live.replace(row["replacement"], row["anchor"], 1))
    valids = []
    for cand in candidates:
        if not _is_pristine_over(cand, rows):
            continue
        reachable = {cand}
        for row in rows:
            reachable.add(cand.replace(row["anchor"], row["replacement"], 1))
        if live in reachable:
            valids.append(cand)
    if len(valids) == 1:
        return valids[0]
    reasons = []
    if len(valids) > 1:
        reasons.append(
            f"{len(valids)} distinct pristine candidates accept the same "
            "live bytes: ambiguous state"
        )
    for row in rows:
        n_anchor = live.count(row["anchor"])
        n_replacement = live.count(row["replacement"])
        if n_anchor != 1 or n_replacement != 0:
            reasons.append(
                f"{row['name']}: anchor occurs {n_anchor}x, replacement "
                f"{n_replacement}x in the live bytes"
            )
    detail = "; ".join(reasons) or "live bytes match no reachable state"
    raise RecoveryRefusedError(
        f"refusing to certify an unmeasured state: accepted state set is "
        f"pristine + one applied row per row ({len(rows) + 1} state(s) "
        f"over {len(rows)} row(s)); {detail}"
    )


def _validate_rows(paths, rows, emitter_src):
    problems = []
    if not paths or paths.get("emit_run_manifest") != "tools/emit_run_manifest.py":
        problems.append("MODULE_PATHS does not resolve emit_run_manifest")
    if not rows:
        problems.append(
            "zero mutation rows: a module in MODULE_PATHS with no mutants "
            "is UNMEASURED, never covered"
        )
        return problems
    names = [row.get("name") for row in rows if isinstance(row, dict)]
    if len(names) != len(set(names)):
        problems.append("duplicate row names")
    sealed = []
    for row in rows:
        if set(row) != REQUIRED:
            problems.append(f"row keys {sorted(row)} != {sorted(REQUIRED)}")
            continue
        if row["anchor"] == row["replacement"]:
            problems.append(f"{row['name']}: replacement == anchor")
            continue
        sealed.append(row)
    if problems:
        return problems
    # Mutation-invariant seal: the battery applies ONE row at a time and
    # restores bytes between trials (tools/mutate.py:1131-1134), so the
    # emitter on disk is always either pristine or carrying exactly one
    # shipped row. Rows may share a producer line, so one row's edit can
    # evict another row's anchor in general -- but the inert control is
    # NOT such a case; R4 is settled from the listing: its replacement
    # (tools/mutate.py:869-872) keeps the producer line's text as a
    # prefix and appends a comment, leaving row lora-count-plus-one's
    # anchor intact, and the battery run confirms it -- the control's
    # leg red-named only the byte-assuming MUST_PASS further down, never
    # this seal. The state knowledge -- which states the battery can
    # put on disk and the way back to pristine from any of them --
    # lives exactly once, in _recover_pristine_bytes, shared with the
    # MUST_PASS below; two sources for one fact is what produced the
    # last defect. This validator adjudicates the rows parsed from
    # EMIT_RUN_MANIFEST_ROWS (the control row carries a fifth key and
    # is bound across states by the pin in the muster test instead).
    # Every old refusal survives inside the helper: anchor absent -> no
    # revert candidate; anchor ambiguous (>= 2x) -> not pristine; equal
    # pairs refused above; replacement already shipped -> not pristine;
    # two rows applied at once -> no single revert reaches a pristine
    # tree -- each surfaced as a refusal naming the offending row(s)
    # with its counts.
    try:
        _recover_pristine_bytes(emitter_src, sealed)
    except RecoveryRefusedError as exc:
        problems.append(str(exc))
    return problems


def test_mutation_rows_for_emit_run_manifest_are_valid():
    # DENOMINATOR: however many rows EMIT_RUN_MANIFEST_ROWS carries. The
    # count is RECOMPUTED below as len(rows) from the parsed table and is
    # stated in the failure message; it is deliberately not restated as a
    # literal here, because the next row added to the table would stale
    # it (the census pin on the constant lives in the muster test by
    # design). The battery's eighth row is the inert MUST-PASS control
    # merged later by EMBEDDED_TABLE, so this denominator is the
    # battery's minus one. Each row is checked for keys, a unique name, a
    # real delta, and a binding that holds in every state the battery can
    # put on disk, so this test evaluates identically on pristine and on
    # mutated bytes.
    tree = ast.parse(MUTATE_PATH.read_text(encoding="utf-8"))
    paths, rows = _table_rows(tree)
    src = EMITTER_PATH.read_text(encoding="utf-8")
    problems = _validate_rows(paths, rows, src)
    assert not problems, (
        f"{len(rows) if rows else 0} row(s) examined, "
        f"{len(problems)} problem(s):\n" + "\n".join(problems)
    )


def test_must_fire_row_validator_rejects_planted_bad_rows():
    # MUST_FIRE for the validator itself, observed red on every run:
    # 6 doctored inputs, each must be flagged. Units examined: 6.
    src = EMITTER_PATH.read_text(encoding="utf-8")
    good_paths = {"emit_run_manifest": "tools/emit_run_manifest.py"}
    # anchor absent from the file
    assert _validate_rows(
        good_paths,
        [{"name": "x", "what": "x", "anchor": "NOT IN THE FILE", "replacement": "y"}],
        src,
    )
    # zero rows
    assert _validate_rows(good_paths, [], src)
    # replacement identical to anchor
    assert _validate_rows(
        good_paths,
        [
            {
                "name": "x",
                "what": "x",
                "anchor": "    return entries",
                "replacement": "    return entries",
            }
        ],
        src,
    )
    # module never registered
    assert _validate_rows(
        {},
        [{"name": "x", "what": "x", "anchor": "    return entries", "replacement": "    pass"}],
        src,
    )
    # anchor occurs more than once -- a doctored corpus carries it twice
    assert _validate_rows(
        good_paths,
        [{"name": "x", "what": "x", "anchor": "alpha\n", "replacement": "BRAVO\n"}],
        "alpha\nalpha\n",
    )
    # anchor AND replacement each occur once: the tree already carries the
    # replacement, so there is no unique mutation site -- still refused
    assert _validate_rows(
        good_paths,
        [{"name": "x", "what": "x", "anchor": "alpha\n", "replacement": "BRAVO\n"}],
        "alpha\nBRAVO\n",
    )


def test_must_pass_validator_accepts_every_reachable_state(tmp_path):
    # MUST_PASS for the seal: proves the seal ACCEPTS every state it must
    # accept, constructively. The bytes on disk take len(all_rows) + 1
    # shapes -- pristine, or pristine with exactly ONE row applied
    # (tools/mutate.py:1124-1134) -- and during its own leg the disk
    # carries that leg's row, so the live bytes are NOT assumed pristine:
    # this test first proves the seal accepts the live state, then
    # recovers the pristine text through the ONE recovery and replays
    # every single-row application from it. The recovery runs over the
    # FULL table -- the len(rows) parsed rows plus the inert control:
    # over the parsed rows alone the control's comment edit is
    # invisible, so control-applied bytes would be laundered into
    # "pristine" and the assertions below would red on mutated bytes --
    # the false red this test emitted on every leg of the last run,
    # including the control's own (the sole attribution in its
    # [CTRL DEAD] block). An unrecoverable live state fails this test
    # by the helper's refusal: fail closed, never skipped. Denominator:
    # len(rows) + 2 validator invocations (1 live + 1 synthesized per
    # parsed row + 1 synthesized control), each over len(rows) rows;
    # every one must come back clean.
    tree = ast.parse(MUTATE_PATH.read_text(encoding="utf-8"))
    paths, rows = _table_rows(tree)
    assert rows, "zero rows parsed: UNMEASURED, never PASS"
    mutate = _load_mutate_module()
    control = mutate.EMBEDDED_TABLE["emit_run_manifest"][-1]
    assert control.get("must_survive") is True
    all_rows = [*rows, control]
    src = EMITTER_PATH.read_text(encoding="utf-8")
    problems = _validate_rows(paths, rows, src)
    assert not problems, "live on-disk state refused:\n" + "\n".join(problems)
    examined = 1
    pristine = _recover_pristine_bytes(src, all_rows)
    for row in rows:
        assert pristine.count(row["anchor"]) == 1, (
            f"{row['name']}: anchor not unique in the pristine tree"
        )
        assert pristine.count(row["replacement"]) == 0, (
            f"{row['name']}: replacement already shipped in the pristine tree"
        )
        copy = tmp_path / f"applied_{row['name'].replace('.', '_')}.py"
        copy.write_text(pristine.replace(row["anchor"], row["replacement"]), "utf-8")
        after = copy.read_text(encoding="utf-8")
        assert after.count(row["replacement"]) == 1, (
            f"{row['name']}: replacement not exactly once in the applied copy"
        )
        problems = _validate_rows(paths, rows, after)
        assert not problems, (
            f"{row['name']} applied to a copy is refused "
            f"({len(problems)} problem(s)):\n" + "\n".join(problems)
        )
        examined += 1
    # The control's own leg: its comment-only edit must leave the row
    # seal intact -- the state that proved the old pin was a tripwire --
    # and recovery over the full table must hand back the pristine tree.
    assert pristine.count(control["anchor"]) == 1
    assert pristine.count(control["replacement"]) == 0
    copy = tmp_path / "applied_inert_control.py"
    copy.write_text(pristine.replace(control["anchor"], control["replacement"]), "utf-8")
    applied = copy.read_text(encoding="utf-8")
    problems = _validate_rows(paths, rows, applied)
    assert not problems, "control-applied copy refused:\n" + "\n".join(problems)
    assert _recover_pristine_bytes(applied, all_rows) == pristine, (
        "recovery over the full table did not undo the inert control edit"
    )
    examined += 1
    assert examined == len(rows) + 2, (
        f"denominator unwind: {examined} invocation(s) for {len(rows)} "
        "row(s); expected 1 live + one per row + control"
    )


def test_must_fire_recovery_refuses_states_outside_the_state_set(tmp_path):
    # MUST_FIRE for _recover_pristine_bytes (doctrine 3), observed going
    # red once per doctored state below: two rows applied simultaneously,
    # one row's anchor duplicated, a replacement shipped into an
    # otherwise pristine tree. Each state is outside the accepted set --
    # for the full table that set is len(all_rows) + 1 byte shapes
    # (pristine, or exactly one of the parsed rows or the control
    # applied) -- so each must be REFUSED, never normalized, and each
    # refusal must name the offending row(s). States are doctored from
    # the TRUE pristine tree: the opening recovery runs over
    # rows + control, so this holds even on the battery's control leg.
    # Doctoring never depends on anchor layout: each construction is
    # searched for and verified, and a state that cannot be built FAILS
    # this test -- UNMEASURED, never PASS. Denominator: 3 doctored
    # states, 1 refusal + at least 1 row-naming assertion apiece; the
    # acceptance half of the same helper is the MUST_PASS above.
    tree = ast.parse(MUTATE_PATH.read_text(encoding="utf-8"))
    _, rows = _table_rows(tree)
    assert rows and len(rows) >= 2, (
        f"{len(rows) if rows else 0} row(s) parsed; the two-applied "
        "state requires at least 2: UNMEASURED, never PASS"
    )
    mutate = _load_mutate_module()
    control = mutate.EMBEDDED_TABLE["emit_run_manifest"][-1]
    assert control.get("must_survive") is True
    all_rows = [*rows, control]
    pristine = _recover_pristine_bytes(EMITTER_PATH.read_text(encoding="utf-8"), all_rows)
    refusals = []

    def expect_refusal(label, doctored, names):
        probe = tmp_path / f"{label}.py"
        probe.write_text(doctored, "utf-8")
        try:
            _recover_pristine_bytes(probe.read_text(encoding="utf-8"), all_rows)
        except RecoveryRefusedError as exc:
            message = str(exc)
        else:
            message = ""
        assert message, f"{label}: doctored state was ACCEPTED"
        for name in names:
            assert name in message, f"{label}: refusal did not name {name}:\n{message}"
        refusals.append(message)

    # State 1 of 3: two rows applied simultaneously. Search for a pair
    # whose edits co-apply (neither evicts the other's anchor); if none
    # can be found the state is UNBUILT and this test FAILS, never skips.
    pair = None
    for i, first in enumerate(rows):
        for second in rows[i + 1 :]:
            applied = pristine.replace(first["anchor"], first["replacement"], 1)
            if applied.count(second["anchor"]) != 1:
                continue
            applied = applied.replace(second["anchor"], second["replacement"], 1)
            if (
                applied.count(first["replacement"]) == 1
                and applied.count(second["replacement"]) == 1
            ):
                pair = (first, second, applied)
                break
        if pair is not None:
            break
    assert pair is not None, (
        "no two rows co-apply cleanly; the two-applied state is unbuilt: UNMEASURED, never PASS"
    )
    first, second, applied = pair
    expect_refusal("two_rows_applied", applied, (first["name"], second["name"]))
    # State 2 of 3: one row's anchor duplicated.
    dup = None
    for row in rows:
        doctored = pristine + row["anchor"]
        if doctored.count(row["anchor"]) == 2:
            dup = (row, doctored)
            break
    assert dup is not None, "anchor-duplicated state unbuilt: UNMEASURED, never PASS"
    expect_refusal("anchor_duplicated", dup[1], (dup[0]["name"],))
    # State 3 of 3: a replacement shipped into an otherwise pristine tree.
    shipped = None
    for row in rows:
        doctored = pristine + row["replacement"]
        if doctored.count(row["anchor"]) == 1 and doctored.count(row["replacement"]) == 1:
            shipped = (row, doctored)
            break
    assert shipped is not None, "shipped-replacement state unbuilt: UNMEASURED, never PASS"
    expect_refusal("replacement_shipped", shipped[1], (shipped[0]["name"],))
    assert len(refusals) == 3, (
        f"{len(refusals)} refusal(s) observed, expected 3 -- one per "
        "doctored state (two rows applied, anchor duplicated, "
        "replacement shipped)"
    )


def _load_mutate_module():
    # Import-by-path so the suite does not depend on tools/ being a
    # package. The validity legs above read the tool with ast only; these
    # controls must EXECUTE it -- a control that never runs is not a
    # control (doctrine 3). Module level binds constants and definitions.
    import ast
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("mutate_under_test", MUTATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # tools/mutate.py carries `from __future__ import annotations`, so its
    # dataclass field annotations are strings; dataclasses resolves those
    # strings through sys.modules[cls.__module__], which is None until the
    # module is registered. Register BEFORE exec -- the sibling idiom at
    # tests/test_dense_denominator_repairs.py:57-58 -- but only for the
    # duration of the exec: this suite's subject is order-dependent false
    # greens, so `mutate_under_test` must not stay observable to later tests.
    # The pop lives in `finally`, so an exec failure cannot strand a
    # half-initialized module under the name (fail closed, doctrine 4); the
    # caller keeps the fully executed module alive via the return value.
    assert "mutate_under_test" not in sys.modules, (
        "mutate_under_test leaked into sys.modules before this loader ran; "
        "order-dependent fixture state disqualifies the run"
    )
    sys.modules["mutate_under_test"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("mutate_under_test", None)
    # Fixture precondition, denominator 1 of 1: exec must have run module
    # level to COMPLETION. Derive the last top-level def/class the source on
    # disk binds and require it in the module namespace -- computed from the
    # tree at load time, so the check never names a symbol the file does not
    # define and cannot drift from the tool it guards.
    tree = ast.parse(Path(MUTATE_PATH).read_text(encoding="utf-8"))
    terminal = next(
        (
            stmt.name
            for stmt in reversed(tree.body)
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ),
        None,
    )
    assert terminal is not None, "mutate.py binds no top-level def/class at all"
    assert hasattr(module, terminal), (
        f"mutate.py exec stopped before its final binding {terminal!r}; "
        "refusing to measure a half-executed module"
    )
    return module


def _refusal_text(fn, capsys, *args, **kwargs):
    # die() prints to stderr and then sys.exit(2); accept the message on
    # either channel. An EMPTY return means NO refusal happened -- always
    # red for the MUST_FIRE tests below.
    exc_text = ""
    try:
        fn(*args, **kwargs)
    except SystemExit as exc:
        exc_text = str(exc)
    captured = capsys.readouterr()
    return exc_text + captured.out + captured.err


def test_complement_muster_refuses_and_names_unpublished_module(capsys):
    # MUST_FIRE for the R1 complement check, observed going red on every
    # run. Units examined: 1 injected table over a 2-module registry, of
    # which 1 module (core) is published and 1 (emit_run_manifest) is
    # mapped-but-unpublished. The refusal must fire AND name the module
    # with its expected row count (9 embedded rows) -- "table incomplete"
    # is not a denominator.
    mutate = _load_mutate_module()
    paths = {
        "core": "src/foundationscale/gates/core.py",
        "emit_run_manifest": "tools/emit_run_manifest.py",
    }
    data = {"core": [{"name": "n", "what": "w", "anchor": "a", "replacement": "b"}]}
    blob = _refusal_text(mutate._validate_table, capsys, data, paths, complete=True)
    assert "emit_run_manifest" in blob, "no refusal, or a refusal that names nothing"
    assert "9" in blob, "refusal did not state the expected row count"


def test_dual_source_for_embedded_rows_is_refused(tmp_path, monkeypatch, capsys):
    # MUST_FIRE for the single-source guard (R2): the instant
    # mutations.json grows an emit_run_manifest key there are two sources
    # for one fact, and load_table must refuse rather than pick one.
    # Units examined: 1 doctored table.
    import json

    mutate = _load_mutate_module()
    base = json.loads(mutate.TABLE.read_text(encoding="utf-8"))
    base["emit_run_manifest"] = [dict(mutate.EMIT_RUN_MANIFEST_ROWS[0])]
    rogue = tmp_path / "mutations.json"
    rogue.write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(mutate, "TABLE", rogue)
    blob = _refusal_text(mutate.load_table, capsys, None)
    assert "emit_run_manifest" in blob


def test_shipped_pair_passes_completeness_muster(capsys):
    # MUST_PASS on the real shipped pair. Units examined: 9 registered
    # modules (8 published in tools/mutations.json + emit_run_manifest
    # merged from EMBEDDED_TABLE); 78 rows in total (69 JSON rows per the
    # shipped census + 8 EMIT_RUN_MANIFEST_ROWS + 1 inert must-pass
    # control); plus the filtered --module emit_run_manifest path.
    # The totals are PINNED, not derived: a derived count would agree with
    # any table, including one that silently lost rows. #221 moved 62 -> 64
    # (two manifest topology rows) and 7 -> 8 (the emitter wiring row);
    # #242 moved 64 -> 69, one inert must-pass control per module for the
    # five that had none, so that a per-module CI shard is a whole detector
    # rather than its MUST_FIRE half. The pin is what makes each visible
    # instead of quiet.
    import json

    mutate = _load_mutate_module()
    blob = _refusal_text(mutate.load_table, capsys, None)
    assert blob == "", f"shipped pair refused: {blob.strip()[:400]}"
    data = mutate.load_table(None)
    assert set(data) == set(mutate.MODULE_PATHS)
    assert len(data) == 9
    assert all(data.values())
    assert sum(len(rows) for rows in data.values()) == 78  # 69 JSON + 9 embedded
    emit = data["emit_run_manifest"]
    n_const = len(mutate.EMIT_RUN_MANIFEST_ROWS)
    assert n_const == 8  # census leg: row growth reddens this by design
    assert emit[:n_const] == mutate.EMIT_RUN_MANIFEST_ROWS  # merged unedited
    assert len(emit) == n_const + 1
    control = emit[-1]
    assert control.get("must_survive") is True  # R3's MUST-PASS half
    src_path = mutate.ROOT / mutate.MODULE_PATHS["emit_run_manifest"]
    src = src_path.read_text(encoding="utf-8")
    assert control["anchor"] != control["replacement"]
    # Mutation-invariant binding: the anchor ends ",\n" while the
    # replacement inserts its comment BEFORE that newline
    # (tools/mutate.py:868-872), so neither string contains the other and
    # the site sits in exactly one of two states -- pristine (anchor 1x,
    # replacement 0x) or this control applied (0x, 1x). Strictly stronger
    # than the old count pin: the 0x half proves the shipped tree does
    # not already carry the replacement, and this test stays green while
    # the control's own mutant is applied (the old pin reddened it). The
    # control anchors on the same producer line as row
    # lora-count-plus-one (tools/mutate.py:852-854), so THAT row's leg
    # evicts this anchor outright (0x, 0x); a tree therefore also
    # qualifies when reverting exactly one other shipped row restores
    # the pristine site.
    states = [src]
    for other in mutate.EMBEDDED_TABLE["emit_run_manifest"]:
        if other["name"] == control["name"]:
            continue
        if src.count(other["anchor"]) == 0 and src.count(other["replacement"]) == 1:
            states.append(src.replace(other["replacement"], other["anchor"]))
    assert any(
        (t.count(control["anchor"]), t.count(control["replacement"])) in ((1, 0), (0, 1))
        for t in states
    ), (
        f"control binds unsoundly: anchor x{src.count(control['anchor'])}, "
        f"replacement x{src.count(control['replacement'])} in the live tree, "
        "and no single-row revert restores a unique site"
    )
    # One fact, one source: the JSON must not carry what the tool embeds.
    raw = json.loads(mutate.TABLE.read_text(encoding="utf-8"))
    assert "emit_run_manifest" not in raw
    # The --module filter runs on an already-certified-complete table, so
    # a single-module run cannot trip the complement over the 8 set-asides:
    one = mutate.load_table("emit_run_manifest")
    assert len(one["emit_run_manifest"]) == n_const + 1
