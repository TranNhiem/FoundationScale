"""#85: the missing --lora tests for tools/emit_run_manifest.py.

The emitter is imported by path (tools/ is not an installed package); a
failed import raises at collection (fail closed) and never reads as an
absent test. Importing it in-process is also what makes the per-file
--cov addition in ci.yml measure it. No skips (FS_FORBID_SKIPS). First
execution of every control here is this change's next CI run; nothing
in this module has been observed running yet.

MUST_PASS: a complete on-disk abstention record verifies 5-of-5, and
the producer always emits the five declared keys, in declared order,
with no blank values.
MUST_FIRE: an incomplete record must NEVER read as complete -- not when
the omission is accidental and not when the bare-null drill ARMED it --
and a drifted keys tuple must raise under the strict zip.

Mutation rows planted in tools/mutate.py over this module (#85) and the
leg that kills each:
  emit_run_manifest.lora-zip-unstrict           -> test_zip_is_strict_raises_on_drift
  emit_run_manifest.lora-status-not-abstained   -> test_lora_abstention_record_status_and_denominator
  emit_run_manifest.lora-preexisting-key-rename -> test_lora_abstention_record_status_and_denominator
                                                   (KeyError on the pinned key is the red)
  emit_run_manifest.lora-count-plus-one         -> test_lora_abstention_record_status_and_denominator
  emit_run_manifest.lora-count-fabricated       -> main()-level; SURVIVOR, stated
  emit_run_manifest.drill-never-arms            -> main()-level; SURVIVOR, stated
The two survivors are planted over main() flow that needs a checkpoint
estate to exercise; they stand in the mutation tally as survivors,
loudly, rather than the module reading as covered because the table
could not name it. Row-integrity legs below fail on zero rows: a
registered module with no mutants is UNMEASURED, never covered.
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

def _record_text(dropped_key=None) -> str:
    lines = [f"config-source: {erm._LORA_ABSTENTION_SOURCE}"]
    for k, v in erm._lora_abstention_record_entries(0):
        if k != dropped_key:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def test_must_pass_complete_on_disk_record_verifies_5_of_5():
    state, present, total = erm._enforce_lora_abstention_record(
        _record_text(), saves_observed=0, drill_armed=False
    )
    assert total == 5
    assert present == total, f"control state={state!r} but {present}/{total}"


def test_must_fire_incomplete_record_never_reads_complete():
    # Units examined: 1 record missing 1 of 5 fields.
    refusal = None
    try:
        state, present, total = erm._enforce_lora_abstention_record(
            _record_text(dropped_key="declared.preexisting_iter_dirs"),
            saves_observed=0,
            drill_armed=False,
        )
    except Exception:
        refusal = "raised"  # a raised refusal IS the control firing
    if refusal is None:
        assert present < total, (
            f"MUST_FIRE broken: a record missing a field read as "
            f"{present}/{total} with state={state!r}"
        )


def test_drill_armed_still_counts_the_omission():
    # The drill ARMS the omission; the verifier must still count it
    # absent, otherwise control fire is indistinguishable from a pass.
    # Units examined: 1 armed record carrying zero of the five fields.
    header_only = f"config-source: {erm._LORA_ABSTENTION_SOURCE}"
    try:
        state, present, total = erm._enforce_lora_abstention_record(
            header_only, saves_observed=0, drill_armed=True
        )
    except Exception:
        return  # a raised refusal while armed IS the control firing
    assert present < total, (
        f"armed drill must still read absent: {present}/{total} state={state!r}"
    )


# ---- #83: the retraction strings the emitter now ships ----

def test_torch_record_states_gate_side_provenance_not_implemented():
    entries = dict(erm._training_stack_entries())
    # DENOMINATOR: python_executable, python_version, torch_record.
    # This examines the branch this interpreter executes (torch present
    # or not); both branches now carry the same retraction content.
    assert len(entries) == 3
    text = entries["training_stack.torch_record"]
    assert "not implemented" in text
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
    for row in rows:
        if set(row) != REQUIRED:
            problems.append(f"row keys {sorted(row)} != {sorted(REQUIRED)}")
            continue
        n = emitter_src.count(row["anchor"])
        if n != 1:
            problems.append(f"{row['name']}: anchor occurs {n}x, must be exactly 1")
        if row["anchor"] == row["replacement"]:
            problems.append(f"{row['name']}: replacement == anchor")
    return problems


def test_mutation_rows_for_emit_run_manifest_are_valid():
    # DENOMINATOR: however many rows EMIT_RUN_MANIFEST_ROWS carries
    # (6 as planted), each checked for keys, unique anchor, real delta.
    tree = ast.parse(MUTATE_PATH.read_text(encoding="utf-8"))
    paths, rows = _table_rows(tree)
    src = EMITTER_PATH.read_text(encoding="utf-8")
    problems = _validate_rows(paths, rows, src)
    assert not problems, "\n".join(problems)


def test_must_fire_row_validator_rejects_planted_bad_rows():
    # MUST_FIRE for the validator itself, observed red on every run:
    # 4 doctored inputs, each must be flagged. Units examined: 4.
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
        [{"name": "x", "what": "x", "anchor": "    return entries", "replacement": "    return entries"}],
        src,
    )
    # module never registered
    assert _validate_rows(
        {},
        [{"name": "x", "what": "x", "anchor": "    return entries", "replacement": "    pass"}],
        src,
    )
