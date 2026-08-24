"""Executable coverage for the --lora emitter path (#79's missing evidence).

#79 shipped the LoRA abstention record with the safety argument "the existing
889-test suite exercises the --lora path, so a serialization surprise is
caught by the merge run itself". Measured afterwards: no test invoked
emit.main with --lora AT ALL (grep denominator: ZERO), so the _SOURCE_RE
violation in _LORA_ABSTENTION_SOURCE would have surfaced for the first time
as an rc-1 refusal at a real GB200 launch, exactly as _TRAINING_STACK_SOURCE
did in CI. This module is the repair of that zero-denominator claim. Doctrine
commitments baked in:

* Every claim reads the bytes that reached DISK (stdout was never the
  artifact); persisted attempts are counted (exactly 1 of 1 on a fresh run).
* The five record keys are re-enumerated HERE, deliberately NOT imported from
  the emitter's _LORA_ABSTENTION_RECORD_KEYS: a control zipping over the same
  tuple as the producer verifies nothing (drop-one in BOTH enumerators would
  read green). Two independent enumerators or it is not a control.
* MUST_FIRE pins NO single rc -- which arm fires is saves-dependent BY DESIGN
  (#79's own note); it pins rc != 0 AND the drill token in combined output.
* KNOWN ABSTENTION, named, not silently skipped: the drop-one leg SHOULD also
  run through the public check_saved_run_declaration, but that function
  appears in NONE of the context given to this shard (no module, no
  signature), and a guessed call site is defect 2 repeating itself -- a false
  alarm priced like a false green. The drop-one below therefore drives all
  five single-field drops through the serialized-bytes oracle
  _abstention_markers_absent (signature IS in context) on REAL emitted bytes,
  and the drill test proves fire is wired to hard refusal end-to-end. The
  follow-up shard needs exactly one thing in context:
  check_saved_run_declaration's module and call signature; then hoist the
  five variants this file constructs.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest
from test_dense_denominator_repairs import _emit_args, _load_tool

_DRILL_ENV = "FS_EMIT_DRILL_BARE_NULL"
_DRILL_TOKEN = "DRILL FIRED (FS_EMIT_DRILL_BARE_NULL=1)"

# Independent second enumerator, pinned to the record order named in #79 --
# see the module docstring for why this is NOT imported from the emitter.
_ABSTENTION_KEYS = (
    "declared.status",
    "declared.abstained_by",
    "declared.abstention_reason",
    "declared.superseded_by",
    "declared.preexisting_iter_dirs",
)


def _lora_emit_args(run_id: str, out_dir: Path, ckpt_dir: Path) -> list[str]:
    """Reuse the dense helper, then swap its full-ft tail for bare --lora.

    The LoRA launcher passes NEITHER --base-checkpoint NOR --hf-config BY
    DESIGN -- that absence IS the abstention under test, so no safetensors
    fixture is written here at all (writing one would test a different run).
    The tail assertion makes drift in the reused helper LOUD, never a
    silently-mangled argv.
    """
    base = out_dir / "absent-by-design-base"
    config = out_dir / "absent-by-design-config.json"
    args = _emit_args(run_id, out_dir, ckpt_dir, base, config)
    assert args[-5:] == [
        "--full-ft",
        "--base-checkpoint",
        str(base),
        "--hf-config",
        str(config),
    ]
    return args[:-5] + ["--lora"]


def _run_lora(tmp_path: Path, run_id: str) -> tuple[int, Path]:
    """Load the tool fresh and run ONE --lora emission; return (rc, out_dir)."""
    emit = _load_tool("emit_run_manifest")
    out_dir = tmp_path / "run-root"
    out_dir.mkdir()
    rc = emit.main(_lora_emit_args(run_id, out_dir, out_dir / "ckpts"))
    return rc, out_dir


def _disk_manifest_text(out_dir: Path) -> str:
    """The bytes that reached disk, with the attempt denominator attached."""
    attempts = sorted((out_dir / "provenance").rglob("attempt-*.json"))
    assert len(attempts) == 1, (
        f"a fresh --lora emission persists exactly 1 attempt manifest; found "
        f"{len(attempts)}: {[str(a) for a in attempts]}"
    )
    return attempts[0].read_text(encoding="utf-8")


def _value_of(entry: object) -> object:
    """Unwrap {value, source, findings} serialization, or return a raw scalar.

    The serde SHAPE is deliberately not pinned by these tests -- presence,
    value, and (when serialized as an object) source are what the record
    owes; storage layout is ManifestStore's business.
    """
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _source_of(entry: object) -> str | None:
    if isinstance(entry, dict):
        source = entry.get("source")
        return source if isinstance(source, str) else None
    return None


def _config_container(node: object) -> dict[str, object] | None:
    """Find the serialized block that holds the declared.* / training_stack.*
    records, wherever the manifest schema nests it."""
    if isinstance(node, dict):
        if any(key in node for key in _ABSTENTION_KEYS) or any(
            key.startswith("training_stack.") for key in node
        ):
            return node
        for value in node.values():
            hit = _config_container(value)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for item in node:
            hit = _config_container(item)
            if hit is not None:
                return hit
    return None


def test_lora_emission_persists_abstention_record_five_of_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST_PASS: plain --lora returns rc 0 AND persists a 5-of-5 record."""
    monkeypatch.delenv(_DRILL_ENV, raising=False)
    rc, out_dir = _run_lora(tmp_path, "lora-null")
    captured = capsys.readouterr()
    assert rc == 0, (
        f"plain --lora emission REFUSED (rc={rc}); captured output:\n"
        f"{captured.out}{captured.err}"
    )

    text = _disk_manifest_text(out_dir)
    # 5 of 5, with the five spelled out and the missing set named on failure.
    present = [key for key in _ABSTENTION_KEYS if f'"{key}"' in text]
    assert present == list(_ABSTENTION_KEYS), (
        f"abstention record is {len(present)} of 5 on disk, missing: "
        f"{sorted(set(_ABSTENTION_KEYS) - set(present))}"
    )

    container = _config_container(json.loads(text))
    assert container is not None, "no serialized block holds the declared.* record"
    assert _value_of(container["declared.status"]) == "abstained"
    assert _value_of(container["declared.preexisting_iter_dirs"]) == "0"
    who = str(_value_of(container["declared.abstained_by"]))
    assert "launch_g4e4b_lora_1tray.sh" in who

    # Defect 1's repair, asserted on the persisted bytes: when the serde
    # carries sources, every record entry must carry the new closed class --
    # a relabelled 'default'/'cli' here is the provenance lie #83 prevents.
    for key in _ABSTENTION_KEYS:
        source = _source_of(container[key])
        if source is not None:
            assert source == "measured:lora-abstention", (
                f"{key} persisted with source {source!r}; a measurement "
                f"relabeled 'default' would be an in-band provenance lie"
            )

    # The emitter's own serialized oracle agrees on the honest baseline.
    emit = _load_tool("emit_run_manifest")
    assert emit._abstention_markers_absent(text) == []


def test_lora_drill_bare_null_refuses_with_drill_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST_FIRE: armed drill refuses, with its token. rc is NOT pinned --
    which arm fires is saves-dependent BY DESIGN (#79's note)."""
    monkeypatch.setenv(_DRILL_ENV, "1")
    rc, _out_dir = _run_lora(tmp_path, "lora-drill")
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert rc != 0, (
        "armed FS_EMIT_DRILL_BARE_NULL=1 drill returned rc 0 -- the on-disk "
        "verifier CANNOT fire, so every green it ever reported is unproven "
        "(#56: a drill the run survives is the control's failure, loudly)"
    )
    assert _DRILL_TOKEN in combined, (
        f"drill refused WITHOUT its token {_DRILL_TOKEN!r} -- an unattributed "
        f"fire is not an auditable control; combined output was:\n{combined}"
    )


def test_lora_abstention_record_drop_one_detected_all_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drop-one on the serialized record: 5 of 5 drops fire, 0 pass silently.

    Named-substitute for the check_saved_run_declaration leg -- see the
    module docstring for that abstention and what the follow-up shard needs.
    A partial record is not a record.
    """
    monkeypatch.delenv(_DRILL_ENV, raising=False)
    rc, out_dir = _run_lora(tmp_path, "lora-drop-one")
    capsys.readouterr()
    assert rc == 0
    emit = _load_tool("emit_run_manifest")
    text = _disk_manifest_text(out_dir)
    assert emit._abstention_markers_absent(text) == [], (
        "baseline broken: the un-dropped record must report ZERO missing markers"
    )
    for key in _ABSTENTION_KEYS:
        needle = f'"{key}":'
        assert text.count(needle) == 1, (
            f"drop-one construction needs exactly 1 on-disk occurrence of "
            f"{needle!r}, found {text.count(needle)} -- refusing to guess which"
        )
        variant = text.replace(needle, '"control.dropped_key":', 1)
        missing = emit._abstention_markers_absent(variant)
        assert len(missing) == 1 and any(key in str(m) for m in missing), (
            f"dropping {key} must flag exactly that one field (1 of 5); "
            f"oracle reported {missing!r}"
        )


def test_lora_training_stack_records_are_first_class_honest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#83 leg: torch record is measured-in-emitter OR stated-ABSTAINED.

    "Torch is present" is deliberately NOT asserted: on a machine without an
    importable torch that is a dead control. Both an in-emitter measurement
    and a stated ABSTAINED are first-class honest states; a guessed version
    string is the only failure this leg exists to catch. python_executable /
    python_version are pinned to THIS interpreter because _load_tool runs the
    emitter in-process, so equality is deterministic, not hopeful.
    """
    monkeypatch.delenv(_DRILL_ENV, raising=False)
    rc, out_dir = _run_lora(tmp_path, "lora-torch")
    capsys.readouterr()
    assert rc == 0
    container = _config_container(json.loads(_disk_manifest_text(out_dir)))
    assert container is not None

    torch_record = str(_value_of(container["training_stack.torch_record"]))
    assert torch_record.startswith(("measured in-emitter", "ABSTAINED:")), (
        f"torch_record is neither an in-emitter measurement nor a stated "
        f"abstention -- a guessed version string is the #83 lie: {torch_record!r}"
    )
    assert _value_of(container["training_stack.python_executable"]) == (
        sys.executable or "<unset>"
    )
    assert _value_of(container["training_stack.python_version"]) == (
        platform.python_version()
    )
    source = _source_of(container["training_stack.torch_record"])
    if source is not None:
        assert source == "measured:training-stack"
