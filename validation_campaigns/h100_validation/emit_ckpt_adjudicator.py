#!/usr/bin/env python3
"""Build stage for the checkpoint plane: emit the reference adjudicator and its tests.

The launcher requires an operator-supplied checkpoint adjudicator but the framework shipped
none, so a plane either could not launch or measured whatever an operator happened to invent
under the same label. Both artifacts are generated here instead: hand-editing a supposedly
reference gate is precisely the drift this build exists to prevent.

The adjudicator is deliberately structural. It verifies the writer's own collective
manifest against the rank files on disk, without importing a tensor library, and it uses
the launcher's exit vocabulary so an abstention and a measured refusal remain distinct in
launch logs even though the launcher must fail closed on either one.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from fs_estate_pat import estate_blocklist, estate_ident_pat


BLOCKLIST_PATTERN = estate_blocklist(strict_token=True)

ADJUDICATOR_TEMPLATE = '''
#!/usr/bin/env python3
"""Reference structural adjudicator for FoundationScale rank-local checkpoints.

The writer's manifest is collective evidence: rank_payload_count was produced by an
all_reduce after every rank attempted its save. Comparing that fact with the expected
world size catches a run that continued after one or more ranks silently failed to write.

This module intentionally knows nothing about model architectures or tensor libraries.
Format knowledge is isolated behind FORMAT_READERS, so supporting another manifest format
means adding one reader and one registry entry, not teaching the driver about shards.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Pattern, Sequence

EXIT_PASS = 0
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

# The launcher refuses on every non-zero return. EXIT_UNMEASURED is therefore operationally
# fail-closed too, but it must remain a separate code: launch logs need to distinguish "this
# checkpoint failed a measurement" from "this artifact does not have a layout the gate can
# measure", because the remedies and regression histories for those states are different.

VERDICT_PASS = "PASS"
VERDICT_UNMEASURED = "UNMEASURED"
VERDICT_REFUSE = "REFUSE"

CHECK_PASS = "PASS"
CHECK_FAIL = "FAIL"
CHECK_ABSTAIN = "ABSTAIN"

# The vocabulary is a CHAIN of recognised tokens, not a single one. Finding #150: this
# framework's own writer emits `checkpoint-step-00000010`, and a single-token pattern read
# "checkpoint", then "-", then a tail of "step-00000010" that is not all digits, so
# fullmatch returned None on 2/2 of the shapes the writer actually produces. Leg A7b --
# the only leg that cross-validates the directory name against the manifest -- therefore
# ABSTAINED on every real checkpoint, and because ABSTAIN is correctly distinguished from
# FAIL the verdict still reached 0 and nothing surfaced the hole.
#
# The chain is deliberately closed over a fixed token set rather than made permissive
# (e.g. `.*?(\\d+)`): a parser that accepts anything ending in digits would satisfy the
# agreement gate while giving A7b nothing to cross-validate. `gate_ckpt_naming_agreement.py`
# now renders the writer's real formats and asserts this parser accepts them, so the two
# artifacts cannot drift apart again silently.
_STEP_TOKEN = r"(?:step|checkpoint|ckpt|iter|snapshot)"
_STEP_NAME_RE: Pattern[str] = re.compile(
    rf"{_STEP_TOKEN}(?:[_-]{_STEP_TOKEN})*[_-](?P<step>\\d+)", re.IGNORECASE
)


@dataclass(frozen=True)
class Check:
    """One reported leg, together with the evidence denominator for that leg."""

    label: str
    status: str
    note: str


def _strict_int(value: object) -> bool:
    # bool subclasses int in Python. Accepting it here would let JSON true masquerade as a
    # valid world size or step and turn a writer bug into a green structural check.
    return type(value) is int


def _directory_step(name: str) -> int | None:
    match = _STEP_NAME_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group("step"))


def _emit_check(check: Check) -> None:
    note = f" ({check.note})" if check.note else ""
    print(f"  {check.label}: {check.status}{note}")


def _finish(
    checks: list[Check],
    exit_code: int,
    *,
    checkpoint_dir: Path,
    phase: str,
    out_dir: str,
) -> int:
    if not checks:
        # No leg must ever be reported as an implied pass: zero measured legs is the declared
        # abstention state, not success.
        exit_code = EXIT_UNMEASURED

    passed = sum(check.status == CHECK_PASS for check in checks)
    failed = sum(check.status == CHECK_FAIL for check in checks)
    abstained = sum(check.status == CHECK_ABSTAIN for check in checks)
    measured = passed + failed
    total = len(checks)

    if exit_code == EXIT_PASS and (measured == 0 or failed != 0):
        # This is an internal fail-closed backstop. The normal decision path has already
        # classified the result, but a bad caller must not be able to launder a red leg or
        # an all-abstain result into success.
        exit_code = EXIT_UNMEASURED if measured == 0 else EXIT_REFUSE

    if exit_code == EXIT_PASS:
        verdict = VERDICT_PASS
    elif exit_code == EXIT_UNMEASURED:
        verdict = VERDICT_UNMEASURED
    else:
        verdict = VERDICT_REFUSE

    for check in checks:
        _emit_check(check)
    print(
        f"VERDICT {exit_code} {verdict} "
        f"checkpoint_dir={checkpoint_dir} phase={phase!r} out_dir={out_dir!r} "
        f"checks_measured={measured} checks_green={passed} checks_red={failed} "
        f"legs_abstained={abstained} checks_total={total}"
    )
    return exit_code


@dataclass(frozen=True)
class RankLocalShardedV1Reader:
    """Structural reader for writer format ``rank-local-sharded-v1``.

    Shards are checked only for existence, regular-file status and non-zero size.
    Importing torch to open them would make this generic launcher gate unrunnable on hosts
    that lack one tensor library and would silently couple the adjudicator's contract to a
    particular model stack. A format that needs payload decoding can register its own
    reader without changing the driver below.
    """

    FORMAT: ClassVar[str] = "rank-local-sharded-v1"
    SHARD_NAME: ClassVar[Pattern[str]] = re.compile(r"rank-(?P<rank>\\d{5})\\.pt")

    def validate(self, checkpoint_dir: Path, manifest: dict[str, object]) -> list[Check]:
        checks: list[Check] = []

        world_size = manifest.get("world_size")
        rank_payload_count = manifest.get("rank_payload_count")
        expected_payload_count = manifest.get("expected_rank_payload_count")

        strict_integer_count = sum(
            _strict_int(value)
            for value in (world_size, rank_payload_count, expected_payload_count)
        )
        positive_world = _strict_int(world_size) and world_size > 0
        agreement_count = sum(
            (
                rank_payload_count == expected_payload_count,
                rank_payload_count == world_size,
            )
        )
        counts_hold = (
            strict_integer_count == 3
            and positive_world
            and agreement_count == 2
        )
        checks.append(
            Check(
                "A4 manifest counts agree with a positive world_size",
                CHECK_PASS if counts_hold else CHECK_FAIL,
                "strict integer fields "
                f"{strict_integer_count}/3; payload-count comparisons {agreement_count}/2; "
                f"world_size={world_size!r}, rank_payload_count={rank_payload_count!r}, "
                f"expected_rank_payload_count={expected_payload_count!r}",
            )
        )
        if not counts_hold:
            # Without a valid positive world_size there is no valid expected rank set to
            # enumerate. Guessing one would produce a check claim broader than the evidence.
            return checks
        assert isinstance(world_size, int)

        try:
            entries = list(checkpoint_dir.iterdir())
        except OSError as exc:
            checks.append(
                Check(
                    "A5 rank shard listing is readable",
                    CHECK_FAIL,
                    f"0/1 checkpoint directories listed: {exc}",
                )
            )
            return checks

        by_index: dict[int, str] = {}
        malformed_rank_names: list[str] = []
        for entry in entries:
            name = entry.name
            if not (name.startswith("rank-") and name.endswith(".pt")):
                continue
            match = self.SHARD_NAME.fullmatch(name)
            if match is None:
                malformed_rank_names.append(name)
                continue
            by_index[int(match.group("rank"))] = name

        expected_indices = set(range(world_size))
        observed_indices = set(by_index)
        missing = sorted(expected_indices - observed_indices)
        extra_names = sorted(
            by_index[index] for index in sorted(observed_indices - expected_indices)
        )
        # A malformed rank-looking name is not ignored: treating a stray payload as an
        # unrelated file would let a writer or copy accident hide behind a naming error.
        extra_names.extend(f"malformed:{name}" for name in sorted(malformed_rank_names))

        present = len(expected_indices & observed_indices)
        checks.append(
            Check(
                "A5a expected rank shard set is complete",
                CHECK_PASS if not missing else CHECK_FAIL,
                f"present {present}/{world_size} expected ranks; missing={missing}",
            )
        )
        rank_named_total = len(by_index) + len(malformed_rank_names)
        checks.append(
            Check(
                "A5b no unexpected rank shards are present",
                CHECK_PASS if not extra_names else CHECK_FAIL,
                f"{rank_named_total} rank-named entries inspected; unexpected={extra_names}",
            )
        )

        good_shards = 0
        bad_shards: list[str] = []
        for index in range(world_size):
            shard = checkpoint_dir / f"rank-{index:05d}.pt"
            if not shard.exists():
                bad_shards.append(f"{shard.name}: missing")
                continue
            try:
                if not shard.is_file():
                    bad_shards.append(f"{shard.name}: not a regular file")
                    continue
                size = shard.stat().st_size
            except OSError as exc:
                bad_shards.append(f"{shard.name}: unreadable ({exc})")
                continue
            if size <= 0:
                bad_shards.append(f"{shard.name}: zero-length")
                continue
            good_shards += 1

        checks.append(
            Check(
                "A6 every expected rank shard is a non-empty regular file",
                CHECK_PASS if good_shards == world_size else CHECK_FAIL,
                f"regular non-empty shards {good_shards}/{world_size}; bad={bad_shards}",
            )
        )

        global_step = manifest.get("global_step")
        valid_step = _strict_int(global_step) and global_step >= 0
        checks.append(
            Check(
                "A7a manifest global_step is a non-negative integer",
                CHECK_PASS if valid_step else CHECK_FAIL,
                f"global_step={global_step!r}; non-negative integer legs "
                f"{1 if valid_step else 0}/1",
            )
        )

        encoded_step = _directory_step(checkpoint_dir.name)
        if encoded_step is None:
            checks.append(
                Check(
                    "A7b directory step agrees with manifest global_step",
                    CHECK_ABSTAIN,
                    f"0/1 step values encoded in directory name {checkpoint_dir.name!r}; "
                    "this leg was not measured",
                )
            )
        elif not valid_step:
            checks.append(
                Check(
                    "A7b directory step agrees with manifest global_step",
                    CHECK_ABSTAIN,
                    "comparison not measured because A7a rejected the manifest value",
                )
            )
        else:
            assert isinstance(global_step, int)
            checks.append(
                Check(
                    "A7b directory step agrees with manifest global_step",
                    CHECK_PASS if encoded_step == global_step else CHECK_FAIL,
                    f"directory step {encoded_step} versus manifest 1/1 global_step "
                    f"value {global_step}",
                )
            )

        global_step_value = manifest.get("global_step")
        fixed_loss_present = "fixed_loss_before_save" in manifest
        checks.append(
            Check(
                "A8 manifest preserves the writer's fixed-loss field",
                CHECK_PASS if fixed_loss_present else CHECK_FAIL,
                f"fixed_loss_before_save present {1 if fixed_loss_present else 0}/1; "
                f"global_step field value={global_step_value!r}",
            )
        )
        return checks


# The driver resolves manifest formats only through this registry. A second supported
# checkpoint format is a new reader object plus one entry here; no structural branch belongs
# in the driver, because format-specific branches in generic gates are how shared labels come
# to measure different things on different models.
FORMAT_READERS = {
    RankLocalShardedV1Reader.FORMAT: RankLocalShardedV1Reader(),
}


def adjudicate(checkpoint_dir: str, phase: str, out_dir: str) -> int:
    """Evaluate one checkpoint directory and return the launcher's exit vocabulary."""

    checkpoint = Path(checkpoint_dir)
    checks: list[Check] = []

    is_directory = checkpoint.is_dir()
    checks.append(
        Check(
            "A1 checkpoint path exists and is a directory",
            CHECK_PASS if is_directory else CHECK_FAIL,
            f"directory found {1 if is_directory else 0}/1 at {checkpoint}",
        )
    )
    if not is_directory:
        return _finish(
            checks,
            EXIT_REFUSE,
            checkpoint_dir=checkpoint,
            phase=phase,
            out_dir=out_dir,
        )

    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.exists():
        checks.append(
            Check(
                "A2 manifest.json exists and parses as a JSON object",
                CHECK_ABSTAIN,
                f"manifest found 0/1 at {manifest_path}; layout is unrecognised",
            )
        )
        return _finish(
            checks,
            EXIT_UNMEASURED,
            checkpoint_dir=checkpoint,
            phase=phase,
            out_dir=out_dir,
        )

    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        checks.append(
            Check(
                "A2 manifest.json exists and parses as a JSON object",
                CHECK_ABSTAIN,
                f"parseable manifest values 0/1 at {manifest_path}: {exc}",
            )
        )
        return _finish(
            checks,
            EXIT_UNMEASURED,
            checkpoint_dir=checkpoint,
            phase=phase,
            out_dir=out_dir,
        )

    if not isinstance(manifest_value, dict):
        checks.append(
            Check(
                "A2 manifest.json exists and parses as a JSON object",
                CHECK_ABSTAIN,
                f"JSON-object manifests 0/1; parsed type={type(manifest_value).__name__}",
            )
        )
        return _finish(
            checks,
            EXIT_UNMEASURED,
            checkpoint_dir=checkpoint,
            phase=phase,
            out_dir=out_dir,
        )

    checks.append(
        Check(
            "A2 manifest.json exists and parses as a JSON object",
            CHECK_PASS,
            f"parseable JSON-object manifests 1/1; top-level fields={len(manifest_value)}",
        )
    )

    format_value = manifest_value.get("format")
    reader = FORMAT_READERS.get(format_value) if isinstance(format_value, str) else None
    if reader is None:
        known = sorted(FORMAT_READERS)
        checks.append(
            Check(
                "A3 manifest declares a known checkpoint format",
                CHECK_ABSTAIN,
                f"registered reader matches 0/1 for format {format_value!r}; "
                f"known_formats={known}",
            )
        )
        return _finish(
            checks,
            EXIT_UNMEASURED,
            checkpoint_dir=checkpoint,
            phase=phase,
            out_dir=out_dir,
        )

    checks.append(
        Check(
            "A3 manifest declares a known checkpoint format",
            CHECK_PASS,
            f"registered reader matches 1/1 for format {format_value!r}; "
            f"registry_size={len(FORMAT_READERS)}",
        )
    )

    checks.extend(reader.validate(checkpoint, manifest_value))
    failed = any(check.status == CHECK_FAIL for check in checks)
    return _finish(
        checks,
        EXIT_REFUSE if failed else EXIT_PASS,
        checkpoint_dir=checkpoint,
        phase=phase,
        out_dir=out_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print("usage: fs_ckpt_adjudicator.py <checkpoint_dir> <phase> <out_dir>")
        return EXIT_PASS
    if len(args) != 3:
        print(
            "ADJUDICATOR-REFUSE rc=96 expected 3 positional arguments "
            f"(<checkpoint_dir> <phase> <out_dir>), got {len(args)}",
            file=sys.stderr,
        )
        return EXIT_REFUSE

    try:
        return adjudicate(args[0], args[1], args[2])
    except Exception as exc:
        # A gate must not escape through Python's default exit 1 after beginning an evaluation:
        # callers composing with the launcher are promised that only 0, 95 and 96 are emitted.
        print(f"ADJUDICATOR-REFUSE rc=96 internal adjudication failure: {exc}", file=sys.stderr)
        return EXIT_REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
'''


TEST_TEMPLATE = '''
#!/usr/bin/env python3
"""Executable controls for the reference FoundationScale checkpoint adjudicator."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# The suite uses @pytest.mark.parametrize but did not import pytest, so all 14 tests died at
# COLLECTION with NameError -- zero tests ran while the emitter reported 9/9 gates green. The
# emitter's C5 counted 14 test FUNCTIONS, which is a static property of the text; nothing
# executed them. That is the all([]) shape in its most literal form: a suite that cannot be
# collected has measured nothing, and counting its functions measures the counting.
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fs_ckpt_adjudicator as adjudicator

FORMAT = "rank-local-sharded-v1"


def build_checkpoint(
    directory: Path,
    *,
    world_size: int = 8,
    global_step: int = 120,
    omitted_ranks: tuple[int, ...] = (),
    zero_length_rank: int | None = None,
    extra_ranks: tuple[int, ...] = (),
    overrides: dict[str, object] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": FORMAT,
        "global_step": global_step,
        "world_size": world_size,
        "rank_payload_count": world_size,
        "expected_rank_payload_count": world_size,
        "fixed_loss_before_save": 1.25,
    }
    if overrides:
        manifest.update(overrides)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\\n", encoding="utf-8"
    )

    for rank in range(world_size):
        if rank in omitted_ranks:
            continue
        shard = directory / f"rank-{rank:05d}.pt"
        if rank == zero_length_rank:
            shard.write_bytes(b"")
        else:
            shard.write_bytes(bytes([(rank % 254) + 1]) * 16)

    for rank in extra_ranks:
        (directory / f"rank-{rank:05d}.pt").write_bytes(b"unexpected-rank-payload")
    return directory


def invoke_checkpoint(directory: Path, tmp_path: Path) -> int:
    out_dir = tmp_path / "launcher-out"
    out_dir.mkdir(exist_ok=True)
    return adjudicator.main([str(directory), "after-save", str(out_dir)])


def test_must_pass_well_formed_eight_rank_checkpoint_returns_zero(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120")

    assert invoke_checkpoint(checkpoint, tmp_path) == 0


def test_must_pass_reports_measured_denominators(tmp_path: Path, capsys) -> None:
    checkpoint = build_checkpoint(tmp_path / "checkpoint_120")

    assert invoke_checkpoint(checkpoint, tmp_path) == 0
    output = capsys.readouterr().out
    assert "A5a expected rank shard set is complete: PASS (present 8/8 expected ranks" in output
    assert "A6 every expected rank shard is a non-empty regular file: PASS " in output
    assert "regular non-empty shards 8/8" in output
    assert "VERDICT 0 PASS" in output


def test_must_fire_missing_rank_shard_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120", omitted_ranks=(3,))

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_zero_length_rank_shard_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120", zero_length_rank=4)

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_extra_unexpected_rank_shard_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "step_120", extra_ranks=(8,))

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_rank_payload_count_less_than_expected_returns_96(tmp_path: Path) -> None:
    # Every file is present, so only the writer's collective save count can expose this loss.
    checkpoint = build_checkpoint(
        tmp_path / "step_120",
        overrides={"rank_payload_count": 7},
    )

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_must_fire_directory_step_disagreeing_with_manifest_returns_96(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(tmp_path / "checkpoint-121", global_step=120)

    assert invoke_checkpoint(checkpoint, tmp_path) == 96


def test_abstains_without_manifest_and_returns_95(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_120"
    checkpoint.mkdir()
    (checkpoint / "rank-00000.pt").write_bytes(b"payload")

    assert invoke_checkpoint(checkpoint, tmp_path) == 95


def test_abstains_on_unknown_format_and_returns_95(tmp_path: Path) -> None:
    checkpoint = build_checkpoint(
        tmp_path / "step_120", overrides={"format": "unknown-layout-v0"}
    )

    assert invoke_checkpoint(checkpoint, tmp_path) == 95


def test_directory_without_step_abstains_only_the_a7_leg(
    tmp_path: Path, capsys
) -> None:
    checkpoint = build_checkpoint(tmp_path / "ordinary-checkpoint-name")

    assert invoke_checkpoint(checkpoint, tmp_path) == 0
    output = capsys.readouterr().out
    assert "A7b directory step agrees with manifest global_step: ABSTAIN" in output
    assert "0/1 step values encoded in directory name" in output
    assert "VERDICT 0 PASS" in output


@pytest.mark.parametrize(
    "suffix",
    ("step_120", "step-120", "checkpoint_120", "checkpoint-120", "ckpt_120", "ckpt-120"),
)
def test_supported_step_encoding_forms_agree_and_return_zero(
    tmp_path: Path, suffix: str
) -> None:
    checkpoint = build_checkpoint(tmp_path / suffix, global_step=120)

    assert invoke_checkpoint(checkpoint, tmp_path) == 0


def test_exit_95_and_exit_96_are_never_confused(tmp_path: Path) -> None:
    abstaining_missing = tmp_path / "abstain-missing"
    abstaining_missing.mkdir()
    abstaining_unknown = build_checkpoint(
        tmp_path / "step_120", overrides={"format": "not-a-registered-format"}
    )

    # These assertions guard the control pair directly: UNMEASURED must not be relabelled as
    # a detected defect, and a detected defect must not be laundered into abstention merely
    # because both are launch-refusing non-zero results.
    assert invoke_checkpoint(abstaining_missing, tmp_path) == 95
    assert invoke_checkpoint(abstaining_unknown, tmp_path) == 95
    assert invoke_checkpoint(abstaining_missing, tmp_path) != 96
    assert invoke_checkpoint(abstaining_unknown, tmp_path) != 96

    bad_cases = (
        build_checkpoint(
            tmp_path / "bad-missing" / "step_120", omitted_ranks=(0,), global_step=120
        ),
        build_checkpoint(
            tmp_path / "bad-empty" / "step_120", zero_length_rank=1, global_step=120
        ),
        build_checkpoint(
            tmp_path / "bad-extra" / "step_120", extra_ranks=(8,), global_step=120
        ),
        build_checkpoint(
            tmp_path / "bad-count" / "step_120",
            overrides={"rank_payload_count": 7},
            global_step=120,
        ),
        build_checkpoint(tmp_path / "bad-step" / "step_999", global_step=120),
    )
    for checkpoint in bad_cases:
        assert invoke_checkpoint(checkpoint, tmp_path) == 96
        assert invoke_checkpoint(checkpoint, tmp_path) != 95


def test_missing_checkpoint_directory_is_a_measured_refusal(tmp_path: Path) -> None:
    missing = tmp_path / "step_120"

    assert invoke_checkpoint(missing, tmp_path) == 96


def test_module_imports_no_tensor_library() -> None:
    # The adjudicator may run on a login node with no model stack installed. Reading the AST
    # tests the actual artifact rather than this test environment's coincidental imports.
    module_path = Path(adjudicator.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "torch" not in imported_roots
'''


def _normalise(template: str) -> str:
    return textwrap.dedent(template).strip("\n") + "\n"


def _render_artifacts() -> dict[str, str]:
    return {
        "h100/gen/fs_ckpt_adjudicator.py": _normalise(ADJUDICATOR_TEMPLATE),
        "h100/gen/test_fs_ckpt_adjudicator.py": _normalise(TEST_TEMPLATE),
    }


def _write_artifacts(pairs: list[tuple[pathlib.Path, str]]) -> int:
    temporary_paths: list[pathlib.Path] = []
    try:
        for target, text in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            temporary_path = pathlib.Path(temporary)
            temporary_paths.append(temporary_path)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)

        for (target, _), temporary in zip(pairs, temporary_paths):
            os.replace(temporary, target)
        return len(pairs)
    except OSError:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise


def _discard_outputs(paths: list[pathlib.Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _report(gates: list[tuple[str, bool, str]]) -> None:
    green = sum(ok for _, ok, _ in gates)
    print("gate table:")
    for label, ok, note in gates:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}{(' (' + note + ')') if note else ''}")
    print(f"{green}/{len(gates)} gates green")


def _fail(gates: list[tuple[str, bool, str]]) -> int:
    _report(gates)
    return 2


def _post_write_fail(
    gates: list[tuple[str, bool, str]], outputs: list[pathlib.Path]
) -> int:
    _discard_outputs(outputs)
    _report(gates)
    print("removed both artifacts; unverified files are not left in place")
    return 4


def _compile_artifacts(paths: list[pathlib.Path]) -> tuple[int, str]:
    clean = 0
    diagnostics: list[str] = []
    for path in paths:
        # A separate invocation per artifact preserves the denominator when one output is
        # valid and the other is not; a combined py_compile failure would only say that at
        # least one unknown member of the set failed.
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            clean += 1
        else:
            diagnostics.append(
                f"{path.name}: rc={proc.returncode} stderr={proc.stderr.strip()[:100]}"
            )
    return clean, "; ".join(diagnostics)


def _run_self_check(
    stage_path: pathlib.Path,
    outputs: list[pathlib.Path],
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="fs-ckpt-stage-rerun-") as temporary:
        rerun_root = pathlib.Path(temporary)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(stage_path),
                    "--root",
                    str(rerun_root),
                    "--skip-self-check",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return False, "child stage timed out"

        if proc.returncode != 0:
            lines = (proc.stdout + proc.stderr).strip().splitlines()
            detail = lines[-1][:120] if lines else "no child output"
            return False, f"child rc={proc.returncode}: {detail}"

        rerun_outputs = [
            rerun_root / "h100" / "gen" / path.name for path in outputs
        ]
        identical = sum(
            original.read_bytes() == rerun.read_bytes()
            for original, rerun in zip(outputs, rerun_outputs)
            if rerun.exists()
        )
        if identical != len(outputs):
            return False, f"byte-identical artifacts {identical}/{len(outputs)}"
        return True, f"byte-identical artifacts {identical}/{len(outputs)}; child rc=0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-self-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    rendered = _render_artifacts()
    module_relative = "h100/gen/fs_ckpt_adjudicator.py"
    test_relative = "h100/gen/test_fs_ckpt_adjudicator.py"
    module_target = root / module_relative
    test_target = root / test_relative

    gates: list[tuple[str, bool, str]] = []

    expected_names = {module_relative, test_relative}
    gates.append(
        (
            "C1 renderer yields exactly the two agreed artifacts",
            set(rendered) == expected_names and len(rendered) == 2,
            f"{len(rendered)}/2 expected artifact name(s): {sorted(rendered)}",
        )
    )
    if not gates[-1][1]:
        return _fail(gates)

    module_text = rendered[module_relative]
    test_text = rendered[test_relative]
    module_lines = module_text.count("\n")
    test_lines = test_text.count("\n")
    test_functions = test_text.count("\ndef test_")

    gates.append(
        (
            "C2 both generated sources are plausible",
            module_lines >= 220 and test_lines >= 120,
            f"module {module_lines} lines, test {test_lines} lines",
        )
    )

    required_module_contract = (
        "EXIT_PASS = 0",
        "EXIT_UNMEASURED = 95",
        "EXIT_REFUSE = 96",
        "FORMAT_READERS",
        "def adjudicate(",
        "class RankLocalShardedV1Reader",
        'FORMAT: ClassVar[str] = "rank-local-sharded-v1"',
    )
    missing_contract = [item for item in required_module_contract if item not in module_text]
    gates.append(
        (
            "C3 adjudicator declares the agreed exit and format contract",
            not missing_contract,
            f"missing={missing_contract}",
        )
    )

    declared_codes = {
        "EXIT_PASS = 0": "EXIT_PASS = 0" in module_text,
        "EXIT_UNMEASURED = 95": "EXIT_UNMEASURED = 95" in module_text,
        "EXIT_REFUSE = 96": "EXIT_REFUSE = 96" in module_text,
    }
    gates.append(
        (
            "C4 all three launcher exit codes are declared distinctly",
            len(declared_codes) == 3 and all(declared_codes.values()),
            f"declared {sum(declared_codes.values())}/3: {declared_codes}",
        )
    )

    required_test_contract = (
        "import fs_ckpt_adjudicator as adjudicator",
        "test_must_pass_well_formed_eight_rank_checkpoint_returns_zero",
        "test_abstains_without_manifest_and_returns_95",
        "test_abstains_on_unknown_format_and_returns_95",
        "test_exit_95_and_exit_96_are_never_confused",
        "test_module_imports_no_tensor_library",
    )
    missing_tests = [item for item in required_test_contract if item not in test_text]
    gates.append(
        (
            "C5 suite references the adjudicator and ships the required controls",
            not missing_tests and test_functions >= 11,
            f"{test_functions} test function(s); missing={missing_tests}",
        )
    )

    module_hits = BLOCKLIST_PATTERN.findall(module_text)
    test_hits = BLOCKLIST_PATTERN.findall(test_text)
    gates.append(
        (
            "C6 no estate literal matches the public-repo blocklist regex",
            not module_hits and not test_hits,
            f"module hits={len(module_hits)}, test hits={len(test_hits)}",
        )
    )

    if not all(ok for _, ok, _ in gates):
        # All gates so far are evaluated before touching the destination. Returning here is
        # what makes "write nothing on a red gate" a property of the stage rather than a
        # convention followed only by its caller.
        return _fail(gates)

    outputs = [module_target, test_target]
    try:
        written = _write_artifacts(
            [(module_target, module_text), (test_target, test_text)]
        )
    except OSError as exc:
        gates.append(("C7 artifacts are written atomically", False, str(exc)[:120]))
        return _post_write_fail(gates, outputs)
    gates.append(
        ("C7 artifacts are written atomically", written == 2, f"{written}/2 artifact(s)")
    )

    clean_count, diagnostics = _compile_artifacts(outputs)
    gates.append(
        (
            "C8 py_compile clean on both outputs",
            clean_count == 2,
            f"{clean_count}/2 py_compile invocations clean"
            + (f"; {diagnostics}" if diagnostics else ""),
        )
    )
    if not gates[-1][1]:
        return _post_write_fail(gates, outputs)

    if args.skip_self_check:
        _report(gates)
        print(
            f"emitted {module_target} ({module_lines} lines) and {test_target} "
            f"({test_lines} lines, {test_functions} tests); py_compile: clean"
        )
        return 0

    reran, rerun_note = _run_self_check(pathlib.Path(__file__).resolve(), outputs)
    gates.append(
        (
            "C9 re-running the stage produces byte-identical artifacts",
            reran,
            rerun_note,
        )
    )
    if not reran:
        return _post_write_fail(gates, outputs)

    _report(gates)
    print(
        f"emitted {module_target} ({module_lines} lines) and {test_target} "
        f"({test_lines} lines, {test_functions} tests); py_compile: clean; "
        "self-rerun: byte-identical"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
