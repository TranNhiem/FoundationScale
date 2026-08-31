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
# (e.g. `.*?(\d+)`): a parser that accepts anything ending in digits would satisfy the
# agreement gate while giving A7b nothing to cross-validate. `gate_ckpt_naming_agreement.py`
# now renders the writer's real formats and asserts this parser accepts them, so the two
# artifacts cannot drift apart again silently.
_STEP_TOKEN = r"(?:step|checkpoint|ckpt|iter|snapshot)"
_STEP_NAME_RE: Pattern[str] = re.compile(
    rf"{_STEP_TOKEN}(?:[_-]{_STEP_TOKEN})*[_-](?P<step>\d+)", re.IGNORECASE
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
    SHARD_NAME: ClassVar[Pattern[str]] = re.compile(r"rank-(?P<rank>\d{5})\.pt")

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
