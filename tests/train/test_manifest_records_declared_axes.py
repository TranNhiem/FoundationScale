"""Manifest provenance for the nine train declaration axes.

These tests pin the distinction the run manifest exists to preserve:

* an axis passed on the command line has source ``cli``;
* an omitted axis has source ``default``;
* an omitted None-valued axis is nevertheless present as an explicit
  absence, so ``None`` cannot later be mistaken for "this version of the
  emitter never populated the key" (#342);
* the declared key set is derived from both sides of the contract --
  ``TrainConfig`` and the manifest emitter -- rather than from a list that
  can silently drift away from either one.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest

from foundationscale.provenance import RunManifest
from foundationscale.train import cli as train_cli
from foundationscale.train.loop import (
    TrainConfig,
    _build_run_manifest,
    _manifest_payload,
)

# The nine axes introduced as deliberately declarable-but-default-free train
# controls. This tuple names the current contract; `_new_axis_names` then
# compares the FULL TrainConfig field set against the emitter's key set, so a
# future eleventh axis that reaches only one side is RED in this test rather
# than relying on a reviewer noticing that this tuple needs editing.
_NINE_DECLARED_AXES: tuple[str, ...] = (
    "optimizer",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "gradient_checkpointing",
    "attn_implementation",
    "lr_scheduler_type",
    "warmup_steps",
    "sharding_strategy",
    "cpu_optimizer_offload",
)


# TrainConfig fields that are not one of the nine axes under test. Precision
# and the adapter declarations predate that contract, while the remaining
# fields are machine facts, objective/runtime controls, or the nested topology
# declaration. A field absent from this set is treated as a new declaration
# axis and must appear in `_manifest_payload`; that is the must-fire coverage
# arm rather than a hand-maintained expected tuple.
_CONFIG_FIELDS_OUTSIDE_THE_NINE: frozenset[str] = frozenset(
    {
        "model",
        "dataset",
        "output_dir",
        "nodes",
        "gpus_per_node",
        "profile",
        "profile_name",
        "profile_path",
        "objective",
        "precision",
        "adapter",
        "adapter_rank",
        "adapter_alpha",
        "adapter_targets",
        "adapter_dropout",
        "max_steps",
        "per_device_batch_size",
        "learning_rate",
        "save_interval",
        "seed",
        "dp",
        "tp",
        "pp",
        "ep",
        "cp",
        "dry_run",
        "launch_corpus",
        # Provenance ABOUT the other fields, not a declaration axis: it records
        # which flags argparse actually saw so every other key can state a
        # measured `source`. It is deliberately absent from the manifest's
        # `config` mapping because it is not lost there -- the whole of its
        # content is recoverable by reading the sources it produced, and adding
        # a key that restates them would be a second copy free to drift from
        # the first.
        "cli_declared",
    }
)


# Keys other than the nine axes that `_manifest_payload` is currently known to
# emit. This is intentionally a second split rather than the config-field
# split above: the latter describes what TrainConfig declares, while this one
# describes what the emitter actually puts into `config`. Comparing the two
# residual sets catches drift in either direction.
_PAYLOAD_KEYS_OUTSIDE_THE_NINE: frozenset[str] = frozenset(
    {
        "model",
        "dataset",
        "output_dir",
        "max_steps",
        "per_device_batch_size",
        "learning_rate",
        "save_interval",
        "seed",
        "topology",
        "nodes",
        "gpus_per_node",
        "profile_name",
        "profile_path",
        "dry_run",
        "precision",
        "adapter",
        "adapter_rank",
        "adapter_alpha",
        "adapter_targets",
        "adapter_dropout",
    }
)


@dataclass(frozen=True, kw_only=True)
class AxisCase:
    """One command-line declaration and the value expected after parsing."""

    name: str
    cli_token: str
    config_value: object
    manifest_value: str

    @property
    def option(self) -> str:
        return f"--{self.name.replace('_', '-')}"


# Values for the nine currently contracted axes. Boolean axes are explicitly
# declared as the string "false": that reaches TrainConfig as False while an
# omitted flag remains None. Testing that explicit False keeps a CLI source is
# what prevents omission and explicit negative declarations from being
# laundered into one state (#342 in miniature).
_DECLARATION_TOKENS: dict[str, tuple[str, object, str]] = {
    "optimizer": ("adafactor", "adafactor", "adafactor"),
    "gradient_accumulation_steps": ("3", 3, "3"),
    "max_grad_norm": ("0.25", 0.25, "0.25"),
    "gradient_checkpointing": ("false", False, "False"),
    "attn_implementation": ("sdpa", "sdpa", "sdpa"),
    "lr_scheduler_type": ("cosine", "cosine", "cosine"),
    "warmup_steps": ("4", 4, "4"),
    "sharding_strategy": ("ddp", "ddp", "ddp"),
    "cpu_optimizer_offload": ("false", False, "False"),
}


def _baseline_config(tmp_path: Path) -> TrainConfig:
    """A synthetic config whose machine facts and profile are synthetic too."""
    return TrainConfig(
        model="synthetic/model",
        dataset="synthetic/data",
        output_dir=tmp_path / "run",
        nodes=1,
        gpus_per_node=4,
        profile_name="synthetic-profile",
        objective="sft",
    )


def _train_config_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(TrainConfig)}


def _new_axis_names(cfg: TrainConfig) -> set[str]:
    """Derive the axis key set from both TrainConfig and the emitter.

    This is deliberately not ``set(_NINE_DECLARED_AXES)``. If somebody adds a
    new TrainConfig field but forgets the manifest payload, adding it yields
    one extra key on the config side and none on the emitter side. If they add
    an emitter key unknown to TrainConfig, the reverse happens. Either drift
    must fail here, because a manifest that misses a field can no longer
    distinguish an abstention from an emitter that never learned about it.
    """
    payload = _manifest_payload(cfg, stage="coverage")
    config_payload = payload.get("config")

    # REACHED-SITE: the emitter produced a config mapping at all.
    assert isinstance(config_payload, dict), (
        "the manifest emitter did not produce its config mapping; axis "
        "provenance cannot be adjudicated"
    )

    train_axis_names = _train_config_field_names() - _CONFIG_FIELDS_OUTSIDE_THE_NINE
    emitter_axis_names = set(config_payload) - _PAYLOAD_KEYS_OUTSIDE_THE_NINE

    # OUTCOME: the two independently derived residual sets are the same set.
    assert train_axis_names == emitter_axis_names, (
        "TrainConfig's undeclared-as-None axis fields and the manifest "
        "emitter's corresponding keys have drifted apart: "
        f"only in TrainConfig={sorted(train_axis_names - emitter_axis_names)}, "
        f"only in emitter={sorted(emitter_axis_names - train_axis_names)}"
    )

    # The existing nine-contract remains explicitly pinned. Without this arm a
    # rename made consistently on both sides could pass the drift comparison
    # while dropping the delivery under test.
    missing_current = set(_NINE_DECLARED_AXES) - train_axis_names
    assert not missing_current, (
        f"the manifest axis contract lost one of the declared nine axes: {sorted(missing_current)}"
    )
    return train_axis_names


def _axis_case(field: dataclasses.Field[Any]) -> AxisCase:
    """Create a deterministic declaration for one TrainConfig axis field."""
    declared = _DECLARATION_TOKENS.get(field.name)
    if declared is not None:
        cli_token, config_value, manifest_value = declared
        return AxisCase(
            name=field.name,
            cli_token=cli_token,
            config_value=config_value,
            manifest_value=manifest_value,
        )

    # Generic support for a future axis that has been added to BOTH TrainConfig
    # and the manifest emitter. Type information keeps the case real rather
    # than minting an arbitrary string for every field. The CLI parser must
    # also know the option; argparse will reject it here, which is the intended
    # visible RED for a third side being missed. There is no skip arm.
    hints = get_type_hints(TrainConfig)
    non_none = tuple(
        argument for argument in get_args(hints[field.name]) if argument is not type(None)
    )
    if len(non_none) != 1:
        pytest.fail(
            f"new manifest axis {field.name!r} has a union shape this test "
            "cannot declare deterministically"
        )
    declaration_type = non_none[0]
    if declaration_type is bool:
        return AxisCase(
            name=field.name,
            cli_token="false",
            config_value=False,
            manifest_value="False",
        )
    if declaration_type is int:
        return AxisCase(
            name=field.name,
            cli_token="7",
            config_value=7,
            manifest_value="7",
        )
    if declaration_type is float:
        return AxisCase(
            name=field.name,
            cli_token="0.125",
            config_value=0.125,
            manifest_value="0.125",
        )
    if declaration_type is str:
        token = f"synthetic-{field.name.replace('_', '-')}"
        return AxisCase(
            name=field.name,
            cli_token=token,
            config_value=token,
            manifest_value=token,
        )
    pytest.fail(
        f"new manifest axis {field.name!r} has unsupported declaration type "
        f"{declaration_type!r}; add a deterministic declaration rather than "
        "silently omitting coverage"
    )
    raise AssertionError("pytest.fail raised")  # pragma: no cover - type narrowing


def _axis_cases(cfg: TrainConfig) -> tuple[AxisCase, ...]:
    fields_by_name = {field.name: field for field in dataclasses.fields(TrainConfig)}
    axis_names = _new_axis_names(cfg)
    missing_fields = axis_names - set(fields_by_name)
    assert not missing_fields, (
        f"the axis contract names fields absent from TrainConfig: {sorted(missing_fields)}"
    )
    return tuple(
        sorted(
            (_axis_case(fields_by_name[name]) for name in axis_names),
            key=lambda case: case.name,
        )
    )


@pytest.fixture
def invoked_train_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    """Run the real argparse path while stopping before training.

    The source under test is the CLI parser and TrainConfig composition, not a
    manually constructed config that happens to have the same values. Replacing
    `train` keeps this CPU-only and takes seconds; both non-replacement and
    multiple invocations are asserted separately below, so a monkeypatch that
    was not reached is a failing arm rather than a happily vacuous one.
    """
    captured: list[TrainConfig] = []

    def capture(cfg: TrainConfig) -> int:
        captured.append(cfg)
        return 0

    monkeypatch.setattr(train_cli, "train", capture, raising=True)

    def invoke(extra_args: Sequence[str]) -> TrainConfig:
        argv = [
            "--model",
            "synthetic/model",
            "--dataset",
            "synthetic/data",
            "--output-dir",
            str(tmp_path / "run"),
            "--nodes",
            "1",
            "--gpus-per-node",
            "4",
            "--profile-name",
            "synthetic-profile",
            *extra_args,
        ]
        rc = train_cli.main(argv)

        # OUTCOME: CLI composition returned the substituted trainer's result.
        assert rc == 0

        # REACHED-SITE: the trainer boundary was reached exactly once. Merely
        # checking `captured[0]` would turn an uncalled CLI into IndexError
        # without identifying which control was missing, and an extra call
        # would silently overwrite provenance assumptions.
        assert len(captured) == 1
        return captured[0]

    return invoke


def _build_manifest(
    cfg: TrainConfig,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> RunManifest:
    """Build the real provenance manifest under a synthetic process context."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "argv", ["foundationscale-train"])

    manifest = _build_run_manifest(
        cfg,
        stage="provenance-test",
        extra=None,
        declared=None,
        notes={},
    )

    # REACHED-SITE: provenance itself was built. A None return would mean the
    # degraded plain-JSON arm ran, and that arm deliberately does not satisfy
    # the structured manifest reader.
    assert manifest is not None, (
        "the structured RunManifest builder returned no manifest; the degraded "
        "writer cannot carry the key/value/source records under test"
    )
    return manifest


def _assert_axis_records(
    manifest: RunManifest,
    *,
    cases: Sequence[AxisCase],
    expected_source: str,
) -> None:
    """Assert key, value, and provenance as three independent outcomes."""
    records = manifest.config

    # REACHED-SITE: every declared axis entered manifest.config.
    missing = sorted(case.name for case in cases if case.name not in records)
    assert not missing, (
        "manifest.config omitted declaration-axis key(s), which erases the "
        f"distinction between abstained and unpopulated: {missing}"
    )

    for case in cases:
        record = records[case.name]

        # The three outcomes are separate so a wrong key is not accidentally
        # reported as a wrong value, and wrong provenance is not hidden by a
        # matching string representation.
        assert record.key == case.name
        assert record.value == case.manifest_value
        assert record.source == expected_source


def _round_trip_through_json(manifest: RunManifest, *, tmp_path: Path) -> RunManifest:
    """Take the manifest over its own to_dict/from_dict and JSON boundary."""
    encoded = manifest.to_dict()

    # REACHED-SITE: to_dict returned a real serialized representation.
    assert isinstance(encoded, dict)

    wire_path = tmp_path / "manifest-wire.json"
    wire_path.write_text(
        json.dumps(encoded, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decoded = json.loads(wire_path.read_text(encoding="utf-8"))

    # OUTCOME: JSON itself was lossless before the schema restore is consulted.
    assert decoded == encoded

    restored = RunManifest.from_dict(decoded)

    # REACHED-SITE: the schema reader produced another RunManifest.
    assert isinstance(restored, RunManifest)
    assert isinstance(restored.config, Mapping)
    return restored


def test_emitter_declared_axis_set_tracks_trainconfig_field_set(tmp_path: Path) -> None:
    """Fail automatically when a TrainConfig field and manifest key drift.

    The bad arm is direct: adding a field to TrainConfig but not to
    `_manifest_payload` changes exactly one side of this comparison. Reversed
    drift is caught as well. This is the arm against the recurring defect
    class in which a control is made configurable but its provenance remains
    absent ([no key means no checkable claim], #342/#375).
    """
    cfg = _baseline_config(tmp_path)
    axis_names = _new_axis_names(cfg)

    # The current required delivery names all nine, while the set comparison
    # above remains open to deliberately added future axes.
    for name in _NINE_DECLARED_AXES:
        assert name in axis_names, (
            f"{name!r} is part of the nine-axis delivery but was not "
            "reachable through the TrainConfig/emitter comparison"
        )


def test_every_parser_dest_resolves_to_a_trainconfig_field() -> None:
    """A flag whose dest names no field would lose its provenance silently.

    ``_declared_fields`` reports argparse DESTS, translated through one small
    exception map; ``_config_source`` then looks those names up against
    manifest keys, which are TrainConfig FIELD names. The two vocabularies
    agree today by one character (``--adapter-target`` appends into
    ``adapter_targets``), and nothing in the source makes them agree tomorrow.

    The failure this guards is quiet in the worst way: a new flag whose dest
    does not spell its field is still parsed, still reaches TrainConfig, and
    still trains -- but its key is not in ``cli_declared`` under the name the
    emitter looks up, so the manifest records the operator's explicit
    declaration as ``default``. A wrong source reads as provenance, so nothing
    downstream can tell it from the truth.
    """
    parser = train_cli.build_parser()
    parsed = parser.parse_args(
        [
            "--model",
            "synthetic/model",
            "--dataset",
            "synthetic/data",
            "--output-dir",
            "/tmp/does-not-need-to-exist",
            "--nodes",
            "1",
            "--gpus-per-node",
            "1",
            "--profile-name",
            "synthetic-profile",
        ]
    )
    field_names = _train_config_field_names()

    # REACHED-SITE: the parser produced dests at all.
    dests = set(vars(parsed))
    assert dests, "the parser exposed no dests; the mapping below is vacuous"

    unresolved = sorted(
        dest for dest in dests if train_cli._DEST_TO_FIELD.get(dest, dest) not in field_names
    )

    # OUTCOME: every dest lands on a real field, so every declaration the
    # operator makes is findable under the name the emitter asks for.
    assert not unresolved, (
        "these argparse dests resolve to no TrainConfig field, so a flag the "
        "operator supplies would be recorded with source 'default': "
        f"{unresolved}"
    )

    # MUST-FIRE: the same check over a deliberately unmapped dest is RED, so a
    # green result above is the mapping working rather than the lookup being
    # unable to fail.
    assert train_cli._DEST_TO_FIELD.get("no_such_dest", "no_such_dest") not in field_names


def test_undeclared_axes_are_recorded_as_default_absences_and_round_trip(
    invoked_train_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An omitted axis stays present, None, and sourced to the default.

    This must never collapse to omitting the key. Present key + None means the
    operator abstained and an engine default applied. Missing key means the
    emitter did not know about the field. Those are observably different
    states, and only the first is an accountable manifest record.
    """
    cfg = invoked_train_config([])
    axis_names = _new_axis_names(cfg)
    cases = tuple(
        AxisCase(
            name=name,
            cli_token="",
            config_value=getattr(cfg, name),
            manifest_value="None",
        )
        for name in sorted(axis_names)
    )

    payload = _manifest_payload(cfg, stage="default-provenance")
    config_payload = payload.get("config")
    assert isinstance(config_payload, dict)

    # REACHED-SITE: each axis feature exists in the raw emitter mapping.
    missing_payload_keys = sorted(name for name in axis_names if name not in config_payload)
    assert not missing_payload_keys

    # OUTCOME: the raw emitter kept None as its value rather than replacing it
    # with Adams', clipping, scheduler, attention, or sharding defaults.
    wrong_absences = {
        name: config_payload[name] for name in axis_names if config_payload[name] is not None
    }
    assert not wrong_absences, (
        "an omitted declaration axis was coerced before provenance; None is "
        f"the declared abstention marker (#342): {wrong_absences}"
    )

    manifest = _build_manifest(cfg, monkeypatch=monkeypatch, tmp_path=tmp_path)
    _assert_axis_records(
        manifest,
        cases=cases,
        expected_source="default",
    )

    # The EffectiveValue schema serializes its value as a string, so "None" is
    # the explicit marker for an absence that reached provenance. The contract
    # being tested is nevertheless presence-plus-absence, not a missing key.
    restored = _round_trip_through_json(manifest, tmp_path=tmp_path)
    _assert_axis_records(
        restored,
        cases=cases,
        expected_source="default",
    )


def test_command_line_axes_are_sourced_to_cli_and_survive_round_trip(
    invoked_train_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit command-line declarations record their values and CLI source."""
    representative_cfg = _baseline_config(tmp_path / "representative")
    cases = _axis_cases(representative_cfg)
    cli_args: list[str] = []
    for case in cases:
        cli_args.extend((case.option, case.cli_token))

    cfg = invoked_train_config(cli_args)

    # REACHED-SITE: argparse carried each declaration into TrainConfig before
    # the manifest builder could see it.
    parsed_mismatches = {
        case.name: getattr(cfg, case.name)
        for case in cases
        if getattr(cfg, case.name) != case.config_value
    }
    assert not parsed_mismatches, (
        f"one or more CLI declarations did not reach TrainConfig: {parsed_mismatches}"
    )

    payload = _manifest_payload(cfg, stage="cli-provenance")
    config_payload = payload.get("config")
    assert isinstance(config_payload, dict)

    # REACHED-SITE before value comparison: an absent key is an emitter loss,
    # not merely an incorrect value.
    missing_payload_keys = sorted(case.name for case in cases if case.name not in config_payload)
    assert not missing_payload_keys, (
        f"the emitter dropped command-line axis key(s): {missing_payload_keys}"
    )

    # OUTCOME: the emitter carried the parsed Python value, including explicit
    # False rather than converting it to an omitted/None declaration.
    payload_mismatches = {
        case.name: config_payload[case.name]
        for case in cases
        if config_payload[case.name] != case.config_value
    }
    assert not payload_mismatches, (
        f"the manifest payload changed one or more command-line axis values: {payload_mismatches}"
    )

    manifest = _build_manifest(cfg, monkeypatch=monkeypatch, tmp_path=tmp_path)
    _assert_axis_records(
        manifest,
        cases=cases,
        expected_source="cli",
    )

    restored = _round_trip_through_json(manifest, tmp_path=tmp_path)
    _assert_axis_records(
        restored,
        cases=cases,
        expected_source="cli",
    )


def test_explicit_false_is_cli_source_not_omission(
    invoked_train_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit negative booleans must not be laundered into defaults.

    `--gradient-checkpointing false` and `--cpu-optimizer-offload false` carry
    information: the operator answered. Not passing either flag is a different
    declaration state (None/default). If both become False, the provenance
    boundary has silently converted abstention into a choice.
    """
    boolean_cases = tuple(
        case
        for case in _axis_cases(_baseline_config(tmp_path / "representative"))
        if isinstance(case.config_value, bool)
    )
    current_booleans = {"gradient_checkpointing", "cpu_optimizer_offload"}

    # REACHED-SITE: both boolean axes covered by this delivery entered the
    # generic axis contract; otherwise the assertions below would be vacuous.
    assert {case.name for case in boolean_cases} == current_booleans

    cli_args: list[str] = []
    for case in boolean_cases:
        # Both current declarations use explicit false, keeping them
        # distinguishable from the omitted-None arm in the default test.
        assert case.cli_token == "false"
        cli_args.extend((case.option, case.cli_token))

    cfg = invoked_train_config(cli_args)

    for case in boolean_cases:
        # OUTCOME at the config boundary, separate from manifest provenance.
        assert getattr(cfg, case.name) is False

    manifest = _build_manifest(cfg, monkeypatch=monkeypatch, tmp_path=tmp_path)

    # REACHED-SITE: both negative declarations entered the structured config.
    for case in boolean_cases:
        assert case.name in manifest.config

    for case in boolean_cases:
        record = manifest.config[case.name]
        assert record.key == case.name
        assert record.value == "False"
        assert record.source == "cli"


def test_round_trip_cannot_turn_default_provenance_into_cli_provenance(
    invoked_train_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Serialization preserves source, not just key and rendered value.

    This is the dangerous directed arm for a restore path that recreates every
    EffectiveValue with source="cli": values and keys would look correct while
    all provenance claims drift. Compare the original and restored records
    field-by-field rather than only asserting that some config mapping exists.
    """
    cfg = invoked_train_config([])
    names = sorted(_new_axis_names(cfg))
    original = _build_manifest(cfg, monkeypatch=monkeypatch, tmp_path=tmp_path)
    restored = _round_trip_through_json(original, tmp_path=tmp_path)

    # REACHED-SITE before any outcome: every axis must be present in each
    # representation, otherwise a missing record would silently define the
    # comparison below as zero tuples.
    original_missing = sorted(name for name in names if name not in original.config)
    restored_missing = sorted(name for name in names if name not in restored.config)
    assert not original_missing
    assert not restored_missing

    for name in names:
        before = original.config[name]
        after = restored.config[name]

        # Key, absence marker, and source are independent outcomes. In
        # particular, source stays "default": serialization adds no command.
        assert after.key == before.key == name
        assert after.value == before.value == "None"
        assert after.source == before.source == "default"
