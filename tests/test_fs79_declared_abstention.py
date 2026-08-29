"""Observed-control tests for the #79 declared-abstention repairs (B1-B4).

Doctrine 3: a detector never observed firing is not a control, and one that
never RUNS is not a control either. Every byte string exercised here comes
from the REAL serializer (``RunManifest.to_json``), never a hand-written
JSON literal: a serializer refactor that changes the bare-null spelling or
stops emitting config entries as object keys turns these tests red by
itself — the difference between pinning an assumption and re-asserting it.

* B1 — the two serialization facts every on-disk marker control stands on
  (``declared=None`` serializes as ``"declared": null``; config entries
  serialize as object keys, with the serializer's inner ``key`` echo
  counted separately), pinned against real bytes, plus the symmetric pin
  that a POPULATED declared block never matches the bare-null oracle — a
  false alarm costs what a false green costs (doctrine 5).
* B2 — the full-ft ``expected_expert_bytes`` abstention ships the same
  five-field record shape as the LoRA record, carries its measured
  denominators, and reaches the serialized bytes as object keys while the
  declared field itself stays honestly null — the discrimination the
  artifact must make.
* B3 — the full-ft serialized-bytes control observed firing on exactly its
  measured shapes (bare-null block, emptied declared_fqns, a dropped
  abstention field whose echo survives) and passing an intact record; a
  scope pin asserts the boundary it does NOT claim.
* B4 — both rc arms of the FS_EMIT_DRILL_BARE_NULL drill observed going
  red (exit-1 class with saves observed, exit-3 class with none), beside
  the control's non-drill MUST_FIRE, its MUST_PASS on a stated abstention,
  and the zero-saves arm staying NOT-EXERCISED — never a pass.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered for exactly the exec window: while the module body runs,
    # stdlib dataclasses resolves string annotations via
    # sys.modules.get(cls.__module__).__dict__, which is None for an
    # unregistered alias. The finally removes it even when exec_module
    # raises, so no half-initialized module lingers for a later test to
    # import by accident (fail closed).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


manifest_mod = _load_module(
    "foundationscale_provenance_manifest_under_test",
    _REPO_ROOT / "src" / "foundationscale" / "provenance" / "manifest.py",
)
emit_mod = _load_module(
    "foundationscale_emit_run_manifest_under_test",
    _REPO_ROOT / "tools" / "emit_run_manifest.py",
)


def _bare_manifest(*, declared: object = None, config: object = None) -> object:
    """A minimal valid RunManifest for serialization-shape assertions."""
    code = manifest_mod.CodeProvenance(
        status=manifest_mod.CaptureStatus.NOT_A_REPOSITORY,
        root=None,
        commit=None,
        dirty_files=0,
        untracked_files=0,
        diff_sha256=None,
        diff_bytes=0,
        paths=(),
    )
    environment = manifest_mod.CapturedEnvironment(
        allowlist=("TEST_",), values={}, source_var_count=0
    )
    topology = manifest_mod.Topology(
        nodes=1,
        gpus_per_node=1,
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=1,
    )
    return manifest_mod.RunManifest(
        run_id="fs79-control-tests",
        attempt=1,
        code=code,
        config={} if config is None else config,
        environment=environment,
        topology=topology,
        declared=declared,
    )


def _config_with_entries(entries: list[tuple[str, str]], source: str) -> object:
    resolver = manifest_mod.ConfigResolver(environ={})
    for key, value in entries:
        resolver.record_effective(key, value, source)
    return resolver.freeze()


def _full_ft_declared() -> object:
    return manifest_mod.DeclaredCheckpoint(
        num_experts=16,
        num_moe_layers=2,
        declared_fqns=("a.tensor", "b.tensor"),
        moe_layer_basis="num_moe_layers",
    )


def _full_ft_text() -> str:
    """Real serialized bytes of a full-ft emission's record + declared block."""
    entries = emit_mod._full_ft_bytes_abstention_entries(
        declared_fqn_count=2, expert_family_census=1
    )
    return _bare_manifest(
        declared=_full_ft_declared(),
        config=_config_with_entries(
            entries, emit_mod._FULL_FT_BYTES_ABSTENTION_SOURCE
        ),
    ).to_json()


class TestSerializationContractB1(unittest.TestCase):
    def test_declared_none_serializes_as_json_null(self) -> None:
        text = _bare_manifest(declared=None).to_json()
        self.assertIn('"declared": null', text)
        # ...and the emitter's own on-disk predicate reads those bytes as the
        # bare-null shape it exists to indict.
        self.assertIsNotNone(emit_mod._DECLARED_NULL_RE.search(text))
        loaded = manifest_mod.RunManifest.from_dict(json.loads(text))
        self.assertIsNone(loaded.declared)

    def test_populated_declared_never_reads_bare_null(self) -> None:
        # The symmetric pin: the oracle must not fire on a green input —
        # doctrine 5 prices a false alarm equal to a false green.
        text = _bare_manifest(declared=_full_ft_declared()).to_json()
        self.assertIsNone(emit_mod._DECLARED_NULL_RE.search(text))
        loaded = manifest_mod.RunManifest.from_dict(json.loads(text))
        self.assertIsNotNone(loaded.declared)

    def test_record_keys_serialize_as_object_keys_with_a_single_echo(self) -> None:
        keys = list(emit_mod._LORA_ABSTENTION_RECORD_KEYS) + list(
            emit_mod._FULL_FT_BYTES_ABSTENTION_RECORD_KEYS
        )
        config = _config_with_entries(
            [(key, "pinned") for key in keys], "measured:test-harness"
        )
        text = _bare_manifest(config=config).to_json()
        loaded = json.loads(text)
        for key in keys:
            with self.subTest(key=key):
                # The measurement the emitter's docstring used to carry in
                # prose alone: the with-colon spelling exactly ONCE (the
                # object key), the quoted spelling exactly TWICE (object key
                # plus the serializer's inner "key" echo). Both on-disk
                # controls stand on these counts, so they are pinned here.
                self.assertEqual(text.count(f'"{key}":'), 1)
                self.assertEqual(text.count(f'"{key}"'), 2)
                self.assertIn(f'"key": "{key}"', text)
                self.assertEqual(loaded["config"][key]["value"], "pinned")
                self.assertEqual(loaded["config"][key]["key"], key)


class TestFullFtBytesAbstentionB2(unittest.TestCase):
    def test_entries_match_the_lora_records_five_field_shape(self) -> None:
        entries = emit_mod._full_ft_bytes_abstention_entries(
            declared_fqn_count=10, expert_family_census=2
        )
        keys = [key for key, _value in entries]
        self.assertEqual(
            keys, list(emit_mod._FULL_FT_BYTES_ABSTENTION_RECORD_KEYS)
        )
        self.assertEqual(len(entries), len(emit_mod._LORA_ABSTENTION_RECORD_KEYS))
        self.assertTrue(
            all(key.startswith("declared.expected_expert_bytes.") for key in keys)
        )
        self.assertTrue(all(value.strip() for _key, value in entries))
        by_key = dict(entries)
        self.assertEqual(
            by_key["declared.expected_expert_bytes.status"], "abstained"
        )
        # The abstention must state what it measured: its denominators live
        # in the context field, machine-checkably (doctrine 2).
        context = by_key["declared.expected_expert_bytes.context"]
        self.assertIn("declared_fqns=10", context)
        self.assertIn("2 of", context)

    def test_abstention_reaches_the_artifact_while_field_stays_null(self) -> None:
        entries = emit_mod._full_ft_bytes_abstention_entries(
            declared_fqn_count=10, expert_family_census=2
        )
        config = _config_with_entries(
            entries, emit_mod._FULL_FT_BYTES_ABSTENTION_SOURCE
        )
        text = _bare_manifest(declared=_full_ft_declared(), config=config).to_json()
        loaded = json.loads(text)
        for key, _value in entries:
            with self.subTest(key=key):
                self.assertIn(f'"{key}":', text)
                self.assertIn(key, loaded["config"])
        # The discrimination #79 demands, asserted on the bytes: the field is
        # still null, and the record says that null was CHOSEN, not omitted.
        self.assertIn('"expected_expert_bytes": null', text)

    def test_recorded_through_the_same_stated_channel_main_uses(self) -> None:
        resolver = manifest_mod.ConfigResolver(environ={})
        emit_mod._record_stated_entries(
            resolver,
            emit_mod._full_ft_bytes_abstention_entries(
                declared_fqn_count=2, expert_family_census=1
            ),
            emit_mod._FULL_FT_BYTES_ABSTENTION_SOURCE,
        )
        frozen = resolver.freeze()
        for key in emit_mod._FULL_FT_BYTES_ABSTENTION_RECORD_KEYS:
            self.assertIn(key, frozen)
            self.assertEqual(
                frozen[key].source, emit_mod._FULL_FT_BYTES_ABSTENTION_SOURCE
            )


class TestFullFtSerializedRecordB3(unittest.TestCase):
    def test_must_pass_on_an_intact_record(self) -> None:
        present, total = emit_mod._enforce_full_ft_declared_on_disk(_full_ft_text())
        self.assertEqual(present, total)
        self.assertEqual(
            total, len(emit_mod._FULL_FT_BYTES_ABSTENTION_RECORD_KEYS)
        )

    def test_must_fire_on_a_bare_null_declared_block(self) -> None:
        # The store/serializer dropped the populated block to bare null.
        entries = emit_mod._full_ft_bytes_abstention_entries(
            declared_fqn_count=2, expert_family_census=1
        )
        bad = _bare_manifest(
            declared=None,
            config=_config_with_entries(
                entries, emit_mod._FULL_FT_BYTES_ABSTENTION_SOURCE
            ),
        ).to_json()
        with self.assertRaises(emit_mod.EmitUnmeasured):
            emit_mod._enforce_full_ft_declared_on_disk(bad)

    def test_must_fire_on_declared_fqns_emptied_on_disk(self) -> None:
        good = _full_ft_text()
        bad = re.sub(
            r'"declared_fqns": \[[^]]*\]', '"declared_fqns": []', good, count=1
        )
        # The transform must really have happened, or this MUST_FIRE proves
        # nothing about the empty-array shape.
        self.assertNotEqual(bad, good)
        with self.assertRaises(emit_mod.EmitUnmeasured):
            emit_mod._enforce_full_ft_declared_on_disk(bad)

    def test_must_fire_on_a_dropped_abstention_field_despite_its_echo(self) -> None:
        good = _full_ft_text()
        dropped = emit_mod._FULL_FT_BYTES_ABSTENTION_RECORD_KEYS[0]
        bad = good.replace(f'"{dropped}": {{', '"__dropped__": {', 1)
        self.assertNotEqual(bad, good)
        # The inner "key" echo survives the loss of its own object key — and
        # must NOT count as presence: only the with-colon spelling does.
        self.assertEqual(bad.count(f'"{dropped}":'), 0)
        self.assertIn(f'"key": "{dropped}"', bad)
        with self.assertRaises(emit_mod.EmitUnmeasured) as caught:
            emit_mod._enforce_full_ft_declared_on_disk(bad)
        self.assertIn(dropped, str(caught.exception))

    def test_partial_fqn_reduction_is_outside_the_measured_shapes(self) -> None:
        good = _full_ft_text()
        partial = re.sub(
            r'"declared_fqns": \[[^]]*\]',
            '"declared_fqns": ["a.tensor"]',
            good,
            count=1,
        )
        self.assertNotEqual(partial, good)
        # Scope pin (doctrine 5 cuts both ways): the control's measured
        # shapes are a null/absent declared block, an emptied declared_fqns,
        # and dropped abstention fields. A PARTIAL reduction is outside that
        # evidence; pinning the boundary here keeps the honest scope claim
        # tested rather than aspirational — nobody may read this control as
        # field-by-field equality verification.
        present, total = emit_mod._enforce_full_ft_declared_on_disk(partial)
        self.assertEqual(present, total)


class TestBareNullDrillB4(unittest.TestCase):
    def _record_text(self, *, with_markers: bool) -> str:
        config = None
        if with_markers:
            config = _config_with_entries(
                [(key, "stated") for key in emit_mod._LORA_ABSTENTION_RECORD_KEYS],
                "measured:test-harness",
            )
        return _bare_manifest(declared=None, config=config).to_json()

    def test_drill_arm1_exit1_class_fires_with_saves_observed(self) -> None:
        with self.assertRaises(emit_mod.EmitRefused) as caught:
            emit_mod._enforce_lora_abstention_record(
                self._record_text(with_markers=False),
                saves_observed=2,
                drill_armed=True,
            )
        self.assertIn("DRILL FIRED", str(caught.exception))

    def test_drill_arm2_exit3_class_fires_without_saves_observed(self) -> None:
        with self.assertRaises(emit_mod.EmitUnmeasured) as caught:
            emit_mod._enforce_lora_abstention_record(
                self._record_text(with_markers=False),
                saves_observed=0,
                drill_armed=True,
            )
        self.assertIn("DRILL FIRED", str(caught.exception))

    def test_control_must_fire_outside_the_drill(self) -> None:
        with self.assertRaises(emit_mod.EmitRefused) as caught:
            emit_mod._enforce_lora_abstention_record(
                self._record_text(with_markers=False),
                saves_observed=1,
                drill_armed=False,
            )
        self.assertNotIn("DRILL FIRED", str(caught.exception))

    def test_control_must_pass_on_a_stated_abstention(self) -> None:
        state, present, total = emit_mod._enforce_lora_abstention_record(
            self._record_text(with_markers=True),
            saves_observed=1,
            drill_armed=False,
        )
        self.assertTrue(state.startswith("STATED-ABSTENTION"))
        self.assertEqual(present, total)
        self.assertEqual(total, len(emit_mod._LORA_ABSTENTION_RECORD_KEYS))

    def test_zero_saves_is_named_not_exercised_never_a_pass(self) -> None:
        entries = [
            (key, "stated") for key in emit_mod._LORA_ABSTENTION_RECORD_KEYS
        ]
        text = _bare_manifest(
            declared=None,
            config=_config_with_entries(entries, "measured:test-harness"),
        ).to_json()
        state = emit_mod.check_saved_run_declaration(text, saves_observed=0)
        self.assertIn("NOT-EXERCISED", state)


if __name__ == "__main__":
    unittest.main()
