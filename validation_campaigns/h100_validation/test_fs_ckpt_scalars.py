#!/usr/bin/env python3
"""Certification harness for fs_ckpt_scalars.py — the torch-free checkpoint scalar reader.

WHY THIS FILE EXISTS, AND WHAT IT IS PINNING
--------------------------------------------
The module under test read ``fixed_loss_before_save`` off eight rank-local checkpoint
payloads on a host with no torch and no GPUs, and found the eight ranks holding two
distinct values with a spread of exactly ``0.17570888996124268`` — bit-for-bit the number
the run had reported as a *restore* error, though the split between ranks was fully
determined before the checkpoint was ever written. A control checkpoint read the same way
showed a spread of exactly 0.0 across 8 of 8 ranks. A measurement that load-bearing needs
a certified instrument; until now the instrument had no in-tree tests.

Two defects-behaviors this harness exists to pin:

  * ``identical_across_ranks`` asserted over a short denominator (7 of 8 payloads present)
    was a real defect: the claim was wider than its evidence. The fix renamed the partial
    claim to ``identical_across_present_ranks`` and refuses to emit the unqualified key
    when any payload is missing. C4 asserts the new key is present AND the old key is
    absent — the absence is the fix.
  * one manifest scalar standing in for eight per-rank values. C9 writes a manifest that
    disagrees with every payload and proves the report comes from the payloads.

THE SHAPE OF THE WHOLE FILE
---------------------------
The estate host that runs this has no torch (that is the point of the tool), so the
fixtures are hand-synthesized torch-shaped zip payloads built with the standard library
alone: a zip whose ``payload/data.pkl`` member (nested, deliberately, to exercise the
module's ``endswith`` search) unpickles to a dict that mixes readable bookkeeping scalars
with values a stock unpickler chokes on. Leg C0 is the control that proves the fixture is
load-bearing: if a stock ``pickle.Unpickler`` ever reads it, every downstream leg is
vacuous and C0 fails rather than letting the harness pass on dead stub machinery.

Exit-state contract under test: 0 measured, 95 unmeasured/partial (fail closed), 96
refuse.

Python 3.6 compatible throughout: no f-string ``=`` specifiers, no walrus, no PEP 585/604
generics, no ``from __future__ import annotations`` (a hard SyntaxError on 3.6).
"""

import contextlib
import importlib.util
import io
import json
import os
import pickle
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path


# --------------------------------------------------------------------------
# Load the module under test by path; nothing here assumes it is on sys.path.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, "fs_ckpt_scalars.py")


def _load_module_under_test():
    if not os.path.isfile(_MODULE_PATH):
        raise RuntimeError(
            "module under test not found next to this harness: %s" % _MODULE_PATH)
    spec = importlib.util.spec_from_file_location("fs_ckpt_scalars", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fsckpt = _load_module_under_test()


# --------------------------------------------------------------------------
# Published measurement, pinned as a regression constant.
# --------------------------------------------------------------------------
# The eight per-rank values the failing run's checkpoint actually contained, rank order.
# The campaign published max-min == 0.17570888996124268; if a future refactor moves the
# arithmetic (sorting, rounding, abs, anything), C2 goes red with expected-vs-observed.
MEASURED_VECTOR = [
    0.5986318588256836, 0.7743407487869263, 0.5986318588256836, 0.5986318588256836,
    0.7743407487869263, 0.7743407487869263, 0.5986318588256836, 0.5986318588256836,
]
PUBLISHED_SPREAD = 0.17570888996124268
FIXED_KEY = "fixed_loss_before_save"


class _Absent(object):
    """Marker: this rank's payload omits the key entirely (the real final save does)."""
    pass


ABSENT = _Absent()


# --------------------------------------------------------------------------
# Hand-synthesized torch-shaped stand-ins. A payload must contain values that a stock
# unpickler cannot read, or C0's control is meaningless and the rest of the harness is
# certifying nothing.
# --------------------------------------------------------------------------
class _StorageSentinel(object):
    """Stand-in for a tensor's storage reference.

    torch emits storages through ``persistent_id`` instead of pickling the bytes; this
    sentinel does the same, so the module's ``persistent_load`` stub is the only thing
    that keeps a load of the fixture alive.
    """

    def __init__(self, storage_key):
        self.storage_key = storage_key


class _ShardedTensorStandIn(object):
    """Pickles as ``torch._fixture_tensors._ShardedTensorStandIn`` — a global that does
    not exist on any host this harness runs on. Unpickling it exercises ``find_class``
    on an unimportable module, exactly as a real ShardedTensor would."""

    def __init__(self, shard_tag):
        self.shard_tag = shard_tag


class _ProcessGroupOuter(object):
    """A nested class, pickled under a fake module. Its ``__qualname__`` contains a dot
    (``_ProcessGroupOuter.ProcessGroupState``), which is the grammar that forces the
    module's ``_StubMeta``: arbitrary attribute access against a stub standing in for a
    class must answer, or the load dies."""

    class ProcessGroupState(object):
        pass


# Rewrite the classes' import addresses so pickling names globals that cannot be
# imported at load time. Bump each defining module out of this file's namespace.
_ShardedTensorStandIn.__module__ = "torch._fixture_tensors"
_ProcessGroupOuter.__module__ = "torch._fixture_c10d"
_ProcessGroupOuter.ProcessGroupState.__module__ = "torch._fixture_c10d"

# What the pickler will look up on each fake module: pickle verifies at dump time that
# ``sys.modules[module].<dotted-name> is obj``, so the attribute names must match the
# classes' qualnames.
_FAKE_MODULE_ATTRS = {
    "torch._fixture_tensors": {"_ShardedTensorStandIn": _ShardedTensorStandIn},
    "torch._fixture_c10d": {"_ProcessGroupOuter": _ProcessGroupOuter},
}


@contextlib.contextmanager
def _fixture_import_surface():
    """Install the fake modules just for the duration of a ``dump``.

    A ``Pickler`` verifies every global by importing its module, so synthesis requires
    the modules to exist *while dumping*; the reader must still never see them. The
    finally block restores ``sys.modules`` exactly, so C10's "no torch" hygiene assert
    is meaningful after fixtures have been built.
    """
    saved = {}
    # The parent package must be installed too, and this is not a nicety. `save_global`
    # calls `__import__(module_name)` with no fromlist, and that form returns
    # `_gcd_import(name.partition(".")[0])` -- it resolves the TOP-LEVEL name even when
    # the full dotted name is already present in `sys.modules`. So a stub at
    # "torch._fixture_tensors" alone still sends the import system after the real
    # `torch`, and on a host without one the dump dies
    # `PicklingError: ... No module named 'torch'`. Measured on the build host: 8 of 11
    # legs errored this way, all of them in fixture synthesis rather than in the reader.
    # Installing the parent does not weaken C10: that leg asserts `"torch" not in
    # sys.modules` AFTER a full survey, so the parent's teardown is pinned by an
    # existing control rather than by this comment.
    for module_name in sorted(_FAKE_MODULE_ATTRS):
        parent_name = module_name.partition(".")[0]
        if parent_name == module_name or parent_name in saved:
            continue
        saved[parent_name] = sys.modules.get(parent_name)
        sys.modules[parent_name] = types.ModuleType(parent_name)
    for module_name, attrs in _FAKE_MODULE_ATTRS.items():
        fake = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(fake, attr_name, value)
        saved[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = fake
        # Bind the child onto the stub parent, mirroring what a real package import
        # does: `_getattribute` walks from the module object the import returned.
        parent = sys.modules.get(module_name.partition(".")[0])
        if parent is not None and parent is not fake:
            setattr(parent, module_name.partition(".")[2], fake)
    try:
        yield
    finally:
        for module_name, previous in saved.items():
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous


class _FixturePickler(pickle.Pickler):
    """Emit storage references the way torch does: through ``persistent_id``."""

    def persistent_id(self, obj):
        if isinstance(obj, _StorageSentinel):
            return ("storage", "FloatStorage", obj.storage_key)
        return None


def _pickle_document(obj):
    # Protocol 4, on purpose: the nested-class global is then emitted as STACK_GLOBAL
    # carrying the dotted qualname "torch._fixture_c10d" /
    # "_ProcessGroupOuter.ProcessGroupState" (not via hand-assembled opcode bytes — the
    # pickler names it, which is the legitimate way to reach this load path), and the
    # persistent id goes out as BINPERSID the way real torch payloads address storage.
    buffer = io.BytesIO()
    with _fixture_import_surface():
        _FixturePickler(buffer, protocol=4).dump(obj)
    return buffer.getvalue()


def _rank_document(fixed_loss=ABSENT, global_step=12440, world_size=8,
                   optimizer_state_count=2):
    """A torch-shaped checkpoint dict: readable scalars mixed with values that must NOT
    be readable without the module's stub machinery."""
    document = {
        "global_step": global_step,
        "world_size": world_size,
        "optimizer_state_count": optimizer_state_count,
    }
    if fixed_loss is not ABSENT:
        document[FIXED_KEY] = fixed_loss
    # Everything below this line is undumpable back into life on this host: a
    # storage reference (only persistent_load can carry it across), an instance whose
    # class lives in a module that does not exist here (only find_class can), and an
    # instance of a nested class whose pickled name contains a dot.
    document["_unpkl_storage_ref"] = _StorageSentinel("0/0/handler")
    document["_unpkl_sharded_tensor"] = _ShardedTensorStandIn("rank shard")
    document["_unpkl_pg_state"] = _ProcessGroupOuter.ProcessGroupState()
    return document


def _write_rank_payload(directory, rank, pickled_document):
    """Write ``rank-NNNNN.pt`` as a zip whose pkl member is NESTED under an archive
    directory, matching real payload layout; the module's member search is an
    ``endswith("data.pkl")`` and a root-level fixture would leave that uncovered."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload/data.pkl", pickled_document)
    (directory / ("rank-%05d.pt" % rank)).write_bytes(archive.getvalue())


def _read_pkl_member(payload_path):
    with zipfile.ZipFile(str(payload_path)) as zf:
        return zf.read("payload/data.pkl")


# --------------------------------------------------------------------------
# The legs.
# --------------------------------------------------------------------------
class TestFsCkptScalars(unittest.TestCase):
    def setUp(self):
        self._scratch = tempfile.TemporaryDirectory(prefix="fs179-")
        self.addCleanup(self._scratch.cleanup)
        self.root = Path(self._scratch.name)

    def _dir(self, name):
        directory = self.root / name
        directory.mkdir()
        return directory

    def _write_rank_set(self, directory, losses, ranks=None):
        if ranks is None:
            ranks = range(len(losses))
        for rank, loss in zip(ranks, losses):
            _write_rank_payload(directory, rank, _pickle_document(
                _rank_document(fixed_loss=loss, world_size=max(1, len(losses)))))

    def _run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fsckpt.main(argv)
        return rc, out.getvalue(), err.getvalue()

    # -- C0: the control that keeps every other leg honest. ------------------
    def test_c00_must_fire_control(self):
        directory = self._dir("c0")
        payload = directory / "rank-00000.pt"
        _write_rank_payload(directory, 0, _pickle_document(
            _rank_document(fixed_loss=0.5986318588256836)))
        pkl = _read_pkl_member(payload)

        # Half 1: a stock unpickler MUST raise on these bytes. If it ever succeeds, the
        # fixture no longer exercises persistent_load/find_class/_StubMeta and every
        # downstream leg is vacuous — this harness must say so, loudly, not pass.
        stock_error = None
        try:
            pickle.Unpickler(io.BytesIO(pkl)).load()
        except Exception as exc:  # UnpicklingError / ImportError / AttributeError
            stock_error = exc
        self.assertIsNotNone(
            stock_error,
            "C0 MUST_FIRE: a stock pickle.Unpickler read the synthesized data.pkl; the "
            "fixture no longer exercises the stub machinery, so every downstream leg is "
            "vacuous and must not be trusted")
        self.assertIsInstance(
            stock_error, (pickle.UnpicklingError, ImportError, AttributeError),
            "C0 MUST_FIRE: stock reader failed, but not for an expected reason: %r"
            % (stock_error,))
        print("C0 MUST_FIRE control: stock pickle.Unpickler raised %s: %.100s"
              % (type(stock_error).__name__, stock_error))

        # Half 1b: isolate the SECOND obstacle. Half 1 is satisfied by the persistent-id
        # opcode alone -- the stock reader dies at BINPERSID and never reaches the
        # `torch.*` global -- so on its own it demonstrates only one of the two problems
        # this module solves, and would keep printing a green "MUST_FIRE fired" if the
        # unimportable-global path stopped being exercised at all. Give a stock reader a
        # permissive persistent_load, clearing obstacle one, and require it to still fail
        # on obstacle two. Two obstacles, two pieces of evidence, each named.
        class _PersistentOnly(pickle.Unpickler):
            def persistent_load(self, pid):
                return ("storage-stub", pid)

        global_error = None
        try:
            _PersistentOnly(io.BytesIO(pkl)).load()
        except Exception as exc:
            global_error = exc
        self.assertIsNotNone(
            global_error,
            "C0 MUST_FIRE: with persistent ids satisfied, a stock reader read the "
            "torch-shaped globals; find_class/_StubMeta are then unexercised and the "
            "stub assertions below prove nothing")
        # ModuleNotFoundError subclasses ImportError, which is what 3.6 through 3.13 all
        # raise here; AttributeError is the acceptable variant if a parent package
        # happens to exist without the submodule.
        self.assertIsInstance(
            global_error, (ImportError, AttributeError),
            "C0 MUST_FIRE: expected the unimportable global to be the second obstacle, "
            "observed %r" % (global_error,))
        print("C0 MUST_FIRE control: persistent-ids-satisfied reader raised %s: %.100s"
              % (type(global_error).__name__, global_error))

        # Half 2: the module's stub unpickler MUST read the same bytes successfully.
        try:
            recovered = fsckpt._ScalarUnpickler(io.BytesIO(pkl)).load()
        except Exception as exc:
            self.fail("C0 MUST_FIRE: _ScalarUnpickler must succeed where the stock "
                      "reader raises; observed %r" % (exc,))
        self.assertIsInstance(recovered, dict,
                              "C0 MUST_FIRE: expected a dict back, observed %r"
                              % (type(recovered).__name__,))
        self.assertEqual(recovered[FIXED_KEY], 0.5986318588256836,
                         "C0 MUST_FIRE: scalar round-trip mismatch: expected "
                         "0.5986318588256836, observed %r" % (recovered.get(FIXED_KEY),))
        for unreadable in ("_unpkl_storage_ref", "_unpkl_sharded_tensor",
                           "_unpkl_pg_state"):
            self.assertNotIsInstance(
                recovered[unreadable], (int, float, str, bool),
                "C0 MUST_FIRE: %r should have come back as a stub, not a scalar: %r"
                % (unreadable, recovered[unreadable]))

    # -- C1: the happy path. 8 of 8, equal values. ---------------------------
    def test_c01_measured_equal(self):
        directory = self._dir("c1")
        self._write_rank_set(directory, [0.4125] * 8)
        report = fsckpt.survey(directory, 8, (FIXED_KEY,))
        entry = report["keys"][FIXED_KEY]
        self.assertEqual(report["ranks"], "8 of 8 rank payloads present",
                         "C1 MEASURED: denominator wrong: %r" % (report["ranks"],))
        self.assertEqual(entry["recorded"], "8 of 8 ranks recorded this key",
                         "C1 MEASURED: recorded count wrong: %r" % (entry["recorded"],))
        self.assertEqual(entry["spread"], 0.0,
                         "C1 MEASURED: equal values must give spread 0.0, observed %r"
                         % (entry["spread"],))
        self.assertIs(entry["identical_across_ranks"], True,
                      "C1 MEASURED: expected identical_across_ranks True, observed %r"
                      % (entry["identical_across_ranks"],))
        self.assertEqual(entry["status"], "MEASURED",
                         "C1 MEASURED: status wrong: %r" % (entry["status"],))
        rc, _out, _err = self._run_main(
            [str(directory), "--world-size", "8", "--key", FIXED_KEY])
        self.assertEqual(rc, 0, "C1 MEASURED: expected exit 0, observed %r" % (rc,))

    # -- C2: the published divergent vector, pinned to the exact spread. ------
    def test_c02_measured_divergent(self):
        directory = self._dir("c2")
        self._write_rank_set(directory, list(MEASURED_VECTOR))
        report = fsckpt.survey(directory, 8, (FIXED_KEY,))
        entry = report["keys"][FIXED_KEY]
        self.assertEqual(entry["per_rank"], MEASURED_VECTOR,
                         "C2 MEASURED: per_rank vector does not match the payloads: %r"
                         % (entry["per_rank"],))
        self.assertEqual(entry["min"], 0.5986318588256836,
                         "C2 MEASURED: min wrong: %r" % (entry["min"],))
        self.assertEqual(entry["max"], 0.7743407487869263,
                         "C2 MEASURED: max wrong: %r" % (entry["max"],))
        # Regression pin, exact float equality: this is the number the campaign
        # published as the so-called *restore* error; the survey must reproduce it
        # bit-for-bit off the rank payloads.
        self.assertEqual(entry["spread"], PUBLISHED_SPREAD,
                         "C2 MEASURED: spread moved off the published value: expected "
                         "%r, observed %r" % (PUBLISHED_SPREAD, entry["spread"]))
        self.assertIs(entry["identical_across_ranks"], False,
                      "C2 MEASURED: values differ across ranks; expected "
                      "identical_across_ranks False, observed %r"
                      % (entry["identical_across_ranks"],))
        self.assertEqual(entry["status"], "MEASURED",
                         "C2 MEASURED: status wrong: %r" % (entry["status"],))
        rc, _out, _err = self._run_main(
            [str(directory), "--world-size", "8", "--key", FIXED_KEY])
        self.assertEqual(rc, 0, "C2 MEASURED: expected exit 0, observed %r" % (rc,))

    # -- C3: everyone present, nobody recorded. Absence is not agreement. -----
    def test_c03_unmeasured(self):
        directory = self._dir("c3")
        self._write_rank_set(directory, [ABSENT] * 8)
        report = fsckpt.survey(directory, 8, (FIXED_KEY,))
        entry = report["keys"][FIXED_KEY]
        self.assertEqual(entry["per_rank"], [None] * 8,
                         "C3 UNMEASURED: omitted keys must read as None per rank: %r"
                         % (entry["per_rank"],))
        self.assertEqual(entry["recorded"], "0 of 8 ranks recorded this key",
                         "C3 UNMEASURED: denominator must read 0 of 8: %r"
                         % (entry["recorded"],))
        self.assertEqual(entry["status"], "UNMEASURED",
                         "C3 UNMEASURED: status wrong: %r" % (entry["status"],))
        # "nobody recorded it" must never read as "everybody agreed": no agreement
        # claim of either spellings may appear in an UNMEASURED entry.
        self.assertNotIn("identical_across_ranks", entry,
                         "C3 UNMEASURED: an agreement claim appeared over 0 of 8: %r"
                         % (entry,))
        self.assertNotIn("identical_across_present_ranks", entry,
                         "C3 UNMEASURED: a partial agreement claim appeared over 0 of "
                         "8: %r" % (entry,))
        rc, out, _err = self._run_main(
            [str(directory), "--world-size", "8", "--key", FIXED_KEY])
        self.assertEqual(rc, 95,
                         "C3 UNMEASURED: nothing to read must fail closed at 95, "
                         "observed %r" % (rc,))
        self.assertIn("UNMEASURED", out,
                      "C3 UNMEASURED: text report must say so: %r" % (out,))

    # -- C4: short denominator. This is the fix the harness exists to pin. ----
    def test_c04_partial(self):
        directory = self._dir("c4")
        # Only ranks 0..6 on disk; --world-size declares 8. rank-00007.pt is missing.
        self._write_rank_set(directory, [0.3375] * 7, ranks=range(7))
        report = fsckpt.survey(directory, 8, (FIXED_KEY,))
        entry = report["keys"][FIXED_KEY]
        self.assertEqual(report["missing_payloads"], ["rank-00007.pt"],
                         "C4 PARTIAL: the missing payload must be named: %r"
                         % (report["missing_payloads"],))
        self.assertEqual(report["ranks"], "7 of 8 rank payloads present",
                         "C4 PARTIAL: denominator wrong: %r" % (report["ranks"],))
        self.assertEqual(entry["status"], "PARTIAL",
                         "C4 PARTIAL: status wrong: %r" % (entry["status"],))
        self.assertIs(entry["identical_across_present_ranks"], True,
                      "C4 PARTIAL: the seven present payloads agree; expected the "
                      "qualified claim True, observed %r"
                      % (entry.get("identical_across_present_ranks"),))
        # The defect this harness pins: 'identical_across_ranks' asserted over 7 of 8
        # was a claim wider than its evidence. The unqualified key must NOT appear.
        self.assertNotIn("identical_across_ranks", entry,
                         "C4 PARTIAL: the unqualified agreement key must be absent "
                         "over a short denominator; observed entry %r" % (entry,))
        rc, out, _err = self._run_main(
            [str(directory), "--world-size", "8", "--key", FIXED_KEY])
        self.assertEqual(rc, 95,
                         "C4 PARTIAL: a short denominator must fail closed at 95, not "
                         "pass at 0; observed %r" % (rc,))
        self.assertIn("rank-00007.pt", out,
                      "C4 PARTIAL: the missing filename must be named in the report: "
                      "%r" % (out,))

    # -- C5: refusals on the calling shape. -----------------------------------
    def test_c05_refuse_bad_invocation(self):
        ghost = self.root / "does-not-exist"
        rc, _out, err = self._run_main([str(ghost), "--world-size", "8"])
        self.assertEqual(rc, 96,
                         "C5 REFUSE: a missing directory must exit 96, observed %r"
                         % (rc,))
        self.assertIn("not a directory", err,
                      "C5 REFUSE: stderr should say why: %r" % (err,))

        empty = self._dir("c5")
        rc, _out, err = self._run_main([str(empty), "--world-size", "0"])
        self.assertEqual(rc, 96,
                         "C5 REFUSE: --world-size 0 must exit 96, observed %r" % (rc,))
        self.assertIn("positive", err,
                      "C5 REFUSE: stderr should say why: %r" % (err,))

    # -- C6: refusals on malformed payloads. ----------------------------------
    def test_c06_refuse_malformed_payloads(self):
        # (a) Not a zip at all. zipfile attributes this failure by TYPE (BadZipFile)
        # rather than by path — "File is not a zip file" carries no filename — so the
        # naming half of this leg is pinned on case (b) below; here we pin the exit
        # code and the typed classification of the refusal.
        bad = self._dir("c6a")
        (bad / "rank-00000.pt").write_bytes(b"this is manifestly not a zip archive")
        rc, _out, err = self._run_main(
            [str(bad), "--world-size", "1", "--key", FIXED_KEY])
        self.assertEqual(rc, 96,
                         "C6 REFUSE: a non-zip payload must exit 96, observed %r"
                         % (rc,))
        self.assertIn("BadZipFile", err,
                      "C6 REFUSE: the refusal must be classified as a zip failure: %r"
                      % (err,))

        # (b) A zip with no data.pkl member: the module's own refusal, and it names
        # the offending file.
        empty_zip = self._dir("c6b")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("payload/weights.bin", b"\x00" * 16)
        (empty_zip / "rank-00000.pt").write_bytes(archive.getvalue())
        rc, _out, err = self._run_main(
            [str(empty_zip), "--world-size", "1", "--key", FIXED_KEY])
        self.assertEqual(rc, 96,
                         "C6 REFUSE: a zip with no data.pkl must exit 96, observed %r"
                         % (rc,))
        self.assertIn("rank-00000.pt", err,
                      "C6 REFUSE: the refusal must name the offending file: %r"
                      % (err,))

    # -- C7: data.pkl that unpickles to a non-dict. ----------------------------
    def test_c07_refuse_non_dict_top_level(self):
        directory = self._dir("c7")
        # A list, and it even contains a persistent-id storage reference so the stub
        # unpickler must survive it before the type check fires.
        _write_rank_payload(directory, 0, _pickle_document(
            [17, _StorageSentinel("0/0/list"), "x"]))
        rc, _out, err = self._run_main(
            [str(directory), "--world-size", "1", "--key", FIXED_KEY])
        self.assertEqual(rc, 96,
                         "C7 REFUSE: a non-dict data.pkl must exit 96, observed %r"
                         % (rc,))
        self.assertIn("rank-00000.pt", err,
                      "C7 REFUSE: the refusal must name the file: %r" % (err,))
        self.assertIn("list", err,
                      "C7 REFUSE: the refusal must name the offending type: %r"
                      % (err,))

    # -- C8: type discipline. Non-numeric recorded values are absent, not coerced.
    def test_c08_type_discipline(self):
        cases = (
            ("string", "recorded-as-text"),
            ("bool", True),
            ("none", None),
        )
        for label, value in cases:
            directory = self._dir("c8-%s" % label)
            self._write_rank_set(directory, [value])
            report = fsckpt.survey(directory, 1, (FIXED_KEY,))
            entry = report["keys"][FIXED_KEY]
            self.assertEqual(entry["per_rank"], [value],
                             "C8 (%s): per_rank must carry the recorded value "
                             "verbatim: %r" % (label, entry["per_rank"]))
            self.assertEqual(entry["recorded"], "0 of 1 ranks recorded this key",
                             "C8 (%s): a %r value must not enter the numeric "
                             "statistics: %r" % (label, value, entry["recorded"]))
            self.assertEqual(entry["status"], "UNMEASURED",
                             "C8 (%s): status wrong: %r" % (label, entry["status"]))
            self.assertNotIn("spread", entry,
                             "C8 (%s): no numeric statistics may appear: %r"
                             % (label, entry))
            self.assertNotIn("identical_across_ranks", entry,
                             "C8 (%s): no agreement claim over non-numerics: %r"
                             % (label, entry))
        # The trap this leg pins: isinstance(True, int) is True in Python; a bool must
        # NOT be counted as a numeric recording. The assert above already proved it
        # for the bool case ("0 of 1"), restated here as its own leg note.
        self.assertTrue(True,
                        "C8 bool: the isinstance(True, int) guard held (0 of 1 "
                        "recorded, UNMEASURED, no spread)")

    # -- C9: manifest independence. The leg that would have caught the old defect.
    def test_c09_manifest_independence(self):
        directory = self._dir("c9")
        self._write_rank_set(directory, [1.25] * 4)
        # The manifest scalar DISAGREES with every rank payload. If the per-rank read
        # ever came from manifest.json — one scalar standing in for four rank values —
        # this leg catches it.
        (directory / "manifest.json").write_text(
            json.dumps({FIXED_KEY: 9.999}))
        report = fsckpt.survey(directory, 4, (FIXED_KEY,))
        entry = report["keys"][FIXED_KEY]
        self.assertEqual(entry["per_rank"], [1.25, 1.25, 1.25, 1.25],
                         "C9: per_rank must come from the payloads, not the manifest: "
                         "%r" % (entry["per_rank"],))
        self.assertEqual(entry["manifest_value"], 9.999,
                         "C9: the manifest's differing number must be reported "
                         "separately: %r" % (entry["manifest_value"],))
        self.assertEqual(entry["spread"], 0.0,
                         "C9: spread must be computed over payloads only: %r"
                         % (entry["spread"],))
        self.assertIs(entry["identical_across_ranks"], True,
                      "C9: payload agreement must not be contaminated by the manifest: "
                      "%r" % (entry["identical_across_ranks"],))

    # -- C10: the whole premise. ----------------------------------------------
    def test_c10_no_torch(self):
        directory = self._dir("c10")
        self._write_rank_set(directory, [0.25] * 8)
        fsckpt.survey(directory, 8, (FIXED_KEY, "global_step", "world_size",
                                     "optimizer_state_count"))
        self.assertNotIn("torch", sys.modules,
                         "C10: 'torch' entered sys.modules during a full survey; the "
                         "reader's premise is that it never needs one")
        # Fixture hygiene: the synthesis surface must have been torn back down too.
        for leaked in _FAKE_MODULE_ATTRS:
            self.assertNotIn(leaked, sys.modules,
                             "C10: fixture module %r leaked out of its dump scope"
                             % (leaked,))


# --------------------------------------------------------------------------
# Runner: plain unittest, both `python3 -m unittest` and direct execution. The direct
# path prints the fs179 line with the denominator and exits 0 only if every leg held.
# --------------------------------------------------------------------------
def _run_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFsCkptScalars)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    legs_passed = result.testsRun - len(result.failures) - len(result.errors)
    print("fs179: %d/%d legs" % (legs_passed, result.testsRun))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(_run_suite())