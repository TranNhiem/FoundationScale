"""#83 staleness control: the emitter's torch_record prose must never again
contradict the gate's code.

The defect this guards shipped for months: the emitter asserted "the gate
writes no torch field ... (grep -c torch_record tools/live_save_gate.py
reads 0)" while the gate already wrote one, and the leg standing guard
pinned the STRING, never the GATE. This module connects the two sides.

GATE side, read statically and BY NAME: the AST of
src/foundationscale/gates/adjudication.py is parsed; the function
_interpreter_provenance must exist and its record
must carry the torch_record key. No count is grepped out of prose -- the
key's presence is re-derived from code on every run. An unparsable gate
errors the parse -- unreadable is not empty. A parsed gate WITHOUT the
named function raises LookupError -- missing is not zero, and prose must
never be graded against an unmeasured gate: the control fails CLOSED.
Only a parsed function lacking the key is a MEASURED zero -- the
pre-#83-gate-half state this control would have caught from the other
direction. DENOMINATOR: 1 named function sought, 1 expected key.

EMITTER side, read from the shipped strings on BOTH arms, each reached by
construction rather than host luck: a stub module forces the measured arm;
None in sys.modules halts `import torch` with ImportError, forcing the
ABSTAINED arm. DENOMINATOR: 2 arms, 3 entries each (executable, version,
torch_record), asserted below -- never zero units, and nothing here skips.

MUST_PASS: the real tree yields zero contradictions across 2 of 2 arms.
MUST_FIRE: the gate side is doctored two ways in tmp_path (the record
re-keyed; the function deleted outright) and the emitter side one way
(the stale retraction spliced back into a single arm), each requiring
red. A detector never observed firing is not a control.

Named abstention (estate idiom, cf. tests/test_lora_emit_coverage.py's
docstring): this estate does NOT yet compare the two torch_records at
runtime -- the emitter has no gate-report reader, so a runtime comparison
here would have to invent one. The follow-up shard needs exactly one fact
in context: the gate report's provenance block made reachable to the
emitting process (path or payload). It must then apply the mismatch
semantics the measured arm states: the two instruments differ by design
(import + __version__/__file__ vs find_spec + dist metadata), so a
mismatch indicts the instrument pairing first and convicts neither the
training build nor the model by itself.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
# The gate's decision API is a library module since T2_lib_script_boundary#0;
# tools/live_save_gate.py is an argparse wrapper that re-exports it. Parsing
# the wrapper would find no FunctionDef and raise LookupError -- which is the
# correct fail-closed behaviour for a moved seam, and the reason this pointer
# has to move with it rather than being widened to "either file".
GATE_PATH = REPO_ROOT / "src" / "foundationscale" / "gates" / "adjudication.py"
TOOLS_DIR = str(REPO_ROOT / "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import emit_run_manifest as erm  # noqa: E402

_GATE_FUNCTION = "_interpreter_provenance"
_GATE_FIELD = "torch_record"
# The exact vocabulary of the stale retraction: each phrase asserts -- or
# sources -- a gate that writes nothing. "grep -c" / "reads 0" also refuse
# the count-swap variant of the same defect: a fresh number frozen into
# prose drifts exactly like the old one did.
_DENIAL_PHRASES = (
    "writes no torch field",
    "not implemented",
    "no gate record",
    "grep -c",
    "reads 0",
)


def _gate_torch_record_key_count(gate_source: str) -> int:
    """torch_record dict keys carried by the gate's NAMED provenance
    function, read BY NAME from its own AST -- never a count grepped from
    prose. A parsed function without the key is a measured 0 (the gate
    regressed; prose naming a counterpart then reads red). A source with
    NO such function is not a zero: missing is not zero, so LookupError
    is raised and the control fails CLOSED instead of grading emitter
    prose against a gate it could not measure. DENOMINATOR: every
    FunctionDef in gate_source is examined for _GATE_FUNCTION.
    """
    for node in ast.walk(ast.parse(gate_source)):
        if isinstance(node, ast.FunctionDef) and node.name == _GATE_FUNCTION:
            return sum(
                1
                for child in ast.walk(node)
                if isinstance(child, ast.Dict)
                for key in child.keys
                if isinstance(key, ast.Constant) and key.value == _GATE_FIELD
            )
    raise LookupError(
        f"no function {_GATE_FUNCTION} in the parsed gate source: "
        "the seam moved; UNMEASURED, fail closed"
    )


def _emitter_torch_record_texts(monkeypatch) -> list[str]:
    """Both arms of _training_stack_entries, by construction, on any host:
    a stub torch module forces the measured arm; None in sys.modules makes
    `import torch` raise ImportError, forcing the ABSTAINED arm.
    DENOMINATOR: exactly 2 texts, one per arm; each arm's entry list
    carries exactly 3 keys (executable, version, torch_record), asserted
    inline, and each arm's identity is PROVEN (stub version string /
    ABSTAINED prefix), never assumed.
    """
    stub = types.ModuleType("torch")
    stub.__version__ = "0.0.0-stub"
    stub.__file__ = "<stub>/torch/__init__.py"
    texts = []
    monkeypatch.setitem(sys.modules, "torch", stub)
    entries = dict(erm._training_stack_entries())
    assert len(entries) == 3
    texts.append(entries["training_stack.torch_record"])
    monkeypatch.setitem(sys.modules, "torch", None)
    entries = dict(erm._training_stack_entries())
    assert len(entries) == 3
    texts.append(entries["training_stack.torch_record"])
    assert "0.0.0-stub" in texts[0]  # the measured arm really ran
    assert texts[1].startswith("ABSTAINED")  # the abstained arm really ran
    return texts


def _torch_provenance_contradictions(gate_source: str, emitter_texts: list[str]) -> list[str]:
    """Contradictions between the gate's REAL record and the emitter's
    shipped text, refused in BOTH directions -- a false alarm costs what
    a false green costs.

    Gate writes the field (count >= 1): an arm may not deny the gate's
    torch provenance (the #83 staleness), and it must NAME the instrument
    and field it claims as counterpart, so the relationship stays
    re-verifiable by THIS control rather than frozen into prose -- and
    green-by-silence (an arm claiming nothing at all) is refused too.
    Gate measured to write nothing (count == 0): an arm naming the
    instrument is the mirror-image lie this control would have caught
    before #83's gate half landed.
    An UNMEASURABLE gate (no named function) never reaches this logic:
    the count helper raises first -- fail closed.
    DENOMINATOR: len(emitter_texts) arms x their full text are graded.
    """
    key_count = _gate_torch_record_key_count(gate_source)
    contradictions = []
    for index, text in enumerate(emitter_texts):
        denials = [phrase for phrase in _DENIAL_PHRASES if phrase in text]
        names_counterpart = _GATE_FUNCTION in text and _GATE_FIELD in text
        if key_count >= 1:
            if denials:
                contradictions.append(
                    f"arm {index}: asserts {denials} while the gate's "
                    f"{_GATE_FUNCTION} carries {key_count} {_GATE_FIELD} "
                    "key(s)"
                )
            if not names_counterpart:
                contradictions.append(
                    f"arm {index}: names no gate counterpart "
                    f"({_GATE_FUNCTION}/{_GATE_FIELD}) while the gate "
                    "writes one"
                )
        elif names_counterpart:
            contradictions.append(
                f"arm {index}: names {_GATE_FUNCTION}/{_GATE_FIELD}, but "
                "the parsed gate carries no such key"
            )
    return contradictions


def test_must_pass_emitter_prose_matches_gate_record(monkeypatch):
    """MUST_PASS over the REAL tree. DENOMINATORS -- gate: 1 named
    function found, carrying exactly 1 torch_record key (measured, not
    assumed); emitter: 2 arms, 3 entries each, both really exercised
    (identity proven in the helper). Zero contradictions accepted from
    2 of 2 arms -- none vacuous: all([]) is never a PASS here, because
    the helper asserts both denominators before anything is graded.
    """
    gate_source = GATE_PATH.read_text(encoding="utf-8")
    assert _gate_torch_record_key_count(gate_source) == 1
    texts = _emitter_torch_record_texts(monkeypatch)
    assert len(texts) == 2
    assert _torch_provenance_contradictions(gate_source, texts) == []


def test_must_fire_gate_side_doctored(tmp_path, monkeypatch):
    """OBSERVED FIRING, gate side, two doctorings in tmp_path. First the
    record is re-keyed (torch_record -> renamed_record) while the emitter
    still names the old counterpart: a measured zero, and BOTH arms must
    read red. Then the function is deleted outright: the seam is MISSING,
    so the control must fail CLOSED with LookupError before any prose is
    consulted -- missing is not zero, UNMEASURED is never PASS.
    DENOMINATOR: 1 real gate measured (count 1, baseline) -> 1 re-keyed
    copy (count 0; 2 of 2 arms indicted) -> 1 gutted copy (0 matching
    functions; raises).
    """
    real = GATE_PATH.read_text(encoding="utf-8")
    assert _gate_torch_record_key_count(real) == 1  # measured baseline
    rekeyed = tmp_path / "live_save_gate_rekeyed.py"
    rekeyed.write_text(
        real.replace('"torch_record": {', '"renamed_record": {', 1),
        encoding="utf-8",
    )
    gate_source = rekeyed.read_text(encoding="utf-8")
    assert _gate_torch_record_key_count(gate_source) == 0
    contradictions = _torch_provenance_contradictions(
        gate_source, _emitter_torch_record_texts(monkeypatch)
    )
    # DENOMINATOR: 2 of 2 arms indicted -- both name the missing record
    assert len(contradictions) == 2

    gutted = tmp_path / "live_save_gate_gutted.py"
    gutted.write_text("def _unrelated_helper():\n    return {}\n", encoding="utf-8")
    with pytest.raises(LookupError):
        _torch_provenance_contradictions(
            gutted.read_text(encoding="utf-8"),
            _emitter_torch_record_texts(monkeypatch),
        )


def test_must_fire_emitter_reasserts_the_retraction(tmp_path, monkeypatch):
    """OBSERVED FIRING, emitter side: the stale retraction's vocabulary
    is spliced back into ONE arm via tmp_path; exactly that arm must read
    red while the clean arm stays green -- the control indicts the lying
    arm, never every arm (a false alarm costs what a false green costs).
    DENOMINATOR: 1 doctored arm of 2 examined; 1 of 2 indicted.
    """
    texts = _emitter_torch_record_texts(monkeypatch)
    doctored_arm = tmp_path / "abstained_arm.txt"
    doctored_arm.write_text(
        texts[1] + " Gate-side torch provenance is not implemented: the "
        "adjudicating gate writes no torch field on any exit path.",
        encoding="utf-8",
    )
    examined = [texts[0], doctored_arm.read_text(encoding="utf-8")]
    contradictions = _torch_provenance_contradictions(
        GATE_PATH.read_text(encoding="utf-8"), examined
    )
    # DENOMINATOR: 1 of 2 arms indicted; the undoctored arm stays green
    assert len(contradictions) == 1
    assert "arm 1" in contradictions[0]
    assert "writes no torch field" in contradictions[0]
