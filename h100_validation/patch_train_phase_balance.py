#!/usr/bin/env python3
"""#173: close the segment-1 train phase and open the segment-2 one in fs_train.fixed.py.

WHAT / WHY. Measured on 8xH100 job 37304 (probe leg, 2026-09-01): the run reached
real distributed training -- load MEASURED at 8 of 8 rank/device pairs, data MEASURED at
4000 rows, and five TRAIN_JSON events with loss falling 1.2414 -> 1.1862 -> 0.9386 ->
0.7519 -> 0.5841 across steps 10..50 of 200 -- and then died at the step-50 early save
with RUN_SUMMARY_JSON {"reason":"contract_refused","verdict":"UNMEASURED",
"detail":"cannot begin save while train is open","phase":"absent"}.

The run trains TWICE -- segment 1 from step 0 to early, a checkpoint, a resume, then
segment 2 from early to budget -- but the train phase was opened once and closed once, in
DIFFERENT segments. Segment 1's train phase was never closed: after the train-ledger
extend idiom the code went straight to the save begin, which the phase machine refused
(the measured failure). Segment 2's train phase was never opened: the closing summary
after the continuation leg would have raised "cannot close train; open phase is None"
had the first defect not fired first. The two halves are mirror images, and only their
pairing kept the imbalance invisible to reading -- proof the train->save boundary had
never once executed.

THE FIX (two insertions, no redesign). (1) Immediately after the train-ledger extend
idiom, a denominated segment-1 train summary -- ledger key train.early, matching the
save.early / save.final convention -- whose closing summary ends the phase; its
peak-memory branch reuses the existing measured/unmeasured shape verbatim, and a
non-finite peak stays UNMEASURED, never a 0.0 substitute. (2) Immediately before the
continuation train_steps call, the train-phase begin segment 2 was missing. The phase
machine, the summary helper, and every emitted metric are untouched; the dedup reading
len(set(...)) over completed phases is verified present, so closing train twice cannot
inflate the measured-phase count.

THE GATE. A phase contract that a straight-line function can violate is an unenforced
contract, so this stage also ships a static, AST-based balance checker that walks the
target's main run function and proves every phase begin is closed by a same-phase
summary before the next begin, and that no summary closes a phase with no open
predecessor. Any future asymmetric edit fails the build. The checker is exercised
against synthetic function texts in a temp dir (MUST_PASS and MUST_FIRE controls) and is
observed RED on the real pre-image and GREEN on the post-image -- a gate never seen red
is not a gate. Controls need no torch and no transformers; the target is parsed, never
imported.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tempfile

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs173:"
N_CONTROLS = 6

# Needles are ASSEMBLED, never written as one literal: this stage counts them in the
# target, and a source that contains its own needle is inside its own denominator.
ANCHOR1 = "ledger.unmeasured.extend(" + "train" + "_ledger.unmeasured)"
ANCHOR2 = "global_step, peak_memory, continuation" + "_ledger = " + "train_steps("
DEDUP = "len(set(" + "machine.completed" + "))"
PHASE_SUMMARY = "_phase" + "_summary"
MACHINE_BEGIN = "machine" + "." + "begin("

SEG1_MARK = "# --- fs173: close segment-1 train phase"
SEG2_MARK = "# --- fs173: open segment-2 train phase"

# Insertion 1: the segment-1 close. The peak-memory measured/unmeasured branch is the
# verbatim shape of the existing final-train block; an unmeasured peak stays UNMEASURED.
SEG1_BLOCK = (
    "    " + SEG1_MARK + " (job 37304) ------------------------------\n"
    "    # Job 37304 died at the next line down: five TRAIN_JSON events fell\n"
    "    # 1.2414 -> 0.5841 over steps 10..50 of 200, then the save begin refused\n"
    "    # because this train phase was still open. Closing it here restores the\n"
    "    # first half of the mirror pair.\n"
    "    segment_metrics = {\n"
    '        "iterations": DenominatedCount(global_step, early, "iterations").payload(),\n'
    '        "peak_gpu_memory": (\n'
    "            {\n"
    '                "status": "measured",\n'
    '                "value": peak_memory,\n'
    '                "unit": "GiB",\n'
    '                "display": f"{peak_memory:.3f} GiB",\n'
    "            }\n"
    "            if math.isfinite(peak_memory)\n"
    "            else {\n"
    '                "status": "unmeasured",\n'
    '                "value": None,\n'
    '                "unit": "GiB",\n'
    '                "display": "peak GPU memory UNMEASURED",\n'
    "            }\n"
    "        ),\n"
    "    }\n"
    '    ledger.check("train.early", segment_metrics)\n'
    "    " + PHASE_SUMMARY + '(machine, "train", segment_metrics, context)\n'
    "    # --- end fs173 segment-1 ---\n"
)

# Insertion 2: the segment-2 open, the mirror image of the close above.
SEG2_BLOCK = (
    "    " + SEG2_MARK + " (job 37304) -------------------------------\n"
    "    # Mirror of the close above: the continuation leg trains from early to\n"
    "    # budget, but its train phase was never opened -- the final train summary\n"
    "    # would have closed None. Opening it here restores the second half of the\n"
    "    # pair. The two defects hid each other; only their pairing read as balanced.\n"
    "    " + MACHINE_BEGIN + '"train")\n'
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _classify_call(node: ast.Call) -> tuple[str, str] | None:
    """Recognise a phase begin or a closing summary with a literal phase name."""
    f = node.func
    if (
        isinstance(f, ast.Attribute)
        and f.attr == "begin"
        and isinstance(f.value, ast.Name)
        and f.value.id == "machine"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return ("begin", node.args[0].value)
    if (
        isinstance(f, ast.Name)
        and f.id == PHASE_SUMMARY
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "machine"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return ("close", node.args[1].value)
    return None


def _own_events(func: ast.AST) -> list[tuple[int, int, str, str]]:
    """Phase events of one function's own straight line, in source order.

    Nested function scopes are not descended into: their transitions are not this
    function's straight line.
    """
    events: list[tuple[int, int, str, str]] = []
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            hit = _classify_call(node)
            if hit is not None:
                events.append((node.lineno, node.col_offset, hit[0], hit[1]))
        stack.extend(ast.iter_child_nodes(node))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _balance_report(source: str) -> dict:
    """Walk the main run function and prove the phase transitions balance.

    Balance means: every begin(P) is followed, before the next begin, by a closing
    summary for the SAME P, and no summary closes a phase with no open predecessor.
    A source with ZERO begins is a zero-height denominator -- reported, never passed.
    """
    tree = ast.parse(source)
    best: ast.AST | None = None
    best_events: list[tuple[int, int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            evs = _own_events(node)
            if any(e[2] == "begin" for e in evs) and len(evs) > len(best_events):
                best, best_events = node, evs
    rep: dict = {
        "function": getattr(best, "name", None),
        "begins": 0,
        "balanced": 0,
        "train_segments": 0,
        "names": set(),
        "errors": [],
    }
    if best is None:
        return rep
    open_phase: str | None = None
    for lineno, _col, kind, phase in best_events:
        if kind == "begin":
            rep["begins"] += 1
            rep["names"].add(phase)
            if phase == "train":
                rep["train_segments"] += 1
            if open_phase is not None:
                rep["errors"].append(
                    f"line {lineno}: begin({phase!r}) refused: {open_phase!r} is still "
                    f"open and was never closed"
                )
            open_phase = phase
        else:
            if open_phase is None:
                rep["errors"].append(
                    f"line {lineno}: summary closes {phase!r} with no open predecessor"
                )
            elif open_phase != phase:
                rep["errors"].append(
                    f"line {lineno}: summary closes {phase!r} but the open phase is "
                    f"{open_phase!r}"
                )
                open_phase = None
            else:
                rep["balanced"] += 1
                open_phase = None
    if open_phase is not None:
        rep["errors"].append(
            f"phase {open_phase!r} abandoned open at end of {rep['function']}"
        )
    return rep


def _balance_gate_line(rep: dict) -> tuple[bool, str]:
    if rep["begins"] == 0:
        return False, (
            MARK + " 0 of 0 phase transitions balanced (0 phase name(s), 0 train "
            "segment(s)); zero-height denominator -- 95 UNMEASURED, never a pass"
        )
    ok = not rep["errors"] and rep["balanced"] == rep["begins"]
    line = (
        MARK + f" {rep['balanced']} of {rep['begins']} phase transitions balanced "
        f"({len(rep['names'])} phase name(s), {rep['train_segments']} train segment(s))"
    )
    if rep["errors"]:
        line += "; IMBALANCED: " + " | ".join(rep["errors"][:3])
    return ok, line


def _transform(text: str) -> tuple[str, dict[str, int], bool]:
    counts = {"anchor1": text.count(ANCHOR1), "anchor2": text.count(ANCHOR2)}
    if SEG1_MARK in text and SEG2_MARK in text:
        return text, counts, True
    out: list[str] = []
    for ln in text.splitlines(keepends=True):
        if ANCHOR2 in ln:
            out.append(SEG2_BLOCK)  # open segment 2 immediately before the continuation
            out.append(ln)
        elif ANCHOR1 in ln:
            out.append(ln)
            out.append(SEG1_BLOCK)  # close segment 1 immediately after the extend idiom
        else:
            out.append(ln)
    return "".join(out), counts, False


def _controls(pre: str, new: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    with tempfile.TemporaryDirectory(prefix="fs173-") as td:

        def synth(name: str, body: str) -> dict:
            p = pathlib.Path(td) / name
            p.write_text(body, "utf-8")
            return _balance_report(p.read_text("utf-8"))

        # C1 MUST_PASS: balanced begin/summary pairs, including a repeated phase name
        # (two train segments, the post-fix shape in miniature).
        r1 = synth("c1_balanced.py", (
            "def run():\n"
            "    " + MACHINE_BEGIN + '"load")\n'
            "    " + PHASE_SUMMARY + '(machine, "load", m, None)\n'
            "    " + MACHINE_BEGIN + '"train")\n'
            "    " + PHASE_SUMMARY + '(machine, "train", m, None)\n'
            "    " + MACHINE_BEGIN + '"train")\n'
            "    " + PHASE_SUMMARY + '(machine, "train", m, None)\n'
            "    " + MACHINE_BEGIN + '"save")\n'
            "    " + PHASE_SUMMARY + '(machine, "save", m, None)\n'
        ))
        good = (r1["begins"] == 4 and r1["balanced"] == 4 and not r1["errors"]
                and r1["train_segments"] == 2 and len(r1["names"]) == 3)
        ok += int(good)
        notes.append(
            f"C1 MUST_PASS balanced synthetic with a repeated phase name: "
            f"begins={r1['begins']} balanced={r1['balanced']} "
            f"train_segments={r1['train_segments']} errors={len(r1['errors'])} "
            + ("PASS" if good else "FAIL " + "; ".join(r1["errors"][:2]))
        )

        # C2 MUST_FIRE: the exact #173 shape -- a train begin followed by a save begin
        # -- is reported imbalanced, and the message names train.
        r2 = synth("c2_fs173_shape.py", (
            "def run():\n"
            "    " + MACHINE_BEGIN + '"train")\n'
            "    " + MACHINE_BEGIN + '"save")\n'
            "    " + PHASE_SUMMARY + '(machine, "save", m, None)\n'
        ))
        good = bool(r2["errors"]) and any("train" in e for e in r2["errors"])
        ok += int(good)
        notes.append(
            f"C2 MUST_FIRE the #173 shape (train begin, then save begin): "
            f"errors={len(r2['errors'])} names_train="
            f"{any('train' in e for e in r2['errors'])} "
            + ("PASS " + r2["errors"][0] if good else "FAIL checker stayed green on the measured defect")
        )

        # C3 MUST_FIRE: a summary closing a phase with no open predecessor.
        r3 = synth("c3_orphan_close.py", (
            "def run():\n"
            "    " + MACHINE_BEGIN + '"load")\n'
            "    " + PHASE_SUMMARY + '(machine, "load", m, None)\n'
            "    " + PHASE_SUMMARY + '(machine, "load", m, None)\n'
        ))
        good = any("no open predecessor" in e for e in r3["errors"])
        ok += int(good)
        notes.append(
            f"C3 MUST_FIRE summary with no open predecessor: errors={len(r3['errors'])} "
            + ("PASS " + r3["errors"][0] if good else "FAIL orphan close not detected")
        )

        # C4: ZERO begins is a zero-height denominator -- 95 UNMEASURED, never a pass.
        r4 = synth("c4_no_begins.py", "def run():\n    return 0\n")
        ok4, line4 = _balance_gate_line(r4)
        good = (r4["begins"] == 0 and not ok4 and "95" in line4 and "0 of 0" in line4)
        ok += int(good)
        notes.append(
            "C4 zero-begin synthetic reports 95 with the zero denominator stated: "
            + ("PASS " + line4 if good else f"FAIL begins={r4['begins']} line={line4!r}")
        )

        # C5 MUST_FIRE on the REAL pre-image: RED before, GREEN after. Both polarities
        # must be observed -- a gate never seen red is not a gate.
        r_pre = _balance_report(pre)
        r_new = _balance_report(new)
        red = r_pre["begins"] > 0 and (
            bool(r_pre["errors"]) or r_pre["balanced"] != r_pre["begins"]
        )
        green = (r_new["begins"] > 0 and not r_new["errors"]
                 and r_new["balanced"] == r_new["begins"])
        good = red and green
        ok += int(good)
        notes.append(
            f"C5 MUST_FIRE real target, both polarities observed: "
            f"pre(begins={r_pre['begins']} balanced={r_pre['balanced']} "
            f"errors={len(r_pre['errors'])})=RED->{red} "
            f"post(begins={r_new['begins']} balanced={r_new['balanced']} "
            f"errors={len(r_new['errors'])})=GREEN->{green} "
            + ("PASS" if good else "FAIL")
        )

        # C6: the peak-memory unmeasured branch is preserved in the inserted block --
        # the "unmeasured" spelling is present and no 0.0 substitute sneaks in.
        good = ('"unmeasured"' in SEG1_BLOCK
                and "peak GPU memory UNMEASURED" in SEG1_BLOCK
                and "0.0" not in SEG1_BLOCK)
        ok += int(good)
        notes.append(
            "C6 peak-memory unmeasured branch preserved (unmeasured spelling present, "
            "no 0.0 substitute): " + ("PASS" if good else "FAIL")
        )
    return ok, notes


def main() -> int:
    # The build driver invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would make the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_train_phase_balance.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    if not TARGET.exists():
        _stderr(f"UNMEASURED 95: target missing: {TARGET}")
        return 95
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    m1, m2 = text.count(SEG1_MARK), text.count(SEG2_MARK)
    if m1 or m2:
        if m1 == 1 and m2 == 1:
            # Second run: byte-exact no-op, but re-prove balance so an already-applied
            # yet imbalanced file is RED (5), not a silent pass.
            ok, line = _balance_gate_line(_balance_report(text))
            print("verdict: already applied; byte-idempotent no-op")
            print(line)
            return 0 if ok else 5
        _stderr(
            f"REFUSE 96: half-applied state (segment-1 marker x{m1}, segment-2 marker "
            f"x{m2}); the stage does not recognise this file and will not guess"
        )
        return 96

    new, counts, _already = _transform(text)

    gates = 0
    gres: list[tuple[str, bool, str]] = []
    gres.append(("G1", counts["anchor1"] == 1,
                 f"segment-1 extend anchor count={counts['anchor1']} need=1 (the "
                 "train_ledger spelling is unique; the continuation spelling belongs "
                 "to segment 2)"))
    gres.append(("G2", counts["anchor2"] == 1,
                 f"continuation-call anchor count={counts['anchor2']} need=1"))

    pre_rep = _balance_report(text)
    if pre_rep["begins"] == 0:
        _, line = _balance_gate_line(pre_rep)
        print(line)
        _stderr("UNMEASURED 95: no phase begins found in the target's run function; "
                "zero-height denominator")
        return 95
    premise = (len(pre_rep["errors"]) >= 2
               and any("train" in e for e in pre_rep["errors"])
               and any("no open predecessor" in e for e in pre_rep["errors"]))
    gres.append(("G3", premise,
                 f"MUST_FIRE premise, the pre-image is genuinely imbalanced: "
                 f"errors={len(pre_rep['errors'])} in {pre_rep['function']}: "
                 + " | ".join(pre_rep["errors"][:2])))
    gres.append(("G4", text.count(DEDUP) == 1 and new.count(DEDUP) == 1,
                 f"dedup reading len(set(...)) over completed phases "
                 f"pre={text.count(DEDUP)} post={new.count(DEDUP)} need=1/1 (verified, "
                 "not assumed: closing train twice must not inflate the measured-phase "
                 "count)"))
    post_rep = _balance_report(new)
    balanced_ok, balance_line = _balance_gate_line(post_rep)
    g5 = (balanced_ok
          and post_rep["begins"] == pre_rep["begins"] + 1
          and post_rep["train_segments"] == pre_rep["train_segments"] + 1 == 2)
    gres.append(("G5", g5, "post-image balance gate: " + balance_line))
    try:
        compile(new, str(TARGET), "exec")
        c6ok, c6msg = True, "clean"
    except SyntaxError as exc:
        c6ok, c6msg = False, f"SyntaxError: {exc}"
    gres.append(("G6", c6ok, "compile() " + c6msg))
    again, _, already2 = _transform(new)
    gres.append(("G7", again == new and already2, "byte-idempotence on own output"))
    nl = new.splitlines()
    i1 = next((i for i, l in enumerate(nl) if ANCHOR1 in l), None)
    i2 = next((i for i, l in enumerate(nl) if ANCHOR2 in l), None)
    placed = (
        i1 is not None and i2 is not None
        and SEG1_MARK in nl[i1 + 1]
        and nl[i2 - 1] == "    " + MACHINE_BEGIN + '"train")'
        # The marker heads the inserted block and the begin ends it, so the marker sits
        # a block-length above the anchor, not two lines above it. Search exactly that
        # window: a wider one would pass on a block placed somewhere else entirely.
        and any(
            SEG2_MARK in l
            for l in nl[max(0, i2 - len(SEG2_BLOCK.splitlines())) : i2]
        )
        and (len(nl[i2]) - len(nl[i2].lstrip()))
        == (len(nl[i2 - 1]) - len(nl[i2 - 1].lstrip()))
        and new.count(ANCHOR1) == 1 and new.count(ANCHOR2) == 1
    )
    gres.append(("G8", placed,
                 f"placement: anchor1@line{i1} anchor2@line{i2}; the segment-1 close "
                 "lands immediately after the extend idiom and the segment-2 begin "
                 "immediately before the continuation call, at matching indentation; "
                 "anchors still unique post-image"))

    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    cok, cnotes = _controls(text, new)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls "
                f"{cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  2 insertion(s) would be applied, {gates}/{len(gres)} "
              f"static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # This stage exists because a phase contract collapsed four states into one
    # refusal; it must not collapse its own: an unhandled exception is a REFUSE with a
    # named message, never a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())