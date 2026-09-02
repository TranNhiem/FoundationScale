#!/usr/bin/env python3
"""fs206: make the held-out evaluation's collective count rank-invariant by construction.

THE DEFECT, AS MEASURED (job 37368, world=7, --eval-count 8, one 8-GPU H100 node)
---------------------------------------------------------------------------------
`evaluate_held_out` gives rank r the held-out rows at positions `p % world_size == r`,
then loops `range(0, len(rows), batch_size)` and calls `bundle.model(**batch)` on each
trip. That model is FSDP-wrapped, so every forward issues parameter all-gathers.

When the held-out count is not a multiple of the world size the low ranks own one more row
than the high ranks. With 8 rows over 7 ranks: rank 0 owns positions 0 and 7, ranks 1..6 own
one row each. At batch_size 1 that is 2 trips on rank 0 and 1 trip everywhere else -- one
extra all-gather that no other rank will ever match, because the other six ranks have
already fallen through to the terminal 4-element `dist.all_reduce(packet)`.

The log is unambiguous. At the SAME work sequence id, with `last completed work: 3141` on
every rank:

    [rank0] WorkNCCL(SeqNum=3142, OpType=_ALLGATHER_BASE,
                     NumelIn=3834588, NumelOut=26842116, Timeout(ms)=600000)
    [rank1] WorkNCCL(SeqNum=3142, OpType=ALLREDUCE, NumelIn=4, NumelOut=4, ...)
    [rank5] WorkNCCL(SeqNum=3142, OpType=ALLREDUCE, NumelIn=4, NumelOut=4, ...)
    [rank6] WorkNCCL(SeqNum=3142, OpType=ALLREDUCE, NumelIn=4, NumelOut=4, ...)

NumelOut/NumelIn is exactly 7, so rank 0's collective is a world-sized parameter gather --
it is still inside a forward. NumelIn=4 is exactly the `[loss_sum, measured_rows,
token_total, failures]` packet. Two different collectives paired at one sequence number:
NCCL blocks both sides, the watchdog fires at 600014 ms, and SIGABRT takes all seven ranks.
The job burned 15:10 of an 8-GPU-class allocation to produce no eval number.

WHY IT WAS INVISIBLE UNTIL NOW
------------------------------
Phase 3 ran world=8 over 8 held-out rows. 8 % 8 == 0, every rank ran exactly one trip, and
the loop was accidentally uniform. The framework was never wrong about the shape it was
developed at -- it was wrong about every other shape, and the shape that exposed it is the
most ordinary one there is: a node with one GPU already taken.

THE REMEDY, AND WHY IT IS THE FRAMEWORK'S AND NOT THE RUN'S
------------------------------------------------------------
Rejected: "require eval_count % world_size == 0". That is a constraint on the operator to
protect an implementation detail, it would have to be re-imposed at every future sharded
loop, and it fails closed only if someone remembers to write the guard.

Rejected: "run eval on rank 0 only". It changes the measurement (one rank's arithmetic) to
work around a control-flow bug, and it silently drops 6/7 of the throughput.

Shipped: the trip count becomes a GLOBAL constant, agreed with an all_reduce(MAX) *before*
the first forward -- the one place every rank is guaranteed to arrive together. Ranks that
run out of real rows still enter the forward, on a padding row every rank can name, and
their result is discarded. Padding enters no numerator and no denominator, so `examples`,
`tokens` and `loss` still describe exactly the real held-out rows.

The count of padding trips is then *published* in the eval record, summed across the world
through the existing reduction (the packet grows from 4 to 5 elements). It is nonzero
exactly when the pre-fix code would have deadlocked, which makes the run's own output the
detector for the shape that used to be fatal.

Residual, declared rather than papered over: the `except Exception` inside the loop still
admits a rank-local failure *mid-forward*, after some all-gathers have been issued and
before the rest. This stage does not fix that -- once a forward has partially executed, the
collectives it did not issue cannot be recovered from outside it -- and it is filed
separately. What this stage guarantees is that the NUMBER OF TRIPS is rank-invariant, which
is the mechanism that actually fired on hardware.
"""

from __future__ import annotations

import ast
import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs206:"
OPEN_MARK = "# --- fs206: rank-invariant collective counting for the held-out evaluation"
N_CONTROLS = 8  # C1..C8, of which C1 and C7 are MUST_FIREs.

FUNC = "evaluate_held_out"
DEF_EVAL = "def evaluate_held_out(\n"
AGREED = "_agreed_max"

# ---------------------------------------------------------------------------- pre-image --
# Byte-exact anchors. Every one is gated for uniqueness before a byte is written: a
# near-miss must REFUSE, never silently splice into the wrong place.

OLD_BODY = '''    sequence_length = int(config.sequence_length.value)
    loss_sum = 0.0
    token_total = 0
    failures = 0
    bundle.model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), int(config.batch_size.value)):
            chosen = rows[offset : offset + int(config.batch_size.value)]
            texts = [_text_at(context, index) for index in chosen]
            try:
                batch = _make_batch(bundle, texts, sequence_length)
                output = bundle.model(**batch)
                loss = getattr(output, "loss", None)
                if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise ValueError("invalid scalar held-out loss")
                loss_sum += float(loss.float().item()) * len(chosen)
                token_total += int(batch["attention_mask"].detach().sum().item())
            except Exception:
                failures += len(chosen)
'''

NEW_BODY = '''    sequence_length = int(config.sequence_length.value)
    batch_size = max(1, int(config.batch_size.value))
    # fs206: every trip below calls an FSDP-wrapped forward, and every such forward issues
    # parameter all-gathers. `rows` is sharded `position % world_size == rank`, so when the
    # held-out count is not a multiple of the world size the low ranks own one more row than
    # the high ranks -- one more trip, one more all-gather, matched by nobody, while the
    # short ranks are already blocked in the terminal reduction below. Measured, not feared:
    # job 37368 (world=7, 8 held-out rows) deadlocked with rank 0 in _ALLGATHER_BASE and
    # ranks 1/5/6 in the 4-element ALLREDUCE at the same SeqNum 3142, and NCCL's watchdog
    # SIGABRTed all seven ranks 600 s later. It never fired in development because that ran
    # world=8 over 8 rows, which divides. So the trip count must be a GLOBAL constant agreed
    # before the first forward, never a per-rank consequence of how the data happened to fall.
    trips = _agreed_max((len(rows) + batch_size - 1) // batch_size, "held-out eval trips")
    # A rank that has run out of real rows still has to enter the forward, so it repeats a
    # row every rank can name. Its result is discarded: padding enters no numerator and no
    # denominator, which is why `measured_rows` below still counts only this rank's own rows.
    pad_index = context.eval_rows[0] if context.eval_rows else None
    padding_trips = 0
    loss_sum = 0.0
    token_total = 0
    failures = 0
    bundle.model.eval()
    with torch.no_grad():
        for trip in range(trips):
            offset = trip * batch_size
            chosen = rows[offset : offset + batch_size]
            if chosen:
                texts = [_text_at(context, index) for index in chosen]
            elif pad_index is None:
                # Unreachable by construction: pad_index is None only when eval_rows is
                # empty, and then every rank's local trip count is 0, so `trips` is 0 and
                # this loop does not run. It raises rather than breaking on purpose -- a
                # `break` here would be a rank-dependent exit from a collective region,
                # which is the exact defect fs206 exists to remove.
                raise OperationFailure(
                    "fs206: a padding trip was required with no held-out row to pad"
                )
            else:
                padding_trips += 1
                texts = [_text_at(context, pad_index)]
            try:
                batch = _make_batch(bundle, texts, sequence_length)
                output = bundle.model(**batch)
                loss = getattr(output, "loss", None)
                if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise ValueError("invalid scalar held-out loss")
                if chosen:
                    loss_sum += float(loss.float().item()) * len(chosen)
                    token_total += int(batch["attention_mask"].detach().sum().item())
            except Exception:
                if chosen:
                    failures += len(chosen)
'''

OLD_PACKET = '''    packet = torch.tensor(
        [loss_sum, measured_rows, token_total, failures],
        dtype=torch.float64,
        device=bundle.device,
    )
'''

NEW_PACKET = '''    packet = torch.tensor(
        # fs206: the padding count rides the reduction that already exists rather than
        # buying a second collective, so publishing it costs nothing and cannot desync.
        [loss_sum, measured_rows, token_total, failures, float(padding_trips)],
        dtype=torch.float64,
        device=bundle.device,
    )
'''

OLD_RETURN = '''        "loss": loss_payload,
    }
'''

NEW_RETURN = '''        "loss": loss_payload,
        # fs206: nonzero exactly when the pre-fix code would have deadlocked. Published, not
        # merely fixed -- an operator reading this record can see that the world did not
        # divide the held-out set evenly and that the framework absorbed it.
        "padding_trips": DenominatedCount(
            int(packet[4].item()),
            trips * bundle.world_size,
            "padding trips across all ranks (uneven held-out shard)",
        ).payload(),
    }
'''

HELPER_BLOCK = '''# --- fs206: rank-invariant collective counting for the held-out evaluation --------------
EVAL_TRIP_CENSUS: dict[str, Any] = {}


def _agreed_max(local: int, what: str) -> int:
    """Raise a per-rank count to the global maximum so a loop's collectives can match.

    A collective inside a per-rank loop is safe only when every rank runs the loop the same
    number of times, and that agreement must be reached with a collective every rank is
    guaranteed to arrive at -- BEFORE the loop. Reaching it afterwards is precisely the
    deadlock this exists to prevent.

    At world_size 1, or with no initialised process group, the answer is recorded as
    UNMEASURED rather than as agreement: a single rank cannot disagree with itself, and
    calling that a PASS would be the vacuous truth this framework refuses to publish.
    """
    if not (dist.is_available() and dist.is_initialized()):
        EVAL_TRIP_CENSUS[what] = {
            "local": int(local),
            "agreed": int(local),
            "world_size": 1,
            "status": "unmeasured",
            "reason": "UNMEASURED: no initialised process group, so no rank could disagree",
        }
        return int(local)
    world = int(dist.get_world_size())
    if world <= 1:
        EVAL_TRIP_CENSUS[what] = {
            "local": int(local),
            "agreed": int(local),
            "world_size": world,
            "status": "unmeasured",
            "reason": "UNMEASURED: a single rank cannot disagree with itself",
        }
        return int(local)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    holder = torch.tensor([int(local)], dtype=torch.int64, device=device)
    dist.all_reduce(holder, op=dist.ReduceOp.MAX)
    agreed = int(holder.item())
    EVAL_TRIP_CENSUS[what] = {
        "local": int(local),
        "agreed": agreed,
        "world_size": world,
        "status": "measured",
        "reason": f"{what}: this rank owned {int(local)}, the world agreed on {agreed}",
    }
    return agreed


'''


def _stderr(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _transform(text: str) -> tuple[str, bool]:
    """Return (patched, changed). Byte-idempotent: a second application is a no-op."""
    if OPEN_MARK in text:
        return text, False
    new = text.replace(DEF_EVAL, HELPER_BLOCK + DEF_EVAL, 1)
    new = new.replace(OLD_BODY, NEW_BODY, 1)
    new = new.replace(OLD_PACKET, NEW_PACKET, 1)
    new = new.replace(OLD_RETURN, NEW_RETURN, 1)
    return new, new != text


def _func(src: str, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _assigned(fn: ast.FunctionDef, target: str) -> ast.AST | None:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == target:
                return node.value
    return None


class _Ctx:
    def __init__(self, rows):
        self.eval_rows = list(rows)


class _Bundle:
    def __init__(self, rank, world):
        self.rank = rank
        self.world_size = world


class _Cfg:
    def __init__(self, bs):
        self.batch_size = type("V", (), {"value": bs})()


def _shipped_shard(src: str, fn: ast.FunctionDef, rows_n: int, rank: int, world: int):
    """Evaluate the SHIPPED row-sharding expression -- not a re-implementation of it."""
    expr = _assigned(fn, "rows")
    seg = ast.get_source_segment(src, expr)
    return eval(seg, {}, {"context": _Ctx(range(rows_n)), "bundle": _Bundle(rank, world)})


def _controls(new: str, pre: str) -> tuple[int, list[str]]:
    ok, notes = 0, []

    def rec(tag, good, detail):
        nonlocal ok
        ok += int(good)
        notes.append(f"{tag}: {'OBSERVED' if good else 'NOT OBSERVED'}  {detail}")

    pre_fn = _func(pre, FUNC)
    new_fn = _func(new, FUNC)

    # C1 (MUST_FIRE): run the PRE-IMAGE's own loop header over the shape that actually
    # deadlocked and count its trips per rank. If they come out equal, this instrument is
    # dead and every green below it is unattributable.
    pre_for = next(n for n in ast.walk(pre_fn) if isinstance(n, ast.For))
    pre_iter = ast.get_source_segment(pre, pre_for.iter)
    pre_trips = []
    for r in range(7):
        rows = _shipped_shard(pre, pre_fn, 8, r, 7)
        pre_trips.append(len(list(eval(pre_iter, {}, {"rows": rows, "config": _Cfg(1)}))))
    rec(
        "C1",
        len(set(pre_trips)) != 1 and pre_trips == [2, 1, 1, 1, 1, 1, 1],
        f"MUST_FIRE: the pre-image loop runs {pre_trips} trips across 7 ranks over 8 rows "
        f"-- {len(set(pre_trips))} distinct counts, so rank 0 issues an all-gather nobody "
        f"matches (this is job 37368)",
    )

    # C2: the SHIPPED trip expression, on that same shape, is one number on every rank.
    trips_call = _assigned(new_fn, "trips")
    assert isinstance(trips_call, ast.Call), "trips must be assigned from a call"
    local_expr = ast.get_source_segment(new, trips_call.args[0])
    locals_7 = []
    for r in range(7):
        rows = _shipped_shard(new, new_fn, 8, r, 7)
        locals_7.append(eval(local_expr, {}, {"rows": rows, "batch_size": 1}))
    agreed_7 = max(locals_7)
    pad_7 = agreed_7 * 7 - 8
    rec(
        "C2",
        locals_7 == [2, 1, 1, 1, 1, 1, 1] and agreed_7 == 2 and pad_7 == 6,
        f"the same shape now agrees on {agreed_7} trip(s) for all 7 ranks (locals {locals_7}), "
        f"with {pad_7} padding trips carrying no real row",
    )

    # C3: no cost on the shape that already worked. If the fix perturbed the even case it
    # would invalidate the Phase 3 measurements taken at world=8.
    locals_8 = [
        eval(local_expr, {}, {"rows": _shipped_shard(new, new_fn, 8, r, 8), "batch_size": 1})
        for r in range(8)
    ]
    rec(
        "C3",
        locals_8 == [1] * 8 and max(locals_8) * 8 - 8 == 0,
        "world=8 over 8 rows is untouched: 1 trip per rank, 0 padding -- the fix is inert "
        "on the shape Phase 3 measured",
    )

    # C4/C5: _agreed_max must never call a solo rank's answer agreement.
    src = ast.get_source_segment(new, _func(new, AGREED))
    ns: dict = {"Any": object}

    def _run(world_or_none, local):
        if world_or_none is None:
            d = type("D", (), {
                "is_available": staticmethod(lambda: False),
                "is_initialized": staticmethod(lambda: False),
            })()
        else:
            d = type("D", (), {
                "is_available": staticmethod(lambda: True),
                "is_initialized": staticmethod(lambda: True),
                "get_world_size": staticmethod(lambda group=None: world_or_none),
            })()
        env = dict(ns)
        env["dist"] = d
        env["torch"] = None
        env["EVAL_TRIP_CENSUS"] = {}
        exec(compile(src, "<fs206-agreed>", "exec"), env)
        return env[AGREED](local, "probe"), env["EVAL_TRIP_CENSUS"]["probe"]

    v0, c0 = _run(None, 3)
    rec(
        "C4",
        v0 == 3 and c0["status"] == "unmeasured" and "no initialised" in c0["reason"],
        f"no process group -> returns the local {v0} and records status={c0['status']} "
        "with a stated reason, not a vacuous agreement",
    )
    v1, c1 = _run(1, 5)
    rec(
        "C5",
        v1 == 5 and c1["status"] == "unmeasured" and "cannot disagree" in c1["reason"],
        f"world_size=1 -> status={c1['status']} (a single rank cannot PASS an agreement)",
    )

    # C6: with a real world the value is the MAX and the record says measured.
    seen = {}

    class _T:
        int64 = "int64"

        @staticmethod
        def tensor(v, dtype=None, device=None):
            seen["v"] = list(v)
            return type("H", (), {"item": staticmethod(lambda: max(seen["v"] + [9]))})()

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(*a):
            return "cpu"

    env = {
        "Any": object,
        "dist": type("D", (), {
            "is_available": staticmethod(lambda: True),
            "is_initialized": staticmethod(lambda: True),
            "get_world_size": staticmethod(lambda group=None: 4),
            "all_reduce": staticmethod(lambda t, op=None: None),
            "ReduceOp": type("R", (), {"MAX": "MAX"}),
        })(),
        "torch": _T,
        "EVAL_TRIP_CENSUS": {},
    }
    exec(compile(src, "<fs206-agreed>", "exec"), env)
    v2 = env[AGREED](2, "probe")
    c2 = env["EVAL_TRIP_CENSUS"]["probe"]
    rec(
        "C6",
        v2 == 9 and c2["status"] == "measured" and c2["local"] == 2 and c2["world_size"] == 4,
        f"world_size=4 -> returns the reduced maximum ({v2}) over the local {c2['local']}, "
        f"status={c2['status']}",
    )

    # C7 (MUST_FIRE): the pre-image must carry the anchors and NOT the mark. A stage whose
    # gates are already satisfied on the unpatched text is measuring nothing.
    pre_bad = (
        OPEN_MARK not in pre
        and AGREED not in pre
        and pre.count(OLD_BODY) == 1
        and pre.count(OLD_PACKET) == 1
    )
    rec(
        "C7",
        pre_bad,
        "MUST_FIRE: the pre-image carries the unpatched loop and packet exactly once each "
        "and neither the helper nor the mark",
    )

    # C8: padding must sit in NO numerator. Proved against the shipped AST rather than a
    # re-implementation: every mutation of loss_sum/token_total/failures inside the loop
    # must be lexically guarded by `if chosen:`.
    new_for = next(n for n in ast.walk(new_fn) if isinstance(n, ast.For))
    guarded, total = 0, 0
    for node in ast.walk(new_for):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "chosen":
            for inner in ast.walk(node):
                if isinstance(inner, ast.AugAssign) and isinstance(inner.target, ast.Name):
                    if inner.target.id in {"loss_sum", "token_total", "failures"}:
                        guarded += 1
    for node in ast.walk(new_for):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"loss_sum", "token_total", "failures"}:
                total += 1
    rec(
        "C8",
        total == 3 and guarded == 3,
        f"{guarded} of {total} accumulator mutations sit under `if chosen:` (need 3 of 3), "
        "so a padding trip contributes to no numerator and to no denominator",
    )
    return ok, notes


def main() -> int:
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_eval_collective_invariance.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: cannot read {TARGET}: {exc}")
        return 95

    new, changed = _transform(text)
    gres: list[tuple[str, bool, str]] = []

    gres.append(("G1", text.count(OLD_BODY) == 1,
                 f"the unpatched eval loop is a unique anchor ({text.count(OLD_BODY)}, need 1)"))
    gres.append(("G2", text.count(OLD_PACKET) == 1 and new.count(OLD_PACKET) == 0
                 and new.count(NEW_PACKET) == 1,
                 f"the 4-element packet is unique pre ({text.count(OLD_PACKET)}) and fully "
                 f"replaced post ({new.count(NEW_PACKET)} 5-element, {new.count(OLD_PACKET)} old)"))
    gres.append(("G3", text.count(DEF_EVAL) == 1 and new.count(DEF_EVAL) == 1,
                 "evaluate_held_out is defined exactly once, pre and post"))

    i_helper, i_def = new.find(OPEN_MARK), new.find(DEF_EVAL)
    gres.append(("G4", -1 < i_helper < i_def,
                 f"the helper lands immediately before evaluate_held_out ({i_helper} < {i_def})"))

    fn = _func(new, FUNC)
    # Every ordering and multiplicity claim below is about THIS function, so it is measured
    # inside THIS function. Scoping them to the file was the first thing this stage got
    # wrong: `dist.all_reduce(packet` also matches `packet[0:4]` in the throughput reduction
    # 180 lines earlier, and `bundle.model(**batch)` has three call sites in the module. A
    # gate whose denominator is wider than its claim reports a defect that is not there --
    # and the same widening in the other direction would have reported a green that is not
    # there either.
    body = ast.get_source_segment(new, fn) if fn is not None else ""
    i_trips, i_loop, i_reduce = (body.find("    trips = _agreed_max("),
                                 body.find("        for trip in range(trips):"),
                                 body.find("dist.all_reduce(packet,"))
    gres.append(("G5", -1 < i_trips < i_loop < i_reduce,
                 f"within {FUNC}: the agreement precedes the loop which precedes the reduction "
                 f"({i_trips} < {i_loop} < {i_reduce}); agreeing AFTER the loop is the deadlock"))
    gres.append(("G6", "range(0, len(rows)" not in new and body.count("range(trips)") == 1,
                 "no rank-local loop bound survives: the data-dependent `range(0, len(rows)...)` "
                 "is gone module-wide and exactly one `range(trips)` replaces it here"))
    gres.append(("G7", body.count("dist.all_reduce(packet,") == 1
                 and body.count("bundle.model(**batch)") == 1
                 and new.count("bundle.model(**batch)") == text.count("bundle.model(**batch)"),
                 f"within {FUNC}: still exactly one terminal reduction and one forward call "
                 f"site, and the module-wide forward count is unchanged "
                 f"({new.count('bundle.model(**batch)')} == {text.count('bundle.model(**batch)')}) "
                 "-- this stage changes how often the loop runs, never how the measurement is taken"))

    bound_ok = False
    if fn is not None:
        for node in ast.walk(fn):
            if isinstance(node, ast.For):
                names = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}
                bound_ok = "rows" not in names and "trips" in names
                break
    gres.append(("G8", bound_ok,
                 "AST: the loop bound names `trips` and does NOT name `rows` -- the trip count "
                 "cannot be a function of this rank's shard"))

    compiled = True
    try:
        compile(new, str(TARGET), "exec")
    except SyntaxError as exc:
        compiled = False
        _stderr(f"patched text does not compile: {exc}")
    gres.append(("G9", compiled, "the patched module compiles"))

    again, changed_again = _transform(new)
    gres.append(("G10", again == new and not changed_again,
                 "byte-idempotence on own output (a second run is a byte-exact no-op)"))
    gres.append(("G11", changed or OPEN_MARK in text,
                 "the transform either changed the file or the mark was already present"))

    gates = 0
    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    try:
        cok, cnotes = _controls(new, text)
    except Exception as exc:  # noqa: BLE001
        cok, cnotes = 0, [f"controls raised {type(exc).__name__}: {exc}"]
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls {cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # A stage whose whole purpose is to stop a cross-rank deadlock must not collapse its own
    # states: an unhandled exception is a REFUSE, not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    sys.exit(_guarded())
