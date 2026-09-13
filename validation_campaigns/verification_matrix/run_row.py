#!/usr/bin/env python3
# validation_campaigns/verification_matrix/run_row.py
# CLI:
#   run_row.py --row T1-9 --out-dir DIR --pass-id P [--dry-run]
#   run_row.py --all --out-dir DIR --pass-id P     (every row with an adjudicator)
#   run_row.py --self-test
"""The RUNNER for the verification matrix -- the piece whose absence left five
verdicts existing only in a shell scrollback.

Reads matrix.json, resolves the row's `adjudicator` field, introspects its
--help, composes argv, runs it, collects the arm payloads, writes the receipt.
--dry-run composes and PRINTS the argv without executing -- so the wiring can
be inspected without burning a GPU.

The defect this module exists to prevent (#416): row T1-6 was recorded
UNMEASURED because an operator passed --out-dir to an adjudicator that has no
run mode at all, and the campaign wrote the consequence of that mis-wiring
down as a fact about the CLAIM. The rules here follow from that:

  1. The argv is built ONLY from long flags the adjudicator's --help actually
     declares. A flag the program does not accept is impossible to send, by
     construction -- never from an assumption.
  2. An adjudicator with no run mode (no --out-dir in --help) is a REFUSAL
     (96) naming the adjudicator and the missing flag -- a defect in the
     INSTRUMENT, never an unmeasured row and never a red.
  3. The model root and dataset arrive ONLY from FS_T1_MODEL and FS_T1_DATASET,
     with no default and no guessing (this is a public repository; no estate
     paths). An absent variable is UNMEASURED (95), with a reason naming the
     variable.
  4. Results become receipts through receipts.write_receipt. matrix.json is
     the declaration and is never written. The runner is the caller, so the
     runner takes the clock and passes written_at_utc in.
  5. Arms are MEASURED, not narrated: the receipt's `arms` mapping is built
     from the <out-dir>/<row>_<arm>.json files the adjudicator wrote. An exit
     of 0 or 5 with zero arm files is the all([]) trap and REFUSES 96.
  6. Four-state exit contract everywhere: 0 CLEAR, 5 RED, 95 UNMEASURED,
     96 REFUSE. Never 1, never 2 -- argparse usage errors included. An
     adjudicator exit code outside receipts.VERDICTS is a REFUSAL naming the
     code, because a program that broke its own contract has not said what it
     measured.
  7. Stdlib only. receipts.py is imported by path from beside this file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
MATRIX_PATH = MODULE_DIR / "matrix.json"
RECEIPTS_ROOT = MODULE_DIR / "receipts"

MODEL_ENV_VAR = "FS_T1_MODEL"
DATASET_ENV_VAR = "FS_T1_DATASET"
REQUIRED_ENV_VARS = (MODEL_ENV_VAR, DATASET_ENV_VAR)

# --help must answer quickly; the GPU run itself gets no timeout, because a
# 15-minute-per-arm training job would be killed by any bound we picked here.
HELP_TIMEOUT_SECONDS = 120

# The scalar fields lifted out of each <row>_<arm>.json payload into the
# receipt's arms mapping. The axis key (optimizer, accumulation, ...) is
# deliberately NOT here: it is per-row and must not be hard-coded.
ARM_SCALAR_KEYS = ("status", "reason", "launcher_exit_code", "telemetry", "loss_curve")

_LONG_FLAG_RE = re.compile(r"--([A-Za-z0-9][A-Za-z0-9-]*)")


def _import_receipts():
    """Import receipts.py by path from beside this file (stdlib only, no
    package installation assumed on the gate machine)."""
    path = MODULE_DIR / "receipts.py"
    if not path.is_file():
        raise ImportError(
            f"run_row.py requires receipts.py beside it at {path}; the runner cannot "
            "write a verdict it cannot shape"
        )
    spec = importlib.util.spec_from_file_location("verification_matrix_receipts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receipts = _import_receipts()


class _UsageError(Exception):
    """Internal: argparse wanted to exit(2); the contract forbids that."""


class ContractArgumentParser(argparse.ArgumentParser):
    """WHY a subclass: stock argparse exits 2 on a usage error, and 2 is not a
    state the campaign can record -- an operator reading '2' cannot tell a typo
    from a crash from a refutation. Usage problems REFUSE (96); --help still
    exits 0, which is the contract's CLEAR."""

    def error(self, message):  # argparse's exit-2 path
        raise _UsageError(message)

    def exit(self, status=0, message=None):
        if status:
            raise _UsageError(message or f"argument parsing failed (argparse status {status})")
        if message:
            print(message)
        raise SystemExit(0)


def load_matrix(matrix_path=None):
    """WHY: matrix.json is the DECLARATION of rows. This function only ever
    reads it -- a file that is both the claim and its own result is
    self-certifying, so the runner never writes here. The on-disk top-level
    shape is tolerated (a bare list of rows, or an object wrapping one) so the
    runner depends on the rows, not on the envelope."""
    path = Path(matrix_path) if matrix_path is not None else MATRIX_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "matrix", "verification_matrix"):
            if isinstance(data.get(key), list):
                return data[key]
        if data and all(isinstance(value, dict) for value in data.values()):
            return list(data.values())
    raise ValueError(f"{path}: cannot find the list of matrix rows in this JSON shape")


def help_flags(help_text):
    """WHY: the argv is built from what the adjudicator DECLARES, never from an
    assumption about what it should accept. This set is the entire authority on
    which flags may be sent. Names are normalized ('out_dir' vs '--out-dir')
    so spelling never decides what is runnable."""
    return {
        match.group(1).replace("_", "-").lower() for match in _LONG_FLAG_RE.finditer(help_text or "")
    }


def _normalize_flag(name):
    return str(name).lstrip("-").replace("_", "-").lower()


def compose_argv(*, interpreter, adjudicator_path, out_dir, declared_flags, requested_flags=None):
    """WHY: this is the single place where operator intent meets the program's
    declared interface, and the declared interface wins. A requested flag that
    is absent from --help is DROPPED, so a flag the program does not accept is
    impossible to send by construction -- the exact wiring error that turned
    T1-6's refusal into a fake measurement. --out-dir comes solely from the
    runner's own --out-dir argument, and composing for a program that does not
    declare it is a hard error: there is no run mode to compose for."""
    declared = {_normalize_flag(flag) for flag in declared_flags}
    if "out-dir" not in declared:
        raise ValueError(
            f"compose_argv: {adjudicator_path!r} does not declare --out-dir in its --help; "
            "it has no run mode and must be refused (96), not composed for"
        )
    argv = [str(interpreter), str(adjudicator_path), "--out-dir", str(out_dir)]
    for raw_name, value in (requested_flags or {}).items():
        name = _normalize_flag(raw_name)
        if name in ("out-dir", "help"):
            continue  # out-dir is already placed; --help mid-run is not a run.
        if name not in declared:
            continue  # the point of the module: undeclared flags never reach the argv.
        if value is None:
            argv.append(f"--{name}")
        else:
            argv += [f"--{name}", str(value)]
    return argv


def _arm_prefix(row_id):
    """'T1-9' -> 't1_9', matching the <ROW>_<arm>.json names the adjudicators
    write. Derived, never hard-coded per row."""
    return re.sub(r"[^a-z0-9]+", "_", str(row_id).lower()).strip("_")


def collect_arms(out_dir, *, row_id):
    """WHY: the arms are MEASURED, not narrated. The receipt's arms mapping is
    built solely from the <out-dir>/<row>_<arm>.json payloads the adjudicator
    wrote -- stdout is never scraped for numbers, because a number that only
    existed in a scrollback is exactly the defect being fixed. A payload that
    cannot be read is not an observation and raises, so the caller can refuse
    rather than record a partial truth."""
    prefix = _arm_prefix(row_id)
    directory = Path(out_dir)
    arms = {}
    for path in sorted(directory.glob(f"{prefix}_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path}: arm payload is unreadable ({exc}); an observation that "
                "cannot be read is not an observation"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"{path}: arm payload is {type(payload).__name__}, not an object; "
                "it is not an observation"
            )
        arm = path.stem[len(prefix) + 1 :]
        if not arm:
            raise ValueError(f"{path}: filename carries no arm name after the row prefix")
        arms[arm] = {key: payload[key] for key in ARM_SCALAR_KEYS if key in payload}
    return arms


def _output_tail(stdout, stderr):
    lines = [
        line.strip()
        for line in ((stdout or "") + "\n" + (stderr or "")).splitlines()
        if line.strip()
    ]
    return lines[-1][:300] if lines else ""


def _child_env(env_map):
    merged = dict(os.environ)
    merged.update(env_map)
    return merged


def run_row(
    *,
    row,
    out_dir,
    pass_id,
    receipts_root,
    written_at_utc,
    repo_root=None,
    requested_flags=None,
    env=None,
    dry_run=False,
    interpreter=None,
    help_timeout=HELP_TIMEOUT_SECONDS,
):
    """WHY: this is the whole design as one pipeline. Each early exit is a
    verdict with a reason that says WHICH input was wrong -- the adjudicator,
    the environment, or the contract -- because the T1-6 defect was precisely
    a reason that blamed the claim for the wiring. Every terminal state is a
    receipt written through receipts.write_receipt with the caller-supplied
    written_at_utc (the runner takes the clock; receipts.py never does).

    Order of checks matters and is deliberate:
      adjudicator declared? -> adjudicator file exists? -> does --help offer a
      run mode? (REFUSE 96, the instrument is defective) -> only then the
      environment (UNMEASURED 95, the inputs are missing). A row that cannot
      run at all must refuse even on a perfect machine, and a runnable row on
      an input-less machine is unmeasured, not refused. --dry-run prints the
      composed argv and writes NOTHING, because nothing was measured."""
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    interpreter = interpreter or sys.executable
    env_map = dict(os.environ if env is None else env)
    requested = dict(requested_flags or {})
    out_dir = Path(out_dir)
    row_id = row.get("id") if isinstance(row, Mapping) else None

    result = {
        "row_id": row_id,
        "argv": None,
        "exit_code": None,
        "verdict": None,
        "reason": None,
        "receipt_path": None,
        "receipt": None,
    }

    def _finish(code, reason, arms=None, adj_field=None, adj_bytes=None):
        result["exit_code"] = code
        result["verdict"] = receipts.VERDICTS[code]
        result["reason"] = reason
        if dry_run:
            print(f"dry-run: would record {code} ({receipts.VERDICTS[code]}): {reason}")
            return result
        path = receipts.write_receipt(
            receipts_root=receipts_root,
            row=row,
            pass_id=pass_id,
            exit_code=code,
            reason=reason,
            arms=arms or {},
            adjudicator_path=adj_field,
            adjudicator_bytes=adj_bytes,
            written_at_utc=written_at_utc,
        )
        result["receipt_path"] = path
        result["receipt"] = json.loads(path.read_text(encoding="utf-8"))
        return result

    # -- 1. The row must name a decider. No adjudicator is not a refusal --
    # the row simply has nothing to execute; it is UNMEASURED.
    adj_field = row.get("adjudicator") if isinstance(row, Mapping) else None
    if not isinstance(adj_field, str) or not adj_field.strip():
        return _finish(
            95,
            f"UNMEASURED 95: row {row_id!r} declares no 'adjudicator'; there is no "
            "program that decides this claim today, so it is unmeasured -- not green, not red",
        )

    # -- 2. A named decider with no bytes behind it is a broken declaration:
    # REFUSE 96 (mirrors receipts.row_verdict rule 3). The receipt records
    # adjudicator=None because a path without its bytes is not evidence.
    adj_path = repo_root / adj_field
    if not adj_path.is_file():
        return _finish(
            96,
            f"REFUSED 96: row {row_id} names adjudicator '{adj_field}' but no such file "
            f"exists under {repo_root}; a row whose decider is missing refuses -- it is "
            "not a red, because nothing was measured and nothing failed",
        )
    try:
        adj_bytes = adj_path.read_bytes()
    except OSError as exc:
        return _finish(96, f"REFUSED 96: adjudicator '{adj_field}' is unreadable: {exc}")

    # -- 3. Introspect. The --help text is the ONLY authority on the argv.
    # Its own exit code is irrelevant to parsing (t1_6's --help exits 95);
    # what matters is which long flags it prints.
    try:
        help_proc = subprocess.run(
            [interpreter, str(adj_path), "--help"],
            capture_output=True,
            text=True,
            timeout=help_timeout,
            env=_child_env(env_map),
        )
        help_output = (help_proc.stdout or "") + "\n" + (help_proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _finish(
            96,
            f"REFUSED 96: could not introspect adjudicator '{adj_field}' with --help "
            f"({exc}); refusing to guess an interface the program would not state",
        )
    flags = help_flags(help_output)

    # -- 4. No run mode: REFUSE 96 naming the adjudicator AND the missing
    # flag. This is the T1-6 case: a defect in the INSTRUMENT, reported as
    # such, never as a property of the claim.
    if "out-dir" not in flags:
        offered = f"offers only {sorted(flags)}" if flags else "offers no long flags at all"
        return _finish(
            96,
            f"REFUSED 96: adjudicator '{adj_field}' has no run mode -- its --help {offered} "
            "and does not declare '--out-dir', so there is no way to hand it an output "
            "directory and no way to execute this row today. That is a defect in the "
            "instrument (the T1-6 symlinked-shard case), not an UNMEASURED property of "
            "the claim and never a red",
            adj_field=adj_field,
            adj_bytes=adj_bytes,
        )

    # -- 5. Compose the argv from what the program declares. Requested flags
    # it does not declare are dropped here, loudly, so the operator sees the
    # wiring they asked for and the wiring they got.
    argv = compose_argv(
        interpreter=interpreter,
        adjudicator_path=adj_path,
        out_dir=out_dir,
        declared_flags=flags,
        requested_flags=requested,
    )
    result["argv"] = argv
    for raw_name in requested:
        flag = _normalize_flag(raw_name)
        if flag == "out-dir":
            print(
                f"note: requested flag '--out-dir' is always taken from the runner's own "
                "--out-dir; ignoring the duplicate request",
                file=sys.stderr,
            )
        elif flag not in flags and flag != "help":
            print(
                f"note: requested flag '--{flag}' is NOT declared by {adj_field}'s --help "
                "and was not placed in the argv",
                file=sys.stderr,
            )

    missing_env = [var for var in REQUIRED_ENV_VARS if not str(env_map.get(var, "")).strip()]

    if dry_run:
        print(shlex.join(str(token) for token in argv))
        if missing_env:
            print(
                f"note: {', '.join(missing_env)} not set in the environment; a real run "
                "would read UNMEASURED 95 before executing"
            )
        result["exit_code"] = 0
        result["reason"] = "dry run: argv composed and printed; nothing executed, no receipt written"
        return result

    # -- 6. Inputs. No estate paths, no defaults, no guessing: absent model or
    # dataset is UNMEASURED 95 with the variable names in the reason.
    if missing_env:
        verb = "is" if len(missing_env) == 1 else "are"
        listed = " and ".join(missing_env)
        return _finish(
            95,
            f"UNMEASURED 95: environment variable{'s' if len(missing_env) > 1 else ''} "
            f"{listed} {verb} not set; the model root and the dataset arrive only from "
            f"{MODEL_ENV_VAR} and {DATASET_ENV_VAR}, which have no default and are never "
            "guessed (no estate paths in a public repository). An absent input is not a "
            "failure of the claim",
            adj_field=adj_field,
            adj_bytes=adj_bytes,
        )

    # -- 7. Execute. No timeout: GPU arms run for tens of minutes, and a
    # runner-chosen bound would manufacture refusals out of slow science.
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, env=_child_env(env_map))
    except OSError as exc:
        return _finish(
            96,
            f"REFUSED 96: could not launch adjudicator '{adj_field}': {exc}",
            adj_field=adj_field,
            adj_bytes=adj_bytes,
        )
    rc = proc.returncode
    tail = _output_tail(proc.stdout, proc.stderr)

    # -- 8. Out-of-contract exit: REFUSE 96 naming the code. It does not
    # become a red and does not become a green.
    if rc not in receipts.VERDICTS:
        suffix = f"; last output line: {tail!r}" if tail else ""
        return _finish(
            96,
            f"REFUSED 96: adjudicator '{adj_field}' exited {rc}, which is outside the "
            f"four-state contract {sorted(receipts.VERDICTS)}{suffix}; a program that "
            "broke its own contract has not told us what it measured, so this refusal "
            "names the code -- it is not a red and not a green",
            adj_field=adj_field,
            adj_bytes=adj_bytes,
        )

    # -- 9. Collect the measured arms.
    try:
        arms = collect_arms(out_dir, row_id=row_id)
    except ValueError as exc:
        return _finish(
            96,
            f"REFUSED 96: {exc}",
            adj_field=adj_field,
            adj_bytes=adj_bytes,
        )

    # -- 10. A verdict with no observations behind it is the all([]) trap.
    if rc in (0, 5) and not arms:
        return _finish(
            96,
            f"REFUSED 96: adjudicator '{adj_field}' exited {rc} ({receipts.VERDICTS[rc]}) "
            f"but wrote no '{_arm_prefix(row_id)}_<arm>.json' files under '{out_dir}'; a "
            "verdict with no observations behind it is the all([]) trap, and an empty "
            "observation is not a clean one",
            adj_field=adj_field,
            adj_bytes=adj_bytes,
        )

    if rc == 0:
        reason = (
            f"claim upheld; {len(arms)} arm(s) measured under '{out_dir}': "
            + ", ".join(sorted(arms))
        )
    elif rc == 5:
        arm_reasons = "; ".join(
            f"{name}: {arms[name]['reason']}"
            for name in sorted(arms)
            if isinstance(arms[name].get("reason"), str) and arms[name]["reason"].strip()
        )
        reason = (
            arm_reasons
            or tail
            or "adjudicator exited 5 (control rule refuted the claim) without a reason "
            "string of its own; the refutation stands, its detail does not"
        )
    else:
        reason = (
            f"adjudicator '{adj_field}' exited {rc} ({receipts.VERDICTS[rc]}): "
            f"{tail or 'it printed nothing'}"
        )
    return _finish(rc, reason, arms=arms, adj_field=adj_field, adj_bytes=adj_bytes)


_FAKE_ADJUDICATOR_TEMPLATE = '''
import json
import os
import sys

HELP_TEXT = %(help_text)r
HELP_EXIT = %(help_exit)d
RUN_EXIT = %(run_exit)d
ARMS = %(arms)r
PREFIX = %(prefix)r


def main(argv):
    if "--help" in argv or "-h" in argv:
        sys.stdout.write(HELP_TEXT)
        return HELP_EXIT
    out_dir = None
    if "--out-dir" in argv:
        index = argv.index("--out-dir")
        if index + 1 < len(argv):
            out_dir = argv[index + 1]
    if ARMS and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for arm_name, payload in ARMS.items():
            target = os.path.join(out_dir, PREFIX + "_" + arm_name + ".json")
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
    return RUN_EXIT


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def self_test():
    """WHY: every control below is a regression test for a way this campaign
    has already lied -- a refusal recorded as a measurement (C1, the T1-6
    case), a flag the program never accepted (C2), an estate path masquerading
    as an input (C3), a verdct with no observations (C4), a red without a
    reason (C5), a crash wearing a verdict (C6), arms that drift from the disk
    (C7), and a clock borrowed from the writer (C8). Fixtures are tiny fake
    adjudicators in a TemporaryDirectory: no GPU, no torch, no estate path,
    and the real matrix.json is never read -- the rows are constructed inline.
    Exit code follows the same four-state contract: 0 if every control fires,
    5 if any fails, because a runner that fails its own controls has refuted
    itself."""
    checks = []

    def record(label, ok, observed):
        text = str(observed)
        if len(text) > 400:
            text = text[:400] + "...[truncated]"
        checks.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {label} -- observed: {text}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        receipts_root = root / "receipts"
        env_ok = {MODEL_ENV_VAR: str(root / "model"), DATASET_ENV_VAR: str(root / "dataset")}

        def fake(filename, *, help_text, run_exit, arms=None, prefix, help_exit=0):
            (root / filename).write_text(
                _FAKE_ADJUDICATOR_TEMPLATE
                % {
                    "help_text": help_text,
                    "help_exit": help_exit,
                    "run_exit": run_exit,
                    "arms": arms or {},
                    "prefix": prefix,
                },
                encoding="utf-8",
            )
            return filename

        def go(row_id, pass_id, adj_field, out_name, written_at, env):
            return run_row(
                row={
                    "id": row_id,
                    "claim": f"self-test claim for {row_id}",
                    "adjudicator": adj_field,
                },
                out_dir=root / out_name,
                pass_id=pass_id,
                receipts_root=receipts_root,
                written_at_utc=written_at,
                repo_root=root,
                env=env,
            )

        # C1 -- the t1_6 case: --help that offers only --self-test.
        c1_adj = fake(
            "c1_adjudicator.py",
            help_text="usage: c1 [-h] [--self-test]\n\nexercise the adjudicator on synthetic records\n",
            run_exit=0,
            prefix="c1_6",
        )
        res = go("C1-6", "c1", c1_adj, "c1_out", "2029-01-01T00:00:00+00:00", env_ok)
        reason = res["reason"] or ""
        record(
            "C1 MUST-FIRE: --help lacking --out-dir REFUSES 96 naming adjudicator and flag",
            res["exit_code"] == 96 and "c1_adjudicator.py" in reason and "--out-dir" in reason,
            {"exit_code": res["exit_code"], "reason": reason},
        )

        # C2 -- an undeclared flag never reaches the argv, even when asked for.
        declared = help_flags("usage: c2 [-h] [--out-dir OUT_DIR] [--seed SEED]\n")
        argv2 = compose_argv(
            interpreter=sys.executable,
            adjudicator_path="c2_adjudicator.py",
            out_dir="c2_out",
            declared_flags=declared,
            requested_flags={"seed": "12345", "never-declared-flag": "zz-forbidden"},
        )
        leaked = [t for t in argv2 if "never-declared" in t or "zz-forbidden" in t]
        record(
            "C2 MUST-FIRE: a flag absent from --help is never placed in the argv",
            "--out-dir" in argv2 and "--seed" in argv2 and "12345" in argv2 and not leaked,
            argv2,
        )

        # C3 -- absent FS_T1_MODEL reads 95 naming the variable.
        c3_adj = fake(
            "c3_adjudicator.py",
            help_text="usage: c3 [-h] [--out-dir OUT_DIR]\n",
            run_exit=0,
            prefix="c3_9",
        )
        res = go(
            "C3-9",
            "c3",
            c3_adj,
            "c3_out",
            "2029-01-01T00:00:01+00:00",
            {DATASET_ENV_VAR: str(root / "dataset")},
        )
        reason = res["reason"] or ""
        record(
            "C3: an absent FS_T1_MODEL reads 95 and the reason names the variable",
            res["exit_code"] == 95 and "FS_T1_MODEL" in reason,
            {"exit_code": res["exit_code"], "reason": reason},
        )

        # C4 -- exit 0 with ZERO arm files refuses 96.
        c4_adj = fake(
            "c4_adjudicator.py",
            help_text="usage: c4 [-h] [--out-dir OUT_DIR]\n",
            run_exit=0,
            arms=None,
            prefix="c4_2",
        )
        res = go("C4-2", "c4", c4_adj, "c4_out", "2029-01-01T00:00:02+00:00", env_ok)
        reason = res["reason"] or ""
        record(
            "C4: exit 0 with zero arm files REFUSES 96 (the all([]) trap)",
            res["exit_code"] == 96 and "arm" in reason.lower() and "all([])" in reason,
            {"exit_code": res["exit_code"], "reason": reason},
        )

        # C5 -- exit 5 yields a red receipt with a non-empty reason.
        c5_adj = fake(
            "c5_adjudicator.py",
            help_text="usage: c5 [-h] [--out-dir OUT_DIR]\n",
            run_exit=5,
            prefix="c5_3",
            arms={
                "run": {
                    "status": "refuted",
                    "reason": "loss curves identical across arms; the knob did nothing",
                    "launcher_exit_code": 0,
                    "telemetry": {"steps": 40},
                    "loss_curve": [{"step": 0, "loss": 2.5}],
                }
            },
        )
        res = go("C5-3", "c5", c5_adj, "c5_out", "2029-01-01T00:00:03+00:00", env_ok)
        rec = res["receipt"] or {}
        record(
            "C5: an adjudicator exiting 5 produces a red receipt with a non-empty reason",
            res["exit_code"] == 5
            and rec.get("verdict") == "red"
            and bool(str(rec.get("reason") or "").strip()),
            {"exit_code": res["exit_code"], "verdict": rec.get("verdict"), "reason": rec.get("reason")},
        )

        # C6 -- exit 7 (out of contract) refuses 96 naming the code.
        c6_adj = fake(
            "c6_adjudicator.py",
            help_text="usage: c6 [-h] [--out-dir OUT_DIR]\n",
            run_exit=7,
            prefix="c6_4",
        )
        res = go("C6-4", "c6", c6_adj, "c6_out", "2029-01-01T00:00:04+00:00", env_ok)
        rec = res["receipt"] or {}
        reason = res["reason"] or ""
        record(
            "C6 MUST-FIRE: exit 7 REFUSES 96 naming the code -- not a red, not a green",
            res["exit_code"] == 96
            and "7" in reason
            and rec.get("verdict") == "refused",
            {"exit_code": res["exit_code"], "verdict": rec.get("verdict"), "reason": reason},
        )

        # C7 + C8 -- a clean green run with two arms and a pinned timestamp.
        arm_payload = {
            "status": "upheld",
            "reason": "arm measured",
            "launcher_exit_code": 0,
            "telemetry": {"steps": 40},
            "loss_curve": [{"step": 0, "loss": 1.0}],
        }
        c7_adj = fake(
            "c7_adjudicator.py",
            help_text="usage: c7 [-h] [--out-dir OUT_DIR]\n",
            run_exit=0,
            prefix="c7_1",
            arms={"alpha": dict(arm_payload), "beta": dict(arm_payload, telemetry={"steps": 41})},
        )
        stamp = "2031-02-03T04:05:06.789+00:00"
        res = go("C7-1", "c7", c7_adj, "c7_out", stamp, env_ok)
        rec = res["receipt"] or {}
        disk_arms = {p.stem[len("c7_1") + 1 :] for p in (root / "c7_out").glob("c7_1_*.json")}
        record(
            "C7: the receipt's arms mapping equals the arm files on disk, by name",
            res["exit_code"] == 0 and set(rec.get("arms", {})) == disk_arms == {"alpha", "beta"},
            {"receipt_arms": sorted(rec.get("arms", {})), "disk_arms": sorted(disk_arms)},
        )
        record(
            "C8: written_at_utc in the receipt is the value the runner passed, verbatim",
            rec.get("written_at_utc") == stamp,
            {"receipt": rec.get("written_at_utc"), "passed_to_run_row": stamp},
        )

    passed = sum(checks)
    print(f"self-test: {passed}/{len(checks)} controls passed")
    return 0 if passed == len(checks) else 5


def main(argv=None):
    """WHY the aggregation order for --all (5, then 96, then 95, else 0): a
    measured refutation is the loudest thing a campaign can say, so red wins;
    a refusal means the INSTRUMENT is defective and outranks merely missing
    inputs; UNMEASURED must never masquerade as either; and CLEAR is earned
    only when every selected row is clear. Usage errors refuse (96) rather
    than exiting 2, because 2 is not a state the campaign can record."""
    parser = ContractArgumentParser(
        prog="run_row.py",
        description=(
            "Run verification-matrix rows through their adjudicators and write receipts. "
            "The argv is built ONLY from flags the adjudicator's --help declares; results "
            "are written as receipts, never into matrix.json. Exit codes: 0 CLEAR, 5 RED, "
            "95 UNMEASURED, 96 REFUSE."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--row", help="a single matrix row id, e.g. T1-9")
    mode.add_argument("--all", action="store_true", help="every row with an adjudicator")
    mode.add_argument("--self-test", action="store_true", help="run controls C1-C8; no GPU, no matrix.json")
    parser.add_argument("--out-dir", help="directory the adjudicator writes <row>_<arm>.json into")
    parser.add_argument("--pass-id", help="the id of this pass; receipts are addressed by it")
    parser.add_argument(
        "--flag",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "a flag to forward to the adjudicator; it is sent ONLY if the adjudicator's "
            "--help declares it, and reported otherwise"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compose and print the argv without executing -- inspect the wiring without a GPU",
    )
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"REFUSED 96: {exc}", file=sys.stderr)
        return 96
    except SystemExit:
        return 0  # --help

    if args.self_test:
        return self_test()
    if not (args.row or args.all):
        print("REFUSED 96: one of --row ROW, --all, or --self-test is required", file=sys.stderr)
        return 96
    if not args.out_dir or not args.pass_id:
        print(
            "REFUSED 96: --out-dir and --pass-id are required to run a row; the "
            "adjudicator needs somewhere to write arm files and the receipt needs a pass",
            file=sys.stderr,
        )
        return 96

    try:
        rows = load_matrix()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSED 96: cannot read the matrix declaration: {exc}", file=sys.stderr)
        return 96
    rows = [r for r in rows if isinstance(r, Mapping)]

    requested_flags = {}
    for item in args.flag:
        name, eq, value = item.partition("=")
        requested_flags[name.strip()] = value if eq else None

    if args.all:
        selected = [r for r in rows if isinstance(r.get("adjudicator"), str) and r["adjudicator"].strip()]
        if not selected:
            print("REFUSED 96: --all selected no rows; no matrix row declares an adjudicator", file=sys.stderr)
            return 96
    else:
        selected = [r for r in rows if r.get("id") == args.row]
        if not selected:
            print(f"REFUSED 96: no matrix row with id {args.row!r}", file=sys.stderr)
            return 96

    codes = []
    for row in selected:
        stamp = datetime.now(timezone.utc).isoformat()  # the runner takes the clock
        try:
            res = run_row(
                row=row,
                out_dir=args.out_dir,
                pass_id=args.pass_id,
                receipts_root=RECEIPTS_ROOT,
                written_at_utc=stamp,
                requested_flags=requested_flags,
                dry_run=args.dry_run,
            )
            code, verdict, reason = res["exit_code"], res["verdict"], res["reason"]
        except Exception as exc:  # e.g. receipt for this pass already exists
            code, verdict, reason = 96, "refused", f"row could not be recorded: {exc}"
            print(f"REFUSED 96: row {row.get('id')!r}: {reason}", file=sys.stderr)
        codes.append(code)
        print(f"{row.get('id')}: {code} {verdict or '-'}: {reason}")

    if 5 in codes:
        return 5
    if 96 in codes:
        return 96
    if 95 in codes:
        return 95
    return 0


if __name__ == "__main__":
    sys.exit(main())
