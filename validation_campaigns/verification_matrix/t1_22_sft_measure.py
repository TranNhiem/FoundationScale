"""t1_22_sft_measure -- measurement probe for verification row T1-22, SFT plane.

WHAT IS MEASURED. Row T1-22 asks whether the SFT training plane honours the
boundary that fix #490 drew: the DECLARATION, never the data, decides whether a
modality column is refused. This script runs the four SFT arms described by an
external plan file (``--plan``) against ``foundationscale.train.cli`` in
subprocesses, extracts a fixed set of fields from each arm's console output and
run manifests, and writes a JSON payload (``--out``) that the already-written
T1-22 adjudicator consumes via its ``--sft-payload`` flag. The object written
to ``--out`` is exactly the value the adjudicator places under ``payload["sft"]``:
a mapping of plan arm name -> the eight fields listed in R2, in that order. No
verdict is computed here; this script measures and never judges.

WHY EACH ARM EXISTS (the "why" text is owned by the plan; reproduced here so the
file stands alone):

  * video_only  -- no text column at all: any refusal here is about 'text' and
    never about video, so a missing-'text' refusal can never be mistaken for a
    modality refusal; under #490 the [fs:train:data] notice names the dropped
    video column first, then the textless corpus earns its own refusal.
  * text_video  -- nothing declared: the corpus merely CONTAINS a video key. It
    must TRAIN -- refusing on the column name is name-sniffing and breaks
    legitimate text corpora -- and the [fs:train:data] notice must name 'video'
    among the dropped columns instead of dropping it silently.
  * text_video_declared -- the deciding arm: identical corpus to text_video
    except the modality is DECLARED through its env var. It must exit 96 via
    [fs:train:refuse] naming the modality, the env var and the declared column
    -- the refusal is keyed on the declaration, which is the whole design.
  * image_control -- the refusal-capability control: a declared-but-
    unsatisfiable image column earns a named 96, proving the SFT plane CAN
    refuse a declaration at all; without it, silence about video is an absence
    of evidence.

The script is driven BY the plan: arm names, corpus columns, declared env vars
and hyperparameters all come from it. Adding a fifth arm to the plan requires
zero edits here.

SELF-HIT AVOIDANCE. The needle is the word "video". Every path this script
creates is needle-free (arm directories are arm_00, arm_01, ... in plan order;
the corpus file is corpus.jsonl), and the full argv of every arm is asserted
needle-free BEFORE the first subprocess runs: if the caller's --model or
--work-dir carried the needle, a console echo of the dataset path would read as
the framework naming the column, so the whole run is refused (96) instead.

EXIT CODES. 0 = every arm measured (a trainer refusal of 96 is a successful
measurement of that arm); 95 = at least one arm timed out or errored; 96 = bad
usage, unreadable/invalid plan, dirty argv, or a failed --self-test. This file
never exits 1 or 2.

--self-test runs under ``python3 -S`` with no torch, no model and no network:
it exercises the pure field extractors against synthetic console text and exits
0 only if every named control passes.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Four-state contract, shared with the adjudicator.
GREEN, RED, UNMEASURED, REFUSE = 0, 5, 95, 96

PROG = "t1_22_sft_measure"

# Lowercase scan needle. Only corpus record KEYS and the arm's declared env
# values may carry it -- never a path this script creates (see docstring).
NEEDLE = "video"

CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_GLOB = "*manifest*.json"

# Console markers printed by the post-#490 trainer via _mark(step, msg), which
# emits f"[{step}]".ljust(24) + f" {msg}".rstrip(). Marker and message are
# therefore separated by VARIABLE padding and are matched separately below.
REFUSE_MARKER = "fs:train:refuse"
DATA_MARKER = "fs:train:data"

_TOKENIZED_RE = re.compile(r"(\d+)\s+examples tokenized")

# ---------------------------------------------------------------------------
# Field extractors (pure: console text in, booleans/counts out)
# ---------------------------------------------------------------------------

# R4 DISCRIMINATION. The pre-#490 missing-text refusal reads exactly
# "the thin path requires a 'text' column". The post-#490 modality refusal also
# contains the word "text" three times ("it encodes text", "pixels when an image
# column is declared", "train text-only on the same corpus"), so matching the
# bare word "text" -- or even "'text'" -- would report the FIX as the original
# defect: a correct exit 96 attached to the missing-text message, which is the
# T1-23 bug this row exists to separate from its fix. The only phrase unique to
# the pre-existing refusal is "thin path requires a 'text' column"; this pattern
# anchors on it (tolerating quote style and whitespace, never on the bare word).
_MISSING_TEXT_RE = re.compile(
    r"thin\s+path\s+requires\s+a\s+['\u2018\u2019\"]text['\u2018\u2019\"]\s+column",
    re.IGNORECASE,
)

# Marker lines: "[" + marker + "]" followed by one or more spaces/tabs of _mark
# padding, then the message. The message is captured separately so a reworded
# or differently padded line still parses; marker and message are NEVER matched
# as one adjacent literal.
def _marker_messages(out: str, marker: str) -> list[str]:
    pat = re.compile(r"^\[" + re.escape(marker) + r"\][ \t]+(?P<msg>.*)$", re.MULTILINE)
    return [m.group("msg") for m in pat.finditer(out)]


_DROPPED_RE = re.compile(r"columns\s+(\[[^\]]*\])\s+are\s+dropped")


def _drop_notice_columns(out: str) -> list[list[str]]:
    """The dropped-column list from every [fs:train:data] notice in ``out``.

    Padding-insensitive by construction: the notice line is found by its marker
    alone and the column list is parsed out of the message with literal_eval.
    """
    dropped: list[list[str]] = []
    for msg in _marker_messages(out, DATA_MARKER):
        m = _DROPPED_RE.search(msg)
        if not m:
            continue
        try:
            cols = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(cols, list):
            dropped.append([str(c) for c in cols])
    return dropped


def _output_names_needle(out: str) -> bool:
    """Did the word 'video' reach the CONSOLE? Combined stdout+stderr only.

    Manifest mentions are reported separately (manifests_naming_needle): the
    adjudicator's READ of the undeclared arm is specifically about the console
    ("the word 'video' never reached the console"), so this flag must not be
    contaminated by manifest content.
    """
    return NEEDLE in out.lower()


def _names_missing_text(out: str) -> bool:
    return bool(_MISSING_TEXT_RE.search(out))


def _names_own_declaration(declared_env: dict[str, str], out: str) -> bool:
    """True only when the arm DECLARED something (env non-empty) AND the output
    names a declared column value or its env var. An arm that declared nothing
    must be False, never True, so the guard on `declared_env` comes first and
    is not folded into the loop.
    """
    if not declared_env:
        return False
    low = out.lower()
    for key, value in declared_env.items():
        if str(value).lower() in low or key.lower() in low:
            return True
    return False


def _scan_manifests(arm_dir: Path) -> dict[str, Any]:
    """Every manifest the arm wrote, names of those containing the needle, and
    their total size in characters (reported so 'not a word in N characters of
    manifest' is a checkable claim, not an impression)."""
    hits: list[str] = []
    total_chars = 0
    if arm_dir.exists():  # a timed-out arm may never have been created
        for path in sorted(arm_dir.rglob(MANIFEST_GLOB)):
            body = path.read_text(errors="replace")
            total_chars += len(body)
            if NEEDLE in body.lower():
                hits.append(path.name)
    return {"manifests_naming_needle": hits, "manifest_chars": total_chars}


def _extract_fields(
    declared_env: dict[str, str],
    rc: int | None,
    out: str,
    arm_dir: Path,
    timed_out: bool,
) -> dict[str, Any]:
    """Assemble the eight-key arm record. EXACTLY these keys, in this order
    (R2); the adjudicator is authoritative about their meanings."""
    man = _scan_manifests(arm_dir)
    tokenized = _TOKENIZED_RE.search(out)
    if timed_out:
        status = "timeout"
    elif rc is not None:
        status = "measured"
    else:
        status = "error"
    return {
        "status": status,
        # None for timeout/error -- never an invented measurement (R5).
        "launcher_exit_code": rc,
        "output_names_needle": _output_names_needle(out),
        "refusal_names_missing_text": _names_missing_text(out),
        "manifests_naming_needle": man["manifests_naming_needle"],
        "manifest_chars": man["manifest_chars"],
        "examples_tokenized": int(tokenized.group(1)) if tokenized else None,
        "names_own_declaration": _names_own_declaration(declared_env, out),
    }


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

def _as_text(stream: Any) -> str:
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")


def _run_arm(
    arm_dir: Path, cmd: list[str], env: dict[str, str],
    declared_env: dict[str, str], timeout_s: int, cwd: Path,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=timeout_s, cwd=os.fspath(cwd), check=False,
        )
        return _extract_fields(declared_env, proc.returncode,
                               _as_text(proc.stdout) + _as_text(proc.stderr),
                               arm_dir, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        # Partial output is real evidence (a notice may already be on the
        # console); rc is None and status is "timeout", never "measured".
        out = _as_text(exc.stdout) + _as_text(exc.stderr)
        return _extract_fields(declared_env, None, out, arm_dir, timed_out=True)
    except OSError as exc:
        # The launcher itself could not be spawned: the arm never ran.
        return _extract_fields(declared_env, None, f"{type(exc).__name__}: {exc}",
                               arm_dir, timed_out=False)


# ---------------------------------------------------------------------------
# Plan handling
# ---------------------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    """R6: argparse would exit 2 on a usage error; this probe never exits 2."""

    def error(self, message: str) -> None:
        print(f"{PROG}: usage error: {message}", file=sys.stderr)
        raise SystemExit(REFUSE)


def _load_plan(path: str) -> dict[str, Any] | None:
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        print(f"{PROG}: cannot read plan {path!r}: {exc}", file=sys.stderr)
        return None
    arms = plan.get("arms") if isinstance(plan, dict) else None
    if not isinstance(arms, dict) or not arms:
        print(f"{PROG}: plan has no non-empty 'arms' mapping", file=sys.stderr)
        return None
    for name, spec in arms.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("columns"), dict):
            print(f"{PROG}: plan arm {name!r} lacks a 'columns' mapping",
                  file=sys.stderr)
            return None
        env = spec.get("env", {})
        if not isinstance(env, dict):
            print(f"{PROG}: plan arm {name!r} has a non-mapping 'env'",
                  file=sys.stderr)
            return None
    return plan


def _self_test() -> bool:
    """Exercise the pure extractors against synthetic console text. Spawns no
    subprocesses; imports nothing beyond stdlib; runs under ``python3 -S``."""

    def fake_mark(step: str, msg: str) -> str:
        # Replicates the trainer's _mark exactly, padding included.
        return f"[{step}]".ljust(24) + f" {msg}".rstrip()

    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'} {name}")

    modality, declared, envvar = "video", "video", "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"

    # -- control 1 (the R4 trap): the #490 modality refusal contains the word
    # "text" three times but must NOT set refusal_names_missing_text. It SHOULD
    # still name the needle and the arm's own declaration.
    refusal_msg = (
        f"{modality} column '{declared}' is declared via {envvar}, but this training "
        f"plane has no {modality} arm: it encodes text, and pixels when an image "
        "column is declared -- nothing else. Training anyway would drop the "
        f"{modality} silently and report success under a {modality} label, which is "
        f"the defect this refuses. Unset {envvar} to train text-only on the same corpus"
    )
    refusal_out = fake_mark(REFUSE_MARKER, refusal_msg) + "\n"
    check(
        "t490_refusal_not_missing_text",
        not _names_missing_text(refusal_out)
        and _output_names_needle(refusal_out)
        and _names_own_declaration({envvar: declared}, refusal_out),
    )

    # -- control 2: a genuine missing-'text' refusal DOES set the flag.
    missing_text_out = fake_mark(REFUSE_MARKER, "the thin path requires a 'text' column") + "\n"
    check(
        "missing_text_refusal_sets_flag",
        _names_missing_text(missing_text_out),
    )

    # -- control 3: the padded [fs:train:data] notice is detected despite the
    # variable marker/message padding, and its dropped list parses.
    notice_msg = (
        "text-only arm: tokenizing 'text'; columns ['video', 'label'] are dropped "
        "and contribute nothing to the loss. If one of them was meant to be "
        "trained on, declare it -- an undeclared column is dropped by design, "
        "not by accident"
    )
    notice_out = fake_mark(DATA_MARKER, notice_msg) + "\n"
    check(
        "padded_data_notice_detected",
        _marker_messages(notice_out, DATA_MARKER) == [notice_msg]
        and _drop_notice_columns(notice_out) == [["video", "label"]]
        and _output_names_needle(notice_out),
    )

    # -- control 4: an arm declaring nothing never sets names_own_declaration,
    # even when the output happens to contain plausible words.
    check(
        "undeclared_arm_never_names_own_declaration",
        not _names_own_declaration({}, refusal_out + notice_out),
    )

    # -- control 5: a timed-out arm yields status "timeout" and a None exit
    # code, straight from the shared extractor.
    ghost = Path("self_test_no_such_arm_dir")
    fields = _extract_fields({}, None, "", ghost, timed_out=True)
    check(
        "timeout_yields_timeout_status_none_exit",
        fields["status"] == "timeout" and fields["launcher_exit_code"] is None,
    )

    passed = sum(1 for _, ok in results if ok)
    print(f"self-test: {passed}/{len(results)} passed")
    print("ALL PASS" if passed == len(results) else "SELF-TEST FAILED")
    return passed == len(results)


def main() -> int:
    ap = _Parser(prog=PROG)
    ap.add_argument("--plan")
    ap.add_argument("--repo")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--work-dir")
    ap.add_argument("--profile-name", default="local-single-node")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return GREEN if _self_test() else REFUSE

    missing = [f"--{k}" for k in ("plan", "repo", "model", "out", "work-dir")
               if getattr(args, k.replace("-", "_")) in (None, "")]
    if missing:
        print(f"{PROG}: missing required argument(s): {', '.join(missing)}",
              file=sys.stderr)
        return REFUSE

    plan = _load_plan(args.plan)
    if plan is None:
        return REFUSE
    arm_specs: dict[str, dict[str, Any]] = plan["arms"]  # plan order is semantic
    hp = plan.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        print(f"{PROG}: plan 'hyperparameters' is not a mapping", file=sys.stderr)
        return REFUSE

    repo = Path(args.repo).resolve()
    src = repo / "src"
    if not src.is_dir():
        print(f"{PROG}: {src} is not a directory; PYTHONPATH would be empty",
              file=sys.stderr)
        return REFUSE
    try:
        work = Path(args.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"{PROG}: cannot create --work-dir: {exc}", file=sys.stderr)
        return REFUSE

    max_steps = int(hp.get("max_steps", 2))
    batch = int(hp.get("per_device_batch_size", 1))
    nodes = int(hp.get("nodes", 1))
    gpus = int(hp.get("gpus_per_node", 1))
    seed = int(hp.get("seed", 42))
    save_interval = int(hp.get("save_interval", 1000))
    timeout_s = int(hp.get("timeout_s", 3600))

    # Every env var ANY arm declares. For arms that do not declare it, the var
    # is scrubbed from the inherited environment, so a declaration cannot leak
    # between arms or in from the caller's shell.
    plan_env_vars = {k for spec in arm_specs.values() for k in (spec.get("env") or {})}

    prepared: dict[str, tuple[Path, list[str], dict[str, str], dict[str, str]]] = {}
    for i, (name, spec) in enumerate(arm_specs.items()):
        columns: dict[str, Any] = spec["columns"]
        declared_env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        # Needle-free arm dir (arm_00, arm_01, ...): the arm NAME may carry the
        # needle (text_video declares it as data), the path must not.
        arm_dir = work / f"arm_{i:02d}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        corpus = arm_dir / CORPUS_FILENAME
        # Two identical rows, mirroring the proven 3-arm probe: with batch 1
        # and 2 max steps a one-row corpus can exhaust its own loader mid-run.
        # Only record KEYS carry the needle; values come from the plan.
        rows = [dict(columns), dict(columns)]
        corpus.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        cmd = [sys.executable, "-m", "foundationscale.train.cli"]
        flags = {
            "--model": str(args.model),
            "--dataset": os.fspath(corpus),
            "--output-dir": os.fspath(arm_dir),
            "--max-steps": str(max_steps),
            "--per-device-batch-size": str(batch),
            "--nodes": str(nodes),
            "--gpus-per-node": str(gpus),
            "--dp": str(nodes * gpus),
            "--seed": str(seed),
            "--save-interval": str(save_interval),
            "--profile-name": str(args.profile_name),
        }
        for flag, value in flags.items():
            cmd.extend([flag, value])
        env = dict(os.environ)
        for var in plan_env_vars:
            if var not in declared_env:
                env.pop(var, None)
        env.update(declared_env)
        env["PYTHONPATH"] = os.fspath(src)
        prepared[name] = (arm_dir, cmd, env, declared_env)

    # Assert argv cleanliness for EVERY arm before the first subprocess runs:
    # a needle anywhere in any argv (a --model or --work-dir path, say) would
    # let the scan match the probe's own fingerprint.
    for name, (_, cmd, _, _) in prepared.items():
        dirty = [el for el in cmd if NEEDLE in el.lower()]
        if dirty:
            print(f"{PROG}: argv for arm {name!r} carries the {NEEDLE!r} needle "
                  f"({dirty!r}); the scan would read the probe's own fingerprint. "
                  "Refusing to run rather than forging a measurement.",
                  file=sys.stderr)
            return REFUSE

    payload: dict[str, Any] = {}
    for name in arm_specs:  # plan order
        arm_dir, cmd, env, declared_env = prepared[name]
        print(f"--- {name} ---", flush=True)
        fields = _run_arm(arm_dir, cmd, env, declared_env, timeout_s, repo)
        payload[name] = fields
        print(f"    status={fields['status']} rc={fields['launcher_exit_code']} "
              f"names_needle={fields['output_names_needle']} "
              f"missing_text={fields['refusal_names_missing_text']} "
              f"names_decl={fields['names_own_declaration']} "
              f"tokenized={fields['examples_tokenized']}", flush=True)

    try:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        print(f"{PROG}: cannot write payload to {args.out!r}: {exc}",
              file=sys.stderr)
        return UNMEASURED

    if all(f["status"] == "measured" for f in payload.values()):
        return GREEN
    unmeasured = sorted(n for n, f in payload.items() if f["status"] != "measured")
    print(f"{PROG}: unmeasured arm(s): {', '.join(unmeasured)}; writing partial "
          "payload and exiting 95 so the gap reads as UNMEASURED, never as "
          "evidence either way", file=sys.stderr)
    return UNMEASURED


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- a wedge must never read as a pass: 95, not 1
        print(f"{PROG}: unexpected failure: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        # `from exc` keeps the original traceback attached: this path exists so a
        # wedge reads as 95 rather than 1, and a 95 whose cause was swallowed is
        # an abstention nobody can act on.
        raise SystemExit(UNMEASURED) from exc
