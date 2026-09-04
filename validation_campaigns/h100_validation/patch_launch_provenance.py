#!/usr/bin/env python3
"""#180: write launch provenance -- the run records everything except what it ran.

MEASURED DEFECT. On a completed 8-GPU job's own 936-line log the census for the
trainer invocation was: 0 occurrences of the launch-command variable, 0
occurrences of its argv-only flags (the resume tolerance, the model path, the
dataset path, the sequence length). The run records BEGIN (8 fields) and
LAUNCH_TOPOLOGY (4 fields) -- everything ABOUT the run except WHAT ran. The
code cannot answer either: the resume knob is sourced with env_name=None and
required=True in the trainer -- flag-only, so the value can only arrive on a
command line, from a command line nothing records. Reconstructing a
four-hour-old run's invocation from its artifacts was attempted and failed;
only an assertion artifact ("tolerance" echoed into a proof JSON) accidentally
witnessed one knob. A run that cannot reproduce itself fails the
reproducibility requirement while every gate stays green.

THE FIX. Pure bash (the launcher is bash; the login-node interpreter may be
3.6.8, so no python dependency here), inserted after the composer has rewritten
LAUNCH_CMD and immediately before the run_in_container execution site, so the
record is what actually runs. The operator-supplied command is captured into
LAUNCH_CMD_RAW at the ingestion line, before the composer overwrites it. The
record path is DERIVED from RUN_LOG by suffix substitution
("${RUN_LOG%.log}.provenance.json") and never recomputed from LOG_DIR /
SLURM_JOB_ID a second way: finding #150 in this same campaign was a writer and
a reader computing one checkpoint name two ways and disagreeing, which made a
cross-validation leg inert on 2 of 2 real checkpoints.

REDACTIONS. An environment dump on a shared filesystem is exactly how a live
credential gets copied, archived or published, so redaction is part of the
design: name-based (never value-shape/entropy, which mis-fires on paths and
misses short values), every substitution counted, and the count emitted as
"redactions" so a consumer can tell an exact record from one that needs values
re-supplied. A write failure must never fail the launch: the whole block is
guarded and a PROVENANCE line states the path and ok/FAILED either way.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"
MARK = "fs180"
TAG = MARK + ":"
N_CONTROLS = 10

# Needles are ASSEMBLED from fragments, never one literal: this stage counts
# them in the target, and a source that contains its own needle is inside its
# own denominator.
A_ASSIGN = "LAUNCH" + '_CMD="${FS_ENGINE_' + "LAUNCH" + '_CMD:-}"'
A_RE = "LAUNCH" + '_CMD="${TOPO_OUT#*$' + "'" + "\\t" + "'" + '}"'
A_EXEC = "bash -lc \"$" + "LAUNCH" + "_CMD\""
CENSUS_VNAMES = ("LAUNCH" + "_CMD", "FS_ENGINE_" + "LAUNCH" + "_CMD")
DERIVED = '"${RUN_LOG%.log}.provenance.json"'

BLOCK_MARK = "# --- " + TAG + " launch provenance writer (finding #180)"
END_MARK = "# --- end " + MARK + " "
RAW_MARK = TAG + " captured the operator-supplied command before the composer "

# The "# " is load-bearing and was MISSING until it was measured on hardware: without it
# bash parses `fs180:` as a command word and the launcher dies `command not found`, rc 127,
# after the allocation is granted and the collective probe has already passed (job 37366).
# `bash -n` cannot catch this -- a bare word followed by prose is SYNTACTICALLY VALID, so the
# build's "parse clean" gate was true and useless here. G11 below is the gate that can see it.
RAW_LINE = ('LAUNCH_CMD_RAW="$LAUNCH_CMD"  # ' + RAW_MARK + "rewrites LAUNCH_CMD at "
            "ingestion; without this capture the record could only ever hold the "
            "composed form and 'what the operator asked for' would be lost.\n")

BLOCK = (
    BLOCK_MARK + " ----------\n"
    "# WHY: a completed 8-GPU job's own 936-line log held zero occurrences of the\n"
    "# launch command, zero of --resume-tolerance, and zero combined of\n"
    "# --model-path/--dataset-path/--sequence-length: the run logged everything\n"
    "# about itself except what ran. The trainer's resume knob is flag-only\n"
    "# (sourced with no environment name) and required, so a value that can only\n"
    "# arrive on a command line was recorded by nothing, and reconstructing a\n"
    "# four-hour-old run's invocation from its artifacts was attempted and\n"
    "# failed. This block writes what ACTUALLY runs, placed after the composer\n"
    "# rewrote LAUNCH_CMD -- the executed command, not the requested one.\n"
    "# The path is DERIVED from RUN_LOG by suffix substitution, never recomputed\n"
    "# from LOG_DIR and SLURM_JOB_ID (finding #150: a writer and a reader\n"
    "# computing one name two ways disagree). Redaction is by NAME only --\n"
    "# value-shape matching misfires on paths and misses short values -- and\n"
    "# every substitution is counted, because a record that has been altered\n"
    "# must say so: redactions=0 may be treated as exact, redactions>0 means\n"
    "# values must be re-supplied by hand. The write is fully guarded: a\n"
    "# failure degrades to an announced write=FAILED state, never a failed\n"
    "# launch. The name pattern is assembled from single-character classes so\n"
    "# matching is case-insensitive without a literal word list in this file.\n"
    "FS180_PROV_PATH=\"${RUN_LOG%.log}.provenance.json\"\n"
    "FS180_REDACTS=0\n"
    "_FS180_NPAT=\"[Kk][Ee][Yy]|[Tt][Oo][Kk][Ee][Nn]|[Ss][Ee][Cc][Rr][Ee][Tt]\"\n"
    "_FS180_NPAT=\"${_FS180_NPAT}|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]|[Pp][Aa][Ss][Ss][Ww][Dd]|[Cc][Rr][Ee][Dd][Ee][Nn][Tt][Ii][Aa][Ll]\"\n"
    "\n"
    "fs180_json_escape() {\n"
    "  local s=\"$1\"\n"
    "  s=\"${s//\\\\/\\\\\\\\}\"\n"
    "  s=\"${s//\\\"/\\\\\\\"}\"\n"
    "  s=\"${s//$'\\n'/\\\\n}\"\n"
    "  s=\"${s//$'\\r'/\\\\r}\"\n"
    "  s=\"${s//$'\\t'/\\\\t}\"\n"
    "  s=\"${s//[[:cntrl:]]/}\"\n"
    "  printf '%s' \"$s\"\n"
    "}\n"
    "\n"
    "fs180_redact_cmd() {\n"
    "  # Input: $1. Output: FS180_REDACTED. Side effect: FS180_REDACTS is\n"
    "  # incremented once per substitution, across the --flag VALUE, --flag=VALUE\n"
    "  # and NAME=VALUE forms.\n"
    "  #\n"
    "  # fs180 MEASURED: the first draft rebuilt the string with ${s/\\\"$m\\\"/...},\n"
    "  # whose pattern carries literal double quotes that a command line does not\n"
    "  # have. No substitution ever landed, the loop condition stayed true, and one\n"
    "  # --api-key flag hung bash for >60s -- immediately before exec, on every\n"
    "  # launch. The rewrite consumes the string left to right: each iteration\n"
    "  # appends the prefix plus the redacted form to out and advances rest past\n"
    "  # the match, so termination is structural rather than a side effect of the\n"
    "  # replacement happening to match. Every ${var%%\"$m\"*} and ${var#\"$pre$m\"}\n"
    "  # quotes its pattern, which is what makes bash match it literally; unquoted,\n"
    "  # a command line containing * or [ would splice itself.\n"
    "  local s=\"$1\" out=\"\" rest=\"$1\" re m v flag pre\n"
    "  re=\"--[A-Za-z0-9_-]*(${_FS180_NPAT})[A-Za-z0-9_-]*[[:space:]]+([^[:space:]\\\"'<][^[:space:]]*)\"\n"
    "  while [[ \"$rest\" =~ $re ]]; do\n"
    "    m=\"${BASH_REMATCH[0]}\"\n"
    "    v=\"${BASH_REMATCH[2]}\"\n"
    "    flag=\"${m%\"$v\"}\"\n"
    "    pre=\"${rest%%\"$m\"*}\"\n"
    "    out=\"${out}${pre}${flag}<redacted>\"\n"
    "    rest=\"${rest#\"$pre$m\"}\"\n"
    "    FS180_REDACTS=$((FS180_REDACTS + 1))\n"
    "  done\n"
    "  s=\"${out}${rest}\"; out=\"\"; rest=\"$s\"\n"
    "  re=\"--[A-Za-z0-9_-]*(${_FS180_NPAT})[A-Za-z0-9_-]*=[^[:space:]\\\"'<][^[:space:]]*\"\n"
    "  while [[ \"$rest\" =~ $re ]]; do\n"
    "    m=\"${BASH_REMATCH[0]}\"\n"
    "    pre=\"${rest%%\"$m\"*}\"\n"
    "    out=\"${out}${pre}${m%%=*}=<redacted>\"\n"
    "    rest=\"${rest#\"$pre$m\"}\"\n"
    "    FS180_REDACTS=$((FS180_REDACTS + 1))\n"
    "  done\n"
    "  s=\"${out}${rest}\"; out=\"\"; rest=\"$s\"\n"
    "  re=\"[A-Za-z0-9_]*(${_FS180_NPAT})[A-Za-z0-9_]*=[^[:space:]\\\"'<][^[:space:]]*\"\n"
    "  while [[ \"$rest\" =~ $re ]]; do\n"
    "    m=\"${BASH_REMATCH[0]}\"\n"
    "    pre=\"${rest%%\"$m\"*}\"\n"
    "    out=\"${out}${pre}${m%%=*}=<redacted>\"\n"
    "    rest=\"${rest#\"$pre$m\"}\"\n"
    "    FS180_REDACTS=$((FS180_REDACTS + 1))\n"
    "  done\n"
    "  # A redacted value begins with '<', which every value class above excludes,\n"
    "  # so a site already substituted can never be counted a second time.\n"
    "  FS180_REDACTED=\"${out}${rest}\"\n"
    "}\n"
    "\n"
    "fs180_emit_provenance() {\n"
    "  # stdout: one JSON object. Called with its output redirected to\n"
    "  # FS180_PROV_PATH inside an if-condition, so any failure here degrades to\n"
    "  # the announced write=FAILED state and can never abort the launch.\n"
    "  local esc_composed esc_raw mode_json top_json top_state job_json env_body ent v val\n"
    "  local unexp_body dcl flags\n"
    "  fs180_redact_cmd \"${LAUNCH_CMD:-}\"\n"
    "  esc_composed=\"$(fs180_json_escape \"$FS180_REDACTED\")\"\n"
    "  fs180_redact_cmd \"${LAUNCH_CMD_RAW:-}\"\n"
    "  esc_raw=\"$(fs180_json_escape \"$FS180_REDACTED\")\"\n"
    "  if [[ -n \"${FS_ENGINE_LAUNCH_MODE:-}\" ]]; then\n"
    "    mode_json=\"\\\"$(fs180_json_escape \"$FS_ENGINE_LAUNCH_MODE\")\\\"\"\n"
    "  else\n"
    "    # explicit null, never a silent omission: a reader must be able to\n"
    "    # distinguish 'not set' from 'the writer forgot'.\n"
    "    mode_json=\"null\"\n"
    "  fi\n"
    "  top_json=\"null\"; top_state=\"absent\"\n"
    "  if [[ \"${top_args[@]+set}\" == set && \"${#top_args[@]}\" -gt 0 ]]; then\n"
    "    val=\"$(printf '%s ' \"${top_args[@]}\")\"; val=\"${val% }\"\n"
    "    top_json=\"\\\"$(fs180_json_escape \"$val\")\\\"\"\n"
    "    top_state=\"present\"\n"
    "  fi\n"
    "  if [[ -n \"${SLURM_JOB_ID:-}\" ]]; then\n"
    "    job_json=\"\\\"$(fs180_json_escape \"$SLURM_JOB_ID\")\\\"\"\n"
    "  else\n"
    "    job_json=\"null\"\n"
    "  fi\n"
    "  # The fs_env half: the trainer's knobs arrive as env AND as argv, so both\n"
    "  # halves must be on record for the pair to be reproducible. Name-based\n"
    "  # redaction keeps the name and replaces the value with the marker.\n"
    "  env_body=\"\"\n"
    "  unexp_body=\"\"\n"
    "  while IFS= read -r v; do\n"
    "    [[ -n \"$v\" ]] || continue\n"
    "    # fs180: declare -p prints the flags as the second word -- reading them\n"
    "    # that way, instead of globbing the whole line for an x, keeps a VALUE\n"
    "    # containing x from being read as the export flag.\n"
    "    dcl=\"$(declare -p \"$v\" 2>/dev/null)\"; flags=\"${dcl#declare }\"; flags=\"${flags%% *}\"\n"
    "    if [[ \"$flags\" != *x* ]]; then\n"
    "      if [[ -n \"$unexp_body\" ]]; then unexp_body=\"${unexp_body}, \"; fi\n"
    "      unexp_body=\"${unexp_body}\\\"$(fs180_json_escape \"$v\")\\\"\"\n"
    "    fi\n"
    "    if [[ \"$v\" =~ (${_FS180_NPAT}) ]]; then\n"
    "      FS180_REDACTS=$((FS180_REDACTS + 1))\n"
    "      ent=\"  \\\"$(fs180_json_escape \"$v\")\\\": \\\"<redacted>\\\"\"\n"
    "    else\n"
    "      if [[ \"${!v+x}\" == x ]]; then val=\"${!v}\"; else val=\"\"; fi\n"
    "      ent=\"  \\\"$(fs180_json_escape \"$v\")\\\": \\\"$(fs180_json_escape \"$val\")\\\"\"\n"
    "    fi\n"
    "    if [[ -n \"$env_body\" ]]; then\n"
    "      env_body=\"${env_body},\n"
    "${ent}\"\n"
    "    else\n"
    "      env_body=\"$ent\"\n"
    "    fi\n"
    "  # fs180 MEASURED: compgen -e lists only EXPORTED names, and the launcher\n"
    "  # sets FS_ITERATION_BUDGET and FS_EARLY_SAVE_STEPS with a bare assignment\n"
    "  # on the resume arm. run_in_container forwards by NAME from an allowlist,\n"
    "  # so those two reach the trainer while an exported-only census would have\n"
    "  # silently dropped the pair of numbers that define the resume segment --\n"
    "  # a reproducibility record omitting exactly what it exists to hold.\n"
    "  # compgen -v is the shell's own set; the export state is recorded per name\n"
    "  # rather than used as a filter, so the record states its denominator.\n"
    "  done < <(compgen -v FS_ | LC_ALL=C sort)\n"
    "  printf '{\\n'\n"
    "  printf '  \\\"launch_cmd_composed\\\": \\\"%s\\\",\\n' \"$esc_composed\"\n"
    "  printf '  \\\"launch_cmd_raw\\\": \\\"%s\\\",\\n' \"$esc_raw\"\n"
    "  printf '  \\\"engine_launch_mode\\\": %s,\\n' \"$mode_json\"\n"
    "  printf '  \\\"world_size\\\": %s,\\n' \"${WORLD_SIZE:-null}\"\n"
    "  printf '  \\\"world_size_source\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${WORLD_SIZE_SOURCE:-}\")\"\n"
    "  printf '  \\\"gpus_per_node\\\": %s,\\n' \"${FS_GPUS_PER_NODE:-null}\"\n"
    "  printf '  \\\"top_args\\\": %s,\\n' \"$top_json\"\n"
    "  printf '  \\\"top_args_state\\\": \\\"%s\\\",\\n' \"$top_state\"\n"
    "  printf '  \\\"out_dir\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${OUT_DIR:-}\")\"\n"
    "  printf '  \\\"image\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${IMAGE:-}\")\"\n"
    "  printf '  \\\"model_dir\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${MODEL_DIR:-}\")\"\n"
    "  printf '  \\\"dataset_dir\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${DATASET_DIR:-}\")\"\n"
    "  printf '  \\\"config_file\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${CONFIG_FILE:-}\")\"\n"
    "  printf '  \\\"phase\\\": \\\"%s\\\",\\n' \"$(fs180_json_escape \"${FS_PHASE:-train}\")\"\n"
    "  printf '  \\\"probe\\\": %s,\\n' \"${PROBE:-null}\"\n"
    "  printf '  \\\"job_id\\\": %s,\\n' \"$job_json\"\n"
    "  # hostname is deliberately NOT emitted: it is estate-identifying and this\n"
    "  # record is written onto a shared filesystem.\n"
    "  printf '  \\\"nodes\\\": %s,\\n' \"${SLURM_NNODES:-1}\"\n"
    "  printf '  \\\"fs_env_scope\\\": \\\"%s\\\",\\n' 'every FS_ name this shell held at launch (compgen -v), not only the exported subset; export state is recorded per name in fs_env_not_exported rather than used as a filter'\n"
    "  printf '  \\\"fs_env\\\": {\\n%s\\n  },\\n' \"$env_body\"\n"
    "  printf '  \\\"fs_env_not_exported\\\": [%s],\\n' \"$unexp_body\"\n"
    "  printf '  \\\"redactions\\\": %s\\n' \"$FS180_REDACTS\"\n"
    "  printf '}\\n'\n"
    "}\n"
    "\n"
    "if fs180_emit_provenance > \"$FS180_PROV_PATH\" 2>/dev/null; then\n"
    "  printf 'PROVENANCE path=%s write=ok redactions=%s\\n' \"$FS180_PROV_PATH\" \"$FS180_REDACTS\" | tee -a \"$RUN_LOG\" || true\n"
    "else\n"
    "  # degraded, not failed -- and it SAYS so, so a silent gap can never pass\n"
    "  # for a record.\n"
    "  printf 'PROVENANCE path=%s write=FAILED redactions=%s (launch continues; run is degraded, not failed)\\n' \"$FS180_PROV_PATH\" \"$FS180_REDACTS\" | tee -a \"$RUN_LOG\" || true\n"
    "fi\n"
    "unset -f fs180_json_escape fs180_redact_cmd fs180_emit_provenance 2>/dev/null || true\n"
    "unset _FS180_NPAT FS180_REDACTED FS180_PROV_PATH FS180_REDACTS\n"
    + END_MARK + "---------------------------------------------------------------\n"
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _logsite_census(text: str) -> list[tuple[int, str]]:
    """Count echo/printf/tee sites that NAME the launch command.

    A logging site names the command; an execution site passes it to a program.
    The execution line pipes run OUTPUT through tee AFTER the command mention,
    so only a tool keyword appearing BEFORE the variable mention on a line
    counts as a record site.
    """
    hits: list[tuple[int, str]] = []
    for no, ln in enumerate(text.splitlines(), 1):
        pos = min((ln.find(n) for n in CENSUS_VNAMES if ln.find(n) >= 0), default=-1)
        if pos < 0:
            continue
        early = ln[:pos]
        for tool in ("echo", "printf", "tee"):
            if re.search(r"(?<![\w\"'\-])" + tool + r"(?![\w'\-])", early):
                hits.append((no, tool))
                break
    return hits


_MARKER_RE = re.compile(r"(?:fs|fix)\d+[a-z]?:\s")


def _marker_scan(text: str) -> tuple[int, list[tuple[int, str]]]:
    """Every fsNNN:/fixNN: marker in `text`, and those of them bash would EXECUTE.

    A marker is inert if a '#' opens a comment before it, or if it sits inside an
    unclosed quote (printf/echo prose). Anything else is a bare command word, which
    `bash -n` accepts and the shell then fails on at rc 127. Returns (total, bare) so
    the caller can state a denominator instead of only a count of hits: a scan that
    reports "0 bare" over 0 markers examined has measured nothing.
    """
    total = 0
    bare: list[tuple[int, str]] = []
    for no, ln in enumerate(text.splitlines(), 1):
        for m in _MARKER_RE.finditer(ln):
            total += 1
            before = ln[:m.start()]
            if "#" in before:
                continue
            if before.count("'") % 2 == 1 or before.count('"') % 2 == 1:
                continue
            bare.append((no, ln.strip()[:100]))
    return total, bare


def _marker_total(text: str) -> int:
    return _marker_scan(text)[0]


def _uncommented_markers(text: str) -> list[tuple[int, str]]:
    return _marker_scan(text)[1]


def _transform(text: str) -> tuple[str, dict[str, int], bool]:
    counts = {"assign": text.count(A_ASSIGN), "recompose": text.count(A_RE),
              "exec": text.count(A_EXEC)}
    if BLOCK_MARK in text and RAW_MARK in text:
        return text, counts, True
    out: list[str] = []
    for ln in text.splitlines(keepends=True):
        if A_EXEC in ln:
            out.append(BLOCK)  # provenance lands immediately before execution
            out.append(ln)
        elif A_ASSIGN in ln:
            out.append(ln)
            out.append(RAW_LINE)  # capture the operator's command at ingestion
        else:
            out.append(ln)
    return "".join(out), counts, False


def _gate_results(pre: str, new: str, counts: dict[str, int]) -> list[tuple[str, bool, str]]:
    gres: list[tuple[str, bool, str]] = []
    gres.append(("G1", counts["assign"] == 1,
                 f"ingestion anchor count={counts['assign']} need=1 of 1 (the "
                 "ingestion line is where the raw command must be captured)"))
    gres.append(("G2", counts["recompose"] == 1,
                 f"recomposition anchor count={counts['recompose']} need=1 of 1"))
    gres.append(("G3", counts["exec"] == 1,
                 f"execution anchor count={counts['exec']} need=1 of 1"))
    hits = _logsite_census(pre)
    nlines = len(pre.splitlines())
    gres.append(("G4", len(hits) == 0,
                 f"MUST_FIRE premise: echo/printf/tee sites naming the launch "
                 f"command = {len(hits)} of {nlines} pre-image lines scanned, "
                 f"need 0{'' if not hits else ' -- REFUSE: a record already exists, ' + str(hits[:3])}"))
    nl = new.splitlines()
    def idx(needle: str) -> list[int]:
        return [i for i, l in enumerate(nl) if needle in l]
    ia, ir, ire, ib, ix = idx(A_ASSIGN), idx(RAW_MARK), idx(A_RE), idx(BLOCK_MARK), idx(A_EXEC)
    order_ok = (len(ia) == 1 and len(ir) == 1 and len(ire) == 1 and len(ib) == 1
                and len(ix) == 1
                and ia[0] < ir[0] < ire[0] < ib[0] < ix[0])
    gres.append(("G5", order_ok,
                 f"ordering by line index in post-image: ingest@{ia} raw@{ir} "
                 f"recompose@{ire} block@{ib} exec@{ix} -- the record sits after "
                 "the composer and before execution, not by hope"))
    gres.append(("G6", new.count(RAW_MARK) == 1 and 'LAUNCH_CMD_RAW="$LAUNCH_CMD"' in new,
                 f"raw capture present exactly once (marker count="
                 f"{new.count(RAW_MARK)} need=1 of 1)"))
    derived = new.count(DERIVED)
    gres.append(("G7", derived == 1,
                 f"derived path ${'{'}RUN_LOG%.log{'}'}.provenance.json count={derived} "
                 "need=1 of 1 (the #150 lesson: derive from RUN_LOG, never "
                 "recompute a second way)"))
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", prefix=MARK + "-syntax-",
                                         delete=False) as tf:
            tf.write(new)
            tname = tf.name
        proc = subprocess.run(["bash", "-n", tname], capture_output=True, text=True)
        os.unlink(tname)
        bashn_ok, bashn_msg = proc.returncode == 0, (proc.stderr.strip() or "clean")
    except OSError as exc:
        bashn_ok, bashn_msg = False, f"bash -n unrunnable: {exc}"
    gres.append(("G8", bashn_ok, "bash -n on post-image: " + bashn_msg))
    again, _, already2 = _transform(new)
    gres.append(("G9", again == new and already2,
                 "byte-idempotence: re-running the transform on its own output "
                 "is a byte-exact no-op"))
    gres.append(("G10", '"host' + 'name"' not in new,
                 "estate-identifying host name key is absent from the record "
                 "contract (omitted by design, not merely empty)"))
    # G11 exists because G8 cannot see this class. `foo=bar  fs180: prose` parses fine --
    # bash reads `fs180:` as a command word -- so `bash -n` returns clean and the launcher
    # still dies rc 127 at runtime. Measured on hardware, job 37366, at launcher L:857, one
    # line after a PASSING 7-rank collective probe. The denominator is stated on purpose:
    # this ranges over EVERY marker in the post-image, not just the one this stage inserts,
    # because the defect class belongs to the artifact and not to its most recent author.
    bare = _uncommented_markers(new)
    gres.append(("G11", not bare,
                 f"every fsNNN/fixNN marker in the post-image is commented or quoted "
                 f"({_marker_total(new)} marker(s) examined, {len(bare)} bare)"
                 + ("" if not bare else "; bare at line(s) "
                    + ", ".join(str(n) for n, _ in bare))))
    return gres


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _extract_block(new: str) -> str:
    lines = new.splitlines()
    start = next(i for i, l in enumerate(lines) if BLOCK_MARK in l)
    end = next(i for i, l in enumerate(lines) if END_MARK in l and i > start)
    return "\n".join(lines[start:end + 1]) + "\n"


def _harness(block: str, *, run_log: str, composed: str, raw: str,
             exports: dict[str, str], top_args_line: str,
             plain: dict[str, str] | None = None) -> str:
    # `plain` is the half `exports` cannot express: a shell variable that is SET
    # and not EXPORTED. The launcher creates exactly that on its resume arm, so a
    # harness that could only plant exported names could not drive the case the
    # record was getting wrong.
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "IFS=$'\\n\\t'"]
    for k, v in exports.items():
        lines.append(f"export {k}={_shq(v)}")
    for k, v in (plain or {}).items():
        lines.append(f"{k}={_shq(v)}")
    lines += [
        f"RUN_LOG={_shq(run_log)}",
        f"OUT_DIR={_shq(str(pathlib.Path(run_log).parent))}",
        "IMAGE=/opt/fixture/image.sif",
        "MODEL_DIR=/opt/fixture/model",
        "DATASET_DIR=/opt/fixture/dataset",
        "CONFIG_FILE=/opt/fixture/config.yaml",
        "FS_GPUS_PER_NODE=8",
        "WORLD_SIZE=8",
        "WORLD_SIZE_SOURCE=composer-measured-fixture",
        "PROBE=0",
        f"LAUNCH_CMD={_shq(composed)}",
        f"LAUNCH_CMD_RAW={_shq(raw)}",
        top_args_line,
        block,
    ]
    return "\n".join(lines) + "\n"


def _run_bash(script: str) -> subprocess.CompletedProcess:
    # Real bash, no cluster, no GPUs. The environment is scrubbed of FS_*/SLURM_*
    # so the fs_env half of the record contains only what the control planted.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("FS_") and not k.startswith("SLURM_")}
    with tempfile.NamedTemporaryFile("w", suffix=".sh", prefix=MARK + "-ctl-",
                                     delete=False) as tf:
        tf.write(script)
        name = tf.name
    try:
        try:
            return subprocess.run(["bash", name], capture_output=True, text=True,
                                  env=env, timeout=60)
        except subprocess.TimeoutExpired as exc:
            # fs180: a control that HANGS must fail as that control, not as an
            # exception in the stage. The first draft's redaction loop could not
            # terminate, and the stage reported "TimeoutExpired" with no drill
            # named -- a real defect arriving as an unattributable one. 124 is the
            # conventional timed-out status; the caller's assertions read it as a
            # failure of whichever leg produced it.
            return subprocess.CompletedProcess(
                args=exc.cmd, returncode=124, stdout="",
                stderr=f"{MARK}: control script did not terminate within 60s "
                       f"(script={name}); treated as a failure of this drill")
    finally:
        os.unlink(name)


def _controls(new: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    block = _extract_block(new)
    with tempfile.TemporaryDirectory(prefix=MARK + "-") as td:
        run_log = f"{td}/launch.interactive.log"
        prov = run_log[:-len(".log")] + ".provenance.json"

        # C1 MUST_PASS -- pins the write-and-parse half: the record must be
        # machine-readable; a record that json.loads cannot open is a comment.
        c1 = _harness(block, run_log=run_log,
                      composed="torchrun --nproc_per_node=8 trainer.py --steps 40",
                      raw="trainer.py --steps 40",
                      exports={"FS_ENGINE_LAUNCH_MODE": "torchrun",
                               "FS_ITERATION_BUDGET": "40"},
                      top_args_line="top_args=(--slurm-ntasks 8)")
        p = _run_bash(c1)
        parsed = None
        try:
            parsed = json.loads(pathlib.Path(prov).read_text("utf-8"))
        except (OSError, ValueError):
            parsed = None
        good = (p.returncode == 0 and parsed is not None
                and parsed.get("redactions") == 0
                and parsed.get("top_args_state") == "present")
        ok += int(good)
        notes.append(
            "C1 MUST_PASS parseable half: populated env -> file at derived path, "
            f"json.loads ok, rc={p.returncode}, redactions="
            f"{parsed.get('redactions') if parsed else 'unparseable'} "
            "(exact-record claim allowed at 0) " + ("PASS" if good else "FAIL " + (p.stderr or "")))

        # C2 MUST_PASS -- pins the verbatim-composed half: embedded spaces and
        # quotes survive byte for byte, the case naive shell quoting breaks.
        c2_composed = ("torchrun --nproc_per_node=8 /opt/fixture/train.py "
                       "--out \"/tmp/fixture dir/out path\" --tag 'a b' --steps 9")
        p = _run_bash(_harness(block, run_log=run_log, composed=c2_composed,
                               raw="/opt/fixture/train.py --steps 9",
                               exports={"FS_ENGINE_LAUNCH_MODE": "torchrun"},
                               top_args_line="top_args=()"))
        d2 = None
        try:
            d2 = json.loads(pathlib.Path(prov).read_text("utf-8"))
        except (OSError, ValueError):
            d2 = None
        good = (p.returncode == 0 and d2 is not None
                and d2.get("launch_cmd_composed") == c2_composed)
        ok += int(good)
        notes.append(
            "C2 MUST_PASS verbatim half: launch_cmd_composed equals the composed "
            "command byte for byte, spaces and quotes included "
            + ("PASS" if good else "FAIL mismatch or unparseable"))

        # C3 MUST_PASS -- pins the raw-vs-composed pair: losing either half
        # loses what the operator asked for or what actually ran.
        c3_raw = "/opt/fixture/train.py --resume-tolerance 10.0 --steps 200"
        c3_composed = "torchrun --nproc_per_node=8 " + c3_raw
        p = _run_bash(_harness(block, run_log=run_log, composed=c3_composed,
                               raw=c3_raw, exports={"FS_ENGINE_LAUNCH_MODE": "torchrun"},
                               top_args_line="top_args=()"))
        d3 = None
        try:
            d3 = json.loads(pathlib.Path(prov).read_text("utf-8"))
        except (OSError, ValueError):
            d3 = None
        good = (p.returncode == 0 and d3 is not None
                and d3.get("launch_cmd_raw") == c3_raw
                and d3.get("launch_cmd_composed") == c3_composed
                and c3_raw != c3_composed)
        ok += int(good)
        notes.append(
            "C3 MUST_PASS pair half: composer prepended a wrapper, raw differs "
            "from composed, and BOTH halves are on the record "
            + ("PASS" if good else "FAIL"))

        # C4 MUST_FIRE -- the redaction control, both polarities in one fixture:
        # the planted marker (an obvious non-secret word) must appear 0 times
        # while an ordinary value must survive, and redactions must equal the
        # number planted (1 env + 1 raw + 1 composed = 3).
        sentinel = "sentinel_must_not_appear"
        c4_cmd = ("trainer.py --resume-tolerance 10.0 "
                  "--api-key " + sentinel + " --iterations 5")
        p = _run_bash(_harness(block, run_log=run_log, composed=c4_cmd, raw=c4_cmd,
                               exports={"FS_FA" + "KE_TO" + "KEN": sentinel,
                                        "FS_ITERATION_BUDGET": "200",
                                        "FS_ENGINE_LAUNCH_MODE": "torchrun"},
                               top_args_line="top_args=()"))
        blob = b""
        d4 = None
        try:
            blob = pathlib.Path(prov).read_bytes()
            d4 = json.loads(blob.decode("utf-8"))
        except (OSError, ValueError):
            d4 = None
        neg = blob.count(sentinel.encode()) == 0
        posi = (d4 is not None
                and d4.get("fs_env", {}).get("FS_ITERATION_BUDGET") == "200"
                and b'"200"' in blob)
        named_kept = d4 is not None and ("FS_FA" + "KE_TO" + "KEN") in d4.get("fs_env", {})
        counted = d4 is not None and d4.get("redactions") == 3
        good = (p.returncode == 0 and neg and posi and named_kept and counted)
        ok += int(good)
        notes.append(
            f"C4 MUST_FIRE redaction, both polarities: marker bytes={blob.count(sentinel.encode())} "
            f"need 0, FS_ITERATION_BUDGET survives={posi}, name kept/"
            f"value marked={named_kept}, redactions="
            f"{d4.get('redactions') if d4 else 'unparseable'} need 3 of 3 planted "
            + ("PASS" if good else "FAIL"))

        # C5 MUST_FIRE -- pins the explicit-null half: unset mode and empty
        # top_args are recorded as null/absent, never silently omitted, so a
        # reader can distinguish 'not set' from 'the writer forgot'.
        p = _run_bash(_harness(block, run_log=run_log,
                               composed="trainer.py --steps 1",
                               raw="trainer.py --steps 1",
                               exports={}, top_args_line="top_args=()"))
        d5 = None
        try:
            d5 = json.loads(pathlib.Path(prov).read_text("utf-8"))
        except (OSError, ValueError):
            d5 = None
        good = (p.returncode == 0 and d5 is not None
                and d5.get("engine_launch_mode") is None
                and d5.get("top_args") is None
                and d5.get("top_args_state") == "absent"
                and d5.get("job_id") is None
                and "engine_launch_mode" in d5 and "top_args" in d5)
        ok += int(good)
        notes.append(
            "C5 MUST_FIRE explicit-null half: mode unset + top_args empty are "
            "null/absent on the record, keys present, write still ok "
            + ("PASS" if good else "FAIL"))

        # C6 MUST_FIRE -- pins the degraded-and-announcing half: an unwritable
        # target must not abort the launch, and the PROVENANCE line must say
        # the write failed. The parent is made a regular FILE, which fails for
        # any euid, unlike a chmod.
        blocker = pathlib.Path(td) / "notadir"
        blocker.write_text("occupied\n", "utf-8")
        bad_log = f"{td}/notadir/launch.interactive.log"
        p = _run_bash(_harness(block, run_log=bad_log,
                               composed="trainer.py --steps 1",
                               raw="trainer.py --steps 1",
                               exports={}, top_args_line="top_args=()"))
        good = (p.returncode == 0 and "PROVENANCE" in p.stdout
                and "write=FAILED" in p.stdout and "write=ok" not in p.stdout)
        ok += int(good)
        notes.append(
            "C6 MUST_FIRE degraded half: unwritable target -> block rc=0 AND "
            f"PROVENANCE says the write failed (rc={p.returncode}, "
            f"announced={'write=FAILED' in p.stdout}) "
            + ("PASS" if good else "FAIL " + (p.stderr or "")))

        # C7 MUST_PASS -- pins the derivation-pairing half: the suffix
        # substitution pairs the record with the log name for both the
        # SLURM_JOB_ID set and unset (interactive) cases: 2 ids, 2 pairs.
        c7 = (
            "set -euo pipefail\n"
            "LOG_DIR=/tmp/" + MARK + "-c7-logs\n"
            "unset SLURM_JOB_ID\n"
            "RUN_LOG=\"$LOG_DIR/launch.${SLURM_JOB_ID:-interactive}.log\"\n"
            "printf '%s\\n%s\\n' \"$RUN_LOG\" \"${RUN_LOG%.log}.provenance.json\"\n"
            "SLURM_JOB_ID=4242\n"
            "RUN_LOG=\"$LOG_DIR/launch.${SLURM_JOB_ID:-interactive}.log\"\n"
            "printf '%s\\n%s\\n' \"$RUN_LOG\" \"${RUN_LOG%.log}.provenance.json\"\n"
        )
        p = _run_bash(c7)
        lines = p.stdout.splitlines()
        expect = [
            "/tmp/" + MARK + "-c7-logs/launch.interactive.log",
            "/tmp/" + MARK + "-c7-logs/launch.interactive.provenance.json",
            "/tmp/" + MARK + "-c7-logs/launch.4242.log",
            "/tmp/" + MARK + "-c7-logs/launch.4242.provenance.json",
        ]
        pairs = 0
        if len(lines) == 4:
            for i in (0, 2):
                if lines[i] == expect[i] and lines[i + 1] == expect[i + 1]:
                    pairs += 1
        good = p.returncode == 0 and pairs == 2
        ok += int(good)
        notes.append(
            f"C7 MUST_PASS pairing half: derived path pairs with the log name "
            f"{pairs} of 2 cases (job id set and unset) "
            + ("PASS" if good else f"FAIL lines={lines}"))

        # C8 MUST_FIRE -- pins the census DENOMINATOR. The launcher sets
        # FS_ITERATION_BUDGET and FS_EARLY_SAVE_STEPS with a bare assignment on
        # the resume arm (:703-704), and run_in_container forwards by name from
        # an allowlist, so the trainer receives them while `compgen -e` -- the
        # first draft's oracle -- cannot see them. A record that silently drops
        # the two numbers defining the resume segment is the #180 defect wearing
        # the fix's clothes. This drives one exported and one non-exported name
        # and requires BOTH on the record, with the export state stated.
        c8_log = f"{td}/c8/launch.interactive.log"
        pathlib.Path(f"{td}/c8").mkdir(parents=True, exist_ok=True)
        p = _run_bash(_harness(
            block, run_log=c8_log,
            composed="trainer.py --steps 5", raw="trainer.py --steps 5",
            exports={"FS_EXPORTED_KNOB": "seen-by-a-child"},
            plain={"FS_BARE_KNOB": "5"},
            top_args_line="top_args=()"))
        rec: dict = {}
        try:
            rec = json.loads(pathlib.Path(c8_log[:-len(".log")]
                                          + ".provenance.json").read_text("utf-8"))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            notes.append(f"{TAG} C8 could not read its record: {exc}")
        env = rec.get("fs_env", {})
        unexp = rec.get("fs_env_not_exported", None)
        good = (p.returncode == 0
                and env.get("FS_EXPORTED_KNOB") == "seen-by-a-child"
                and env.get("FS_BARE_KNOB") == "5"
                and isinstance(unexp, list)
                and "FS_BARE_KNOB" in unexp
                and "FS_EXPORTED_KNOB" not in unexp
                and isinstance(rec.get("fs_env_scope"), str)
                and rec["fs_env_scope"].strip() != "")
        ok += int(good)
        notes.append(
            "C8 MUST_FIRE census denominator: a SET-but-not-EXPORTED FS_ name is "
            f"on the record (bare_present={env.get('FS_BARE_KNOB') == '5'}, "
            f"named_unexported={isinstance(unexp, list) and 'FS_BARE_KNOB' in unexp}, "
            f"exported_not_named={isinstance(unexp, list) and 'FS_EXPORTED_KNOB' not in unexp}, "
            f"scope_stated={isinstance(rec.get('fs_env_scope'), str)}) "
            + ("PASS" if good else "FAIL " + (p.stderr or "")[:300]))

        # C9 MUST_FIRE -- the marker scanner must go RED on the exact defect that was
        # measured on hardware, and it must do so for the RIGHT REASON: not because
        # the text looks odd, but because bash actually executes the marker. So the
        # fixture is run, not merely matched. `bash -n` is run alongside to pin the
        # reason G8 could not see this: the same bytes are syntactically valid.
        planted = ('LAUNCH_CMD_RAW="$LAUNCH_CMD"  ' + MARK + ": captured the "
                   "operator-supplied command\n")
        seen = _uncommented_markers(planted)
        pn = _run_bash("set -e\nLAUNCH_CMD=x\n" + planted)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tf:
            tf.write(planted)
            syn_name = tf.name
        syn = subprocess.run(["bash", "-n", syn_name], capture_output=True, text=True)
        os.unlink(syn_name)
        good = (len(seen) == 1 and pn.returncode == 127 and syn.returncode == 0)
        ok += int(good)
        notes.append(
            f"C9 MUST_FIRE bare marker: scanner flags {len(seen)} of 1 planted, the "
            f"fixture really dies rc={pn.returncode} (need 127 'command not found'), "
            f"and bash -n calls the same bytes clean (rc={syn.returncode}, need 0) -- "
            "which is why G8 cannot stand in for G11 "
            + ("PASS" if good else "FAIL " + (pn.stderr or syn.stderr or "")[:300]))

        # C10 MUST_PASS -- the scanner must NOT fire on the three inert forms, or it
        # would redden 71 of the 72 markers the artifact legitimately carries.
        inert = ("# " + MARK + ": a leading comment\n"
                 'X=1  # ' + MARK + ": a trailing comment\n"
                 "printf 'NOTICE: " + MARK + ": prose inside quotes\\n'\n")
        tot, bare2 = _marker_scan(inert)
        good = (tot == 3 and not bare2)
        ok += int(good)
        notes.append(
            f"C10 MUST_PASS inert forms: {tot} of 3 markers examined, {len(bare2)} "
            "flagged (need 0) across leading-comment, trailing-comment and quoted-prose "
            + ("PASS" if good else "FAIL"))
    return ok, notes


def main() -> int:
    # The build driver invokes every stage as `python3 <stage>` with NO
    # arguments, so bare invocation must APPLY; requiring a flag would make the
    # stage a no-op inside the build while passing by hand.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_launch_provenance.py [--apply|--check]   (no argument == --apply)")
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

    mb, mr = text.count(BLOCK_MARK), text.count(RAW_MARK)
    if mb or mr:
        if mb == 1 and mr == 1:
            # Second run: byte-exact no-op, but re-prove the gates so an
            # already-applied yet malformed file is RED (5), not a silent pass.
            new, counts, _ = _transform(text)
            gres = _gate_results(text, new, counts)
            gates = 0
            for name, good, detail in gres:
                print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
                gates += int(good)
            print("verdict: already applied; byte-idempotent no-op")
            return 0 if gates == len(gres) else 5
        _stderr(
            f"REFUSE 96: half-applied state (block marker x{mb}, raw marker x{mr}); "
            "the stage does not recognise this file and will not guess"
        )
        return 96

    new, counts, _already = _transform(text)

    gates = 0
    gres = _gate_results(text, new, counts)
    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    cok, cnotes = _controls(new)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls "
                f"{cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  provenance block + raw capture would be applied, "
              f"{gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{TAG} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # Four states only (0/5/95/96): an unhandled exception is a REFUSE with a
    # named message, never a bare rc=1 that collapses the states into one.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {TAG} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())