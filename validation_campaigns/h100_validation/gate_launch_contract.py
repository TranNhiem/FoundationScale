#!/usr/bin/env python3
"""gate_launch_contract.py -- Deliverable D: a training command GENERATED from the code
and gated against it.

WHY THIS EXISTS
  The reproducible training command a new engineer can run without reverse-engineering
  the codebase cannot be a hand-written document: it drifts the moment the launcher
  grows a new required variable. That has happened twice already (FS_ALLOWED_PATH_ROOTS
  arrived; FS_ENGINE_LAUNCH_MODE is landing). So the command is DERIVED from the two
  generated bash artifacts and pinned to them by six gates (L1..L6) -- the same trick
  the env-drift gate plays by pinning the allowlist against the exports. A document
  that can drift from the code is a claim nothing backs.

READS
  h100/gen/launch_fs_h100.fixed.sh        (generated launcher)
  h100/gen/fs_container_backend.bound.sh  (generated container backend)

WRITES
  h100/gen/LAUNCH.md -- only when every gate is green and the redaction scan is clean
  (this repo is PUBLIC).

SCOPE LIMITS
  * Exactly three refusal idioms are understood (I1 req_env calls, I2 inline emptiness
    refusals, I3 the marker phrase "required, no default by design"). Lines that look
    like refusals but match no idiom are counted as UNPARSED and printed; coverage is
    reported as a floor, never silently claimed complete.
  * The two bash artifacts are never modified; the L6 drills run on in-memory copies.
  * stdlib only.
"""

import os
import re
import sys
from pathlib import Path
from fs_estate_pat import estate_ident_pat

ROOT = Path(__file__).resolve().parent
LAUNCH = ROOT / "h100/gen/launch_fs_h100.fixed.sh"
BACKEND = ROOT / "h100/gen/fs_container_backend.bound.sh"
OUT = ROOT / "h100/gen/LAUNCH.md"

MARKER = "required, no default by design"

# L3 waiver list. Every entry MUST carry a stated reason: a waiver without a reason is
# just a longer list, and the gate below refuses to treat it as declared otherwise.
OPTIONAL = {
    "FS_PHASE": "defaults to train; resume is the other value",
    "FS_ITERATION_BUDGET": "bounded-probe budget; the trainer refuses rather than "
                           "defaulting, but it is not needed for a plain train phase",
    "FS_EARLY_SAVE_STEPS": "bounded-probe budget; the trainer refuses rather than "
                           "defaulting, but it is not needed for a plain train phase",
    "FS_BIND_PATHS": "declared mount plane; empty is legal",
    "FS_ENGINE_LAUNCH_CMD": "supplied by the engine adapter, not by the operator",
}

# L4 buckets. estate/run are exact-match sets; topology is exact names plus the
# *PROCS* / *NTASKS* families. Anything extracted that matches none of these lands in
# 'unclassified' and is PRINTED -- a new required variable must be visible, not
# silently absorbed.
ESTATE = {"FS_ALLOWED_NODE", "FS_ALLOWED_PATH_ROOTS", "FS_CONTAINER_RUNTIME",
          "FS_ALLOCATION", "FS_CONTAINER_SQSH", "IMAGE"}
RUN = {"MODEL_DIR", "DATASET_DIR", "CONFIG_FILE", "OUT_DIR_STABLE", "PROBE", "FS_PHASE"}
TOPOLOGY_EXACT = {"FS_GPUS_PER_NODE", "FS_ENGINE_LAUNCH_MODE"}
TOPOLOGY_PARTS = ("PROCS", "NTASKS")

# The names this generator knows how to place into the command template, in display
# order. An extracted name that is NOT here fails L2 loudly instead of being silently
# dropped -- that failure is the drift alarm, and drill L6/MUST_FIRE relies on it.
KNOWN_EXPORT_ORDER = [
    "MODEL_DIR", "DATASET_DIR", "CONFIG_FILE", "OUT_DIR_STABLE",
    "IMAGE", "FS_GPUS_PER_NODE", "PROBE",
    "FS_ALLOWED_NODE", "FS_ALLOWED_PATH_ROOTS", "FS_CONTAINER_RUNTIME",
    "FS_ALLOCATION", "FS_CONTAINER_SQSH", "FS_ENGINE_LAUNCH_MODE",
]

# Optional names we deliberately surface in the operator-facing command (they are safe
# defaults to publish); the rest of OPTIONAL is a whitelist the L3 gate consults.
TEMPLATE_OPTIONALS = ["FS_PHASE", "FS_BIND_PATHS"]

# LAUNCH.md is written into h100/gen/, next to the launcher, so a relative entry keeps
# the document free of machine-local absolute paths.
ENTRY = "launch_fs_h100.fixed.sh"
ENTRY_ARGS = "--model <model> --dataset <dataset> --num-gpus 8 --config <config>"

# PUBLIC-repo redaction list. Cluster-identifying substrings must never reach OUT;
# the scan runs on the final rendered document, and the same expression is used to
# scrub quoted refusal messages inside the table.
REDACTION_PATTERNS = [p for p in (estate_ident_pat(trailing_pipe=False),
                                  r"/work/", r"ghp_") if p]
REDACT_RE = re.compile("|".join("(?:%s)" % p for p in REDACTION_PATTERNS), re.IGNORECASE)

NAME_RE = r"[A-Z][A-Z0-9_]*"
REQ_ENV_DEF_RE = re.compile(r"\breq_env\s*\(")               # definition: machinery, not a call site
REQ_ENV_CALL_RE = re.compile(r"\breq_env\s+(%s)" % NAME_RE)  # I1 call sites
I2_TEST_RE = re.compile(r"\[\[\s*-n\s+\"?\$\{(%s):-\}\"?\s*\]\]" % NAME_RE)  # I2 condition
CAPS_RE = re.compile(r"\b(%s)\b" % NAME_RE)
ASSIGN_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=")
QUOTE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')                # spans wrapped lines: [^"] eats \n
FAIL_TOKEN_RE = re.compile(r"\bfail 96\b|\bfs_die\b")

# Tokens that look ALL_CAPS but are prose, not variable names; keeps the unparsed-site
# denominator honest without drowning it in false positives.
STOPLIST = {"ENV", "ERROR", "ERR", "WARN", "WARNING", "NOTE", "TODO", "FIXME",
            "USAGE", "EXIT", "ALL", "TRUE", "FALSE", "NONE", "YES", "NO"}

RUN_HINTS = {
    "MODEL_DIR": "path to the model weights / checkpoints for this run",
    "DATASET_DIR": "path to the tokenized dataset for this run",
    "CONFIG_FILE": "trainer config file for this run",
    "OUT_DIR_STABLE": "stable output directory for checkpoints and metrics",
    "PROBE": "probe selection for this run",
    "FS_PHASE": "train (the default) or resume",
}

PLACEHOLDERS = {
    "MODEL_DIR": "<model>",
    "DATASET_DIR": "<dataset>",
    "CONFIG_FILE": "<config>",
    "OUT_DIR_STABLE": "<out-dir>",
    "IMAGE": "<image>",
    "FS_GPUS_PER_NODE": "8",
    "PROBE": "<probe>",
    "FS_PHASE": "train",
    "FS_BIND_PATHS": "",
}


def placeholder(name):
    """Operator-facing value hint for the template; unknown names get <their-own-name>."""
    return PLACEHOLDERS.get(name, "<" + name.lower().replace("_", "-") + ">")


class Extraction:
    """Result of one pass over the two artifacts: names, per-idiom site counts, and the
    unparsed denominator (U) that keeps the coverage claim honest."""

    def __init__(self):
        self.info = {}       # name -> {"idiom", "message", "where"}
        self.i1 = self.i2 = self.i3 = self.u = 0
        self.unparsed = []   # (file label, line number, truncated line text)


def _add(res, idiom, name, message, where):
    old = res.info.get(name)
    # A literal refusal message (I2/I3) is more informative than the generic req_env
    # helper text, so a later literal attribution upgrades an I1 entry.
    if old is None or (old["idiom"] == "I1" and idiom != "I1"):
        res.info[name] = {"idiom": idiom, "message": message, "where": where}


def _fail_message(window):
    """First quoted string after the first fail/fs_die token in the window, or ''."""
    m = FAIL_TOKEN_RE.search(window)
    if not m:
        return ""
    q = QUOTE_RE.search(window, m.end())
    return q.group(1) if q else ""


def _parse_file(res, text, label):
    lines = text.splitlines()
    covered = set()  # line indices already attributed to an idiom (for the U scan)

    # I1 refusal text lives in the helper definition, not at the call sites; capture it
    # so the LAUNCH.md table can quote something real for req_env-only names.
    helper_msg = ""
    dm = re.search(r"req_env\s*\(\)\s*\{(.*?)\}", text, re.S)
    if dm:
        helper_msg = _fail_message(dm.group(1))

    # --- I1: req_env NAME call sites (the definition line is machinery, not a call) ---
    for ln, line in enumerate(lines):
        if REQ_ENV_DEF_RE.search(line):
            continue
        for m in REQ_ENV_CALL_RE.finditer(line):
            name = m.group(1)
            msg = helper_msg.replace("${n}", name).replace("$n", name)
            _add(res, "I1", name, msg or ("refused empty by req_env helper"),
                 "%s:%d" % (label, ln + 1))
            res.i1 += 1

    # --- I2: inline emptiness refusal, possibly wrapped across up to 3 lines ---
    for ln, line in enumerate(lines):
        m2 = I2_TEST_RE.search(line)
        if not m2:
            continue
        window = "\n".join(lines[ln:ln + 3])
        if not FAIL_TOKEN_RE.search(window):
            continue
        _add(res, "I2", m2.group(1), _fail_message(window), "%s:%d" % (label, ln + 1))
        res.i2 += 1
        for j in range(ln, min(ln + 3, len(lines))):
            if FAIL_TOKEN_RE.search(lines[j]):
                covered.add(j)

    # --- I3: marker phrase; attribute to the earliest NAME inside its message string.
    # Offsets are computed on the whole text so a phrase wrapped mid-message still
    # counts exactly once, and the quoted-string scan spans the wrap.
    for m in re.finditer(re.escape(MARKER), text):
        lno = text.count("\n", 0, m.start())
        lstart = text.rfind("\n", 0, m.start()) + 1
        lend = text.find("\n", m.end())
        line_text = text[lstart:lend if lend != -1 else len(text)]
        if REQ_ENV_DEF_RE.search(line_text):
            # The generic helper body cannot name a specific variable; it is the
            # implementation of I1, not a per-name refusal site.
            continue
        qm = None
        for q in QUOTE_RE.finditer(text):
            if q.start(1) <= m.start() <= q.end(1):
                qm = q
                break
        hay = qm.group(1) if qm else line_text
        nm = CAPS_RE.search(hay)
        if not nm:
            continue
        _add(res, "I3", nm.group(1), hay, "%s:%d" % (label, lno + 1))
        res.i3 += 1
        span_end = text.count("\n", 0, qm.end(1)) + 1 if qm else lno + 1
        for j in range(lno, min(span_end + 1, len(lines))):
            if FAIL_TOKEN_RE.search(lines[j]):
                covered.add(j)

    # --- U: refusal-looking lines that matched NO idiom. Computed, never assumed. ---
    for ln, line in enumerate(lines):
        if ln in covered or REQ_ENV_DEF_RE.search(line):
            continue
        if not FAIL_TOKEN_RE.search(line):
            continue
        toks = [t for t in CAPS_RE.findall(line) if t not in STOPLIST]
        if toks:
            res.u += 1
            res.unparsed.append((label, ln + 1, line.strip()[:110]))


def assigned_names(template_text):
    return set(ASSIGN_RE.findall(template_text))


def build_template(extracted):
    """Render the sbatch + plain-env command pair from the KNOWN ordered inventory.
    Extracted names outside that inventory are deliberately left OUT so that L2 goes
    red -- silence here is exactly the drift this gate exists to catch."""
    names = [n for n in KNOWN_EXPORT_ORDER if n in extracted]
    names += [n for n in TEMPLATE_OPTIONALS if n in OPTIONAL and n not in names]
    exports = ",".join("%s=%s" % (n, placeholder(n)) for n in names)
    sbatch = "sbatch --export=ALL,%s %s %s" % (exports, ENTRY, ENTRY_ARGS)
    plain = "env %s bash %s %s" % (
        " ".join("%s=%s" % (n, placeholder(n)) for n in names), ENTRY, ENTRY_ARGS)
    return sbatch, plain


def classify(names):
    """Exactly one bucket per name; the four parts are asserted to sum to the whole."""
    buckets = {"estate": [], "run": [], "topology": [], "unclassified": []}
    for n in sorted(names):
        if n in ESTATE:
            buckets["estate"].append(n)
        elif n in RUN:
            buckets["run"].append(n)
        elif n in TOPOLOGY_EXACT or any(p in n for p in TOPOLOGY_PARTS):
            buckets["topology"].append(n)
        else:
            buckets["unclassified"].append(n)
    return buckets


def redact_text(s):
    return REDACT_RE.sub("<redacted>", s)


def _table_row(name, bucket, msg):
    msg = redact_text(msg or "(no literal message; refused by the req_env helper)")
    msg = msg.replace("\n", " ").replace("|", "\\|").strip()
    if len(msg) > 160:
        msg = msg[:157].rstrip() + "..."
    return '| `%s` | %s | "%s" |' % (name, bucket, msg)


def render_markdown(res, buckets, sbatch, plain):
    n_names = len(res.info)
    m_sites = res.i1 + res.i2 + res.i3
    bucket_of = {n: b for b, names in buckets.items() for n in names}
    ordered = [n for n in KNOWN_EXPORT_ORDER if n in res.info]
    ordered += sorted(set(res.info) - set(KNOWN_EXPORT_ORDER))

    L = []
    L.append("# LAUNCH -- the generated training command (Deliverable D)")
    L.append("")
    L.append(
        "This command trains a foundation-scale model on this allocation. Every variable "
        "below is required because the launcher or its container backend **refuses to "
        "guess it**: an empty or missing value dies with an exit-96-style refusal instead "
        'of silently defaulting -- "%s" is the deliberate marker phrase, and the refusal '
        "messages are quoted verbatim (redacted) in the table. Extraction found "
        "%d required names from %d refusal sites (I1 %d, I2 %d, I3 %d, %d unparsed). This "
        "file is regenerated and re-gated whenever those artifacts change; a name the code "
        "requires that is missing from the command below fails the gate and this file is "
        "not written." % (MARKER, n_names, m_sites, res.i1, res.i2, res.i3, res.u))
    L.append("")
    L.append("## Variables")
    L.append("")
    L.append("| Variable | Bucket | Why it is required |")
    L.append("|----------|--------|--------------------|")
    for name in ordered:
        L.append(_table_row(name, bucket_of[name], res.info[name]["message"]))
    L.append("")
    L.append("## The command")
    L.append("")
    L.append("Shape: `<foundation-scale-command> %s`, realised against Slurm as an "
             "`sbatch` invocation whose `--export` list carries every required variable "
             "(`ALL` additionally forwards the submitting shell):" % ENTRY_ARGS)
    L.append("")
    L.append("```bash")
    L.append(sbatch)
    L.append("```")
    L.append("")
    L.append("Equivalent plain-env form for the local / off-Slurm allocation -- exactly "
             "the same variables, same entry point, no scheduler:")
    L.append("")
    L.append("```bash")
    L.append(plain)
    L.append("```")
    L.append("")
    L.append("## What a new engineer changes per run")
    L.append("")
    L.append("Only the `run` bucket. Everything else is set once per site (estate) or is "
             "a property of the job shape (topology):")
    L.append("")
    for name in [n for n in KNOWN_EXPORT_ORDER if n in buckets["run"]] + \
                sorted(set(buckets["run"]) - set(KNOWN_EXPORT_ORDER)):
        hint = RUN_HINTS.get(name, "changes every invocation")
        L.append("- `%s=%s` -- %s" % (name, placeholder(name), hint))
    L.append("")
    L.append("---")
    L.append("_Generated by `gate_launch_contract.py` from `launch_fs_h100.fixed.sh` and "
             "`fs_container_backend.bound.sh`. This file is a generated artifact and must "
             "not be hand-edited; regenerate it instead. Gates L1-L6 pin it to the code, "
             "and any drift fails the build._")
    return "\n".join(L) + "\n"


def run_drills(launch_text, backend_text, res):
    """L6: every drill must be OBSERVED going red/green. All mutations happen on
    in-memory copies; LAUNCH and BACKEND on disk are never touched."""
    ok = True

    # MUST_FIRE 1: plant a req_env call; the extractor must list the name AND L2 must
    # go red because the planted name is absent from the known-template inventory.
    planted = Extraction()
    _parse_file(planted, launch_text + "\n# L6 drill copy -- never written to disk\n"
                                       "req_env FS_PLANTED_REQ\n", LAUNCH.name)
    _parse_file(planted, backend_text, BACKEND.name)
    ps, pp = build_template(set(planted.info))
    missing = sorted(set(planted.info) - assigned_names(ps + "\n" + pp))
    if "FS_PLANTED_REQ" in planted.info and "FS_PLANTED_REQ" in missing:
        print("  PASS L6  MUST_FIRE planted-required: FS_PLANTED_REQ extracted on the "
              "copy (%d names) and flagged missing from the template by L2"
              % len(planted.info))
    else:
        ok = False
        print("  FAIL L6  MUST_FIRE planted-required drill did not fire "
              "(extracted=%s, flagged=%s)"
              % ("FS_PLANTED_REQ" in planted.info, "FS_PLANTED_REQ" in missing))

    # MUST_FIRE 2: plant an extra assignment into the template; L3 must flag it as an
    # undeclared extra.
    s2, p2 = build_template(set(res.info))
    t2 = s2 + "\n" + p2 + "\nFS_PLANTED_EXTRA=1"
    undeclared = sorted(assigned_names(t2) - set(res.info) - set(OPTIONAL))
    if "FS_PLANTED_EXTRA" in undeclared:
        print("  PASS L6  MUST_FIRE planted-extra: FS_PLANTED_EXTRA flagged undeclared "
              "by L3 (%d undeclared of %d assignments on the copy)"
              % (len(undeclared), len(assigned_names(t2))))
    else:
        ok = False
        print("  FAIL L6  MUST_FIRE planted-extra drill did not fire")

    # MUST_PASS: the unmodified pair yields a non-empty required set (>= 8 names).
    if len(res.info) >= 8:
        print("  PASS L6  MUST_PASS unmodified pair yields %d required names (>= 8)"
              % len(res.info))
    else:
        ok = False
        print("  FAIL L6  MUST_PASS unmodified pair yielded only %d required names (< 8)"
              % len(res.info))
    return ok


def main():
    ok = True

    missing_files = [str(p) for p in (LAUNCH, BACKEND) if not p.is_file()]
    if missing_files:
        print("  FAIL L1  cannot read generated artifacts: " + ", ".join(missing_files))
        print("           run the bash generators first; there is no command to gate.")
        return 1

    launch_text = LAUNCH.read_text(encoding="utf-8")
    backend_text = BACKEND.read_text(encoding="utf-8")

    # --- L1: extraction with a denominator AND an unparsed count -------------------
    res = Extraction()
    _parse_file(res, launch_text, LAUNCH.name)
    _parse_file(res, backend_text, BACKEND.name)
    n_names = len(res.info)
    m_sites = res.i1 + res.i2 + res.i3
    if res.u:
        # Never claim completeness while sites matched no idiom.
        print("  PASS L1  %d required names (%d sites unparsed -- coverage is a floor, "
              "not a total)" % (n_names, res.u))
        print("           parsed: %d refusal sites (I1 %d, I2 %d, I3 %d)"
              % (m_sites, res.i1, res.i2, res.i3))
        for label, ln, txt in res.unparsed:
            print("           UNPARSED %s:%d: %s" % (label, ln, txt))
    else:
        print("  PASS L1  %d required names from %d refusal sites (I1 %d, I2 %d, I3 %d); "
              "0 sites matched no idiom" % (n_names, m_sites, res.i1, res.i2, res.i3))

    sbatch, plain = build_template(set(res.info))
    template_text = sbatch + "\n" + plain
    tmpl_names = assigned_names(template_text)

    # --- L2 FORWARD: every extracted name must appear in the command template -------
    missing_fwd = sorted(set(res.info) - tmpl_names)
    if missing_fwd:
        ok = False
        print("  FAIL L2  %d of %d extracted required names absent from the command "
              "template: %s" % (len(missing_fwd), n_names, ", ".join(missing_fwd)))
    else:
        print("  PASS L2  forward: all %d extracted required names appear in the "
              "command template (%d of %d)" % (n_names, n_names, n_names))

    # --- L3 REVERSE: every template assignment is required or declared-optional -----
    if not all(isinstance(r, str) and r.strip() for r in OPTIONAL.values()):
        ok = False
        print("  FAIL L3  OPTIONAL contains a waiver without a stated reason")
    undeclared = sorted(tmpl_names - set(res.info) - set(OPTIONAL))
    declared_opt = sorted(tmpl_names & set(OPTIONAL))
    if undeclared:
        ok = False
        print("  FAIL L3  %d template assignment(s) neither extracted-required nor "
              "declared-optional: %s" % (len(undeclared), ", ".join(undeclared)))
    else:
        print("  PASS L3  reverse: %d template assignments accounted for "
              "(%d required, %d optional-with-stated-reason, 0 undeclared)"
              % (len(tmpl_names), len(tmpl_names) - len(declared_opt), len(declared_opt)))

    # --- L4 CLASSIFY: disjoint buckets whose parts sum to the whole ----------------
    buckets = classify(set(res.info))
    total = sum(len(v) for v in buckets.values())
    flat = [n for v in buckets.values() for n in v]
    if total != n_names or len(set(flat)) != total:
        ok = False
        print("  FAIL L4  bucket partition broken: %d names classified, %d required, "
              "%d distinct" % (total, n_names, len(set(flat))))
    else:
        print("  PASS L4  buckets disjoint and complete: estate %d + run %d + topology "
              "%d + unclassified %d = %d required names"
              % (len(buckets["estate"]), len(buckets["run"]),
                 len(buckets["topology"]), len(buckets["unclassified"]), n_names))
    for b in ("estate", "run", "topology"):
        print("           %-12s %s" % (b, ", ".join(buckets[b]) or "(empty)"))
    if buckets["unclassified"]:
        # Visible, not absorbed: a newly required variable nobody bucketed yet.
        print("           UNCLASSIFIED (new required variable -- give it a bucket): "
              + ", ".join(buckets["unclassified"]))

    # --- L5 EMIT: refuse to write while any gate is red; redaction scan first ------
    if not ok:
        print("  FAIL L5  refusing to write %s while a gate is red" % OUT)
    else:
        md = render_markdown(res, buckets, sbatch, plain)
        md_lines = md.splitlines()
        hits = [i + 1 for i, l in enumerate(md_lines) if REDACT_RE.search(l)]
        if hits:
            ok = False
            print("  FAIL L5  redaction: %d of %d lines matched cluster-identifying "
                  "patterns (lines %s); refusing to write %s"
                  % (len(hits), len(md_lines), ", ".join(map(str, hits[:10])), OUT))
        else:
            OUT.write_text(md, encoding="utf-8")
            print("  PASS L5  wrote %s (%d lines; redaction: 0 of %d lines matched)"
                  % (OUT, len(md_lines), len(md_lines)))

    # --- L6 DRILLS -----------------------------------------------------------------
    if not run_drills(launch_text, backend_text, res):
        ok = False

    print("RESULT: " + ("ALL GATES GREEN" if ok else "GATES RED -- LAUNCH.md not trusted"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())