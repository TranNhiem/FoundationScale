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
  Nothing, by default. The rendered template is a BYPRODUCT; the verdict is the exit
  code. `--emit PATH` writes the render, and the caller names the path, so no default
  location can silently accrete an undeclared artifact -- which is what happened when
  this gate wrote h100/gen/LAUNCH.md unconditionally: that directory is asserted by
  gate_build_inputs.py to hold EXACTLY the declared PRODUCED set, and every member of
  that set also ships in PUBLISH_SET.txt, so the write forced a choice between a red
  input-partition gate and a second published statement of the required-knob countable
  (#194/#220/#233/#266). The render and the redaction scan still run on EVERY invocation
  regardless of --emit; only the write is conditional. This repo is PUBLIC.

WHERE THE REQUIRED NAMES COME FROM (#276)
  The authoritative answer to "which variables does the launch plane refuse to start
  without?" is fs_required_knobs.extract(), shared with gate_launch_doc.py. The two
  gates answer the same question over the SAME two bash artifacts and then do opposite
  things with the answer -- this one GENERATES a command from it, that one AUDITS the
  hand-written h100/LAUNCH.md against it. (The operator document this gate must not
  contradict is h100/LAUNCH.md; the template rendered here reaches disk only under
  --emit, and is a byproduct -- see WRITES.) Two readers of one artifact holding
  two different required sets is a drift neither can see from inside itself, because
  each is internally consistent.

  The three local idioms below are NOT retired -- they still run, but as a SITE scan,
  not as the name source. They answer a different question: how much of each artifact
  did any reader understand, and how much matched nothing (the U floor). Keeping both
  is what makes the disagreement printable: every name the site scan mints that the
  census refuses is reported with the census rule that refused it.

  This split is load-bearing rather than tidy. The I3 rule reads the first ALL-CAPS
  token out of a refusal MESSAGE, so on launch_fs_h100.fixed.sh:39 it mints a variable
  called REFUSE -- a word from the prose, not a knob any operator can set. Nothing in
  STOPLIST covered it. Before this change that phantom was in the required set, one
  wiring commit away from being published into the operator-facing table.

SCOPE LIMITS
  * Exactly three refusal idioms are understood by the SITE scan (I1 req_env calls, I2
    inline emptiness refusals, I3 the marker phrase "required, no default by design").
    Lines that look like refusals but match no idiom are counted as UNPARSED and
    printed; site coverage is reported as a floor, never silently claimed complete.
  * The two bash artifacts are never modified; the L6 drills run on in-memory copies.
  * stdlib only.

EXIT CODES -- the plane's four-state contract (0 clean / 5 red / 95 unmeasured / 96
refused). Until #278 this gate returned 0 or 1, so REFUSE, RED and a dead control were
one indistinguishable code, and a consumer could not tell "the artifact is wrong" from
"the detector never ran". Only states this gate can actually reach are listed; declaring
an unreachable code is the #198/#200 defect in the other direction.
  0   every gate green and every control fired.
  5   a substantive gate is red: a required name is missing from the command template
      (L2), a template assignment is neither required nor waived (L3), a required name
      lands in no bucket (L4), or the rendered document failed the redaction scan (L5).
  95  a MUST_FIRE or MUST_PASS control did not fire (L6). The detector is uncertified,
      so any verdict it produced is unattributable -- UNMEASURED, not clean.
  96  refused before any measurement: an unrecognised argument or a bare --emit with no
      path; the generated artifacts are unreadable (run the bash generators first); or
      -- raised at import by fs_estate_pat -- the estate's redaction vocabulary is unset,
      which cannot certify a document against a secret it was never given.
"""

import os
import re
import sys
from pathlib import Path

import fs_required_knobs
from fs_estate_pat import estate_ident_pat

ROOT = Path(__file__).resolve().parent
LAUNCH = ROOT / "h100/gen/launch_fs_h100.fixed.sh"
BACKEND = ROOT / "h100/gen/fs_container_backend.bound.sh"
# There is deliberately no OUT constant. A module-level default output path is what made
# the write unconditional, and h100/gen/ is a DECLARED artifact directory, not scratch
# (#278). The render's destination now comes from --emit or does not exist.

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
# 'unclassified', which is FATAL (#127).
#
# It used to be merely PRINTED, under a PASS line reading "buckets disjoint and
# complete". Both halves of that sentence cannot be true at once: a residue nobody
# bucketed is exactly the case where the partition is NOT complete, and the gate went
# on to write the operator document anyway. The unclassified name would have rendered
# with the generic <fs-whatever> placeholder and no statement of where an operator is
# supposed to get its value -- which is the one question the buckets exist to answer.
# A visible line in a green report is not a gate; it is a note, and notes do not stop
# a document from shipping.
#
# The RED is also the only thing that makes the bucket sets maintainable. A name
# reaching unclassified means the launcher grew a requirement, and the correct response
# is a one-line decision about which bucket it belongs to -- routine, provided someone
# is made to make it. Absorbing it costs nothing at the moment it happens and produces
# an operator document that quietly under-specifies the launch.
#
# The estate/topology line is "does it change how many ranks run?" FS_GPUS_PER_NODE
# does, so it is topology. FS_PARTITION, FS_WALLTIME, FS_CPUS_PER_TASK and FS_MEM do
# not -- they say which queue and what allocation shape, which is an estate fact the
# operator inherits from their site. FS_FABRIC_TRIPWIRE names one estate's IMEX master
# (#163), so it is estate too. Those five arrived in the required set when this gate
# adopted the shared census (#276): a fifth bucket would have been the easy move and
# the wrong one, because the buckets exist to tell an operator which knobs they must
# get from their cluster admin, not to record which reader found them.
ESTATE = {"FS_ALLOWED_NODE", "FS_ALLOWED_PATH_ROOTS", "FS_CONTAINER_RUNTIME",
          "FS_ALLOCATION", "FS_CONTAINER_SQSH", "IMAGE",
          "FS_PARTITION", "FS_WALLTIME", "FS_CPUS_PER_TASK", "FS_MEM",
          "FS_FABRIC_TRIPWIRE"}
RUN = {"MODEL_DIR", "DATASET_DIR", "CONFIG_FILE", "OUT_DIR_STABLE", "PROBE", "FS_PHASE"}
TOPOLOGY_EXACT = {"FS_GPUS_PER_NODE", "FS_ENGINE_LAUNCH_MODE"}
TOPOLOGY_PARTS = ("PROCS", "NTASKS")

# The names this generator knows how to place into the command template, in display
# order. An extracted name that is NOT here fails L2 loudly instead of being silently
# dropped -- that failure is the drift alarm, and drill L6/MUST_FIRE relies on it.
#
# This is an ORDERING HINT, not a claim about requiredness: build_template() keeps only
# the entries that are actually in the census (`if n in res.info`), so an entry that
# stops being required costs nothing and is not a drift. FS_CONTAINER_SQSH and
# FS_ENGINE_LAUNCH_MODE are exactly that -- the shared census files them R5 (produced
# in-artifact) and R3 (conditional), so they no longer render. They stay because
# deleting them would throw away where they belong if they ever come back, and because
# a reader comparing this list to the table should see the same names in the same
# order either way.
KNOWN_EXPORT_ORDER = [
    "MODEL_DIR", "DATASET_DIR", "CONFIG_FILE", "OUT_DIR_STABLE",
    "IMAGE", "FS_GPUS_PER_NODE", "PROBE",
    "FS_ALLOWED_NODE", "FS_ALLOWED_PATH_ROOTS", "FS_CONTAINER_RUNTIME",
    "FS_ALLOCATION", "FS_CONTAINER_SQSH", "FS_ENGINE_LAUNCH_MODE",
    "FS_PARTITION", "FS_WALLTIME", "FS_CPUS_PER_TASK", "FS_MEM",
    "FS_FABRIC_TRIPWIRE",
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
    # Estate knobs the operator gets from their cluster admin. The generic fallback
    # would render <fs-partition>, which reads like a value; these say what SHAPE of
    # value the scheduler wants. No estate literal may appear here -- the rendered
    # document is scanned by REDACT_RE, and a real partition name would fail L5.
    "FS_PARTITION": "<partition>",
    "FS_WALLTIME": "<d-hh:mm:ss>",
    "FS_CPUS_PER_TASK": "<cpus-per-task>",
    "FS_MEM": "<mem-per-node>",
    "FS_FABRIC_TRIPWIRE": "<imex-master-host>",
}


def placeholder(name):
    """Operator-facing value hint for the template; unknown names get <their-own-name>."""
    return PLACEHOLDERS.get(name, "<" + name.lower().replace("_", "-") + ">")


class Extraction:
    """Result of one pass over the two artifacts: names, per-idiom site counts, and the
    unparsed denominator (U) that keeps the coverage claim honest.

    After _apply_census() runs, `info` holds the SHARED census's required set and the
    three disagreement lists below hold the difference against the local site scan.
    `i1/i2/i3/u/unparsed` are never rewritten -- they stay the site scan's own report,
    because the U floor is a statement about what the artifacts look like, not about
    which names are required."""

    def __init__(self):
        self.info = {}       # name -> {"idiom", "message", "where"}
        self.i1 = self.i2 = self.i3 = self.u = 0
        self.unparsed = []   # (file label, line number, truncated line text)
        self.excluded = {}   # name -> (rule, why) for every name the census refused
        self.site_only = []  # site scan minted it, census refused it (REFUSE lives here)
        self.census_only = []  # census requires it, the three local idioms cannot see it


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


def _apply_census(res, sources):
    """Replace the site scan's name set with the SHARED census, and record the diff.

    `sources` is {label: text}, the same mapping the site scan was fed, so the drills
    can run this over a mutated in-memory copy exactly as main() runs it over the real
    artifacts. Anything else would leave the drills testing dead code.

    The site scan's counters survive untouched -- see Extraction's docstring. What
    changes is only WHICH NAMES are called required, and every difference in either
    direction is recorded so the next reader can see it rather than infer it."""
    census = fs_required_knobs.extract(sources)

    site_names = set(res.info)
    res.excluded = dict(census.excluded)
    res.site_only = sorted(site_names - set(census.required))
    res.census_only = sorted(set(census.required) - site_names)

    res.info = {}
    for name, knob in census.required.items():
        res.info[name] = {
            "idiom": knob.rule,
            "message": knob.message or "refused when unset",
            "where": knob.site,
        }
    return census


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


def l4_verdict(names):
    """L4's ENTIRE decision, returned rather than printed, so that MUST_FIRE 4 can run
    the code the gate runs instead of restating its condition. A drill that re-derives
    `buckets["unclassified"] != []` for itself proves that classify() works and says
    nothing about whether the gate acts on it -- which is precisely the thing #127 got
    wrong for the whole life of this file.

    Returns (ok, buckets, lines). main() prints the lines and ANDs the ok.
    """
    buckets = classify(set(names))
    n_names = len(set(names))
    total = sum(len(v) for v in buckets.values())
    flat = [n for v in buckets.values() for n in v]
    lines, ok = [], True
    if total != n_names or len(set(flat)) != total:
        ok = False
        lines.append("  FAIL L4  bucket partition broken: %d names classified, %d "
                     "required, %d distinct" % (total, n_names, len(set(flat))))
    elif buckets["unclassified"]:
        # #127. Fatal, not a note. The partition sums correctly here -- that is what
        # the branch above checked -- so the only thing wrong is that some of the parts
        # mean "unbucketed", and a sum that includes an I-don't-know term is not a
        # classification. L5 reads `ok` and will now refuse to write the document.
        ok = False
        lines.append("  FAIL L4  %d required name(s) in no bucket -- estate %d + run %d "
                     "+ topology %d = %d of %d, and a name with no bucket renders in "
                     "the operator document with no statement of where its value comes "
                     "from: %s"
                     % (len(buckets["unclassified"]), len(buckets["estate"]),
                        len(buckets["run"]), len(buckets["topology"]),
                        n_names - len(buckets["unclassified"]), n_names,
                        ", ".join(buckets["unclassified"])))
    else:
        lines.append("  PASS L4  buckets disjoint and complete: estate %d + run %d + "
                     "topology %d + unclassified %d = %d required names"
                     % (len(buckets["estate"]), len(buckets["run"]),
                        len(buckets["topology"]), len(buckets["unclassified"]),
                        n_names))
    for b in ("estate", "run", "topology"):
        lines.append("           %-12s %s" % (b, ", ".join(buckets[b]) or "(empty)"))
    return ok, buckets, lines


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
    # The drill runs the SAME two passes main() runs -- site scan then _apply_census --
    # because after #276 it is the census that decides the name set. A drill that only
    # ran the site scan would be exercising a path no longer on the critical route.
    mutated = (launch_text + "\n# L6 drill copy -- never written to disk\n"
                             "req_env FS_PLANTED_REQ\n")
    planted = Extraction()
    _parse_file(planted, mutated, LAUNCH.name)
    _parse_file(planted, backend_text, BACKEND.name)
    _apply_census(planted, {LAUNCH.name: mutated, BACKEND.name: backend_text})
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

    # MUST_FIRE 3: the anti-phantom drill, and the reason #276 is worth doing. I3
    # attributes a refusal to the first ALL-CAPS token inside the refusal MESSAGE, so a
    # message opening with a capitalised prose word mints that word as a variable. On
    # the real launcher that word is REFUSE, and before the census it was in the
    # required set -- one wiring commit from being published to operators.
    #
    # The fixture is standalone rather than appended to launch_text, and that is a
    # measured decision, not tidiness: QUOTE_RE pairs quotes sequentially from offset 0
    # over the WHOLE file, and the launcher currently holds an ODD number of them, so an
    # appended line pairs off-by-one, falls back to hay=whole-line and mints the
    # VARIABLE instead of the prose word. The appended version of this drill therefore
    # reported "no phantom" while the real artifact had one -- a control whose result
    # depended on the parity of quote characters in unrelated lines above it. (No count
    # is quoted here on purpose: it would be a number in no drift denominator.)
    #
    # Both halves are asserted. Checking only that the census is clean would still read
    # PASS if the site scan had quietly stopped minting anything at all.
    fixture = ('[[ -n "${FS_PLANTED_PHANTOM:-}" ]] || { echo "DENY 96: '
               'FS_PLANTED_PHANTOM is unset (' + MARKER + ')." >&2; exit 96; }\n')
    fx_label = "l6_phantom_fixture.sh"
    ph_site = Extraction()
    _parse_file(ph_site, fixture, fx_label)
    minted_by_site = "DENY" in ph_site.info
    ph_census = Extraction()
    _parse_file(ph_census, fixture, fx_label)
    _apply_census(ph_census, {fx_label: fixture})
    refused = "DENY" not in ph_census.info and "FS_PLANTED_PHANTOM" in ph_census.info
    if minted_by_site and refused:
        # The real pair is reported too, but as an observation: a phantom is a name the
        # site scan minted that the census has no RULE for, because it was never a guard
        # site at all. Zero here would be good news, not a failure.
        live = [n for n in res.site_only if n not in res.excluded]
        print("  PASS L6  MUST_FIRE anti-phantom: the site idiom mints DENY out of the "
              "refusal message (control is live) and the census refuses it while "
              "keeping FS_PLANTED_PHANTOM; on the real pair %d such phantom(s): %s"
              % (len(live), ", ".join(live) or "none"))
    else:
        ok = False
        print("  FAIL L6  MUST_FIRE anti-phantom drill did not fire "
              "(site minted DENY=%s, census clean=%s) -- a dead control here means "
              "UNMEASURED, not clean" % (minted_by_site, refused))

    # MUST_FIRE 4: plant a required name that belongs to no bucket; L4 must go RED and
    # therefore L5 must refuse to write. This is #127's other half. Until it was added,
    # the unclassified state was PRINTED under a PASS line reading "buckets disjoint and
    # complete" -- a note, and notes do not stop a document from shipping.
    #
    # The drill calls l4_verdict() rather than re-deriving `unclassified != []` for
    # itself, which is the whole reason that function exists. A drill that restates the
    # condition proves classify() works and says nothing about whether the gate ACTS on
    # it, and "the gate did not act on it" is the defect being closed here.
    #
    # Both directions are asserted. Planting alone is not a control: if L4 returned
    # False unconditionally the planted half would still read PASS, so the real set is
    # run through the same function and must come back ok with an EMPTY unclassified.
    # And the planted name must land in unclassified SPECIFICALLY -- if it were absorbed
    # by the topology substring rule (TOPOLOGY_PARTS is a substring match, so a name can
    # be bucketed by accident) the equality below fails and this drill says so.
    nb_ok, nb_buckets, _ = l4_verdict(set(res.info) | {"FS_PLANTED_NOBUCKET"})
    real_ok, real_buckets, _ = l4_verdict(set(res.info))
    fired = not nb_ok and nb_buckets["unclassified"] == ["FS_PLANTED_NOBUCKET"]
    discriminates = real_ok and not real_buckets["unclassified"]
    if fired and discriminates:
        print("  PASS L6  MUST_FIRE unbucketed-required: FS_PLANTED_NOBUCKET lands in "
              "'unclassified' and L4 returns RED (so L5 refuses to write), while the "
              "real %d-name set returns GREEN with 0 unclassified" % len(res.info))
    else:
        ok = False
        print("  FAIL L6  MUST_FIRE unbucketed-required drill did not fire "
              "(planted RED=%s, planted unclassified=%s, real GREEN=%s) -- a dead "
              "control here means UNMEASURED, not clean"
              % (not nb_ok, nb_buckets["unclassified"] or "none", discriminates))

    # MUST_PASS: the unmodified pair yields a non-empty required set (>= 8 names).
    if len(res.info) >= 8:
        print("  PASS L6  MUST_PASS unmodified pair yields %d required names (>= 8)"
              % len(res.info))
    else:
        ok = False
        print("  FAIL L6  MUST_PASS unmodified pair yielded only %d required names (< 8)"
              % len(res.info))
    return ok


def main(argv=None):
    ok = True

    # --emit is the ONLY way the rendered template reaches disk, and the caller names the
    # path. Parsed by hand rather than through argparse to keep this gate importable and
    # runnable under the 3.6.8 login-node interpreter the plane targets (#138), which is
    # the same reason every other gate here avoids the newer argparse conveniences.
    argv = list(sys.argv[1:] if argv is None else argv)
    emit_path = None
    while argv:
        arg = argv.pop(0)
        if arg == "--emit":
            if not argv:
                print("  REFUSE  --emit requires a path argument")
                return 96
            emit_path = Path(argv.pop(0))
        elif arg.startswith("--emit="):
            emit_path = Path(arg.split("=", 1)[1])
        else:
            print("  REFUSE  unrecognised argument: %s (accepts --emit PATH)" % arg)
            return 96

    missing_files = [str(p) for p in (LAUNCH, BACKEND) if not p.is_file()]
    if missing_files:
        print("  REFUSE L1  cannot read generated artifacts: " + ", ".join(missing_files))
        print("           run the bash generators first; there is no command to gate.")
        # 96, not 5. Nothing was measured, so there is no finding -- and a build that
        # cannot tell "the inputs are absent" from "the command template is wrong" will
        # go looking for a defect that does not exist. #278.
        return 96

    launch_text = LAUNCH.read_text(encoding="utf-8")
    backend_text = BACKEND.read_text(encoding="utf-8")

    # --- L1: extraction with a denominator AND an unparsed count -------------------
    # Two passes over the same bytes, on purpose. The site scan measures how much of
    # the artifacts any reader here understands (I1/I2/I3 and the U floor); the shared
    # census decides which names are actually required. #276.
    sources = {LAUNCH.name: launch_text, BACKEND.name: backend_text}
    res = Extraction()
    _parse_file(res, launch_text, LAUNCH.name)
    _parse_file(res, backend_text, BACKEND.name)
    census = _apply_census(res, sources)
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

    # The two readers disagreeing is the interesting output, so print it rather than
    # letting the census quietly win. Each site-only name carries the census rule that
    # refused it; a name with no rule was never a guard site at all (that is the
    # phantom shape -- a word read out of a refusal MESSAGE).
    print("           census: %d required, %d excluded, %d sites seen by the shared "
          "extractor" % (len(census.required), len(census.excluded), census.sites_seen))
    for name in res.site_only:
        rule = res.excluded.get(name)
        why = "%s: %s" % rule if rule else "not a guard site -- read out of message text"
        print("           SITE-ONLY %-22s %s" % (name, why))
    for name in res.census_only:
        print("           CENSUS-ONLY %-20s %s (idiom the three local rules cannot see)"
              % (name, res.info[name]["where"]))

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
    l4_ok, buckets, l4_lines = l4_verdict(set(res.info))
    if not l4_ok:
        ok = False
    for line in l4_lines:
        print(line)

    # --- L6 DRILLS: run BEFORE the emit, because a dead control is not a licence to ---
    # ship. Until #278 the drills ran after L5, so the document could be written while
    # the control layer certifying this gate's own detectors was red -- an artifact
    # emitted under an uncertified detector carries a claim nothing checked. Ordering
    # them ahead of the write makes them a precondition of it rather than a postscript.
    # The report therefore prints L6 before L5; that is deliberate, not a numbering slip.
    drills_ok = run_drills(launch_text, backend_text, res)

    # --- L5 RENDER + REDACTION SCAN: always in memory; write only if asked ---------
    # #278 moved the write behind --emit and made it default OFF. It used to land
    # unconditionally in h100/gen/, which is not a scratch directory: gate_build_inputs.py
    # I1 asserts that h100/gen/ holds EXACTLY the declared PRODUCED set, and every member
    # of that set is also in PUBLISH_SET.txt. So an unconditional write forced a choice
    # between two bad states -- leave it undeclared and turn the build's own input/output
    # partition gate red (measured: "10 present, 9 declared; UNDECLARED: ['LAUNCH.md']"),
    # or declare it and thereby queue a SECOND shipped statement of the 16-required-knobs
    # countable that h100/LAUNCH.md already makes, which is the drift class behind #194,
    # #220, #233 and #266.
    #
    # Neither was necessary, because the document had zero consumers: nothing reads it, no
    # stage depends on it, and its only tracked sibling relationship was to files that do
    # ship. What this gate is FOR is the verdict -- forward and reverse accounting of the
    # launch command template -- and the verdict travels in the exit code. So the render
    # still happens on every run and the redaction scan still runs over it (the scan is a
    # real gate and must not become conditional on a flag nobody passes); only the write
    # is opt-in. Passing --emit is how a human gets the rendered template on demand, and
    # the caller names the path, so no default location can silently accrete an artifact.
    if not ok or not drills_ok:
        print("  FAIL L5  refusing to render the template while a gate is red or a "
              "control is dead")
    else:
        md = render_markdown(res, buckets, sbatch, plain)
        md_lines = md.splitlines()
        hits = [i + 1 for i, l in enumerate(md_lines) if REDACT_RE.search(l)]
        if hits:
            ok = False
            print("  FAIL L5  redaction: %d of %d lines matched cluster-identifying "
                  "patterns (lines %s); not emitting"
                  % (len(hits), len(md_lines), ", ".join(map(str, hits[:10]))))
        else:
            where = "not written (byproduct; pass --emit PATH to write it)"
            if emit_path is not None:
                emit_path.write_text(md, encoding="utf-8")
                where = "wrote %s" % emit_path
            print("  PASS L5  rendered %d lines, redaction 0 of %d matched; %s"
                  % (len(md_lines), len(md_lines), where))

    # The verdict order is deliberate: a dead control OUTRANKS a red gate. A RED produced
    # by a detector whose own controls did not fire is unattributable -- it may be the
    # artifact, it may be the detector -- and reporting RED would claim a measurement
    # that was not made. UNMEASURED is the honest state and is equally blocking, so
    # nothing ships either way; the difference is only in what the operator goes to fix.
    if not drills_ok:
        print("RESULT: CONTROLS DEAD -- UNMEASURED, not clean"
              + ("" if ok else "; L1-L5 also reported red, unattributably"))
        return 95
    if not ok:
        print("RESULT: GATES RED -- LAUNCH.md not trusted")
        return 5
    print("RESULT: ALL GATES GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())