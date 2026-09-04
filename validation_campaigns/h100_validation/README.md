# h100_validation

## What this tree is

`h100_validation/` is a **build plane**, not a training run. It regenerates an H100
launch plane from upstream sources by running 24 ordered stages, then gates the
result. What lives here is the machinery that produces and certifies the launch
plane, plus the validation evidence gathered along the way. The plane itself is an
output of the build; the proof that the output is sound is the point of the
exercise.

Because this repository is public, the tree is also deliberately partitioned: the
build logic, gates, and evidence ship; the estate-specific inputs that the logic
*transforms* do not. That boundary is enforced, not merely documented — see below.

## The one build command

Run from the root of this tree, with your estate environment sourced first:

```bash
set -a; . /path/to/your/estate.env; set +a
bash build_h100_plane.sh
```

That is the entire interface. Source the environment, run the script, read the
exit code. If the build refuses, it tells you exactly what it could not find.

## What to set before running it — and why there are no defaults

Four knobs are **required with no default**. If any one is unset the build exits
96 (REFUSE) before doing any work. This is a stance, not an omission: for these
inputs, "nobody decided" must never masquerade as "clean". An unset knob would
either silently skip a transform or silently skip a scan, and both produce an
artifact that looks certified but is not. Where a knob genuinely has nothing to
say, you set it to `NONE`, meaning "declared empty". UNSET means undecided, and
undecided is UNMEASURED, not clean.

  * `FS_ESTATE_ROOT` — the filesystem root the launcher must stop hard-coding
    (#123). The build rewrites it out of the launcher; it must know what to
    rewrite.
  * `FS_PARTITION_LITERAL` — the Slurm partition name being **removed** from the
    launcher (#152). It also expands anchors in `h100/fix_113.json` and in three
    patch stages whose before-text names it (#157). Accepts `NONE`.
  * `FS_ESTATE_IDENT_PAT` — a `|`-separated regex alternation of this estate's
    identifiers: node names, account ids, org segments, private hostnames,
    management IPs. It is an *input* precisely because a redaction list checked
    into a public repo would publish the exact estate it was written to protect
    (#155). Accepts `NONE`.
  * `FS_REDACT_EXTRA` — `|`-separated literals the pattern cannot safely express.
    Example: a bare account id as `[0-9]{5}` would also match line counts.
    Accepts `NONE`.

One deliberate pair to not collapse: `FS_PARTITION_LITERAL` is the build input
being deleted; `FS_PARTITION` is the operator-side runtime knob that replaces it,
read by the *generated* launcher at launch time. They are different things at
different times on purpose. See `estate.env.example` for the full annotated list,
and `h100/LAUNCH.md` for the runtime knobs.

## What is deliberately absent, and how to supply it

Two build-time inputs are not in the repository, and cannot be:

  * **the estate's own launcher snapshot** — supply via
    `--upstream-launcher PATH`, or `$FS_UPSTREAM_H100_LAUNCHER`, or
    `$FS_UPSTREAM_DIR`, or drop it in `h100/upstream/`.
  * **the framework repo** (for the backend splice) — supply via `--repo PATH`,
    or `$FS_UPSTREAM_REPO`, or rely on it being an ancestor of `--root` (which is
    automatic when this plane lives inside the framework repo).

Both resolvers try candidates most-explicit-first, accept a candidate only if the
file it actually needs is really there, and **name every candidate tried** in the
refusal message. Neither resolver will silently fall back to a previously
generated artifact: a build that cannot find its input must refuse, because
otherwise it would certify a stale output as freshly produced (#136).

The reason these inputs are absent is that they are the estate's own before-text,
and one of them names an organisational filesystem root four times. The details,
and the mechanics of supplying them, are in `h100/upstream/README.md`.

## How to read the exit codes

The whole tree uses one contract, gates included:

  * **0 — PASS.** The stage or gate verified what it claims to verify.
  * **5 — RED.** A gate found a real defect. Read the output; do not re-run and
    hope.
  * **95 — UNMEASURED.** The check could not run at all. This is explicitly *not*
    a pass. (Example: the pytest-suite gate needs an interpreter with pytest and
    declares 95 rather than skipping if `$FS_PYTEST` and `./.venv/bin/python`
    both fail to resolve.)
  * **96 — REFUSE.** A required input is unset. There is no default, by design.

The build derives its scan denominator from `h100/PUBLISH_SET.txt` — every file
listed there, with no count restated here, because a count in prose is a second
oracle for the set's size and it was already stale once. A coverage control
refuses UNMEASURED if any published file falls
outside every scan category. That control exists because the publish set and the
scan denominator used to be different sets derived by different rules, and
nothing compared them (#157).

## Why the two scanners stay here and are not published (#248)

`prepush_gate.py` and `gate133_estate.py` scan for estate leaks before a push.
The obvious objection is that they therefore belong in the published repo's
`checks/`, where CI would run them on every commit instead of only when a
developer remembers. They stay here anyway, and the reason is the same doctrine
the rest of this tree enforces.

**Their vocabulary is the thing they must never publish.** The fatal patterns are
built at import time from `FS_ESTATE_IDENT_PAT`, `FS_PARTITION_LITERAL`,
`FS_REDACT_EXTRA` and `FS_ESTATE_ROOT`, sourced from the operator's own estate
file (mode 600, outside any repo). A published copy would run in CI with none of
those set. It
would then do one of two things, and both are worse than not running:

  * **Refuse (96) on every commit**, because the inputs are required with no
    default — a gate that is red on the tree it guards, which is exactly the
    defect #239 was about; or
  * **Degrade to the literals it can compile in** and print `PASS`, having
    scanned for an empty estate vocabulary. That is a vacuous pass: `all([])` is
    True, the denominator is zero, and the banner says clean because nothing ever
    told it what a leak looks like. Publishing the scanner without its vocabulary
    manufactures precisely the false confidence this repo exists to eliminate.

Hard-coding the vocabulary to escape that dilemma is self-defeating: the leak
detector would become the leak.

**Their denominator is also wrong for CI.** `prepush_gate.py` scans the *push
set* — `git diff --stat origin/main..HEAD` plus the working tree — because the
question it answers is "is it safe to publish *this*". CI runs post-push, over a
tree from which any estate literal has, by then, already been published. The
instrument is pre-publication by construction; moving it after the event does not
make it a weaker check, it makes it a check of a different and useless claim.

Consequences accepted, and the mitigations:

  * The gate runs only when invoked. It is wired into the documented push
    procedure, not into CI, and that is a human-discipline dependency with no
    automated backstop. Accepted knowingly.
  * `prepush_gate.py` keeps its own historical exit codes — **0 PASS, 1 LEAK,
    2 GATE ERROR, 96 REFUSE** — rather than the 0/5/95/96 contract above. It has
    no programmatic callers (`sync_to_repo.py` only names it in a print), so
    nothing decodes those numbers but a human. Renumbering it to the tree
    contract would be churn with a migration risk and no reader.
  * What *can* be published safely is published: `h100_validation/` carries the
    use-vs-mention controls and the token-tier patterns that carry no estate
    identifier. The split is deliberate — public checks get the vocabulary that
    is safe in public, and only that.

## Where the deliverables are

  * `h100/LAUNCH.md` — the operator-facing launch document (Deliverable D).
  * `h100/DELIVERABLE_B_validation_report.md` — the H100 validation report.
  * `h100/DELIVERABLE_E_matrix.md` — the compatibility matrix.
  * `h100/EVIDENCE.md` — the measurement log every claim cites.
  * `h100/upstream/README.md` — what the absent inputs are and how to supply
    them.

## What is not yet validated

Read this before citing anything in this tree as proof of a working system:

  * **Phase 3, the multi-node 8xH100 run, has NOT been executed.**
  * **Phase 4, the Gemma+Qwen cross-model validation, has NOT been executed.**

Cluster access is pending. Nothing here demonstrates — or claims to demonstrate —
that the framework has been proven end-to-end on hardware. The evidence in this
tree covers what the build and its gates can measure today; the hardware phases
above are future work, and any statement stronger than that is not supported by
`h100/EVIDENCE.md`.
