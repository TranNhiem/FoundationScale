# Verification: the audit, and why the gates look like this

*This document was the front page of FoundationScale until the README became a framework
README. It is the rationale for the whole gate plane, kept intact and moved here rather than
condensed, because it is the argument the code exists to satisfy.*

*Numbers in section 3 were re-measured when this document moved; see `docs/review/` for the
finding (#220) that the front page's countables had drifted.*

---

This repository contains two things, in the order that produced them: an audit of a
distributed-training estate, written as an evidence chain and published in full; and a
verification framework built to make the audit's central failure impossible to repeat,
including inside the framework itself.

## 1. The audit

FoundationScale is built from a forensic audit of a real estate rather than from a
blank page: two production training codebases (published here under the pseudonyms
`omni-accel` and `omni-bridge`), 1,774 Slurm jobs, 205 export artefacts, ten target
training stages, MoE and dense model families across H100 and GB200. The full
evidence chain is in [`docs/deliverables/`](deliverables/README.md), published
with cluster-internal identifiers replaced by stable pseudonyms and nothing else
altered — the failure record, the numbers and the retractions are as written.

The finding that shapes the design:

> **The dominant failure mode in large-scale training is not a crash. It is a run that
> reports success.**

Two items from the record carry the argument:

* A checkpoint holding **5.71 GB where 45.70 GB was correct** passed every check in
  front of it — exit code, resume, tensor counts, dtypes, healthy loss — because
  every property anyone measured was genuinely correct. The only thing wrong was the
  content.
* The tool written to *detect* that corruption reported
  **`expert_perm_all_identity: true`** on a known-corrupt artefact, because the expert
  tensors it meant to compare were absent, the comparison set was empty, and
  **`all([])` is `True`** in Python. The engineer who wrote the detector for that bug
  wrote the bug into the detector. The same class of error had gone undetected for
  months, four separate times, in the audited systems; a negative control caught the
  tool's own copy of it in minutes.

Nor was this confined to checkpointing. One run logged `grad_norm` exactly 0.000 for
472 steps while reporting `reward/mean=0.794, success=1.00`. Five of the ten target
stages have never executed at all. Every check that existed asked *"did anything go
wrong?"* — a question a broken system answers "no" just as fluently as a working one.

The analysis is written as seven deliverables plus a decision log, graded claim by
claim — **[M]** measured, **[V]** verified by source inspection, **[A]** artefact
census, **[K]** asserted by a model and not independently confirmed, **[U]**
explicitly unverified. Start with the reasoning spine,
[`docs/DECISIONS.md`](DECISIONS.md), or jump to the
[deliverables index](deliverables/README.md):

| | Document | What it answers |
|---|---|---|
| **A1** | [Existing system — structure](deliverables/A1_existing_system.md) | Two repos, workflows end-to-end (SFT / RL / export), infrastructure, dependency map |
| **A2** | [Existing system — algorithms](deliverables/A2_algorithms.md) | Every objective, the reward cascade, pipeline coverage matrix |
| **A3** | [Existing system — debt & risk](deliverables/A3_debt.md) | Duplication census, the **silent-failure catalogue**, blast-radius accounting, risk register |
| **B1** | [Architecture — core](deliverables/B1_architecture.md) | The L0–L6 layered design, gate catalogue, run manifest, falsifiable claims |
| **B2** | [Architecture — unified LLM/VLM + scaling](deliverables/B2_scaling.md) | One data contract, one Stage abstraction, 1 GPU → 100+ nodes as a ladder with measured rungs |
| **C** | [Codebase mapping](deliverables/C_mapping.md) | Per-component Reusable / Refactor / Wrap / Replace / Build-New |
| **D** | [Development roadmap](deliverables/D_roadmap.md) | Phases 1–10, each with exit criteria and a **falsification condition** |

The documents apply two rules to themselves. First: **an unqualified count is not a
fact**. Second: **every claim that something does not exist must name the positive
control proving its detector could have fired.** The second rule exists because the
audit broke it four times and was wrong each time — and the second rule is what the
framework below is, once you make it code.

## 2. The framework: a verdict is a claim about coverage

The central abstraction is not a trainer or a parallelism strategy. It is a **gate** —
a correctness check that runs at a defined point in a job's lifecycle (launch, build,
data, first optimizer step, first save, every save, export, promotion) and can block.
What separates a gate from an assertion is that its verdict includes its coverage:

```python
class Verdict(str, Enum):
    PASS          # examined N units, all correct
    FAIL          # examined N units, found a defect
    VACUOUS       # examined ZERO units, and therefore proved nothing — blocks
    UNDERCOVERED  # examined fewer than expected, without declaring a sample — blocks
    SKIP          # explicitly not applicable; reason required; reported; non-blocking
    ERROR         # the gate itself raised — gates fail closed; blocks
```

**VACUOUS and UNDERCOVERED are the design, not edge cases.** There are two ways to
not block and four ways to block, and the asymmetry is deliberate: it is far easier
to produce a meaningless success than a meaningless failure. A gate reporting "3
layers checked" out of 205 is UNDERCOVERED unless it declares itself a sample, with
the reason; a gate reporting "all experts match" over an empty comparison set is
VACUOUS. Both block.

The rule is enforced in the base class
([`src/foundationscale/gates/core.py`](../src/foundationscale/gates/core.py)), not in
the gate author's discipline: `Gate.ok()` cannot return `PASS` on zero coverage, no
matter what the author writes — it downgrades. An author who means "nothing to check
here" must say so via `skip()` with a reason, which is recorded and surfaced.

The reference gate — `checkpoint.expert_alias`, in
[`src/foundationscale/gates/example.py`](../src/foundationscale/gates/example.py) —
encodes the estate's defining incident, in which 128 experts were saved under local
names as 16 experts replicated eight times, and two full training runs executed on a
model that was 87.5% wrong. Its `check()` contains no special case for an empty
expert set. Deliberately:

```python
        # 3. The empty case is NOT special-cased. self.ok with zero coverage is
        #    downgraded to VACUOUS by the contract — that downgrade is the fix for
        #    the `all([]) is True` verification tool, and the `empty-expert-set`
        #    control below exists to prove nobody "helpfully" bypasses it here.
        if not by_expert:
            return self.ok(
                f"checkpoint exposes {len(names)} expert tensors total but none with "
                f"a resolvable global expert index",
                coverage,
            )
```

And here is that code path running, over a checkpoint whose expert set is entirely
absent — the exact shape of artefact the original tool passed:

```python
from foundationscale.gates.core import REGISTRY, Verdict
from foundationscale.gates.example import ExpertCheckContext   # importing registers the gate
from foundationscale.gates.fixtures import make_empty_experts

gate = REGISTRY.get("checkpoint.expert_alias")
ctx = ExpertCheckContext.from_expert_set(make_empty_experts(declared_expert_count=128))
result = gate.run(ctx)

print(result.render())
# [VACUOUS] checkpoint.expert_alias: 0/128 experts — gate examined 0 experts and
# therefore proves nothing (claimed: …)
assert result.verdict is Verdict.VACUOUS
assert result.blocking
```

Three more properties of the contract, because each is drawn from the record:

* **Controls are executable.** Every gate declares fixtures — at least one
  `MUST_FIRE` (a deliberately broken input it must block) and typically a
  `MUST_PASS` (known-good, which catches gates that block everything and get
  disabled). `verify_controls()` runs them and is wired into CI as its own job; a
  gate with no `MUST_FIRE` control fails the build, because a gate that has never
  been shown to fire is not evidence of anything. This is the audit's second review
  rule, made executable.
* **Gates fail closed.** An exception inside a gate is `ERROR`, and `ERROR` blocks.
  In the audited estate, a reward-module import failure silently disabled a
  degeneracy veto, and a verifier exception counted as a pass.
* **Absence blocks, one level up.** Callers can declare which gates are required at
  a lifecycle point; any that never ran render as `MISSING` and block the report. In
  the audited estate the export byte check lived as a copy-pasted heredoc in one
  script and was simply absent from the other, which is how a truncated export
  reached `rc=0`. A registry that silently ran zero gates is the same failure as a
  gate that silently checked zero units, one level up.

## 3. Proof the checks can fail

A verification framework whose own checks cannot fail would be the thing it exists
to prevent. So the framework's test infrastructure is part of the claim, and it is
measured, not asserted:

* **982 tests and 15 subtests, 0 skipped**, with 95% statement coverage of
  `src/foundationscale` (4,218 statements, 230 missed) — and CI is configured to fail on
  *any* skip at all. How it acquired that policy is the last item in the next section.
  The zero is *declared*, not inferred: the run that produced it set `FS_FORBID_SKIPS=1`,
  which arms a guard that reports `zero skips observed` and turns the run red if even one
  test skips. An unarmed run reporting no skips would be the weaker claim.
* **A mutation battery** ([`tools/mutate.py`](../tools/mutate.py)): rules of the contract
  deliberately broken — one mutant per rule — with the suite required to catch each one.
  The corpus is **70 rows across 9 modules: 66 scoreable MUST-FIRE mutations and 4
  MUST-PASS control rows.** A full run exits 0 with **66 killed, 0 alive, 0 unscored, and
  caught=100% in every one of the 9 modules**, the 4 inert control edits surviving as they
  must, and all 9 modules restored byte-for-byte afterwards. The battery prints that
  denominator itself, per module, unprompted.

  It did not start there, and the history is the point. **When the table held 42 rules**,
  the first full run caught 38, and the 4 survivors were four rules the modules stated and
  nothing tested — including the byte-deficit tolerance that is the incident itself. They
  were recorded as gaps, then closed by writing the tests that kill them. A battery is
  worth having because it finds survivors, not because it reports none. The corpus has
  roughly doubled since, which is what growth looks like when the rules are enumerated by
  hand: the 42 and the 70 describe different tables, not a change of score.
* **That tally is printed beside negative controls, and it did not used to be.**
  The table carries **four** `must_survive` rows — comment-only edits that change no
  behaviour, so a sound suite cannot detect them. If the battery reports killing one,
  the run is voided at exit 2: a harness that can manufacture kills has nothing to
  say about the ones it printed. This is not decoration. An earlier revision of this
  repository published `42 killed, 100% caught` from a harness that scored exactly
  that inert edit as `[killed]` — the meta-tests were driving the battery against the
  live mutation table, so any mutant that disturbed an anchor was killed on contact
  regardless of behaviour. A run whose table configures *no* control row now says so
  in words and exits 2 rather than printing a tally with its negative half missing.
  The full retraction, with the before-and-after measurement, is in
  [`docs/SELF_AUDIT.md`](SELF_AUDIT.md) §3.10.
* **A survivor and an unapplied mutation exit differently from a clean run.** If an
  anchor no longer matches its module, that mutation did not run, and the battery
  exits nonzero rather than counting it as caught. This is not hypothetical either: a
  run reported `40 killed, 0 alive, 2 n/a` and exited 0 — the tool for detecting
  reported success over unexamined work, doing exactly that.
* **Controls run as their own CI job**, over the live registry, so a gate cannot rot
  into a no-op that reports success on everything while the unit tests stay green.

## 4. The bug, inside this repository

None of the above prevented this repository from shipping the same failure modes it
audits for. While the framework was being built, its own verification produced each
of the classic vacuities — and each was found by measuring rather than assuming, via
the discipline this repository advocates: coverage-carrying verdicts, positive
controls, and mutation testing.

* A composite gate reported **"distinctness, byte volume and completeness all hold"
  while two of its three sub-gates had abstained.** Aggregation read abstention as
  assent; the parent now propagates coverage instead of minting a pass its children
  never earned.
* A gate treated **"no manifest found" as "nothing wrong"** — an empty discovery
  result read as a clean bill of health.
* A parity report **passed when the only key it compared contained zero elements**,
  because coverage counted keys — and a key is a container, not evidence.
* A loss gate **compared weights against `0.0` with `==`**, so a NaN weight — a term
  already poisoning the objective — took the false branch of every comparison and
  was reported as fine.
* The gate-audit entry point — the thing that checks the checkers — **crashed
  part-way through the registry instead of reporting.** A gate whose `controls()`
  raised, and a gate *subpackage* that raised on import, both escaped as tracebacks,
  so every finding collected before them was lost with the unprinted report. A
  checker that dies rather than reports is a checker whose findings vanish.
* **41 of this repository's own tests were skipping in CI while the run reported
  green.** The skip-list version of VACUOUS, shipped by the repository that exists
  to catch it. Any skip is now a CI failure — enforced by a guard that CI proves can
  fail, by feeding it a deliberately skipped test on every run. That is how the suite
  reached 412 with 0 skipped, and it is why these sections are written in the past
  tense with numbers attached.

This is stated plainly because it is the argument. The adversary is not careless
engineers; it is the systematic ease with which a broken check says "no defects
found." The discipline caught it here, repeatedly, at the framework's own expense —
including twice inside the tooling built to enforce the discipline. That is not a
confession to bury and it is not a victory lap: what is still weak is named under
Status below rather than left for a reader to discover.

