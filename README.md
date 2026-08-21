<p align="center">
  <img src="assets/hero.png" alt="FoundationScale — a scalable distributed training framework for foundation models, from single-GPU experiments to large-scale multi-node training" width="100%">
</p>

<h1 align="center">FoundationScale</h1>

<p align="center">
  <b>A forensic audit of a real training estate, and the checkpoint-verification framework it demanded.</b>
</p>

<p align="center">
  Gates whose verdicts carry their own coverage &mdash;<br>
  because a check that cannot fail is worse than no check. It also buys confidence.
</p>

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
evidence chain is in [`docs/deliverables/`](docs/deliverables/README.md), published
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
[`docs/DECISIONS.md`](docs/DECISIONS.md), or jump to the
[deliverables index](docs/deliverables/README.md):

| | Document | What it answers |
|---|---|---|
| **A1** | [Existing system — structure](docs/deliverables/A1_existing_system.md) | Two repos, workflows end-to-end (SFT / RL / export), infrastructure, dependency map |
| **A2** | [Existing system — algorithms](docs/deliverables/A2_algorithms.md) | Every objective, the reward cascade, pipeline coverage matrix |
| **A3** | [Existing system — debt & risk](docs/deliverables/A3_debt.md) | Duplication census, the **silent-failure catalogue**, blast-radius accounting, risk register |
| **B1** | [Architecture — core](docs/deliverables/B1_architecture.md) | The L0–L6 layered design, gate catalogue, run manifest, falsifiable claims |
| **B2** | [Architecture — unified LLM/VLM + scaling](docs/deliverables/B2_scaling.md) | One data contract, one Stage abstraction, 1 GPU → 100+ nodes as a ladder with measured rungs |
| **C** | [Codebase mapping](docs/deliverables/C_mapping.md) | Per-component Reusable / Refactor / Wrap / Replace / Build-New |
| **D** | [Development roadmap](docs/deliverables/D_roadmap.md) | Phases 1–10, each with exit criteria and a **falsification condition** |

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
([`src/foundationscale/gates/core.py`](src/foundationscale/gates/core.py)), not in
the gate author's discipline: `Gate.ok()` cannot return `PASS` on zero coverage, no
matter what the author writes — it downgrades. An author who means "nothing to check
here" must say so via `skip()` with a reason, which is recorded and surfaced.

The reference gate — `checkpoint.expert_alias`, in
[`src/foundationscale/gates/example.py`](src/foundationscale/gates/example.py) —
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

* **582 tests, 0 skipped**, 94.2% line coverage — and CI is configured to fail on *any*
  skip at all. How it acquired that policy is the last item in the next section.
* **A mutation battery** ([`tools/mutate.py`](tools/mutate.py)): 42 rules of the
  contract deliberately broken — one mutant per rule — with the suite required to
  catch each one. It currently catches 42 of 42. It did not start there: the first
  full run caught 38, and the 4 survivors were four rules the modules stated and
  nothing tested, including the byte-deficit tolerance that is the incident itself.
  They were recorded as gaps, then closed by writing the tests that kill them. That
  history is the point — a battery is worth having because it finds survivors, not
  because it reports none.
* **That 42-of-42 is printed beside a negative control, and it did not used to be.**
  The table carries a `must_survive` row — a comment-only edit that changes no
  behaviour, so a sound suite cannot detect it. If the battery reports killing it,
  the run is voided at exit 2: a harness that can manufacture kills has nothing to
  say about the ones it printed. This is not decoration. An earlier revision of this
  repository published `42 killed, 100% caught` from a harness that scored exactly
  that inert edit as `[killed]` — the meta-tests were driving the battery against the
  live mutation table, so any mutant that disturbed an anchor was killed on contact
  regardless of behaviour. A run whose table configures *no* control row now says so
  in words and exits 2 rather than printing a tally with its negative half missing.
  The full retraction, with the before-and-after measurement, is in
  [`docs/SELF_AUDIT.md`](docs/SELF_AUDIT.md) §3.10.
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

## Quickstart

From a clean clone:

```bash
git clone <this-repository> && cd foundationscale

# [checkpoint] carries torch. Install only [dev] and 41 tests skip instead of run —
# which is precisely how this repository's CI was green over them. Any skip fails
# CI now, but on a laptop the suite would simply be quieter and smaller.
python -m pip install -e ".[checkpoint,dev]"

# 1. The unit and gate suite — 582 tests, 0 skipped. Every skip is named in the
#    summary with its reason; in CI (FS_FORBID_SKIPS=1) each one fails the build.
python -m pytest

# 2. The controls — every gate's MUST_FIRE / MUST_PASS fixtures, run over the live
#    registry. This is verbatim what CI's `controls` job runs. It exits nonzero if a
#    gate fails to block its defective input, if a gate declares no MUST_FIRE control
#    at all, or if the registry is empty — a controls run that verified nothing is
#    the vacuous pass one level up.
python -m foundationscale.gates.controls

# 3. The mutation battery — 42 deliberate breakages of the gate contract, each
#    required to turn the suite red. ~2 minutes. Exit 0 only if every mutation was
#    applied AND killed; 1 if a mutant survived; 2 if it could not measure (red
#    suite, any skipped test, or an anchor that no longer matches its module).
python tools/mutate.py
```

`make check` runs all of the above plus lint, types, and the skip-guard probe.

## What the diagram shows

The poster above describes the training frame that is being built *on top of* the
gate plane — the audit's conclusion was that this layer had to exist first. The
trainer itself is early (see Status below).

**Model stack** — curated recipes and framework libraries covering the full model
development path, from pre-training and SFT/LoRA (VFM, LLM &amp; VLM) through the
framework layer (Megatron-LM, Megatron-Bridge, AutoModel) to post-training
alignment and RL.

**Parallelism strategies** — the four ways work is distributed across GPUs:

| | strategy | what is split |
|---|---|---|
| **DP** | Data Parallelism | Same model replicated on each GPU, different data per GPU |
| **TP** | Tensor Parallelism | Each GPU holds a slice of every layer (tensor / hidden dimension split) |
| **PP** | Pipeline Parallelism | Different layers (stages) on different GPUs, executed sequentially |
| **EP** | Expert Parallelism | Different experts on different GPUs, a router selects which experts run |

**Distributed infrastructure** — the same stack is designed to run on local
machines, Slurm clusters, DGX Cloud / Lepton, and Kubernetes clusters.

## Status, and what is still missing

Early, and deliberately so. The gate contract
(`src/foundationscale/gates/core.py`) is frozen; the reference gate, fixtures,
controls job, mutation battery and skip policy above all exist and run in CI. The
layers above the gate plane are being built against the roadmap in
[`docs/deliverables/D_roadmap.md`](docs/deliverables/D_roadmap.md), each phase with
a falsification condition. The audit and the gate plane are the useful artefacts
today.

What is still missing, stated rather than implied:

* **Coverage is 94.2% overall, but it is not evenly earned.** The weakest module is
  `checkpoint/dcp_meta.py` at 74%, followed by `gates/checkpoint_gates.py` at 94% —
  and `checkpoint_gates.py` is the module that encodes the incident, so its 6% is the
  6% that matters most. The enforced floor is a total, which means a well-covered
  module can subsidise a thin one; per-module floors are not yet in place.
* **The mutation battery measures the rules someone thought to write down.** 42 of 42
  are killed, which says every listed rule has a test behind it. It says nothing
  about a rule nobody listed, and the table is hand-maintained: if a module is
  refactored, its anchors go stale, and the battery reports that rather than hiding
  it, but re-deriving them is still manual work. The negative control is likewise
  *one* row: it proves the harness can still say "no", not that the suite's
  attribution is sound for every mutant it scored.
* **No gate has run against a real distributed checkpoint in this repository's CI.**
  The suite writes real checkpoints to disk and reads them back, single-process. The
  audited estate's failures were multi-rank; the shapes are reproduced from the
  record, not re-observed here.
* From the audit's "still open" list: the estate's export bytes are verified —
  200 of 205 export directories tensor-by-tensor against the training format at
  0 DIFFER, 3,840/3,840 experts bitwise identical, with controls firing at 128 and
  112 — but **no exported artefact has ever been asked to produce a token.**
  Bitwise-identical weights survive a wrong rope base, a mismatched attention
  kernel, a tokenizer drift and a bad generation config. That probe is one node,
  one GPU, about twenty minutes, and it is specified to land as the export path's
  first semantic gate rather than as a one-off script.

## Assets

| file | description |
|---|---|
| [`assets/hero.png`](assets/hero.png) | 2000 &times; 2242 — the image used above |
| [`assets/hero@2x.png`](assets/hero%402x.png) | 4000 &times; 4484 — retina / print master |
| [`assets/hero.html`](assets/hero.html) | Self-contained source (system fonts, inline SVG, no external requests) |

The poster is rendered from `hero.html`, which has no network dependencies — open
it in any browser, or re-render it headlessly at any resolution:

```bash
chrome --headless=new --disable-gpu --hide-scrollbars \
       --window-size=2000,2242 --force-device-scale-factor=2 \
       --screenshot=hero@2x.png file://$PWD/assets/hero.html
```

## License

[MIT](LICENSE).
