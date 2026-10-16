# Verification: the audit, and why the gates look like this

This document is the rationale for the gate plane: how the audit's central failure — a run that reports success — is made structurally impossible to repeat.

## 1. The audit

FoundationScale's design grew out of an audit of two production training codebases — 1,774 Slurm jobs and 205 export artefacts across MoE and dense models on H100 and GB200.

> **The dominant failure mode in large-scale training is not a crash. It is a run that reports success.**

The founding incident — a checkpoint holding **5.71 GB where 45.70 GB was correct** while the tool written to detect that corruption reported **`expert_perm_all_identity: true`**, because the expert tensors it meant to compare were absent and **`all([])` is `True`** in Python — is told in full in [ARCHITECTURE.md](ARCHITECTURE.md).

A second item from the record: one run logged `grad_norm` exactly 0.000 for 472 steps while reporting `reward/mean=0.794, success=1.00`. Five of the ten target stages have never executed at all. Every check that existed asked *"did anything go wrong?"* — a question a broken system answers "no" just as fluently as a working one.

The project applies two rules to its own claims. First: **an unqualified count is not a fact**. Second: **every claim that something does not exist must name the positive control proving its detector could have fired.** The second rule is what the framework below is, once you make it code.

## 2. The framework: a verdict is a claim about coverage

The central abstraction is not a trainer or a parallelism strategy. It is a **gate** — a correctness check that runs at a defined point in a job's lifecycle (launch, build, data, first optimizer step, first save, every save, export, promotion) and can block. What separates a gate from an assertion is that its verdict includes its coverage:

```python
class Verdict(str, Enum):
    PASS          # examined N units, all correct
    FAIL          # examined N units, found a defect
    VACUOUS       # examined ZERO units, and therefore proved nothing — blocks
    UNDERCOVERED  # examined fewer than expected, without declaring a sample — blocks
    SKIP          # explicitly not applicable; reason required; reported; non-blocking
    ERROR         # the gate itself raised — gates fail closed; blocks
```

**VACUOUS and UNDERCOVERED are the design, not edge cases.** There are two ways to not block and four ways to block, and the asymmetry is deliberate: it is far easier to produce a meaningless success than a meaningless failure. A gate reporting "3 layers checked" out of 205 is UNDERCOVERED unless it declares itself a sample, with the reason; a gate reporting "all experts match" over an empty comparison set is VACUOUS. Both block.

The rule is enforced in the base class ([`src/foundationscale/gates/core.py`](../src/foundationscale/gates/core.py)), not in the gate author's discipline: `Gate.ok()` cannot return `PASS` on zero coverage, no matter what the author writes — it downgrades. An author who means "nothing to check here" must say so via `skip()` with a reason, which is recorded and surfaced.

The reference gate — `checkpoint.expert_alias`, in [`src/foundationscale/gates/example.py`](../src/foundationscale/gates/example.py) — encodes the founding MoE-aliasing incident ([ARCHITECTURE.md](ARCHITECTURE.md)). Its `check()` contains no special case for an empty expert set. Deliberately:

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

Here is that code path running over a checkpoint whose expert set is entirely absent — the exact shape of artefact the original tool passed:

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

Three more properties of the contract:

* **Controls are executable.** Every gate declares fixtures — at least one `MUST_FIRE` (a deliberately broken input it must block) and typically a `MUST_PASS` (known-good, which catches gates that block everything and get disabled). `verify_controls()` runs them and is wired into CI as its own job; a gate with no `MUST_FIRE` control fails the build, because a gate that has never been shown to fire is not evidence of anything.
* **Gates fail closed.** An exception inside a gate is `ERROR`, and `ERROR` blocks; the audited estate lost a degeneracy veto to a silent reward-module import failure and treated a verifier exception as a pass.
* **Absence blocks, one level up.** Callers can declare which gates are required at a lifecycle point; any that never ran render as `MISSING` and block the report. In the audited estate the export byte check existed in one script and was simply absent from the other, which is how a truncated export reached `rc=0`. A registry that silently runs zero gates is the same failure as a gate that silently checked zero units, one level up.

## 3. Proof the checks can fail

A verification framework whose own checks cannot fail would be the thing it exists to prevent. The test infrastructure is therefore part of the claim, and it is measured: current numbers — tests, coverage, mutation corpus — are recorded in [TESTING.md](TESTING.md).

* **Zero skips are declared, not inferred.** The run sets `FS_FORBID_SKIPS=1`, which arms a guard that reports `zero skips observed` and turns the run red if even one test skips. CI is configured to fail on *any* skip, so the zero is a property of the run, not an inference from its absence.
* **A mutation battery** ([`tools/mutate.py`](../tools/mutate.py)): rules of the contract deliberately broken — one mutant per rule — with the suite required to catch each one. The corpus is scored as MUST-FIRE mutation rows against MUST-PASS control rows, and the battery prints the denominator itself, per module. A full run must exit 0 with every MUST-FIRE mutation caught in every module, the inert control edits surviving as they must, and all modules restored byte-for-byte afterwards.
* **Negative controls void a run that kills them.** The table carries `must_survive` rows — comment-only edits that change no behaviour, so a sound suite cannot detect them. If the battery reports killing one, the run is voided at exit 2: a harness that can manufacture kills has nothing to say about the ones it printed. A run whose table configures *no* control row says so in words and exits 2 rather than printing a tally with its negative half missing.
* **A survivor and an unapplied mutation exit differently from a clean run.** If an anchor no longer matches its module, that mutation did not run, and the battery exits nonzero rather than counting it as caught.
* **Controls run as their own CI job**, over the live registry, so a gate cannot rot into a no-op that reports success on everything while the unit tests stay green.

## 4. The bug, inside this repository

None of the above prevented this repository from shipping the same failure modes it audits for; each class of vacuity was found here by measuring rather than assuming, and each leaves a rule behind:

* A composite gate treated abstention as assent and reported all checks holding while sub-gates had abstained. The parent propagates coverage instead of minting a pass its children never earned.
* A gate treated "no manifest found" as "nothing wrong"; an empty discovery result is not a clean bill of health.
* A parity report passed when the only key it compared contained zero elements; coverage counts units compared, and a key is a container, not evidence.
* A loss gate compared weights against `0.0` with `==`, so a NaN weight took the false branch of every comparison and was reported as fine.
* The gate-audit entry point crashed part-way through the registry instead of reporting, so findings collected before the crash were lost with the unprinted report. A checker that dies rather than reports is a checker whose findings vanish.
* The test suite skipped in CI while the run reported green. Any skip is now a CI failure — enforced by a guard that CI proves can fail, by feeding it a deliberately skipped test on every run.

The adversary is not careless engineers; it is the systematic ease with which a broken check says "no defects found."
