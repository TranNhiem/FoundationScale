# The RL plane: design record and binding seams

FoundationScale has an RL plane at `src/foundationscale/rl/`. This record was written on 2026-09-09 **before any algorithm was bound to it**, to fix the shape of the seams first: five independent design reviews were run against the interfaces as they then stood, and all five concluded those interfaces did not fit: not in a way that any single signature change repairs, but in the shape of the seams themselves. NeMo-RL and comparable frameworks show which algorithm families exist; this record states what the seams must be so every one of those families plugs in without the core branching on algorithm identity.

WHAT IS CLAIMED:
- A design. Six decisions about interface shape, each anchored to a verified site in the current source, each argued from the failure it prevents.
- That the seams as described are sufficient for the algorithm families in the table below, at the level of interface coverage.

WHAT IS NOT CLAIMED:
- A measurement. The reviews rated interfaces, not running code, and they rated them as they stood on the day above — no review here was re-run against a bound family.
- Any benchmark, convergence claim, or performance result. None exists for this plane.
- Completeness. The open questions at the end were live when written; five families have since bound and only the one noted there as CLOSED has been retested.

**STATUS — corrected after the fact, measured at this commit.** The lead paragraph and the
open questions below originally stated, in the present tense, that nothing had bound. That
was true for one day. Five families have since bound — GRPO, RLOO, REINFORCE-with-baseline,
Reinforce++ and PPO — over a plane that now carries four advantage estimators, eight
protocols and ten loss classes. The design argument is left exactly as it was argued; only
the sentences that read as claims about the tree's present state are corrected, and they are
corrected in place rather than deleted, so the record still shows what was believed when the
seams were chosen. Counts here are measured from the source, not maintained by hand: re-take
them with `make countables` and the class census.

## The five seams

Five sites carry the coupling that must be broken:

1. **Requirements surface** — `AlgorithmRequirements` (`algorithm.py:160`) declares role presence via `requires_rollout` / `requires_advantage` / `requires_weight_sync` / `requires_reference_policy` booleans, plus `declared_components` and `declared_metrics`.
2. **Wiring enumeration** — `check_algorithm_wiring` (`algorithm.py:450-453`) iterates a hard-coded 3-tuple of roles: `rollout_source`, `advantage_fn`, `weight_sync`. The enumeration is source, not data.
3. **Advantage input** — `AdvantageFn.compute` (`advantage.py:259`) takes `prompt_ids`, `rewards`, `mask`. Three estimators implement it today: `GroupNormalisedAdvantage`, `LeaveOneOutAdvantage`, `GeneralisedAdvantageEstimation` (`advantage.py:472/530/578`).
4. **Step report** — `StepReport` carries `step`, a single `loss: LossOutput`, `rows`, `reward_stats: RewardStats | None`, `sync: SyncReport | None`. Critic-based families produce more than one loss per step.
5. **Objective gate context** — `build_objective_gate_context` hard-codes `uses_rewards=False` with a comment acknowledging the shape (`interfaces.py:278`, `interfaces.py:286`).

Two facts constrain the rework rather than invite it. First, the offline case is already anticipated: the comment at `algorithm.py:242-249` states that `requires_advantage` without `requires_rollout` is a legitimate offline shape, so DPO-style algorithms are not foreign bodies here. Second, an absence-as-data precedent exists: `algorithm.py:573` declines to refuse a missing `sync` under `requires_weight_sync` because a weight sync happens on a cadence — the framework already distinguishes "never required" from "required but not yet due".

## The six decisions

### D1. Add a semantics declaration beside the role-presence layer

`AlgorithmRequirements` stays a role-presence layer. The `requires_*` booleans answer "is a rollout source wired in?"; they cannot answer "does this loss and this algorithm agree about what a group is?" Add a second, independent declaration of **consumed semantics** — group size K, ratio scheme, KL estimator, clip bounds, reference-freeness — stated by *both* the loss *and* the algorithm, and compared by `check_algorithm_wiring`:

```python
@dataclass(frozen=True, slots=True)
class AlgorithmSemantics:
    group_size: int | None
    ratio_scope: Literal["token", "sequence"] | None
    kl_estimator: str | None
    clip_bounds: tuple[float, float] | None
    reference_free: bool | None
```

Two declarations that must agree is an instrument; one shared constant is not. **Why not put semantics on the algorithm alone:** a single declaration has no second side to disagree with, so a mismatched loss reads it, believes it, and trains wrongly — the handshake exists precisely because both halves can state the wrong thing independently.

### D2. The roles tuple becomes data, not source

Replace the hard-coded 3-tuple at `algorithm.py:450-453` with two mappings:

```python
requires: Mapping[str, bool]
supplied: Mapping[str, Any]
```

The checked denominator is the **union** of both key sets. This matters concretely: were the denominator `requires` alone, a newly supplied role — a critic, a reward model, a reference policy — would sit in no denominator and the wiring check would print CLEAR over it. That is this repository's signature defect class (an instrument reporting CLEAR over a set that does not contain the thing the claim is about) and it must not be reintroduced here. **Why not extend the tuple when a new role is needed:** every new role becomes a core edit, and the edit is exactly where the denominator silently stays stale. With data, adding a critic role requires no core edit.

### D3. Do not widen `AdvantageFn.compute`

Do not add parameters and do not add a `**kwargs` escape hatch. Widening breaks all three existing estimators; `**kwargs` makes every estimator's real input set unmeasurable. Pass one frozen record and require each estimator to declare what it reads:

```python
@dataclass(frozen=True, slots=True)
class AdvantageInputs:
    prompt_ids: Sequence[str]
    rewards: Sequence[float] | None
    mask: Sequence[Sequence[int]] | None
    logprobs: Sequence[Sequence[float]] | None
    values: Sequence[Sequence[float]] | None

@runtime_checkable
class AdvantageFn(Protocol):
    reads: frozenset[str]
    def compute(self, inputs: AdvantageInputs) -> AdvantageResult: ...
```

Reading an undeclared field is a refusal naming the field and both sides' sets — never a silent success. **Why not the widened signature:** `**kwargs` answers the question "what does this estimator consume?" with "unknowable without reading its body"; the `reads` declaration keeps that question checkable.

### D4. `StepReport` must carry more than one loss

PPO has a policy loss and a value loss; a single `LossOutput` cannot state both. Replace `loss` with a mapping of named outputs:

```python
@dataclass(frozen=True, slots=True)
class StepReport:
    step: int
    losses: Mapping[str, LossOutput]
    rows: int
    reward_stats: RewardStats | None
    sync: SyncReport | None
```

The existing `reward_stats.count == rows` cross-check must be restated against the correct denominator: with multiple losses and partial batches, `rows` is no longer the set the reward statistics were computed over. A check against the wrong denominator manufactures false refusals — a denominator that is a subset of the claim is worse than no denominator, because it fails loudly and wrongly. **Why not keep one loss and a tuple of auxiliaries:** the named mapping makes each loss addressable and its presence checkable; an unnamed tuple laundered through index arithmetic is a denominator bug waiting to be written.

### D5. Derive `uses_rewards`; never hard-code it

`uses_rewards=False` at `interfaces.py:286` is honest today — nothing consumes rewards — and becomes a lie the moment the first RL algorithm binds. The fix is **not** to flip it to `True`. Derive it from whether a reward source is declared, and make the no-source case abstain: `uses_rewards: bool | None`, where undeclared reports `None`. `False` is a claim ("rewards were declared and not used"); `None` is an abstention ("not measured"). The framework's rule is that an unmeasured quantity is `None`, never `0.0` and never `False` [-> docs/DESIGN_PRINCIPLES.md]. **Why not just set it from config:** a config flag is a promise, not a measurement; deriving from a declared reward source ties the report to what is wired, which is what wiring checks exist to attest.

### D6. `ratio_scope` is loss geometry, refused at wiring time

Sequence-level ratio is objective geometry and belongs to the loss declaration. It cannot live in the advantage function — the advantage never sees logprobs — and it cannot be implicit in the loss, because `SFTLoss` owns a fixed token-level denominator. So D1's `ratio_scope: token | sequence` sits on the loss, the batch must carry the matching denominator, and a disagreement is refused at wiring time naming the field and both sides' stated scopes. **Why not test for the mismatch instead:** a test detects one instance of the GSPO token/sequence-denominator bug; a typed declaration with a refused mismatch makes the bug unrepresentable. Refusals age; tests rot.

## How to add an algorithm

1. Write the algorithm's `AlgorithmRequirements`: role booleans, `declared_components`, `declared_metrics`. Do not reclassify presence booleans as semantics.
2. Write the algorithm's `AlgorithmSemantics` independently of the loss you intend to pair — from the paper, not from the implementation.
3. Declare every role you wire through the `supplied` mapping (D2), including roles with no `requires` counterpart; a role in no denominator is unmeasured by construction.
4. If the estimator you need is new, set `reads` to exactly the `AdvantageInputs` fields it consumes (D3). Reading anything outside that set must fail in your own unit test before CI sees it.
5. Name every loss by function (`policy`, `value`, `kl`) in the `StepReport.losses` mapping, and state which denominator each `RewardStats` was computed over (D4).
6. Declare a reward source if the algorithm consumes rewards; otherwise expect `uses_rewards=None` and treat a `False` in your gate output as a bug to file, not a value to assert (D5).
7. State `ratio_scope` on your loss; if your batch denominator disagrees, fix the batch, not the declaration (D6).
8. Run `check_algorithm_wiring`. A CLEAR result over an empty `supplied` union is vacuous and must be refused — `all([]) is True` remains the founding defect [-> docs/DESIGN_PRINCIPLES.md].

## Families and the seams they exercise

| Family (example algorithms) | D1 semantics | D2 roles | D3 advantage | D4 losses | D5 rewards | D6 ratio |
|---|---|---|---|---|---|---|
| Critic-free on-policy (GRPO, RLOO, REINFORCE, Reinforce++, Dr.GRPO, DAPO) | ● | ○ | ● | ○ | ● | ● |
| Critic-based (PPO, VinePPO) | ● | ● | ● | ● | ● | ● |
| Offline preference (DPO, IPO, KTO, ORPO, SimPO, CPO) | ● | ● | ○ | ● | ○ | ○ |
| Online/iterative (Online DPO, Iterative DPO, RAFT, RSO, Best-of-N) | ● | ● | ○ | ● | ● | ● |
| Sequence-level (GSPO, POLAR, length control) | ● | ○ | ○ | ○ | ● | ● |

● the family cannot bind without the seam; ○ the seam is exercised but the current shape survives. No row is all ○: every family needs at least D1.

## Open questions

- **Critic plumbing.** D2 makes the critic role free to declare but says nothing about value-target shapes flowing into `AdvantageInputs.values` (D3); the GAE migration path is unwritten.
- **`sync` cadence as data.** The `algorithm.py:573` precedent (a cadence-bound absence is not a refusal) suggests `supplied` may need a per-role cadence field; no shape has been chosen.
- **~~Is `ratio_scope` a closed union?~~ CLOSED by B5 at two members — and the second of the two guesses above was the right one.** The question offered a length-normalised sequence ratio as either a third member or "a sequence ratio plus a declared denominator", and the second is what the geometry turns out to be. GSPO's ratio is `exp(mean_t log_ratio_t)` — the length normalisation sits *inside* the exponent, so it is a property of how the sequence ratio is formed, not a separate scope. It is also not optional: the unnormalised alternative, `exp(sum_t log_ratio_t)`, leaves any usable floating-point range over a response of a few hundred tokens for any non-trivial policy shift, so a sequence ratio that is not length-normalised is not a geometry anything binds. Normalisation is what makes a sequence ratio exist, not a variant of one. **This is an argument from the arithmetic, not a measurement** — nothing has trained.
  What *is* independent between the families is the denominator, and it is the axis that actually separates them: GSPO averages one scalar surrogate per response, Dr.GRPO divides by a configured constant length, and DAPO divides by the batch's total supervised-token count. That is a genuine three-way distinction with nowhere to live under `ratio_scope`, so B5 carries it on a separate `reduction` declaration (`sequence_mean | constant | token_mean`) beside the scope. **Why not widen `ratio_scope` to carry both:** a scope that also encoded its denominator would make `("sequence", token_mean)` unrepresentable at the type level while leaving `("token", sequence_mean)` spellable, which is precisely inverted — two independent axes collapsed into one union refuse the wrong combinations. D6's refusal therefore stayed an instrument rather than becoming an obstacle, which is what the question was watching for.
- **Whether semantics comparison should be structural or advisory.** `check_algorithm_wiring` refusing on any `None`-vs-value semantics mismatch may over-block algorithms that genuinely do not fix a KL estimator; the refusal policy per field is open.
- **~~No second family has bound.~~ CLOSED by the bindings above.** This was written expecting that the first binding to force a revision would be a feature of the process rather than a failure of it, and that is what happened: the seams held for all five families, and the one shape they did not anticipate was a SECOND learned model inside a single step. PPO needs a value estimate that two consumers — the value regression and the temporal estimator — must agree about, which the seams as designed had no place for; it became the `ValueHead` protocol rather than a field on the composite. The remaining open questions below have not been retested.
