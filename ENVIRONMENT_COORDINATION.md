# Environment coordination — three development environments (2026-09-22)

**Supersedes** `artifacts/LANE_COORDINATION.md` (2026-09-20). Reason: that file was posted to the cluster workdir's `artifacts/` only and was never pushed to GitHub, so the Mac lane could not read it. A coordination protocol only one party can see cannot work. THIS file lives in the repo root and is pushed, or it repeats the failure.

## 1. The three environments

| Environment | Role | Authority | Identity fingerprint | Where its work lands |
|---|---|---|---|---|
| E1 — Claude Code on operator's Mac | primary feature development | **PUBLISHER** — only environment that pushes to GitHub main | git author `TranNhiem <86524375+TranNhiem@users.noreply.github.com>` | github.com/TranNhiem/FoundationScale, branch `main` (sole branch; currently `9d3c436`, 2026-09-22 12:10) |
| E2 — Kimi-Code on the GB200 server | parallel implementation + on-silicon campaigns | **PROPOSER** — may commit to local lane branches and export bundles; may NEVER push to GitHub | git author `TranNhiem <user.name>` (misconfigured identity — literally the string `user.name` as the email); also `hh28144 (cluster verification) <hh28144@users.noreply>` | `/home/hhri-ai/hh28144/Project-Developments/FoundationScale/fs-repo`, branch `kimicode/*`, plus git bundles in `artifacts/` |
| E3 — this Claude Code session on the GB200 login node | integration coordinator | **REVIEWER/ADJUDICATOR** — reviews, adjudicates conflicts, prepares merges; does not author features; pushes only on explicit operator instruction | to be set, see R1 | — |

## 2. Standing rules

- **R1. Identity is the audit trail.** Every environment sets a distinct `git user.email` before committing. E2's current `user.name` email is a misconfiguration and must be fixed to a real distinguishing address; until then the string is treated as the E2 marker. Never rewrite existing history to fix it.
- **R2. One publisher.** Only E1 pushes to `main`. E2 and E3 deliver via lane branches + bundles. A cluster push is a doctrine inversion and requires explicit operator instruction per instance.
- **R3. Coordination artifacts are repo-tracked and pushed, never cluster-only.** Ownership claims live in this file at the repo root.
- **R4. Claim before you edit.** An environment about to touch a shared file records the file and the symbol range in the ownership table below, in a pushed commit, before editing.
- **R5. Shared-territory files require a claim:** `src/foundationscale/train/loop.py` (the standing collision surface — it is 1 of the 13 files currently conflicting), `provenance/manifest.py`, `gates/checkpoint_gates.py`, `rl/trainer.py`, `checks/coverage_floor.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/review/D*.md`.
- **R6. Never merge on hash equality alone.** Verify content: patch-id (`git cherry`) for commits, byte compare for files. Both have produced findings here that file-existence checks missed.
- **R7. Mutation battery is exclusive.** `tools/mutate.py` mutates the live tree; one battery at a time, no reads/edits/commits by any environment while it runs.

## 3. Integration workflow

1. **Inspect:**
   ```
   git fetch origin --prune; git log --oneline HEAD..origin/main; git log --oneline origin/main..HEAD
   ```
2. **Attribute** — map each commit to E1/E2/E3:
   ```
   git log --format='%h | A:%an <%ae> | %s' origin/main..HEAD
   ```
3. **Test real uniqueness** — `+` genuinely new, `-` already upstream by patch-id:
   ```
   git cherry -v origin/main HEAD
   ```
4. **Predict conflicts without touching state:**
   ```
   git merge-tree --write-tree HEAD origin/main
   ```
5. **Review paired diffs per subsystem:** extract lane-side and upstream-side diffs of the SAME files from the merge-base and compare them against each other, not against the merge-base alone. Delegate the reading to Kimi-K3 fan-out; the coordinator verifies every load-bearing claim against source before acting on it.
6. **Adjudicate per subsystem**, recording one of: take-lane / take-upstream / manual-merge, with the reason.
7. **Document:** update this file and write the decision + evidence to `artifacts/evidence/<sha>/`.

## 4. Current integration state (as of 2026-09-22 17:30)

GitHub `main` `9d3c436`. Cluster lane `kimicode/coverage-thin-modules` `0c94f25`: 18 behind, 8 ahead, tree clean. All 8 lane commits are patch-unique (`git cherry`: all `+`); GitHub has never seen SHA `0c94f25`. 13 files will conflict on merge. The lane is the sole copy of `AGENTS.md` and of the #471 fix. Backup exists: `artifacts/fs_all_lanes_20260921.bundle` contains head `0c94f25`.

Adjudications (all three verified against source by the coordinator):

- **Manifest / rank-scope → MANUAL-MERGE.** Both environments independently fixed #471 config blindness. Upstream's `_config_expert_counts` (depth-3 walk, cycle-guarded, excludes bool) is strictly more general than the lane's 3-scope first-match probe and preserves tower-disagreement detection — take upstream's reader. But the lane's atomic rank-0 manifest writer is lane-only: upstream `_emit_manifest` (`train/loop.py:3194`) still calls bare `path.write_text` at lines 3224 and 3254 with no rank guard, so every rank races one path. Keep the lane's tmp+fsync+`Path.replace` and `_effective_rank`, and the `DeclaredCheckpoint.expert_scope` discriminator. Lane bug to fix on the way in: `isinstance(getattr(sub,key),int)` admits bool; upstream excludes bool everywhere (`loop.py:1865`, `2654`).
- **Checkpoint gates #471 → MANUAL-MERGE.** Neither side subsumes the other. Upstream `_is_adapter_only_artifact` (`checkpoint_gates.py:413`) is a MEASURED property of on-disk FQNs but is gated behind `num_experts is None` (line 933), so declared-N adapter saves still go vacuous-RED. The lane skips on a DECLARED scope string, which closes that case but trusts a declaration — a misdeclared manifest would silence the expert gates. Resolution: keep upstream's measured detector as the firing condition, extend it past the `None` guard, demote the lane's scope plumbing to audit metadata. The lane's skip message wording is stale post-#529 (the number now means active routed experts).
- **RL grpo refusal → MANUAL-MERGE, cheapest of the three.** The two refusals fire on disjoint branches at the same anchor: lane when objective is None, upstream when objective exists but lacks `advantage_fn` (#513). Take upstream wholesale and graft the lane's grpo reference-policy arm inside the objective-is-None branch. Drop the lane's `corpus.py` docstring edit — superseded by upstream's `gold_key` path. Lane risk: a broad `except Exception` around `requirements()` could mask a broken registry factory, and no MUST_FIRE test pins the arm, so it would silently stop firing if `GRPOAlgorithm` later gains `_objective`.

## 5. Scratch-tree sprawl

Seven git trees exist under the project folder besides `fs-repo`. Four had uncommitted changes. All were checked by byte comparison against upstream:

- `fs-rc` (30 dirty files): every new file byte-identical to upstream. Harvested. Safe to archive.
- `fs-490` (4 dirty): residual delta vs upstream is 8 + 2 lines, pure line-rewrapping and a trailing newline. No semantic content. Harvested. Safe to archive.
- `fs-ci` (2 dirty), `fs-verify` (3 untracked): no unique tracked work. `fs-verify` holds untracked `import_cost_probe.py` (51 lines, not upstream) and `receipts/` evidence dirs — salvage those two before archiving.
- `checkouts/` holds 5 more trees; only `fs-e66cdda` and `fs-head-tmp` are real repos, both stale.
- `repo/fs-repo` has no `.git` by design; never run git mutations there.
- `_incoming_kimi_work/` (2026-09-14, base `0c4663612cca`) is an older E2 handoff: `loop_py.patch` and `test_train_multi_rank.py`, the multi-rank manifest seam. Its subject matter was later committed as lane commit `d15cb32`, so it is superseded, but `test_train_multi_rank.py` never reached any branch — check it for salvageable coverage before deleting.

**Disposition rule:** scratch trees are disposable, never authoritative. Work leaves a scratch tree as a commit on a lane branch or a bundle in `artifacts/`, or it does not exist.

## 6. Ownership table

| Path/symbol | Claimed by | Scope | Status |
|---|---|---|---|
| `train/loop.py::_emit_manifest` + `_effective_rank` | E2 | manifest emission only | claimed 2026-09-20, CONFLICTED, pending manual-merge |
| `train/loop.py::_declare_checkpoint` expert probe | E1 and E2 both | #471 | COLLIDED, upstream reader wins |
| `gates/checkpoint_gates.py` expert gates | E1 and E2 both | #471 adapter scope | COLLIDED, manual-merge |
| `rl/trainer.py::_resolve_objective` | E1 and E2 both | refusal arms | COMPATIBLE, graft lane arm |
| `train/loop.py` axis handling (`_effective_topology`, dp/tp/pp/ep/cp, sharding, offload) | E1 (6D lane) | do not edit without ping | unchanged |
| `AGENTS.md` | E2 | doctrine | lane-only, must reach GitHub |

## 7. Open decision for the operator

How the 8 lane commits reach main. Three options, no recommendation stated:

- **(a)** Mac ingests `artifacts/fs_all_lanes_20260921.bundle` and resolves there.
- **(b)** Fast-forward the cluster to upstream first, then rebase the lane and re-export.
- **(c)** Push the lane branch to GitHub for a PR — inverts the Mac-is-publisher rule; needs explicit operator approval.

## Update 2026-09-26 (E3)

- **Commit identity.** Every environment commits as `TranNhiem <86524375+TranNhiem@users.noreply.github.com>`, the operator's contributor rule. The environment fingerprint moves to a trailer: `Environment: E3-cluster` (and `E2-kimicode` for the server lane). This supersedes the email-based fingerprint in R1.
- **E3 push route.** A write deploy key scoped to this repo (`hh28144@slogin01`, via `ssh.github.com:443`, since port 22 is blocked). E3 pushes **branches only** and only on explicit operator instruction. `main` changes only through a reviewed PR or a fast-forward the operator approves.
- **FoundationSkills.** The agent skills layer lives in `FoundationSkills/`, on branch `skills/foundationskills` and PR to main. It is its own territory: core workstreams need not touch it, and it measures the installed core rather than assuming it.
- **E2 lane `kimicode/coverage-thin-modules`, adjudicated on `integrate/e2-lane` against main d9e9a47:**

  | Lane commit | Verdict | Why |
  |---|---|---|
  | 189c002 thin-module tests | take | pass on main as-is (coverage only) |
  | d15cb32 one atomic manifest writer | port | main still wrote the manifest from every rank with in-place `write_text`; ported as `_effective_rank` + `_atomic_write_text`, keeping main's #487 unattributable-run warning. The corpus change is docstring-only (main already collapses "BB"). |
  | a9a19ab grpo refusal names the reference plane | port | main's refusal was still the generic no-axes message; the Open Code Review rule pack is taken too |
  | f8ad416 #471 expert_scope | drop (superseded) | main solved #471 with `config_expert_keys` + `_is_adapter_only_artifact`; the lane's tests target the lane's alternative API |
  | 40ec22d rl fallback test | take | |
  | 411a001, 0c94f25 countables resyncs | drop | regenerated fresh with `countables --fix` |
  | 5ee2303 AGENTS.md | take | |

  Four lane trainer tests assumed pre-#546 behaviour (every-step-UNMEASURED runs are now refused as vacuous; saturation wording changed). They were updated to main's behaviour, keeping their intent.
