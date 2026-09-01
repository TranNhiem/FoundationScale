#!/usr/bin/env bash
# Reproducible build of the H100 launch plane (Deliverable C).
#
# Every artifact under h100/gen/ is GENERATED. Nothing here may be hand-edited:
# a hand edit puts the fix in the file you read and leaves it out of the file
# that runs. Every change is a gated stage below, and the whole plane is rebuilt
# from scratch on every invocation so "it works" cannot mean "it works because
# of a state nobody recorded".
#
# Each stage refuses to write while any of its own gates is red, so a red stage
# leaves the tree at the last good state rather than half-patched.
#
# After the stages, three standing checks run over the result:
#   * gate_env_drift.py  -- the env allowlist and the launcher's exports agree,
#                           in BOTH directions
#   * blocklist          -- nothing estate-identifying survives into a public repo
#   * bash -n            -- both artifacts parse
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# #151: the estate root is an INPUT to the build, never a literal inside it.
# patch_estate_roots.py must MATCH the hard-coded root it is deleting from the upstream
# launcher, so the value has to exist somewhere -- but a value checked into this repository
# is a published value. Required with no default, matching FS_ALLOWED_NODE and
# FS_CONTAINER_RUNTIME. Checked HERE rather than only in the stage so the build refuses in
# its first second instead of eight stages deep with three artifacts already rewritten.
: "${FS_ESTATE_ROOT:?FS_ESTATE_ROOT is unset (required, no default). It is the estate filesystem root the upstream launcher hard-codes and this build removes. Export it for the build; do not add it to any tracked file.}"

# #152, same contract, second literal. Checked here for the same reason: patch_partition_knob.py
# runs fourteen stages deep, and a build that dies there has already rewritten three artifacts.
# NOT the same variable as FS_PARTITION -- that one is a knob on the EMITTED LAUNCHER, read at
# submit time by an operator who may be on a different estate entirely. This one is the name
# being DELETED, and it exists only during the build.
: "${FS_PARTITION_LITERAL:?FS_PARTITION_LITERAL is unset (required, no default). It is the Slurm partition name the upstream launcher hard-codes 13x and this build replaces with the FS_PARTITION knob. A partition name is a bare lowercase word with no path or hostname shape, so no scan pattern can find it -- it is reachable only by being named, which is why it must be named HERE and never in a tracked file.}"

GEN=h100/gen
LAUNCHER=$GEN/launch_fs_h100.fixed.sh
BACKEND=$GEN/fs_container_backend.bound.sh
ENTRY=$GEN/fs_train.fixed.py
# The backend's base text, spliced from the upstream repo. It is an INTERMEDIATE, not a
# shipped artifact, but it is generated and therefore removed and rebuilt like one -- see
# the rm below. Until #136 it was neither: apply_splice.py was not a stage, so this 73 KB
# file survived every "from scratch" rebuild and the shipped backend was derived from a
# state nobody recorded. Measured: moving it aside turned the build RED at stage 1.
SPLICED=$GEN/fs_container_backend.spliced.sh
# #133: the model-root plane. Two generated artifacts, and the TEST is generated too --
# a hand-kept test beside a generated module drifts the moment the module's rule changes,
# which is exactly what the shallowest-depth fix did to the old ambiguity assertion.
MODELROOT=$GEN/fs_model_root.py
MRTEST=$GEN/test_fs_model_root.py
# #141: the launcher REQUIRED FS_CHECKPOINT_ADJUDICATORS and the build shipped zero
# adjudicators, so the only way to launch was to point the knob at a placeholder. A
# required knob with no satisfying artifact is a knob that teaches operators to fake it.
# Generated, and its suite generated with it, for the same reason as the model-root pair.
ADJ=$GEN/fs_ckpt_adjudicator.py
ADJTEST=$GEN/test_fs_ckpt_adjudicator.py

# Order is load-bearing where noted; the rest is stable for reproducibility.
#   apply_113        generates the launcher
#   apply_117        generates the backend  -- must precede every stage that
#                    touches it (resume_env, env_drift, launch_topology)
#   patch_env_drift  mints MASTER_PORT      -- must precede launch_topology,
#                    whose composer does ${MASTER_PORT:?...}
STAGES=(
  apply_splice.py             # #136: produces the backend base text from the upstream
                              # repo. MUST be first: both apply_113 and apply_117 read
                              # its output, so if it does not run the plane is built on
                              # whatever spliced.sh happened to be lying in gen/.
  apply_113.py
  patch_bindpop.py
  patch_estate_roots.py
  apply_117.py
  patch_resume_env.py
  patch_env_drift.py
  patch_launch_topology.py
  patch_wlm_allocation.py     # needs fs_compose_launch, so must follow launch_topology
  patch_launcher_interpreter.py  # same dependency
  patch_alloc_nodefault.py
  patch_bind_closure.py       # #132: R4 must resolve symlinks BENEATH each declared
                              # bind, not just stat the declared path. Last of the
                              # backend stages, because it only needs apply_117's
                              # fs117_verify_script to exist and no later stage
                              # rewrites that line.
  patch_collective_probe.py   # #129: mounts and torch were verified, a COLLECTIVE was
                              # not. Touches both shell artifacts and re-runs the drift
                              # gate itself, so it goes last of the two-file stages.
  patch_fabric_tripwire.py    # #163 BLOCKER: the s9 pre-launch tripwire hard-coded ONE
                              # estate's IMEX master endpoint, so the backend refused every
                              # Slurm launch on any site without a host of that name -- it
                              # is what killed job 37280 on the H100 estate. Replaced by the
                              # required-no-default knob FS_FABRIC_TRIPWIRE (host:port, or
                              # the explicit sentinel `none` to DECLARE that this estate has
                              # no pre-launch fabric check). Same stage also repairs fs_die,
                              # which exited 1 at all 72 call sites and so collapsed the
                              # published 0/5/95/96 contract into one state. Backend-only,
                              # and it rewrites the fs_die DEFINITION, so it must follow
                              # every stage that adds fs_die call sites.
  patch_nv_runtime.py         # #167 BLOCKER: the singularity arm built its exec argv with
                              # no --nv, so no host NVIDIA driver user-space library was
                              # bound in. Silent: libcuda.so.1 still resolves out of the
                              # image's CUDA compat layer, so torch.cuda.is_available()
                              # and device_count() both report green and the run dies at
                              # the FIRST NCCL collective on libnvidia-ml.so.1 (job 37284,
                              # 8/8 ranks). Adds the flag AND the R6 probe that measures
                              # the capability instead of trusting the argv. Must follow
                              # patch_fabric_tripwire.py: its refusals are `fs_die 96`,
                              # which only means 96 once fs163 has made fs_die code-aware.
  patch_master_addr.py        # #168 BLOCKER: fs_compose_launch expands ${MASTER_ADDR:?...}
                              # and its message names fs_backend_init as the producer, but
                              # the ONLY producer was the off-Slurm arm's local default.
                              # On a Slurm allocation nothing minted it, so the composer
                              # aborted in expansion after every real probe had passed --
                              # bind verification, drain gate, container entry, the torch
                              # and NVML probes, and a real 8-rank all_reduce (job 37291).
                              # fs116's own comment admits the asymmetry it left: deriving
                              # MASTER_PORT next to MASTER_ADDR "would have covered only
                              # the off-Slurm branch". Mints the pair in one place. Must
                              # follow patch_fabric_tripwire.py for the same reason #167
                              # does -- its refusals are `fs_die 96`; its own C3/C4/C5
                              # controls enforce that ordering, since a pre-fs163 fs_die
                              # would exit 1 and they would go red rather than pass.
  extract_fs_train.py         # the training entrypoint is the third generated artifact
  patch_fs_train.py           # #134: bounds guard read the flag, so every env-sourced
                              # launch refused. Must follow the extract.
  extract_fs_model_root.py    # #133: the model-root resolver and its suite. Independent
                              # of the two shell artifacts, so order here is free.
  patch_fs_model_root.py      # #133: ambiguity is a property of the SHALLOWEST populated
                              # depth, not of the whole subtree. The flat rule refused
                              # stock upstream layouts (gpt-oss `original/`,
                              # sentence-transformers `1_Pooling/`). Must follow the
                              # extract, and re-application is a byte-exact no-op (S1).
  patch_fs_train_model_root.py  # #133 stage C: until this ran, the resolver was an
                              # ORPHAN -- 12 green tests and zero callers, which is the
                              # #86 shape. Binds load_artifacts to the resolved config
                              # dir. Must follow BOTH patch_fs_train.py (it patches that
                              # file) and extract_fs_model_root.py (PRE-2 requires the
                              # resolver to be co-located).
  patch_fsdp_wrap_policy.py   # #172: the size-based FSDP policy is structure-blind. On job
                              # 37300 it wrapped Qwen3's 388,956,160-param embedding into its
                              # own FULL_SHARD unit that re-shards after the embedding
                              # forward, so the TIED lm_head then read the flat 1-D shard:
                              # `mat (1024x2560), vec (48619520)`, and 48,619,520 = 388,956,160/8
                              # on all 8 ranks. Replaced by transformer_auto_wrap_policy keyed
                              # on the model's OWN _no_split_modules, so the model-specific
                              # fact stays in the model. It rewrites build_runtime, which
                              # patch_fs_train_model_root.py must have finished with first.
  patch_train_phase_balance.py # #173: with #172 fixed, job 37304 trained for real -- loss
                              # 1.2414 -> 0.5841 over steps 10..50 of 200 -- and then refused:
                              # "cannot begin save while train is open". The run trains TWICE
                              # (0->early, save, resume, early->budget) but the train phase was
                              # opened for segment 1 and closed for segment 2, so each half hid
                              # the other and the train->save boundary had never executed. Two
                              # insertions restore the pair; the stage also ships an AST balance
                              # gate over _run, so no future edit can open or close a phase
                              # asymmetrically. Edits _run, not build_runtime, so it does not
                              # collide with #172 -- but keep it after, one owner per pass.
  patch_resume_proof_attribution.py # #177: the resume proof compared every rank's post-restore
                              # loss against ONE manifest scalar (rank 0's) and all-reduced MAX,
                              # so a single number answered two questions -- restore fidelity AND
                              # cross-rank agreement -- and could not attribute its own failure.
                              # Job 37319 reported after_resume and before_save BIT-IDENTICAL on
                              # rank 0 while that statistic read 0.17570888996124268; the eight
                              # rank payloads of the refused checkpoint hold exactly that spread,
                              # a quantity fully determined BEFORE the save. Each rank's own
                              # pre-save value was already on disk in its own payload, so the fix
                              # needs no format change and is backward-compatible: compare each
                              # rank against ITSELF for the tolerance, report cross-rank spread as
                              # its own named measurement, and declare fixed_eval_rank_invariance
                              # UNMEASURED when the ranks diverge. A real lossy restore stays RED
                              # and outranks divergence. Edits resume_and_prove, not _run, so it
                              # does not collide with #173 -- but keep it after, one owner per pass.
  patch_resume_tolerance_split.py # #192: the stage above left ONE tolerance deciding two
                              # unrelated questions. `--resume-tolerance` gated restore
                              # fidelity (RED, final) AND cross-rank agreement (an instrument
                              # property, measurable before any checkpoint exists). Job 37336
                              # at 0.0005 measured restore_delta 0.0 with both spreads
                              # 0.2940967082977295 and abstained; job 37319, the same arm at
                              # 10.0, recorded zero abstentions -- a cross-rank pass bought by
                              # raising the knob that also governs restore, where a delta of
                              # 9.9 is a PASS. This stage splits them: `--resume-tolerance`
                              # keeps its name and means RESTORE only; the optional
                              # `--rank-agreement-tolerance` carries the absolute cross-rank
                              # claim. Unset, the before-save spread self-calibrates only the
                              # PRESERVATION question -- rank_invariant may be True ONLY under
                              # an explicit tolerance, because a floor derived from the same
                              # run it judges cannot certify that run's absolute agreement.
                              # Must run AFTER #177, which authors the function it edits.
  emit_ckpt_adjudicator.py    # #141: the adjudicator the launcher's required knob has been
                              # asking for since #68 wired the call sites. Independent of
                              # the shell artifacts -- order here is free.
  patch_plane_dir.py          # #142 BLOCKER: the launcher sourced fs_container_backend.sh --
                              # a filename the build never produces -- from whatever directory
                              # it happened to sit in. Replaces that with a four-step plane
                              # resolver (FS_PLANE_DIR override, co-located siblings, the WLM's
                              # own record of the submitted Command, then a named refusal) and
                              # re-points the four submit-chain self-references at the resolved
                              # plane instead of the spool directory.
  patch_partition_knob.py     # #152: the launcher hard-coded one estate's Slurm partition
                              # 13x, of which 2 were functional. Invisible to every artifact
                              # scan for a reason worth stating: a partition name is a bare
                              # lowercase word with no hostname or path SHAPE, so no pattern
                              # can express it -- it is reachable only by being named, and
                              # naming it in a tracked file publishes it. Hence the pair:
                              # FS_PARTITION_LITERAL (build input, the name being deleted)
                              # and FS_PARTITION (runtime knob on the emitted launcher).
                              # Launcher-only, so it sits with the launcher stages.
  patch_list_separators.py    # #139: FS_ALLOWED_PATH_ROOTS documented itself as
                              # "space-separated" while the launcher's IFS made spaces not
                              # split, so a two-root value was read as one impossible root.
                              # MUST BE LAST of the launcher-touching stages: it rewrites
                              # split sites anywhere in the launcher, so any later stage
                              # that added one would silently escape the fix.
  patch_adjudicator_binding.py  # #146: FS_CHECKPOINT_ADJUDICATORS was required, parsed BELOW
                              # both of its consumers, never containment-checked like the
                              # other four path knobs, and never folded into FS_BIND_PATHS --
                              # so the one artifact the operator is REQUIRED to supply was the
                              # one artifact the container could not see.
                              #
                              # AFTER patch_list_separators.py, and the "must be last" note
                              # above is the reason this needs stating rather than assuming.
                              # This stage MOVES the adjudicator parse; its anchor is the
                              # POST-fs139 text (the scoped `IFS=$' \t\n' read -r -a`), so
                              # running it earlier finds A2=0 and refuses -- measured, that is
                              # exactly what happened. Placing it here is safe for the narrow
                              # reason that it RELOCATES an already-corrected split site and
                              # introduces no new one; the re-run of patch_list_separators.py
                              # below turns that from an argument into a measurement.
  patch_adjudicate_denominator.py # #188: #175's fix was authored and never installed. This
                              # stage was in neither STAGES nor PUBLISH_SET.txt, so the
                              # launcher kept shipping the one-liner `while read` loop whose
                              # body drains its own process substitution. MEASURED on jobs
                              # 37341/37342/37344: a tree holding 3 checkpoint dirs was
                              # adjudicated 1 dir deep, three separate times, under the banner
                              # "ADJUDICATE complete". The replacement collects the dirs into
                              # an array first and reports "adjudicated=N of M", so the
                              # denominator is printed and a short sweep cannot read as a
                              # full one. Runs before the fs187 stage below only because it
                              # edits an earlier region; their anchors do not overlap.
  patch_postmortem_adjudicate.py # #187: the post-mortem afterany link printed one sentence
                              # and exited 0 -- PASS over zero adjudications, in exactly the
                              # case the link exists for (production died, so the ordinary
                              # path returns before adjudicate_tree and nothing ever reads the
                              # checkpoints the failed run left behind). Placed here, last of
                              # the launcher-touching stages, for two reasons: its edit-2
                              # anchor is the training-launch line that launch_topology,
                              # wlm_allocation, launcher_interpreter and collective_probe all
                              # rewrite, so it must follow every one of them; and running it
                              # before the second patch_list_separators.py below puts its two
                              # new blocks inside that stage's re-measured invariant.
  patch_launcher_exit_discipline.py # #189, which REOPENS #174/#169/#171. All three were
                              # recorded fixed and none of them was: `fs175:` occurred 0x in
                              # both generated artifacts, the launcher carried 0 traps, and
                              # line 851 still handed the backend's hard-stop helper "$$" --
                              # the launcher's OWN pid -- so on every failure bash took
                              # SIGTERM's default action and the END line and the exit below
                              # it were both unreachable, along with the helper's enroot
                              # force-remove arm. The fix was authored, verified and committed
                              # into neither this list nor PUBLISH_SET.txt: the third instance
                              # of the orphan-stage class after #136 and #188, and the reason
                              # #191 exists. Placed after every launcher-touching stage above
                              # because its OLD_BLOCK anchor is the failure path that follows
                              # the training launch those stages rewrite, and before the second
                              # patch_list_separators.py below so its new blocks fall inside
                              # that stage's re-measured invariant.
  patch_postmortem_verdict_scope.py # #193, which REGRESSES #187. The stage above inserts
                              # #171's success-arm verdict mapper ABOVE the shared adjudication
                              # tail that #187 deliberately routes the post-mortem link into.
                              # The mapper has an unstated precondition -- a trainer ran and was
                              # supposed to declare a verdict -- which the post-mortem arm
                              # violates by construction, so it maps to 95 and exits before
                              # adjudicate_tree is ever called. Measured: job 37344 (pre-#189)
                              # reached the tail and printed
                              # `END rc=0 phase=post-mortem checkpoint_saves_adjudicated=1`;
                              # job 37347 (post-#189) printed `END rc=0 mapped_rc=95 phase=train
                              # FAILED (...)`, 0 ADJUDICATE lines, sacct FAILED 95:0. This stage
                              # scopes the mapper to FS_SKIP_TRAIN != 1 and logs the deliberate
                              # non-application on the other arm. Necessarily placed directly
                              # after the exit-discipline stage, whose output is its anchor, and
                              # before the second patch_list_separators.py for the same reason
                              # that stage gives. The lesson is #193's own: a fix that inserts a
                              # guard on a shared path owes a drill on every arm that path
                              # serves, and the launcher enumerates its arms in one FS_PHASE
                              # variable -- #189 drilled one of three.
  patch_launch_provenance.py  # #180: the composed launch command was executed and recorded
                              # nowhere. Job 37319's own 936-line log contains the launch
                              # command 0 times, --resume-tolerance 0 times, and
                              # --model-path/--dataset-path/--sequence-length 0 times combined,
                              # so no run in this campaign is reproducible from its own output
                              # -- and the trainer's resume knob is flag-only, meaning a value
                              # that can arrive ONLY on a command line was held by no artifact.
                              # This writes ${RUN_LOG%.log}.provenance.json immediately before
                              # exec, on the training arm only: the post-mortem arm launches no
                              # trainer, so a record there would assert a run that never
                              # happened. The path is derived from RUN_LOG by suffix
                              # substitution rather than recomputed from LOG_DIR and the job id
                              # (#150: two computations of one name diverge). Placed here, after
                              # every stage that rewrites the exec line or the composer, because
                              # its three anchors are those stages' output; and before the
                              # second patch_list_separators.py so the new block is inside that
                              # pass's denominator rather than after it.
  patch_list_separators.py    # SECOND application, deliberately. The invariant "no unfixed
                              # split site survives" was previously enforced by ordering
                              # alone, i.e. by a comment -- and a comment is not a detector.
                              # The stage is byte-idempotent, so re-running it after the last
                              # launcher-touching stage costs nothing when the invariant holds
                              # and goes red the moment some future stage breaks it.
)

echo "=== rebuilding from scratch (removing generated artifacts first) ==="
rm -f "$LAUNCHER" "$BACKEND" "$ENTRY" "$SPLICED" "$MODELROOT" "$MRTEST" "$ADJ" "$ADJTEST"

fails=0
for s in "${STAGES[@]}"; do
  printf '\n--- %s ---\n' "$s"
  if python3 "$s"; then :; else
    echo "STAGE RED: $s (rc=$?)" >&2
    fails=$((fails + 1))
    break   # a later stage's anchors assume the earlier one applied
  fi
done
[[ $fails -eq 0 ]] || { echo -e "\nBUILD RED — ${#STAGES[@]} stages, $fails red" >&2; exit 5; }

echo -e "\n=== standing gate: bidirectional env drift ==="
python3 gate_env_drift.py || { echo "DRIFT GATE RED" >&2; exit 5; }

echo -e "\n=== standing gate: input/output partition (#136, #137) ==="
# Every file the build touches must be a DECLARED artifact or a DOCUMENTED upstream.
# Both #136 and #137 were a third thing -- a file read from the output directory that no
# stage produced -- and both were found by accident. This is the on-purpose version.
python3 gate_build_inputs.py || { echo "INPUT PARTITION GATE RED" >&2; exit 5; }

echo -e "\n=== standing gate: exit-code contract (#161) ==="
# The plane publishes a four-state contract (0 / 5 / 95 / 96) and twelve of its own exit
# sites did not honour it: `raise SystemExit("text")` prints the text and exits 1. Two
# sibling stages already carried a docstring warning about exactly this trap, which is
# how we know prose beside the code is not a control. This is the control.
python3 gate_exit_contract.py || {
  rc=$?
  case $rc in
    4)  echo "EXIT-CONTRACT GATE CONTROLS FAILED (rc=4) — the gate could not catch a planted" >&2
        echo "  violation, so its clean verdict on the tree means nothing." >&2 ;;
    5)  echo "EXIT-CONTRACT GATE RED (rc=5) — a published stage exits 1 where it declares 5/95/96." >&2 ;;
    95) echo "EXIT-CONTRACT GATE UNMEASURED (rc=95 -> build 95) — the publish set did not resolve" >&2
        echo "  enough Python files to scan. Not scanning is not passing." >&2
        exit 95 ;;
    *)  echo "EXIT-CONTRACT GATE unexpected rc=$rc" >&2 ;;
  esac
  exit 5
}

echo -e "\n=== standing gate: stage/publish-set orphans (#191) ==="
# Four findings came out of the gap between the two membership lists this script keeps --
# the STAGES array above, and h100/PUBLISH_SET.txt. #136 (a producer in neither list, under
# a header claiming "rebuilt from scratch"), #188 (an adjudicator fix regenerated away on
# every build), #189 (the self-kill fix, recorded as three separate closed findings and
# present in the artifact zero times) and #190 (five stages that ran and never shipped).
# A fix that is correct, committed, and in no execution list is indistinguishable from a
# fix that was never written, and until this gate nothing read either list against the
# other. Roles for the files that are legitimately in exactly one list are declared in
# h100/STAGE_ROLES.tsv and MEASURED here -- a declaration the gate cannot check is a
# comment, not a contract.
python3 gate_stage_orphans.py || {
  rc=$?
  case $rc in
    5)  echo "STAGE-ORPHAN GATE RED (rc=5) — a file is in a state the contract forbids:" >&2
        echo "  a stage that runs and does not ship, or a candidate in neither list with no" >&2
        echo "  declaration. See the per-file lines above." >&2 ;;
    95) echo "STAGE-ORPHAN GATE UNMEASURED (rc=95 -> build 95) — the gate could not derive its" >&2
        echo "  own inputs (STAGES array, publish set, or h100/STAGE_ROLES.tsv). Not deriving" >&2
        echo "  is not passing." >&2
        exit 95 ;;
    96) echo "STAGE-ORPHAN GATE REFUSE (rc=96) — a control failed or a role is unknown. A" >&2
        echo "  detector that cannot be shown to fire has not been shown to work." >&2
        exit 96 ;;
    *)  echo "STAGE-ORPHAN GATE unexpected rc=$rc" >&2 ;;
  esac
  exit 5
}

echo -e "\n=== doc stage-count agreement (#194) ==="
# The stage count is DERIVED here and was TYPED in four published documents, so every new
# stage silently falsified four sentences at once. LAUNCH.md said 17/17, Deliverable B said
# 33, E said 17/17, and this script printed 36. Same shape as #157 and #190: a claim whose
# denominator has no producer. It ran the other way too -- LAUNCH.md's lead box told
# operators Phase 3 had "never been executed" for a full campaign after job 37310 executed
# 8/8 legs. Understatement is the same defect as overstatement.
# Runs here, after every stage, for the same reason as the gate above: it reads the STAGES
# array and must not read it before the array has been executed.
python3 gate_doc_stage_count.py || {
  rc=$?
  case $rc in
    5)  echo "DOC STAGE-COUNT GATE RED (rc=5) — a shipped document states a stage count the" >&2
        echo "  build does not have. Fix the sentence, or make it past tense if it is" >&2
        echo "  deliberately about an earlier build. See the per-claim lines above." >&2 ;;
    95) echo "DOC STAGE-COUNT GATE UNMEASURED (rc=95 -> build 95) — zero claims matched, or the" >&2
        echo "  STAGES array could not be parsed. A scanner that matches nothing cannot tell a" >&2
        echo "  clean corpus from a broken pattern." >&2
        exit 95 ;;
    96) echo "DOC STAGE-COUNT GATE REFUSE (rc=96) — a control failed. A detector that cannot be" >&2
        echo "  shown to fire has not been shown to work." >&2
        exit 96 ;;
    *)  echo "DOC STAGE-COUNT GATE unexpected rc=$rc" >&2 ;;
  esac
  exit 5
}

echo -e "\n=== public-repo blocklist (case-insensitive) ==="
# This repo is public. The pattern is the standing one; report a count, not a
# verdict, so an empty grep cannot be mistaken for an unrun grep.
# #144: two tiers, because this was one pattern doing two incompatible jobs. An ESTATE
# IDENTIFIER is a disclosure by itself. A SECRET PREFIX is not -- `ghp_` is public knowledge,
# and only `ghp_` FOLLOWED BY A TOKEN BODY is a leak. Merged, the gate refused a generated
# stage whose only sin was carrying a redaction list, i.e. it pressured the code toward the
# LESS safe state. Split, with the bare prefix kept visible as a non-fatal notice so the
# narrowing cannot become a blind spot.
PAT="$FS_ESTATE_IDENT_PAT|/work/|ghp_[A-Za-z0-9]{20,}"
NOTICE_PAT='ghp_|-----BEGIN[A-Z ]*PRIVATE KEY'

# #151b: the extra-literal vocabulary is an INPUT, not a checked-in list.
# Some estate identifiers cannot be expressed as a safe pattern -- a bare account number is
# five digits, and `[0-9]{5}` would match line counts, byte sizes and half the corpus. The
# only way to catch them is to name them, and naming them IN THIS FILE would publish them,
# which is the same trap #145 describes one level up: the redaction list becomes the leak.
# So they arrive by environment, and the empty case is DECLARED rather than assumed --
# FS_REDACT_EXTRA=NONE means "this estate has no extra literals", unset means nobody decided.
: "${FS_ESTATE_IDENT_PAT:?FS_ESTATE_IDENT_PAT is unset (required, no default). Set it to the |-separated regex alternation of the identifiers of this estate -- node names, account ids, org segments, private hostnames -- or to the literal string NONE to declare there are none. #155: a redaction list compiled into a public repository publishes the estate it was written to protect.}"
: "${FS_REDACT_EXTRA:?FS_REDACT_EXTRA is unset (required, no default). Set it to a |-separated list of estate literals the pattern cannot express (bare account ids, private hostnames), or to the literal string NONE to declare that there are none. Unset is UNMEASURED, not clean.}"
if [[ "$FS_REDACT_EXTRA" != "NONE" ]]; then
  PAT="$PAT|$FS_REDACT_EXTRA"
  echo "  extra literals folded in from FS_REDACT_EXTRA (value not echoed)"
else
  echo "  FS_REDACT_EXTRA=NONE — declared: this estate contributes no extra literals"
fi

# #151: scan the GENERATORS, not only the generated. The old loop covered exactly the 7
# output artifacts, so the build could report "0 blocklist hits" while 32 sat in the inputs
# that produced them. patch_estate_roots.py -- the stage whose whole job is deleting an
# estate root -- carried that root hard-coded, and nothing here could see it. Same
# generated-vs-source asymmetry as #142 and #146, in the redaction plane.
# Build inputs are scanned with the SAME pattern but reported separately, because a hit in
# a generator is a different kind of problem from a hit in a shipped artifact.
GENERATED=("$LAUNCHER" "$BACKEND" "$ENTRY" "$MODELROOT" "$MRTEST" "$ADJ" "$ADJTEST")
# NOT mapfile: this runs on bash 3.2 (macOS system bash), where mapfile does not exist and
# the failure is a silent empty array -- i.e. a scan of zero files reporting zero hits.
GENERATORS=()
# ENUMERATED FROM DISK, not from a hand-maintained list. The first version of this named its
# six non-stage inputs explicitly and silently omitted gate_launch_contract.py -- a published
# file outside the scan's denominator, which is #151 one notch smaller and in the very check
# written to fix #151. A list of things to scan drifts from the things that exist by exactly
# the mechanism this build keeps finding; a glob cannot.
# #157, second turn. The glob above is derived from what LOOKS LIKE A STAGE; the publish set
# is derived from what SHIPS. Two rules, one denominator each, and nothing compared them --
# so `fs_estate_pat.py`, `h100_backend_splice.py` and five `h100/*.json` generator envelopes
# were published and scanned by nothing. One of those envelopes carried 11 estate identifiers.
#
# The first fix was a coverage GATE that detected the gap. That was still the wrong shape: a
# detector for a class of defect that construction can make impossible is a detector somebody
# has to keep answering. So the scan denominator is now DERIVED FROM the publish set --
# everything published that is not a generated artifact and not a document is a generator --
# and the gate below is demoted from discovery to a MUST_FIRE control on that derivation.
#
# The stage glob is KEPT and unioned in, not replaced: it covers files that are scanned
# without being published, which is the safe direction to over-scan in.
#
# #158, found by the coverage control below the first time it ran in anger: a producer inside
# this brace group that EXITS NONZERO truncates the group. `set -e` is on and is inherited by
# the process-substitution subshell, so `ls gate_*.py 2>/dev/null` returning 1 on an estate
# with no gates would kill the group before the publish-set branch runs -- and `2>/dev/null`
# means it does that in total silence. The union would degrade to the stage list and report a
# smaller denominator as a clean scan. The DOCS union below actually did this. So no producer
# here is allowed to fail: the glob is expanded by the shell under `nullglob` instead of by
# `ls`, which turns "no matches" from an error into an empty list.
_gate_files=()
shopt -s nullglob; _gate_files=(gate_*.py); shopt -u nullglob
_pub_gen=()
if [[ -r h100/PUBLISH_SET.txt ]]; then
  while IFS= read -r _p; do _pub_gen+=("$_p"); done < <(
    grep -vE '^[[:space:]]*(#|$)' h100/PUBLISH_SET.txt | sed 's|^\./||' \
      | grep -vE '\.md$' \
      | grep -vxF -e "$LAUNCHER" -e "$BACKEND" -e "$ENTRY" -e "$MODELROOT" \
                  -e "$MRTEST" -e "$ADJ" -e "$ADJTEST" || true)
fi
# Positive control on the sub-producer, not just on the union. The coverage gate at the end
# catches a truncation only when the dropped file is PUBLISHED; a silently empty publish-set
# branch that happens to drop nothing published would still be a dead limb reporting clean.
[[ -r h100/PUBLISH_SET.txt && "${#_pub_gen[@]}" -eq 0 ]] && {
  echo "  GENERATOR DENOMINATOR UNMEASURED: the publish set is readable but contributed 0 generators — the branch is dead, not the set empty" >&2; exit 95; }
while IFS= read -r _g; do GENERATORS+=("$_g"); done < <(
  { printf '%s\n' "${STAGES[@]}"; \
    [[ "${#_gate_files[@]}" -gt 0 ]] && printf '%s\n' "${_gate_files[@]}"; \
    printf '%s\n' build_h100_plane.sh extract_stage.py; \
    [[ "${#_pub_gen[@]}" -gt 0 ]] && printf '%s\n' "${_pub_gen[@]}"; \
    true; } | sort -u)

hits=0
for f in "${GENERATED[@]}"; do
  n=$(grep -cInEi "$PAT" "$f" || true)
  printf '  %-34s %s hit(s)\n' "$(basename "$f")" "$n"
  hits=$((hits + n))
done
echo "  --- build inputs (#151: generators are published too) ---"
# The generators are scanned with a DIFFERENT, narrower pattern, and the reason matters.
#
# A first attempt excluded lines containing the word BLOCKLIST, on the theory that a
# redaction list is vocabulary rather than disclosure (#145). That filter immediately
# excused this fixture in extract_stage.py:
#     BLOCKLIST = ("/work/<real-estate-segment>/public/weights",)
# which is a REAL estate path hidden inside a fake blocklist tuple -- the exact case that
# fixture exists to prove is still fatal. Excluding by keyword let the disclosure through
# by agreeing with it.
#
# The sound split is not by line shape but by TOKEN CLASS:
#   * A generic token -- `/work/`, a bare `ghp_` -- is vocabulary. A generator must be able
#     to name it to redact it, and naming it discloses nothing.
#   * An ESTATE IDENTIFIER -- the org segment, a private hostname, an account id -- is a
#     disclosure wherever it appears, INCLUDING inside a redaction list.
# So generators are scanned for identifiers only. Shipped artifacts keep the full pattern,
# because a generated artifact has no legitimate reason to contain either class.
IDENT_CORE="$FS_ESTATE_IDENT_PAT"
[[ "$FS_REDACT_EXTRA" == "NONE" ]] || IDENT_CORE="$IDENT_CORE|$FS_REDACT_EXTRA"
# #157: this scan used to apply a narrower PATH-ADJACENCY rule here -- an identifier counted
# only when it touched a `/`, so that an identifier sitting in an alternation between pipes
# read as vocabulary rather than as disclosure. That rule was forced by a real constraint: a
# generator that redacts these tokens had to CONTAIN them, so matching the bare token flagged
# every redactor for redacting.
#
# #155 removed the constraint. The whole identifier tier is now an environment input, so NO
# generator carries an identifier for any legitimate reason, and the exemption bought nothing
# while still excusing the case it had itself declared as a blind spot: an identifier in
# generator PROSE is not path-adjacent. Measured over the same 31 generators: adjacency 0
# hits, full pattern 3 -- two comments naming an identifier outright, and a MUST_FIRE drill
# that assembled its org segment and then wrote a node name literally on the same line.
#
# The exemption outlived its premise by one ticket. That is the general shape worth naming:
# a narrowing is only as sound as the constraint that forced it, and nothing re-derives the
# constraint when the code around it changes. So the rule is now the plain identifier tier,
# and the token tier stays exempt for generators -- that part of #144/#145 is untouched and
# still correct, because a generic token is vocabulary and an identifier never is.
IDENT_PAT="$IDENT_CORE"
# Four controls. The pattern is a second detector and needs its own proof of life in both
# directions. The control tokens are ASSEMBLED, not written: a literal estate identifier here
# would be a disclosure in this file and would be caught by this very loop -- a control that
# fails the check it certifies. Split across quotes so the source text cannot match while the
# runtime value does; the concatenation is the point, not obfuscation.
_ctl_id="hh""ri""-AI"
_ctl_node="dg""pn""04"
ictl=$(printf 'BLOCKLIST = ("/work/%s/x",)\n' "$_ctl_id" | grep -cInEi "$IDENT_PAT" || true)
[[ "$ictl" -eq 1 ]] || { echo "  GENERATOR CONTROL FAILED: identifier pattern missed an estate segment inside a redaction list ($ictl/1)" >&2; exit 5; }
ictl_neg=$(printf 'BLOCKLIST = ("/work/", "ghp_")\n' | grep -cInEi "$IDENT_PAT" || true)
[[ "$ictl_neg" -eq 0 ]] || { echo "  GENERATOR CONTROL FAILED: identifier pattern fired on pure vocabulary ($ictl_neg, expected 0) — it would pressure generators toward carrying no redaction list at all" >&2; exit 5; }
# The two #157 MUST_FIREs -- the cases the adjacency rule excused. Their expected value is
# the INVERSE of what the old control asserted, and that inversion is the fix: a bare
# alternation carrying an identifier is a disclosure now that no generator needs to carry one.
ictl_alt=$(printf 'r"a|%s|b"\n' "$_ctl_id" | grep -cInEi "$IDENT_PAT" || true)
[[ "$ictl_alt" -eq 1 ]] || { echo "  GENERATOR CONTROL FAILED: identifier in a bare alternation not caught ($ictl_alt, expected 1) — this is the #157 case: post-#155 no generator carries the vocabulary, so an identifier between pipes is a disclosure like any other" >&2; exit 5; }
ictl_prose=$(printf '# measured on %s during bring-up\n' "$_ctl_node" | grep -cInEi "$IDENT_PAT" || true)
[[ "$ictl_prose" -eq 1 ]] || { echo "  GENERATOR CONTROL FAILED: identifier in a comment not caught ($ictl_prose, expected 1) — prose in a published generator is published prose" >&2; exit 5; }
echo "  control: identifier in path 1/1, in alternation 1/1, in prose 1/1 caught; pure vocabulary 0 false-positive"
gen_hits=0; gen_files=0
for f in "${GENERATORS[@]}"; do
  [[ -f "$f" ]] || continue
  gen_files=$((gen_files + 1))
  n=$(grep -cInEi "$IDENT_PAT" "$f" 2>/dev/null || true)
  if [[ "$n" -gt 0 ]]; then printf '  %-34s %s hit(s)\n' "$(basename "$f")" "$n"; fi
  gen_hits=$((gen_hits + n))
done
printf '  %-34s %s file(s) scanned, %s hit(s)\n' "[generators]" "$gen_files" "$gen_hits"
[[ "$gen_files" -ge 20 ]] || { echo "  GENERATOR SCAN UNMEASURED: only $gen_files input(s) resolved (expected >=20) — a shrunken denominator reads exactly like a clean scan" >&2; exit 95; }
hits=$((hits + gen_hits))
# `|| true` before the pipe, not after it: `set -o pipefail` is on, and grep exits 1 when it
# matches nothing -- which is the CLEAN case here. Without this the build dies on success.
notices=$( { grep -lInEi "$NOTICE_PAT" "$LAUNCHER" "$BACKEND" "$ENTRY" "$MODELROOT" "$MRTEST" "$ADJ" "$ADJTEST" 2>/dev/null || true; } | wc -l | tr -d ' ')
[[ "$notices" -eq 0 ]] || {
  echo "  NOTICE (not fatal): secret vocabulary with no token body in $notices file(s):"
  grep -nIHEi "$NOTICE_PAT" "$LAUNCHER" "$BACKEND" "$ENTRY" "$MODELROOT" "$MRTEST" "$ADJ" "$ADJTEST" 2>/dev/null | sed 's/^/    /'
}
# The control: BOTH tiers must be able to match. A blocklist that matches nothing anywhere
# reports 0 for a dead regex just as loudly as for a clean file. The planted FATAL string is
# token-SHAPED (36 body chars, a real PAT's length) because the tightened pattern is exactly
# what distinguishes it from the planted NOTICE string, and a control that could not tell those
# two apart would certify the very confusion #144 is about.
probe=$(printf '%s\n' "ghp""_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789" | grep -cInEi "$PAT" || true)
[[ "$probe" -eq 1 ]] || { echo "  BLOCKLIST CONTROL FAILED: FATAL pattern matched a planted token-shaped string $probe/1 times" >&2; exit 5; }
probe_bare=$(printf 'ghp_\n' | grep -cInEi "$PAT" || true)
[[ "$probe_bare" -eq 0 ]] || { echo "  BLOCKLIST CONTROL FAILED: FATAL pattern still matches a bare prefix ($probe_bare) — #144 not actually fixed" >&2; exit 5; }
probe_notice=$(printf 'ghp_\n' | grep -cInEi "$NOTICE_PAT" || true)
[[ "$probe_notice" -eq 1 ]] || { echo "  BLOCKLIST CONTROL FAILED: NOTICE pattern missed a bare prefix $probe_notice/1 — the narrowing IS a blind spot" >&2; exit 5; }
echo "  control: token-shaped 1/1 fatal, bare prefix 0/1 fatal and 1/1 notice — both tiers live"

echo "  --- published documents (#151c: the third category) ---"
# The generator scan above declares a blind spot: a bare identifier in PROSE is not
# path-adjacent, so it slips through. That declaration was a promise with nothing behind it
# until this block existed -- a claim broader than its evidence, which is doctrine point 6
# turned on the build itself. Documents are where prose lives, so they are scanned with a
# THIRD pattern: identifiers with no adjacency requirement at all.
#
# Measured when this was written: EVIDENCE.md carried the node name in 3 places, the
# partition in 5, both account ids in 7, and phase3_spec.md carried a full model path.
# None of them were reachable by either scan above, because neither reads .md.
#
# `/work/` and `ghp_` stay TOKEN-CLASS here for the #145 reason, and this file is the proof
# that the distinction is load-bearing: EVIDENCE.md legitimately DESCRIBES the blocklist
# ("`/work/` plus the estate segment are both on the blocklist"). A pattern that could not
# tell that sentence from a real path would force the documentation to stop documenting.
DOC_PAT="$IDENT_CORE|/work/[A-Za-z0-9]|ghp_[A-Za-z0-9]{8}"
# #159: the partition literal used to sit in that pattern, written out inside a word-boundary
# escape -- `\b<PARTITION>\b` with the value where the placeholder is. Two defects in one
# four-character string, and they hid each other:
#   * DISCLOSURE. A published build script named the estate's Slurm partition -- the same
#     literal #152 spent a ticket removing from the launcher and #157 removed from three
#     patch anchors. FS_PARTITION_LITERAL was already the one oracle for it; this was a
#     fourteenth site nobody counted.
#   * A SCAN THAT COULD NOT SEE IT. FS_REDACT_EXTRA carries the token in word-boundary form,
#     and in the source text the character immediately before the token was the `b` of the
#     preceding `\b` escape -- so `\b` found no boundary there and the generator scan measured
#     0 hits on this file. The narrowing that made the pattern precise in prose made it blind
#     in source, and it was blind exactly where the token was written down.
# Same shape as #157: a narrowing outliving the constraint that justified it. Built from the
# environment now, and the bare-literal sweep below is the control that keeps it built.
#
# The sweep then fired a second time, on this comment: the first draft explained the defect by
# quoting the offending text verbatim, which re-published the value while describing why it
# must not be published. Unlike `ghp_`, a partition NAME is the secret itself and not a public
# prefix, so #144's use-vs-mention licence does not extend to it. Placeholder above; the value
# appears nowhere in this tree.
[[ "$FS_PARTITION_LITERAL" == "NONE" ]] || DOC_PAT="$DOC_PAT|\\b${FS_PARTITION_LITERAL}\\b"
DOCS=()
# #157: unioned with the publish set for the same reason as GENERATORS. `find -maxdepth 2`
# happens to reach every published .md today, which is precisely the kind of accident that
# stops being true the first time a doc lands one directory deeper.
#
# #158: `find h100 docs` exits 1 when `docs/` does not exist -- and it does not exist. Under
# `set -e`, inherited by the process-substitution subshell, that killed the brace group before
# the publish-set branch ran, so this union silently degraded to find-only and the new
# top-level README.md was never scanned. `2>/dev/null` hid the one line that would have said
# so. The coverage control at the end of the build is what caught it, which is the whole
# argument for keeping that control after deriving the denominator: derivation removes the
# class of defect, it does not remove the bugs in the derivation.
#
# Fixed by construction rather than by `|| true`: only existing roots are passed to find, so
# there is no failure to swallow. `|| true` would have worked here and would also have
# swallowed a genuine permission error, producing the same silent partial denominator.
_doc_roots=()
for _r in h100 docs; do [[ -d "$_r" ]] && _doc_roots+=("$_r"); done
_pub_doc=()
if [[ -r h100/PUBLISH_SET.txt ]]; then
  while IFS= read -r _p; do _pub_doc+=("$_p"); done < <(
    grep -vE '^[[:space:]]*(#|$)' h100/PUBLISH_SET.txt | sed 's|^\./||' | grep -E '\.md$' || true)
fi
[[ -r h100/PUBLISH_SET.txt && "${#_pub_doc[@]}" -eq 0 ]] && {
  echo "  DOCUMENT DENOMINATOR UNMEASURED: the publish set is readable but contributed 0 documents — the branch is dead, not the set empty" >&2; exit 95; }
while IFS= read -r _d; do [[ -f "$_d" ]] && DOCS+=("$_d"); done < <(
  { [[ "${#_doc_roots[@]}" -gt 0 ]] && find "${_doc_roots[@]}" -maxdepth 2 -name '*.md'
    [[ "${#_pub_doc[@]}" -gt 0 ]] && printf '%s\n' "${_pub_doc[@]}"
    true; } | sort -u)

# Four controls, because this pattern makes a finer distinction than the other two and the
# one that matters is the THIRD: a real path smuggled inside a sentence about redaction.
# Without it, "we allow prose about the blocklist" is indistinguishable from "we allow
# anything on a line that mentions the blocklist" -- the keyword-exclusion mistake again.
_d_org="HH""RI-AI"
# #157: the node name used to sit here as a literal, in the same printf whose org segment was
# carefully assembled. Half-assembling a control is worse than not assembling it, because the
# surrounding care reads as evidence that the line was checked. Both tokens are assembled now,
# and the generator scan above is what would have caught this had its adjacency rule not
# excused a token that touches a space instead of a slash.
dmf1=$(printf 'ran on %s under /work/%s/POC\n' "$_ctl_node" "$_d_org" | grep -cInEi "$DOC_PAT" || true)
dmf2=$(printf 'token %s is live\n' "ghp""_AbCdEfGhIjKlMnOp" | grep -cInEi "$DOC_PAT" || true)
dmf3=$(printf 'the blocklist covers /work/ but /work/%s/x slipped in\n' "$_d_org" | grep -cInEi "$DOC_PAT" || true)
dmp1=$(printf 'the blocklist covers /work/ and the ghp_ prefix\n' | grep -cInEi "$DOC_PAT" || true)
dmp2=$(printf 'run on <h100-node> under <estate-root> as <primary-account>\n' | grep -cInEi "$DOC_PAT" || true)
[[ "$dmf1" -eq 1 && "$dmf2" -eq 1 ]] || { echo "  DOC CONTROL FAILED: pattern missed an estate path ($dmf1/1) or a token ($dmf2/1)" >&2; exit 5; }
[[ "$dmf3" -eq 1 ]] || { echo "  DOC CONTROL FAILED: a real estate path inside redaction prose was NOT caught ($dmf3/1) — the token-class split has become a keyword exemption" >&2; exit 5; }
[[ "$dmp1" -eq 0 ]] || { echo "  DOC CONTROL FAILED: fired on prose describing the blocklist ($dmp1, expected 0) — the docs could not document the gate" >&2; exit 5; }
[[ "$dmp2" -eq 0 ]] || { echo "  DOC CONTROL FAILED: fired on a fully-redacted line ($dmp2, expected 0)" >&2; exit 5; }
echo "  control: estate path 1/1, token 1/1, path-inside-redaction-prose 1/1; redaction prose 0 and redacted line 0 false-positive"

doc_hits=0; doc_files=0
for f in "${DOCS[@]}"; do
  doc_files=$((doc_files + 1))
  n=$(grep -cInEi "$DOC_PAT" "$f" 2>/dev/null || true)
  if [[ "$n" -gt 0 ]]; then printf '  %-34s %s hit(s)\n' "$f" "$n"; fi
  doc_hits=$((doc_hits + n))
done
printf '  %-34s %s file(s) scanned, %s hit(s)\n' "[documents]" "$doc_files" "$doc_hits"
# Same denominator floor as the generator scan, same reason: `find` returning nothing reads
# exactly like a clean estate. This is the shape of the vacuous-truth failure in a scanner.
[[ "$doc_files" -ge 4 ]] || { echo "  DOCUMENT SCAN UNMEASURED: only $doc_files document(s) resolved (expected >=4) — a shrunken denominator reads exactly like a clean scan" >&2; exit 95; }
hits=$((hits + doc_hits))

# One oracle for "what ships", used by the sweep below and by the coverage gate after it.
PUBSET="h100/PUBLISH_SET.txt"

echo "  --- partition literal, fixed-string (#159) ---"
# A fourth scan, deliberately not a regex. The three above all express the partition literal
# through a pattern, and #159 is the case where the pattern was written in a form that could
# not match its own source text. A regex that names a token can be wrong about that token in a
# way a fixed-string search cannot, so this one uses `grep -F` on the value itself and sweeps
# the PUBLISH SET directly -- not a category, the shipment.
#
# It is also the check that keeps DOC_PAT built from the environment: re-hard-coding the
# literal anywhere in the tree turns this red immediately, whatever escape form it is written
# in. Construction over detection where possible; where the fix is "keep using the variable",
# a control on the value is what makes that stick.
if [[ "$FS_PARTITION_LITERAL" == "NONE" ]]; then
  echo "  [partition literal]                declared NONE — nothing to sweep for"
else
  plit_files=0; plit_hits=0; plit_named=""
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    plit_files=$((plit_files + 1))
    n=$(grep -cF "$FS_PARTITION_LITERAL" "$f" 2>/dev/null || true)
    if [[ "$n" -gt 0 ]]; then plit_named="$plit_named
    $f ($n)"; fi
    plit_hits=$((plit_hits + n))
  done < <(grep -vE '^[[:space:]]*(#|$)' "$PUBSET" 2>/dev/null | sed 's|^\./||' || true)
  printf '  %-34s %s file(s) swept, %s hit(s)\n' "[partition literal]" "$plit_files" "$plit_hits"
  [[ "$plit_files" -ge 20 ]] || { echo "  PARTITION SWEEP UNMEASURED: only $plit_files published file(s) resolved (expected >=20)" >&2; exit 95; }
  # MUST_FIRE: prove grep -F can see the value at all before believing a zero. The needle is
  # written to a temp file rather than assembled inline, because assembling it here would put
  # the literal back in this file and the sweep would (correctly) fail on itself.
  _plit_probe=$(mktemp); printf 'partition=%s\n' "$FS_PARTITION_LITERAL" > "$_plit_probe"
  _plit_ctl=$(grep -cF "$FS_PARTITION_LITERAL" "$_plit_probe" || true); rm -f "$_plit_probe"
  [[ "$_plit_ctl" -eq 1 ]] || { echo "  PARTITION CONTROL FAILED: fixed-string sweep missed a planted literal ($_plit_ctl/1)" >&2; exit 5; }
  echo "  control: planted literal 1/1 caught"
  [[ "$plit_hits" -eq 0 ]] || { echo "  PARTITION LITERAL PUBLISHED: the estate's partition name appears verbatim in:$plit_named
    Build it from \$FS_PARTITION_LITERAL instead — see #152, #157, #159." >&2; exit 5; }
fi

echo "  --- build-host home paths, fixed-shape (#162) ---"
# Every scan above is tuned for ESTATE identifiers. None of them says anything about the
# BUILD HOST, so `/Users/<someone>/fs-repo/...` hard-coded in a published stage passed all
# four and reached the pre-push gate, which is not itself published and so is not a control
# anyone else inherits. Same class as #123/#151 (hard-coded root), one layer down.
# The character class deliberately excludes a leading dot so the elided form `/home/.../x`
# -- which is how a path is CITED without being disclosed -- is not a hit.
HOME_PAT='/(Users|home)/[A-Za-z0-9_-][A-Za-z0-9._-]*/'
home_files=0; home_hits=0; home_named=""
while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  home_files=$((home_files + 1))
  n=$(grep -cE "$HOME_PAT" "$f" 2>/dev/null || true)
  if [[ "$n" -gt 0 ]]; then home_named="$home_named
    $f ($n)"; fi
  home_hits=$((home_hits + n))
done < <(grep -vE '^[[:space:]]*(#|$)' "$PUBSET" 2>/dev/null | sed 's|^\./||' || true)
printf '  %-34s %s file(s) swept, %s hit(s)\n' "[build-host home]" "$home_files" "$home_hits"
[[ "$home_files" -ge 20 ]] || { echo "  HOME SWEEP UNMEASURED: only $home_files file(s) resolved" >&2; exit 95; }
_home_probe=$(mktemp)
# The needle is ASSEMBLED, never written literally: this file is itself in the swept set,
# so writing a home-rooted absolute path out in full would make the sweep correctly red on
# the sweep. Two drafts of THIS comment hit, each while warning against exactly that -- a
# placeholder is no help if the placeholder still matches the pattern. #159 learned the same
# lesson with the partition literal; it recurs for every fixed-shape probe.
{ printf 'SRC = "/%s/%s/fs-repo/x.sh"\n' Users somebody
  printf 'cited /%s/.../probe_sub/x.sh\n' home; } > "$_home_probe"
_home_ctl=$(grep -cE "$HOME_PAT" "$_home_probe" || true); rm -f "$_home_probe"
[[ "$_home_ctl" -eq 1 ]] || {
  echo "  HOME CONTROL FAILED: planted path + elided citation should give exactly 1 hit ($_home_ctl/1)" >&2
  exit 5; }
echo "  control: planted home path 1/1 caught, elided citation 0/1 (not a disclosure)"
[[ "$home_hits" -eq 0 ]] || { echo "  BUILD-HOST PATH PUBLISHED:$home_named
  Resolve it from \$FS_UPSTREAM_REPO or from \$(dirname \$0) instead — see #162." >&2; exit 5; }

echo "  --- publish-set coverage (#157: a scan is only as wide as its denominator) ---"
# Three scans ran above, each with its own honest denominator, each reporting clean. The
# question none of them can answer is the one that matters at push time: is every file we
# PUBLISH inside one of those three denominators?
#
# It was not. h100/upstream/*.sh and *.py were in the publish set and in no category --
# GENERATED lists seven named artifacts, GENERATORS globs the stages and gates, DOCUMENTS is
# `find h100 docs -name '*.md'`. Nothing owned the rest, so four estate identifiers shipped
# through three consecutive clean scans. The two sets were maintained independently and
# never compared, which is #151 and #155 one level up: those were "the scan missed a file",
# this is "the scan and the shipment disagree about what the files ARE".
#
# So the publish set is a declared artifact in the tree, and coverage is computed against it.
# An unowned published file is UNMEASURED, not clean, and is named in the refusal -- a count
# would send the reader to diff two lists by hand.
if [[ -r "$PUBSET" ]]; then
  _cov_tmp=$(mktemp); _cov_pub=$(mktemp); _cov_un=$(mktemp)
  printf '%s\n' "${GENERATED[@]}" "${GENERATORS[@]}" "${DOCS[@]}" | sed 's|^\./||' | sort -u > "$_cov_tmp"
  grep -vE '^\s*(#|$)' "$PUBSET" | sed 's|^\./||' | sort -u > "$_cov_pub"
  _pub_n=$(wc -l < "$_cov_pub" | tr -d ' ')
  [[ "$_pub_n" -ge 20 ]] || { echo "  COVERAGE UNMEASURED: publish set lists only $_pub_n file(s) (expected >=20) — an empty manifest covers vacuously" >&2; exit 95; }
  # A manifest naming a file that does not exist is its own defect: it inflates the
  # denominator with something no scan could ever have read.
  _missing=0
  while IFS= read -r _p; do [[ -e "$_p" ]] || { echo "  COVERAGE: publish set names a file that does not exist: $_p" >&2; _missing=$((_missing+1)); }; done < "$_cov_pub"
  [[ "$_missing" -eq 0 ]] || { echo "  COVERAGE RED: $_missing published file(s) absent from the tree" >&2; exit 5; }
  comm -23 "$_cov_pub" "$_cov_tmp" > "$_cov_un"
  _un_n=$(wc -l < "$_cov_un" | tr -d ' ')
  # MUST_FIRE: an unowned file must be observed producing the refusal, or this gate is a
  # print statement. Synthesised into the comparison, never onto disk.
  _ctl_un=$(printf 'h100/upstream/__control__.sh\n' | comm -23 - "$_cov_tmp" | wc -l | tr -d ' ')
  [[ "$_ctl_un" -eq 1 ]] || { echo "  COVERAGE CONTROL FAILED: a file in no category was not reported unowned ($_ctl_un, expected 1)" >&2; exit 5; }
  printf '  %-34s %s published, %s covered, %s unowned  (control: unowned file detected 1/1)\n' \
    "[publish-set coverage]" "$_pub_n" "$((_pub_n - _un_n))" "$_un_n"
  if [[ "$_un_n" -gt 0 ]]; then
    echo "  COVERAGE UNMEASURED: the following published file(s) are in no scan category —" >&2
    sed 's/^/      /' "$_cov_un" >&2
    echo "  A file can only be reported clean by a scan that read it. Add it to a category or remove it from the publish set." >&2
    rm -f "$_cov_tmp" "$_cov_pub" "$_cov_un"; exit 95
  fi
  rm -f "$_cov_tmp" "$_cov_pub" "$_cov_un"
else
  echo "  COVERAGE UNMEASURED: $PUBSET absent — the set of published files is undeclared, so no scan can claim to cover it" >&2
  exit 95
fi

[[ "$hits" -eq 0 ]] || { echo "BLOCKLIST RED: $hits hit(s)" >&2; exit 5; }

echo -e "\n=== parse ==="
# Two shell artifacts and one Python one, each checked with its own language's parser.
# A single loop with `bash -n` would report the .py file as clean-because-unchecked.
ok=0
for f in "$LAUNCHER" "$BACKEND"; do
  bash -n "$f" && { printf '  clean  %-32s (bash -n)\n' "$(basename "$f")"; ok=$((ok + 1)); }
done
for f in "$ENTRY" "$MODELROOT" "$MRTEST" "$ADJ" "$ADJTEST"; do
  python3 -m py_compile "$f" \
    && { printf '  clean  %-32s (py_compile)\n' "$(basename "$f")"; ok=$((ok + 1)); }
done
[[ $ok -eq 7 ]] || { echo "SYNTAX RED: $ok/7 clean" >&2; exit 5; }

echo -e "\n=== generated unit suites (#133 model-root plane, #141 adjudicator) ==="
# A generated test suite that the build does not run is an orphan -- #86 was exactly that,
# eight passing legs nobody executed. So this runs, and a missing pytest is an UNMEASURED
# that FAILS the build rather than a skip that reads like a pass. FS_SKIP_SUITE=1 waives it,
# but the waiver has to be said out loud and is printed in the summary.
#
# #160: this branch printed UNMEASURED and then exited 5. Both codes fail the build, so the
# defect was invisible to anyone running it by hand -- and it was found by running the build in
# the PUBLISHED tree, which has no `.venv` because a virtualenv is not a publishable artifact.
# The whole point of a distinct 95 is that a consumer can tell "a gate found a defect" from
# "a gate could not run"; collapsing them re-merges exactly the two states this build spent
# #56 and #149 separating. It also made the README false at the moment it shipped -- that doc
# states this gate declares 95 -- which is doctrine point 6 with the roles reversed: not a
# claim broader than its evidence, but a claim the code had quietly stopped honouring.
PY=${FS_PYTEST:-./.venv/bin/python}
if [[ "${FS_SKIP_SUITE:-0}" == 1 ]]; then
  suite="WAIVED (FS_SKIP_SUITE=1)"
  echo "  $suite — the generated suite was NOT run"
elif [[ -x "$PY" ]] && "$PY" -c 'import pytest' 2>/dev/null; then
  "$PY" -m pytest "$MRTEST" "$ADJTEST" -q || { echo "SUITE RED" >&2; exit 5; }
  suite="$("$PY" -m pytest "$MRTEST" "$ADJTEST" -q 2>/dev/null | tail -1)"
else
  echo "  UNMEASURED (95): no pytest at '$PY'. Set FS_PYTEST to an interpreter that has it," >&2
  echo "  or FS_SKIP_SUITE=1 to waive explicitly. Not running a suite is not passing it." >&2
  exit 95
fi

echo -e "\n=== standing gate: writer/adjudicator naming agreement (#150) ==="
# Runs AFTER the suites, deliberately. #150 was invisible to both suites: the writer and the
# adjudicator are emitted by two different stages, each passes its own tests, and they
# disagreed with each other -- 31 green tests while leg A7b abstained on 2/2 of the
# checkpoint shapes this framework actually produces. A cross-artifact contract needs a
# cross-artifact gate; per-artifact suites structurally cannot see it.
# #160, second site: this case arm translated the gate's own four-state result into a single
# `exit 5`, so an UNMEASURED gate reported as a RED build. The gate goes to the trouble of
# distinguishing "I could not measure" (3) from "we disagree" (5); flattening that at the call
# site throws away the only thing the distinction was for. Controls-failed (4) stays RED on
# purpose -- a gate that fails its own controls is not unmeasured, it is untrustworthy, and the
# build must not offer the softer word for it.
python3 gate_ckpt_naming_agreement.py || {
  rc=$?
  case $rc in
    3) echo "NAMING GATE UNMEASURED (rc=3 -> build 95) — zero writer sites, or the adjudicator would not import" >&2; exit 95 ;;
    4) echo "NAMING GATE CONTROLS FAILED (rc=4) — the gate cannot be trusted, so neither can this build" >&2 ;;
    5) echo "NAMING GATE RED (rc=5) — writer and adjudicator disagree about checkpoint naming" >&2 ;;
    *) echo "NAMING GATE unexpected rc=$rc" >&2 ;;
  esac
  exit 5
}

echo -e "\n=== standing gate: inter-artifact linkage (#142) ==="
# The generalisation of #142: every literal filename one shipped artifact reaches for must be
# a filename this build actually produces. #142 was one instance (a launcher sourcing
# fs_container_backend.sh, which no stage emits); the gate asks the question of every edge, so
# the next instance is caught the day it is introduced rather than on a cluster. Variable-name
# edges are counted as UNRESOLVABLE and explicitly NOT judged -- an edge a static reader cannot
# follow is unmeasured, and unmeasured is not clean.
python3 gate_artifact_linkage.py || {
  rc=$?
  case $rc in
    5)  echo "LINKAGE GATE RED (rc=5) — a shipped artifact reaches for a file the build never produces" >&2 ;;
    95) echo "LINKAGE GATE UNMEASURED (rc=95) — zero inter-artifact edges resolved; a shrunken denominator is not a clean scan" >&2 ;;
    *)  echo "LINKAGE GATE unexpected rc=$rc" >&2 ;;
  esac
  exit 5
}

# #154: the operator document cites the launcher by line number, and the launcher grows.
# Measured when this gate was written: 19 of 19 citations had rotted, all of them pushed
# down by #142's resolver, and FS_PARTITION -- required, no default, refuses at L28 --
# appeared 18x in the launcher and 0x in the document. Both defects are invisible to
# every other gate here: the document is not shell, so it parses clean, and a wrong line
# number is still a number. A doc is a claim about the code; unchecked, it is the one
# artifact that can rot to 100% wrong while the build stays green.
echo -e "\n=== standing gate: operator document vs launcher (#154) ==="
python3 gate_launch_doc.py || {
  rc=$?
  case $rc in
    5)  echo "LAUNCH DOC GATE RED (rc=5) — the document names a knob the launcher does not enforce, cites a line that does not support it, or carries an estate literal" >&2 ;;
    95) echo "LAUNCH DOC GATE UNMEASURED (rc=95) — a zero denominator or an unplantable drill; an uncertified detector is not a clean scan" >&2 ;;
    96) echo "LAUNCH DOC GATE REFUSED (rc=96) — an input is unreadable or a required redaction pattern is unset" >&2 ;;
    *)  echo "LAUNCH DOC GATE unexpected rc=$rc" >&2 ;;
  esac
  exit 5
}

printf '\nBUILD GREEN — %s stages, drift gate green, %s blocklist hits, %s/7 parse clean, naming agreement green, linkage green, launch-doc green, suite: %s\n' \
  "${#STAGES[@]}" "$hits" "$ok" "$suite"
for f in "$LAUNCHER" "$BACKEND" "$ENTRY" "$MODELROOT" "$MRTEST" "$ADJ" "$ADJTEST"; do
  printf '  %-34s %s lines\n' "$f" "$(wc -l < "$f" | tr -d ' ')"
done
