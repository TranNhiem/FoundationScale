#!/usr/bin/env python3
"""tools/live_save_gate.py -- adjudicate a checkpoint written by a LIVE training run.

Why this tool exists
--------------------
``tools/real_checkpoint_probe.py`` proved the gates meet real, static artifacts
honestly: it refused to invent denominators, saved VACUOUS rather than faked a
completeness PASS, and documented every abstention. A production run needs one
step further: a *decision*, returned to the launcher, cheap enough to run inside
the job seconds after a save, on both artifact shapes this estate produces:

  * FULL-FT checkpoint   -- full trainable tensor population (Megatron DCP *or*
                            HF safetensors layout, depending on stage).
  * LoRA adapter         -- a *tiny* population; the base weights are frozen and
                            absent BY DESIGN. Adjudicating it against the full
                            model's denominator is the canonical false alarm
                            that gets verification tooling uninstalled.

Provenance discipline (the core rule, unchanged from the probe)
---------------------------------------------------------------
Every denominator comes from a source the run under judgment did not write:

  denominator                source (read fresh, every call)
  -------------------------  --------------------------------------------------
  run_kind (full|lora)       --train-config (launcher-resolved config, or --set)
  num_experts, num_moe_layers base config.json, read by the probe's
                              derive_declared (text_config scope first), under
                              its two-source contract: 0 is minted ONLY from an
                              affirmative dense statement CORROBORATED by an
                              expert-family census over the UNFILTERED base
                              header (MINT_ZERO_ONLY_IN_PROBE; see
                              derive_declared_block). Measured on the estate,
                              discharging the UNVERIFIED marker this row
                              replaces: the E4B config.json carries
                              text_config.enable_moe_block = False with
                              text_config.num_experts present-but-null, and its
                              headers hold zero expert-family names -- so the
                              census reads 0 and the mint is satisfiable on the
                              real artifact.
  declared_fqns (full)       ONE value, TWO mutually exclusive bases; which
                             one applies is MEASURED per save, never assumed
                             per row. (a) base model.safetensors *header key
                             set* (2,130 tensors per estate measurement --
                             re-read, never trusted from this comment), minus
                             documented frozen scope -- applicable ONLY when
                             the save on disk IS the HF layout, which the
                             gate verifies, never assumes: overlap with this
                             key set must reach 0.90 of the on-disk artifact
                             FQNs, else this basis is refused with the
                             refusal stated (declared_fqns stays None and
                             save_complete VACUOUS-blocks) rather than
                             compare against a guessed mapping.
  declared_fqns (full, DCP)  (b) --fqn-map: a JSON list of ARTIFACT-namespace
                             FQNs -- the ONLY completeness basis when the
                             artifact layout is not the HF namespace (the
                             estate's Megatron/DCP saves). Censused from the
                             INDEPENDENT base before the run, read back by
                             the launcher from the emitter's attempt record,
                             never from the run under judgment; and since
                             the #78 re-scope its namespace is MEASURED AT
                             SUBMIT by this gate's own 0.90 discriminator in
                             reverse (the census overlapped against the
                             model*.safetensors header key set under
                             $HF_MODEL -- >=0.90 means an HF-layout base
                             resolved in and the submit refuses; a partial
                             overlap is a stated abstention and also
                             refuses). The label is now a measured property
                             with a control that was observed firing
                             (doctrine 3), not the producer's say-so; still
                             the ONLY way completeness is measured when the
                             artifact layout is not the HF namespace.
  declared_fqns (lora)       --adapter-modules: the launch-time live-module
                             census -- the module population the trainer's own
                             matcher attaches to, censused from the INDEPENDENT
                             base tree before the run starts, in the artifact's
                             own namespace. (#78: the pre-census basis -- HF
                             base header x targets x rank -- was MEASURED
                             disjoint from every save this estate can produce:
                             on a healthy PROBE save on <compute-node> it fell to 0
                             declared, save_complete went VACUOUS and the drop
                             control was unconstructable -- the founding
                             all([]) shape at the oracle layer.) The census
                             file is refused if it resolves inside the judged
                             tree, and its residual shared-code fate (census
                             and run share the trainer's matcher) is named in
                             every report's fqns_basis rather than hidden.
                             Shapes are declared only when the census carries
                             parent dims AND the config resolves a rank;
                             otherwise shape-checking abstains by name.
  expected_expert_bytes      sum over expert-family keys in the BASE header
                             (layout-invariant byte total; survives the DCP
                             FQN rename that defeats declared_fqns)

The checkpoint's own metadata and the run's own manifest are NEVER denominator
sources. The decision code this claim is about now lives in
``foundationscale.gates.adjudication``, which calls neither ``load_manifest``
nor ``CheckpointGateContext.from_path``: it builds its context from
denominators resolved independently of the artifact under judgement. Scoping
the claim to THIS file would make it true by construction -- the file is an
argparse wrapper and has no denominator logic left to constrain.

The library gate does not share that policy, and the difference is load-bearing
(#218). ``CheckpointGateContext.from_path(path)`` without an explicit
``declared=`` takes its four denominators off the manifest sitting beside the
checkpoint, and ``_coerce`` routes any bare path there -- so ``gate.run(path)``
can judge a save against a declaration that save wrote. The launchers avoid it
by deriving the declaration from an independent base checkpoint, but that is
their discipline, not the library's guarantee.

Exit codes
----------
  0  CLEAR          -- every applicable gate reached a real verdict, AND every
                       constructable MUST_FIRE control FIRED on content copied
                       from this artifact.
  1  BLOCKED        -- a blocking verdict, a control that failed to fire, an
                       exercised control that came back INCONCLUSIVE (the
                       baseline already blocked so the injection cannot be
                       attributed, no baseline existed, or the detector answered
                       the injection with ERROR/a coverage verdict -- a
                       malfunction, not a detection), a control status this
                       loop does not recognize, or a control that could not be
                       CONSTRUCTED on this artifact
                       (an unexercised detector proves nothing; that is a
                       blocking reason here, not a footnote -- except where the
                       control's *claim itself* is absent, e.g. expert aliasing
                       on a genuinely dense artifact, which is stated as
                       "inapplicable" and covered by the universal drop control).
  3  UNMEASURED     -- the tool could not measure (unreadable artifact, missing
                       base model files, unresolvable run mode, tool bug), or the
                       operator's adapter-naming knobs disagreed with each other
                       and adjudication was refused BEFORE any verdict existed
                       (a mis-wired tool is not a checkpoint verdict, and the
                       launcher's retry policy turns on the 1-vs-3 difference).
                       Never surfaced as CLEAR.

Library use (this is the callable the launcher / hooks invoke):

    decision = adjudicate_checkpoint(
        ckpt_dir,
        event="first_save",
        run_kind="auto",
        base_model_dir=BASE, train_config_path=RESOLVED_CFG,
    )
    decision.raise_if_blocking()   # or inspect decision.exit_code

CLI:
    python tools/live_save_gate.py CKPT_DIR --event first_save \\
        --base-model-dir $HF_MODEL --train-config resolved-train-config.json \\
        --json $OUT_DIR/fs_gate/report-first-save.json
"""

from __future__ import annotations

# The CLI wrapper's own imports. Everything else this file needs is
# re-exported from foundationscale.gates.adjudication below.
import argparse
import sys
import traceback

# ---------------------------------------------------------------------------
# T2_lib_script_boundary#0: the decision API moved into the package.
#
# Everything below used to be defined HERE, in a script, which meant the only
# supported way to adjudicate a checkpoint was to shell out to this CLI. It now
# lives in foundationscale.gates.adjudication and is re-exported so that every
# existing reference -- tests doing `lsg.derive_declared_block(...)`, the
# mutation table's anchors, the launchers' call sites -- resolves unchanged.
#
# CAUTION for test authors: re-export binds a NAME, not the defining module's
# globals. Monkeypatching `live_save_gate._probe_derive_declared` does NOT change
# what the moved functions read, because they read their own module's globals.
# Patch `foundationscale.gates.adjudication` instead. The suites that do this
# were migrated with the move; see tests/test_live_save_gate.py.
#
# The surface is 94 names, and it used to be 98. Finding #219 inverted the
# probe dependency -- the declaration machinery moved INTO the package as
# foundationscale.gates.probe, so the try/except ImportError ladder that used
# to reach sideways into tools/real_checkpoint_probe.py is gone. Four names
# went with it: `_PROBE_IMPORT_ERROR` (the sentinel), `_DeriveDeclaredFn` and
# `_AliasControlFn` (the Protocols that typed the slots the ladder filled), and
# `Protocol` itself (re-exported only because those two needed it). They are
# not deprecated aliases kept for a release; they name machinery that no longer
# exists, and re-exporting a name for a thing that is gone would advertise a
# compatibility this file cannot provide. The shrink is deliberate, and it is
# pinned in both directions rather than by a count: tests/test_reexport_surface.py
# derives the library's unconditional surface from its AST and fails if any of
# it is missing here, while re-listing a name the library no longer defines
# fails harder still -- as an ImportError at the top of this file, which is how
# the four were found.
# ---------------------------------------------------------------------------
from foundationscale.gates.adjudication import (
    _ALIAS_STORAGE_ID,  # noqa: F401
    _ALWAYS_GATES,  # noqa: F401
    _CANONICAL_LORA_LAYOUT_RE,  # noqa: F401
    _CONTROL_BUILDERS,  # noqa: F401
    _DEFAULT_ADAPTER_SUFFIX_A,  # noqa: F401
    _DEFAULT_ADAPTER_SUFFIX_B,  # noqa: F401
    _DEFAULT_ADAPTER_SUFFIX_RE,  # noqa: F401
    _DEFAULT_ADAPTER_SUFFIXES,  # noqa: F401
    _DTYPE_BYTES,  # noqa: F401
    _EXPECTED_INTERPRETER_ENV,  # noqa: F401
    _FREEZE_KEYS,  # noqa: F401
    _HF_PEFT_ADAPTER_SUFFIX_A,  # noqa: F401
    _HF_PEFT_ADAPTER_SUFFIX_B,  # noqa: F401
    _HF_PEFT_ADAPTER_SUFFIX_RE,  # noqa: F401
    _HF_PEFT_ADAPTER_SUFFIXES,  # noqa: F401
    _KIND_KEYS,  # noqa: F401
    _MEGATRON_BRIDGE_ADAPTER_SUFFIX_A,  # noqa: F401
    _MEGATRON_BRIDGE_ADAPTER_SUFFIX_B,  # noqa: F401
    _MEGATRON_BRIDGE_ADAPTER_SUFFIX_RE,  # noqa: F401
    _MEGATRON_BRIDGE_ADAPTER_SUFFIXES,  # noqa: F401
    _NON_ADAPTER_CHECKPOINT_NAMESPACE_ROOTS,  # noqa: F401
    _RANK_KEYS,  # noqa: F401
    _REFUSAL_ADAPTER_CENSUS_UNAVAILABLE,  # noqa: F401
    _REFUSAL_ADAPTER_PREFIX_UNPINNED,  # noqa: F401
    _REFUSAL_CHECKPOINT_UNREADABLE,  # noqa: F401
    _REFUSAL_OTHER,  # noqa: F401
    _SELF_MOUNTED_ADAPTER_LAYOUT_RE,  # noqa: F401
    _SHARD_SUFFIX_RE,  # noqa: F401
    _ST_DTYPE,  # noqa: F401
    _TARGET_KEYS,  # noqa: F401
    EXIT_BLOCKED,  # noqa: F401
    EXIT_CLEAR,  # noqa: F401
    EXIT_UNMEASURED,  # noqa: F401
    Any,  # noqa: F401
    BaseModel,  # noqa: F401
    CheckpointFormatError,  # noqa: F401
    CheckpointGateContext,  # noqa: F401
    CheckpointMetadata,  # noqa: F401
    Declared,  # noqa: F401
    DeclaredBasis,  # noqa: F401
    ExpertByteVolumeGate,  # noqa: F401
    ExpertDistinctnessGate,  # noqa: F401
    FirstSaveGate,  # noqa: F401
    Gate,  # noqa: F401
    GateDecision,  # noqa: F401
    GateResult,  # noqa: F401
    GateUnmeasured,  # noqa: F401
    Path,  # noqa: F401
    SaveCompletenessGate,  # noqa: F401
    TensorMeta,  # noqa: F401
    TrainSpec,  # noqa: F401
    TypedDict,  # noqa: F401
    Verdict,  # noqa: F401
    _AdapterModuleCensus,  # noqa: F401
    _attributed_status,  # noqa: F401
    _context,  # noqa: F401
    _expert_named,  # noqa: F401
    _expert_weight_candidates,  # noqa: F401
    _expert_weights,  # noqa: F401
    _first_key,  # noqa: F401
    _infer_auto_kind,  # noqa: F401
    _interpreter_provenance,  # noqa: F401
    _interpreter_report_entry,  # noqa: F401
    _is_non_adapter_namespace,  # noqa: F401
    _layer_normalized_stem,  # noqa: F401
    _load_adapter_modules,  # noqa: F401
    _load_fqn_map,  # noqa: F401
    _load_train_config,  # noqa: F401
    _lora_target_attaches,  # noqa: F401
    _matches_expert_family,  # noqa: F401
    _measure,  # noqa: F401
    _probe_alias_control,  # noqa: F401
    _probe_derive_declared,  # noqa: F401
    _read_json,  # noqa: F401
    _read_safetensors_header,  # noqa: F401
    _real,  # noqa: F401
    _record_refusal,  # noqa: F401
    _refusal_class,  # noqa: F401
    _refusal_interpreter_entry,  # noqa: F401
    _refuse_on_interpreter_mismatch,  # noqa: F401
    _resolve_expected_interpreter,  # noqa: F401
    _split_expert_layouts,  # noqa: F401
    _verify_adapter_naming_agreement,  # noqa: F401
    adjudicate_checkpoint,  # noqa: F401
    control_alias,  # noqa: F401
    control_drop,  # noqa: F401
    control_underfill,  # noqa: F401
    cross_check_population,  # noqa: F401
    dataclass,  # noqa: F401
    derive_declared_block,  # noqa: F401
    field,  # noqa: F401
    lora_structural_findings,  # noqa: F401
    read_metadata,  # noqa: F401
    resolve_train_spec,  # noqa: F401
)


def _h(title: str) -> None:
    print(f"\n== {title} " + "=" * max(4, 68 - len(title)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="live_save_gate",
        description="Adjudicate a checkpoint written by a live training run. "
        "Exit 0=CLEAR, 1=BLOCKED (a verdict, a quiet control, or an "
        "unconstructable control), 3=could not measure.",
    )
    p.add_argument("ckpt_dir")
    p.add_argument("--event", choices=("save", "first_save"), default="save")
    p.add_argument("--run-kind", choices=("auto", "full", "lora"), default="auto")
    p.add_argument(
        "--base-model-dir",
        default=None,
        help="default: $HF_MODEL, then the estate Gemma-4 E4B path",
    )
    p.add_argument(
        "--train-config", default=None, help="launcher-RESOLVED config (JSON or KEY=VALUE dump)"
    )
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="train-config overrides (repeatable)",
    )
    p.add_argument("--controls", default="drop,alias,underfill")
    p.add_argument("--adapter-marker", default=r"(?:lora_[AB]|adapter)")
    p.add_argument(
        "--adapter-prefix",
        default=None,
        help="adapter FQN prefix: a CONSTANT LEADING SEGMENT an "
        "adapter save carries before the base-module stem WITHIN "
        "ONE naming convention (an HF-PEFT 'base_model.model.'-"
        "style wrapper root), DISTINCT from the suffix naming "
        "calibrated below. REAL USE, narrowed by measurement "
        "(#78): this knob repairs an in-namespace wrapper segment "
        "ONLY. It cannot bridge an HF-vs-Megatron namespace split "
        "-- measured on <compute-node> against a real, healthy PROBE "
        "save: pinning '' still produced BLOCKED with 0 declared, "
        "because the save's module names live in a different "
        "NAMESPACE than the HF base header the old oracle "
        "consulted; no prefix value makes those intersect. The "
        "old recipe ('run once with '' asserted, read the phantom "
        "stems, pin that segment') is RETIRED for this estate -- "
        "valid only where saves and the declared source share one "
        "namespace. For a namespace split the denominator is "
        "--adapter-modules (default REFUSAL, exit 3). For lora "
        "the default remains REFUSAL rather than a silently "
        "guessed empty prefix; '' is asserted deliberately. "
        "Full-FT adjudication never consults this knob.",
    )
    p.add_argument(
        "--adapter-suffix",
        default=_DEFAULT_ADAPTER_SUFFIX_RE,
        help="regex RECOGNIZING adapter FQN suffixes for the "
        "structural binding sweep. Default (fix30): the measured "
        "Megatron-Bridge shape, matching BOTH --adapter-suffix-a/-b "
        "literals below; the HF PEFT convention remains available "
        "for explicit pinning as the _HF_PEFT_ADAPTER_* preset. "
        "Any replacement must match the generator literals "
        "exactly: that agreement is verified at startup and any "
        "disagreement refuses adjudication before a verdict "
        "exists. Calibrate ONCE against one saved adapter -- "
        "together with the generator templates -- then pin it in "
        "the wrapper script.",
    )
    p.add_argument(
        "--adapter-suffix-a",
        default=_DEFAULT_ADAPTER_SUFFIX_A,
        help="literal template GENERATING the expected FQN suffix "
        "of the (rank, in_features) adapter matrix. Default "
        "'.adapter.linear_in.weight' (fix30, measured: "
        "Megatron-Bridge's ParallelLinearAdapter saves linear_in "
        "as (dim, in_features) under '<module>.adapter.'); the HF "
        "PEFT '.lora_A.weight'/'/.lora_B.weight' pair survives as "
        "_HF_PEFT_ADAPTER_SUFFIXES for estates that train with HF "
        "peft. Generator and recognizer are deliberately different "
        "knobs of different types -- a regex cannot generate -- "
        "and only their agreement is mandatory.",
    )
    p.add_argument(
        "--adapter-suffix-b",
        default=_DEFAULT_ADAPTER_SUFFIX_B,
        help="literal template GENERATING the expected FQN suffix "
        "of the (out_features, rank) adapter matrix. Default "
        "'.adapter.linear_out.weight' (Megatron-Bridge, measured); "
        "the HF PEFT second half is '.lora_B.weight'.",
    )
    p.add_argument(
        "--adapter-modules",
        default=None,
        metavar="PATH",
        help="JSON census of the adapter TARGET modules in the "
        "artifact's own namespace -- THE lora declared denominator "
        "(#78): a list of module FQNs, or an object carrying "
        "'adapter_modules' entries that are strings or "
        "{'fqn','out_features','in_features'} plus optional "
        "'source'. PRODUCER: the launcher at submit time, from the "
        "step-(5) live-module census over the BASE tree (today "
        "that census computes the names with the shipped matcher; "
        "persisting them -- with parent dims where available -- to "
        "fs_gate/adapter-modules.json and passing this flag is the "
        "producer wiring owed by the launcher side). NEVER the "
        "checkpoint under judgment: the tool refuses a census "
        "whose path resolves inside it, refuses an empty, "
        "duplicate-ridden or partially-dimmed one, and (exit 3, "
        "refusal_class adapter_census_unavailable) refuses a lora "
        "adjudication carried no census at all. FULL "
        "INDEPENDENCE, stated not faked: that census shares the "
        "trainer's matcher with the run; a planner/"
        "conversion-pipeline-produced, versioned expectation "
        "frozen at conversion time (the --fqn-map producer's "
        "shape, discharged by whoever owns conversion) is the "
        "fully independent form -- the residual is named in every "
        "report's declared basis until that exists. Ignored for "
        "full runs, which have their own two sources.",
    )
    p.add_argument(
        "--fqn-map",
        default=None,
        metavar="PATH",
        help="JSON list (or {'declared_fqns': [...]}) of the "
        "artifact-namespace FQNs a FULL checkpoint must contain. The "
        "denominator the low-overlap basis text points at: export it "
        "from the parallelism planner at submit time, never from the "
        "run under judgment.",
    )
    p.add_argument("--modules-to-save", default="")
    p.add_argument("--strict-extras", action="store_true")
    p.add_argument("--json", dest="json_out", default=None)
    args = p.parse_args(argv)

    overrides = dict(kv.split("=", 1) for kv in args.sets)
    try:
        d = adjudicate_checkpoint(
            args.ckpt_dir,
            event=args.event,
            run_kind=args.run_kind,
            base_model_dir=args.base_model_dir,
            train_config_path=args.train_config,
            overrides=overrides or None,
            controls=tuple(s for s in args.controls.split(",") if s),
            adapter_marker=args.adapter_marker,
            adapter_prefix=args.adapter_prefix,
            adapter_suffix_re=args.adapter_suffix,
            adapter_suffixes=(args.adapter_suffix_a, args.adapter_suffix_b),
            fqn_map=args.fqn_map,
            adapter_modules=args.adapter_modules,
            modules_to_save=tuple(s for s in args.modules_to_save.split(",") if s),
            strict_extras=args.strict_extras,
            json_out=args.json_out,
        )
    except GateUnmeasured as exc:
        print(f"live_gate could not measure: {exc}", file=sys.stderr)
        _record_refusal(args, str(exc))
        return EXIT_UNMEASURED
    except Exception:
        traceback.print_exc()
        refusal = "unexpected failure above (a tool bug is not a checkpoint verdict)"
        print(f"live_gate could not measure: {refusal}", file=sys.stderr)
        _record_refusal(args, refusal)
        return EXIT_UNMEASURED

    _h(f"live save gate: {d.checkpoint} (event={d.event}, kind={d.run_kind})")
    inv = d.report["inventory"]
    print(
        f"  artifact      : {inv['origin']} [{inv['format']}] "
        f"{inv['real_tensors']:,} real / {inv['entries_total']:,} entries"
    )
    print(f"  independent   : base {inv['base_source']} ({inv['base_tensors']:,} tensors)")
    _h("declared denominators -- provenance (independent sources only)")
    for k, v in d.declared_basis.items():
        if k == "notes":
            continue
        print(f"  {k:24s}: {v}")
    for note in d.declared_basis["notes"]:
        print(f"  note: {note}")
    _h("gates")
    for g in d.gate_results:
        # GateResult.to_dict() serializes the id under "gate", not "gate_id"; the
        # attribute is gate_id but the wire key is not, and reading the attribute
        # name off a dict crashed this renderer with KeyError on every run that got
        # far enough to print. Nothing exercised the render path until now, which is
        # why a KeyError shipped: the only honest fix is the key the schema defines.
        print(f"  [{g['verdict']:>12s}] {g['gate']}: {str(g.get('detail'))[:160]}")
    _h("MUST_FIRE controls (on copies of this artifact)")
    for c in d.controls:
        print(f"  {c['control']:36s} : {c['status']}")
        if c["status"] != "fired":
            print(f"      {c.get('reason') or c.get('detail', '')}")
    _h("summary")
    print(f"  blocking reasons: {len(d.blocking_reasons)}")
    for r in d.blocking_reasons:
        print(f"      {r}")
    print(f"LIVE GATE VERDICT: {d.verdict} (exit {d.exit_code})")
    return d.exit_code


if __name__ == "__main__":
    sys.exit(main())
