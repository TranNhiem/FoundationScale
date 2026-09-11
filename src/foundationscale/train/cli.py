"""``foundationscale-train`` -- the thinnest real training entry point.

Installed as a console script by ``[project.scripts]``; the heavy dependencies
it needs are the ``train`` extra (``pip install 'foundationscale[train]'``).
Both are declared in pyproject.toml.

They were not always. This docstring used to carry those two stanzas as a note
prefaced "shipped as a note because the manifest is not modifiable from here" --
packaging written in the register of documentation, which reads like a decision
that was made rather than a task that was skipped. Nobody ran it, so for one
release ``prog="foundationscale-train"`` below advertised a command that ``pip
install`` never created, and loop.py's refusal path recommended an extra that
did not exist. Finding #224. The lesson is not "remember to edit pyproject" --
it is that a TODO indistinguishable from a description will be read as one, so
checks/packaging_reachability.py now asks the installed distribution whether
every advertised name resolves.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from foundationscale.train.loop import EXIT_RED, EXIT_REFUSE, TrainConfig, fs_version, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foundationscale-train",
        description=(
            "Thin real training path: validate Topology against a "
            "ClusterProfile BEFORE touching a GPU, then launch "
            "transformers.Trainer DDP with the FoundationScale save gate "
            "wired in as a Trainer callback. Exit codes: 0 PASS, 5 RED, "
            "95 UNMEASURED, 96 REFUSE."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {fs_version()}")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--dataset",
        required=True,
        help="HF dataset id, or a .json/.jsonl file, or a directory of them",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--save-interval", type=int, default=50)
    # Defaulted here and NOT on TrainConfig, and the asymmetry is deliberate.
    # This entry implements exactly one objective -- supervised fine-tuning, via
    # DataCollatorForLanguageModeling(mlm=False) -- so "sft" is a statement about
    # what the code does, not a plausible guess. A programmatic caller building
    # TrainConfig directly gets no such default: it may be driving a path this
    # entry does not, so it has to say which, and an unstated objective is
    # refused by the objective gate at the first observed step.
    p.add_argument(
        "--objective",
        default="sft",
        help=(
            "what the run optimises; recorded with provenance and checked by the "
            "objective gates at the first observed step (default: sft, the only "
            "objective this entry implements)"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dp",
        type=int,
        default=1,
        help=(
            "declared data-parallel degree; compared against the runtime's "
            "WORLD_SIZE and recorded, not used to configure anything"
        ),
    )
    # These four are DECLARATIONS, not knobs. The plane builds a
    # transformers.Trainer whose kwargs carry no tensor/pipeline/expert/
    # context-parallel key, so any degree > 1 is refused (EXIT_REFUSE) before a
    # model is loaded rather than trained as pure DDP under a parallel label.
    # Stated here so an operator learns it at --help time instead of after an
    # allocation is burned -- a knob that appears configurable and is not is
    # worse than a missing knob.
    _unwired = (
        "declared {name}-parallel degree. REFUSED when > 1: no {name} is wired "
        "into this plane, so a larger degree would be recorded and never "
        "executed (finding #375)"
    )
    p.add_argument("--tp", type=int, default=1, help=_unwired.format(name="tensor"))
    p.add_argument("--pp", type=int, default=1, help=_unwired.format(name="pipeline"))
    p.add_argument("--ep", type=int, default=1, help=_unwired.format(name="expert"))
    p.add_argument("--cp", type=int, default=1, help=_unwired.format(name="context"))
    # Machine facts: no defaults, fail closed (doctrine 4).
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile-name", help="name of a built-in ClusterProfile")
    group.add_argument("--profile-path", type=Path, help="path to a ClusterProfile JSON")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="run the ENTIRE validation prologue and exit without touching a GPU",
    )
    p.add_argument(
        "--launch-corpus",
        type=Path,
        default=None,
        help=(
            "directory of LAUNCHER scripts (.sh/.sbatch/.slurm) to scan for "
            "partition spelling variants. Omit it and the scan reports "
            "UNMEASURED -- it is never assumed clean"
        ),
    )
    # Precision and adapter flags default to None, NOT to a plausible value.
    # None means "not declared": the manifest records None, and the
    # observed-vs-declared precision check abstains rather than passing
    # (an unstated precision must never be laundered into "bf16" -- #342).
    p.add_argument(
        "--precision",
        default=None,
        help=(
            "declared training precision, one of: bf16, fp16, fp32, nvfp4. "
            "nvfp4 is declarable but train() refuses it (96) until the package "
            "plane grows an nvfp4 backend -- it never falls back silently. "
            "Omit to declare nothing"
        ),
    )
    p.add_argument(
        "--adapter",
        default=None,
        help=(
            "adapter mode, one of: lora. Omit for a full fine-tune -- that is "
            "the explicit default, not an unstated one. Requires --adapter-rank; "
            "a missing peft install is a REFUSE (96), never a silent full "
            "fine-tune"
        ),
    )
    p.add_argument("--adapter-rank", type=int, default=None)
    p.add_argument("--adapter-alpha", type=float, default=None)
    p.add_argument(
        "--adapter-target",
        action="append",
        default=None,
        help=(
            "LoRA target module pattern; repeatable, collected in order. "
            "Omit to use peft's per-model defaults. If the adapter resolves "
            "to zero modules the run is refused after wrapping"
        ),
    )
    p.add_argument("--adapter-dropout", type=float, default=None)
    # --- The nine declaration axes ------------------------------------------
    # Every flag below defaults to None for exactly the reason --precision does:
    # None means NOT DECLARED. An omitted flag applies the transformers engine
    # default (AdamW, accumulation 1, max_grad_norm 1.0, no recompute, linear LR
    # with zero warmup, model-config attention) and records that default as
    # NOTHING -- the manifest key is present with value None, an explicit
    # abstention, never a claim the engine default was chosen (#342). Declared
    # values are wired into TrainingArguments only when provided, and every one
    # flows through the introspection refusal in loop.py, so an older
    # transformers refuses (96) on a knob it cannot accept rather than silently
    # dropping it.
    p.add_argument(
        "--optimizer",
        default=None,
        help=(
            "optimizer name passed to TrainingArguments as `optim` (e.g. "
            "adamw_torch, adafactor, sgd). Omit: the engine default applies "
            "and is NOT recorded as a claim"
        ),
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=(
            "forward/backward passes per optimizer step. Omit: the engine "
            "default (1) applies and is NOT recorded as a claim"
        ),
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help=(
            "gradient clipping norm passed to TrainingArguments. Omit: the "
            "engine default (1.0) applies and is NOT recorded as a claim"
        ),
    )
    p.add_argument(
        "--gradient-checkpointing",
        choices=("true", "false"),
        default=None,
        help=(
            "activation recomputation, declared as true/false. A string, not "
            "store_true, so an OMITTED flag (None) stays distinguishable from "
            "an explicit false. Omit: the engine default (off) applies and is "
            "NOT recorded as a claim"
        ),
    )
    p.add_argument(
        "--attn-implementation",
        default=None,
        help=(
            "attention implementation for MODEL construction (e.g. sdpa, "
            "flash_attention_2, eager). Refused (96) when this transformers "
            "release's model loader cannot provably accept it -- never "
            "silently dropped through **kwargs. Omit: the model-config "
            "default applies and is NOT recorded as a claim"
        ),
    )
    p.add_argument(
        "--lr-scheduler-type",
        default=None,
        help=(
            "LR scheduler name passed to TrainingArguments (e.g. linear, "
            "cosine). Omit: the engine default (linear) applies and is NOT "
            "recorded as a claim"
        ),
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help=(
            "learning-rate warmup steps for the scheduler. Omit: the engine "
            "default (0) applies and is NOT recorded as a claim"
        ),
    )
    p.add_argument(
        "--sharding-strategy",
        default=None,
        help=(
            "declared sharding strategy. Only 'ddp' has execution behind it "
            "in this plane -- any other value is REFUSED (96) rather than run "
            "as plain DDP under a manifest that claims otherwise (#375's "
            "defect class); FSDP/ZeRO/DeepSpeed were adjudicated and recorded, "
            "never built. Omit: data parallelism only, and no claim is made"
        ),
    )
    p.add_argument(
        "--cpu-optimizer-offload",
        choices=("true", "false"),
        default=None,
        help=(
            "optimizer-state offload, declared as true/false. 'true' is "
            "REFUSED (96): this plane wires no offload backend, so honouring "
            "it would require executor code that was adjudicated and never "
            "built. Omit: no offload, and no claim is recorded"
        ),
    )
    return p


# Sentinel for "argparse never assigned this dest from the command line".
# A dedicated object rather than None, because None is a legal DECLARED value
# nowhere here but a legal RESOLVED one everywhere: reusing it would make an
# omission and a stated absence read the same, which is the #342 laundering
# this record exists to prevent.
_UNSUPPLIED = object()

# The one argparse dest that does not spell its TrainConfig field. Everything
# else is its own field name; tests/train/test_manifest_records_declared_axes.py
# asserts that every dest resolves to a real field, so a future flag whose dest
# matches nothing fails there rather than silently losing its provenance.
_DEST_TO_FIELD: dict[str, str] = {"adapter_target": "adapter_targets"}


def _declared_fields(
    argv: Sequence[str] | None,
    parsed: argparse.Namespace,
) -> tuple[str, ...]:
    """Return the TrainConfig field names the command line actually supplied.

    Measured, not inferred from the values. ``optimizer is None`` happens to
    mean "not declared" only because every tier-0 axis defaults to None;
    ``max_steps=1000`` is genuinely ambiguous between a typed flag and the
    field default, and a manifest that guesses there records a source it never
    observed. So the same argv is parsed a second time against a parser whose
    every default is a sentinel: whatever argparse assigns is what the operator
    wrote, and whatever is still the sentinel is what nobody said.

    ``set_defaults`` is argparse's public API for this and it reaches the
    actions themselves, so the sentinel replaces the per-argument defaults
    rather than sitting behind them. The real parse runs first so that a
    malformed command line produces the real parser's error message.
    """
    probe = build_parser()
    probe.set_defaults(**dict.fromkeys(vars(parsed), _UNSUPPLIED))
    supplied = vars(probe.parse_args(argv))
    return tuple(
        sorted(
            _DEST_TO_FIELD.get(dest, dest)
            for dest, value in supplied.items()
            if value is not _UNSUPPLIED
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Bind the command line, then hand off to ``train`` -- totally.

    #380 made ``train`` total over the exit contract, and the reasoning there
    stops at ``train``'s own boundary: everything BEFORE the handoff still ran
    naked. ``build_parser`` calls ``fs_version()``, which asks importlib for
    installed distribution metadata and raises when the package is not
    installed the way it thinks; ``_build_config`` reads forty-odd namespace
    attributes and is guarded for ``ValueError`` and nothing else, so a
    ``TypeError`` from a mistyped argparse action leaves it untouched. Either
    one reached the interpreter and exited **1** -- outside 0/5/95/96, the
    code #171 showed a launcher cannot interpret, and the same escape #380
    closed one layer along.

    So the pre-handoff region is wrapped and the handoff is not. That split
    is the whole design:

    * the outer ``except Exception`` covers parser construction, parsing and
      config binding -- the statements that have no other adjudicator;
    * ``return train(cfg)`` sits OUTSIDE it, because ``train`` is already
      total and re-wrapping it would put a second reporter over a verdict
      that has already been made (and, if it were folded into the refusal
      block instead, would report a training failure as a refusal -- the
      laundering ``_build_config``'s docstring exists to prevent).

    RED, not UNMEASURED: it is the answer ``train`` gives for "crashed", and
    a caller that cannot tell which side of the handoff the crash happened on
    should not have to. ``BaseException`` is deliberately not caught, so a
    ``SystemExit`` raised anywhere else still passes through.

    #387 asked what to do about argparse's own exits, which reach the shell
    from inside the stdlib: 2 for a usage error, 0 for ``--help`` and
    ``--version``. Resolved by SPLITTING them, because they are not one
    surface:

    * A usage error is the SAME event as the ``ValueError`` arm below -- the
      operator stated something the plane will not honour, and nothing was
      measured -- reported differently only because of which layer noticed
      first. That is not academic here: #181, #183 and #184 were all
      composed-argv defects, so a launcher meeting a mistyped flag is this
      plane's recurring failure rather than a hypothetical, and 2 is the one
      code its case statements do not handle. A nonzero parser exit is
      therefore translated to EXIT_REFUSE, under the same marker as the arm
      below. argparse has already written its own diagnosis to stderr by the
      time it raises, so the operator's evidence is unchanged; only the code
      the launcher reads is.
    * ``--help`` and ``--version`` are re-raised untouched. Exit 0 there is
      the tool answering a question about itself, it is in-contract by value,
      and every other CLI on the machine behaves this way.

    The translation catches ``SystemExit`` and nothing wider, and that is what
    keeps it from being the laundering ``test_cli_refuses_bad_declarations``
    exists to forbid: a ``TypeError`` from a mistyped argparse action is not a
    ``SystemExit``, so it still falls through to the outer handler and is
    adjudicated RED rather than dressed up as a rejected declaration.
    """
    try:
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            # Narrow on purpose -- see the docstring. `exc.code` is None for a
            # bare SystemExit() and may be a string if some future parser
            # calls `parser.exit(message=...)`; both are handled by testing
            # for the ONE code that passes through rather than for the codes
            # that do not, so an unanticipated spelling refuses rather than
            # escaping as an unclassified exit.
            if exc.code is None or exc.code == 0:
                raise
            print(
                f"[fs:train:refuse] declaration rejected: the command line did "
                f"not parse (argparse exit {exc.code}); its own diagnosis is on "
                f"stderr above this line"
            )
            return EXIT_REFUSE
        try:
            cfg = _build_config(argv, args)
        except ValueError as exc:
            # TrainConfig.__post_init__ validates every declaration it can
            # check without a GPU, and it signals a bad one by raising.
            # Uncaught, that left the interpreter to print a traceback and
            # exit 1 -- a code this plane does not define, in the one
            # namespace (0/5/95/96) whose whole purpose is that a caller can
            # tell REFUSED from RED from UNMEASURED without parsing prose.
            # `--max-grad-norm -1` exited 1 with a stack trace; a launcher
            # reading that saw neither a refusal nor a verdict.
            #
            # 96, not 5: nothing was measured. The operator stated something
            # the plane will not honour, and it says so before a profile is
            # resolved, before a model is loaded, and before an allocation is
            # burned. The marker matches train()'s own refusal site so one
            # grep finds both.
            #
            # This handler is nested INSIDE the crash guard rather than made
            # a sibling of it so that it still wraps construction and nothing
            # else -- a ValueError out of `build_parser` or `parse_args` is
            # not a rejected declaration, and reporting it as one would be
            # the laundering _build_config's docstring exists to prevent.
            print(f"[fs:train:refuse] declaration rejected: {exc}")
            return EXIT_REFUSE
    except Exception as exc:  # noqa: BLE001 -- the contract has no code for "crashed"
        # Same shape as train()'s boundary handler: the traceback is the only
        # diagnosis of an unclassified crash, so it goes to stderr exactly
        # where the interpreter would have put it. What changes is the exit
        # code, not the operator's evidence.
        traceback.print_exc(file=sys.stderr)
        print(
            f"[fs:train:red] unhandled {type(exc).__name__} escaped command-line "
            f"binding: {exc}. Adjudicated RED (5) at the main() boundary rather "
            f"than allowed to exit 1, which sits outside the 0/5/95/96 contract. "
            f"Full traceback on stderr"
        )
        return EXIT_RED
    return train(cfg)


def _build_config(argv: Sequence[str] | None, args: argparse.Namespace) -> TrainConfig:
    """Bind the parsed namespace to a TrainConfig.

    Split out of ``main`` so the ``except ValueError`` above wraps construction
    and NOTHING else. Wrapping ``train(cfg)`` in the same block would swallow a
    ValueError raised deep in a real run -- a training failure reported as a
    rejected declaration, which is the same laundering one layer along.
    """
    return TrainConfig(
        cli_declared=_declared_fields(argv, args),
        model=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        profile_name=args.profile_name,
        profile_path=args.profile_path,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        objective=args.objective,
        seed=args.seed,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        cp=args.cp,
        dry_run=args.dry_run,
        launch_corpus=args.launch_corpus,
        precision=args.precision,
        adapter=args.adapter,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        adapter_targets=(tuple(args.adapter_target) if args.adapter_target is not None else None),
        adapter_dropout=args.adapter_dropout,
        # The two tri-state booleans arrive as "true"/"false" strings so that
        # omission stays visible: None -> None (abstained), an explicit value
        # -> a real bool. Collapsing them with store_true would mint an
        # undeclared False that is indistinguishable from a stated one --
        # the #342 laundering in miniature.
        gradient_checkpointing=(
            None if args.gradient_checkpointing is None else args.gradient_checkpointing == "true"
        ),
        cpu_optimizer_offload=(
            None if args.cpu_optimizer_offload is None else args.cpu_optimizer_offload == "true"
        ),
        optimizer=args.optimizer,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        attn_implementation=args.attn_implementation,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        sharding_strategy=args.sharding_strategy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
