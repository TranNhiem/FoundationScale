"""``foundationscale-agentic-rl``: build and run one agentic RL training run from
a declared, provenance-tracked JSON config (see ``config.py``).

Exit taxonomy is EXACTLY the four codes ``train/cli.py`` uses, imported from
the same place (``train.loop``) rather than re-declared, so one grep finds
every definition: ``EXIT_PASS`` (0, success), ``EXIT_RED`` (5, a crash --
``train/cli.py``'s own words: "the answer train gives for 'crashed'"),
``EXIT_UNMEASURED`` (95, ran but measured nothing -- the all([]) shape this
whole plane refuses as a silent pass), ``EXIT_REFUSE`` (96, a declaration this
plane will not honour, caught before or instead of running).

Two phases, two different levels of totality, mirroring ``train/cli.py``'s own
split and for the same reason (see its docstring): command-line parsing and
config loading have no other adjudicator and are wrapped in a broad
``except Exception`` here; ``--dry-run``'s object construction and the real
run's construction + ``RLTrainer.run()`` are each wrapped in their OWN
dedicated ``try``/``except`` (catching the plane's own ``ValueError``-rooted
refusals and ``TrainerRefusal`` as REFUSE, anything else as RED) so that a
``SystemExit`` raised deep inside ``RLTrainer.run()`` (``rl.trainer.
_refuse_exit_96``) still passes through untouched -- ``SystemExit`` is a
``BaseException``, never caught by ``except Exception``, so it reaches the
process with the exit code it already carries, exactly as ``train/cli.py``
relies on for its own ``train()`` call.

The real run imports ``transformers.AutoTokenizer`` LAZILY, inside
``_build_real_run``, and nowhere else in this module imports torch or
transformers: everything else this package needs (``RLTrainConfig``,
``RLTrainer``, the engine/env/harness/reward/task machinery) is torch-free by
construction (see each module's own docstring), so ``--dry-run`` -- which
never reaches ``_build_real_run`` -- imports neither.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from foundationscale.agentic_rl.config import (
    AgenticConfigRefusal,
    AgenticRLConfig,
    load_config,
)
from foundationscale.agentic_rl.engines.fleet import EngineFleet, EngineServer, EngineServerSpec
from foundationscale.agentic_rl.envs.base import EnvSpec, get_env_backend
from foundationscale.agentic_rl.harness.base import ChatTokenizer, EpisodeBudget, SamplingParams
from foundationscale.agentic_rl.harness.native_tool_loop import NativeToolLoop
from foundationscale.agentic_rl.rewards.base import MusicRewardFn
from foundationscale.agentic_rl.rewards.music import MusicReward
from foundationscale.agentic_rl.rollout_host import RolloutHost
from foundationscale.agentic_rl.tasks import TaskSource
from foundationscale.agentic_rl.tools import (
    HermesJsonToolCallParser,
    QwenXmlToolCallParser,
    ToolCallParser,
    default_tools,
)
from foundationscale.agentic_rl.weight_sync import DiskWeightSync
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal

__all__ = ("EXIT_PASS", "EXIT_RED", "EXIT_REFUSE", "EXIT_UNMEASURED", "build_parser", "main")

# SAME VALUES as foundationscale.train.loop's EXIT_PASS/EXIT_RED/EXIT_UNMEASURED/
# EXIT_REFUSE (0/5/95/96) -- re-STATED here rather than imported, deliberately.
# `foundationscale.train.loop` imports torch and transformers at MODULE scope
# (transitively, via `foundationscale.families`), so `from
# foundationscale.train.loop import EXIT_PASS` would import torch the moment
# this module is imported -- defeating --dry-run's own "imports no torch"
# claim before a single line of main() ran. tests/agentic_rl/test_cli.py pins
# these four literals against train.loop's real values (in a test process,
# where that cost is already paid) so the two sets cannot silently drift.
EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

_REFUSE_MARKER = "[fs:agentic-rl:refuse]"
_RED_MARKER = "[fs:agentic-rl:red]"
_UNMEASURED_MARKER = "[fs:agentic-rl:unmeasured]"


@dataclass(frozen=True)
class _HFChatTokenizer:
    """Adapts a ``transformers`` tokenizer to the ``ChatTokenizer`` protocol via
    ``apply_chat_template(..., tokenize=True)``. Built only on the real-run path
    (see ``_build_real_run``) -- never during ``--dry-run``.
    """

    tokenizer: Any

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools is not None else None,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
        )
        # transformers 5 can still hand back a BatchEncoding (a mapping carrying
        # "input_ids") -- measured on the first GPU smoke run, where list() of it
        # produced the KEY strings. Read the ids out by name; anything else is
        # refused rather than guessed at.
        if isinstance(rendered, Mapping):
            if "input_ids" not in rendered:
                raise AgenticConfigRefusal(
                    f"tokenizer.apply_chat_template returned a mapping without "
                    f"'input_ids' (keys: {sorted(rendered)}); cannot read token ids"
                )
            rendered = rendered["input_ids"]
        ids = list(rendered)
        if ids and isinstance(ids[0], (list, tuple)):
            if len(ids) != 1:
                raise AgenticConfigRefusal(
                    f"tokenizer.apply_chat_template returned {len(ids)} sequences "
                    f"for 1 conversation; expected exactly 1"
                )
            ids = list(ids[0])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        decoded = self.tokenizer.decode(list(ids))
        return str(decoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundationscale-agentic-rl",
        description=(
            "Multi-turn, tool-using agentic RL training from a declared JSON "
            "config. Exit codes: 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE."
        ),
    )
    parser.add_argument("--config", required=True, help="path to a JSON config file")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="override one declared config key; repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate the config and build every object that needs no GPU or "
            "server, print the resolved config with provenance as JSON, and "
            "exit 0 without training"
        ),
    )
    return parser


def _make_parser(parser_name: str) -> ToolCallParser:
    if parser_name == "qwen_xml":
        return QwenXmlToolCallParser()
    return HermesJsonToolCallParser()


def _sampling_params(config: AgenticRLConfig) -> SamplingParams:
    return SamplingParams(
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        top_k=config.sampling.top_k,
        min_p=config.sampling.min_p,
        presence_penalty=config.sampling.presence_penalty,
        repetition_penalty=config.sampling.repetition_penalty,
        max_new_tokens=config.sampling.max_new_tokens,
    )


def _episode_budget(config: AgenticRLConfig) -> EpisodeBudget:
    return EpisodeBudget(
        step_limit=config.budget.step_limit,
        max_response_tokens=config.budget.max_response_tokens,
        max_observation_chars=config.budget.max_observation_chars,
    )


def _env_spec(config: AgenticRLConfig) -> EnvSpec:
    return EnvSpec(
        backend=config.env.backend,
        env=config.env.env,
        exec_timeout_s=config.env.exec_timeout_s,
        episode_timeout_s=config.env.episode_timeout_s,
        max_output_bytes=config.env.max_output_bytes,
        allow_unsandboxed=config.env.allow_unsandboxed,
    )


def _engine_server_spec(config: AgenticRLConfig) -> EngineServerSpec:
    return EngineServerSpec(
        kind=config.engine.kind,
        command=config.engine.command,
        port=config.engine.port,
        served_model_name=config.engine.served_model_name,
        startup_timeout_s=config.engine.startup_timeout_s,
        request_timeout_s=config.engine.request_timeout_s,
        host=config.engine.host,
        log_dir=config.engine.log_dir,
        reload_mode=config.engine.reload_mode,
        env=config.engine.env,
    )


def _music_reward(config: AgenticRLConfig) -> MusicReward:
    return MusicReward(abc2midi_bin=config.reward.abc2midi_bin, timeout_s=config.reward.timeout_s)


def _task_source(config: AgenticRLConfig) -> TaskSource:
    return TaskSource(path=config.rollout.tasks_path, seed=config.rollout.seed)


def _build_non_gpu_objects(config: AgenticRLConfig) -> dict[str, Any]:
    """Validate and build every object ``--dry-run`` can build without a GPU or
    a server running: the tool registry and parser, the sampling/budget/env
    declarations, the (unstarted) engine server and fleet, the reward scorer,
    and the task pool. Raises the plane's own named ``ValueError``-rooted
    refusals on a bad declaration; never imports torch or transformers.
    """
    tools = default_tools()
    parser = _make_parser(config.harness.parser)
    sampling = _sampling_params(config)
    budget = _episode_budget(config)
    env_spec = _env_spec(config)
    engine_spec = _engine_server_spec(config)
    fleet = EngineFleet(servers=(EngineServer(engine_spec),))
    reward = _music_reward(config)
    tasks = _task_source(config)
    return {
        "tools": tools,
        "parser": parser,
        "sampling": sampling,
        "budget": budget,
        "env_spec": env_spec,
        "engine_spec": engine_spec,
        "fleet": fleet,
        "reward": reward,
        "tasks": tasks,
    }


def _build_real_run(config: AgenticRLConfig) -> RolloutHost:
    """Build the full real-run object graph: tokenizer, the ``ChatTokenizer``
    adapter over it, the harness, the (unstarted) engine fleet, the
    environment backend and the reward adapter -- everything
    ``RolloutHost.rollout(step)`` needs. Starting the fleet and running the
    trainer are the caller's job (see ``_run``), so this function never talks
    to a GPU or a socket either.
    """
    from transformers import AutoTokenizer  # lazy: only this real-run path needs it

    tokenizer = AutoTokenizer.from_pretrained(config.policy.model_path)
    chat_tokenizer: ChatTokenizer = _HFChatTokenizer(tokenizer=tokenizer)
    harness = NativeToolLoop(
        tools=default_tools(),
        parser=_make_parser(config.harness.parser),
        tokenizer=chat_tokenizer,
        sampling=_sampling_params(config),
    )
    fleet = EngineFleet(servers=(EngineServer(_engine_server_spec(config)),))
    reward = MusicRewardFn(scorer=_music_reward(config), decode=chat_tokenizer.decode)
    # save_fn stays None: RolloutHost.publish() builds its OWN save around the
    # real model/tokenizer/ctx it receives from RLTrainer.run() (via
    # rl.distributed.save_checkpoint), never through weight_sync.save_fn --
    # see rollout_host.py's and weight_sync.py's own docstrings for why no
    # closure built HERE, before the policy exists, could serve that call.
    weight_sync = DiskWeightSync(
        publish_root=config.publish_root,
        fleet=fleet,
        mode=config.engine.weight_sync_mode,
    )
    return RolloutHost(
        tasks=_task_source(config),
        tasks_per_step=config.rollout.tasks_per_step,
        group_size=config.rollout.group_size,
        harness=harness,
        env_backend=get_env_backend(config.env.backend),
        env_spec=_env_spec(config),
        reward=reward,
        budget=_episode_budget(config),
        max_concurrency=config.rollout.max_concurrency,
        fleet=fleet,
        group_by_harness=config.rollout.group_by_harness,
        weight_sync=weight_sync,
        sharding=config.trainer["sharding"],
        declared_max_infra_rate=config.rollout.max_infra_rate,
        declared_max_lag=config.engine.max_policy_lag,
    )


def _print_refusal(exc: Exception) -> None:
    print(f"{_REFUSE_MARKER} declaration rejected: {exc}")


def _print_red(exc: Exception, *, where: str) -> None:
    traceback.print_exc(file=sys.stderr)
    print(
        f"{_RED_MARKER} unhandled {type(exc).__name__} {where}: {exc}. Adjudicated "
        f"RED (5) rather than allowed to exit 1, which sits outside the 0/5/95/96 "
        f"contract. Full traceback on stderr"
    )


def _dry_run(config: AgenticRLConfig) -> int:
    try:
        _build_non_gpu_objects(config)
    except ValueError as exc:
        _print_refusal(exc)
        return EXIT_REFUSE
    except Exception as exc:  # noqa: BLE001 -- adjudicated RED, see module docstring
        _print_red(exc, where="during --dry-run")
        return EXIT_RED
    print(json.dumps(config.as_json(), indent=2, sort_keys=True, default=str))
    return EXIT_PASS


def _run(config: AgenticRLConfig) -> int:
    try:
        host = _build_real_run(config)
    except ValueError as exc:
        _print_refusal(exc)
        return EXIT_REFUSE
    except Exception as exc:  # noqa: BLE001 -- adjudicated RED, see module docstring
        _print_red(exc, where="building the run")
        return EXIT_RED

    max_steps = config.trainer["max_steps"]
    if max_steps > 1 and host.weight_sync is None and not config.engine.allow_stale_policy:
        # Named, explicit guard -- never relied on as a mere invariant of
        # _build_real_run always supplying a DiskWeightSync (see the module
        # docstring): running more than one step with no weight sync wired
        # means the fleet keeps serving the INITIAL weights forever, so every
        # rollout after the first is silently off-policy. engine.
        # allow_stale_policy=true is the explicit, provenance-recorded opt-in
        # out of this refusal.
        print(
            f"{_REFUSE_MARKER} declaration rejected: trainer.max_steps={max_steps} > 1 "
            f"with no weight sync wired: the engine fleet would keep serving the "
            f"policy's INITIAL weights for every step, making every rollout after "
            f"the first silently off-policy. Declare engine.allow_stale_policy=true "
            f"to run anyway (e.g. for a deliberate smoke test), or wire weight sync."
        )
        return EXIT_REFUSE

    assert host.fleet is not None  # _build_real_run always supplies one
    fleet = host.fleet
    try:
        fleet.start_all()
        trainer_config = RLTrainConfig(
            model=config.policy.model_path,
            algorithm=config.policy.algorithm,
            rollout_source=host,
            **config.trainer,
        )
        reports = RLTrainer(trainer_config).run()
    except TrainerRefusal as exc:
        _print_refusal(exc)
        return EXIT_REFUSE
    except Exception as exc:  # noqa: BLE001 -- adjudicated RED, see module docstring
        _print_red(exc, where="running the trainer")
        return EXIT_RED
    finally:
        # A failed start_all() leaves earlier servers running (EngineFleet's
        # own documented contract); stop_all() is idempotent per server, so
        # this cleans up a partial start exactly as fully as a complete one.
        fleet.stop_all()

    if not reports:
        print(
            f"{_UNMEASURED_MARKER} run() completed but reported 0 steps: nothing "
            f"was measured -- the all([]) shape this plane refuses as a silent pass"
        )
        return EXIT_UNMEASURED
    return EXIT_PASS


def main(argv: Sequence[str] | None = None) -> int:
    """Bind the command line and config, then hand off to ``_dry_run``/``_run``.

    Mirrors ``train/cli.py``'s own split exactly (see its docstring for the
    full reasoning): the pre-handoff region (parsing, config loading) is
    wrapped in one broad ``except Exception`` here because nothing else
    adjudicates it; the handoff itself is NOT wrapped here, because
    ``_dry_run``/``_run`` are each already total over 0/5/95/96 on their own.
    """
    try:
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            if exc.code is None or exc.code == 0:
                raise
            print(
                f"{_REFUSE_MARKER} declaration rejected: the command line did not "
                f"parse (argparse exit {exc.code}); its own diagnosis is on stderr "
                f"above this line"
            )
            return EXIT_REFUSE
        try:
            config = load_config(args.config, args.set_overrides)
        except AgenticConfigRefusal as exc:
            print(f"{_REFUSE_MARKER} config refused: {exc}")
            return EXIT_REFUSE
    except Exception as exc:  # noqa: BLE001 -- adjudicated RED, see module docstring
        _print_red(exc, where="escaped command-line binding")
        return EXIT_RED

    if args.dry_run:
        return _dry_run(config)
    return _run(config)


if __name__ == "__main__":
    raise SystemExit(main())
