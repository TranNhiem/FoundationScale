"""VLA reinforcement learning through verl-vla's own trainers, from one FoundationScale spec.

Owned contract #3 of the VLA plan, applied to RL: FoundationScale holds the run declaration and
renders it into the backend's native command; verl-vla trains. :data:`PUBLISHED_RECIPES`
transcribes verl-vla's documented recipes (``examples/rl/<algo>/.../run_train*.sh`` and
``docs/reinforcement-learning/``) with the success counts they report, so a reproduction is a
rendered command plus a published curve to compare against. The algorithm, its schedule and its
losses stay in the recipe's own Hydra config; the spec overrides only placement (GPUs, output
directory, renderer, environment variables Ray workers need) -- every other override is the
recipe's, verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = [
    "PUBLISHED_RECIPES",
    "PublishedCurve",
    "VerlVlaRlError",
    "VerlVlaRlSpec",
    "on_gpus",
    "render_command",
    "with_output_dir",
    "with_runtime_env",
]

_ENTRYPOINTS = frozenset({"vvla-train-fpo", "vvla-train-sac", "vvla-train-recap"})
_RUNTIME_ENV = "ray_kwargs.ray_init.runtime_env.env_vars"


class VerlVlaRlError(ValueError):
    """An RL declaration verl-vla's trainers cannot run as declared."""


@dataclass(frozen=True)
class PublishedCurve:
    """The success counts a recipe's doc reports: before training, at the end, and at the best."""

    trials: int
    start: int
    end: int
    best: int
    source: str

    def __post_init__(self) -> None:
        if type(self.trials) is not int or self.trials < 1:
            raise VerlVlaRlError(f"trials is {self.trials!r}; expected an int >= 1")
        for name in ("start", "end", "best"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.trials:
                raise VerlVlaRlError(f"{name} is {value!r}; expected an int in [0, {self.trials}]")
        if self.best < max(self.start, self.end):
            raise VerlVlaRlError(f"best {self.best} is below start {self.start} or end {self.end}")


@dataclass(frozen=True)
class VerlVlaRlSpec:
    """One verl-vla RL run: an entrypoint, a recipe config, and ordered ``key=value`` overrides."""

    entrypoint: str
    config_dir: str
    config_name: str
    overrides: tuple[tuple[str, str], ...]
    # override keys holding model-GPU counts; on_gpus() rewrites exactly these
    gpu_keys: tuple[str, ...] = ()
    published: PublishedCurve | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.entrypoint not in _ENTRYPOINTS:
            raise VerlVlaRlError(
                f"entrypoint {self.entrypoint!r}; expected one of {sorted(_ENTRYPOINTS)}"
            )
        for name in ("config_dir", "config_name"):
            if not str(getattr(self, name)).strip():
                raise VerlVlaRlError(f"{name} is empty; verl-vla needs the recipe's Hydra config")
        keys = [key.lstrip("+") for key, _ in self.overrides]
        if len(keys) != len(set(keys)):
            raise VerlVlaRlError(
                f"an override key repeats in {keys}; Hydra would keep only the last"
            )
        missing = [key for key in self.gpu_keys if key not in keys]
        if missing:
            raise VerlVlaRlError(f"gpu_keys {missing} are not overrides of this recipe")

    def value(self, key: str) -> str:
        """The override value for ``key`` (with or without a leading ``+``)."""
        for k, v in self.overrides:
            if k.lstrip("+") == key:
                return v
        raise VerlVlaRlError(f"{key!r} is not an override of this recipe")


def _set(spec: VerlVlaRlSpec, key: str, value: str) -> VerlVlaRlSpec:
    out, seen = [], False
    for k, v in spec.overrides:
        if k.lstrip("+") == key.lstrip("+"):
            out.append((k, value))
            seen = True
        else:
            out.append((k, v))
    if not seen:
        out.append((key, value))
    return replace(spec, overrides=tuple(out))


def on_gpus(spec: VerlVlaRlSpec, num_gpus: int) -> VerlVlaRlSpec:
    """The same recipe on ``num_gpus`` model GPUs: each ``gpu_keys`` count capped at ``num_gpus``.

    A count already below ``num_gpus`` (a 1-worker DSRL actor, a 2-GPU value model) is kept.
    """
    if type(num_gpus) is not int or num_gpus < 1:
        raise VerlVlaRlError(f"num_gpus is {num_gpus!r}; expected an int >= 1")
    for key in spec.gpu_keys:
        spec = _set(spec, key, str(min(int(spec.value(key)), num_gpus)))
    return spec


def with_output_dir(spec: VerlVlaRlSpec, output_dir: str) -> VerlVlaRlSpec:
    """``output_dir`` and the TensorBoard directory under it, the two paths every recipe writes."""
    if not output_dir.strip():
        raise VerlVlaRlError("output_dir is empty")
    spec = _set(spec, "output_dir", output_dir)
    return _set(spec, f"{_RUNTIME_ENV}.TENSORBOARD_DIR", f"{output_dir}/tensorboard")


def with_runtime_env(spec: VerlVlaRlSpec, env: dict[str, str]) -> VerlVlaRlSpec:
    """Environment variables for the Ray workers (renderer, PYTHONPATH, caches).

    Keys the recipe's overrides already set are overwritten in place; others are appended with
    Hydra's ``++`` (add-or-override) prefix, because the recipe's YAML may define them even when its
    command line does not (verl-vla's FPO config sets ``PYTHONPATH``; a single ``+`` then fails).
    """
    for name, value in env.items():
        key = f"{_RUNTIME_ENV}.{name}"
        known = any(k.lstrip("+") == key for k, _ in spec.overrides)
        spec = _set(spec, key if known else f"++{key}", value)
    return spec


def render_command(spec: VerlVlaRlSpec, *, extra: tuple[tuple[str, str], ...] = ()) -> list[str]:
    """verl-vla's trainer argv (run from the verl-vla checkout); ``extra`` follow the recipe's."""
    for key, _ in extra:
        spec = _set(spec, key, dict(extra)[key])
    return [
        spec.entrypoint,
        "--config-dir",
        spec.config_dir,
        "--config-name",
        spec.config_name,
        *[f"{key}={value}" for key, value in spec.overrides],
    ]


_PI05_SFT = "Miical/pi05-libero-spatial-sft-step-100"
_EX = "./examples/rl"

# verl-vla examples/rl/*/run_train*.sh, transcribed; counts from docs/reinforcement-learning/*.md
PUBLISHED_RECIPES: dict[str, VerlVlaRlSpec] = {
    "fpo-pi05-libero-spatial-task2": VerlVlaRlSpec(
        entrypoint="vvla-train-fpo",
        config_dir=f"{_EX}/fpo/pi05/libero_spatial_task2_online_from_sft_step100",
        config_name="fpo",
        overrides=(
            ("output_dir", "./outputs/rl/fpo/pi05/libero-spatial-task2"),
            ("cluster.actor_rollout_ref.model.path", _PI05_SFT),
            ("cluster.resource.model.gpus_per_node", "8"),
            ("cluster.resource.model.workers_per_node", "8"),
            ("cluster.resource.env.device", "cpu"),
            ("cluster.resource.env.workers_per_node", "4"),
            ("cluster.env.env_worker.num_envs", "8"),
            (f"{_RUNTIME_ENV}.MUJOCO_GL", "osmesa"),
        ),
        gpu_keys=(
            "cluster.resource.model.gpus_per_node",
            "cluster.resource.model.workers_per_node",
        ),
        published=PublishedCurve(
            50, 29, 50, 50, "docs/reinforcement-learning/fpo/pi05/libero-spatial.md"
        ),
    ),
    "dsrl-pi05-libero-spatial-task2": VerlVlaRlSpec(
        entrypoint="vvla-train-sac",
        config_dir=f"{_EX}/dsrl/pi05/libero_spatial_online_from_sft_step100",
        config_name="dsrl",
        overrides=(
            ("output_dir", "./outputs/rl/dsrl/pi05/libero-spatial-task2-cnn-noise05"),
            ("cluster.actor_rollout_ref.model.path", _PI05_SFT),
            ("cluster.resource.model.gpus_per_node", "8"),
            ("cluster.resource.model.workers_per_node", "1"),
            ("cluster.resource.env.device", "cpu"),
            ("cluster.resource.env.workers_per_node", "16"),
            ("cluster.env.env_worker.num_envs", "2"),
            ("cluster.env.env_worker.simulator.libero.task_ids", "[2]"),
            ("cluster.actor_rollout_ref.actor.mini_batch_size", "128"),
            ("cluster.actor_rollout_ref.actor.micro_batch_size", "16"),
            ("cluster.actor_rollout_ref.model.adapter.dsrl.rollout_noise_scale", "0.5"),
            ("trainer.experiment_name", "pi05-libero-spatial-task2-cnn-noise05-dsrl"),
            (f"{_RUNTIME_ENV}.MUJOCO_GL", "osmesa"),
        ),
        gpu_keys=(
            "cluster.resource.model.gpus_per_node",
            "cluster.resource.model.workers_per_node",
        ),
        published=PublishedCurve(
            50, 37, 44, 44, "docs/reinforcement-learning/dsrl/pi05/libero-spatial.md"
        ),
    ),
    "dsrl-pi05-libero-spatial-task9": VerlVlaRlSpec(
        entrypoint="vvla-train-sac",
        config_dir=f"{_EX}/dsrl/pi05/libero_spatial_online_from_sft_step100",
        config_name="dsrl",
        overrides=(
            ("output_dir", "./outputs/rl/dsrl/pi05/libero-spatial-task9-cnn-noise002"),
            ("cluster.actor_rollout_ref.model.path", _PI05_SFT),
            ("cluster.resource.model.gpus_per_node", "8"),
            ("cluster.resource.model.workers_per_node", "1"),
            ("cluster.resource.env.device", "cpu"),
            ("cluster.resource.env.workers_per_node", "16"),
            ("cluster.env.env_worker.num_envs", "2"),
            ("cluster.env.env_worker.simulator.libero.task_ids", "[9]"),
            ("cluster.actor_rollout_ref.actor.mini_batch_size", "128"),
            ("cluster.actor_rollout_ref.actor.micro_batch_size", "16"),
            ("cluster.actor_rollout_ref.model.adapter.dsrl.rollout_noise_scale", "0.02"),
            ("trainer.experiment_name", "pi05-libero-spatial-task9-cnn-noise002-dsrl"),
            (f"{_RUNTIME_ENV}.MUJOCO_GL", "osmesa"),
        ),
        gpu_keys=(
            "cluster.resource.model.gpus_per_node",
            "cluster.resource.model.workers_per_node",
        ),
        published=PublishedCurve(
            50, 30, 37, 41, "docs/reinforcement-learning/dsrl/pi05/libero-spatial.md"
        ),
    ),
    "td3bc-pi05-libero-spatial-task2": VerlVlaRlSpec(
        entrypoint="vvla-train-sac",
        config_dir=f"{_EX}/td3_bc/pi05/libero_spatial_task2_online_from_sft_step100",
        config_name="td3_bc",
        overrides=(
            ("output_dir", "./outputs/rl/td3-bc/pi05/libero-spatial-task2-online-from-sft-step100"),
            ("cluster.actor_rollout_ref.model.path", _PI05_SFT),
            ("cluster.resource.model.gpus_per_node", "8"),
            ("cluster.resource.model.workers_per_node", "8"),
            ("cluster.resource.env.device", "cpu"),
            ("cluster.resource.env.workers_per_node", "4"),
            ("cluster.env.env_worker.num_envs", "8"),
            ("cluster.actor_rollout_ref.actor.mini_batch_size", "128"),
            ("cluster.actor_rollout_ref.actor.micro_batch_size", "16"),
            (f"{_RUNTIME_ENV}.MUJOCO_GL", "osmesa"),
        ),
        gpu_keys=(
            "cluster.resource.model.gpus_per_node",
            "cluster.resource.model.workers_per_node",
        ),
        published=PublishedCurve(
            50, 32, 40, 40, "docs/reinforcement-learning/td3-bc/pi05/libero-spatial.md"
        ),
    ),
    "td3bc-gaussian-libero-spatial-task0": VerlVlaRlSpec(
        entrypoint="vvla-train-sac",
        config_dir=f"{_EX}/td3_bc/gaussian_actor/libero_spatial_task0_online_from_hf_step100",
        config_name="td3_bc",
        overrides=(
            (
                "output_dir",
                "./outputs/rl/td3-bc/gaussian-actor/libero-spatial-task0-online-from-hf-step100",
            ),
            (
                "cluster.actor_rollout_ref.model.path",
                "Miical/gaussian-actor-libero-spatial-task0-step100-baseline",
            ),
            ("cluster.resource.model.gpus_per_node", "1"),
            ("cluster.resource.model.workers_per_node", "1"),
            ("cluster.resource.env.device", "cuda"),
            ("cluster.resource.env.gpus_per_node", "2"),
            ("cluster.env.env_worker.num_envs", "8"),
            (f"{_RUNTIME_ENV}.MUJOCO_GL", "egl"),
        ),
        gpu_keys=(
            "cluster.resource.model.gpus_per_node",
            "cluster.resource.model.workers_per_node",
        ),
        published=PublishedCurve(
            50, 2, 48, 48, "docs/reinforcement-learning/td3-bc/gaussian-actor/libero-spatial.md"
        ),
    ),
}
