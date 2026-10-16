"""Alpamayo 2 SFT through alpamayo-recipes' own Hydra launchers, from one FoundationScale spec.

Owned contract #3 of the VLA plan, applied to SFT: FoundationScale holds the run declaration and
renders it into alpamayo-recipes' ``torchrun``/``python`` command line; ``alpamayo2_sft`` trains and
evaluates. :data:`PUBLISHED_RECIPES` transcribes the recipe's documented smoke runs (``README.md``,
"Outputs and Expected Signals") with the losses they report, so a reproduction is a rendered
command plus a published curve to compare against. The model, its schedule and its DeepSpeed config
stay in the recipe's own Hydra config (``pkg://alpamayo2_sft/configs``); the spec overrides only the
keys the README's commands override -- the checkpoint, the PAI chunk split and the output
directory -- every other override is the recipe's, verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

__all__ = [
    "PUBLISHED_RECIPES",
    "AlpamayoSftError",
    "AlpamayoSftRecipe",
    "AlpamayoSftSpec",
    "PublishedLosses",
    "PublishedMetric",
]

_CHUNKS = re.compile(r"^\d+-\d+$")
_OWNED_KEYS = frozenset(
    {
        "model.pretrained_model_name_or_path",
        "evaluate.eval_ckpt",
        "data.train_dataset.local_dir",
        "data.train_dataset.chunk_ids",
        "data.val_dataset.local_dir",
        "data.val_dataset.chunk_ids",
        "paths.output_dir",
    }
)


class AlpamayoSftError(ValueError):
    """An SFT declaration alpamayo-recipes' launchers cannot run as declared."""


def _chunks(spec: AlpamayoSftSpec, name: str) -> tuple[int, int]:
    value = getattr(spec, name)
    text = str(value)
    if not _CHUNKS.match(text):
        raise AlpamayoSftError(f"{name} is {value!r}; expected a half-open range like '0-99'")
    start, end = (int(part) for part in text.split("-"))
    if end <= start:
        raise AlpamayoSftError(f"{name} is {value!r}; a half-open range needs start < end")
    return start, end


@dataclass(frozen=True)
class PublishedLosses:
    """The loss numbers a recipe's README section reports, in the order the log shows them."""

    values: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.values:
            raise AlpamayoSftError("values is empty; the recipe README reports at least one loss")


@dataclass(frozen=True)
class PublishedMetric:
    """One held-out number the recipe README reports (``eval_loss`` or ``eval_min_ade``)."""

    metric: str
    value: float
    source: str


@dataclass(frozen=True)
class AlpamayoSftRecipe:
    """A published recipe: the Hydra entry module, its launcher kind, and the numbers to compare."""

    config_name: str
    entry: str
    multi_process: bool
    published: dict[str, PublishedLosses | PublishedMetric]
    overrides: tuple[str, ...] = ()  # the README command's own overrides beyond the spec's keys

    def __post_init__(self) -> None:
        if self.entry not in {"alpamayo2_sft.train_hf", "alpamayo2_sft.evaluate_hf"}:
            raise AlpamayoSftError(
                f"entry {self.entry!r}; expected 'alpamayo2_sft.train_hf'"
                " or 'alpamayo2_sft.evaluate_hf'"
            )
        if not self.published:
            raise AlpamayoSftError(f"recipe {self.config_name!r} publishes no metric")


@dataclass(frozen=True)
class AlpamayoSftSpec:
    """One alpamayo-recipes SFT run: a published recipe, a checkpoint, PAI chunks and placement."""

    config_name: str
    model: str
    pai_dir: str
    train_chunks: str = "0-1"
    val_chunks: str = "99-100"
    output_dir: str | None = None
    gpus: tuple[int, ...] = ()
    extra_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.config_name not in PUBLISHED_RECIPES:
            raise AlpamayoSftError(
                f"config_name {self.config_name!r}; expected one of {sorted(PUBLISHED_RECIPES)}"
            )
        if not self.pai_dir.strip():
            raise AlpamayoSftError("pai_dir is empty; alpamayo-recipes needs the PAI snapshot root")
        if not self.model.strip():
            raise AlpamayoSftError(
                "model is empty; alpamayo-recipes needs a checkpoint dir or HF id"
            )
        train = _chunks(self, "train_chunks")
        val = _chunks(self, "val_chunks")
        if train[0] < val[1] and val[0] < train[1]:
            raise AlpamayoSftError(
                f"train_chunks {self.train_chunks!r} overlaps val_chunks"
                f" {self.val_chunks!r}; chunk 99 must not train"
            )
        recipe = PUBLISHED_RECIPES[self.config_name]
        if recipe.multi_process and not self.gpus:
            raise AlpamayoSftError(
                f"recipe {self.config_name!r} runs under torchrun and needs GPUs"
            )
        if not recipe.multi_process and len(self.gpus) > 1:
            raise AlpamayoSftError(
                f"recipe {self.config_name!r} on {len(self.gpus)} GPUs; trajectory evaluation"
                " rejects multiple processes"
            )
        seen: set[str] = set()
        for override in self.extra_overrides:
            key = override.split("=", 1)[0].lstrip("+")
            if key in _OWNED_KEYS:
                raise AlpamayoSftError(
                    f"extra_overrides set {key!r}, which this spec already owns; change the field"
                )
            if key in seen:
                raise AlpamayoSftError(
                    f"extra_overrides set {key!r} twice; Hydra keeps only the last"
                )
            seen.add(key)

    def on_gpus(self, gpus: tuple[int, ...]) -> AlpamayoSftSpec:
        """The same recipe on ``gpus`` (the CUDA device ids to expose, in order)."""
        return replace(self, gpus=gpus)

    def with_output_dir(self, path: str) -> AlpamayoSftSpec:
        """The same recipe writing its ``paths.output_dir`` to ``path`` (train recipes only)."""
        return replace(self, output_dir=path)

    def render_command(self) -> list[str]:
        """alpamayo-recipes' launcher argv (run from ``recipes/alpamayo2_sft``)."""
        recipe = PUBLISHED_RECIPES[self.config_name]
        if recipe.multi_process:
            argv = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={len(self.gpus)}",
                "-m",
                recipe.entry,
            ]
        elif recipe.entry == "alpamayo2_sft.train_hf":
            raise AlpamayoSftError(f"recipe {self.config_name!r} cannot train without torchrun")
        else:
            argv = ["python", "-m", recipe.entry]
        argv += ["--config-path", "pkg://alpamayo2_sft/configs", "--config-name", self.config_name]
        is_train = recipe.entry == "alpamayo2_sft.train_hf"
        key = "model.pretrained_model_name_or_path" if is_train else "evaluate.eval_ckpt"
        argv.append(f"{key}={self.model}")
        if is_train:
            if self.output_dir is None:
                raise AlpamayoSftError(
                    f"recipe {self.config_name!r} trains and needs with_output_dir before rendering"
                )
            argv.append(f"data.train_dataset.local_dir={self.pai_dir}")
            argv.append(f"data.train_dataset.chunk_ids={self.train_chunks}")
        argv.append(f"data.val_dataset.local_dir={self.pai_dir}")
        argv.append(f"data.val_dataset.chunk_ids={self.val_chunks}")
        if self.output_dir is not None:
            argv.append(f"paths.output_dir={self.output_dir}")
        if is_train:
            # train_hf saves the Hydra config with resolve=True; in a single (non-multirun) job
            # three Hydra fields interpolate keys that exist only in multirun or that it drops
            # (runtime), so it raises before step 1. Literal values for those fields only.
            argv += [
                "hydra.sweep.subdir=0",
                "hydra.hydra_help.template=hydra_help",
                f"hydra.job_logging.handlers.file.filename={self.output_dir}/train_hf.log",
            ]
        argv.extend(recipe.overrides)
        argv.extend(self.extra_overrides)
        return argv

    def render_env(self) -> dict[str, str]:
        """Environment for the launcher: ``CUDA_VISIBLE_DEVICES`` from ``gpus``, if set."""
        return (
            {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in self.gpus)} if self.gpus else {}
        )


_README = "README.md, 'Outputs and Expected Signals'"

# alpamayo-recipes recipes/alpamayo2_sft README.md ("Smoke Test", "Validation Loss",
# "Trajectory Validation") transcribed; every number is the README's own, verbatim.
PUBLISHED_RECIPES: dict[str, AlpamayoSftRecipe] = {
    "sft_stage1_single_node_debug": AlpamayoSftRecipe(
        config_name="sft_stage1_single_node_debug",
        entry="alpamayo2_sft.train_hf",
        multi_process=True,
        published={
            "loss": PublishedLosses((1.8095, 1.7255), f"two Stage-1 steps, {_README}"),
        },
    ),
    "sft_stage2_single_node_debug": AlpamayoSftRecipe(
        config_name="sft_stage2_single_node_debug",
        entry="alpamayo2_sft.train_hf",
        multi_process=True,
        published={
            "loss": PublishedLosses((1.2710, 0.9367), f"two Stage-2 steps, {_README}"),
        },
    ),
    "sft_eval_loss": AlpamayoSftRecipe(
        config_name="sft_eval_loss",
        entry="alpamayo2_sft.evaluate_hf",
        multi_process=True,
        published={
            "eval_loss": PublishedMetric(
                "eval_loss", 1.699801, f"full held-out chunk-99 validation, {_README}"
            ),
        },
    ),
    "sft_eval_trajectory": AlpamayoSftRecipe(
        config_name="sft_eval_trajectory",
        entry="alpamayo2_sft.evaluate_hf",
        multi_process=False,
        published={
            "eval_min_ade": PublishedMetric(
                "eval_min_ade", 0.8288, f"all 100 chunk-99 clips, six samples, {_README}"
            ),
        },
        overrides=("evaluate.max_eval_steps=100",),
    ),
}
