"""Long-WAM LIBERO training through Long-WAM's own ``scripts/train.py``, from one spec.

Owned contract for Long-WAM fine-tuning: FoundationScale holds the run declaration and renders it
into the ``torchrun`` command Long-WAM's ``scripts/train.sh`` would run; Long-WAM trains. The spec
renders the entry (``scripts/train.py``) directly and never reimplements training, the model or
the schedule -- those stay in the Long-WAM checkout's own Hydra config tree (``configs/train.yaml``,
``configs/task``, ``configs/data``). :data:`PUBLISHED_RECIPES` transcribes
the released LIBERO recipe (``docs/TRAINING.md``, the recipes table) with the ranks, batch,
accumulation, schedule and the released checkpoints it publishes, so a reproduction is a rendered
command plus a published run to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "PUBLISHED_RECIPES",
    "LongWamFinetuneError",
    "LongWamFinetuneSpec",
    "LongWamRecipe",
]

_OWNED_KEYS = frozenset(
    {
        "task",
        "batch_size",
        "gradient_accumulation_steps",
        "output_dir",
        "expected_global_batch_size",
    }
)


class LongWamFinetuneError(ValueError):
    """A fine-tuning declaration Long-WAM cannot run as declared."""


@dataclass(frozen=True)
class LongWamRecipe:
    """A published Long-WAM recipe and the topology its training table records."""

    task: str
    expected_global_batch_size: int
    published_gpus: int  # the training table's GPU count (its "GPUs x batch x accumulation")
    per_rank_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    schedule: str
    released_checkpoints: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if self.expected_global_batch_size != (
            self.published_gpus * self.per_rank_batch_size * self.gradient_accumulation_steps
        ):
            raise LongWamFinetuneError(
                f"recipe {self.task!r} publishes a global batch of"
                f" {self.expected_global_batch_size}, not its own layout {self.published_gpus}"
                f"*{self.per_rank_batch_size}*{self.gradient_accumulation_steps}"
            )
        if not self.released_checkpoints:
            raise LongWamFinetuneError(f"recipe {self.task!r} publishes no checkpoint")


@dataclass(frozen=True)
class LongWamFinetuneSpec:
    """One Long-WAM fine-tuning run: a documented recipe, storage roots and GPU placement."""

    task: str = "libero"
    data_root: str = ""
    checkpoint_root: str = ""
    output_dir: str | None = None
    gpus: tuple[int, ...] = ()
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    extra_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in PUBLISHED_RECIPES:
            raise LongWamFinetuneError(
                f"task {self.task!r}; expected one of {sorted(PUBLISHED_RECIPES)}"
            )
        if not self.data_root.strip():
            raise LongWamFinetuneError(
                "data_root is empty; Long-WAM needs LONGWAM_DATA_ROOT for the LeRobot snapshots"
            )
        if not self.checkpoint_root.strip():
            raise LongWamFinetuneError(
                "checkpoint_root is empty; Long-WAM needs LONGWAM_CHECKPOINT_ROOT for the"
                " initialized action weights"
            )
        if not self.gpus:
            raise LongWamFinetuneError(
                "gpus is empty; fine-tuning runs under torchrun and needs GPUs"
            )
        if any(not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0 for gpu in self.gpus):
            raise LongWamFinetuneError(
                f"gpus {self.gpus!r}; Long-WAM expects non-negative CUDA ids"
            )
        if len(set(self.gpus)) != len(self.gpus):
            raise LongWamFinetuneError(f"gpus {self.gpus!r} repeats one CUDA id")
        for name in ("batch_size", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise LongWamFinetuneError(
                    f"{name} is {value!r}; expected a positive integer per rank"
                )
        recipe = PUBLISHED_RECIPES[self.task]
        actual = len(self.gpus) * self.batch_size * self.gradient_accumulation_steps
        if actual != recipe.expected_global_batch_size:
            raise LongWamFinetuneError(
                f"{len(self.gpus)} gpus * batch_size {self.batch_size} *"
                f" gradient_accumulation_steps {self.gradient_accumulation_steps} = {actual}, not"
                " the recipe's target"
                f" expected_global_batch_size {recipe.expected_global_batch_size}; Long-WAM's"
                " training contract refuses the launch, so adjust the layout first"
            )
        seen: set[str] = set()
        for override in self.extra_overrides:
            key = override.split("=", 1)[0].strip().lstrip("+")
            if key in _OWNED_KEYS:
                raise LongWamFinetuneError(
                    f"extra_overrides set {key!r}, which this spec already owns; change the field"
                )
            if key in seen:
                raise LongWamFinetuneError(
                    f"extra_overrides set {key!r} twice; Hydra keeps only the last"
                )
            seen.add(key)

    def on_gpus(self, gpus: tuple[int, ...]) -> LongWamFinetuneSpec:
        """The same recipe on ``gpus``, re-deriving accumulation to hold the global batch."""
        recipe = PUBLISHED_RECIPES[self.task]
        if not gpus or recipe.expected_global_batch_size % (len(gpus) * self.batch_size):
            raise LongWamFinetuneError(
                f"{len(gpus)} gpus * batch_size {self.batch_size} does not divide the recipe's"
                f" target expected_global_batch_size {recipe.expected_global_batch_size}"
            )
        return replace(
            self,
            gpus=gpus,
            gradient_accumulation_steps=recipe.expected_global_batch_size
            // (len(gpus) * self.batch_size),
        )

    def with_output_dir(self, path: str) -> LongWamFinetuneSpec:
        """The same recipe writing its ``output_dir`` to ``path``."""
        return replace(self, output_dir=path)

    def render_command(self) -> list[str]:
        """Long-WAM's ``scripts/train.py`` argv (run from the Long-WAM checkout)."""
        if self.output_dir is None:
            raise LongWamFinetuneError(
                "output_dir is None; training needs with_output_dir before rendering"
            )
        argv = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={len(self.gpus)}",
            "scripts/train.py",
            f"task={self.task}",
            f"batch_size={self.batch_size}",
            f"gradient_accumulation_steps={self.gradient_accumulation_steps}",
            f"output_dir={self.output_dir}",
        ]
        argv.extend(self.extra_overrides)
        return argv

    def render_env(self) -> dict[str, str]:
        """The environment ``scripts/train.py`` reads: roots, placement and the outputs root."""
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in self.gpus),
            "LONGWAM_DATA_ROOT": self.data_root,
            "LONGWAM_CHECKPOINT_ROOT": self.checkpoint_root,
        }
        if self.output_dir is not None:
            env["LONGWAM_OUTPUT_ROOT"] = self.output_dir
        return env


_TRAINING = "docs/TRAINING.md, 'Training recipes'"

# docs/TRAINING.md ("Training recipes" table) transcribed; the ranks column is the
# GPU count the recipe was trained with, and every number is the doc's own.
PUBLISHED_RECIPES: dict[str, LongWamRecipe] = {
    "libero": LongWamRecipe(
        task="libero",
        expected_global_batch_size=128,
        published_gpus=8,  # docs/TRAINING.md "8 × 8 × 2 | 128": 8 GPUs x batch 8 x accumulation 2
        per_rank_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        schedule="10 epochs, cosine",
        released_checkpoints=(
            "Efficient-Large-Model/Long-WAM-LIBERO-IDM",
            "Efficient-Large-Model/Long-WAM-LIBERO-COD",
        ),
        source=f"LIBERO row, {_TRAINING}",
    ),
}
