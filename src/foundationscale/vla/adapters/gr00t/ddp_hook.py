"""Run GR00T's own ``launch_finetune.py`` with GR00T's own DDP path switched on.

GR00T's training config has ``use_ddp``, but ``launch_finetune.py`` does not expose it, so every
``num_gpus > 1`` run is routed to DeepSpeed -- which has no aarch64 build and compiles its ops
with nvcc, absent on our GB200 trays. This module sets ``config.training.use_ddp = True`` just
before GR00T's ``experiment.run`` and then executes GR00T's launcher unchanged: no upstream file
is edited (upstream LEDGER entry ``gr00t-launch-finetune-no-ddp-flag``). It runs INSIDE the GR00T
environment, started by torchrun: ``torchrun ... -m foundationscale.vla.adapters.gr00t.ddp_hook
<launch_finetune.py args>``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


def install() -> None:
    """Wrap ``gr00t.experiment.experiment.run`` so the run it receives uses DDP."""
    import gr00t.experiment.experiment as experiment  # noqa: PLC0415

    original = experiment.run

    def run(config: Any, *args: Any, **kwargs: Any) -> Any:
        config.training.use_ddp = True
        return original(config, *args, **kwargs)

    experiment.run = run


def launcher_path() -> Path:
    """GR00T's own ``gr00t/experiment/launch_finetune.py``, located from the installed package."""
    import gr00t  # noqa: PLC0415

    return Path(gr00t.__file__).resolve().parent / "experiment" / "launch_finetune.py"


def main(argv: list[str] | None = None) -> None:
    install()
    script = launcher_path()
    sys.argv = [str(script), *(sys.argv[1:] if argv is None else argv)]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
