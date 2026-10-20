"""Emit the fs_launch_spec of a LoRA merge (A4.7): FS's own merge script, probed first, launched only on a re-shown hash.

The merge step is FoundationScale's ``examples/vlm-2xH200/scripts/merge_lora.py`` (PEFT ``merge_and_unload`` plus seven
structural checks and a results JSON); nothing here merges anything. ``emit_merge`` measures what the launch needs, and
leaves the spec non-executable with every gap named when a probe fails:

- base and adapter are directories; the adapter carries ``adapter_config.json`` and adapter weights;
- the adapter's recorded ``base_model_name_or_path`` resolves to the declared base (a different base is a different model);
- the output directory is absent or empty (a merge never overwrites);
- ``peft`` and ``transformers`` are importable in the interpreter that will run the script, and the script exists;
- free disk under the output's parent covers the merged size (base size x store-dtype factor, plus 10%).

The spec runs through ``fskills launch --spec ... --confirm <hash>`` like every launch: its plan hash is printed by
``fskills hash`` for the user to type back, never echoed in a refusal.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from foundationskills.interfaces.fs.emit_train import fs_repo_root

MERGE_SCRIPT = Path("examples/vlm-2xH200/scripts/merge_lora.py")
STORE_DTYPES = ("float32", "bfloat16")
_DTYPE_FACTOR = {"float32": 2.0, "bfloat16": 1.0}  # merged size relative to a bf16 base on disk


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def adapter_base(adapter: Path) -> str | None:
    """The base model the adapter was trained on, as its adapter_config.json records it (None when unreadable)."""
    try:
        return json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (OSError, ValueError, AttributeError):
        return None


def merge_probes(base: Path, adapter: Path, out: Path, store_dtype: str, python: str = sys.executable) -> list[str]:
    """Every measured gap that keeps the merge from running here; [] means it can run."""
    gaps: list[str] = []
    if store_dtype not in STORE_DTYPES:
        gaps.append(f"store_dtype {store_dtype!r} is not one of {STORE_DTYPES}")
    if not base.is_dir() or not (base / "config.json").is_file():
        gaps.append(f"base {base} is not a model directory (no config.json)")
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        gaps.append(f"adapter {adapter} has no adapter_config.json")
    elif not any((adapter / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        gaps.append(f"adapter {adapter} has no adapter weights (adapter_model.safetensors/.bin)")
    recorded = adapter_base(adapter) if adapter.is_dir() else None
    if recorded and base.is_dir():
        same = Path(recorded).expanduser().resolve() == base.resolve() if Path(recorded).expanduser().exists() \
            else Path(recorded).name == base.name
        if not same:
            gaps.append(f"adapter was trained on {recorded!r}, not the declared base {str(base)!r}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        gaps.append(f"output {out} exists and is not empty (a merge never overwrites)")
    root = fs_repo_root()
    if root is None or not (Path(root) / MERGE_SCRIPT).is_file():
        gaps.append(f"FoundationScale's {MERGE_SCRIPT} is not found (no FS checkout located)")
    if python == sys.executable:
        missing = [m for m in ("peft", "transformers", "torch") if importlib.util.find_spec(m) is None]
        if missing:
            gaps.append(f"not importable in {python}: {', '.join(missing)}")
    if base.is_dir() and store_dtype in _DTYPE_FACTOR:
        need = int(_dir_bytes(base) * _DTYPE_FACTOR[store_dtype] * 1.1)
        parent = out.parent if out.parent.exists() else Path.home()
        free = shutil.disk_usage(parent).free
        if free < need:
            gaps.append(f"free disk under {parent} is {free / 1e9:.1f} GB < {need / 1e9:.1f} GB needed")
    return gaps


def emit_merge(base: str, adapter: str, out: str, store_dtype: str = "float32", device_map: str = "cpu",
               python: str = sys.executable) -> dict[str, Any]:
    """The merge fs_launch_spec (schema fs_launch_spec, entry ``fskills-merge``)."""
    b, a, o = Path(base).expanduser(), Path(adapter).expanduser(), Path(out).expanduser()
    gaps = merge_probes(b, a, o, store_dtype, python)
    root = fs_repo_root()
    root = str(root) if root else None  # the spec schema types cwd as a string
    script = str(Path(root) / MERGE_SCRIPT) if root else str(MERGE_SCRIPT)
    results = str(o.parent / f"{o.name}.merge_results.json")
    argv = [python, script, "--base", str(b), "--adapter", str(a), "--out", str(o),
            "--dtype", store_dtype, "--device-map", device_map, "--results", results]
    return {
        "stage_name": "merge", "entry": "fskills-merge", "argv": argv, "env": {}, "sbatch": None,
        "dry_run_argv": None, "expected_outputs": [str(o), results], "executable": not gaps,
        "missing": "; ".join(gaps) or None, "cwd": root, "output_dir": str(o.parent),
        "merge": {"base": str(b), "adapter": str(a), "out": str(o), "results": results, "store_dtype": store_dtype,
                  "device_map": device_map, "adapter_recorded_base": adapter_base(a) if a.is_dir() else None},
    }
