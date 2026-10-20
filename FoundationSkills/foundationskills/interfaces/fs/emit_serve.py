"""Emit the fs_launch_spec of a vLLM deployment (A4.9): probed first, GPU-confined by construction, launched on a re-shown hash.

The server is the measured ``vllm031`` enroot container (vLLM 0.31.0, arm64). The device is confined INSIDE the
container (``NVIDIA_VISIBLE_DEVICES`` and ``CUDA_VISIBLE_DEVICES`` = the leased GPU UUID). Outside a Slurm step nothing
else confines it: on 2026-10-20 a server started this way ran on another user's GPU. ``emit_serve`` measures:

- the model directory has ``config.json`` (its ``model_type`` is the envelope family); a LoRA deployment's adapter has
  ``adapter_config.json`` and records the same base;
- a GPU UUID is given (``GPU-...``);
- the port is free on this host (bind test).

The launch runs the server in the foreground: put it under ``nohup`` or an ``srun --overlap`` step.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from foundationskills.interfaces.fs.emit_merge import adapter_base

CONTAINER = "vllm031"
ENROOT_DATA = str(Path.home() / ".enroot-vllm")  # absolute: launch runs argv without a shell


def model_family(model: Path) -> str | None:
    try:
        return json.loads((model / "config.json").read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError, AttributeError):
        return None


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def serve_probes(model: Path, adapter: Path | None, gpu: str, port: int) -> list[str]:
    gaps = []
    if not (model / "config.json").is_file():
        gaps.append(f"model {model} is not a model directory (no config.json)")
    if adapter is not None:
        if not (adapter / "adapter_config.json").is_file():
            gaps.append(f"adapter {adapter} has no adapter_config.json")
        else:
            recorded = adapter_base(adapter)
            if recorded and Path(recorded).name != model.name:
                gaps.append(f"adapter was trained on {recorded!r}, not the served base {str(model)!r}")
    if not gpu.startswith("GPU-"):
        gaps.append(f"gpu {gpu!r} is not a GPU UUID (lease one with fs-gpus claim and pass its UUID)")
    if not port_free(port):
        gaps.append(f"port {port} is already in use on this host")
    return gaps


def emit_serve(model: str, gpu: str, port: int = 18841, adapter: str | None = None, served_name: str = "model",
               modality: str = "text", max_model_len: int = 16384, gpu_memory_utilization: float = 0.85) -> dict[str, Any]:
    m = Path(model).expanduser()
    a = Path(adapter).expanduser() if adapter else None
    gaps = serve_probes(m, a, gpu, port)
    mode = "lora" if a else "full"
    argv = ["env", f"ENROOT_DATA_PATH={ENROOT_DATA}", "enroot", "start", "--rw", "--env", "HOME=/root",
            "--env", f"NVIDIA_VISIBLE_DEVICES={gpu}", "--env", f"CUDA_VISIBLE_DEVICES={gpu}", "--env", "HF_HUB_OFFLINE=1",
            "--mount", f"{m}:/models/served:x-create=dir,bind,ro"]
    if a:
        argv += ["--mount", f"{a}:/adapters/lora:x-create=dir,bind,ro"]
    argv += [CONTAINER, "/models/served", "--host", "127.0.0.1", "--port", str(port), "--served-model-name",
             served_name, "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization)]
    lora_name = None
    if a:
        lora_name = f"{served_name}-lora"
        rank = (json.loads((a / "adapter_config.json").read_text(encoding="utf-8")).get("r") or 16) \
            if (a / "adapter_config.json").is_file() else 16
        argv += ["--enable-lora", "--lora-modules", f"{lora_name}=/adapters/lora", "--max-lora-rank", str(rank)]
        if modality == "vision":
            argv += ["--enable-tower-connector-lora"]  # vLLM skips tower LoRA without it (experimental)
    return {
        "stage_name": "serve", "entry": "fskills-serve", "argv": argv, "env": {}, "sbatch": None, "dry_run_argv": None,
        "expected_outputs": [], "executable": not gaps, "missing": "; ".join(gaps) or None, "cwd": None,
        "serve": {"model": str(m), "adapter": str(a) if a else None, "family": model_family(m), "mode": mode,
                  "modality": modality, "gpu": gpu, "port": port, "endpoint": f"http://127.0.0.1:{port}",
                  "served_names": [served_name] + ([lora_name] if lora_name else []), "container": CONTAINER},
    }
