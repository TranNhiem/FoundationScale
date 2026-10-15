"""Run Qwen3-TTS's official single-speaker SFT (finetuning/sft_12hz.py) unmodified, on SDPA.

Upstream hardcodes ``attn_implementation="flash_attention_2"``. flash-attn ships only as source and
has no aarch64 / CUDA 13 build in the GB200 worker env, so this wrapper patches
``Qwen3TTSModel.from_pretrained`` to request PyTorch's ``sdpa`` kernel instead (the same attention
computation, a different kernel) and then executes the upstream script as ``__main__`` with its own
arguments. Nothing else in the recipe changes.

Usage: qwen3tts_sft.py --upstream-script <Qwen3-TTS>/finetuning/sft_12hz.py -- <sft_12hz.py args>
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if "--upstream-script" not in argv or "--" not in argv:
        raise SystemExit("usage: qwen3tts_sft.py --upstream-script PATH -- <sft_12hz.py args>")
    script = argv[argv.index("--upstream-script") + 1]
    upstream_args = argv[argv.index("--") + 1 :]

    from qwen_tts import Qwen3TTSModel

    original = Qwen3TTSModel.from_pretrained.__func__

    def from_pretrained_sdpa(cls, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return original(cls, *args, **kwargs)

    Qwen3TTSModel.from_pretrained = classmethod(from_pretrained_sdpa)  # type: ignore[method-assign]
    # as `python sft_12hz.py` would: the script's own directory first on sys.path (it imports
    # its sibling dataset.py)
    sys.path.insert(0, str(Path(script).resolve().parent))
    sys.argv = [script, *upstream_args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
