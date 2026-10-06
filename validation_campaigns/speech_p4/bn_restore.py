"""Fine-tuned Parakeet weights + the BASE model's BatchNorm running statistics -> new dir."""

import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

base, ft, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
for f in ft.iterdir():
    if f.suffix != ".safetensors":
        shutil.copy(f, out / f.name)
b = {}
for f in base.glob("*.safetensors"):
    b.update(load_file(str(f)))
n = 0
for f in ft.glob("*.safetensors"):
    t = load_file(str(f))
    for k in t:
        if k.endswith((".running_mean", ".running_var", ".num_batches_tracked")) and k in b:
            t[k] = b[k].to(t[k].dtype)
            n += 1
    save_file(t, str(out / f.name))
print("BN restored buffers:", n)
