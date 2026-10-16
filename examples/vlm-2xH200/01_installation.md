# 01 — Installation: conda environment or Docker image

In this chapter you clone the repo, pick **one** environment (conda, uv/venv, or Docker), install the VLM training stack, and download the target models. By the end you have a working `foundationscale-train` and the weights on disk.

**Target machine:** a 2x H200 machine (Ubuntu 22.04) — 2x NVIDIA H200 (141 GB each), driver 580 / CUDA 13.0, 44 CPU cores, 472 GB RAM. Docker is not preinstalled (Option C shows the install) and there is no system CUDA toolkit (no `nvcc`) — this matters later.

**Time needed:** about 7 min for the environment + install (measured breakdown: Miniforge 15 s, `conda env create` 3.3 min, `pip install -e` 1 min, `flash-linear-attention` 1.5 min). Model downloads depend on network speed.

All commands below were run on a 2x H200 machine, including the Docker path (Option C).

---

## Target models

Use these names for downloads and later runs:

- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-27B`
- `google/gemma-4-31B-it`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-12B-it`

---

## Step 0 — Clone the repo

Run everything from the repository root unless stated otherwise.

```bash
git clone https://github.com/TranNhiem/FoundationScale.git && cd FoundationScale
```

From here on, every command runs from the repository root `FoundationScale/` unless stated.

---

## Check the GPUs first

**Before any run:** GPUs may already be in use by other processes on a shared machine. Check and free them first:

```bash
nvidia-smi
```

Now pick **one** of the three options below.

---

## Option A — conda environment

### Step A1 — Install Miniforge

```bash
curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash mf.sh -b -p $HOME/miniforge && source $HOME/miniforge/etc/profile.d/conda.sh
```

### Step A2 — Create and activate the env

```bash
cd examples/vlm-2xH200/install && conda env create -f environment.yml && conda activate fs-vlm && cd ../../..
```

### Step A3 — Install FoundationScale in editable mode

```bash
pip install -e ".[train]"
```

### Check it worked

```bash
python -c "import torch,transformers,peft,liger_kernel,av,foundationscale;print(torch.__version__,torch.cuda.is_available(),transformers.__version__,peft.__version__)"
```

Expected output (exact):

```
2.13.0+cu132 True 5.18.0 0.21.2
```

```bash
which foundationscale-train
```

Expected output (measured): `<your-env>/bin/foundationscale-train`

GPU sanity check:

```bash
python -c "import torch;print(torch.cuda.device_count())"
```

Expected output (exact):

```
2
```

### Files used

`examples/vlm-2xH200/install/environment.yml`:

```yaml
# conda env create -f environment.yml && conda activate fs-vlm
# then, from the repository root:  pip install -e ".[train]"
name: fs-vlm
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - ffmpeg
  - git
  - pip:
      - -r requirements-vlm.txt
```

`examples/vlm-2xH200/install/requirements-vlm.txt` (the pins verified together on 2x H200, driver 580 / CUDA 13.0, on 2026-10-08):

```
# Versions verified together on 2x H200 (driver 580, CUDA 13.0) on 2026-10-08.
# torch/torchvision come from the PyTorch cu132 index (see environment.yml / Dockerfile).
--extra-index-url https://download.pytorch.org/whl/cu132
torch==2.13.0
torchvision==0.28.0
transformers==5.18.0
peft==0.21.2
accelerate==1.15.0
datasets==5.1.0
liger-kernel==0.8.4
av==19.0.1
pillow==12.3.0
safetensors==0.8.0
soundfile==0.14.0
```

---

## Option B — uv + venv (fastest)

### Step B1 — Install uv and create a venv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 venv
```

### Step B2 — Install the stack

```bash
uv pip install --python venv/bin/python -r examples/vlm-2xH200/install/requirements-vlm.txt --index-strategy unsafe-best-match
uv pip install --python venv/bin/python -e ".[train]"
```

### Step B3 — Activate the venv

```bash
source venv/bin/activate
```

### Check it worked

Run the same checks as in Option A, but through the venv:

```bash
venv/bin/python -c "import torch,transformers,peft,liger_kernel,av,foundationscale;print(torch.__version__,torch.cuda.is_available(),transformers.__version__,peft.__version__)"
venv/bin/python -c "import torch;print(torch.cuda.device_count())"
```

Expected output: `2.13.0+cu132 True 5.18.0 0.21.2` on the first line (measured on the uv path with the same pinned requirements as Option A), and `2` on the second.

`which foundationscale-train` (or `venv/bin/foundationscale-train`): expected output (measured) is a path inside the env, i.e. `venv/bin/foundationscale-train`.

**Every later chapter assumes this environment is active in the shell** (conda: `conda activate fs-vlm`; uv: `source venv/bin/activate`). Otherwise `python`/`torchrun` resolve to the system Python 3.10 and `torchrun` is "not found".

---

## Option C — Docker image

**Docker is not preinstalled on this machine.** Installing Docker Engine + the NVIDIA Container Toolkit needs **root** (about 2 min, Docker 29.9):

```bash
apt-get update && apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" \
  > /etc/apt/sources.list.d/docker.list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L   # lists both H200s
```

Upstream guides: https://docs.docker.com/engine/install/ubuntu/ and
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### Step C1 — Build (from the repository root)

```bash
docker build -f examples/vlm-2xH200/install/Dockerfile -t foundationscale-vlm .
```

Measured: about 6 min from a fresh clone; the image is 14.5 GB.

### Step C2 — Run (models and data mounted from the host)

```bash
docker run --gpus all --ipc=host --shm-size 64g -v /workspace:/workspace -it foundationscale-vlm
```

### Check it worked

The last build step runs an import check and prints:

```
ok 2.13.0+cu132 5.18.0
```

Inside the container, `python -c "import torch;print(torch.cuda.device_count())"` prints `2`.

Verified inside the container: `scripts/train_sft.sh` (`STEPS=10`) trains gemma-4-12B-it and Qwen3.6-35B-A3B at 16384 on the image + video + text mix — exit 0, adapters saved, save gates PASS. One caveat: the run manifest reports `code.status=not_a_repository` because the working directory is the workspace, not the repository; that is a warning, not a failure.

### The Dockerfile

`examples/vlm-2xH200/install/Dockerfile`:

```dockerfile
# Build from the repository root:
#   docker build -f examples/vlm-2xH200/install/Dockerfile -t foundationscale-vlm .
# Run (models and data mounted from the host):
#   docker run --gpus all --ipc=host --shm-size 64g -v /workspace:/workspace -it foundationscale-vlm
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev git ffmpeg curl ca-certificates \
    || (apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa \
        && apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev git ffmpeg curl) \
    && rm -rf /var/lib/apt/lists/*
# Triton (liger fused loss, flash-linear-attention) compiles a C launcher at first use.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /opt/FoundationScale
COPY examples/vlm-2xH200/install/requirements-vlm.txt /tmp/requirements-vlm.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-vlm.txt
# Qwen3.6 gated-delta layers: 502 -> 1461-3156 tokens/s at 32K (01_installation.md).
RUN pip install flash-linear-attention==0.5.2
COPY . /opt/FoundationScale
RUN pip install -e ".[train]" && python -c "import foundationscale, torch, transformers, peft, liger_kernel, fla; print('ok', torch.__version__, transformers.__version__)"
```

The `gcc` line is required: Triton (used by the fused loss and flash-linear-attention) compiles a small C launcher the first time a kernel runs, and the CUDA runtime image ships no compiler. Without it training fails at step 1 with `Failed to find C compiler`.

---

## Step (all options) — Speed-up for Qwen3.6: flash-linear-attention

The Qwen3.6 models (`Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-27B`) need **flash-linear-attention** for speed (measured pin: 0.5.2):

```bash
pip install flash-linear-attention==0.5.2
```

(Use `uv pip install --python venv/bin/python flash-linear-attention==0.5.2` on the uv path. The Docker image already includes it.)

Measured on this setup (Qwen3.6-27B LoRA at 32K):

| | tokens/s |
|---|---|
| before | 502 |
| after | 1461 |

That's a **2.9x** speedup.

**About `causal-conv1d`:** it is **optional** and **cannot be built on this machine**, because it needs `nvcc` from a CUDA toolkit and only the driver is installed (no system CUDA toolkit). Skip it — the remaining slow path is small. The Docker **devel** route can build it (not measured).

---

## Step (all options) — Download the weights

Log in first for faster, rate-limit-free downloads:

```bash
hf auth login
```

Then download from the repository root:

```bash
hf download Qwen/Qwen3.6-27B --local-dir models/Qwen3.6-27B
hf download Qwen/Qwen3.6-35B-A3B --local-dir models/Qwen3.6-35B-A3B
hf download google/gemma-4-31B-it --local-dir models/gemma-4-31B-it
hf download google/gemma-4-26B-A4B-it --local-dir models/gemma-4-26B-A4B-it
hf download google/gemma-4-12B-it --local-dir models/gemma-4-12B-it
```

(measured) sizes on disk:

| Model | On disk |
|---|---|
| `Qwen/Qwen3.6-27B` | 55.6 GB |
| `Qwen/Qwen3.6-35B-A3B` | 71.9 GB |
| `google/gemma-4-31B-it` | 62.5 GB |
| `google/gemma-4-26B-A4B-it` | 51.6 GB |
| `google/gemma-4-12B-it` | 23.9 GB |
| **Total** | **~266 GB** |

Disk space (measured): you need **~300 GB** free for all five models plus scratch space. Run `df -h` before you start, and download only the models you'll tune if space is tight.

### Check it worked

```bash
ls models/Qwen3.6-27B
```

Expected output: the model's weight and config files in `models/Qwen3.6-27B/` (exact listing not measured).

---

## Machine notes you must know before the next chapter

1. **`localhost` resolves only to the IPv6 address `::1` on this machine.** Because of that, `torchrun --standalone` **hangs forever in rendezvous**, and the default hostname resolves to a public address that times out. Always launch single-node runs like this:

   ```bash
   torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 -m foundationscale.train ...
   ```

   Never use `torchrun --standalone ...` on this machine.

2. **GPUs may already be in use.** Check `nvidia-smi` and free them before starting a run.

---

## Troubleshooting

- **`torchrun` hangs forever at rendezvous / can't connect to the default address (timeouts).** See note 1 above. Use the explicit `--master_addr 127.0.0.1 --master_port 29500` launch line and drop `--standalone`.

- **`nvidia-smi` shows other processes on the GPUs / runs fail with out-of-memory right away.** See note 2: free the GPUs first.

- **Import check prints different versions than `2.13.0+cu132 True 5.18.0 0.21.2`.** Make sure the right env is active and that you installed the repo from the repository root with `pip install -e ".[train]"` (`uv pip install --python venv/bin/python -e ".[train]"` on the uv path). The torch/torchvision wheels come from the PyTorch cu132 index (`--extra-index-url https://download.pytorch.org/whl/cu132` is in `requirements-vlm.txt`).

- **`foundationscale-train: command not found`.** The env isn't active, or Step A3/B2 didn't run. Check with `which foundationscale-train` — it should print a path inside the env.

- **`causal-conv1d` fails to build** with errors about `nvcc` / CUDA toolkit. Expected: this machine has no system CUDA toolkit. `causal-conv1d` is optional — leave it out. Only the Docker devel route can build it (not measured).

- **Qwen3.6 runs are slow (hundreds of tokens/s instead of ~1.5k at 32K).** Install `flash-linear-attention` (see above).

- **`docker: command not found`.** Docker is not preinstalled, and installing Docker Engine + the NVIDIA Container Toolkit needs root (links in Option C). If you don't have root, use Option A or B.

- **Model download is slow or gets rate-limited.** Run `hf auth login` first.

---

**Next:** [02 — Data preparation](02_data_preparation.md)
