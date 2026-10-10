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
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /opt/FoundationScale
COPY examples/vlm-2xH200/install/requirements-vlm.txt /tmp/requirements-vlm.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-vlm.txt
COPY . /opt/FoundationScale
RUN pip install -e ".[train]" && python -c "import foundationscale, torch, transformers, peft, liger_kernel; print('ok', torch.__version__, transformers.__version__)"
