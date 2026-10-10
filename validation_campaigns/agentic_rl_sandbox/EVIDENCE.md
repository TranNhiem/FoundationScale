# Agentic RL sandboxes — real-node evidence (plan v2, P0.4)

**Estate facts that shaped the design (measured).** GB200 trays are `aarch64` with no x86 emulation;
the only container runtime is `enroot` (no Docker, Podman, Apptainer or configured Kubernetes); Docker
Hub, GHCR, PyPI and GitHub are reachable. `enroot` bind-mounts the host `/tmp` into every container, and
`unshare -n` is refused inside a container but `unshare -rn enroot start …` runs with no network.

**What was run.** `real_sandbox.py` builds an arm64 sandbox image from `ctx/Dockerfile` with
`foundationscale.agentic_rl.sandbox.build.build_image` (a pre-`FROM` `ARG`, `RUN pip install pytest`,
`ENV`, `WORKDIR`, `COPY` of a toy repo), starts a no-network `EnrootSandbox` from it and checks it.

**Result (2026-10-11, one GB200 tray, twice):**

| Check | Result |
|---|---|
| Build (6 steps, `FROM python:${PY}-slim` with `PY=3.12`) | 19–26 s |
| Sandbox start from the built image | 7.3–7.4 s |
| `pytest` inside the sandbox | `1 passed` |
| `ENV` / `WORKDIR` from the build | `FOO=bar`, `/testbed` |
| Outbound network under `no-network` | blocked |
| Upload into / download out of the sandbox | correct |
| Private `/tmp` per trial | yes |
| Container removed on stop | yes |

**Security defects found in review and fixed before this run** (each now has a test that failed on the
original code): directory uploads could write through an agent-planted symlink onto the host;
directory downloads followed symlinks and could copy host files (e.g. a secrets file) into artifacts.

## Harbor end-to-end trial (P0.4b)

**What was run.** Harbor (pinned `harbor-framework/harbor@09f5b96`, installed in its own Python 3.12 venv
on a GB200 tray) ran its `examples/tasks/hello-alpine` task with the `oracle` agent, loading the FS
backend by import path:

```
harbor run --path <harbor>/examples/tasks/hello-alpine --agent oracle \
  --environment-import-path foundationscale.agentic_rl.sandbox.harbor_env:EnrootEnvironment \
  --ek data_root=<enroot data dir> --ek image_store=<image dir> --ek scratch_root=<scratch dir> \
  --jobs-dir <jobs dir>
```

**Result (2026-10-11):** 1 trial, 0 exceptions, **reward 1.0**, 24 s. The task's `Dockerfile`
(`FROM alpine:3.22`, `RUN apk add bash`, `WORKDIR /app`) was rebuilt for arm64 by `build_image`, the
oracle solution ran in an `EnrootSandbox`, and the task's own verifier (which installs `curl` and `uv`
from the internet and runs pytest) wrote the reward that Harbor collected through the backend's file
transfer.

**Defects this run found and fixed before the result above** (each now has a unit test):

1. Harbor's convention directories `/logs/{agent,user-agent,verifier,artifacts}` did not exist: the
   docker backend bind-mounts them, so the enroot backend now creates them at start.
2. `WORKDIR` recorded the default directory but did not create it, so every exec failed at
   `cd /app` -- Docker's `WORKDIR` creates the directory and the builder now does too. The earlier toy
   build masked this because its `COPY` happened to create the workdir.
