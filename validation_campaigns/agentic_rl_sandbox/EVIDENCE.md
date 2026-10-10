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
