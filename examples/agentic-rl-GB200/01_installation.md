# 01: Installation

**What you have:** a GB200 tray (aarch64, CUDA) and a Python 3.12 environment you can install into.

## Versions we ran

MEASURED on GB200: every run in this example used one container image with these versions:

| Package | Version |
|---|---|
| python | 3.12.13 |
| vllm | 0.21.1rc1 (a nightly build) |
| torch | 2.11.0+cu130 |
| transformers | 5.9.0 |
| accelerate | 1.13.0 |
| abcMIDI (`abc2midi`) | 5.05, built from source |

Other versions may work, but we haven't measured them. The engine and the trainer ran from the same image.

## Step 1: install FoundationScale

From the repository root, inside that environment:

```bash
export PYTHONNOUSERSITE=1
pip install -e .
```

**TRAP (shared accounts):** a `~/.local/lib/python3.12/site-packages` on a shared account leaks into every environment.
pip then treats its packages as already installed. Set `PYTHONNOUSERSITE=1` in every shell; the training launcher sets it too.

## Step 2: build abc2midi

The music reward converts each submitted piece to MIDI with `abc2midi` from abcMIDI. Build it from the abcMIDI source
release and note the path of the binary; you pass it as `ABC2MIDI` later.

## Check it worked

```bash
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
python -m foundationscale.agentic_rl --help
/path/to/abc2midi -ver
```

Expected output (versions as in the table, then):

```
usage: foundationscale-agentic-rl [-h] --config CONFIG
                                  [--set dotted.key=value] [--dry-run]
...
5.05 October 02 2026 abc2midi
```

Next: [02_tasks_and_rewards.md](02_tasks_and_rewards.md).
