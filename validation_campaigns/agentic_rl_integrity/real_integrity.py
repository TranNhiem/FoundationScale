"""Real-hardware check of the P0.7 sandbox mechanisms on an enroot node.

Reuses the toy image built by validation_campaigns/agentic_rl_sandbox/real_sandbox.py.
Prints one JSON report; every MUST_FIRE / MUST_PASS outcome is a field.
"""

import json
import os
import shutil
import sys
from pathlib import Path

from foundationscale.agentic_rl.integrity import IntegrityRefusal
from foundationscale.agentic_rl.sandbox.enroot import EnrootSandbox, SandboxSpec
from foundationscale.agentic_rl.sandbox.hidden import HiddenSpec, HiddenVault, probe_egress

H = Path(os.environ["HOME"])
W = H / "fsar-data" / "sbxreal"
roots = dict(
    data_root=str(H / ".enroot-sbx"), image_store=str(W / "images"), scratch_root=str(W / "scratch")
)
image = str(W / "toy.sqsh")
SEED = "mkdir -p /tests /testbed/.git && echo gold > /tests/t.py && echo ref > /testbed/.git/HEAD"
rep: dict[str, object] = {}
for stale in W.glob("vault-*"):  # a refused unseal leaves its vault behind by design
    shutil.rmtree(stale)


def fires(fn) -> str:
    try:
        fn()
    except IntegrityRefusal as e:
        return f"FIRED: {str(e)[:120]}"
    return "DID-NOT-FIRE"


# 1. Seal / unseal round trip with an agent phase in between.
sb = EnrootSandbox(SandboxSpec(name="integ-seal", image=image, network="no-network", **roots))
sb.start()
try:
    sb.exec(SEED)
    vault = HiddenVault(sb, HiddenSpec(git_worktrees=("/testbed",)), W / "vault-seal")
    rep["sealed"] = vault.seal()
    vault.assert_sealed()
    agent = sb.exec("ls /tests /testbed/.git 2>&1; cat /tests/t.py 2>&1; echo rc=$?")
    rep["agent_view"] = agent.stdout.strip().splitlines()[-3:]
    rep["unsealed"] = vault.unseal()
    rep["verifier_view"] = sb.exec("cat /tests/t.py /testbed/.git/HEAD").stdout.split()
finally:
    sb.stop()

# 2. Attacks during the agent phase: dangling symlink, planted file, symlinked parent.
for attack, cmd in [
    ("dangling_symlink", "ln -s /nowhere /tests"),
    ("planted_file", "mkdir -p /tests && echo fake > /tests/t.py"),
    (
        "symlinked_parent",
        "rm -rf /testbed && mkdir -p /elsewhere/.git && ln -s /elsewhere /testbed",
    ),
]:
    name = f"integ-{attack.replace('_', '-')}"
    sb = EnrootSandbox(SandboxSpec(name=name, image=image, network="no-network", **roots))
    sb.start()
    try:
        sb.exec(SEED)
        vault = HiddenVault(sb, HiddenSpec(git_worktrees=("/testbed",)), W / f"vault-{attack}")
        vault.seal()
        sb.exec(cmd)
        rep[f"attack_{attack}"] = {
            "assert_sealed": fires(vault.assert_sealed),
            "unseal": fires(vault.unseal),
        }
    finally:
        sb.stop()

# 3. Egress probe: must be False under no-network and True under public.
for net in ("no-network", "public"):
    sb = EnrootSandbox(SandboxSpec(name=f"integ-net-{net}", image=image, network=net, **roots))
    sb.start()
    try:
        rep[f"egress_{net}"] = probe_egress(sb)
    finally:
        sb.stop()

rep["arch"] = os.uname().machine
json.dump(rep, sys.stdout, indent=1, default=str)
print()
