"""require_pinned_verl against verl's REAL PEP 610 metadata (pinned git install)."""

import json

from foundationscale.agentic_rl.verl_adapter import VerlAdapterRefusal, require_pinned_verl

rep = {"pinned_commit": require_pinned_verl()}
for name, kw in [
    ("wrong_pin", {"pin": "0" * 40}),
    ("absent_dist", {"distribution": "verl-absent"}),
]:
    try:
        require_pinned_verl(**kw)
        rep[name] = "DID-NOT-FIRE"
    except VerlAdapterRefusal as e:
        rep[name] = "FIRED: " + str(e)[:120]
print(json.dumps(rep, indent=1))
