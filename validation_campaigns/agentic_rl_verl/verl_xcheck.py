"""Cross-check the FS verl adapter against verl's REAL AgentLoopOutput and PEP 610 metadata."""

import json
import math

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from foundationscale.agentic_rl import trajectory_v2 as tv
from foundationscale.agentic_rl.verl_adapter import (
    VerlAdapterRefusal,
    episode_from_agent_loop_output,
    require_pinned_verl,
)

rep = {}
rep["pinned_commit"] = require_pinned_verl()
try:
    require_pinned_verl(pin="0" * 40)
    rep["wrong_pin"] = "DID-NOT-FIRE"
except VerlAdapterRefusal as e:
    rep["wrong_pin"] = "FIRED: " + str(e)[:110]

out = AgentLoopOutput(
    prompt_ids=[10, 11, 12],
    response_ids=[20, 21, 30, 31, 32, 22, 23],
    response_mask=[1, 1, 0, 0, 0, 1, 1],
    response_logprobs=[-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4],
    reward_score=1.0,
    num_turns=3,
    metrics=AgentLoopMetrics(),
)
ep = episode_from_agent_loop_output(
    out,
    episode_id="e0",
    group_id="g0",
    policy_version=7,
    model=tv.ModelMeta(name="m", version=7, checkpoint_sha="sha"),
    env=tv.EnvMeta(kind=tv.EnvKind.FS_LOCAL, task_id="t0"),
    gateway=tv.GatewayMeta(api_format=tv.ApiFormat.WHITE_BOX, session_id="s0"),
)
seg = ep.segments[0]
rep["spans"] = [list(x) for x in seg.generation_spans]
rep["mask"] = list(seg.loss_mask)
rep["nan_on_tool_tokens"] = all(
    math.isnan(lp) for lp, m in zip(seg.logprobs, seg.loss_mask, strict=True) if m == 0
)
rep["versions"] = list(seg.policy_version)
rep["reward_events"] = [(r.source, r.value) for r in ep.reward_events]
rep["v1_roundtrip_rows"] = len(tv.flatten_v2([ep])) if hasattr(tv, "flatten_v2") else "n/a"
print(json.dumps(rep, indent=1, default=str))
