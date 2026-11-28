"""engine.replica_gpus: one engine server per GPU, distinct ports and CUDA_VISIBLE_DEVICES."""

import json
from pathlib import Path

import pytest

from foundationscale.agentic_rl import cli
from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config

CFG = (
    Path(__file__).resolve().parents[2]
    / "examples/agentic-rl-GB200/configs/harbor_qwen35_4b_2gpu.json"
)


def _config(overrides):
    s = CFG.read_text(encoding="utf-8")
    for k, v in {
        "@MODEL@": "/m",
        "@OUT@": "/o",
        "@PORT@": "18400",
        '"@PORT_INT@"': "18400",
        "@HOME@": "/h",
        "@ENGINE_GPU@": "0",
        "@TASKS@": "/t.jsonl",
    }.items():
        s = s.replace(k, v)
    return build_config(json.loads(s), overrides)


def test_default_is_a_single_engine():
    servers = cli._engine_servers(_config({}))
    assert len(servers) == 1


def test_replicas_get_their_own_port_and_gpu():
    servers = cli._engine_servers(_config({"engine.replica_gpus": "1,2,3"}))
    specs = [s.spec if hasattr(s, "spec") else s._spec for s in servers]
    assert [sp.port for sp in specs] == [18400, 18401, 18402]
    assert [sp.env["CUDA_VISIBLE_DEVICES"] for sp in specs] == ["1", "2", "3"]
    for sp in specs:
        cmd = list(sp.command)
        assert cmd[cmd.index("--port") + 1] == str(sp.port)


def test_bad_replica_list_is_refused():
    with pytest.raises(AgenticConfigRefusal):
        _config({"engine.replica_gpus": "0,x"})
