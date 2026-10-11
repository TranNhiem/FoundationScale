"""End-to-end: mini-SWE-agent (host) -> FS gateway -> vLLM; commands in an FS enroot sandbox."""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from transformers import AutoTokenizer

from foundationscale.agentic_rl.cli import _HFChatTokenizer
from foundationscale.agentic_rl.engines.vllm import VLLMClient
from foundationscale.agentic_rl.gateway.server import GatewayConfig, GatewayCore, serve
from foundationscale.agentic_rl.gateway.session import Mode, SamplingContract
from foundationscale.agentic_rl.sandbox.build import build_image
from foundationscale.agentic_rl.sandbox.mini_swe_env import EnrootMiniSweEnvironment
from foundationscale.agentic_rl.tools import QwenXmlToolCallParser

vllm_url, model_path, task_dir, out = sys.argv[1:5]
H = Path(os.environ["HOME"])
W = H / "fsar-data" / "agente2e-work"
W.mkdir(parents=True, exist_ok=True)
roots = dict(
    data_root=str(H / ".enroot-sbx"), image_store=str(W / "images"), scratch_root=str(W / "scratch")
)
for d in roots.values():
    Path(d).mkdir(parents=True, exist_ok=True)
rep = {}
t = time.time()
img = W / "toy_swe.sqsh"
if not img.exists():
    build_image(Path(task_dir) / "Dockerfile", task_dir, img, **roots)
rep["build_s"] = round(time.time() - t, 1)
tok = AutoTokenizer.from_pretrained(model_path)
core = GatewayCore(
    GatewayConfig(
        served_model_name="policy",
        sampling=SamplingContract(mode=Mode.TRAIN, temperature=1.0, max_tokens_cap=2048),
        admin_key="e2e-admin",
        base_url="http://127.0.0.1:0",
    ),
    client=VLLMClient(base_url=vllm_url, timeout_s=600.0, served_model_name="policy"),
    tokenizer=_HFChatTokenizer(tokenizer=tok),
    parser=QwenXmlToolCallParser(),
    decode=lambda ids: tok.decode(list(ids), skip_special_tokens=False),
)
srv = serve(core, "127.0.0.1", 0)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = core.advertised_base_url
admin = {"Authorization": "Bearer e2e-admin", "Content-Type": "application/json"}


def post(path, body):
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(base + path, json.dumps(body).encode(), admin), timeout=120
        )
    except urllib.error.HTTPError as e:
        rep.setdefault("http_errors", []).append(
            {"path": path.split("/")[-1], "code": e.code, "body": e.read().decode()[:600]}
        )
        print(json.dumps(rep))
        raise
    return json.loads(r.read() or b"null")


sess = post("/fs/v1/sessions", {"mode": "train", "group_id": "toy-swe"})
sid = sess["session_id"]
cfg = yaml.safe_load(
    (Path(sys.modules["minisweagent"].__file__).parent / "config" / "default.yaml").read_text()
)
agent_cfg = dict(cfg.get("agent", {}))
agent_cfg["step_limit"] = 12
model = LitellmModel(
    model_name="openai/policy",
    cost_tracking="ignore_errors",
    model_kwargs={
        "api_base": sess["base_urls"]["chat_completions"],
        "api_key": sess["api_key"],
        "temperature": 1.0,
    },
)
env = EnrootMiniSweEnvironment(
    image=str(img), network="no-network", cwd="/testbed", timeout=60, **roots
)
task = (
    "The test suite in /testbed fails. Find and fix the bug in calc.py so that "
    "`python -m pytest -q` passes. "
    "Do not edit the tests."
)
t = time.time()
try:
    agent = DefaultAgent(model, env, **agent_cfg)
    result = agent.run(task)
    rep["agent_exit"] = str(result)[:200]
except Exception as e:  # noqa: BLE001
    rep["agent_exit"] = f"{type(e).__name__}: {e}"[:300]
rep["agent_s"] = round(time.time() - t, 1)
check = env.sandbox.exec("python -m pytest -q 2>&1 | tail -1", cwd="/testbed")
rep["tests_after"] = check.stdout.strip()
reward = 1.0 if " passed" in check.stdout and "failed" not in check.stdout else 0.0
rep["reward"] = reward
rep["calc_py"] = env.sandbox.exec("cat calc.py", cwd="/testbed").stdout
rep["gateway_calls"] = len(core.store.get(sid).calls)
env.cleanup()
post(
    f"/fs/v1/sessions/{sid}/reward",
    {
        "source": "tests",
        "verifier_id": "pytest",
        "verifier_version": "1",
        "value": reward,
        "scope": "episode",
    },
)
post(f"/fs/v1/sessions/{sid}/end", {"status": "ok", "termination": "stop"})
ep = post(
    f"/fs/v1/sessions/{sid}/trajectory",
    {
        "model": {"name": "policy", "version": 0, "checkpoint_sha": "e2e"},
        "harness": {"name": "mini-swe-agent", "version": "2"},
        "env": {"kind": "fs_local", "task_id": "toy-swe"},
    },
)
segs = ep["segments"]
rep["segments"] = len(segs)
rep["spans_per_segment"] = [len(s["generation_spans"]) for s in segs]
rep["trainable_tokens"] = sum(sum(s["loss_mask"]) for s in segs)
rep["episode_reward"] = [e["value"] for e in ep["reward_events"]]
Path(out).write_text(json.dumps(ep))
print(json.dumps(rep))
