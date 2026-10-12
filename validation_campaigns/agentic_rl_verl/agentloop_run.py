# ruff: noqa: E501, ARG002, PTH  -- harness ported from verl tests/experimental/agent_loop (verl style kept)
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file (the AgentLoopManager bootstrap and the DataProto
# construction) are inlined from verl's tests/experimental/agent_loop/
# (agent_utils.py, test_basic_agent_loop.py, function_tool_examples.py),
# which are not installed with the verl package. Those files are Apache-2.0
# licensed as noted above.
"""Run verl's real AgentLoop (pinned 9058412) on one GPU and check the FS adapter.

Usage: python agentloop_run.py <model_path> <out_json>
"""

import json
import os
import sys
import tempfile

import numpy as np
import ray
import verl
from omegaconf import DictConfig
from transformers.utils import get_json_schema
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.function_tool import function_tool
from verl.tools.schemas import ToolResponse
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager

from foundationscale.agentic_rl.trajectory_v2 import (
    ApiFormat,
    EnvKind,
    EnvMeta,
    GatewayMeta,
    ModelMeta,
)
from foundationscale.agentic_rl.verl_adapter import VERL_PIN as PINNED_COMMIT  # noqa: E402
from foundationscale.agentic_rl.verl_adapter import (
    VerlAdapterRefusal,
    episode_from_agent_loop_output,
    require_pinned_verl,
    samples_from_rollout_batch,
)


# ---------------------------------------------------------------------------
# Inlined from tests/experimental/agent_loop/function_tool_examples.py
# ---------------------------------------------------------------------------
@function_tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum.

    Use this tool for any addition instead of doing mental arithmetic.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b


# ---------------------------------------------------------------------------
# Inlined from tests/experimental/agent_loop/test_basic_agent_loop.py
# ---------------------------------------------------------------------------
class AddTool(BaseTool):
    def add(self, a: int, b: int) -> int:
        """Add two integers and return their sum.

        Args:
            a: The first addend.
            b: The second addend.
        """
        return a + b

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.add)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(
        self, instance_id: str, parameters: dict, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.add(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


# ---------------------------------------------------------------------------
# Inlined from tests/experimental/agent_loop/agent_utils.py
# ---------------------------------------------------------------------------
def init_agent_loop_manager(config: DictConfig) -> AgentLoopManager | RayWorkerGroup:
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    actor_rollout_cls = ActorRolloutRefWorker
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    if config.reward.reward_model.enable:
        if config.reward.reward_model.enable_resource_pool:
            mapping[Role.RewardModel] = "reward_pool"
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError(
                    "config.reward.reward_model.n_gpus_per_node must be greater than 0"
                )
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            reward_pool = [
                config.reward.reward_model.n_gpus_per_node
            ] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            mapping[Role.RewardModel] = global_pool_id
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    device_name = get_device_name()
    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=device_name
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    if config.actor_rollout_ref.rollout.mode == "sync":
        raise ValueError(
            "Agent loop tests require async rollout mode. Please set rollout.mode=async."
        )

    # =========================== 2. Create AgentLoopManager ===========================
    rm_resource_pool = (
        resource_pool_manager.get_resource_pool(Role.RewardModel)
        if config.reward.reward_model.enable
        else None
    )
    reward_loop_manager = RewardLoopManager(
        config=config,
        rm_resource_pool=rm_resource_pool,
    )
    llm_server_manager = LLMServerManager.create(config=config, worker_group=actor_rollout_wg)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=reward_loop_manager.reward_loop_workers,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        actor_wg=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()

    return agent_loop_manager


def build_config(model_path: str, function_tool_path: str) -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=os.path.abspath(config_dir)):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "reward.reward_manager.name=dapo",
                "+reward.reward_kwargs.overlong_buffer_cfg.enable=False",
                "+reward.reward_kwargs.overlong_buffer_cfg.len=3072",
                "+reward.reward_kwargs.max_resp_len=4096",
            ],
        )

    config.actor_rollout_ref.model.path = model_path
    # No FlashAttention wheel exists for this aarch64 container: score with SDPA and keep
    # padded batches (remove-padding needs FlashAttention's varlen kernels).
    from omegaconf import open_dict

    with open_dict(config):
        config.actor_rollout_ref.model.override_config = {"attn_implementation": "sdpa"}
        config.actor_rollout_ref.model.use_remove_padding = False
        # MiMo-V2.6 (Qwen3.6 family) emits Qwen3-Coder XML tool calls
        # (<function=add><parameter=a>..), not hermes JSON; measured on the first real run.
        config.actor_rollout_ref.rollout.multi_turn.format = "qwen3_coder"
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 1024
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.rollout.calculate_log_probs = True
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.4
    config.actor_rollout_ref.rollout.agent.num_workers = 2
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    config.actor_rollout_ref.rollout.multi_turn.function_tool_path = function_tool_path
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 2
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 3
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = 1
    config.reward.reward_model.enable = False

    return config


def build_prompts() -> list[list[dict]]:
    texts = [
        "Use the add tool to compute 17+25.",
        "Use the add tool to compute 3+4.",
        "Use the add tool to compute 10+32.",
        "Use the add tool to compute 7+8.",
        "Use the add tool to compute 100+11.",
        "Use the add tool to compute 5+6.",
        "Use the add tool to compute 21+22.",
        "Use the add tool to compute 9+10.",
    ]
    return [[{"role": "user", "content": t}] for t in texts]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <model_path> <out_json>", file=sys.stderr)
        return 1
    model_path = os.path.abspath(sys.argv[1])
    out_json = os.path.abspath(sys.argv[2])

    require_pinned_verl()

    function_tool_path = os.path.join(
        tempfile.mkdtemp(prefix="fsar_tools_"), "function_tool_examples.py"
    )
    with open(function_tool_path, "w") as f:
        f.write(FUNCTION_TOOL_SOURCE)

    config = build_config(model_path, function_tool_path)

    ray.init(
        num_gpus=1,
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        agent_loop_manager = init_agent_loop_manager(config)

        raw_prompts = build_prompts()
        batch = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array([np.array(p) for p in raw_prompts], dtype=object),
                "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
                "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
                "reward_model": np.array(
                    [{"style": "rule", "ground_truth": "<answer>"}] * len(raw_prompts), dtype=object
                ),
            },
        )
        out = agent_loop_manager.generate_sequences(prompts=batch)

        tensor_batch = {k: v for k, v in out.batch.items()}
        non_tensor_batch = dict(out.non_tensor_batch)
        # Raw batch to disk first, so the adapter can be re-checked offline without a GPU rerun.
        raw_path = os.path.splitext(sys.argv[2])[0] + ".raw_batch.json"
        with open(raw_path, "w") as fh:
            json.dump(
                {
                    "batch": {k: v.tolist() for k, v in tensor_batch.items()},
                    "non_tensor": {
                        k: [x.item() if hasattr(x, "item") else str(x) for x in v]
                        for k, v in non_tensor_batch.items()
                        if k in ("__num_turns__", "data_source")
                    },
                },
                fh,
            )
        samples = samples_from_rollout_batch(tensor_batch, non_tensor_batch)

        episodes = []
        refusals = []
        for i, sample in enumerate(samples):
            try:
                episodes.append(
                    episode_from_agent_loop_output(
                        sample,
                        episode_id=f"verl-{i}",
                        group_id="g0",
                        policy_version=0,
                        model=ModelMeta(name="policy", version=0, checkpoint_sha="base"),
                        env=EnvMeta(kind=EnvKind.FS_LOCAL, task_id=f"t{i}"),
                        gateway=GatewayMeta(api_format=ApiFormat.WHITE_BOX, session_id=f"s{i}"),
                    )
                )
            except VerlAdapterRefusal as e:
                refusals.append(str(e))

        turns = [int(t) for t in non_tensor_batch.get("__num_turns__", [])]
        response_mask = tensor_batch["response_mask"]
        attention_mask = tensor_batch["attention_mask"]
        prompt_length = tensor_batch["prompts"].size(1)
        tool_tokens_per_sample = []
        model_tokens_per_sample = []
        for i in range(len(samples)):
            resp_valid = (
                attention_mask[i, prompt_length : prompt_length + response_mask.size(1)] > 0
            )
            rm = response_mask[i][resp_valid]
            tool_tokens_per_sample.append(int((rm == 0).sum().item()))
            model_tokens_per_sample.append(int((rm != 0).sum().item()))

        rm_scores = tensor_batch.get("rm_scores")
        if rm_scores is not None:
            reward_scores = [float(rm_scores[i].sum().item()) for i in range(len(samples))]
        else:
            reward_scores = []

        result = {
            "pinned_commit": PINNED_COMMIT,
            "samples": len(samples),
            "valid_episodes": len(episodes),
            "refusals": refusals,
            "turns": turns,
            "tool_tokens_per_sample": tool_tokens_per_sample,
            "model_tokens_per_sample": model_tokens_per_sample,
            "reward_scores": reward_scores,
        }
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))

        return 0 if (len(episodes) == len(samples) and len(samples) > 0) else 1
    finally:
        ray.shutdown()


FUNCTION_TOOL_SOURCE = '''\
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inlined from verl tests/experimental/agent_loop/function_tool_examples.py."""

from verl.tools.function_tool import function_tool


@function_tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum.

    Use this tool for any addition instead of doing mental arithmetic.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b
'''


if __name__ == "__main__":
    sys.exit(main())
