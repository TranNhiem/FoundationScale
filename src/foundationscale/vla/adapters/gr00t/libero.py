"""GR00T N1.x policies on LIBERO through GR00T's public policy-server protocol.

The observation is GR00T's own LIBERO encoding (``gr00t/eval/sim/LIBERO/libero_env.py``):
both cameras rotated 180 degrees as ``video.image`` / ``video.wrist_image``, state split into
``state.x .. state.yaw`` (end-effector position and robosuite's ``quat2axisangle``) and the
two-joint ``state.gripper``, the instruction under ``annotation.human.action.task_description``
-- batched ``(B=1, T=1)`` the way GR00T's multi-step wrapper hands it to the server started with
``--use-sim-policy-wrapper``. Each returned step is concatenated ``x, y, z, roll, pitch, yaw,
gripper`` and post-processed with GR00T's own ``normalize_gripper_action`` then
``invert_gripper_action``. Every upstream helper is called, never copied, and imported
function-locally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundationscale.vla.eval.libero_runner import LiberoRunnerError

__all__ = ["Gr00tLiberoPolicy"]

_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


class Gr00tLiberoPolicy:
    """A :class:`~foundationscale.vla.eval.protocol.PolicyAdapter` over a GR00T policy server.

    ``host``/``port`` name a running ``gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper``;
    the client connects lazily on the first :meth:`act`. ``n_action_steps`` (8 in GR00T's LIBERO
    recipe) is how many steps of each chunk the harness executes before replanning.
    """

    def __init__(self, host: str, port: int, *, n_action_steps: int = 8) -> None:
        if not isinstance(host, str) or not host:
            raise LiberoRunnerError(f"GR00T host is {host!r}; expected a non-empty string")
        for label, value in (("port", port), ("n_action_steps", n_action_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LiberoRunnerError(f"GR00T {label} is {value!r}; expected an int >= 1")
        self.name = f"gr00t@{host}:{port}"
        self.replan_steps = n_action_steps
        self._host = host
        self._port = port
        self._client: Any = None

    def reset(self) -> None:
        """Ask the server to drop per-episode state when the client supports it."""
        if self._client is not None and hasattr(self._client, "reset"):
            self._client.reset()

    def act(self, raw_obs: Mapping[str, Any], task_text: str) -> Any:
        """One inference: GR00T's batched observation in, an env-ready ``[k, 7]`` chunk out."""
        if self._client is None:
            from gr00t.policy.server_client import (  # noqa: PLC0415
                PolicyClient,
            )

            self._client = PolicyClient(host=self._host, port=self._port)
        answer = self._client.get_action(self.encode(raw_obs, task_text))
        action = answer[0] if isinstance(answer, tuple) else answer
        return self.decode(action)

    def encode(self, raw_obs: Mapping[str, Any], task_text: str) -> dict[str, Any]:
        """GR00T's LIBERO observation for one raw LIBERO observation, batched (B=1, T=1)."""
        import numpy as np  # noqa: PLC0415
        from robosuite.utils.transform_utils import (  # noqa: PLC0415
            quat2axisangle,
        )

        needed = (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        for key in needed:
            if key not in raw_obs:
                raise LiberoRunnerError(
                    f"LIBERO observation has no {key!r}; it carries {sorted(raw_obs)}"
                )

        def video(key: str) -> Any:
            image = np.ascontiguousarray(np.asarray(raw_obs[key])[::-1, ::-1], dtype=np.uint8)
            return image[None, None]

        # GR00T's server requires float32 state arrays (measured: float64 is refused by name).
        xyz = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        rpy = np.asarray(quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"])), dtype=np.float32)
        gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)
        scalars = dict(zip(("x", "y", "z"), xyz, strict=True)) | dict(
            zip(("roll", "pitch", "yaw"), rpy, strict=True)
        )
        observation: dict[str, Any] = {
            "video.image": video("agentview_image"),
            "video.wrist_image": video("robot0_eye_in_hand_image"),
            "state.gripper": gripper[None, None, :],
            "annotation.human.action.task_description": (str(task_text),),
        }
        for key, value in scalars.items():
            observation[f"state.{key}"] = np.asarray([[[value]]], dtype=np.float32)
        return observation

    def decode(self, action: Any) -> Any:
        """``action.<key>`` arrays ``(1, k, d)`` -> a ``[k, 7]`` chunk, gripper post-processed."""
        import numpy as np  # noqa: PLC0415
        from gr00t.eval.sim.LIBERO.libero_env import (  # noqa: PLC0415
            invert_gripper_action,
            normalize_gripper_action,
        )

        if not isinstance(action, Mapping):
            raise LiberoRunnerError(
                f"GR00T server answered {type(action).__name__}; expected a mapping of action.<key>"
            )
        parts = []
        for key in _ACTION_KEYS:
            name = f"action.{key}"
            if name not in action:
                raise LiberoRunnerError(
                    f"GR00T action has no {name!r}; it carries {sorted(action)}"
                )
            values = np.asarray(action[name], dtype=np.float64)
            if values.ndim == 3:
                values = values[0]
            if values.ndim == 1:
                values = values[:, None]
            parts.append(values)
        steps = {part.shape[0] for part in parts}
        if len(steps) != 1:
            raise LiberoRunnerError(
                f"GR00T action keys disagree on the chunk length: {sorted(steps)} steps"
            )
        chunk = np.concatenate(parts, axis=1)
        return np.stack(
            [invert_gripper_action(normalize_gripper_action(row.copy())) for row in chunk]
        )
