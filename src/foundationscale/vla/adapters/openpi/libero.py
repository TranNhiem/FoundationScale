"""openpi policies on LIBERO through openpi's public websocket protocol.

The observation is encoded exactly as openpi's ``examples/libero/main.py`` encodes it -- both
cameras rotated 180 degrees, resized with padding to ``resize`` and cast to uint8 by openpi's own
``openpi_client.image_tools``; state = end-effector position, robosuite's ``quat2axisangle`` of the
end-effector quaternion, and the two gripper joints -- so a published openpi checkpoint sees what
it was evaluated with upstream. Every upstream helper is CALLED, never copied, and imported
function-locally: the core install carries none of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundationscale.vla.eval.libero_runner import LiberoRunnerError

__all__ = ["OpenPiLiberoPolicy"]


class OpenPiLiberoPolicy:
    """A :class:`~foundationscale.vla.eval.protocol.PolicyAdapter` over an openpi policy server.

    ``host``/``port`` name a running ``scripts/serve_policy.py``; the client connects lazily on
    the first :meth:`act`. ``resize`` (224 upstream) and ``replan_steps`` (5 upstream) are openpi's
    own LIBERO settings, declared rather than hidden.
    """

    def __init__(self, host: str, port: int, *, resize: int = 224, replan_steps: int = 5) -> None:
        if not isinstance(host, str) or not host:
            raise LiberoRunnerError(f"openpi host is {host!r}; expected a non-empty string")
        for label, value in (("port", port), ("resize", resize), ("replan_steps", replan_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LiberoRunnerError(f"openpi {label} is {value!r}; expected an int >= 1")
        self.name = f"openpi@{host}:{port}"
        self.replan_steps = replan_steps
        self._host = host
        self._port = port
        self._resize = resize
        self._client: Any = None

    def reset(self) -> None:
        """Nothing to reset: openpi's LIBERO client keeps no per-episode state on the server."""

    def act(self, raw_obs: Mapping[str, Any], task_text: str) -> Any:
        """One inference: the encoded request to the server, its ``actions`` chunk back."""
        import numpy as np  # noqa: PLC0415

        if self._client is None:
            from openpi_client import (  # noqa: PLC0415
                websocket_client_policy,
            )

            self._client = websocket_client_policy.WebsocketClientPolicy(self._host, self._port)
        response = self._client.infer(self.encode(raw_obs, task_text))
        if not isinstance(response, Mapping) or "actions" not in response:
            raise LiberoRunnerError(
                f"openpi server at {self._host}:{self._port} answered {type(response).__name__} "
                "without an 'actions' entry"
            )
        return np.asarray(response["actions"], dtype=np.float64)

    def encode(self, raw_obs: Mapping[str, Any], task_text: str) -> dict[str, Any]:
        """The request openpi's LIBERO client builds, from one raw LIBERO observation."""
        import numpy as np  # noqa: PLC0415
        from openpi_client import image_tools  # noqa: PLC0415
        from robosuite.utils.transform_utils import (  # noqa: PLC0415
            quat2axisangle,
        )

        def camera(key: str) -> Any:
            if key not in raw_obs:
                raise LiberoRunnerError(
                    f"LIBERO observation has no {key!r}; it carries {sorted(raw_obs)}"
                )
            image = np.ascontiguousarray(np.asarray(raw_obs[key])[::-1, ::-1])
            return image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, self._resize, self._resize)
            )

        for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if key not in raw_obs:
                raise LiberoRunnerError(
                    f"LIBERO observation has no {key!r}; it carries {sorted(raw_obs)}"
                )
        state = np.concatenate(
            (
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64),
                np.asarray(
                    quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"])), dtype=np.float64
                ),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64),
            )
        )
        return {
            "observation/image": camera("agentview_image"),
            "observation/wrist_image": camera("robot0_eye_in_hand_image"),
            "observation/state": state,
            "prompt": str(task_text),
        }
