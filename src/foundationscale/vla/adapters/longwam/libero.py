"""Long-WAM LIBERO policy adapter: a client of the Long-WAM policy server, one action per step.

Long-WAM conditions on the last 48 control steps, and its own LIBERO loop
(``longwam/benchmarks/libero/eval_libero_single.py``) calls the policy's ``select_action`` on every
step after the settle steps: the policy keeps its own observation history and action queue and
infers again on its own cadence. So this adapter declares ``replan_steps = 1``. The FS runner then
calls :meth:`act` once per step, every observation reaches Long-WAM, and the replanning stays
Long-WAM's. The server (:mod:`.server`) runs in Long-WAM's own environment; the two sides exchange
plain dicts over ``multiprocessing.connection``, with arrays as dtype/shape/bytes, because the two
environments carry different NumPy versions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from foundationscale.vla.wire import DEFAULT_AUTHKEY, decode_array, encode_array

if TYPE_CHECKING:
    import numpy as np

__all__ = ["OBS_KEYS", "LongWamError", "LongWamLiberoPolicy", "decode_array", "encode_array"]

# the raw LIBERO keys Long-WAM's policy_batch reads (two cameras, end-effector pose, gripper)
OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


class LongWamError(RuntimeError):
    """The Long-WAM server refused a request or answered something that is not an action."""


class LongWamLiberoPolicy:
    """The FS PolicyAdapter for a Long-WAM policy server at ``host:port``."""

    replan_steps = 1

    def __init__(
        self,
        host: str,
        port: int,
        *,
        authkey: bytes = DEFAULT_AUTHKEY,
        name: str | None = None,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name or f"longwam@{host}:{port}"
        self._address = (host, port)
        self._authkey = authkey
        self._connect = connect
        self._conn: Any = None

    def _call(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._conn is None:
            if self._connect is None:
                from multiprocessing.connection import Client  # noqa: PLC0415

                self._connect = Client
            self._conn = self._connect(self._address, authkey=self._authkey)
        self._conn.send(message)
        reply = self._conn.recv()
        if not isinstance(reply, dict) or reply.get("ok") is not True:
            detail = reply.get("error") if isinstance(reply, dict) else repr(reply)
            raise LongWamError(f"Long-WAM server refused {message.get('op')!r}: {detail}")
        return reply

    def reset(self) -> None:
        """Clear the policy's history and action queue on the server (Long-WAM's policy.reset())."""
        self._call({"op": "reset"})

    def act(self, raw_obs: Mapping[str, Any], task_text: str) -> np.ndarray:
        """One env-ready action ``[1, 7]`` for this step's raw LIBERO observation."""
        import numpy as np  # noqa: PLC0415

        missing = [key for key in OBS_KEYS if key not in raw_obs]
        if missing:
            raise LongWamError(
                f"LIBERO observation lacks {missing}; Long-WAM reads {list(OBS_KEYS)}"
            )
        obs = {key: encode_array(raw_obs[key]) for key in OBS_KEYS}
        reply = self._call({"op": "act", "task": task_text, "obs": obs})
        action = np.asarray(decode_array(reply["action"]), dtype=np.float64).reshape(1, -1)
        if action.shape != (1, 7):
            raise LongWamError(
                f"Long-WAM answered an action of shape {action.shape}; expected (1, 7)"
            )
        return action

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
