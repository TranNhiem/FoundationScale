"""Client of the Alpamayo trajectory server: one clip's trajectories and ground truth per call."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from foundationscale.vla.wire import DEFAULT_AUTHKEY, decode_array

if TYPE_CHECKING:
    import numpy as np

__all__ = ["AlpamayoError", "AlpamayoTrajectoryClient"]


class AlpamayoError(RuntimeError):
    """The Alpamayo server refused a request or answered something that is not a trajectory set."""


class AlpamayoTrajectoryClient:
    """Talks to ``foundationscale.vla.adapters.alpamayo.server`` at ``host:port``."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        authkey: bytes = DEFAULT_AUTHKEY,
        name: str | None = None,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name or f"alpamayo@{host}:{port}"
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
            raise AlpamayoError(f"Alpamayo server refused {message.get('op')!r}: {detail}")
        return reply

    def total(self) -> int:
        """How many clips the server's evaluation set holds."""
        total = self._call({"op": "info"}).get("total")
        if type(total) is not int or total < 1:
            raise AlpamayoError(f"Alpamayo server reports {total!r} clips; expected an int >= 1")
        return total

    def next(self) -> tuple[str, np.ndarray, np.ndarray]:
        """``(clip_id, pred [N, K, T, 3], gt [T, 3])`` for the next clip, in the server's order."""
        reply = self._call({"op": "next"})
        pred, gt = decode_array(reply["pred"]), decode_array(reply["gt"])
        if pred.ndim != 4 or gt.ndim != 2 or pred.shape[-2:] != gt.shape:
            raise AlpamayoError(
                f"Alpamayo answered pred {pred.shape} and gt {gt.shape}; "
                "expected [N, K, T, 3] and [T, 3]"
            )
        return str(reply.get("clip_id")), pred, gt

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
