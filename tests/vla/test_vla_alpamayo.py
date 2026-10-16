"""Alpamayo trajectory server/client and the FS driving evaluation CLI."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import foundationscale.vla.adapters.alpamayo.server as server
import foundationscale.vla.driving.eval as drive
from foundationscale.vla.adapters.alpamayo import AlpamayoError, AlpamayoTrajectoryClient
from foundationscale.vla.wire import decode_array, encode_array


def _clip(offset: float, t: int = 64) -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros((t, 3))
    pred = np.zeros((1, 2, t, 3))
    pred[0, 0, :, 0] = offset  # best sample: offset metres in x
    pred[0, 1, :, 0] = offset + 5.0
    return pred, gt


def _stream(offsets: list[float], fail_at: int | None = None) -> server.TrajectoryStream:
    def predict(batch: Any) -> tuple[np.ndarray, np.ndarray, str]:
        if batch == fail_at:
            raise RuntimeError("CUDA OOM")
        pred, gt = _clip(offsets[batch])
        return pred, gt, f"clip{batch}"

    return server.TrajectoryStream(iter(range(len(offsets))), predict, len(offsets))


def test_stream_walks_the_clips_in_order_and_refuses_past_the_end() -> None:
    stream = _stream([0.5, 1.5])
    assert server.handle(stream, {"op": "info"}) == {"ok": True, "total": 2, "index": 0}
    first = server.handle(stream, {"op": "next"})
    assert first["ok"] and first["index"] == 0 and first["clip_id"] == "clip0"
    assert decode_array(first["pred"]).shape == (1, 2, 64, 3) and decode_array(
        first["gt"]
    ).shape == (64, 3)
    assert server.handle(stream, {"op": "next"})["index"] == 1
    assert server.handle(stream, {"op": "next"})["ok"] is False  # exhausted


@pytest.mark.parametrize("message", [{"op": "train"}, None, "next"])
def test_server_refuses_unknown_requests(message: Any) -> None:
    assert server.handle(_stream([0.0]), message)["ok"] is False


def test_server_reports_a_model_error_instead_of_crashing() -> None:
    reply = server.handle(_stream([0.0], fail_at=0), {"op": "next"})
    assert reply == {"ok": False, "error": "RuntimeError: CUDA OOM"}


def test_serve_skips_a_port_probe_and_answers_a_connection() -> None:
    from multiprocessing import AuthenticationError

    class _Conn:
        def __init__(self) -> None:
            self.msgs, self.sent, self.closed = [{"op": "info"}], [], False

        def recv(self) -> Any:
            if not self.msgs:
                raise EOFError
            return self.msgs.pop(0)

        def send(self, m: Any) -> None:
            self.sent.append(m)

        def close(self) -> None:
            self.closed = True

    conn = _Conn()
    attempts: list[Any] = [AuthenticationError("probe"), conn]

    class _Listener:
        def accept(self) -> Any:
            item = attempts.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    assert server.serve(_stream([0.0]), _Listener(), max_connections=1) == 1
    assert conn.sent == [{"ok": True, "total": 1, "index": 0}] and conn.closed


class _FakeServerConn:
    def __init__(self, stream: server.TrajectoryStream) -> None:
        self.stream, self.reply = stream, None

    def send(self, message: Any) -> None:
        self.reply = server.handle(self.stream, message)

    def recv(self) -> Any:
        return self.reply

    def close(self) -> None:
        pass


def _client(stream: server.TrajectoryStream) -> AlpamayoTrajectoryClient:
    return AlpamayoTrajectoryClient(
        "h", 1, connect=lambda address, authkey: _FakeServerConn(stream)
    )


def test_client_returns_clip_predictions_and_ground_truth() -> None:
    client = _client(_stream([0.5]))
    assert client.total() == 1 and client.name == "alpamayo@h:1"
    clip, pred, gt = client.next()
    assert clip == "clip0" and pred.shape == (1, 2, 64, 3) and gt.shape == (64, 3)
    with pytest.raises(AlpamayoError):
        client.next()  # exhausted -> refusal surfaces as an error
    client.close()


@pytest.mark.parametrize(
    "reply",
    [
        {"ok": True, "total": 0},
        {"ok": True, "total": "3"},
    ],
)
def test_client_refuses_a_nonsense_clip_count(reply: Any) -> None:
    class _C:
        def send(self, m: Any) -> None: ...
        def recv(self) -> Any:
            return reply

    with pytest.raises(AlpamayoError):
        AlpamayoTrajectoryClient("h", 1, connect=lambda a, authkey: _C()).total()


def test_client_refuses_mismatched_shapes() -> None:
    class _C:
        def send(self, m: Any) -> None: ...
        def recv(self) -> Any:
            return {
                "ok": True,
                "pred": encode_array(np.zeros((2, 64, 3))),
                "gt": encode_array(np.zeros((64, 3))),
            }

    with pytest.raises(AlpamayoError, match="expected"):
        AlpamayoTrajectoryClient("h", 1, connect=lambda a, authkey: _C()).next()


def test_evaluate_averages_per_clip_min_ade_and_keeps_failed_clips() -> None:
    report = drive.evaluate(_client(_stream([0.5, 1.5])), 2)
    assert report["measured"] and report["scored"] == 2
    assert report["mean_min_ade"] == pytest.approx(1.0)  # (0.5 + 1.5) / 2, best sample per clip
    assert report["mean_min_ade/by_t=0.5"] == pytest.approx(1.0)
    failed = drive.evaluate(_client(_stream([0.5, 1.5], fail_at=1)), 2)
    assert failed["measured"] is False and failed["clip_errors"] == 1 and failed["clips"] == 2
    assert failed["mean_min_ade"] == pytest.approx(0.5)
    assert drive.evaluate(_client(_stream([0.5], fail_at=0)), 1)["scored"] == 0


def _cli(monkeypatch: pytest.MonkeyPatch, stream: server.TrajectoryStream) -> None:
    monkeypatch.setattr(drive, "AlpamayoTrajectoryClient", lambda host, port: _client(stream))


def _listening() -> tuple[socket.socket, int]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s, s.getsockname()[1]


def test_cli_writes_a_measured_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _cli(monkeypatch, _stream([0.5, 1.5]))
    sock, port = _listening()
    with sock:
        rc = drive.main(
            [
                "--backend",
                "alpamayo",
                "--model",
                "m",
                "--out",
                str(tmp_path / "r.json"),
                "--port",
                str(port),
            ]
        )
    report = json.loads((tmp_path / "r.json").read_text())
    assert rc == drive.EXIT_OK and report["clips"] == 2 and report["backend"] == "alpamayo"


def test_cli_is_unmeasured_when_a_clip_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _cli(monkeypatch, _stream([0.5, 1.5], fail_at=0))
    sock, port = _listening()
    with sock:
        rc = drive.main(
            [
                "--backend",
                "alpamayo",
                "--model",
                "m",
                "--out",
                str(tmp_path / "r.json"),
                "--port",
                str(port),
                "--clips",
                "2",
            ]
        )
    assert (
        rc == drive.EXIT_UNMEASURED
        and json.loads((tmp_path / "r.json").read_text())["clip_errors"] == 1
    )


def test_cli_is_unmeasured_without_a_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sock, port = _listening()
    sock.close()
    monkeypatch.setattr(drive.time, "sleep", lambda s: None)
    rc = drive.main(
        [
            "--backend",
            "alpamayo",
            "--model",
            "m",
            "--out",
            str(tmp_path / "r.json"),
            "--port",
            str(port),
            "--serve-timeout-s",
            "0.2",
        ]
    )
    assert rc == drive.EXIT_UNMEASURED and not (tmp_path / "r.json").exists()


@pytest.mark.parametrize(
    "extra",
    [["--clips", "0"], ["--serve-cmd", "serve {checkpoint}"]],
)
def test_cli_refuses_bad_arguments_before_starting_anything(
    tmp_path: Path, extra: list[str]
) -> None:
    rc = drive.main(
        ["--backend", "alpamayo", "--model", "m", "--out", str(tmp_path / "r.json"), *extra]
    )
    assert rc == drive.EXIT_REFUSE


def test_cli_refuses_a_directory_out(tmp_path: Path) -> None:
    assert (
        drive.main(["--backend", "alpamayo", "--model", "m", "--out", str(tmp_path)])
        == drive.EXIT_REFUSE
    )


def test_cli_starts_and_stops_the_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started: list[Any] = []

    class _Proc:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            started.append("terminated")

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(
        drive.subprocess, "Popen", lambda argv, start_new_session: started.append(argv) or _Proc()
    )
    _cli(monkeypatch, _stream([0.5]))
    sock, port = _listening()
    with sock:
        rc = drive.main(
            [
                "--backend",
                "alpamayo",
                "--model",
                "ckpt",
                "--out",
                str(tmp_path / "r.json"),
                "--port",
                str(port),
                "--serve-cmd",
                "serve --m {model} --p {port}",
            ]
        )
    assert (
        rc == drive.EXIT_OK
        and started[0] == ["serve", "--m", "ckpt", "--p", str(port)]
        and started[-1] == "terminated"
    )


def test_server_main_composes_the_recipe_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    calls: dict[str, Any] = {}

    class _Init:
        def __init__(self, config_module: str, version_base: Any) -> None:
            calls["module"] = config_module

        def __enter__(self) -> None:
            return None

        def __exit__(self, *e: object) -> None:
            return None

    hydra = types.ModuleType("hydra")
    hydra.compose = lambda config_name, overrides: (
        calls.setdefault("compose", (config_name, overrides)) and "CFG"
    )  # type: ignore[attr-defined]
    hydra.initialize_config_module = _Init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setattr(
        server, "build_stream", lambda cfg: calls.setdefault("cfg", cfg) and _stream([0.0])
    )

    class _Listener:
        def __init__(self, address: Any, authkey: bytes) -> None:
            calls["listen"] = address

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *e: object) -> None:
            return None

    import multiprocessing.connection as mpc

    monkeypatch.setattr(mpc, "Listener", _Listener)
    monkeypatch.setattr(
        server, "serve", lambda stream, listener: calls.setdefault("served", stream.total)
    )
    assert server.main(["--port", "7", "evaluate.max_eval_steps=100"]) == 0
    assert calls["module"] == "alpamayo2_sft.configs"
    assert calls["compose"] == ("sft_eval_trajectory", ["evaluate.max_eval_steps=100"])
    assert calls["cfg"] == "CFG" and calls["listen"] == ("127.0.0.1", 7) and calls["served"] == 1
