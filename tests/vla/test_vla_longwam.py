"""Long-WAM adapter: array transport, the per-step client, the server and the protocol row."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

import foundationscale.vla.adapters.longwam.server as server
import foundationscale.vla.eval.__main__ as cli
from foundationscale.vla.adapters.longwam import LongWamError, LongWamLiberoPolicy
from foundationscale.vla.adapters.longwam.libero import OBS_KEYS, decode_array, encode_array


def _obs() -> dict[str, Any]:
    return {
        "agentview_image": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eye_in_hand_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "unused": np.ones(5),
    }


class _Conn:
    def __init__(self, replies: list[Any]) -> None:
        self.sent: list[Any] = []
        self.replies = list(replies)
        self.closed = False

    def send(self, message: Any) -> None:
        self.sent.append(message)

    def recv(self) -> Any:
        if not self.replies:
            raise EOFError
        return self.replies.pop(0)

    def close(self) -> None:
        self.closed = True


def _client(replies: list[Any]) -> tuple[LongWamLiberoPolicy, _Conn, list[Any]]:
    conn, calls = _Conn(replies), []

    def connect(address: Any, authkey: bytes) -> _Conn:
        calls.append((address, authkey))
        return conn

    return LongWamLiberoPolicy("h", 7, connect=connect), conn, calls


@pytest.mark.parametrize(
    "array", [np.arange(6, dtype=np.float32).reshape(2, 3), np.zeros((4, 4, 3), np.uint8)]
)
def test_arrays_round_trip_as_builtins(array: np.ndarray) -> None:
    payload = encode_array(array)
    assert {type(v) for v in payload.values()} <= {str, list, bytes}
    back = decode_array(payload)
    assert back.dtype == array.dtype and np.array_equal(back, array) and back.flags.writeable


def test_the_adapter_is_called_every_step_and_returns_one_action() -> None:
    action = encode_array(np.arange(7, dtype=np.float32))
    policy, conn, calls = _client([{"ok": True}, {"ok": True, "action": action}])
    assert policy.replan_steps == 1 and policy.name == "longwam@h:7"
    policy.reset()
    out = policy.act(_obs(), "pick up the bowl")
    assert calls == [(("h", 7), b"fs-longwam")]  # connects once, lazily
    assert conn.sent[0] == {"op": "reset"}
    assert conn.sent[1]["op"] == "act" and conn.sent[1]["task"] == "pick up the bowl"
    assert set(conn.sent[1]["obs"]) == set(OBS_KEYS)  # only the keys Long-WAM reads
    assert out.shape == (1, 7) and out.dtype == np.float64 and out[0, 6] == 6.0
    policy.close()
    assert conn.closed


@pytest.mark.parametrize(
    "reply",
    [
        {"ok": False, "error": "CUDA OOM"},
        "garbage",
        {"ok": True, "action": encode_array(np.zeros(6))},
    ],
)
def test_a_refusal_or_a_wrong_action_raises(reply: Any) -> None:
    policy, _, _ = _client([reply])
    with pytest.raises(LongWamError):
        policy.act(_obs(), "t")


def test_an_observation_missing_a_key_is_refused_before_sending() -> None:
    policy, conn, _ = _client([])
    obs = _obs()
    del obs["robot0_eef_quat"]
    with pytest.raises(LongWamError, match="robot0_eef_quat"):
        policy.act(obs, "t")
    assert conn.sent == []


class _Policy:
    def __init__(self, fail: bool = False) -> None:
        self.resets = 0
        self.fail = fail
        self.batches: list[Any] = []

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, batch: Any) -> list[Any]:
        if self.fail:
            raise RuntimeError("boom")
        self.batches.append(batch)
        return [np.full(7, 0.5)]


def test_server_answers_reset_and_act_like_long_wams_own_loop() -> None:
    policy, seen = _Policy(), []

    def batch_fn(obs: dict[str, Any], task: str) -> str:
        seen.append((set(obs), task))
        return "batch"

    assert server.handle(policy, batch_fn, {"op": "reset"}) == {"ok": True}
    reply = server.handle(
        policy, batch_fn, {"op": "act", "task": "t", "obs": {"a": encode_array(np.ones(2))}}
    )
    assert reply["ok"] is True and np.array_equal(decode_array(reply["action"]), np.full(7, 0.5))
    assert policy.resets == 1 and policy.batches == ["batch"] and seen == [({"a"}, "t")]


@pytest.mark.parametrize("message", [{"op": "train"}, "x", None])
def test_server_refuses_unknown_requests(message: Any) -> None:
    assert server.handle(_Policy(), lambda o, t: o, message)["ok"] is False


def test_server_reports_a_policy_error_instead_of_crashing() -> None:
    reply = server.handle(_Policy(fail=True), lambda o, t: o, {"op": "act", "task": "t", "obs": {}})
    assert reply == {"ok": False, "error": "RuntimeError: boom"}


def test_serve_answers_every_request_of_a_connection() -> None:
    conn = _Conn([{"op": "reset"}, {"op": "bad"}])

    class _Listener:
        def accept(self) -> _Conn:
            return conn

    served = server.serve(_Policy(), lambda o, t: o, _Listener(), max_connections=1)
    assert (
        served == 2 and conn.closed and conn.sent[0] == {"ok": True} and conn.sent[1]["ok"] is False
    )


def test_server_main_builds_the_policy_with_long_wams_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    settings = types.SimpleNamespace(seed=1)

    def mod(name: str, **attrs: Any) -> None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    mod("longwam")
    mod("longwam.benchmarks")
    mod("longwam.benchmarks.libero")
    mod("longwam.benchmarks.libero.adapter", policy_batch="PB")
    mod(
        "longwam.benchmarks.libero.eval_libero_single",
        set_global_seed=lambda seed, get_worker_init_fn: calls.setdefault("seed", seed),
    )
    mod(
        "longwam.evaluation",
        read_settings=lambda path, ov: calls.setdefault("read", (path, ov)) and settings,
    )
    mod("longwam.runtime", create_policy=lambda s: calls.setdefault("policy_from", s) and "POLICY")

    class _Listener:
        def __init__(self, address: Any, authkey: bytes) -> None:
            calls["listen"] = (address, authkey)

        def __enter__(self) -> _Listener:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    import multiprocessing.connection as mpc

    monkeypatch.setattr(mpc, "Listener", _Listener)
    monkeypatch.setattr(server, "serve", lambda p, b, lst: calls.setdefault("serve", (p, b)))
    assert (
        server.main(["--config", "eval.yaml", "--port", "9", "checkpoint=c.pt", "episodes=50"]) == 0
    )
    assert calls["read"] == ("eval.yaml", ["checkpoint=c.pt", "episodes=50"])
    assert calls["seed"] == 1 and calls["policy_from"] is settings
    assert calls["listen"] == (("127.0.0.1", 9), b"fs-longwam")
    assert calls["serve"] == ("POLICY", "PB")


def test_long_wam_protocol_row_is_its_published_eval_config() -> None:
    p = cli.PROTOCOLS["longwam"]
    assert (p.init, p.num_steps_wait, p.replan_steps, p.trials_per_task, p.seed) == (
        "suite_init_states",
        30,
        1,
        50,
        1,
    )
    assert [
        p.steps_for(s) for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    ] == [
        400,
        400,
        400,
        700,
    ]
    assert cli.PROTOCOLS["gr00t"].steps_for("libero_10") == 720  # other rows unchanged
    assert cli.PROTOCOLS["openpi"].steps_for("libero_10") is None
    adapter = cli._policy("longwam", "127.0.0.1", 5600, p)
    assert isinstance(adapter, LongWamLiberoPolicy) and adapter.replan_steps == 1


def test_the_default_client_is_multiprocessing_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing.connection as mpc

    conn = _Conn([{"ok": True}])
    monkeypatch.setattr(mpc, "Client", lambda address, authkey: conn)
    LongWamLiberoPolicy("h", 7).reset()
    assert conn.sent == [{"op": "reset"}]
