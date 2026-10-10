"""The openpi and GR00T LIBERO adapters, pinned against stand-in upstream modules.

Neither ``openpi_client``, ``robosuite`` nor ``gr00t`` is installed in CI, so each test injects a
fake with the exact call surface the adapter uses (the way tests/vla/test_vla_frames.py injects a
fake ``av``). What is pinned is the CONTRACT with each upstream: request keys, the 180-degree
rotation, the state composition order, GR00T's normalize-then-invert gripper post-processing, and
every refusal -- the parts a silent drift would corrupt without failing loudly.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from foundationscale.vla.adapters.gr00t.libero import Gr00tLiberoPolicy
from foundationscale.vla.adapters.openpi.libero import OpenPiLiberoPolicy
from foundationscale.vla.eval.libero_runner import LiberoRunnerError


def _raw_obs() -> dict[str, Any]:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 1,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def robosuite(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """robosuite's quat2axisangle stand-in: a marker vector so composition order is visible."""
    calls: list[Any] = []

    def quat2axisangle(quat: Any) -> Any:
        calls.append(np.asarray(quat))
        return np.array([7.0, 8.0, 9.0])

    _module(monkeypatch, "robosuite")
    _module(monkeypatch, "robosuite.utils")
    _module(monkeypatch, "robosuite.utils.transform_utils", quat2axisangle=quat2axisangle)
    return calls


@pytest.fixture
def openpi(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """openpi_client stand-in: image_tools that tag their calls, a websocket client that records."""
    seen: dict[str, Any] = {"clients": [], "requests": [], "resize": []}

    def resize_with_pad(image: Any, height: int, width: int) -> Any:
        seen["resize"].append((height, width))
        return image

    class WebsocketClientPolicy:
        def __init__(self, host: str, port: int) -> None:
            seen["clients"].append((host, port))

        def infer(self, request: dict[str, Any]) -> Any:
            seen["requests"].append(request)
            return seen.get("response", {"actions": np.zeros((10, 7))})

    image_tools = types.SimpleNamespace(
        resize_with_pad=resize_with_pad, convert_to_uint8=lambda x: np.asarray(x, dtype=np.uint8)
    )
    _module(
        monkeypatch,
        "openpi_client",
        image_tools=image_tools,
        websocket_client_policy=types.SimpleNamespace(WebsocketClientPolicy=WebsocketClientPolicy),
    )
    return seen


@pytest.fixture
def gr00t(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """gr00t stand-in: a PolicyClient that records and GR00T's two gripper post-processors."""
    seen: dict[str, Any] = {"clients": [], "observations": [], "post": []}

    class PolicyClient:
        def __init__(self, host: str, port: int) -> None:
            seen["clients"].append((host, port))

        def get_action(self, observation: dict[str, Any]) -> Any:
            seen["observations"].append(observation)
            return seen["action"], {}

    def normalize_gripper_action(action: Any, binarize: bool = True) -> Any:
        seen["post"].append("normalize")
        out = action.copy()
        out[-1] = 2.0 * out[-1] - 1.0
        return out

    def invert_gripper_action(action: Any) -> Any:
        seen["post"].append("invert")
        out = action.copy()
        out[-1] = -out[-1]
        return out

    for name in ("gr00t", "gr00t.policy", "gr00t.eval", "gr00t.eval.sim", "gr00t.eval.sim.LIBERO"):
        _module(monkeypatch, name)
    _module(monkeypatch, "gr00t.policy.server_client", PolicyClient=PolicyClient)
    _module(
        monkeypatch,
        "gr00t.eval.sim.LIBERO.libero_env",
        normalize_gripper_action=normalize_gripper_action,
        invert_gripper_action=invert_gripper_action,
    )
    return seen


def _gr00t_action(steps: int = 8) -> dict[str, Any]:
    action = {
        f"action.{k}": np.full((1, steps, 1), i, dtype=np.float64)
        for i, k in enumerate(("x", "y", "z", "roll", "pitch", "yaw"))
    }
    action["action.gripper"] = np.full((1, steps, 1), 1.0)
    return action


# -- openpi ---------------------------------------------------------------------------------


def test_openpi_request_matches_openpis_libero_client(openpi, robosuite) -> None:
    policy = OpenPiLiberoPolicy("127.0.0.1", 8000)
    request = policy.encode(_raw_obs(), "put the bowl on the plate")
    assert set(request) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    raw = _raw_obs()
    np.testing.assert_array_equal(request["observation/image"], raw["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(
        request["observation/wrist_image"], raw["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    # eef position, then robosuite's axis-angle, then the two gripper joints: 8 numbers
    np.testing.assert_allclose(
        request["observation/state"], [0.1, 0.2, 0.3, 7.0, 8.0, 9.0, 0.04, -0.04]
    )
    assert request["prompt"] == "put the bowl on the plate"
    assert openpi["resize"] == [(224, 224), (224, 224)]


def test_openpi_client_is_created_lazily_once_and_actions_come_back(openpi, robosuite) -> None:
    policy = OpenPiLiberoPolicy("10.0.0.2", 8763, resize=128, replan_steps=5)
    assert openpi["clients"] == []
    openpi["response"] = {"actions": np.ones((10, 7))}
    first = policy.act(_raw_obs(), "task")
    policy.act(_raw_obs(), "task")
    assert openpi["clients"] == [("10.0.0.2", 8763)]
    assert first.shape == (10, 7) and openpi["resize"][0] == (128, 128)
    assert policy.name == "openpi@10.0.0.2:8763" and policy.replan_steps == 5


def test_openpi_response_without_actions_is_refused(openpi, robosuite) -> None:
    openpi["response"] = {"chunk": []}
    with pytest.raises(LiberoRunnerError, match="without an 'actions' entry"):
        OpenPiLiberoPolicy("h", 1).act(_raw_obs(), "task")


def test_openpi_observation_missing_a_key_is_refused(openpi, robosuite) -> None:
    obs = _raw_obs()
    del obs["robot0_eef_quat"]
    with pytest.raises(LiberoRunnerError, match="robot0_eef_quat"):
        OpenPiLiberoPolicy("h", 1).encode(obs, "task")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": "", "port": 1},
        {"host": "h", "port": 0},
        {"host": "h", "port": True},
        {"host": "h", "port": 1, "resize": 0},
        {"host": "h", "port": 1, "replan_steps": 0},
    ],
)
def test_openpi_constructor_refuses_bad_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(LiberoRunnerError):
        OpenPiLiberoPolicy(**kwargs)


# -- GR00T ----------------------------------------------------------------------------------


def test_gr00t_observation_matches_gr00ts_libero_env(gr00t, robosuite) -> None:
    observation = Gr00tLiberoPolicy("127.0.0.1", 5555).encode(_raw_obs(), "put the bowl")
    raw = _raw_obs()
    assert observation["video.image"].shape == (1, 1, 4, 4, 3)
    np.testing.assert_array_equal(
        observation["video.image"][0, 0], raw["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        observation["video.wrist_image"][0, 0], raw["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    for key, value in zip(
        ("x", "y", "z", "roll", "pitch", "yaw"), (0.1, 0.2, 0.3, 7, 8, 9), strict=True
    ):
        assert observation[f"state.{key}"].shape == (1, 1, 1)
        assert observation[f"state.{key}"].dtype == np.float32  # GR00T refuses float64
        assert observation[f"state.{key}"][0, 0, 0] == pytest.approx(value)
    np.testing.assert_allclose(observation["state.gripper"], [[[0.04, -0.04]]], rtol=1e-6)
    assert observation["state.gripper"].dtype == np.float32
    assert observation["annotation.human.action.task_description"] == ("put the bowl",)


def test_gr00t_chunk_concatenates_keys_in_order_then_normalizes_then_inverts(gr00t, robosuite):
    gr00t["action"] = _gr00t_action(steps=8)
    policy = Gr00tLiberoPolicy("127.0.0.1", 5689, n_action_steps=8)
    chunk = policy.act(_raw_obs(), "task")
    assert chunk.shape == (8, 7)
    np.testing.assert_allclose(chunk[0, :6], [0, 1, 2, 3, 4, 5])
    # gripper 1.0 -> normalize: 2*1-1 = 1.0 -> invert: -1.0
    assert chunk[0, 6] == pytest.approx(-1.0)
    assert gr00t["post"][:2] == ["normalize", "invert"]
    assert gr00t["clients"] == [("127.0.0.1", 5689)] and policy.replan_steps == 8


def test_gr00t_missing_action_key_is_refused(gr00t, robosuite) -> None:
    action = _gr00t_action()
    del action["action.yaw"]
    gr00t["action"] = action
    with pytest.raises(LiberoRunnerError, match="action.yaw"):
        Gr00tLiberoPolicy("h", 1).act(_raw_obs(), "task")


def test_gr00t_chunk_length_disagreement_is_refused(gr00t, robosuite) -> None:
    action = _gr00t_action(steps=8)
    action["action.gripper"] = np.ones((1, 4, 1))
    gr00t["action"] = action
    with pytest.raises(LiberoRunnerError, match="disagree on the chunk length"):
        Gr00tLiberoPolicy("h", 1).act(_raw_obs(), "task")


def test_gr00t_non_mapping_action_is_refused(gr00t, robosuite) -> None:
    gr00t["action"] = [1, 2, 3]
    with pytest.raises(LiberoRunnerError, match="expected a mapping"):
        Gr00tLiberoPolicy("h", 1).act(_raw_obs(), "task")


def test_gr00t_observation_missing_a_camera_is_refused(gr00t, robosuite) -> None:
    obs = _raw_obs()
    del obs["robot0_eye_in_hand_image"]
    with pytest.raises(LiberoRunnerError, match="robot0_eye_in_hand_image"):
        Gr00tLiberoPolicy("h", 1).encode(obs, "task")


def test_gr00t_reset_is_forwarded_only_once_a_client_exists(gr00t, robosuite) -> None:
    policy = Gr00tLiberoPolicy("h", 1)
    policy.reset()  # no client yet: nothing to reset, nothing created
    assert gr00t["clients"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": "", "port": 1},
        {"host": "h", "port": -1},
        {"host": "h", "port": 1, "n_action_steps": 0},
    ],
)
def test_gr00t_constructor_refuses_bad_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(LiberoRunnerError):
        Gr00tLiberoPolicy(**kwargs)
