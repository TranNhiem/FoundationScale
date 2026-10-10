"""``python -m foundationscale.vla.eval``: protocol defaults, report payload, exit codes."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

import foundationscale.vla.eval.__main__ as cli
from foundationscale.vla.eval.libero_runner import EpisodeRecord, EvalReport


def _listening_port() -> tuple[socket.socket, int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock, sock.getsockname()[1]


def _fake_run(seen: dict[str, Any]) -> Any:
    def run(plan: Any, policy: Any) -> EvalReport:
        seen["plan"], seen["policy"] = plan, policy
        episodes = tuple(
            EpisodeRecord(
                task_id=t,
                trial=i,
                success=(i % 4 != 0),
                steps=10,
                inferences=2,
                wall_s=0.1,
                error=None,
            )
            for t, i in plan.episodes()
        )
        return EvalReport(plan=plan, policy=policy.name, episodes=episodes)

    return run


@pytest.mark.parametrize(
    ("backend", "init", "wait", "replan", "trials", "seed"),
    [("openpi", "suite_init_states", 10, 5, 50, 7), ("gr00t", "seeded_reset", 0, 8, 20, 0)],
)
def test_each_backend_rolls_out_its_published_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    init: str,
    wait: int,
    replan: int,
    trials: int,
    seed: int,
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "run_libero", _fake_run(seen))
    server, port = _listening_port()
    out = tmp_path / "report.json"
    with server:
        rc = cli.main(
            [
                "--backend",
                backend,
                "--model",
                "ckpt",
                "--suite",
                "libero_spatial",
                "--out",
                str(out),
                "--port",
                str(port),
            ]
        )
    assert rc == cli.EXIT_OK
    plan = seen["plan"]
    assert (plan.init, plan.num_steps_wait, plan.trials_per_task, plan.seed) == (
        init,
        wait,
        trials,
        seed,
    )
    assert seen["policy"].replan_steps == replan and plan.task_ids == tuple(range(10))
    payload = json.loads(out.read_text())
    assert payload["backend"] == backend and payload["model"] == "ckpt"
    # the fake fails every trial index divisible by 4
    expected = 100.0 * sum(1 for _, i in plan.episodes() if i % 4 != 0) / len(plan.episodes())
    assert payload["success_rate_pct"] == pytest.approx(expected)
    assert payload["wilson95_pct"][0] < expected < payload["wilson95_pct"][1]


def test_trials_and_seed_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "run_libero", _fake_run(seen))
    server, port = _listening_port()
    with server:
        cli.main(
            [
                "--backend",
                "gr00t",
                "--model",
                "m",
                "--suite",
                "libero_spatial",
                "--out",
                str(tmp_path / "r.json"),
                "--port",
                str(port),
                "--trials-per-task",
                "3",
                "--seed",
                "11",
            ]
        )
    assert (seen["plan"].trials_per_task, seen["plan"].seed) == (3, 11)


def test_no_server_is_unmeasured_not_a_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server, port = _listening_port()
    server.close()  # nothing listens on this port any more
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    rc = cli.main(
        [
            "--backend",
            "openpi",
            "--model",
            "m",
            "--suite",
            "libero_spatial",
            "--out",
            str(tmp_path / "r.json"),
            "--port",
            str(port),
            "--serve-timeout-s",
            "0.5",
        ]
    )
    assert rc == cli.EXIT_UNMEASURED and not (tmp_path / "r.json").exists()


def test_a_refused_plan_exits_96(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rc = cli.main(
        [
            "--backend",
            "openpi",
            "--model",
            "m",
            "--suite",
            "libero_spatial",
            "--out",
            str(tmp_path / "r.json"),
            "--trials-per-task",
            "-1",
            "--port",
            "1",
        ]
    )
    assert rc == cli.EXIT_REFUSE


def test_serve_cmd_starts_and_stops_the_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "run_libero", _fake_run(seen))
    server, port = _listening_port()
    server.close()
    script = tmp_path / "serve.py"
    script.write_text(
        "import socket, sys, time\n"
        "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(); time.sleep(60)\n"
    )
    rc = cli.main(
        [
            "--backend",
            "gr00t",
            "--model",
            "m",
            "--suite",
            "libero_spatial",
            "--out",
            str(tmp_path / "r.json"),
            "--port",
            str(port),
            "--trials-per-task",
            "1",
            "--serve-cmd",
            f"{sys.executable} {script} {{port}}",
            "--serve-timeout-s",
            "20",
        ]
    )
    assert rc == cli.EXIT_OK and seen["plan"].trials_per_task == 1
