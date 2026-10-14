"""FS rewards and tasks in verl's shapes, for the P0.3 parity runs.

verl calls a custom reward function as ``compute_score(data_source, solution_str, ground_truth,
extra_info)`` (``reward.custom_reward_function`` in verl's trainer config) and reads tasks from
parquet rows with ``data_source`` / ``prompt`` / ``reward_model`` / ``extra_info``. This module
gives both shapes to the FS music task without importing verl.

verl has no abstention: every sample gets a number. FS's ``MusicReward`` abstains (``value``
``None``) exactly when its own infrastructure failed (``reason`` starting ``"infra:"``). Scoring
that as 0.0 would train the policy on a fabricated zero, so ``music_compute_score`` raises
``VerlRewardAbstention`` instead: the run stops with the reason named, never a silent score.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
from typing import Any

from foundationscale.agentic_rl.rewards.music import MusicReward

DATA_SOURCE_MUSIC = "fs_music"


class VerlRewardAbstention(RuntimeError):
    """The FS scorer abstained (infrastructure failure); verl cannot represent that."""


class VerlTaskRefusal(ValueError):
    """A task row cannot be converted; the message names the row and the field."""


@functools.lru_cache(maxsize=8)
def _music_reward(abc2midi_bin: str, timeout_s: float) -> MusicReward:
    return MusicReward(abc2midi_bin=abc2midi_bin, timeout_s=timeout_s)


def music_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> float:
    """verl ``compute_score`` for the FS music task.

    ``abc2midi_bin`` and ``timeout_s`` come from the row's ``extra_info`` (declared in the task
    file, never read from the environment). ``ground_truth`` is unused: music has no reference.
    """
    del ground_truth
    if data_source != DATA_SOURCE_MUSIC:
        raise VerlTaskRefusal(
            f"music_compute_score: data_source {data_source!r} is not {DATA_SOURCE_MUSIC!r}"
        )
    info = extra_info or {}
    binary = info.get("abc2midi_bin")
    if not isinstance(binary, str) or not binary:
        raise VerlTaskRefusal(
            f"music_compute_score: extra_info.abc2midi_bin is {binary!r}: declare the binary in "
            f"the task row"
        )
    timeout = info.get("timeout_s", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise VerlTaskRefusal(f"music_compute_score: extra_info.timeout_s is {timeout!r}")
    result = _music_reward(binary, float(timeout)).score(solution_str)
    if result.value is None:
        raise VerlRewardAbstention(
            f"music_compute_score: the FS scorer abstained ({result.reason}); verl would score "
            f"it as a number, so the run stops instead of training on a fabricated value"
        )
    return float(result.value)


def music_rows_from_jsonl(
    lines: Iterable[str], *, abc2midi_bin: str, timeout_s: float = 60.0
) -> list[dict[str, Any]]:
    """Convert FS music task JSONL lines (``uid`` + chat ``messages``) to verl parquet rows."""
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerlTaskRefusal(f"line {lineno}: not JSON ({exc.msg})") from exc
        uid = task.get("uid") if isinstance(task, dict) else None
        messages = task.get("messages") if isinstance(task, dict) else None
        if not isinstance(uid, str) or not uid:
            raise VerlTaskRefusal(f"line {lineno}: field 'uid' is {uid!r}")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(m, dict)
                and isinstance(m.get("role"), str)
                and isinstance(m.get("content"), str)
                for m in messages
            )
        ):
            raise VerlTaskRefusal(f"line {lineno} ({uid}): field 'messages' is not a chat list")
        rows.append(
            {
                "data_source": DATA_SOURCE_MUSIC,
                "prompt": [{"role": m["role"], "content": m["content"]} for m in messages],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "uid": uid,
                    "index": len(rows),
                    "abc2midi_bin": abc2midi_bin,
                    "timeout_s": float(timeout_s),
                },
            }
        )
    if not rows:
        raise VerlTaskRefusal("0 task rows: an empty task file converts to nothing")
    return rows
