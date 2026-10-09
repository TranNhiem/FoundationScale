"""Hygiene and provenance pins for the FoundationScale Agentic RL contract plane."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import foundationscale.agentic_rl as agentic_rl
from foundationscale.agentic_rl import token_trace
from foundationscale.agentic_rl.contracts import (
    SegmentKind,
    Termination,
    Trajectory,
    TrajectoryRefusal,
    Turn,
)
from foundationscale.agentic_rl.token_trace import TokenTrace, TokenTraceRefusal

_PACKAGE_DIR = Path(agentic_rl.__file__).parent
_TEST_DIR = Path(__file__).parent
_REPO_ROOT = _PACKAGE_DIR.parents[2]
_HEADER = (
    "# Portions adapted from Xiaomi"
    + "Mi"
    + "Mo"
    + "/verl recipes/arvo/token_trace.py (commit a2ad9f61),\n"
    "# Copyright the original authors, licensed under the Apache License, Version 2.0.\n"
    "# Modifications Copyright (c) 2026 TranNhiem, licensed under the MIT License (see LICENSE).\n"
    "# See THIRD_PARTY_NOTICES.md.\n"
)

# The music scorer port (slice 4c) keeps each upstream file's FULL Apache
# header verbatim and adds a 3-line attribution block below it -- a different
# shape from token_trace's condensed one-paragraph header above, because the
# upstream music scorer files carry the full boilerplate and the spec says to
# keep each file's original header lines intact rather than re-write them.
_APACHE_BYTEDANCE_HEADER = (
    "# Copyright 2026 Bytedance Ltd. and/or its affiliates\n"
    "#\n"
    '# Licensed under the Apache License, Version 2.0 (the "License");\n'
    "# you may not use this file except in compliance with the License.\n"
    "# You may obtain a copy of the License at\n"
    "#\n"
    "#     http://www.apache.org/licenses/LICENSE-2.0\n"
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    '# distributed under the License is distributed on an "AS IS" BASIS,\n'
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
)
_MUSIC_DIR = _PACKAGE_DIR / "rewards" / "music"
_MUSIC_UPSTREAM_PATH = {
    "core.py": "recipes/design/music/scorer/core.py",
    "feats.py": "recipes/design/music/scorer/feats.py",
    "score.py": "recipes/design/music/scorer/score.py",
    "pipeline.py": "recipes/design/music/scorer/pipeline.py",
    "baseline.py": "recipes/design/music/scorer/baselines/ref_full4k.json",
}


def _music_header(filename: str, upstream_path: str) -> str:
    brand = "Xiaomi" + "Mi" + "Mo"
    # baseline.py's upstream path is long enough that "# Adapted from
    # <brand>/verl <path> (commit a2ad9f61)." alone exceeds the 100-col limit
    # (ruff E501), so that one file wraps the path onto its own line; the
    # other four fit on one line.
    if filename == "baseline.py":
        attribution = (
            f"# Adapted from {brand}/verl recipes/design/music/scorer/baselines/\n"
            "# ref_full4k.json (commit a2ad9f61).\n"
        )
    else:
        attribution = f"# Adapted from {brand}/verl {upstream_path} (commit a2ad9f61).\n"
    return (
        _APACHE_BYTEDANCE_HEADER
        + attribution
        + "# Modifications Copyright (c) 2026 TranNhiem, licensed under the MIT License "
        "(see LICENSE).\n"
        "# See THIRD_PARTY_NOTICES.md.\n"
    )


def test_the_ported_module_opens_with_the_exact_apache_attribution_header() -> None:
    source = Path(token_trace.__file__).read_text(encoding="utf-8")
    assert source.startswith(_HEADER)


@pytest.mark.parametrize("filename", sorted(_MUSIC_UPSTREAM_PATH))
def test_each_ported_music_module_opens_with_the_exact_apache_attribution_header(
    filename: str,
) -> None:
    source = (_MUSIC_DIR / filename).read_text(encoding="utf-8")
    assert source.startswith(_music_header(filename, _MUSIC_UPSTREAM_PATH[filename]))


def test_third_party_notices_names_the_port_its_license_and_the_upstream_notice() -> None:
    notices = (_REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for needle in (
        "src/foundationscale/agentic_rl/token_trace.py",
        "recipes/arvo/token_trace.py",
        "a2ad9f61",
        "Apache License, Version 2.0",
        "Copyright 2023-2024 Bytedance Ltd. and/or its affiliates",
    ):
        assert needle in notices, needle


def test_no_feature_file_carries_the_upstream_brand_outside_the_attribution() -> None:
    # The feature is "FoundationScale Agentic RL"; the only lawful occurrence of the
    # upstream organisation's name is the attribution header of the ported module.
    pattern = re.compile("mi" + "mo", re.IGNORECASE)
    files = sorted(_PACKAGE_DIR.rglob("*.py")) + sorted(_TEST_DIR.rglob("*.py"))
    assert files
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if path.name == "token_trace.py" and path.parent == _PACKAGE_DIR:
            text = text[len(_HEADER) :] if text.startswith(_HEADER) else text
        elif path.parent == _MUSIC_DIR and path.name in _MUSIC_UPSTREAM_PATH:
            header = _music_header(path.name, _MUSIC_UPSTREAM_PATH[path.name])
            text = text[len(header) :] if text.startswith(header) else text
        if pattern.search(text):
            offenders.append(str(path))
    assert offenders == []


def test_third_party_notices_names_the_music_port_its_license_and_the_upstream_notice() -> None:
    notices = (_REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for needle in (
        "src/foundationscale/agentic_rl/rewards/music/core.py",
        "recipes/design/music/scorer/core.py",
        "recipes/design/music/scorer/baselines/ref_full4k.json",
        "a2ad9f61",
        "Apache License, Version 2.0",
        "Copyright 2026 Bytedance Ltd. and/or its affiliates",
    ):
        assert needle in notices, needle


@pytest.mark.parametrize("buffer", [bytearray(b"\x03\x04"), memoryview(b"\x03\x04")])
def test_byte_buffers_are_never_read_as_token_ids(buffer: object) -> None:
    with pytest.raises(TrajectoryRefusal, match="token_ids"):
        Turn(0, SegmentKind.USER, buffer, False, None)  # type: ignore[arg-type]
    trace = TokenTrace(8)
    trace.append_prompt([1, 2])
    with pytest.raises(TokenTraceRefusal):
        trace.append_observation(buffer)  # type: ignore[arg-type]


def test_an_infra_trajectory_supervises_nothing_even_when_it_holds_sampled_tokens() -> None:
    trajectory = Trajectory(
        uid="g",
        session_id="s",
        harness="h",
        prompt_turns=(Turn(0, SegmentKind.USER, (1, 2), False, None),),
        turns=(Turn(1, SegmentKind.ASSISTANT, (3, 4), True, (-0.1, -0.2)),),
        reward=None,
        abstention_reason="infra:pod_lost",
        termination=Termination.INFRA,
        policy_version_min=None,
        policy_version_max=None,
    )
    assert trajectory.is_infra is True
    assert trajectory.loss_mask() == (0, 0)
    assert trajectory.supervised_token_count == 0
