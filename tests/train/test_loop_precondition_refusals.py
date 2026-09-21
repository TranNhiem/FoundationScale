"""Coverage tests for train()'s two PRECONDITION refusal sites.

loop.py 3431-3433 and 3439-3441 refuse (exit 96) before any optional
dependency is imported or any GPU touched: when the declared Topology cannot
even be constructed, and when the ClusterProfile cannot be resolved. The two
call sites are monkeypatched to raise, and each test asserts REACHED-SITE
(the ``[fs:train:refuse]`` marker plus this site's own message text)
separately from OUTCOME (exactly 96 -- REFUSE, distinct from 5 RED, because
nothing ran to adjudicate -- and the absence of every later pipeline marker,
which is the observable proof that no GPU was touched).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundationscale.train import loop


def _base_kwargs(tmp_path: Path) -> dict[str, Any]:
    # The refused arms below exit before profile resolution, dataset loading
    # or Trainer construction, so model/dataset/profile_name values are
    # placeholders that are never consulted.
    return {
        "model": "fake-model",
        "dataset": "fake-dataset",
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "synthetic-profile",
        "max_steps": 2,
        "save_interval": 1,
    }


def test_topology_not_constructible_refuses_96_before_any_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pins loop.py 3431-3433: a Topology constructor failure becomes REFUSE
    96 with the text "topology is not constructible (nothing touched)".

    If deleted: the ``except`` around ``Topology(...)`` can be dropped or
    narrowed, and a misdeclared geometry escapes as a raw traceback -- a
    crash where the contract promises an adjudicated 96. The "(nothing
    touched)" wording is itself part of the contract: it tells the caller no
    side effect happened before the refusal.
    """

    def _unconstructible_topology(**kwargs: Any) -> None:
        raise ValueError("declared degrees cannot tile a device mesh")

    monkeypatch.setattr(loop, "Topology", _unconstructible_topology)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path))

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE: the refusal marker plus THIS site's own message text -- a
    # 96 from any other site (e.g. the unwired-axes batch further up train())
    # cannot satisfy these assertions.
    assert "[fs:train:refuse]" in out
    assert (
        "topology is not constructible (nothing touched): "
        "declared degrees cannot tile a device mesh" in out
    )

    # OUTCOME: exactly 96 (REFUSE) -- never "non-zero": 5 would mean RED, an
    # adjudicated run that failed after starting, and nothing started here.
    assert rc == 96
    assert rc == loop.EXIT_REFUSE

    # NO-GPU PROOF: the refusal precedes the topology announcement itself and
    # every later stage; none of their markers may appear.
    assert "[fs:train:topology]" not in out
    assert "[fs:train:deps]" not in out
    assert "[fs:train:trainer]" not in out


def test_cluster_profile_refused_refuses_96_before_any_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pins loop.py 3439-3441: a ``_resolve_profile`` failure becomes REFUSE
    96 with the text "cluster profile refused:".

    If deleted: a missing or typo'd profile name surfaces as an unhandled
    exception mid-prologue (collapse of 96-REFUSE into a crash), or the
    topology/placement steps re-order and the refusal fires from the wrong
    site. The topology marker assertion is what proves the refusal came from
    the PROFILE site and not the topology site above it.
    """

    class _ConstructibleTopology:
        """Stands in for a valid Topology so the run reaches the profile stage."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def describe(self) -> str:
            return "synthetic 1x1 topology (validation patched out)"

    def _refusing_profile(cfg: Any) -> Any:
        raise RuntimeError("cluster profile 'synthetic-profile' is not known to this host")

    monkeypatch.setattr(loop, "Topology", _ConstructibleTopology)
    monkeypatch.setattr(loop, "_resolve_profile", _refusing_profile)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path))

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE: topology succeeded and was announced first -- this is the
    # control proving we reached the second refusal site -- then the PROFILE
    # site's own refusal fired, carrying the resolver's reason verbatim.
    assert "[fs:train:topology]" in out
    assert "[fs:train:refuse]" in out
    assert "cluster profile refused:" in out
    assert "cluster profile 'synthetic-profile' is not known to this host" in out

    # OUTCOME: exactly 96 (REFUSE). Asserting the code directly -- not a
    # truthy/non-zero check -- is the exit contract: 96 means "refused
    # precondition", which is a different statement from 5 (RED).
    assert rc == 96
    assert rc == loop.EXIT_REFUSE

    # NO-GPU PROOF: the profile never got announced and nothing downstream
    # (dependency import, trainer) ran -- no GPU, no partial state.
    assert "[fs:train:profile]" not in out
    assert "[fs:train:deps]" not in out
    assert "[fs:train:trainer]" not in out
