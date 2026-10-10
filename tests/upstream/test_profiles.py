from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationscale.upstream.profiles import (
    PROFILES,
    TRACKED_PACKAGES,
    UpstreamProfile,
    get_profile,
    installed_versions,
    profile_mismatches,
    write_profile_record,
)


def test_profiles_pin_every_tracked_package_and_differ_where_measured() -> None:
    for profile in PROFILES:
        assert set(profile.packages) == set(TRACKED_PACKAGES)
    nemo, hf = get_profile("nemo-26.08"), get_profile("hf-26.04")
    assert nemo.backend == "nemo" and hf.backend == "hf"
    assert nemo.packages["torch"] != hf.packages["torch"]  # the stacks really differ (rule 3)
    assert hf.packages["nemo_toolkit"] is None


def test_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown upstream profile"):
        get_profile("nope")


def test_mismatches_name_every_difference() -> None:
    profile = UpstreamProfile(
        name="p",
        backend="hf",
        container="c",
        python="3.12",
        packages={"a": "1", "b": None, "c": "2"},
    )
    assert profile_mismatches(profile, {"a": "1", "b": None, "c": "2"}) == []
    assert profile_mismatches(profile, {"a": "1.1", "b": "0.1", "c": None}) == [
        "version:a:1!=1.1",
        "unexpected:b",
        "missing:c",
    ]


def test_installed_versions_reports_absent_packages_as_none() -> None:
    got = installed_versions(["pytest", "definitely-not-a-real-package-xyz"])
    assert got["pytest"] is not None
    assert got["definitely-not-a-real-package-xyz"] is None


def test_write_profile_record(tmp_path: Path) -> None:
    record = write_profile_record(tmp_path / "out", "hf-26.04")
    on_disk = json.loads((tmp_path / "out" / "upstream_profile.json").read_text())
    assert on_disk == json.loads(json.dumps(record, sort_keys=True))
    assert on_disk["profile"] == "hf-26.04"
    assert isinstance(on_disk["mismatches"], list)
    bare = write_profile_record(tmp_path / "bare")
    assert "mismatches" not in bare and bare["profile"] is None
