"""A5.2: FS capability manifests are read generically and every entry is verified by measurement."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from foundationskills.cli import main
from foundationskills.interfaces.fs import manifest as mf


def with_sources(monkeypatch: pytest.MonkeyPatch, **manifests: object) -> None:
    monkeypatch.setattr(
        mf, "_sources", lambda: ({k: (lambda v=v: v) for k, v in manifests.items()}, [])
    )


def entry(key: str, declared: object, verify: object, compare: str = "equal") -> dict:
    return {"key": key, "declared": declared, "verify": verify, "compare": compare}


def status(rows: list[dict]) -> dict[str, str]:
    return {r["key"]: r["status"] for r in rows}


def test_each_compare_mode_verifies_or_contradicts(monkeypatch: pytest.MonkeyPatch) -> None:
    with_sources(
        monkeypatch,
        mod=[
            entry("eq.ok", ["a", "b"], lambda: ("b", "a")),
            entry("sub.ok", ["a", "b", "c"], lambda: ["a"], "subset"),
            entry("sub.empty", ["a"], lambda: [], "subset"),
            entry("sup.ok", ["local"], lambda: ("enroot", "local"), "superset"),
            entry("sup.bad", ["local", "enroot"], lambda: ("local",), "superset"),
        ],
    )
    rows = mf.read()
    assert status(rows) == {
        "eq.ok": "verified",
        "sub.ok": "verified",
        "sub.empty": "contradicted",
        "sup.ok": "verified",
        "sup.bad": "contradicted",
    }
    bad = next(r for r in rows if r["key"] == "sup.bad")
    assert (
        bad["measured"] == ["local"] and "declared" in bad["reason"]
    )  # measured overrides the claim
    assert mf.exit_code(rows) == 5


def test_nothing_measurable_is_unmeasured_never_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[str]:
        raise RuntimeError("decoder missing")

    with_sources(
        monkeypatch,
        mod=[
            entry("no.verify", ["x"], None),
            entry("raises", ["x"], boom),
            entry("bad.compare", ["x"], lambda: ["x"], "roughly"),
            entry("bad.target", ["x"], "not-a-dotted-path"),
            {"declared": ["no key"]},
            entry("raises", ["dup"], lambda: ["dup"]),
        ],
    )
    rows = mf.read()
    assert all(r["status"] == "unmeasured" for r in rows) and len(rows) == 6
    reasons = " | ".join(r["reason"] for r in rows)
    for fragment in (
        "no verify",
        "decoder missing",
        "compare must be",
        "module:callable",
        "not a mapping",
        "duplicate key",
    ):
        assert fragment in reasons
    assert mf.exit_code(rows) == 95


def test_unmeasured_outranks_contradicted_and_empty_is_unmeasured() -> None:
    assert mf.exit_code([{"status": "contradicted"}, {"status": "unmeasured"}]) == 95
    assert mf.exit_code([{"status": "verified"}]) == 0
    assert mf.exit_code([]) == 95


def test_a_manifest_that_fails_to_load_is_a_row_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> list:
        raise ImportError("heavy dependency missing")

    monkeypatch.setattr(
        mf, "_sources", lambda: ({"pkg.broken": broken, "pkg.notlist": lambda: "x"}, [])
    )
    rows = mf.read()
    assert [r["status"] for r in rows] == ["unmeasured", "unmeasured"]
    assert "heavy dependency missing" in rows[0]["reason"] and "must be a list" in rows[1]["reason"]


def test_discovers_capability_manifest_files_without_entry_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "foundationscale"
    (pkg / "newmod").mkdir(parents=True)
    (pkg / "newmod" / "__init__.py").write_text("")
    (pkg / "newmod" / "capability_manifest.py").write_text(
        "MANIFEST = [{'key': 'newmod.ok', 'declared': [], 'verify': 'os:getcwd'}]\n"
    )
    fake = types.ModuleType("foundationscale")
    fake.__path__ = [str(pkg)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "foundationscale", fake)
    for name in ("foundationscale.newmod", "foundationscale.newmod.capability_manifest"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(mf.metadata, "entry_points", lambda **_kw: [])
    sources, failures = mf._sources()
    assert failures == [] and list(sources) == ["foundationscale.newmod.capability_manifest"]
    loaded = sources["foundationscale.newmod.capability_manifest"]()
    assert (
        loaded[0]["key"] == "newmod.ok"
    )  # a new module is picked up with no FoundationSkills edit


def test_fs_missing_is_one_unmeasured_discovery_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "foundationscale", None)
    rows = mf.read()
    assert len(rows) == 1 and rows[0]["key"] == "<discovery>" and rows[0]["status"] == "unmeasured"


def test_real_fs_manifests_verify_here() -> None:
    rows = mf.read()
    assert {"video.decode_backends", "agentic_rl.env_backends"} <= set(status(rows))
    assert all(r["status"] in ("verified", "contradicted", "unmeasured") for r in rows)


def test_cli_manifest_check_exit_follows_the_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with_sources(monkeypatch, mod=[entry("k", ["a"], lambda: ["a"])])
    assert main(["manifest", "check"]) == 0
    with_sources(monkeypatch, mod=[entry("k", ["a"], lambda: ["b"])])
    assert main(["manifest", "check", "--json"]) == 5
    assert '"contradicted"' in capsys.readouterr().out
