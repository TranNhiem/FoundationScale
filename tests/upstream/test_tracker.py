"""Offline tests for the read-only upstream release tracker (PHASE 5).

Every test scripts the whole upstream world through a local ``get_json`` -- GitHub
releases and Hugging Face listings are plain Python data here, and a "dead" API is an
Exception instance a route raises -- so no test opens a socket, and the one HTTP entry
point (``http_get_json``) is only ever swapped, never called. What is pinned down: version
ordering and its refusals, draft/prerelease skipping and tag stripping, the model filters
and license transcription, "already registered" flagging, per-source error degradation and
counting, the rendered digest, and the CLI's exit codes (0 while reporting, 2 when every
source is dead).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from foundationscale.upstream import tracker

SINCE = "2026-10-01"


# --- offline upstream world ---------------------------------------------------------------


def _fake_get(routes: dict[str, object]) -> Callable[[str], Any]:
    """A ``get_json`` answering from ``routes``, substring-matched on the URL.

    A route whose value is an ``Exception`` instance is RAISED instead of returned, so one
    dead upstream is scripted without a second fake. A URL no route names fails loudly:
    the tests state every source the tracker is allowed to touch.
    """

    def get_json(url: str) -> Any:
        for fragment, value in routes.items():
            if fragment in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected URL fetched: {url}")

    return get_json


def _pypi(*versions: tuple[str, str] | tuple[str, str, bool]) -> dict[str, Any]:
    """A PyPI /json payload: (version, upload_time[, yanked]) per release."""
    releases: dict[str, Any] = {}
    for v in versions:
        version, uploaded = v[0], v[1]
        yanked = v[2] if len(v) > 2 else False  # type: ignore[misc]
        releases[version] = [{"upload_time_iso_8601": uploaded, "yanked": yanked}]
    return {"releases": releases}


def _hf_model(
    repo_id: str,
    created: str = "2026-10-05T12:00:00.000Z",
    pipeline_tag: str = "automatic-speech-recognition",
    card: dict[str, Any] | None = None,
    tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": repo_id,
        "createdAt": created,
        "pipeline_tag": pipeline_tag,
        "tags": list(tags),
    }
    if card is not None:
        entry["cardData"] = card
    return entry


def _nemo_routes() -> dict[str, object]:
    """The whole upstream world for profile ``nemo-26.08``, with ONE dead source (peft)."""
    return {
        "pypi/nemo_toolkit/json": _pypi(("3.2.0", "2026-11-03T00:00:00Z")),
        "pypi/transformers/json": _pypi(
            ("5.13.0", "2026-11-02T10:00:00Z"),
            ("5.10.0", "2026-09-01T10:00:00Z"),  # below the 5.12.1 pin
        ),
        "pypi/peft/json": OSError("connection reset by peer"),
        "pypi/lhotse/json": _pypi(),
        "pypi/accelerate/json": _pypi(),
        "author=nvidia": [
            _hf_model(
                "nvidia/parakeet-ctc-1.1b",
                created="2026-10-05T12:00:00.000Z",
                card={"license": "cc-by-4.0"},
            ),
            _hf_model(
                "nvidia/new-speech-26b",
                created="2026-10-06T12:00:00.000Z",
                tags=("transformers", "license:cc-by-4.0"),
            ),
        ],
        "author=Qwen": [],
        "author=openai": [],
        "author=google": [],
    }


def _hf_routes() -> dict[str, object]:
    """The whole upstream world for profile ``hf-26.04``: one update, one new model, no failures."""
    return {
        "pypi/transformers/json": _pypi(("5.13.0", "2026-11-02T10:00:00Z")),
        "pypi/peft/json": _pypi(),
        "pypi/accelerate/json": _pypi(),
        "author=nvidia": [
            _hf_model(
                "nvidia/new-speech-26b",
                created="2026-10-06T12:00:00.000Z",
                tags=("license:cc-by-4.0",),
            ),
        ],
        "author=Qwen": [],
        "author=openai": [],
        "author=google": [],
    }


# --- version arithmetic ---------------------------------------------------------------------------


def test_tracked_sources_are_the_documented_ones() -> None:
    assert tracker.TRACKED_REPOS["nemo_toolkit"] == "NVIDIA-NeMo/NeMo"
    assert tracker.TRACKED_REPOS["transformers"] == "huggingface/transformers"
    assert tracker.TRACKED_REPOS["peft"] == "huggingface/peft"
    assert tracker.TRACKED_REPOS["lhotse"] == "lhotse-speech/lhotse"
    assert tracker.TRACKED_REPOS["accelerate"] == "huggingface/accelerate"
    assert tracker.MODEL_ORGS == ("nvidia", "Qwen", "openai", "google")
    assert tracker.SPEECH_TAGS == ("automatic-speech-recognition", "audio-text-to-text")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3.1.0+912d96b", (3, 1, 0)),
        ("2.0.0a6", (2, 0, 0)),
        ("2.13.0a0+8145d630e8.nv26.6.54250401", (2, 13, 0)),
        ("5.5.0", (5, 5, 0)),
        ("0.21.2", (0, 21, 2)),
        ("5.5.0-rc1", (5, 5, 0)),
        ("2.0.0.post1", (2, 0, 0)),
        ("1.26.4", (1, 26, 4)),
        ("  2.4.0 ", (2, 4, 0)),
    ],
)
def test_parse_version_keeps_only_the_release_numbers(raw: str, expected: tuple[int, ...]) -> None:
    assert tracker.parse_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "rc1", "v3.1.0", "V2.0.0", "a1.2.3", "+3.1", ".1", "release-1", "nemo-26.08"],
)
def test_parse_version_refuses_a_string_without_a_leading_number(raw: str) -> None:
    with pytest.raises(ValueError):
        tracker.parse_version(raw)


def _timed(*versions: str) -> list[tracker.Release]:
    return [
        tracker.Release(
            package="transformers",
            version=version,
            published="2026-10-01T00:00:00Z",
            url=f"https://example.invalid/{version}",
        )
        for version in versions
    ]


def test_newer_releases_compares_numbers_and_puts_the_newest_first() -> None:
    # lexical order would call 5.6 above 5.12.1; the order must be numeric.
    got = tracker.newer_releases(_timed("5.5.0", "5.6", "5.12.1", "5.4.0"), "5.5.0")
    assert [release.version for release in got] == ["5.12.1", "5.6"]


def test_newer_releases_returns_everything_when_no_version_is_pinned() -> None:
    got = tracker.newer_releases(_timed("5.5.0", "5.12.1", "5.6"), None)
    assert [release.version for release in got] == ["5.12.1", "5.6", "5.5.0"]


def test_newer_releases_drops_the_pin_itself_and_anything_older() -> None:
    releases = [
        tracker.Release("transformers", "5.12.1", "2026-10-01T00:00:00Z", "u-equal"),
        tracker.Release("transformers", "5.11.0", "2026-09-01T00:00:00Z", "u-older"),
    ]
    assert tracker.newer_releases(releases, "5.12.1") == []


def test_newer_releases_treats_equal_release_numbers_as_not_newer() -> None:
    # 2.0.0 IS 2.0 with a third zero: padding is what keeps a package from looking stale.
    releases = [tracker.Release("lhotse", "2.0.0", "2026-07-01T00:00:00Z", "u")]
    assert tracker.newer_releases(releases, "2.0") == []


def test_newer_releases_refuses_a_version_it_cannot_parse() -> None:
    releases = [tracker.Release("peft", "release-1", "2026-07-01T00:00:00Z", "u")]
    with pytest.raises(ValueError):
        tracker.newer_releases(releases, "0.18.1")


# --- fetchers (offline) ---------------------------------------------------------------------------


def test_fetch_releases_reads_pypi_and_skips_prereleases_yanked_and_empty() -> None:
    payload = _pypi(
        ("5.13.0", "2026-11-02T10:00:00Z"),
        ("5.14.0rc1", "2026-11-05T10:00:00Z"),  # pre-release
        ("5.15.0.dev0", "2026-11-06T10:00:00Z"),  # pre-release
        ("5.11.0", "2026-10-20T10:00:00Z", True),  # fully yanked
        ("5.12.1", "2026-10-01T10:00:00Z"),
    )
    payload["releases"]["5.16.0"] = []  # no files published
    seen: list[str] = []

    def get_json(url: str) -> Any:
        seen.append(url)
        return payload

    got = tracker.fetch_releases("transformers", get_json=get_json)

    assert {release.version for release in got} == {"5.13.0", "5.12.1"}
    assert all(release.package == "transformers" for release in got)
    r = next(x for x in got if x.version == "5.13.0")
    assert r.published == "2026-11-02T10:00:00Z"
    assert r.url == "https://pypi.org/project/transformers/5.13.0/"
    assert seen == ["https://pypi.org/pypi/transformers/json"]


def test_container_style_tags_are_not_compared_as_newer() -> None:
    """NeMo tags container builds (25.09-alpha.rc2); PyPI versions compare like the pin."""
    got = tracker.fetch_releases(
        "nemo_toolkit", get_json=lambda url: _pypi(("3.0.0", "2026-09-01T00:00:00Z"))
    )
    assert tracker.newer_releases(got, "3.1.0+912d96b") == []


def test_fetch_new_models_filters_by_pipeline_tag_and_date_and_reads_licenses() -> None:
    payload = [
        _hf_model(
            "nvidia/with-card-license",
            created="2026-10-05T12:00:00.000Z",
            card={"license": "cc-by-4.0"},
        ),
        _hf_model(
            "nvidia/with-tag-license",
            created="2026-10-06T12:00:00.000Z",
            tags=("transformers", "license:apache-2.0"),
        ),
        _hf_model(
            "nvidia/old-speech",
            created="2026-09-30T12:00:00.000Z",  # before the cutoff
            card={"license": "cc-by-4.0"},
        ),
        _hf_model(
            "nvidia/text-only",
            created="2026-10-07T12:00:00.000Z",  # not a speech pipeline tag
            pipeline_tag="text-generation",
            card={"license": "cc-by-4.0"},
        ),
        _hf_model(
            "nvidia/audio-lg",
            created="2026-10-08T12:00:00.000Z",  # the second speech tag, no license at all
            pipeline_tag="audio-text-to-text",
        ),
        _hf_model(
            "nvidia/exactly-at-cutoff",
            created="2026-10-01T00:00:00.000Z",  # strictly newer only: not newer
        ),
    ]
    seen: list[str] = []

    def get_json(url: str) -> Any:
        seen.append(url)
        return payload

    got = tracker.fetch_new_models("nvidia", SINCE, get_json=get_json)

    # Server-side filtering is the API's job (the fake returns one list for both tag calls);
    # the tracker keeps the date cut, de-duplicates across tags, and sorts newest first.
    assert [model.repo_id for model in got] == [
        "nvidia/audio-lg",
        "nvidia/text-only",
        "nvidia/with-tag-license",
        "nvidia/with-card-license",
    ]
    by_id = {m.repo_id: m for m in got}
    assert by_id["nvidia/with-card-license"].license == "cc-by-4.0"
    assert by_id["nvidia/with-tag-license"].license == "apache-2.0"
    assert by_id["nvidia/audio-lg"].license is None
    assert by_id["nvidia/with-card-license"].created == "2026-10-05T12:00:00.000Z"
    assert seen == [
        "https://huggingface.co/api/models?author=nvidia&pipeline_tag=automatic-speech-recognition"
        "&sort=createdAt&direction=-1&limit=50&full=true",
        "https://huggingface.co/api/models?author=nvidia&pipeline_tag=audio-text-to-text"
        "&sort=createdAt&direction=-1&limit=50&full=true",
    ]


# --- digest ---------------------------------------------------------------------------------------


def test_digest_marks_registered_models_and_counts_a_failed_source_without_aborting() -> None:
    report = tracker.digest("nemo-26.08", SINCE, get_json=_fake_get(_nemo_routes()))

    assert report["profile"] == "nemo-26.08"
    assert report["errors"] == 1

    packages = report["packages"]
    assert set(packages) == {"nemo_toolkit", "transformers", "peft", "lhotse", "accelerate"}
    assert packages["transformers"]["pinned"] == "5.12.1"
    assert packages["transformers"]["newer"] == [
        {
            "package": "transformers",
            "version": "5.13.0",
            "published": "2026-11-02T10:00:00Z",
            "url": "https://pypi.org/project/transformers/5.13.0/",
        }
    ]
    # a pinned local build compares against its own release numbers (3.1.0 vs 3.2.0)
    assert [release["version"] for release in packages["nemo_toolkit"]["newer"]] == ["3.2.0"]
    assert packages["lhotse"]["newer"] == []
    assert packages["accelerate"]["newer"] == []
    # the dead source degrades to an error record and keeps the pin it was asked about
    assert packages["peft"]["error"] == "OSError: connection reset by peer"
    assert packages["peft"]["pinned"] == "0.21.2"

    models = report["new_models"]["nvidia"]["models"]
    # newest first
    assert [model["repo_id"] for model in models] == [
        "nvidia/new-speech-26b",
        "nvidia/parakeet-ctc-1.1b",
    ]
    assert models[0]["registered"] is False
    assert models[0]["license"] == "cc-by-4.0"
    assert models[1]["registered"] is True
    assert report["new_models"]["Qwen"] == {"models": []}
    assert report["new_models"]["google"] == {"models": []}

    refs = report["registered_model_refs"]
    assert "google/gemma-4-E4B-it" in refs
    assert "nvidia/canary-qwen-2.5b" in refs


def test_digest_skips_packages_the_profile_treats_as_absent() -> None:
    # hf-26.04 pins nemo_toolkit and lhotse as None: they are not this backend's stack,
    # so they are not sources at all (and no URL for them is fetched -- the fake would
    # reject an unscripted fetch).
    report = tracker.digest("hf-26.04", SINCE, get_json=_fake_get(_hf_routes()))

    assert set(report["packages"]) == {"transformers", "peft", "accelerate"}
    assert report["errors"] == 0


def test_digest_refuses_a_cutoff_it_cannot_read() -> None:
    with pytest.raises(ValueError):
        tracker.digest("nemo-26.08", "last week", get_json=lambda url: [])


def test_digest_refuses_an_unknown_profile() -> None:
    with pytest.raises(ValueError):
        tracker.digest("nemo-26.99", SINCE, get_json=lambda url: [])


# --- rendering ------------------------------------------------------------------------------------


def test_render_markdown_names_packages_versions_and_errors() -> None:
    report = tracker.digest("nemo-26.08", SINCE, get_json=_fake_get(_nemo_routes()))

    markdown = tracker.render_markdown(report)

    assert "nemo-26.08" in markdown
    assert "transformers -- pinned 5.12.1" in markdown
    assert "5.12.1 -> 5.13.0" in markdown
    assert "https://pypi.org/project/transformers/5.13.0/" in markdown
    assert "nemo_toolkit -- pinned 3.1.0+912d96b" in markdown
    assert "3.1.0+912d96b -> 3.2.0" in markdown
    assert "lhotse -- pinned 2.0.0a6" in markdown
    assert "up to date" in markdown
    assert "peft" in markdown
    assert "OSError: connection reset by peer" in markdown
    assert "nvidia/parakeet-ctc-1.1b" in markdown
    assert "cc-by-4.0" in markdown
    assert "[registered]" in markdown
    assert "[not registered]" in markdown
    assert "Errors: 1 of 9 sources." in markdown


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_exit_0_and_writes_the_report(tmp_path: Any, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(tracker, "http_get_json", _fake_get(_hf_routes()))
    json_out = tmp_path / "nested" / "digest.json"
    md_out = tmp_path / "digest.md"

    code = tracker.main(
        [
            "--profile",
            "hf-26.04",
            "--since",
            SINCE,
            "--json",
            str(json_out),
            "--markdown",
            str(md_out),
        ]
    )

    # updates exist and it still reports only: exit 0 is reserved for "could not report"
    assert code == 0
    captured = capsys.readouterr()
    assert "5.5.0 -> 5.13.0" in captured.out
    assert "nvidia/new-speech-26b" in captured.out
    assert "Errors: 0 of 7 sources." in captured.out
    assert "nemo_toolkit" not in captured.out  # not part of this profile's stack

    written = json.loads(json_out.read_text())
    assert written["profile"] == "hf-26.04"
    assert written["errors"] == 0
    assert written["packages"]["transformers"]["newer"][0]["version"] == "5.13.0"
    assert "5.13.0" in md_out.read_text()


def test_cli_exits_0_when_only_one_source_fails(monkeypatch: Any, capsys: Any) -> None:
    routes = _hf_routes()
    routes["pypi/peft/json"] = OSError("connection reset by peer")
    monkeypatch.setattr(tracker, "http_get_json", _fake_get(routes))

    assert tracker.main(["--profile", "hf-26.04", "--since", SINCE]) == 0
    assert "Errors: 1 of 7 sources." in capsys.readouterr().out


def test_cli_exits_2_when_every_source_fails(monkeypatch: Any, capsys: Any) -> None:
    def always_down(url: str) -> Any:
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr(tracker, "http_get_json", always_down)

    assert tracker.main(["--profile", "nemo-26.08", "--since", SINCE]) == 2

    captured = capsys.readouterr()
    # it degrades to a per-source report of the outage instead of a traceback
    assert "transformers" in captured.out
    assert "OSError: getaddrinfo failed" in captured.out
    assert "Errors: 9 of 9 sources." in captured.out
    assert "all 9 upstream sources failed" in captured.err
