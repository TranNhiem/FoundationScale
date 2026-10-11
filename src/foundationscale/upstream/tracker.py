"""Read-only upstream release tracker: what moved above our pins, and what speech models are new.

PHASE 5 of docs/research/upstream_integration.md. The upgrade process asks exactly two
questions and the tracker answers exactly those: has a tracked framework published a
release NEWER than the version an upstream profile (profiles.py) was verified on, and has
one of the tracked upstream orgs published a speech model since the cutoff the operator
stated. The answer is a digest of data -- pinned version, newer versions with their
release URLs, new models with their pipeline tag and license -- and every new model is
marked with whether the model registry (models.py) already knows it, so a person sees at a
glance what a release or a model would cost to adopt. Nothing here decides anything: the
tracker reports and exits 0 even when updates exist, because "an upgrade is available" is
information, not a failure. It exits 2 only when EVERY source was unreachable -- a report
assembled out of nothing is not a report -- and even then it renders what it knows, source
by source.

This module is READ-ONLY by construction: one unauthenticated HTTP GET per source (no
credentials, no writes, 20 s timeout) through :func:`http_get_json`, the only place that
touches the network. Every fetcher takes an injectable ``get_json(url)`` and defaults to
it, so tests/upstream/test_tracker.py scripts the whole upstream world offline.

A fetch failure is recorded ON the source it came from -- ``{"error": "<type>: <msg>"}``
next to that package or that org -- counted in ``digest()["errors"]``, and never aborts
the rest: one dead API must not blind the operator to every other upstream (fail closed
per source, degrade as a whole).

Two definitions make the digest comparable:

* :func:`parse_version` reduces a version to its leading release numbers
  (``"3.1.0+912d96b" -> (3, 1, 0)``, ``"2.0.0a6" -> (2, 0, 0)``), so a pinned local build
  compares equal to the release it was built from, and a pre-release suffix cannot invent
  a newer version. A string with no leading number is REFUSED (ValueError): a version we
  cannot parse is not "old", and quietly sorting it somewhere is how a tracker starts
  lying.
* A new model is one whose ``createdAt`` is strictly after ``since_iso``, compared as
  instants (ISO-8601; a timestamp with no zone is read as UTC), with a lexical fallback
  only for a timestamp that does not parse at all.

Stdlib only, Python >= 3.10: this module must import, catalogue and REFUSE in a bare venv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from foundationscale.upstream import models as upstream_models
from foundationscale.upstream import profiles as upstream_profiles

__all__ = [
    "GetJson",
    "MODEL_ORGS",
    "NewModel",
    "Release",
    "SPEECH_TAGS",
    "TRACKED_REPOS",
    "digest",
    "fetch_new_models",
    "fetch_releases",
    "http_get_json",
    "main",
    "newer_releases",
    "parse_version",
    "render_markdown",
]

# The one fetch shape this module speaks: a GET, answered as parsed JSON or raised.
GetJson = Callable[[str], Any]

HTTP_TIMEOUT_SECONDS = 20
USER_AGENT = "foundationscale-upstream-tracker (read-only; stdlib urllib)"

RELEASES_PER_CALL = 20
MODELS_PER_CALL = 50

# The framework Python packages whose releases decide what a backend can run, and the
# GitHub repository each releases from. It is a SUBSET of profiles.TRACKED_PACKAGES on
# purpose: only these five publish GitHub releases worth tracking (numpy/soundfile/... are
# released through PyPI and would add noise to an upgrade decision).
TRACKED_REPOS: dict[str, str] = {
    "nemo_toolkit": "NVIDIA-NeMo/NeMo",
    "transformers": "huggingface/transformers",
    "peft": "huggingface/peft",
    "lhotse": "lhotse-speech/lhotse",
    "accelerate": "huggingface/accelerate",
}

# The upstream orgs whose speech uploads matter to this plane. Spelling is the Hugging Face
# author spelling, case included ("Qwen"), so the query is exact and reproducible.
MODEL_ORGS: tuple[str, ...] = ("nvidia", "Qwen", "openai", "google")

# The pipeline tags this plane can actually drive (the ModelKind shapes of models.py):
# transcription and audio->text generation, nothing else.
SPEECH_TAGS: tuple[str, ...] = ("automatic-speech-recognition", "audio-text-to-text")


@dataclass(frozen=True)
class Release:
    """One stable upstream release of a tracked package, as pointers only.

    A signpost, not a snapshot: the four fields are what a person needs to go read the
    release notes and decide, and nothing about installability (the asset, the wheel, a
    hash) is here, because the tracker never installs anything. ``version`` is the release
    TAG with one leading ``v`` stripped, so it compares against profile pins with
    :func:`parse_version`; ``published`` is kept as the API's ISO text rather than a
    parsed instant, because the digest transcribes what upstream said.
    """

    package: str
    version: str  # the GitHub tag with one leading "v"/"V" removed
    published: str  # ISO timestamp; "" when the API published none
    url: str


@dataclass(frozen=True)
class NewModel:
    """One speech model a tracked upstream org published after the digest's cutoff.

    Only the facts the API states and licensing reads: the repo id exactly as upstream
    spells it (registry entries pin these case-included), when it was created, its
    pipeline tag, and its license from the model card (``cardData.license``) or a
    ``license:`` tag -- None when the card says nothing, which is itself a finding and is
    rendered as "license: none" rather than guessed.
    """

    repo_id: str
    created: str  # ISO timestamp as the API returned it
    pipeline_tag: str
    license: str | None


_VERSION_HEAD = re.compile(r"\d+(?:\.\d+)*")


def parse_version(version: str) -> tuple[int, ...]:
    """The leading release numbers of ``version``, for ordering and equality.

    Only the release numbers count; everything after them is build or pre-release detail
    of the SAME release and is ignored on purpose:

    * ``"3.1.0+912d96b"`` -> ``(3, 1, 0)`` -- a pinned local build IS its release.
    * ``"2.0.0a6"`` -> ``(2, 0, 0)`` -- ``a6`` must not sort above ``2.0.0``.
    * ``"2.13.0a0+8145d630e8.nv26.6.54250401"`` -> ``(2, 13, 0)`` -- the NV wheel suffix
      is not a longer version number.

    A string with no leading number (``"rc1"``, ``"v3.1.0"``, ``""``) raises ValueError:
    a "v" is a TAG decoration that :class:`Release` strips on the way in, and a version we
    cannot read must not be quietly treated as older than anything.
    """
    text = version.strip()
    match = _VERSION_HEAD.match(text)
    if match is None:
        raise ValueError(
            f"cannot parse a release version from {version!r}: expected a leading release "
            "number such as '3.2.0' (local and pre-release suffixes are ignored on purpose)"
        )
    return tuple(int(part) for part in match.group(0).split("."))


def _pad(key: tuple[int, ...], width: int) -> tuple[int, ...]:
    """Zero-pad a release tuple to ``width`` so ``(2, 13)`` and ``(2, 13, 0)`` compare equal.

    Comparing raw tuples would call ``(2, 13, 0)`` newer than ``(2, 13)``: a shorter
    spelling of the same release would make a package look out of date forever.
    """
    return key + (0,) * (width - len(key))


def newer_releases(releases: list[Release], pinned: str | None) -> list[Release]:
    """The releases strictly newer than ``pinned``, newest first.

    ``pinned=None`` means "no version is pinned here", so every release is reported (still
    newest first). The pinned release itself and anything older is dropped: a tracker that
    re-reports the version we already run trains operators to stop reading it. Ordering is
    by :func:`parse_version`, which is why 5.12.1 sorts above 5.6 (numeric, not lexical).

    Raises ValueError when a pinned or released version is not parseable, so a comparison
    is either done honestly or refused.
    """
    keys: list[tuple[Release, tuple[int, ...]]] = [
        (release, parse_version(release.version)) for release in releases
    ]
    width = max((len(key) for _, key in keys), default=0)
    if pinned is not None:
        pinned_key = parse_version(pinned)
        width = max(width, len(pinned_key))
        keys = [
            (release, key) for release, key in keys if _pad(key, width) > _pad(pinned_key, width)
        ]
    ordered = sorted(keys, key=lambda pair: _pad(pair[1], width), reverse=True)
    return [release for release, _ in ordered]


def http_get_json(url: str) -> Any:
    """GET ``url`` and decode the body as JSON: the one place this module touches the network.

    Read-only and unauthenticated -- a GET with a User-Agent (GitHub refuses anonymous
    calls without one) and GitHub's JSON ``Accept`` header on api.github.com. A 20-second
    timeout and no retries: the caller records a failure per source and this must answer
    or fail promptly so a half-dead internet still yields a digest.
    """
    headers = {"User-Agent": USER_AGENT}
    if "api.github.com" in url:
        headers["Accept"] = "application/vnd.github+json"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _resolve_getter(get_json: GetJson | None) -> GetJson:
    """The fetch callable to use.

    ``None`` (the default of every fetcher) means :func:`http_get_json` resolved at CALL
    time: tests and offline runs swap the single HTTP entry point without threading a
    callable through every call.
    """
    return http_get_json if get_json is None else get_json


def _strip_tag_v(tag: str) -> str:
    """One leading ``v``/``V`` of a release tag removed: ``"v5.13.0"`` -> ``"5.13.0"``.

    Only when a digit follows, so a tag like ``"version-tests"`` is left exactly as
    upstream spelled it and later refused by :func:`parse_version` instead of being
    mangled into one first.
    """
    if len(tag) > 1 and tag[0] in "vV" and tag[1].isdigit():
        return tag[1:]
    return tag


def _error_entry(exc: Exception) -> dict[str, str]:
    """The per-source failure record: ``{"error": "<type>: <msg>"}``."""
    return {"error": f"{type(exc).__name__}: {exc}"}


_PRERELEASE = re.compile(r"(a|b|rc|dev|alpha|beta|pre)\d*", re.IGNORECASE)


def fetch_releases(package: str, get_json: GetJson | None = None) -> list[Release]:
    """The stable published releases of ``package``, from PyPI (``/pypi/{package}/json``).

    PyPI, not GitHub tags: profile pins are pip versions, and repositories tag differently
    (NeMo tags container builds like ``25.09-alpha.rc2``, which would compare as "newer" than
    pip's ``3.1.0`` -- measured 2026-10-10). Pre-releases (a/b/rc/dev), fully yanked releases and
    releases with no files are skipped: a digest is an upgrade trigger, and those are not
    upgrades. ``package`` must be tracked; payloads that are not the expected shape are refused.
    """
    if package not in TRACKED_REPOS:
        tracked = ", ".join(sorted(TRACKED_REPOS))
        raise ValueError(
            f"not a tracked upstream package: {package!r}; TRACKED_REPOS holds: {tracked}"
        )
    payload = _resolve_getter(get_json)(f"https://pypi.org/pypi/{package}/json")
    releases = payload.get("releases") if isinstance(payload, Mapping) else None
    if not isinstance(releases, Mapping):
        raise ValueError(f"unexpected PyPI payload for {package!r}: no 'releases' mapping")
    out: list[Release] = []
    for version, files in releases.items():
        if not isinstance(version, str) or not isinstance(files, list) or not files:
            continue
        if _PRERELEASE.search(version.split("+")[0]):
            continue
        if all(isinstance(f, Mapping) and f.get("yanked") for f in files):
            continue
        uploads = [
            str(f.get("upload_time_iso_8601") or f.get("upload_time") or "")
            for f in files
            if isinstance(f, Mapping)
        ]
        out.append(
            Release(
                package=package,
                version=version,
                published=min(u for u in uploads if u) if any(uploads) else "",
                url=f"https://pypi.org/project/{package}/{version}/",
            )
        )
    return out


def _license_of(item: Mapping[str, Any]) -> str | None:
    """The license of one Hugging Face model listing: card first, then a ``license:`` tag.

    ``cardData.license`` is what the card declares; the ``license:`` tags are what the
    upload metadata carries when it is absent (the API mirrors both). A multi-license card
    (a list) is kept as one string joined with ``+``; nothing is inferred, and a model that
    says nothing yields None so the digest can say "license: none".
    """
    card = item.get("cardData")
    if isinstance(card, Mapping):
        value = card.get("license")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = [part.strip() for part in value if isinstance(part, str) and part.strip()]
            if parts:
                return "+".join(parts)
    tags = item.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                name = tag.split(":", 1)[1].strip()
                if name:
                    return name
    return None


def fetch_new_models(org: str, since_iso: str, get_json: GetJson | None = None) -> list[NewModel]:
    """The speech models ``org`` published strictly after ``since_iso``, newest first.

    One Hugging Face listing call PER speech tag, filtered server-side
    (``pipeline_tag=...&sort=createdAt&direction=-1&full=true``). Filtering the org's latest
    fifty uploads client-side missed speech models whenever the org published more than fifty
    other models in between (nvidia: 22 text-generation, 6 robotics ... in the latest fifty,
    measured 2026-10-10). ``full=true`` carries the card metadata, where the license lives.
    Unreadable payloads and entries without an id or a creation date are refused.
    """
    getter = _resolve_getter(get_json)
    seen: dict[str, NewModel] = {}
    for tag in SPEECH_TAGS:
        query = urllib.parse.urlencode(
            {
                "author": org,
                "pipeline_tag": tag,
                "sort": "createdAt",
                "direction": "-1",
                "limit": str(MODELS_PER_CALL),
                "full": "true",
            }
        )
        payload = getter(f"https://huggingface.co/api/models?{query}")
        if not isinstance(payload, list):
            raise ValueError(
                f"unexpected Hugging Face /api/models payload for {org!r}: expected a list, "
                f"got {type(payload).__name__}"
            )
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"unexpected Hugging Face /api/models entry for {org!r}: expected an "
                    f"object, got {type(item).__name__}"
                )
            repo_id = item.get("id")
            created = item.get("createdAt")
            if (
                not isinstance(repo_id, str)
                or not repo_id.strip()
                or not isinstance(created, str)
                or not created.strip()
            ):
                raise ValueError(
                    f"Hugging Face /api/models entry without id/createdAt for {org!r}: {repo_id!r}"
                )
            if not _strictly_after(created, since_iso):
                continue
            seen.setdefault(
                repo_id,
                NewModel(
                    repo_id=repo_id,
                    created=created,
                    pipeline_tag=str(item.get("pipeline_tag") or tag),
                    license=_license_of(item),
                ),
            )
    return sorted(seen.values(), key=lambda m: m.created, reverse=True)


_ISO_TEXT = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"(?:[T ](?P<hour>\d{2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?:[.,]\d+)?)?)?"
    r"(?P<zone>Z|z|[+-]\d{2}:?\d{2})?$"
)


def _utc_offset(zone: str | None) -> timezone:
    """The timezone of a trailing ISO zone (``Z``, ``+02:00``); UTC when there is none."""
    text = (zone or "").strip()
    if text in ("", "Z", "z"):
        return timezone.utc
    sign = -1 if text[0] == "-" else 1
    digits = text[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))


def _iso_instant(value: str) -> datetime | None:
    """An ISO date or timestamp as a timezone-aware instant (no zone means UTC), else None.

    Accepts exactly the shapes the upstream APIs produce -- ``2026-10-01`` and
    ``2026-10-05T12:00:00.000Z`` -- plus offsets, and returns None rather than raising on
    anything else (an out-of-range clock is not a timestamp we can compare).
    """
    match = _ISO_TEXT.match(value.strip())
    if match is None:
        return None
    field = match.groupdict()
    try:
        naive = datetime(
            int(field["year"]),
            int(field["month"]),
            int(field["day"]),
            int(field["hour"] or 0),
            int(field["minute"] or 0),
            int(field["second"] or 0),
        )
        offset = _utc_offset(field["zone"])
    except ValueError:
        return None
    return naive.replace(tzinfo=offset)


def _strictly_after(value: str, since: str) -> bool:
    """True when the ISO timestamp ``value`` is strictly after ``since``.

    Instants when both sides parse (zone-less is read as UTC): a model created any hour of
    2026-10-05 is after ``2026-10-01``. When one side does not parse, the comparison falls
    back to lexical order on the raw text instead of dropping the model on the floor -- a
    wrong order on a garbage timestamp is the lesser failure here.
    """
    left = _iso_instant(value)
    right = _iso_instant(since)
    if left is not None and right is not None:
        return left > right
    return value.strip() > since.strip()


def digest(profile_name: str, since_iso: str, get_json: GetJson | None = None) -> dict[str, Any]:
    """One read-only report over every tracked source, degrading source by source.

    The mapping is JSON-shaped (``--json`` writes it verbatim), so its nested values are
    typed conservatively::

        {
          "profile": "nemo-26.08", "backend": "nemo", "since": "2026-10-01",
          "generated_at": "2026-10-12T09:00:00+00:00",
          "packages": {"transformers": {"pinned": "5.12.1",
                                        "newer": [{"package": "transformers", "version": "5.13.0",
                                                   "published": "...", "url": "..."}]}},
          "new_models": {"nvidia": {"models": [{"repo_id": "...", "created": "...",
                                                "pipeline_tag": "...", "license": "...",
                                                "registered": false}]}},
          "registered_model_refs": ["google/gemma-4-E4B-it", "..."],
          "errors": 1,
        }

    ``packages`` holds every tracked package the profile PINS (a package the profile
    records as absent -- its pin is None -- is not in this backend's stack and is not a
    source at all). ``new_models`` covers every tracked org. A fetch failure -- GitHub for
    one package, Hugging Face for one org -- replaces that source's answer with
    ``{"error": "<type>: <msg>"}`` (the package keeps its ``pinned`` so the report still
    says what we run), is counted in ``errors``, and the other sources still answer.

    ``registered_model_refs`` is the set of ``upstream_ref`` strings the model registry
    (upstream.models.MODELS) already knows, and a new model whose repo id matches one of
    them carries ``registered: true`` -- matched case-insensitively because the hub is,
    while the refs themselves keep upstream's exact spelling.

    Raises ValueError for a profile or a ``since_iso`` that cannot be read: those are
    argument errors, not an upstream outage, and a digest compared against nonsense is
    worse than no digest.
    """
    if _iso_instant(since_iso) is None:
        raise ValueError(
            f'since_iso is not an ISO-8601 date or timestamp: {since_iso!r} (e.g. "2026-10-01")'
        )
    profile = upstream_profiles.get_profile(profile_name)
    getter = _resolve_getter(get_json)
    registered = {entry.upstream_ref.lower(): entry.id for entry in upstream_models.MODELS}

    packages: dict[str, Any] = {}
    new_models: dict[str, Any] = {}
    errors = 0

    for package in TRACKED_REPOS:
        pinned = profile.packages.get(package)
        if pinned is None:
            continue
        source: dict[str, Any] = {"pinned": pinned}
        try:
            newer = newer_releases(fetch_releases(package, get_json=getter), pinned)
            source["newer"] = [
                {
                    "package": release.package,
                    "version": release.version,
                    "published": release.published,
                    "url": release.url,
                }
                for release in newer
            ]
        except Exception as exc:  # one dead API degrades ITS source, never the digest
            errors += 1
            source.update(_error_entry(exc))
        packages[package] = source

    for org in MODEL_ORGS:
        try:
            found = fetch_new_models(org, since_iso, get_json=getter)
            new_models[org] = {
                "models": [
                    {
                        "repo_id": model.repo_id,
                        "created": model.created,
                        "pipeline_tag": model.pipeline_tag,
                        "license": model.license,
                        "registered": model.repo_id.lower() in registered,
                    }
                    for model in found
                ]
            }
        except Exception as exc:  # one dead API degrades ITS source, never the digest
            errors += 1
            new_models[org] = _error_entry(exc)

    return {
        "profile": profile.name,
        "backend": profile.backend,
        "since": since_iso,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packages": packages,
        "new_models": new_models,
        "registered_model_refs": sorted(entry.upstream_ref for entry in upstream_models.MODELS),
        "errors": errors,
    }


def _as_mapping(value: object) -> Mapping[str, Any]:
    """A mapping when ``value`` is one, ``{}`` otherwise: rendering never raises."""
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def render_markdown(d: Mapping[str, Any]) -> str:
    """The digest as a short human report: packages, then new models, then the errors line.

    One section per tracked package naming its pin and every release above it with its
    date and URL ("up to date" when there is none), one section per upstream org listing
    new speech models with creation date, pipeline tag, license and the registered flag,
    and a final line with how many sources failed out of how many were attempted. A failed
    source renders as its ``{"error": ...}`` entry -- the report shows the hole instead of
    papering over it.
    """
    profile = _as_text(d.get("profile")) or "unknown profile"
    since = _as_text(d.get("since")) or "unspecified cutoff"
    backend = _as_text(d.get("backend"))
    lines: list[str] = [
        f"Upstream digest (read-only) -- profile {profile}{' / ' + backend if backend else ''}",
        f"Cutoff: speech models created after {since}.",
        "",
        "## releases above the pins",
    ]

    packages = _as_mapping(d.get("packages"))
    if not packages:
        lines += ["", "(no tracked package is pinned in this profile)"]
    for package, raw in packages.items():
        source = _as_mapping(raw)
        pinned = _as_text(source.get("pinned")) or "unpinned"
        lines += ["", f"### {package} -- pinned {pinned}"]
        error = _as_text(source.get("error"))
        if error:
            lines.append(f"- error: {error}")
            continue
        newer = _as_list(source.get("newer"))
        if not newer:
            lines.append("- up to date")
        for item in newer:
            release = _as_mapping(item)
            version = _as_text(release.get("version"))
            published = _as_text(release.get("published"))
            url = _as_text(release.get("url"))
            lines.append(f"- {pinned} -> {version}{f' ({published})' if published else ''}: {url}")

    lines += ["", "## new speech models"]
    new_models = _as_mapping(d.get("new_models"))
    if not new_models:
        lines += ["", "(no upstream org tracked)"]
    for org, raw in new_models.items():
        source = _as_mapping(raw)
        lines += ["", f"### {org}"]
        error = _as_text(source.get("error"))
        if error:
            lines.append(f"- error: {error}")
            continue
        models = _as_list(source.get("models"))
        if not models:
            lines.append("- none since the cutoff")
        for item in models:
            model = _as_mapping(item)
            pipeline = _as_text(model.get("pipeline_tag")) or "pipeline: none"
            license_name = _as_text(model.get("license")) or "license: none"
            flag = "[registered]" if model.get("registered") is True else "[not registered]"
            lines.append(
                f"- {_as_text(model.get('repo_id'))} "
                f"({_as_text(model.get('created'))}, {pipeline}, {license_name}) {flag}"
            )

    total = len(packages) + len(new_models)
    lines += ["", f"Errors: {_as_count(d.get('errors'))} of {total} sources."]
    return "\n".join(lines)


def _write_output(path: str, text: str) -> None:
    """Write ``text`` to ``path``, creating the parent directories first."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI: ``python -m foundationscale.upstream.tracker --profile P --since YYYY-MM-DD``.

    The rendered digest always goes to stdout; ``--json`` and ``--markdown`` write the same
    digest to files as well. The exit code says whether the tracker could do its job, never
    whether an upgrade is available: 0 whenever a digest was produced, with updates and
    with partial API failures alike, and 2 when every source failed (or the profile or the
    cutoff cannot be read), because a report built out of nothing is not a report.
    """
    parser = argparse.ArgumentParser(
        prog="python -m foundationscale.upstream.tracker",
        description=(
            "Read-only upstream digest: releases above a backend profile's pins and new "
            "speech models from the tracked upstream orgs. Reports only; never decides."
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=[profile.name for profile in upstream_profiles.PROFILES],
        metavar="NAME",
        help="upstream profile whose pinned versions the releases are compared against",
    )
    parser.add_argument(
        "--since",
        required=True,
        help="only models created strictly after this ISO date/timestamp are reported",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        metavar="OUT",
        help="also write the digest as JSON to OUT",
    )
    parser.add_argument(
        "--markdown",
        dest="markdown_out",
        metavar="OUT",
        help="also write the rendered digest to OUT",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = digest(args.profile, args.since)
    except ValueError as exc:
        print(f"tracker: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(report)
    print(markdown)
    if args.json_out is not None:
        _write_output(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out is not None:
        _write_output(args.markdown_out, markdown + "\n")

    sources = len(_as_mapping(report.get("packages"))) + len(_as_mapping(report.get("new_models")))
    errors = _as_count(report.get("errors"))
    if sources > 0 and errors >= sources:
        print(f"tracker: all {sources} upstream sources failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
