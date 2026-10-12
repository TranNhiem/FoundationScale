"""foundationskills.intake.arch — provable arm64 sibling of an amd64-pinned base image.

Thin stdlib-only (``urllib.request``, ``json``) helper module.  All network
access is read-only HTTPS GETs against public registry APIs and every fetch is
injectable so the claims below can be re-measured offline with a FAKE fetcher.

Statuses: ``PASS`` / ``RED`` / ``UNMEASURED`` / ``REFUSED``.  Absence of proof is
``UNMEASURED`` — never ``PASS``.  ``resolve_arm64`` never raises.

Proven protocol (measured on ``python:3.12-slim-bookworm`` @ ``sha256:6e13...``)
-----------------------------------------------------------------------------
The current tag index does NOT contain the pin, but the exact-version tag
``python:3.12.13-slim-bookworm`` index DOES; its arm64 entry is the same
release.  Registry access proven on that path:

* ``docker.io`` — token ``GET
  https://auth.docker.io/token?service=registry.docker.io&scope=repository:<repo>:pull``
  -> ``json["token"]``; API base ``https://registry-1.docker.io/v2/<repo>``.
* ``ghcr.io`` — token ``GET https://ghcr.io/token?scope=repository:<repo>:pull``
  -> ``json["token"]``; base ``https://ghcr.io/v2/<repo>``.
* ``mcr.microsoft.com`` and ``quay.io`` — anonymous; base ``https://<host>/v2/<repo>``.
* any other host -> ``UNMEASURED`` ("registry not supported").

Manifests: ``GET <base>/manifests/<tag-or-digest>`` with the ``Accept`` header
carrying the 4 media types (OCI index, docker manifest list, OCI manifest,
docker manifest v2) plus ``Authorization: Bearer <token>`` when a token exists.
An index has ``"manifests": [{"digest", "platform": {"architecture",
"os", "variant"}}]``.  Config blob: ``GET <base>/blobs/<config digest>``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "ArchResolution",
    "PASS",
    "RED",
    "UNMEASURED",
    "REFUSED",
    "Status",
    "parse_ref",
    "resolve_arm64",
    "resolve_all",
    "stage_task",
    "default_fetch_json",
    "MANIFEST_ACCEPT",
]

# --------------------------------------------------------------------- statuses

PASS = "PASS"
RED = "RED"
UNMEASURED = "UNMEASURED"
REFUSED = "REFUSED"


class Status:
    """Status vocabulary (plain strings: ``Status.PASS == "PASS"``)."""

    PASS = "PASS"
    RED = "RED"
    UNMEASURED = "UNMEASURED"
    REFUSED = "REFUSED"


# --------------------------------------------------------------- media constants

OCI_INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_MANIFEST_V2 = "application/vnd.docker.distribution.manifest.v2+json"

MANIFEST_ACCEPT = ", ".join(
    [OCI_INDEX, DOCKER_MANIFEST_LIST, OCI_MANIFEST, DOCKER_MANIFEST_V2]
)

_TOKEN_ENDPOINTS: Dict[str, str] = {
    "docker.io": (
        "https://auth.docker.io/token?service=registry.docker.io"
        "&scope=repository:{repo}:pull"
    ),
    "ghcr.io": "https://ghcr.io/token?scope=repository:{repo}:pull",
}

_API_BASES: Dict[str, str] = {
    "docker.io": "https://registry-1.docker.io/v2/",
    "ghcr.io": "https://ghcr.io/v2/",
    "mcr.microsoft.com": "https://mcr.microsoft.com/v2/",
    "quay.io": "https://quay.io/v2/",
}

_VERSION_ENV_NAMES: Tuple[str, ...] = (
    "PYTHON_VERSION",
    "NODE_VERSION",
    "GOLANG_VERSION",
    "RUBY_VERSION",
    "JAVA_VERSION",
    "PHP_VERSION",
)
_VERSION_LABEL = "org.opencontainers.image.version"

_FROM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<kw>[Ff][Rr][Oo][Mm])(?P<sp>[ \t]+)(?P<body>\S.*)$"
)

FetchJson = Callable[[str, Mapping[str, str]], Tuple[Dict[str, Any], Mapping[str, str]]]


# -------------------------------------------------------------------- fetch glue


def default_fetch_json(
    url: str, headers: Optional[Mapping[str, str]] = None
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Read-only HTTPS GET through ``urllib`` with a 30 s timeout."""
    request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        response_headers = {str(k).lower(): v for k, v in response.headers.items()}
    document = json.loads(raw.decode("utf-8"))
    return document, response_headers


def _err(exc: BaseException) -> str:
    return "%s: %s" % (type(exc).__name__, exc)


# --------------------------------------------------------------------- parse_ref


def parse_ref(ref: str) -> Tuple[str, str, str]:
    """Split ``ref`` into ``(host, repo, tag)``.

    e.g. "python:3.12-slim" -> ("docker.io","library/python","3.12-slim");
    "mcr.microsoft.com/playwright:v1.53.1-noble"
        -> ("mcr.microsoft.com","playwright","v1.53.1-noble");
    "ghcr.io/o/r:t" -> ("ghcr.io","o/r","t").
    """
    name = str(ref).strip()
    if "@" in name:
        name = name.split("@", 1)[0]

    tag = "latest"
    last_slash = name.rfind("/")
    last_colon = name.rfind(":")
    if last_colon > last_slash:
        tag = name[last_colon + 1:] or "latest"
        name = name[:last_colon]

    slash = name.find("/")
    if slash > 0:
        first = name[:slash]
        if "." in first or ":" in first or first == "localhost":
            host, repo = first, name[slash + 1:]
        else:
            host, repo = "docker.io", name
    else:
        host, repo = "docker.io", name

    if host == "docker.io" and "/" not in repo:
        repo = "library/" + repo

    return host, repo, tag


# ------------------------------------------------------------- resolution object


@dataclass(frozen=True)
class ArchResolution:
    """One measured claim: is the pin's exact release proven to have arm64?"""

    ref: str
    pinned: str
    status: str
    arm64_digest: Optional[str] = None
    proof_tag: Optional[str] = None
    reason: str = ""


def _make(
    ref: str,
    pinned: str,
    status: str,
    arm64_digest: Optional[str] = None,
    proof_tag: Optional[str] = None,
    reason: str = "",
) -> ArchResolution:
    return ArchResolution(
        ref=ref,
        pinned=pinned,
        status=status,
        arm64_digest=arm64_digest,
        proof_tag=proof_tag,
        reason=reason,
    )


# ----------------------------------------------------------------- doc utilities


def _is_index(document: Any) -> bool:
    return isinstance(document, Mapping) and "manifests" in document


def _entries(document: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = document.get("manifests")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, Mapping)]


def _idx_has_pin(index: Mapping[str, Any], pinned: str) -> bool:
    return any(e.get("digest") == pinned for e in _entries(index))


def _arm64_entry(index: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    hits = []
    for entry in _entries(index):
        if not entry.get("digest"):
            continue
        platform = entry.get("platform") or {}
        if platform.get("os") != "linux" or platform.get("architecture") != "arm64":
            continue
        hits.append((platform.get("variant"), entry))
    if not hits:
        return None
    for variant, entry in hits:
        if variant in (None, "", "v8"):
            return entry
    return hits[0][1]


def _probe_entry(index: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Best entry to descend into when a pin is an index: prefer linux/amd64."""
    scored: List[Tuple[int, Mapping[str, Any]]] = []
    for entry in _entries(index):
        if not entry.get("digest"):
            continue
        platform = entry.get("platform") or {}
        if platform.get("os") == "unknown":
            continue
        score = 0
        if platform.get("os") == "linux":
            score += 2
        if platform.get("architecture") == "amd64":
            score += 4
        scored.append((score, entry))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    return scored[0][1]


def _manifest(
    base: str, headers: Mapping[str, str], fetch: FetchJson, reference: str
) -> Any:
    document, _ = fetch("%s/manifests/%s" % (base, reference), dict(headers))
    return document


def _config_doc(
    base: str, headers: Mapping[str, str], fetch: FetchJson, manifest: Any
) -> Optional[Mapping[str, Any]]:
    if not isinstance(manifest, Mapping):
        return None
    config = manifest.get("config") or {}
    digest = config.get("digest") if isinstance(config, Mapping) else None
    if not isinstance(digest, str) or not digest:
        return None
    document, _ = fetch("%s/blobs/%s" % (base, digest), dict(headers))
    return document if isinstance(document, Mapping) else None


def _pin_config(
    base: str, headers: Mapping[str, str], fetch: FetchJson, manifest: Any
) -> Optional[Mapping[str, Any]]:
    """Config blob of the pinned release (descending into index entries)."""
    if not isinstance(manifest, Mapping):
        return None
    if not _is_index(manifest):
        return _config_doc(base, headers, fetch, manifest)
    probe: Any = manifest
    for _ in range(4):
        if not _is_index(probe):
            return _config_doc(base, headers, fetch, probe)
        entry = _probe_entry(probe)
        if entry is None:
            return None
        probe = _manifest(base, headers, fetch, str(entry.get("digest")))
    return None


def _release_version(config: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(config, Mapping):
        return None
    section = config.get("config")
    if not isinstance(section, Mapping):
        section = config.get("Config") if isinstance(config.get("Config"), Mapping) else {}
    env: Dict[str, str] = {}
    for item in section.get("Env") or []:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    for name in _VERSION_ENV_NAMES:
        value = env.get(name)
        if value:
            return value.strip()
    labels = section.get("Labels")
    if isinstance(labels, Mapping):
        value = labels.get(_VERSION_LABEL)
        if value:
            return str(value).strip()
    return None


def _candidate_tag(tag: str, version: str) -> Optional[str]:
    """Leading version token replaced by ``version`` (token = text before the
    first "-", only when it starts with a digit or "v")."""
    head, _sep, _rest = tag.partition("-")
    if not head:
        return None
    if not (head[0].isdigit() or head[0] == "v"):
        return None
    return version + tag[len(head):]


# ----------------------------------------------------------------- registry glue


def _client(
    host: str, repo: str, fetch: FetchJson, extra_headers: Optional[Mapping[str, str]]
) -> Tuple[str, Dict[str, str]]:
    api_base = _API_BASES[host] + repo
    headers: Dict[str, str] = {"Accept": MANIFEST_ACCEPT}
    token_endpoint = _TOKEN_ENDPOINTS.get(host)
    if token_endpoint is not None:
        token_document, _ = fetch(token_endpoint.format(repo=repo), dict(headers))
        token = ""
        if isinstance(token_document, Mapping):
            token = token_document.get("token") or token_document.get("access_token") or ""
        if not token:
            raise ValueError("no pull token from %s" % host)
        headers["Authorization"] = "Bearer " + str(token)
    if extra_headers:
        headers.update(dict(extra_headers))
    return api_base, headers


# --------------------------------------------------------------------- resolve


def _resolve_impl(
    ref: str,
    pinned: str,
    fetch: FetchJson,
    fetch_headers: Optional[Mapping[str, str]],
) -> ArchResolution:
    host, repo, tag = parse_ref(ref)
    if host not in _API_BASES:
        return _make(ref, pinned, UNMEASURED, reason="registry not supported")

    base, headers = _client(host, repo, fetch, fetch_headers)

    pin_manifest = _manifest(base, headers, fetch, pinned)
    release_index_without_arm64 = False
    pin_config: Optional[Mapping[str, Any]] = None

    # 1. What the pin itself proves.
    if _is_index(pin_manifest):
        entry = _arm64_entry(pin_manifest)
        if entry is not None and entry.get("digest"):
            return _make(
                ref,
                pinned,
                PASS,
                entry.get("digest"),
                None,
                "pin is a multi-arch index",
            )
        release_index_without_arm64 = True
    else:
        pin_config = _config_doc(base, headers, fetch, pin_manifest)
        if pin_config is not None and str(pin_config.get("architecture")).lower() == "arm64":
            return _make(ref, pinned, PASS, pinned, None, "already arm64")

    # 2. The current tag's release index.
    tag_manifest = _manifest(base, headers, fetch, tag)
    if _is_index(tag_manifest):
        if _idx_has_pin(tag_manifest, pinned):
            entry = _arm64_entry(tag_manifest)
            if entry is not None and entry.get("digest"):
                return _make(
                    ref,
                    pinned,
                    PASS,
                    entry.get("digest"),
                    tag,
                    "pin is in the %s release index; linux/arm64 entry present" % tag,
                )
            return _make(
                ref,
                pinned,
                UNMEASURED,
                reason="no linux/arm64 variant in the release index",
            )
        if tag_manifest == pin_manifest and _arm64_entry(tag_manifest) is None:
            release_index_without_arm64 = True

    # 3. Exact-version tag derived from the pinned release's config.
    if pin_config is None:
        pin_config = _pin_config(base, headers, fetch, pin_manifest)
    version = _release_version(pin_config)
    candidate = _candidate_tag(tag, version) if version else None
    if candidate and candidate != tag:
        candidate_manifest = _manifest(base, headers, fetch, candidate)
        if _is_index(candidate_manifest):
            if _idx_has_pin(candidate_manifest, pinned):
                entry = _arm64_entry(candidate_manifest)
                if entry is not None and entry.get("digest"):
                    return _make(
                        ref,
                        pinned,
                        PASS,
                        entry.get("digest"),
                        candidate,
                        "pin is in the %s release index; linux/arm64 entry present"
                        % candidate,
                    )
                return _make(
                    ref,
                    pinned,
                    UNMEASURED,
                    reason="no linux/arm64 variant in the release index",
                )
            if candidate_manifest == pin_manifest and _arm64_entry(candidate_manifest) is None:
                release_index_without_arm64 = True

    # 4. Release index exists but has no linux/arm64 variant.
    if release_index_without_arm64:
        return _make(
            ref, pinned, UNMEASURED, reason="no linux/arm64 variant in the release index"
        )

    # 5. Nothing proved.
    return _make(
        ref,
        pinned,
        UNMEASURED,
        reason="cannot prove the same release: pin not in the tag index nor the "
        "exact-version tag index",
    )


def resolve_arm64(
    ref: str,
    pinned: str,
    *,
    fetch_json: FetchJson,
    fetch_headers: Optional[Mapping[str, str]] = None,
) -> ArchResolution:
    """Resolve whether ``pinned`` has a proven linux/arm64 sibling release.

    ``fetch_json(url, headers) -> (dict, headers_dict)`` is injectable (default:
    ``default_fetch_json``).  ``fetch_headers`` is merged into every request
    header set.  Never raises: network/HTTP/JSON errors become ``UNMEASURED``
    with the error named.
    """
    fetch: FetchJson = fetch_json if fetch_json is not None else default_fetch_json
    try:
        return _resolve_impl(str(ref), str(pinned), fetch, fetch_headers)
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        return _make(str(ref), str(pinned), UNMEASURED, reason=_err(exc))


# ------------------------------------------------------------------ cache layer


_CACHE_FIELDS = ("ref", "pinned", "status", "arm64_digest", "proof_tag", "reason")


def _load_cache(path: Path) -> Dict[Tuple[str, str], ArchResolution]:
    out: Dict[Tuple[str, str], ArchResolution] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    entries: List[Any] = []
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, Mapping):
        maybe = raw.get("resolutions")
        if isinstance(maybe, list):
            entries = maybe
        else:
            entries = [v for v in raw.values() if isinstance(v, Mapping)]
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if "ref" not in entry or "pinned" not in entry or "status" not in entry:
            continue
        resolution = ArchResolution(
            ref=str(entry["ref"]),
            pinned=str(entry["pinned"]),
            status=str(entry["status"]),
            arm64_digest=entry.get("arm64_digest"),
            proof_tag=entry.get("proof_tag"),
            reason=str(entry.get("reason") or ""),
        )
        out[(resolution.ref, resolution.pinned)] = resolution
    return out


def _write_cache(path: Path, data: Mapping[Tuple[str, str], ArchResolution]) -> None:
    payload = {
        "resolutions": [
            {
                "ref": resolution.ref,
                "pinned": resolution.pinned,
                "status": resolution.status,
                "arm64_digest": resolution.arm64_digest,
                "proof_tag": resolution.proof_tag,
                "reason": resolution.reason,
            }
            for _key, resolution in sorted(data.items(), key=lambda item: item[0])
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, str(path))
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def resolve_all(
    pins: Iterable[Tuple[str, str]],
    *,
    fetch_json: Optional[FetchJson] = None,
    cache_path: Optional[Any] = None,
) -> Dict[Tuple[str, str], ArchResolution]:
    """Deduplicate ``(ref, digest)`` pins, resolve each once, optional JSON
    cache (loaded when present, written atomically after)."""
    fetch: FetchJson = fetch_json if fetch_json is not None else default_fetch_json

    keys: List[Tuple[str, str]] = []
    seen = set()
    for pin in pins:
        key = (str(pin[0]), str(pin[1]))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    cache_file: Optional[Path] = Path(cache_path) if cache_path is not None else None
    cached: Dict[Tuple[str, str], ArchResolution] = {}
    if cache_file is not None:
        cached = _load_cache(cache_file)

    results: Dict[Tuple[str, str], ArchResolution] = {}
    for key in keys:
        if key in cached:
            results[key] = cached[key]
            continue
        results[key] = resolve_arm64(key[0], key[1], fetch_json=fetch)

    if cache_file is not None:
        merged = dict(cached)
        merged.update(results)
        try:
            _write_cache(cache_file, merged)
        except Exception:
            pass
    return results


# ------------------------------------------------------------------- staging


def _first_symlink(root: Path) -> Optional[Path]:
    if root.is_symlink():
        return root
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                return candidate
    return None


def _clean_dest(dest: Path) -> None:
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)


# Directories git requires even when empty. The published Skill2Env corpus dropped empty directories
# (45 of 196 task repos lack refs/ or objects/, so `git rev-parse HEAD` fails "not a git repository").
_GIT_REQUIRED_DIRS = ("refs/heads", "refs/tags", "objects/info", "objects/pack")


def _repair_git_repos(env_dir: Path) -> List[Dict[str, Any]]:
    """Restore what the corpus distribution lost from each git repo in a STAGED environment: git's
    required empty directories, then executable bits recorded in the repo's own index (100755).
    Both are recoverable from the repo itself; nothing else is changed.  Returns one record per repair."""
    repairs: List[Dict[str, Any]] = []
    for git_dir in sorted(env_dir.rglob(".git")):
        if not git_dir.is_dir() or git_dir.is_symlink():
            continue
        repo = git_dir.parent
        rel = repo.relative_to(env_dir).as_posix()
        created = [d for d in _GIT_REQUIRED_DIRS if not (git_dir / d).is_dir()]
        for d in created:
            (git_dir / d).mkdir(parents=True, exist_ok=True)
        if created:
            repairs.append({"repo": rel, "repair": "git_required_dirs", "created": created})
        listing = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(repo), "ls-files", "-s", "-z"],
            capture_output=True, check=False,
        )
        if listing.returncode != 0:
            repairs.append({"repo": rel, "repair": "exec_bits", "error": listing.stderr.decode(errors="replace")[-300:]})
            continue
        restored = []
        for entry in listing.stdout.split(b"\0"):
            if not entry.startswith(b"100755 "):
                continue
            path = repo / entry.split(b"\t", 1)[1].decode()
            if path.is_file() and not path.is_symlink() and not os.access(path, os.X_OK):
                path.chmod(path.stat().st_mode | 0o111)
                restored.append(path.relative_to(repo).as_posix())
        if restored:
            repairs.append({"repo": rel, "repair": "exec_bits", "restored": restored})
    return repairs


def stage_task(
    task_dir: Path,
    stage_root: Path,
    resolutions: Mapping[Tuple[str, str], ArchResolution],
    pins: Mapping[str, str],
) -> Tuple[Optional[Path], str]:
    """Copy ``task_dir`` to ``stage_root/<dirname>`` and rewrite
    ``environment/Dockerfile`` ``FROM`` lines to the proven arm64 digests.

    ``FROM <ref>[@digest]`` becomes ``FROM <ref>@<arm64_digest>`` (``AS name``
    kept, leading ``--platform`` dropped).  ``environment/ARCH_PROVENANCE.json``
    lists ``{"ref","amd64_pin","arm64_digest","proof_tag"}`` per rewrite.  The
    source corpus is never modified.
    """
    task_dir = Path(task_dir)
    stage_root = Path(stage_root)

    if _first_symlink(task_dir) is not None:
        return None, "symlink in task tree"

    dest = stage_root / task_dir.name
    _clean_dest(dest)
    shutil.copytree(task_dir, dest, symlinks=True)
    repairs = _repair_git_repos(dest / "environment") if (dest / "environment").is_dir() else []
    if repairs:
        (dest / "environment" / "STAGING_REPAIRS.json").write_text(json.dumps(repairs, indent=2) + "\n")

    dockerfile = dest / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None, "environment/Dockerfile not found"

    text = dockerfile.read_text(encoding="utf-8")
    stage_names = set()
    provenance: List[Dict[str, Any]] = []
    lines = text.split("\n")

    for index, raw in enumerate(lines):
        match = _FROM_RE.match(raw)
        if match is None:
            continue
        tokens = match.group("body").split()
        cursor = 0
        while cursor < len(tokens) and tokens[cursor].startswith("-"):
            flag = tokens[cursor]
            if flag.lower().startswith("--platform") and "=" not in flag:
                cursor += 2
            else:
                cursor += 1
        if cursor >= len(tokens):
            return None, "cannot parse FROM line: %s" % raw.strip()

        image = tokens[cursor]
        rest = tokens[cursor + 1:]
        spec, separator, digest = image.partition("@")
        if not separator:
            digest = None

        next_stage = None
        if rest and rest[0].lower() == "as" and len(rest) >= 2:
            next_stage = rest[1].lower()

        lowered = spec.lower()
        if lowered == "scratch" or lowered in stage_names:
            if next_stage:
                stage_names.add(next_stage)
            continue

        if next_stage:
            stage_names.add(next_stage)

        pin_digest = digest if digest else pins.get(spec)
        if pin_digest is None:
            return None, "no pinned digest for FROM %s" % spec

        resolution = resolutions.get((spec, pin_digest))
        if resolution is None:
            return None, "no arch resolution for %s@%s" % (spec, pin_digest)
        if resolution.status != PASS:
            return None, resolution.reason or ("%s is %s" % (spec, resolution.status))
        if not resolution.arm64_digest:
            return None, "no arm64 digest for %s@%s" % (spec, pin_digest)

        rebuilt = [match.group("indent"), "FROM", "%s@%s" % (spec, resolution.arm64_digest)]
        rebuilt.extend(rest)
        lines[index] = " ".join(rebuilt)

        provenance.append(
            {
                "ref": spec,
                "amd64_pin": pin_digest,
                "arm64_digest": resolution.arm64_digest,
                "proof_tag": resolution.proof_tag,
            }
        )

    dockerfile.write_text("\n".join(lines), encoding="utf-8")
    (dest / "environment" / "ARCH_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return dest, ""
