"""Unit tests for foundationskills/intake/arch.py.

Contract under test: a provable arm64 sibling of an amd64-pinned base image.
Statuses are PASS/RED/UNMEASURED/REFUSED and an absence of proof is UNMEASURED,
never PASS. All registry answers are served by an in-memory fake through the
injectable `fetch_json(url, headers) -> (body, response_headers)` seam, so the
suite opens no sockets and can never flake on the wire.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path

from foundationskills.intake import arch

ACCEPT_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)

FAKE_TOKEN = "FAKE-REGISTRY-TOKEN"

# API hosts the base URLs are built from (contract: docker.io speaks to
# https://registry-1.docker.io, everything else talks to itself).
API_HOSTS = {"docker.io": "registry-1.docker.io"}

Call = collections.namedtuple("Call", "kind url headers reference")


# --------------------------------------------------------------------------- payloads


def dgst(name: str) -> str:
    """A syntactically valid digest that stays readable in assertion messages."""
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def make_manifest(config_digest, layers=()):
    """Single-platform image manifest: the kind a `FROM ref@digest` pin points at."""
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "digest": config_digest,
            "size": 256,
            "mediaType": "application/vnd.oci.image.config.v1+json",
        },
        "layers": [
            {
                "digest": layer,
                "size": 4096,
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            }
            for layer in layers
        ],
    }


def make_index(*entries):
    """Multi-arch index: each entry is (manifest digest, architecture, os)."""
    manifests = []
    for digest, platform_arch, platform_os, variant in entries:
        platform = {"architecture": platform_arch, "os": platform_os}
        if variant:
            platform["variant"] = variant
        manifests.append(
            {
                "digest": digest,
                "size": 512,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "platform": platform,
            }
        )
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": manifests,
    }


def make_config(*, architecture="amd64", os_name="linux", env=(), labels=None):
    """Image config blob.

    The double exposes architecture / Env / Labels both at the top level and in the
    standard `config` nesting, so the tests pin down the resolution algorithm rather
    than one particular JSON layout.
    """
    env = list(env)
    labels = dict(labels or {})
    return {
        "architecture": architecture,
        "os": os_name,
        "Env": env,
        "Labels": labels,
        "config": {"Env": env, "Labels": labels, "architecture": architecture},
        "rootfs": {"type": "layers", "diff_ids": []},
    }


def http_error(code=404, msg="Not Found"):
    """The HTTPError `urlopen` raises for a non-2xx answer."""
    error = urllib.error.HTTPError(
        "https://registry-1.docker.io/v2/library/python/manifests/pin", code, msg, None, None
    )
    try:
        if not hasattr(error, "reason"):
            error.reason = msg
    except Exception:
        pass
    return error


# --------------------------------------------------------------------------- helpers


def header(headers, name):
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return value
    return None


def status_object(name):
    """The module's own status token for `name`, however it happens to be exported."""
    name = name.upper()
    for holder in (
        arch,
        getattr(arch, "Status", None),
        getattr(arch, "Statuses", None),
        getattr(arch, "State", None),
        getattr(arch, "States", None),
    ):
        if holder is None:
            continue
        for attr in (name, name.title(), name.lower()):
            token = getattr(holder, attr, None)
            if token is not None and not isinstance(token, type) and not callable(token):
                return token
    return name


def status_text(resolution):
    """A resolution's status as plain text, whether it is a string or a status enum."""
    status = resolution.status
    for attr in ("name", "value"):
        text = getattr(status, attr, None)
        if isinstance(text, str):
            return text.upper()
    return str(status).upper().rsplit(".", 1)[-1]


def is_pass(resolution):
    return status_text(resolution) == "PASS"


def is_unmeasured(resolution):
    return status_text(resolution) == "UNMEASURED"


def record(resolution):
    """A resolution reduced to comparable values (status enums and strings match)."""
    return (
        resolution.ref,
        resolution.pinned,
        status_text(resolution),
        resolution.arm64_digest,
        resolution.proof_tag,
        resolution.reason,
    )


def normalize(line):
    return " ".join(line.split()).casefold()


def tree_snapshot(root: Path):
    root = Path(root)
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            out[rel] = ("symlink", os.readlink(path))
        elif path.is_dir():
            out[rel] = ("dir",)
        else:
            out[rel] = ("file", path.read_bytes())
    return out


# --------------------------------------------------------------------------- fake wire


class FakeRegistry:
    """Stand-in for the registry APIs: `fetch_json(url, headers) -> (body, response headers)`.

    Routes are keyed by (api host, repo, kind, reference) so query-string ordering and
    percent-encoding chosen by the module under test cannot make a test flaky. Nothing
    is served over plain HTTP and unknown routes are loud: they raise ValueError (a
    "network/JSON error" under the module's own rules) naming the unexpected request.
    """

    def __init__(self):
        self.manifests = {}
        self.blobs = {}
        self.calls = []

    # -- routing ----------------------------------------------------------
    def add_manifest(self, host, repo, reference, payload):
        self.manifests[(API_HOSTS.get(host, host), repo, reference)] = payload

    def add_blob(self, host, repo, digest, payload):
        self.blobs[(API_HOSTS.get(host, host), repo, digest)] = payload

    @property
    def token_calls(self):
        return [call for call in self.calls if call.kind == "token"]

    @property
    def manifest_calls(self):
        return [call for call in self.calls if call.kind == "manifest"]

    @property
    def blob_calls(self):
        return [call for call in self.calls if call.kind == "blob"]

    @property
    def manifest_refs(self):
        return [call.reference for call in self.manifest_calls]

    # -- the injectable fetch_json ----------------------------------------
    def fetch(self, url, headers=None):
        if not isinstance(url, str):
            url = getattr(url, "full_url", None) or str(url)
        if not url.startswith("https://"):
            raise ValueError("registry access must be a read-only HTTPS GET: " + repr(url))
        sent = dict(headers or {})
        parts = urllib.parse.urlsplit(url)
        segments = [urllib.parse.unquote(s) for s in parts.path.split("/") if s]
        if segments and segments[-1] == "token":
            self.calls.append(Call("token", url, sent, None))
            return {"token": FAKE_TOKEN}, {"Content-Type": "application/json"}

        kind = segments[-2] if len(segments) >= 3 and segments[-2] in ("manifests", "blobs") else None
        if kind is None or segments[0] != "v2":
            raise ValueError("unexpected request: " + url)
        repo = "/".join(segments[1:-2])
        reference = segments[-1]
        table = self.manifests if kind == "manifests" else self.blobs
        self.calls.append(Call(kind[:-1], url, sent, reference))

        key = (parts.netloc, repo, reference)
        if key not in table:
            raise ValueError("unexpected {} request: {} (known: {})".format(kind, url, sorted(table)))
        payload = table[key]
        if isinstance(payload, BaseException):
            raise payload
        response_headers = {"Content-Type": payload.get("mediaType", "application/json")}
        return copy.deepcopy(payload), response_headers


# --------------------------------------------------------------------------- parse_ref


class ParseRefTests(unittest.TestCase):

    MATRIX = [
        ("python:3.12-slim", ("docker.io", "library/python", "3.12-slim")),
        ("ubuntu:24.04", ("docker.io", "library/ubuntu", "24.04")),
        ("redis:7-alpine", ("docker.io", "library/redis", "7-alpine")),
        ("myorg/myimage:1.2.3", ("docker.io", "myorg/myimage", "1.2.3")),
        (
            "docker.io/library/python:3.12-slim",
            ("docker.io", "library/python", "3.12-slim"),
        ),
        (
            "mcr.microsoft.com/playwright:v1.53.1-noble",
            ("mcr.microsoft.com", "playwright", "v1.53.1-noble"),
        ),
        ("ghcr.io/o/r:t", ("ghcr.io", "o/r", "t")),
        (
            "quay.io/prometheus/busybox:latest",
            ("quay.io", "prometheus/busybox", "latest"),
        ),
        (
            "registry.example.com/team/app:1.0",
            ("registry.example.com", "team/app", "1.0"),
        ),
    ]

    def test_parse_ref_matrix(self):
        for raw, expected in self.MATRIX:
            with self.subTest(ref=raw):
                self.assertEqual(tuple(arch.parse_ref(raw)), expected)

    def test_parse_ref_without_tag_is_the_docker_latest_tag(self):
        host, repo, tag = arch.parse_ref("python")
        self.assertEqual((host, repo), ("docker.io", "library/python"))
        # the contract does not bind the implied tag; Docker's own is "latest"
        self.assertIn(tag, ("latest", None, ""))

    def test_parse_ref_without_tag_on_an_explicit_host(self):
        host, repo, tag = arch.parse_ref("ghcr.io/o/r")
        self.assertEqual((host, repo), ("ghcr.io", "o/r"))
        self.assertIn(tag, ("latest", None, ""))


# --------------------------------------------------------------------------- resolve_arm64


class ResolveArm64Tests(unittest.TestCase):
    """One scenario per step of the proof algorithm."""

    def test_pin_is_the_release_multi_arch_index(self):
        """Step 1: the pin already is a multi-arch index that lists linux/arm64."""
        ref = "python:3.12-slim-bookworm"
        pin = dgst("release index")
        arm64_entry = dgst("manifest arm64")
        reg = FakeRegistry()
        reg.add_manifest(
            "docker.io",
            "library/python",
            pin,
            make_index(
                (dgst("manifest amd64"), "amd64", "linux", None),
                (arm64_entry, "arm64", "linux", "v8"),
            ),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertEqual((res.ref, res.pinned), (ref, pin))
        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.arm64_digest, arm64_entry)
        self.assertIsNone(res.proof_tag)
        self.assertIn("multi-arch index", res.reason)
        # the pin is the proof: nothing else is fetched
        self.assertEqual([call.reference for call in reg.manifest_calls], [pin])
        self.assertEqual(reg.blob_calls, [])

        # docker.io: exactly one token handshake and then registry-1.docker.io calls
        self.assertEqual(len(reg.token_calls), 1)
        query = dict(
            urllib.parse.parse_qsl(urllib.parse.urlsplit(reg.token_calls[0].url).query)
        )
        self.assertEqual(query.get("service"), "registry.docker.io")
        self.assertEqual(query.get("scope"), "repository:library/python:pull")
        for call in reg.manifest_calls + reg.blob_calls:
            self.assertEqual(
                header(call.headers, "Authorization"), "Bearer " + FAKE_TOKEN
            )
            accept = header(call.headers, "Accept") or ""
            for media_type in ACCEPT_TYPES:
                self.assertIn(media_type, accept)
            self.assertTrue(
                call.url.startswith("https://registry-1.docker.io/v2/library/python/"),
                call.url,
            )

    def test_pin_is_already_a_single_arm64_manifest(self):
        """Step 1: a single manifest whose config says arm64 is its own sibling."""
        ref = "mcr.microsoft.com/playwright:v1.53.1-noble"
        pin = dgst("playwright arm64 manifest")
        config = dgst("playwright arm64 config")
        reg = FakeRegistry()
        reg.add_manifest(
            "mcr.microsoft.com", "playwright", pin, make_manifest(config, [dgst("layer one")])
        )
        reg.add_blob(
            "mcr.microsoft.com",
            "playwright",
            config,
            make_config(architecture="arm64", env=["NODE_VERSION=22.14.0"]),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.arm64_digest, pin)  # the pin *is* the arm64 image
        # the contract does not bind proof_tag here: the pin is its own proof
        self.assertIn(res.proof_tag, (None, "v1.53.1-noble"))
        self.assertIn("arm64", res.reason.lower())
        # mcr.microsoft.com is anonymous
        self.assertEqual(reg.token_calls, [])
        for call in reg.calls:
            self.assertIsNone(header(call.headers, "Authorization"))
            self.assertTrue(
                call.url.startswith("https://mcr.microsoft.com/v2/playwright/"), call.url
            )

    def test_tag_index_contains_the_pin_and_its_arm64_sibling(self):
        """Step 2: the pin and its linux/arm64 sibling sit in the tag's index."""
        ref = "ghcr.io/o/r:v1.4.2-noble"
        tag = "v1.4.2-noble"
        pin = dgst("ghcr manifest amd64")
        arm64_entry = dgst("ghcr manifest arm64")
        config = dgst("ghcr config")
        reg = FakeRegistry()
        reg.add_manifest("ghcr.io", "o/r", pin, make_manifest(config))
        reg.add_blob(
            "ghcr.io",
            "o/r",
            config,
            make_config(architecture="amd64", env=["PYTHON_VERSION=3.12.13"]),
        )
        reg.add_manifest(
            "ghcr.io",
            "o/r",
            tag,
            make_index(
                (pin, "amd64", "linux", None),
                (arm64_entry, "arm64", "linux", "v8"),
            ),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.arm64_digest, arm64_entry)
        self.assertEqual(res.proof_tag, tag)  # the tag is what proves "same release"
        self.assertIn(pin, reg.manifest_refs)
        self.assertLessEqual(set(reg.manifest_refs), {pin, tag})

        self.assertEqual(len(reg.token_calls), 1)
        self.assertIn("ghcr.io/token", reg.token_calls[0].url)
        query = dict(
            urllib.parse.parse_qsl(urllib.parse.urlsplit(reg.token_calls[0].url).query)
        )
        self.assertEqual(query.get("scope"), "repository:o/r:pull")
        for call in reg.manifest_calls:
            self.assertEqual(
                header(call.headers, "Authorization"), "Bearer " + FAKE_TOKEN
            )
            self.assertTrue(call.url.startswith("https://ghcr.io/v2/o/r/"), call.url)

    def test_exact_version_tag_index_is_the_proof_python(self):
        """Step 3: 3.12-slim-bookworm moved on, 3.12.13-slim-bookworm still ships the pin."""
        ref = "python:3.12-slim-bookworm"
        tag = "3.12-slim-bookworm"
        exact_tag = "3.12.13-slim-bookworm"
        pin = dgst("python 3.12.13 manifest amd64")
        arm64_entry = dgst("python 3.12.13 manifest arm64")
        config = dgst("python 3.12.13 config")
        reg = FakeRegistry()
        reg.add_manifest("docker.io", "library/python", pin, make_manifest(config))
        reg.add_blob(
            "docker.io",
            "library/python",
            config,
            make_config(
                architecture="amd64",
                env=[
                    "PATH=usr/local/bin",
                    "PYTHON_VERSION=3.12.13",
                    "PYTHON_PIP_VERSION=24.0",
                ],
            ),
        )
        # the *current* tag has moved on and no longer lists the pinned manifest ...
        reg.add_manifest(
            "docker.io",
            "library/python",
            tag,
            make_index(
                (dgst("python 3.12.5 manifest amd64"), "amd64", "linux", None),
                (dgst("python 3.12.5 manifest arm64"), "arm64", "linux", "v8"),
            ),
        )
        # ... the exact-version tag still does, arm64 sibling and all
        reg.add_manifest(
            "docker.io",
            "library/python",
            exact_tag,
            make_index((pin, "amd64", "linux", None), (arm64_entry, "arm64", "linux", "v8")),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.arm64_digest, arm64_entry)
        self.assertEqual(res.proof_tag, exact_tag)
        self.assertIn(exact_tag, reg.manifest_refs)
        self.assertLessEqual(set(reg.manifest_refs), {pin, tag, exact_tag})

    def test_version_token_is_only_the_text_before_the_first_hyphen(self):
        """Step 3 with a node style tag: 22-bookworm-slim -> 22.14.0-bookworm-slim."""
        ref = "node:22-bookworm-slim"
        tag = "22-bookworm-slim"
        exact_tag = "22.14.0-bookworm-slim"
        pin = dgst("node 22.14.0 manifest amd64")
        arm64_entry = dgst("node 22.14.0 manifest arm64")
        config = dgst("node 22.14.0 config")
        reg = FakeRegistry()
        reg.add_manifest("docker.io", "library/node", pin, make_manifest(config))
        reg.add_blob(
            "docker.io",
            "library/node",
            config,
            make_config(architecture="amd64", env=["NODE_VERSION=22.14.0"]),
        )
        reg.add_manifest(
            "docker.io",
            "library/node",
            tag,
            make_index((dgst("node 22.2.0 manifest amd64"), "amd64", "linux", None)),
        )
        reg.add_manifest(
            "docker.io",
            "library/node",
            exact_tag,
            make_index((pin, "amd64", "linux", None), (arm64_entry, "arm64", "linux", None)),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.arm64_digest, arm64_entry)
        self.assertEqual(res.proof_tag, exact_tag)
        self.assertIn(exact_tag, reg.manifest_refs)
        self.assertLessEqual(set(reg.manifest_refs), {pin, tag, exact_tag})

    def test_known_env_version_wins_in_the_documented_order(self):
        """PYTHON_VERSION outranks NODE_VERSION regardless of the Env list order."""
        ref = "python:3.12-slim-bookworm"
        tag = "3.12-slim-bookworm"
        exact_tag = "3.12.13-slim-bookworm"
        pin = dgst("python precedence manifest amd64")
        arm64_entry = dgst("python precedence manifest arm64")
        config = dgst("python precedence config")
        reg = FakeRegistry()
        reg.add_manifest("docker.io", "library/python", pin, make_manifest(config))
        reg.add_blob(
            "docker.io",
            "library/python",
            config,
            make_config(
                architecture="amd64",
                env=["NODE_VERSION=22.14.0", "PYTHON_VERSION=3.12.13"],
            ),
        )
        reg.add_manifest(
            "docker.io",
            "library/python",
            tag,
            make_index((dgst("python precedence other"), "amd64", "linux", None)),
        )
        reg.add_manifest(
            "docker.io",
            "library/python",
            exact_tag,
            make_index((pin, "amd64", "linux", None), (arm64_entry, "arm64", "linux", None)),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.proof_tag, exact_tag)
        self.assertLessEqual(set(reg.manifest_refs), {pin, tag, exact_tag})

    def test_opencontainers_version_label_is_the_fallback(self):
        """Step 3 when no known Env version exists: the version label supplies it."""
        ref = "quay.io/apps/tool:8.2-bookworm"
        tag = "8.2-bookworm"
        exact_tag = "8.2.3-bookworm"
        pin = dgst("tool 8.2.3 manifest amd64")
        arm64_entry = dgst("tool 8.2.3 manifest arm64")
        config = dgst("tool 8.2.3 config")
        reg = FakeRegistry()
        reg.add_manifest("quay.io", "apps/tool", pin, make_manifest(config))
        reg.add_blob(
            "quay.io",
            "apps/tool",
            config,
            make_config(
                architecture="amd64",
                env=["PATH=bin", "LANG=C.UTF-8"],
                labels={"org.opencontainers.image.version": "8.2.3"},
            ),
        )
        reg.add_manifest(
            "quay.io",
            "apps/tool",
            tag,
            make_index((dgst("tool 8.2.1 manifest amd64"), "amd64", "linux", None)),
        )
        reg.add_manifest(
            "quay.io",
            "apps/tool",
            exact_tag,
            make_index((pin, "amd64", "linux", None), (arm64_entry, "arm64", "linux", None)),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_pass(res), repr(res))
        self.assertEqual(res.arm64_digest, arm64_entry)
        self.assertEqual(res.proof_tag, exact_tag)

    def test_release_index_without_linux_arm64_is_unmeasured(self):
        """Step 4: an index that lists the pin but has no linux/arm64 variant."""
        ref = "quay.io/apps/tool:2.1-bookworm"
        tag = "2.1-bookworm"
        exact_tag = "2.1.0-bookworm"
        pin = dgst("tool 2.1.0 manifest amd64")
        config = dgst("tool 2.1.0 config")
        release_entries = (
            (pin, "amd64", "linux", None),
            (dgst("tool 2.1.0 manifest amd64 b"), "amd64", "linux", None),
            (dgst("tool 2.1.0 manifest windows"), "amd64", "windows", None),
        )
        reg = FakeRegistry()
        reg.add_manifest("quay.io", "apps/tool", pin, make_manifest(config))
        reg.add_blob(
            "quay.io",
            "apps/tool",
            config,
            make_config(architecture="amd64", env=["GOLANG_VERSION=2.1.0"]),
        )
        reg.add_manifest("quay.io", "apps/tool", tag, make_index(*release_entries))
        reg.add_manifest("quay.io", "apps/tool", exact_tag, make_index(*release_entries))

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_unmeasured(res), repr(res))
        self.assertIsNone(res.arm64_digest)
        self.assertEqual(res.proof_tag, None)
        self.assertIn("no linux/arm64 variant", res.reason)

    def test_pin_that_no_index_lists_is_unmeasured(self):
        """Step 5: neither the tag index nor the exact-version tag index ships the pin."""
        ref = "python:3.12-slim-bookworm"
        tag = "3.12-slim-bookworm"
        exact_tag = "3.12.13-slim-bookworm"
        pin = dgst("python orphan manifest amd64")
        config = dgst("python orphan config")
        reg = FakeRegistry()
        reg.add_manifest("docker.io", "library/python", pin, make_manifest(config))
        reg.add_blob(
            "docker.io",
            "library/python",
            config,
            make_config(architecture="amd64", env=["PYTHON_VERSION=3.12.13"]),
        )
        reg.add_manifest(
            "docker.io",
            "library/python",
            tag,
            make_index((dgst("python current amd64"), "amd64", "linux", None)),
        )
        reg.add_manifest(
            "docker.io",
            "library/python",
            exact_tag,
            make_index(
                (dgst("python 3.12.13 amd64"), "amd64", "linux", None),
                (dgst("python 3.12.13 arm64"), "arm64", "linux", "v8"),
            ),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_unmeasured(res), repr(res))
        self.assertIsNone(res.arm64_digest)
        self.assertIn("cannot prove the same release", res.reason)
        self.assertLessEqual(set(reg.manifest_refs), {pin, tag, exact_tag})

    def test_a_tag_without_a_version_token_is_not_fabricated(self):
        """Step 5: "slim-bullseye" has no version token, so no candidate tag is invented."""
        ref = "quay.io/apps/tool:slim-bullseye"
        tag = "slim-bullseye"
        pin = dgst("tool slim manifest amd64")
        config = dgst("tool slim config")
        reg = FakeRegistry()
        reg.add_manifest("quay.io", "apps/tool", pin, make_manifest(config))
        reg.add_blob(
            "quay.io",
            "apps/tool",
            config,
            make_config(architecture="amd64", env=["GOLANG_VERSION=2.1.0"]),
        )
        reg.add_manifest(
            "quay.io",
            "apps/tool",
            tag,
            make_index((dgst("tool slim sibling"), "amd64", "linux", None)),
        )

        res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)

        self.assertTrue(is_unmeasured(res), repr(res))
        self.assertIn("cannot prove the same release", res.reason)
        self.assertLessEqual(set(reg.manifest_refs), {pin, tag})

    def test_unsupported_registry_is_unmeasured_and_stays_offline(self):
        reg = FakeRegistry()
        cases = [
            ("registry.example.com/team/app:1.0", dgst("other host")),
            ("gcr.io/foo/bar:1", dgst("gcr host")),
        ]
        for ref, pin in cases:
            with self.subTest(ref=ref):
                res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)
                self.assertTrue(is_unmeasured(res), repr(res))
                self.assertIsNone(res.arm64_digest)
                self.assertIn("not supported", res.reason.lower())
        self.assertEqual(reg.calls, [])

    def test_network_and_json_failures_are_unmeasured_and_name_the_error(self):
        rows = [
            ("http-404", http_error(), ("404", "not found", "http", "urlopen error")),
            (
                "dns",
                urllib.error.URLError("no route to host"),
                ("no route to host", "urlopen error", "urlerror"),
            ),
            ("timeout", TimeoutError("timed out"), ("timed out", "timeout")),
            (
                "html-instead-of-json",
                json.JSONDecodeError("Expecting value", "<html>", 0),
                ("json", "expecting value", "decode"),
            ),
        ]
        for name, failure, needles in rows:
            with self.subTest(case=name):
                ref = "python:3.12-slim-bookworm"
                pin = dgst("failing pin " + name)
                reg = FakeRegistry()
                reg.add_manifest("docker.io", "library/python", pin, failure)

                res = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)  # never raises

                self.assertTrue(is_unmeasured(res), repr(res))
                self.assertIsNone(res.arm64_digest)
                low = res.reason.lower()
                self.assertTrue(
                    any(needle in low for needle in needles),
                    "reason {} names none of {}".format(res.reason, needles),
                )


# --------------------------------------------------------------------------- resolve_all


class ResolveAllTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    @staticmethod
    def pairs():
        return [
            ("python:3.12-slim-bookworm", dgst("all py pin")),
            ("node:22-bookworm-slim", dgst("all node pin")),
        ]

    @staticmethod
    def serve(reg):
        """Minimal pass chain: each pin is a multi-arch index that lists linux/arm64."""
        reg.add_manifest(
            "docker.io",
            "library/python",
            dgst("all py pin"),
            make_index((dgst("all py arm64"), "arm64", "linux", "v8")),
        )
        reg.add_manifest(
            "docker.io",
            "library/node",
            dgst("all node pin"),
            make_index((dgst("all node arm64"), "arm64", "linux", "v8")),
        )
        return reg

    def test_resolve_all_dedupes_and_resolves_each_pair_once(self):
        reg = self.serve(FakeRegistry())
        duplicated = self.pairs() + [self.pairs()[0], self.pairs()[0]]

        result = arch.resolve_all(duplicated, fetch_json=reg.fetch)

        self.assertEqual(set(result), set(self.pairs()))
        for key, resolution in result.items():
            self.assertEqual((resolution.ref, resolution.pinned), key)
            self.assertTrue(is_pass(resolution), repr(resolution))

        control_reg = self.serve(FakeRegistry())
        arch.resolve_all(self.pairs(), fetch_json=control_reg.fetch)
        self.assertEqual(
            [call.url for call in reg.calls], [call.url for call in control_reg.calls]
        )

    def test_resolve_all_cache_round_trip_does_not_touch_the_network(self):
        cache_path = self.tmp / "cache" / "arch_cache.json"
        cache_path.parent.mkdir()
        reg = self.serve(FakeRegistry())

        first = arch.resolve_all(self.pairs(), fetch_json=reg.fetch, cache_path=cache_path)

        self.assertEqual(set(first), set(self.pairs()))
        self.assertTrue(cache_path.is_file())
        self.assertTrue(json.loads(cache_path.read_text(encoding="utf-8")))

        def no_network(url, headers=None):
            raise AssertionError("the cache should have answered without " + repr(url))

        second = arch.resolve_all(self.pairs(), fetch_json=no_network, cache_path=cache_path)

        self.assertEqual(set(second), set(first))
        for key in first:
            self.assertEqual(record(first[key]), record(second[key]), key)


# --------------------------------------------------------------------------- stage_task


class StageTaskTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.stage_root = self.tmp / "stage"
        self.stage_root.mkdir()

    def make_task(self, dockerfile, *, name="task-alpha"):
        task_dir = self.tmp / name
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        (task_dir / "environment" / "requirements.txt").write_text(
            "requests==2.32.3\n", encoding="utf-8"
        )
        (task_dir / "task.yaml").write_text("title: sample\n", encoding="utf-8")
        return task_dir

    @staticmethod
    def resolve_pair(reg, ref, repo, pin, arm64_entry):
        reg.add_manifest(
            "docker.io", repo, pin, make_index((arm64_entry, "arm64", "linux", "v8"))
        )
        resolution = arch.resolve_arm64(ref, pin, fetch_json=reg.fetch)
        return resolution

    def test_stage_task_rewrites_from_lines_and_stamps_provenance(self):
        py_ref = "python:3.12-slim-bookworm"
        node_ref = "node:22-bookworm-slim"
        py_pin, node_pin = dgst("stage py pin"), dgst("stage node pin")
        py_arm, node_arm = dgst("stage py arm64"), dgst("stage node arm64")

        reg = FakeRegistry()
        py_res = self.resolve_pair(reg, py_ref, "library/python", py_pin, py_arm)
        node_res = self.resolve_pair(reg, node_ref, "library/node", node_pin, node_arm)
        self.assertTrue(is_pass(py_res) and is_pass(node_res), repr((py_res, node_res)))

        pins = {py_ref: py_pin, node_ref: node_pin}
        resolutions = {(py_ref, py_pin): py_res, (node_ref, node_pin): node_res}

        dockerfile = "\n".join(
            [
                "# base image",
                "FROM --platform=linux/amd64 " + py_ref + "@" + py_pin + " AS base",
                "RUN true",
                "# FROM " + node_ref + "@" + node_pin + " AS decoy",
                "FROM " + node_ref + "@" + node_pin + " AS build",
                'ENTRYPOINT ["/bin/sh"]',
            ]
        ) + "\n"
        task_dir = self.make_task(dockerfile)
        before = tree_snapshot(task_dir)

        path, msg = arch.stage_task(task_dir, self.stage_root, resolutions, pins)

        self.assertIsNotNone(path, msg)
        self.assertIsInstance(msg, str)
        self.assertEqual(Path(path), self.stage_root / "task-alpha")
        self.assertTrue(Path(path).is_dir())

        staged = (Path(path) / "environment" / "Dockerfile").read_text(encoding="utf-8")
        from_lines = [
            line for line in staged.splitlines() if normalize(line).startswith("from")
        ]
        self.assertEqual(len(from_lines), 2, staged)
        self.assertEqual(
            {normalize(line) for line in from_lines},
            {
                normalize("FROM {}@{} AS base".format(py_ref, py_arm)),
                normalize("FROM {}@{} AS build".format(node_ref, node_arm)),
            },
        )
        self.assertNotIn("--platform", staged)
        self.assertIn("RUN true", staged.splitlines())
        self.assertIn('ENTRYPOINT ["/bin/sh"]', staged.splitlines())
        # a comment is not a FROM line: it is copied verbatim, amd64 pin and all
        self.assertIn(
            "# FROM " + node_ref + "@" + node_pin + " AS decoy", staged.splitlines()
        )
        # the rest of the task tree travels along
        self.assertEqual(
            (Path(path) / "task.yaml").read_text(encoding="utf-8"), "title: sample\n"
        )
        self.assertEqual(
            (Path(path) / "environment" / "requirements.txt").read_text(encoding="utf-8"),
            "requests==2.32.3\n",
        )

        provenance_path = Path(path) / "environment" / "ARCH_PROVENANCE.json"
        self.assertTrue(provenance_path.is_file())
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        self.assertIsInstance(provenance, list)
        self.assertEqual(len(provenance), 2, provenance)
        for row in provenance:
            self.assertTrue(
                {"ref", "amd64_pin", "arm64_digest", "proof_tag"} <= set(row), row
            )
        rows = {row["ref"]: row for row in provenance}
        self.assertEqual(set(rows), {py_ref, node_ref})
        self.assertEqual(rows[py_ref]["amd64_pin"], py_pin)
        self.assertEqual(rows[py_ref]["arm64_digest"], py_arm)
        self.assertEqual(rows[py_ref]["proof_tag"], py_res.proof_tag)
        self.assertEqual(rows[node_ref]["amd64_pin"], node_pin)
        self.assertEqual(rows[node_ref]["arm64_digest"], node_arm)
        self.assertEqual(rows[node_ref]["proof_tag"], node_res.proof_tag)

        # the source corpus is never touched
        self.assertEqual(tree_snapshot(task_dir), before)
        self.assertFalse((task_dir / "environment" / "ARCH_PROVENANCE.json").exists())

    def test_stage_task_pins_from_lines_without_a_digest(self):
        ref = "python:3.12-slim"
        pin, arm64_entry = dgst("bare py pin"), dgst("bare py arm64")
        reg = FakeRegistry()
        resolution = self.resolve_pair(reg, ref, "library/python", pin, arm64_entry)
        self.assertTrue(is_pass(resolution), repr(resolution))
        pins = {ref: pin}
        resolutions = {(ref, pin): resolution}

        task_dir = self.make_task("FROM {}\nRUN true\n".format(ref))
        before = tree_snapshot(task_dir)

        path, msg = arch.stage_task(task_dir, self.stage_root, resolutions, pins)

        self.assertIsNotNone(path, msg)
        staged = (Path(path) / "environment" / "Dockerfile").read_text(encoding="utf-8")
        from_lines = [
            line for line in staged.splitlines() if normalize(line).startswith("from")
        ]
        self.assertEqual(
            [normalize(line) for line in from_lines],
            [normalize("FROM {}@{}".format(ref, arm64_entry))],
        )
        rows = json.loads(
            (Path(path) / "environment" / "ARCH_PROVENANCE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [(row["ref"], row["amd64_pin"], row["arm64_digest"], row["proof_tag"]) for row in rows],
            [(ref, pin, arm64_entry, resolution.proof_tag)],
        )
        self.assertEqual(tree_snapshot(task_dir), before)

    def test_stage_task_refuses_tasks_with_a_symlink_in_the_tree(self):
        ref = "python:3.12-slim-bookworm"
        pin, arm64_entry = dgst("link py pin"), dgst("link py arm64")
        reg = FakeRegistry()
        resolution = self.resolve_pair(reg, ref, "library/python", pin, arm64_entry)
        pins = {ref: pin}
        resolutions = {(ref, pin): resolution}

        task_dir = self.make_task("FROM {}@{} AS base\n".format(ref, pin))
        link = task_dir / "environment" / "sitecustomize.py"
        try:
            os.symlink(task_dir / "environment" / "requirements.txt", link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available here")
        link = task_dir / "task-extra"
        try:
            os.symlink(task_dir / "environment", link)
        except (OSError, NotImplementedError):
            pass
        before = tree_snapshot(task_dir)

        path, msg = arch.stage_task(task_dir, self.stage_root, resolutions, pins)

        self.assertIsNone(path, msg)
        self.assertIn("symlink", msg.lower())
        self.assertEqual(tree_snapshot(task_dir), before)
        self.assertFalse(
            list(self.stage_root.rglob("ARCH_PROVENANCE.json")),
            "a refused task must not be proven",
        )

    def test_stage_task_refuses_when_a_from_line_is_not_proven(self):
        ref = "python:3.12-slim-bookworm"
        pin = dgst("unproven py pin")
        reason = (
            "cannot prove the same release: pin not in the tag index "
            "nor the exact-version tag index"
        )
        pending = arch.ArchResolution(
            ref=ref,
            pinned=pin,
            status=status_object("UNMEASURED"),
            arm64_digest=None,
            proof_tag=None,
            reason=reason,
        )
        pins = {ref: pin}
        resolutions = {(ref, pin): pending}

        task_dir = self.make_task("FROM {}@{} AS base\nRUN true\n".format(ref, pin))
        before = tree_snapshot(task_dir)

        path, msg = arch.stage_task(task_dir, self.stage_root, resolutions, pins)

        self.assertIsNone(path, msg)
        self.assertIsInstance(msg, str)
        self.assertTrue(msg)
        self.assertTrue(
            ("cannot prove" in msg.lower())
            or ("unmeasured" in msg.lower())
            or (ref in msg),
            msg,
        )
        self.assertEqual(tree_snapshot(task_dir), before)
        self.assertFalse(list(self.stage_root.rglob("ARCH_PROVENANCE.json")))


if __name__ == "__main__":
    unittest.main()


def test_repair_git_repos_restores_what_the_corpus_lost(tmp_path):
    """The published corpus dropped empty dirs (.git/refs) and exec bits; both come back from the repo itself."""
    import os
    import shutil
    import subprocess

    from foundationskills.intake.arch import _repair_git_repos

    repo = tmp_path / "environment" / "repo"
    repo.mkdir(parents=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "safe.directory=*", "-C", str(repo)]
    subprocess.run(git + ["init", "-q", "-b", "main"], check=True)
    script = repo / "verify.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    subprocess.run(git + ["add", "verify.sh"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "seed"], check=True)
    subprocess.run(git + ["pack-refs", "--all"], check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    # what the distribution did: empty dirs gone, exec bit gone
    shutil.rmtree(repo / ".git" / "refs")
    script.chmod(0o644)
    assert subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True).returncode != 0

    repairs = _repair_git_repos(tmp_path / "environment")

    assert subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == head
    assert os.access(script, os.X_OK)
    assert subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True).stdout == ""
    kinds = {r["repair"] for r in repairs}
    assert kinds == {"git_required_dirs", "exec_bits"}
    assert any("verify.sh" in r.get("restored", []) for r in repairs)


def test_repair_git_repos_leaves_intact_repos_alone(tmp_path):
    import subprocess

    from foundationskills.intake.arch import _repair_git_repos

    repo = tmp_path / "environment" / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    for d in ("refs/heads", "refs/tags", "objects/info", "objects/pack"):
        (repo / ".git" / d).mkdir(parents=True, exist_ok=True)
    assert _repair_git_repos(tmp_path / "environment") == []
