"""LLM inference backends and deterministic request helpers for Data Engine ops.

This module intentionally keeps the backend contract small. Ops receive a backend
with ``kind`` and ``model`` attributes and one chat-completion method; all
configuration failures raise ``LLMOpError`` before a request is attempted.

Built-in protocol config::

    {
        "kind": "openai_compatible",
        "base_url": "http://host:8000",       # required; root or .../v1 URL
        "model": "model-name",               # required
        "api_key_env": "SERVICE_TOKEN",      # optional env var NAME, not a token
        "timeout_s": 300.0,                   # per-request timeout, default 300
        "max_retries": 2,                     # retries, therefore 3 total attempts
        "extra_body": {"custom": true},       # merged into the JSON request body
        "cache_dir": "/path/to/inference-cache" # optional exact-response cache
    }

The OpenAI-compatible payload contains exactly ``model``, ``messages``,
``temperature``, ``max_tokens`` and ``seed``, plus ``response_format`` when JSON
mode is requested. Keys from ``extra_body`` are then merged; an explicit
``json_mode`` still installs ``{"type": "json_object"}`` last.

Documented simplifications and failure policy:

* Retryable failures are ``URLError``, timeout failures, HTTP 429 and HTTP
  5xx. ``max_retries=2`` means attempts 0, 1 and 2, sleeping ``2**attempt``
  seconds between retryable attempts through the injectable ``sleep`` callable.
* Only the first choice is read. Non-string content or finish values are
  exposed as ``None`` rather than guessed. Absent, malformed or negative usage
  token values are reported as zero.
* Per-request transport, timeout, HTTP and JSON-decoding failures after policy
  are returned as ``LLMResponse(content=None, error=...)``. ``complete()`` does
  not raise them, because an op must count that item as unmeasured instead of
  manufacturing a result. Configuration problems still raise ``LLMOpError``.
* HTTP redirects, proxy configuration, TLS handling and connection pooling are
  urllib defaults. This module does not implement its own redirects or transport
  authentication. ``timeout_s`` is a per-request timeout, not a retry-loop budget.
* The cache key is SHA-256 over the specified JSON object containing model,
  messages, temperature, max_tokens, seed and json_mode. Only successful
  responses are stored. Each successful response is one JSON object at
  ``cache_dir/<key[:2]>/<key>.json`` and is installed by an atomic
  temporary-file write plus ``os.replace``.
* Cache reads are best effort: unreadable or malformed cache entries are misses
  and persistence failures return the live response uncached. Non-JSON
  message parameters bypass caching because the specified key cannot be made.
* Atomicity here is the filesystem rename guarantee, not an fsync/durability
  guarantee. A power failure can lose a cached response, but readers never get
  a partially replaced JSON file from a normal completed store.
* ``map_ordered`` holds at most ``window`` futures. Invalid window values are
  clamped to one. Worker exceptions propagate when their ordered result is
  consumed; they are not converted into sentinel values. A size-one worker pool
  executes directly in the consuming iterator.
* ``parse_json_object`` understands an explicit fenced ``json`` block and
  leading prose. It returns a dict only; prose, arrays, malformed objects and
  absent JSON produce ``None`` and never raise. The extractor balances braces
  outside strings but is not a full natural-language parser.
* ``prompt_hash("a", "b")`` hashes the directly concatenated parts ``"ab"``.
  Callers that need field boundaries should include a separator in the parts.

No backend named ``none`` is registered deliberately: there is no safe
heuristic inference fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, TypeVar


class LLMOpError(ValueError):
    """An LLM op refused or failed; the message names the missing input."""


@dataclass(frozen=True)
class LLMResponse:
    """One completion result, with measured usage or a short request error."""

    content: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool
    error: str | None


class LLMBackend(Protocol):
    """A deterministic chat-completion endpoint."""

    kind: str
    model: str

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Complete one chat request without raising a per-request failure."""


_T = TypeVar('_T')
_R = TypeVar('_R')
BACKENDS: dict[str, Callable[[dict], LLMBackend]] = {}
_JSON_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.IGNORECASE | re.DOTALL)


def _failure(reason: str) -> LLMResponse:
    text = ' '.join(str(reason).split())
    return LLMResponse(
        content=None,
        finish_reason=None,
        prompt_tokens=0,
        completion_tokens=0,
        cache_hit=False,
        error=text[:160] or 'request failed',
    )


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return max(int(value), 0)
    except (OverflowError, ValueError):
        return 0


def _known_kinds() -> str:
    return ', '.join(sorted(BACKENDS))


def _missing_backend_error(op_name: str) -> LLMOpError:
    return LLMOpError(
        f'{op_name}: missing input: config.backend - LLM ops need an inference backend '
        f'(kinds: {_known_kinds()}); there is no heuristic fallback'
    )


def register_backend(kind: str, factory: Callable[[dict], LLMBackend]) -> None:
    """Register one backend factory; re-registering a kind replaces it."""

    if not isinstance(kind, str) or not kind:
        raise ValueError('backend kind must be a non-empty string')
    if not callable(factory):
        raise ValueError('backend factory must be callable')
    BACKENDS[kind] = factory


def make_backend(cfg: dict | None, op_name: str) -> LLMBackend:
    """Build a validated backend from an op's optional ``config.backend`` object."""

    if cfg is None:
        raise _missing_backend_error(op_name)
    if not isinstance(cfg, Mapping):
        raise LLMOpError(
            f'{op_name}: missing input: config.backend - expected a backend object '
            f'(kinds: {_known_kinds()})'
        )
    registered = dict(cfg)
    kind = registered.get('kind')
    if kind is None or kind == '':
        raise _missing_backend_error(op_name)
    if not isinstance(kind, str):
        raise LLMOpError(
            f'{op_name}: missing input: config.backend.kind - backend kind must be a '
            f'string (kinds: {_known_kinds()})'
        )
    if kind == 'none':
        raise _missing_backend_error(op_name)
    factory = BACKENDS.get(kind)
    if factory is None:
        raise LLMOpError(
            f'{op_name}: unknown LLM backend kind {kind!r}; known kinds: {_known_kinds()}'
        )
    return factory(registered)


class OpenAICompatibleBackend:
    """Small urllib client for an OpenAI-compatible chat-completions endpoint."""

    kind = 'openai_compatible'

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        if not isinstance(cfg, Mapping):
            raise LLMOpError(
                'openai_compatible: missing input: config.backend - expected an object'
            )
        raw_url = cfg.get('base_url')
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise LLMOpError(
                'openai_compatible: missing input: config.backend.base_url '
                '(an http(s) service root or /v1 URL)'
            )
        raw_model = cfg.get('model')
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise LLMOpError(
                'openai_compatible: missing input: config.backend.model (a served model id)'
            )
        env_name = cfg.get('api_key_env')
        if env_name is not None and (not isinstance(env_name, str) or not env_name):
            raise LLMOpError(
                'openai_compatible: missing input: config.backend.api_key_env '
                '(an environment variable name)'
            )
        raw_extra = cfg.get('extra_body', {})
        if raw_extra is None:
            extra_body: dict[str, Any] = {}
        elif isinstance(raw_extra, Mapping):
            extra_body = dict(raw_extra)
        else:
            raise LLMOpError(
                'openai_compatible: missing input: config.backend.extra_body '
                '(a JSON object whose keys merge into the request)'
            )
        raw_timeout = cfg.get('timeout_s', 300.0)
        if (
            isinstance(raw_timeout, bool)
            or not isinstance(raw_timeout, (int, float))
            or float(raw_timeout) <= 0
        ):
            raise LLMOpError(
                'openai_compatible: invalid input: config.backend.timeout_s '
                '(a positive number of seconds)'
            )
        raw_retries = cfg.get('max_retries', 2)
        if (
            isinstance(raw_retries, bool)
            or not isinstance(raw_retries, int)
            or raw_retries < 0
        ):
            raise LLMOpError(
                'openai_compatible: invalid input: config.backend.max_retries '
                '(a non-negative integer)'
            )

        self.base_url = raw_url.strip()
        self.endpoint = self._chat_endpoint(self.base_url)
        self.model = raw_model.strip()
        self.api_key_env = None if env_name is None else env_name.strip()
        self.timeout_s = float(raw_timeout)
        self.max_retries = raw_retries
        self.extra_body = extra_body
        self.opener: Callable[..., Any] = urllib.request.urlopen
        self.sleep: Callable[[float], Any] = time.sleep
        self._bearer: str | None = None
        if self.api_key_env is not None:
            token = os.environ.get(self.api_key_env)
            if token is None:
                raise LLMOpError(
                    f'openai_compatible: missing input: env var {self.api_key_env}'
                )
            self._bearer = token

    @staticmethod
    def _chat_endpoint(base_url: str) -> str:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise LLMOpError(
                'openai_compatible: invalid input: config.backend.base_url '
                '(must be an http(s) URL naming a host)'
            )
        if parsed.query or parsed.fragment:
            raise LLMOpError(
                'openai_compatible: invalid input: config.backend.base_url '
                '(queries and fragments are not part of a service root)'
            )
        root_path = parsed.path.rstrip('/')
        if root_path == '/v1':
            root_path = ''
        elif root_path.endswith('/v1'):
            root_path = root_path[:-3]
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f'{root_path}/v1/chat/completions',
                '',
                '',
            )
        )

    def _request_body(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        json_mode: bool,
    ) -> bytes:
        body: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'seed': seed,
        }
        body.update(self.extra_body)
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        return json.dumps(body).encode('utf-8')

    def _request(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        json_mode: bool,
    ) -> urllib.request.Request:
        payload = self._request_body(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            json_mode=json_mode,
        )
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            method='POST',
        )
        if self._bearer is not None:
            request.add_header('Authorization', f'Bearer {self._bearer}')
        return request

    @staticmethod
    def _parse_body(payload: bytes) -> LLMResponse:
        body = json.loads(payload.decode('utf-8', errors='replace'))
        if not isinstance(body, Mapping):
            raise ValueError('response is not a JSON object')
        choices = body.get('choices')
        if not isinstance(choices, list) or not choices:
            raise ValueError('response has no choices')
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError('first response choice is not an object')
        message = first.get('message')
        if message is not None and not isinstance(message, Mapping):
            raise ValueError('first response message is not an object')
        usage = body.get('usage')
        if not isinstance(usage, Mapping):
            usage = {}
        if isinstance(message, Mapping):
            content = _optional_text(message.get('content'))
        else:
            content = None
        return LLMResponse(
            content=content,
            finish_reason=_optional_text(first.get('finish_reason')),
            prompt_tokens=_token_count(usage.get('prompt_tokens')),
            completion_tokens=_token_count(usage.get('completion_tokens')),
            cache_hit=False,
            error=None,
        )

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Complete a chat request; request failures are returned, never guessed."""

        try:
            request = self._request(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                json_mode=json_mode,
            )
        except (TypeError, ValueError) as exc:
            return _failure(f'request encode: {exc}')

        for attempt in range(self.max_retries + 1):
            failure: str
            retryable: bool
            try:
                response = self.opener(request, timeout=self.timeout_s)
                try:
                    payload = response.read()
                finally:
                    close = getattr(response, 'close', None)
                    if callable(close):
                        close()
                return self._parse_body(payload)
            except urllib.error.HTTPError as exc:
                failure = f'http {exc.code}'
                retryable = exc.code == 429 or 500 <= exc.code <= 599
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = getattr(exc, 'reason', exc)
                failure = f'network: {reason}'
                retryable = True
            except (json.JSONDecodeError, ValueError) as exc:
                failure = f'response JSON: {exc}'
                retryable = False
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
                retryable = False
            if retryable and attempt < self.max_retries:
                self.sleep(float(2**attempt))
                continue
            return _failure(failure)
        return _failure('request retry loop ended unexpectedly')


class CachedBackend:
    """Exact-response disk cache wrapping any ``LLMBackend`` implementation."""

    def __init__(self, inner: LLMBackend, cache_dir: str | None) -> None:
        self.inner = inner
        self.kind = inner.kind
        self.model = inner.model
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

    def _key(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
        json_mode: bool,
    ) -> str | None:
        material = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'seed': seed,
            'json_mode': bool(json_mode),
            # merged into the request body, so it changes the answer (e.g. thinking on/off)
            'extra_body': getattr(self.inner, 'extra_body', None) or {},
        }
        try:
            encoded = json.dumps(material, sort_keys=True).encode('utf-8')
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(encoded).hexdigest()

    def _path(self, key: str) -> Path:
        if self.cache_dir is None:
            raise ValueError('cache path requested with cache disabled')
        return self.cache_dir / key[:2] / f'{key}.json'

    @staticmethod
    def _response_from_json(value: Any) -> LLMResponse | None:
        if not isinstance(value, Mapping):
            return None
        return LLMResponse(
            content=_optional_text(value.get('content')),
            finish_reason=_optional_text(value.get('finish_reason')),
            prompt_tokens=_token_count(value.get('prompt_tokens')),
            completion_tokens=_token_count(value.get('completion_tokens')),
            cache_hit=True,
            error=None,
        )

    def _read(self, key: str) -> LLMResponse | None:
        if self.cache_dir is None:
            return None
        try:
            payload = json.loads(self._path(key).read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return self._response_from_json(payload)

    def _write(self, key: str, response: LLMResponse) -> None:
        if self.cache_dir is None:
            return
        path = self._path(key)
        temporary: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=path.parent,
                prefix=f'.{key}.',
                suffix='.tmp',
                delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(asdict(response), handle, sort_keys=True)
            os.replace(temporary, path)
            temporary = None
        except OSError:
            return
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a cached successful response or perform and store a live call."""

        key = self._key(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            json_mode=json_mode,
        )
        if key is not None:
            cached = self._read(key)
            if cached is not None:
                return cached
        response = self.inner.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            json_mode=json_mode,
        )
        if key is not None and response.error is None:
            self._write(key, response)
        return response


def map_ordered(
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    workers: int,
    window: int = 64,
) -> Iterator[_R]:
    """Map with a bounded in-flight window while preserving input order."""

    worker_count = int(workers)
    if worker_count <= 1:
        for item in items:
            yield fn(item)
        return
    limit = max(int(window), 1)
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        pending: deque[Any] = deque()

        def submit_available() -> None:
            while len(pending) < limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                pending.append(pool.submit(fn, item))

        submit_available()
        while pending:
            future = pending.popleft()
            yield future.result()
            submit_available()


def prompt_hash(*parts: str) -> str:
    """Return the stable 16-hex-character prompt identifier used in provenance."""

    return hashlib.sha256(''.join(parts).encode('utf-8')).hexdigest()[:16]


def _balanced_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def parse_json_object(text: str | None) -> dict | None:
    """Extract the first parseable JSON object, tolerantly and without raising."""

    if not isinstance(text, str):
        return None
    try:
        fenced = _JSON_FENCE.search(text)
        candidate = fenced.group(1) if fenced is not None else text
        start = 0
        while True:
            position = candidate.find('{', start)
            if position < 0:
                return None
            end = _balanced_object_end(candidate, position)
            if end is None:
                return None
            try:
                parsed = json.loads(candidate[position:end])
            except json.JSONDecodeError:
                start = end
                continue
            return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class CallBudget:
    """Thread-safe accountant limiting the number of successful LLM initiations."""

    def __init__(self, max_calls: int | None) -> None:
        if max_calls is not None:
            if isinstance(max_calls, bool) or not isinstance(max_calls, int):
                raise LLMOpError('call budget: max_calls must be an integer or None')
            if max_calls < 0:
                raise LLMOpError('call budget: max_calls must be non-negative')
        self.max_calls = max_calls
        self._used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Consume one call, returning false exactly once the budget is exhausted."""

        with self._lock:
            if self.max_calls is not None and self._used >= self.max_calls:
                return False
            self._used += 1
            return True

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.max_calls is not None and self._used >= self.max_calls

    @property
    def used(self) -> int:
        with self._lock:
            return self._used


def budget(max_calls: int | None) -> CallBudget:
    """Build a thread-safe call budget; ``None`` means unlimited calls."""

    return CallBudget(max_calls)


def _openai_compatible_factory(cfg: dict) -> LLMBackend:
    backend = OpenAICompatibleBackend(cfg)
    cache_dir = cfg.get('cache_dir')
    if cache_dir is None:
        return backend
    if not isinstance(cache_dir, (str, os.PathLike)) or not str(cache_dir):
        raise LLMOpError(
            'openai_compatible: invalid input: config.backend.cache_dir '
            '(a filesystem directory path or null)'
        )
    return CachedBackend(backend, str(cache_dir))


register_backend('openai_compatible', _openai_compatible_factory)
