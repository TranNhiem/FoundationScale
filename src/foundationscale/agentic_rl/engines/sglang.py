"""``SGLangClient``: a token-in/token-out ``GenerationClient`` over SGLang's
native HTTP server, plus the control calls one rollout host needs for
colocated weight sync.

ASSUMPTIONS (stated here because no SGLang install exists in this environment
to probe -- the ``sglang`` package itself is never imported by this module or
anything in this plane; a cluster probe must confirm the installed server
version honours them before this client is pointed at a real engine):

  * The native (non-OpenAI-compatible) ``/generate`` endpoint is used, with
    request body ``{"input_ids": [...], "sampling_params": {...},
    "return_logprob": true}`` and a JSON response shaped
    ``{"text": <str>, "output_ids": [<int>, ...],
    "meta_info": {"output_token_logprobs": [[<logprob>, <token_id>, ...], ...],
    "finish_reason": {"type": "stop" | "length" | "abort"}}}``.
  * ``sampling_params`` field names are exactly ``temperature``, ``top_p``,
    ``top_k``, ``min_p``, ``presence_penalty``, ``repetition_penalty``,
    ``max_new_tokens`` -- SGLang's own native spelling, not an OpenAI alias.
  * ``top_k`` has no "disabled" value of its own in
    ``harness.base.SamplingParams`` (it only requires ``>= 0``); this client
    treats FoundationScale's ``top_k == 0`` as "no top-k filtering" and sends
    SGLang's native disablement sentinel ``-1`` for it, never a bare ``0``
    (which a sampler could read as "keep zero candidates" instead of "keep
    every candidate").
  * ``response_json["text"]`` is the fully decoded text of ``output_ids``,
    decoded server-side (SGLang holds the tokenizer; this client never
    tokenises or decodes anything itself).
  * Control endpoints ``/health`` (GET), ``/update_weights_from_disk``,
    ``/flush_cache``, ``/release_memory_occupation`` and
    ``/resume_memory_occupation`` (all POST) return HTTP 200 with a JSON body
    on success; ``/update_weights_from_disk``'s body carries a boolean
    ``"success"`` key (checked in addition to the HTTP status, since a
    same-process engine failure can still answer 200 with a failure body).

WHAT IS NOT CLAIMED: any retry, backoff or connection pooling -- one call is
one socket, opened and closed by ``urllib``; and no behaviour for any
SGLang endpoint this module does not name above.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from foundationscale.agentic_rl.engines import EngineInfraError, EngineRefusal
from foundationscale.agentic_rl.harness.base import Generation, SamplingParams

__all__ = ("SGLangClient",)

_FINISH_TYPES = ("stop", "length", "abort")


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


def _sampling_payload(sampling: SamplingParams) -> dict[str, float | int]:
    """Translate ``SamplingParams`` into SGLang's native ``sampling_params`` dict.

    ``top_k == 0`` (FoundationScale's "no restriction") becomes SGLang's native
    disablement sentinel ``-1`` -- see the module docstring's ASSUMPTIONS.
    """
    top_k = -1 if sampling.top_k == 0 else sampling.top_k
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": top_k,
        "min_p": sampling.min_p,
        "presence_penalty": sampling.presence_penalty,
        "repetition_penalty": sampling.repetition_penalty,
        "max_new_tokens": sampling.max_new_tokens,
    }


def _request(
    url: str, body: Mapping[str, Any] | None, *, timeout_s: float, expect_json: bool = True
) -> dict[str, Any]:
    """POST ``body`` as JSON to ``url`` (GET when ``body`` is ``None``) and return the
    decoded JSON response, raising :class:`EngineInfraError` for every way this can
    fail short of a well-formed 200 response: a transport failure (``kind="transport"``),
    a timeout (``kind="timeout"``), a non-2xx status (``kind="http_<code>"``) or a body
    that is not valid JSON (``kind="bad_response"``).
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            raw = response.read()
    except TimeoutError as exc:
        raise EngineInfraError("timeout", f"{url}: no response within {timeout_s}s") from exc
    except urllib.error.HTTPError as exc:
        raise EngineInfraError(f"http_{exc.code}", f"{url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise EngineInfraError("timeout", f"{url}: no response within {timeout_s}s") from exc
        raise EngineInfraError("transport", f"{url}: {exc.reason}") from exc
    except OSError as exc:
        raise EngineInfraError("transport", f"{url}: {exc}") from exc
    if status != 200:
        raise EngineInfraError("http_" + str(status), f"{url}: HTTP {status}")
    if not expect_json:
        # Control endpoints (/health, /sleep, /wake_up, cache and memory calls)
        # answer 200 with an EMPTY body on the real server (probed 2026-10-08):
        # their success is the status, and parsing the body would turn every
        # healthy answer into a bad_response.
        return {}
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineInfraError(
            "bad_response", f"{url}: response body is not valid JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise EngineInfraError(
            "bad_response",
            f"{url}: response body is {_described(decoded)}, not a JSON object",
        )
    return decoded


@dataclass(frozen=True)
class SGLangClient:
    """Token-in/token-out ``GenerationClient`` over one SGLang server's native HTTP API.

    ``base_url`` carries no trailing slash requirement (one is stripped if present);
    every call appends its own ``/<endpoint>``. See the module docstring for the
    full set of ASSUMPTIONS about SGLang's native request/response shapes this
    client was written against.
    """

    base_url: str
    timeout_s: float

    def __post_init__(self) -> None:
        where = "SGLangClient"
        if not isinstance(self.base_url, str) or not self.base_url:
            raise EngineRefusal(
                f"{where}: field 'base_url' is {_described(self.base_url)}: it must be a "
                f"non-empty str -- an empty base URL names no server to talk to"
            )
        parsed_url = urllib.parse.urlsplit(self.base_url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise EngineRefusal(
                f"{where}: field 'base_url' is {_described(self.base_url)}: it must be an "
                f"http:// or https:// URL naming a host -- any other scheme (e.g. "
                f"file://) would let _request() reach something that is not an HTTP "
                f"server, such as a local file"
            )
        if type(self.timeout_s) is bool or not isinstance(self.timeout_s, (int, float)):
            raise EngineRefusal(
                f"{where}: field 'timeout_s' is {_described(self.timeout_s)}: it must be a "
                f"real number (bool excluded) -- a bool timeout would silently become 0 or "
                f"1 second"
            )
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise EngineRefusal(
                f"{where}: field 'timeout_s' is {self.timeout_s!r}: it must be finite and "
                f"> 0 -- a non-positive or infinite timeout would never bound a request"
            )
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        object.__setattr__(self, "timeout_s", float(self.timeout_s))

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint}"

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        """POST ``/generate`` and parse the response into a ``Generation``.

        Refuses with ``EngineInfraError(kind="bad_response")`` on any shape violation:
        ``output_ids``/``output_token_logprobs`` length mismatch (both counts named),
        a logged token id that disagrees with the sampled output id (position, both
        ids named), or a ``finish_reason.type`` outside {"stop","length","abort"}.
        """
        payload = {
            "input_ids": list(prompt_ids),
            "sampling_params": _sampling_payload(sampling),
            "return_logprob": True,
        }
        response = await asyncio.to_thread(
            _request, self._url("generate"), payload, timeout_s=self.timeout_s
        )
        return _parse_generate_response(response, origin=self._url("generate"))

    def health(self) -> bool:
        """GET ``/health``; ``True`` on HTTP 200, ``False`` on any failure to reach it or
        any non-200 status -- this call never raises, since a caller polling for
        readiness (``engines.fleet.SGLangServer.start``) needs a plain yes/no.
        """
        try:
            _request(self._url("health"), None, timeout_s=self.timeout_s, expect_json=False)
        except EngineInfraError:
            return False
        return True

    def update_weights_from_disk(self, model_path: str) -> None:
        """POST ``/update_weights_from_disk`` with ``{"model_path": model_path}``.

        Raises ``EngineInfraError`` on a non-2xx status (via ``_request``) or when the
        decoded body's ``"success"`` key is present and falsy -- the one SGLang
        control response this module treats as fallible-with-a-200, see the module
        docstring's ASSUMPTIONS.
        """
        if not isinstance(model_path, str) or not model_path:
            raise EngineRefusal(
                f"SGLangClient.update_weights_from_disk: parameter 'model_path' is "
                f"{_described(model_path)}: it must be a non-empty str -- an empty path "
                f"names no checkpoint directory to load"
            )
        url = self._url("update_weights_from_disk")
        response = _request(url, {"model_path": model_path}, timeout_s=self.timeout_s)
        success = response.get("success", True)
        if success is not True and success is not False:
            raise EngineInfraError(
                "bad_response",
                f"{url}: field 'success' is {_described(success)}, not a bool",
            )
        if success is False:
            message = response.get("message", "<no message>")
            raise EngineInfraError("update_failed", f"{url}: success=False: {message}")

    def flush_cache(self) -> None:
        """POST ``/flush_cache`` with an empty body."""
        _request(self._url("flush_cache"), {}, timeout_s=self.timeout_s, expect_json=False)

    def release_memory(self) -> None:
        """POST ``/release_memory_occupation`` with an empty body, for colocation."""
        _request(
            self._url("release_memory_occupation"), {}, timeout_s=self.timeout_s, expect_json=False
        )

    def resume_memory(self) -> None:
        """POST ``/resume_memory_occupation`` with an empty body, for colocation."""
        _request(
            self._url("resume_memory_occupation"), {}, timeout_s=self.timeout_s, expect_json=False
        )


def _parse_generate_response(response: Mapping[str, Any], *, origin: str) -> Generation:
    output_ids = response.get("output_ids")
    if not isinstance(output_ids, list) or not all(type(i) is int for i in output_ids):
        raise EngineInfraError(
            "bad_response", f"{origin}: field 'output_ids' is {_described(output_ids)}"
        )
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise EngineInfraError(
            "bad_response", f"{origin}: field 'meta_info' is {_described(meta_info)}"
        )
    logprob_entries = meta_info.get("output_token_logprobs")
    if not isinstance(logprob_entries, list):
        raise EngineInfraError(
            "bad_response",
            f"{origin}: field 'meta_info.output_token_logprobs' is {_described(logprob_entries)}",
        )
    if len(logprob_entries) != len(output_ids):
        raise EngineInfraError(
            "bad_response",
            f"{origin}: 'output_ids' has {len(output_ids)} token(s) but "
            f"'meta_info.output_token_logprobs' has {len(logprob_entries)} entry/entries "
            f"-- one logprob entry per sampled token is required",
        )
    logprobs: list[float] = []
    for position, (entry, output_id) in enumerate(zip(logprob_entries, output_ids, strict=True)):
        if (
            not isinstance(entry, list)
            or len(entry) < 2
            or type(entry[0]) is bool
            or not isinstance(entry[0], (int, float))
            or not isfinite(float(entry[0]))
        ):
            raise EngineInfraError(
                "bad_response",
                f"{origin}: 'meta_info.output_token_logprobs' entry {position} is "
                f"{_described(entry)}, not a [logprob, token_id, ...] list with a real "
                f"finite logprob (bool excluded)",
            )
        logged_token_id = entry[1]
        if type(logged_token_id) is not int or logged_token_id != output_id:
            raise EngineInfraError(
                "bad_response",
                f"{origin}: 'meta_info.output_token_logprobs' entry {position} names "
                f"token_id {logged_token_id!r} but 'output_ids' entry {position} is "
                f"{output_id!r}: the logged token id must be a real int "
                f"(type(x) is int, bool excluded) equal to the sampled output id",
            )
        logprobs.append(float(entry[0]))
    finish_reason = meta_info.get("finish_reason")
    finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else None
    if finish_type not in _FINISH_TYPES:
        raise EngineInfraError(
            "bad_response",
            f"{origin}: field 'meta_info.finish_reason.type' is {_described(finish_type)}: "
            f"it must be one of {_FINISH_TYPES}",
        )
    text = response.get("text")
    if not isinstance(text, str):
        raise EngineInfraError("bad_response", f"{origin}: field 'text' is {_described(text)}")
    return Generation(
        token_ids=tuple(output_ids),
        logprobs=tuple(logprobs),
        finish_reason=finish_type,
        text=text,
    )
