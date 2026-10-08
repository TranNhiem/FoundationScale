"""``VLLMClient``: a token-in/token-out ``GenerationClient`` over vLLM's
OpenAI-compatible HTTP server, plus the control calls one rollout host needs
for colocated weight sync.

PROBED 2026-10-08 against vLLM 0.21.1rc1 (enroot image on the GB200 estate, model
qwen3_5 9B, server started with VLLM_SERVER_DEV_MODE=1): token-in ``/v1/completions``
with ``return_token_ids`` returns ``choices[0].token_ids`` that agree with the
``"token_id:<int>"`` logprob tokens; greedy output is deterministic; ``/sleep``,
``/wake_up`` and ``/collective_rpc`` answer 200; ``reload_weights`` accepts
``kwargs={"weights_path": <dir>}`` (``model_path`` is rejected with a 500) and leaves
greedy logprobs bit-identical when reloading identical weights. The server also
exposes ``/init_weight_transfer_engine``, ``/start_weight_update``, ``/update_weights``
and ``/finish_weight_update`` (not used yet).

UNPROBED ASSUMPTIONS (stated here because no vLLM install exists in this
environment to probe -- the ``vllm`` package itself is never imported by this
module or anything in this plane; a cluster probe must confirm the installed
server version honours every one of these before this client is pointed at a
real GB200 engine):

  * The OpenAI-compatible ``/v1/completions`` endpoint (never ``/v1/chat/
    completions``: this plane is token-in/token-out, and chat completions
    would re-apply a chat template server-side) accepts a request body shaped
    ``{"model": <served model name>, "prompt": [<token id>, ...],
    "max_tokens": <int>, "temperature": <float>, "top_p": <float>,
    "top_k": <int>, "min_p": <float>, "presence_penalty": <float>,
    "repetition_penalty": <float>, "logprobs": 1, "return_tokens_as_token_ids":
    true, "return_token_ids": true, "skip_special_tokens": false}``, and that
    every one of these non-OpenAI-standard fields (``top_k``, ``min_p``,
    ``repetition_penalty``, ``return_tokens_as_token_ids``, ``return_token_ids``,
    ``skip_special_tokens``) is accepted by the deployed vLLM version rather
    than refused or silently ignored.
  * ``"model"`` must exactly equal the string the server was started with
    (``--served-model-name``, or the model path/id when that flag was not
    given) -- this client never discovers it and the caller must declare it.
  * ``top_k`` has no "disabled" value of its own in ``harness.base.
    SamplingParams`` (it only requires ``>= 0``); this client treats
    FoundationScale's ``top_k == 0`` as "no top-k filtering" and sends vLLM's
    documented disablement sentinel ``-1`` for it, mirroring
    ``engines.sglang``'s identical translation for SGLang's native API --
    UNPROBED for vLLM specifically, carried over by analogy.
  * Sampled token ids: NEWER vLLM builds return them directly as
    ``choices[0].token_ids`` (a list of ints); this client prefers that field
    when present. OLDER builds require ``return_tokens_as_token_ids=true`` and
    expose them only as ``choices[0].logprobs.tokens``, a list of STRINGS of
    the form ``"token_id:<int>"`` -- this client parses that prefix when
    ``choices[0].token_ids`` is absent. Which shape a given deployment uses is
    unprobed; both paths are implemented so the client works either way.
  * ``choices[0].logprobs.token_logprobs`` is a list of per-token log
    probabilities, one per sampled token, aligned by position with the
    recovered token ids -- regardless of which of the two paths above produced
    them.
  * ``choices[0].finish_reason`` is one of the OpenAI-standard strings; this
    client accepts exactly ``"stop"`` and ``"length"`` and maps every other
    value -- including ``"abort"``, ``null``, or any string this plane does
    not otherwise recognise -- to ``"abort"``, per the module's own declared
    three-way ``Generation.finish_reason`` contract.
  * ``/health`` (GET) returns HTTP 200 with no particular body on success --
    vLLM's own documented health-check route, carried over unverified from the
    same assumption in ``engines.sglang``.
  * ``reload_weights`` (POST ``/collective_rpc`` with body ``{"method":
    "reload_weights", "kwargs": {"weights_path": <path>}}``) is NOT a standard
    vLLM HTTP route: ``collective_rpc`` is a real vLLM endpoint for invoking an
    arbitrary method on every worker, but whether a ``"reload_weights"``
    method exists, what it is named, and what keyword arguments it takes are
    ALL version- and deployment-dependent (it typically requires the server to
    have been started in a development/RPC-enabled mode). Because this is the
    least-proven assumption in this module, the call is gated behind the
    explicit ``reload_mode`` field -- ``reload_weights`` refuses outright
    unless the caller declares ``reload_mode="collective_rpc"``, so an
    unconfigured client never silently attempts an unproven RPC. The cluster
    probe decides whether this mode is usable at all; until it runs,
    ``weight_sync.DiskWeightSync(mode="restart")`` is the proven S0 path (see
    that module).
  * ``/sleep?level=<int>`` and ``/wake_up`` (both POST) are real vLLM routes
    added for RLHF-style colocated serving (freeing/restoring GPU memory
    between rollout and training phases); this client calls them unconditionally
    (unlike ``reload_weights``) because they are documented vLLM features, but
    the exact set of legal ``level`` values (vLLM's own docs describe 1 and 2
    with different trade-offs) and the response shape on success are unprobed
    here -- a non-2xx status or a transport failure is reported as
    ``EngineInfraError`` the same way every other control call reports one.

WHAT IS NOT CLAIMED: any retry, backoff or connection pooling -- one call is
one socket, opened and closed by ``urllib``, mirroring ``engines.sglang``; and
no behaviour for ``/v1/chat/completions`` or any other vLLM endpoint this
module does not name above.
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
from typing import Any, Literal

from foundationscale.agentic_rl.engines import EngineInfraError, EngineRefusal
from foundationscale.agentic_rl.harness.base import Generation, SamplingParams

__all__ = ("VLLMClient",)

_FINISH_TYPES = ("stop", "length")
_TOKEN_ID_ENTRY_PREFIX = "token_id:"
_RELOAD_MODES = ("collective_rpc",)


def _described(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


def _completions_payload(
    served_model_name: str, prompt_ids: Sequence[int], sampling: SamplingParams
) -> dict[str, Any]:
    """Translate ``SamplingParams`` into vLLM's ``/v1/completions`` request body.

    ``top_k == 0`` (FoundationScale's "no restriction") becomes vLLM's documented
    disablement sentinel ``-1`` -- see the module docstring's UNPROBED ASSUMPTIONS.
    """
    top_k = -1 if sampling.top_k == 0 else sampling.top_k
    return {
        "model": served_model_name,
        "prompt": list(prompt_ids),
        "max_tokens": sampling.max_new_tokens,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": top_k,
        "min_p": sampling.min_p,
        "presence_penalty": sampling.presence_penalty,
        "repetition_penalty": sampling.repetition_penalty,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "skip_special_tokens": False,
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
class VLLMClient:
    """Token-in/token-out ``GenerationClient`` over one vLLM server's OpenAI-compatible
    HTTP API.

    ``base_url`` carries no trailing slash requirement (one is stripped if present);
    every call appends its own ``/<endpoint>``. ``served_model_name`` is the exact
    ``"model"`` string every ``/v1/completions`` request must carry -- see the module
    docstring's UNPROBED ASSUMPTIONS for the full set this client was written against.
    ``reload_mode`` gates :meth:`reload_weights`: ``None`` (the default) means the
    caller has not proven the ``/collective_rpc`` route works against this deployment,
    and the method refuses rather than attempt an unconfirmed API.
    """

    base_url: str
    timeout_s: float
    served_model_name: str
    reload_mode: Literal["collective_rpc"] | None = None

    def __post_init__(self) -> None:
        where = "VLLMClient"
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
        if not isinstance(self.served_model_name, str) or not self.served_model_name:
            raise EngineRefusal(
                f"{where}: field 'served_model_name' is {_described(self.served_model_name)}: "
                f"it must be a non-empty str -- vLLM's /v1/completions refuses a request "
                f"whose 'model' does not match the name the server was started with, and an "
                f"empty name can match none"
            )
        if self.reload_mode is not None and self.reload_mode not in _RELOAD_MODES:
            raise EngineRefusal(
                f"{where}: field 'reload_mode' is {_described(self.reload_mode)}: it must be "
                f"None or one of {_RELOAD_MODES} -- an undeclared mode cannot be dispatched "
                f"by reload_weights()"
            )
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        object.__setattr__(self, "timeout_s", float(self.timeout_s))

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint}"

    async def generate(self, prompt_ids: Sequence[int], sampling: SamplingParams) -> Generation:
        """POST ``/v1/completions`` and parse the response into a ``Generation``.

        Refuses with ``EngineInfraError(kind="bad_response")`` on any shape violation:
        a missing/malformed ``choices``, an unparseable token id, a
        ``token_logprobs``/token-id-count mismatch (both counts named), or a
        non-str ``text``. ``finish_reason`` outside {"stop", "length"} (including
        "abort" or a missing/null value) becomes ``"abort"``, per the module's
        UNPROBED ASSUMPTIONS.
        """
        payload = _completions_payload(self.served_model_name, prompt_ids, sampling)
        response = await asyncio.to_thread(
            _request, self._url("v1/completions"), payload, timeout_s=self.timeout_s
        )
        return _parse_completions_response(response, origin=self._url("v1/completions"))

    def health(self) -> bool:
        """GET ``/health``; ``True`` on HTTP 200, ``False`` on any failure to reach it or
        any non-200 status -- this call never raises, since a caller polling for
        readiness (``engines.fleet.EngineServer.start``) needs a plain yes/no.
        """
        try:
            _request(self._url("health"), None, timeout_s=self.timeout_s, expect_json=False)
        except EngineInfraError:
            return False
        return True

    def reload_weights(self, model_path: str) -> None:
        """POST ``/collective_rpc`` with ``{"method": "reload_weights", "kwargs":
        {"weights_path": model_path}}`` -- ONLY when ``reload_mode == "collective_rpc"``.

        Refuses with ``EngineRefusal`` when ``reload_mode`` is unset: this is the
        least-proven assumption in the module (see its docstring), and an
        unconfigured client must never silently attempt it. Unlike
        ``engines.sglang.SGLangClient.update_weights_from_disk``, no ``"success"``
        response field is checked -- vLLM's ``collective_rpc`` response shape for
        this (non-standard) method is itself unprobed, so only the HTTP-level
        failure modes ``_request`` already reports are treated as failure.
        """
        if not isinstance(model_path, str) or not model_path:
            raise EngineRefusal(
                f"VLLMClient.reload_weights: parameter 'model_path' is "
                f"{_described(model_path)}: it must be a non-empty str -- an empty path "
                f"names no checkpoint directory to load"
            )
        if self.reload_mode != "collective_rpc":
            raise EngineRefusal(
                f"VLLMClient.reload_weights: field 'reload_mode' is "
                f"{_described(self.reload_mode)}: reload_weights() requires "
                f"reload_mode='collective_rpc', declared explicitly at construction -- "
                f"the /collective_rpc 'reload_weights' RPC is unproven on this "
                f"deployment until a cluster probe confirms it (see the module "
                f"docstring's UNPROBED ASSUMPTIONS); weight_sync.DiskWeightSync("
                f"mode='restart') is the proven S0 path until then"
            )
        _request(
            self._url("collective_rpc"),
            {"method": "reload_weights", "kwargs": {"weights_path": model_path}},
            timeout_s=self.timeout_s,
        )

    def update_weights_from_disk(self, model_path: str) -> None:
        """Thin alias of :meth:`reload_weights`, named to match
        ``engines.sglang.SGLangClient.update_weights_from_disk`` so
        ``engines.fleet.EngineFleet.update_weights_from_disk`` can fan out to a fleet
        mixing both engine kinds through one uniform client method name, with no
        kind-switch in the fleet itself.
        """
        self.reload_weights(model_path)

    def sleep(self, level: int) -> None:
        """POST ``/sleep?level=<level>`` -- a declared OPTIONAL capability (see the
        module docstring); called unconditionally, with no ``reload_mode``-style gate,
        because it is a documented vLLM route rather than an unproven RPC method.
        """
        if type(level) is bool or type(level) is not int or level < 0:
            raise EngineRefusal(
                f"VLLMClient.sleep: parameter 'level' is {_described(level)}: it must be "
                f"a real int >= 0 (bool excluded) -- vLLM's sleep levels are small "
                f"non-negative integers naming a memory-release depth"
            )
        _request(
            f"{self._url('sleep')}?level={level}", {}, timeout_s=self.timeout_s, expect_json=False
        )

    def wake_up(self) -> None:
        """POST ``/wake_up`` with an empty body -- the inverse of :meth:`sleep`, for
        colocation.
        """
        _request(self._url("wake_up"), {}, timeout_s=self.timeout_s, expect_json=False)


def _recover_token_ids(choice: Mapping[str, Any], *, origin: str) -> list[int]:
    """Prefer ``choice["token_ids"]`` (newer vLLM); else parse
    ``choice["logprobs"]["tokens"]`` entries of the form ``"token_id:<int>"``.
    """
    raw_token_ids = choice.get("token_ids")
    if raw_token_ids is not None:
        if not isinstance(raw_token_ids, list) or not all(type(i) is int for i in raw_token_ids):
            raise EngineInfraError(
                "bad_response",
                f"{origin}: field 'choices[0].token_ids' is {_described(raw_token_ids)}: "
                f"it must be a list[int] when present",
            )
        return list(raw_token_ids)
    logprobs_field = choice.get("logprobs")
    if not isinstance(logprobs_field, dict):
        raise EngineInfraError(
            "bad_response",
            f"{origin}: field 'choices[0].token_ids' is absent and "
            f"'choices[0].logprobs' is {_described(logprobs_field)}: with no "
            f"token_ids field, sampled token ids must be recoverable from "
            f"logprobs.tokens",
        )
    raw_tokens = logprobs_field.get("tokens")
    if not isinstance(raw_tokens, list):
        raise EngineInfraError(
            "bad_response",
            f"{origin}: field 'choices[0].logprobs.tokens' is {_described(raw_tokens)}",
        )
    token_ids: list[int] = []
    for position, entry in enumerate(raw_tokens):
        if not isinstance(entry, str) or not entry.startswith(_TOKEN_ID_ENTRY_PREFIX):
            raise EngineInfraError(
                "bad_response",
                f"{origin}: field 'choices[0].logprobs.tokens' entry {position} is "
                f"{_described(entry)}: expected a str of the form "
                f"{_TOKEN_ID_ENTRY_PREFIX!r}<int> (requires "
                f"return_tokens_as_token_ids=true)",
            )
        raw_id = entry[len(_TOKEN_ID_ENTRY_PREFIX) :]
        try:
            token_ids.append(int(raw_id))
        except ValueError as exc:
            raise EngineInfraError(
                "bad_response",
                f"{origin}: field 'choices[0].logprobs.tokens' entry {position} is "
                f"{entry!r}: {raw_id!r} is not a valid int after the "
                f"{_TOKEN_ID_ENTRY_PREFIX!r} prefix",
            ) from exc
    return token_ids


def _recover_logprobs(choice: Mapping[str, Any], *, token_count: int, origin: str) -> list[float]:
    logprobs_field = choice.get("logprobs")
    if not isinstance(logprobs_field, dict):
        raise EngineInfraError(
            "bad_response", f"{origin}: field 'choices[0].logprobs' is {_described(logprobs_field)}"
        )
    raw_logprobs = logprobs_field.get("token_logprobs")
    if not isinstance(raw_logprobs, list):
        raise EngineInfraError(
            "bad_response",
            f"{origin}: field 'choices[0].logprobs.token_logprobs' is {_described(raw_logprobs)}",
        )
    if len(raw_logprobs) != token_count:
        raise EngineInfraError(
            "bad_response",
            f"{origin}: 'choices[0].logprobs.token_logprobs' has {len(raw_logprobs)} "
            f"entry/entries but {token_count} sampled token(s) were recovered -- one "
            f"logprob per sampled token is required",
        )
    logprobs: list[float] = []
    for position, entry in enumerate(raw_logprobs):
        if type(entry) is bool or not isinstance(entry, (int, float)) or not isfinite(float(entry)):
            raise EngineInfraError(
                "bad_response",
                f"{origin}: 'choices[0].logprobs.token_logprobs' entry {position} is "
                f"{_described(entry)}, not a real finite logprob (bool excluded)",
            )
        logprobs.append(float(entry))
    return logprobs


def _parse_completions_response(response: Mapping[str, Any], *, origin: str) -> Generation:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise EngineInfraError(
            "bad_response", f"{origin}: field 'choices' is {_described(choices)}"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise EngineInfraError(
            "bad_response",
            f"{origin}: field 'choices[0]' is {_described(choice)}, not a JSON object",
        )
    token_ids = _recover_token_ids(choice, origin=origin)
    logprobs = _recover_logprobs(choice, token_count=len(token_ids), origin=origin)
    finish_reason_raw = choice.get("finish_reason")
    finish_type: Literal["stop", "length", "abort"] = (
        finish_reason_raw if finish_reason_raw in _FINISH_TYPES else "abort"
    )
    text = choice.get("text")
    if not isinstance(text, str):
        raise EngineInfraError(
            "bad_response", f"{origin}: field 'choices[0].text' is {_described(text)}"
        )
    return Generation(
        token_ids=tuple(token_ids),
        logprobs=tuple(logprobs),
        finish_reason=finish_type,
        text=text,
    )
