"""ARL-G02: coverage for the judge plane's transport, credential and shape refusals.

``OpenAIJudgeClient`` is driven against a REAL loopback ``ThreadingHTTPServer`` (never the
external network) which can serve a valid chat completion, an HTTP 500, an empty or malformed
body, missing ``choices``, empty content, or no answer before the client's timeout. Everything
else executes each validation branch this module owns: ``JudgeInfraError``'s kind gate, the
``_check_*`` helpers behind their public dataclasses, ``Rubric``'s shape refusals, the bearer
read at call time, ``_completion_text``'s body parsing and the calibration entry points.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.contracts import Termination
from foundationscale.agentic_rl.harness.base import EpisodeOutcome, EpisodeTask
from foundationscale.agentic_rl.rewards.base import RewardVerdict
from foundationscale.agentic_rl.rewards.judge import (
    INFRA_KINDS,
    CalibrationReport,
    Criterion,
    JudgeInfraError,
    JudgeParseError,
    JudgeRewardFn,
    OpenAIJudgeClient,
    Rubric,
    _judge_schema,
    calibrate,
    require_calibrated,
)

ONE = Criterion("correctness", "Is the answer right?", 1.0)
_ANSWER = "The answer is four."
_MESSAGES = [{"role": "system", "content": "Judge it."}, {"role": "user", "content": "2+2?"}]
_SCHEMA = {"type": "object"}


class _Wire:
    """One loopback judge endpoint's canned reply and the request it saw."""

    def __init__(self) -> None:
        self.status = 200
        self.body = b""
        self.sleep_s = 0.0
        self.seen_auth: list[str | None] = []
        self.seen_body = b""


class _QuietServer(ThreadingHTTPServer):
    """A loopback judge that shrugs at the BrokenPipe an aborted request provokes."""

    def handle_error(self, request: object, client_address: object) -> None:
        """Swallow the traceback of a client timing out mid-reply."""


@pytest.fixture
def wire() -> Iterator[tuple[_Wire, str]]:
    state = _Wire()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (the stdlib names this hook)
            state.seen_auth.append(self.headers.get("Authorization"))
            state.seen_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if state.sleep_s:
                time.sleep(state.sleep_s)
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(state.body)))
            self.end_headers()
            self.wfile.write(state.body)

        def log_message(self, *args: object) -> None:
            """Silence the stdlib's per-request stderr chatter."""

    server = _QuietServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def _completion(content: str) -> bytes:
    """One valid chat completion whose assistant content is ``content``."""
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


def _client(
    base_url: str, *, api_key_env: str | None = "FS_JUDGE_KEY", timeout_s: float = 5.0
) -> OpenAIJudgeClient:
    return OpenAIJudgeClient(
        base_url=base_url, model="judge-1", api_key_env=api_key_env, timeout_s=timeout_s
    )


def _complete(client: OpenAIJudgeClient) -> str:
    return asyncio.run(
        client.complete([dict(message) for message in _MESSAGES], json_schema=_SCHEMA)
    )


def _rubric() -> Rubric:
    return Rubric("rubric_1", (ONE,), "Judge the answer.")


def _task() -> EpisodeTask:
    return EpisodeTask(
        uid="judge-1",
        session_id="judge-1#0",
        messages=({"role": "user", "content": "What is 2+2?"},),
        metadata={"answer": "4"},
    )


def _outcome() -> EpisodeOutcome:
    return EpisodeOutcome(
        trajectory_turns=(),
        prompt_turns=(),
        termination=Termination.STOP,
        submitted_answer=_ANSWER,
        abstention_reason=None,
    )


def _judged(client: OpenAIJudgeClient) -> RewardVerdict:
    fn = JudgeRewardFn(client=client, rubric=_rubric(), decode=lambda ids: "")
    return asyncio.run(fn.score(_task(), _outcome()))


# -- JudgeInfraError and the _check_* internals ------------------------------------------------


@pytest.mark.parametrize("kind", ["expired", True, None, 3])
def test_judge_infra_error_refuses_an_illegal_kind_and_stamps_a_legal_one(kind: object) -> None:
    with pytest.raises(JudgeParseError, match="legal set"):
        JudgeInfraError(kind)  # type: ignore[arg-type]
    failure = JudgeInfraError("timeout")
    assert (failure.kind, "timeout" in str(failure)) == ("timeout", True)


def test_check_int_refuses_the_shapes_that_are_not_counts_and_those_below_the_floor() -> None:
    for bad in (True, 3.5, "30"):  # bool / float / str never stands in for a measured count
        with pytest.raises(JudgeParseError, match=r"an int \(bool excluded\)"):
            CalibrationReport("rubric_1", bad, 0.5, 0.5)  # type: ignore[arg-type]
    with pytest.raises(JudgeParseError, match=r"legal range is \[0, \+inf\)"):
        CalibrationReport("rubric_1", -1, 0.5, 0.5)


def test_check_str_refuses_a_description_that_is_not_a_non_empty_str() -> None:
    for bad in (b"bytes", 42, ""):
        with pytest.raises(JudgeParseError, match="must be a non-empty str"):
            Criterion("correctness", bad, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload", ["[]", '{"rubric_id": "r", "instructions": "", "criteria": [1]}']
)
def test_from_json_refuses_a_non_object_where_an_object_is_required(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "rubric.json"
    path.write_text(payload)
    with pytest.raises(JudgeParseError, match="JSON object mapping"):
        Rubric.from_json(str(path))


# -- Rubric shape refusals ---------------------------------------------------------------------


@pytest.mark.parametrize("criteria", [5, "names"])
def test_rubric_refuses_criteria_that_is_not_a_sequence_of_criteria(criteria: object) -> None:
    with pytest.raises(JudgeParseError, match=r"tuple\[Criterion, \.\.\.\]"):
        Rubric("rubric_1", criteria, "Judge the answer.")  # type: ignore[arg-type]


def test_rubric_refuses_a_criterion_entry_of_the_wrong_type() -> None:
    with pytest.raises(JudgeParseError, match="every entry must be a Criterion"):
        Rubric("rubric_1", ("not-a-criterion",), "Judge.")  # type: ignore[arg-type]


@pytest.mark.parametrize("scores", [None, "correctness=1", 5])
def test_score_refuses_a_criteria_scores_that_is_not_a_mapping(scores: object) -> None:
    with pytest.raises(JudgeParseError, match=r"Mapping\[str, float\]"):
        _rubric().score(scores)  # type: ignore[arg-type]


@pytest.mark.parametrize("criteria", [5, "crowd"])
def test_from_json_refuses_criteria_that_is_not_a_list_of_objects(
    tmp_path: Path, criteria: object
) -> None:
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps({"rubric_id": "r", "instructions": "", "criteria": criteria}))
    with pytest.raises(JudgeParseError, match="must be a list of criterion objects"):
        Rubric.from_json(str(path))


# -- JudgeRewardFn / calibrate / require_calibrated entry refusals ------------------------------


def test_judge_reward_fn_refuses_an_unusable_client_rubric_or_decode() -> None:
    good = _client("http://127.0.0.1:9")
    with pytest.raises(JudgeParseError, match="JudgeClient protocol"):
        JudgeRewardFn(client=object(), rubric=_rubric(), decode=lambda ids: "")
    with pytest.raises(JudgeParseError, match="it must be a Rubric"):
        JudgeRewardFn(client=good, rubric="rubric_1", decode=lambda ids: "")  # type: ignore
    with pytest.raises(JudgeParseError, match="must be callable"):
        JudgeRewardFn(client=good, rubric=_rubric(), decode=None)  # type: ignore[arg-type]


def test_calibrate_refuses_a_judge_scores_or_gold_that_is_not_a_sequence() -> None:
    with pytest.raises(JudgeParseError, match=r"Sequence\[float \| None\]"):
        calibrate(judge_scores=None, gold=[0.5])  # type: ignore[arg-type]
    with pytest.raises(JudgeParseError, match=r"it must be a Sequence\[float\]"):
        calibrate(judge_scores=[0.5], gold="gold")  # type: ignore[arg-type]


def test_require_calibrated_refuses_a_rubric_that_is_not_a_rubric(tmp_path: Path) -> None:
    with pytest.raises(JudgeParseError, match="it must be a Rubric"):
        require_calibrated(str(tmp_path / "calibration.json"), "rubric_1", 0.9)  # type: ignore


# -- OpenAIJudgeClient against a real loopback judge -------------------------------------------


def test_complete_returns_the_content_and_sends_the_bearer(
    wire: tuple[_Wire, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state, base = wire
    monkeypatch.setenv("FS_JUDGE_KEY", "sk-fs-secret")
    state.body = _completion('{"criteria": {"correctness": 1}}')
    assert _complete(_client(base)) == '{"criteria": {"correctness": 1}}'
    assert state.seen_auth == ["Bearer sk-fs-secret"]
    posted = json.loads(state.seen_body)
    assert posted["messages"] == _MESSAGES and posted["temperature"] == 0
    assert posted["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "judge", "schema": _SCHEMA, "strict": True},
    }


def test_complete_without_an_api_key_env_sends_no_authorization(wire: tuple[_Wire, str]) -> None:
    state, base = wire
    state.body = _completion("plain content")
    assert _complete(_client(base, api_key_env=None)) == "plain content"
    assert state.seen_auth == [None]


def test_missing_env_stops_before_the_wire_and_names_neither_key_nor_value(
    wire: tuple[_Wire, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state, base = wire
    monkeypatch.delenv("FS_JUDGE_KEY", raising=False)
    with pytest.raises(JudgeInfraError) as failure:
        _complete(_client(base))
    assert failure.value.kind == "connection"
    assert state.seen_auth == [] and state.seen_body == b""
    assert "FS_JUDGE_KEY" not in str(failure.value) and "sk-" not in str(failure.value)


def test_a_500_from_the_endpoint_is_an_http_fault(wire: tuple[_Wire, str]) -> None:
    state, base = wire
    state.status = 500
    state.body = b'{"error": "the judge model exploded"}'
    with pytest.raises(JudgeInfraError) as failure:
        _complete(_client(base, api_key_env=None))
    assert failure.value.kind == "http"
    assert "exploded" not in str(failure.value)  # no response body leaks into the error


def test_an_endpoint_slower_than_the_timeout_makes_the_reward_abstain(
    wire: tuple[_Wire, str],
) -> None:
    state, base = wire
    state.sleep_s, state.body = 0.8, _completion("too late to matter")
    with pytest.raises(JudgeInfraError) as failure:
        _complete(_client(base, api_key_env=None, timeout_s=0.3))
    assert failure.value.kind == "timeout"


@pytest.mark.parametrize(
    ("raised", "kind"),
    [
        (TimeoutError("the socket gave up"), "timeout"),
        (urllib.error.URLError(TimeoutError("the socket gave up")), "timeout"),
        (urllib.error.URLError(OSError("connection refused")), "connection"),
        (OSError("the socket died under us"), "connection"),
    ],
)
def test_every_transport_failure_maps_to_an_infra_kind(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, kind: str
) -> None:
    def raiser(*_args: object, **_kwargs: object) -> Any:
        raise raised

    monkeypatch.setattr(urllib.request, "urlopen", raiser)
    with pytest.raises(JudgeInfraError) as failure:
        # api_key_env=None: a missing key would fail first, as "connection", before urlopen
        _complete(_client("http://127.0.0.1:9", api_key_env=None, timeout_s=0.3))
    assert failure.value.kind == kind


def test_an_unreachable_endpoint_is_a_connection_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FS_JUDGE_KEY", "sk-fs-secret")
    with socket.socket() as probe:  # bind, learn a port, then leave nothing listening on it
        probe.bind(("127.0.0.1", 0))
        dead = int(probe.getsockname()[1])
    with pytest.raises(JudgeInfraError) as failure:
        _complete(_client(f"http://127.0.0.1:{dead}"))
    assert failure.value.kind == "connection"
    assert "sk-fs-secret" not in str(failure.value)


@pytest.mark.parametrize(
    ("body", "kind"),
    [
        (b"", "http"),
        (b"<html>not json</html>", "http"),
        (b"[1, 2, 3]", "http"),
        (b"{}", "empty"),
        (b'{"choices": []}', "empty"),
        (b'{"choices": "gone"}', "empty"),
        (b'{"choices": [{"message": {}}]}', "empty"),
        (b'{"choices": [{"message": {"content": ""}}]}', "empty"),
        (b'{"choices": [{"message": {"content": "   "}}]}', "empty"),
        (b'{"choices": [{"message": {"content": 7}}]}', "empty"),
    ],
)
def test_a_hostile_completion_body_is_named_by_its_infra_kind(
    wire: tuple[_Wire, str], body: bytes, kind: str
) -> None:
    state, base = wire
    state.body = body
    with pytest.raises(JudgeInfraError) as failure:
        _complete(_client(base, api_key_env=None))
    assert (failure.value.kind in INFRA_KINDS, failure.value.kind) == (True, kind)


def test_one_vote_through_the_real_client_scores_the_answer(wire: tuple[_Wire, str]) -> None:
    state, base = wire
    state.body = _completion('{"criteria": {"correctness": 0.25}, "rationale": "partly right"}')
    verdict = _judged(_client(base, api_key_env=None))
    assert (verdict.value, verdict.reason) == (0.25, None)
    posted = json.loads(state.seen_body)
    assert posted["chat_template_kwargs"] == {"enable_thinking": False}
    assert posted["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "judge", "schema": _judge_schema(["correctness"]), "strict": True},
    }


def test_a_vote_missing_its_criteria_key_is_judge_malformed(wire: tuple[_Wire, str]) -> None:
    state, base = wire
    state.body = _completion('{"rationale": "no scores here"}')
    verdict = _judged(_client(base, api_key_env=None))
    assert (verdict.value, verdict.reason) == (None, "judge_malformed")
