"""Unit tests for the Data Engine LLM backend contract."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest
from foundationskills.skills.data_engine import llm_backend as lb


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True


class _CaptureOpener:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request: object, timeout: float) -> object:
        self.requests.append(request)
        self.timeouts.append(timeout)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _ScriptBackend:
    kind = 'script'
    model = 'script-model'

    def __init__(self, responses: list[lb.LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict], float, int, int, bool]] = []

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 0,
        json_mode: bool = False,
    ) -> lb.LLMResponse:
        self.calls.append((list(messages), temperature, max_tokens, seed, json_mode))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class _FakeBackend(
    _ScriptBackend
):
    kind = 'fake'
    model = 'fake-model'

    def __init__(self, cfg: dict) -> None:
        content = str(cfg.get('content', 'fake-response'))
        super().__init__(
            [
                lb.LLMResponse(
                    content=content,
                    finish_reason='stop',
                    prompt_tokens=1,
                    completion_tokens=2,
                    cache_hit=False,
                    error=None,
                )
            ]
        )


@pytest.fixture
def fake_factory() -> tuple:
    old_backends = dict(lb.BACKENDS)
    created: list[_FakeBackend] = []

    def factory(cfg: dict) -> _FakeBackend:
        backend = _FakeBackend(cfg)
        created.append(backend)
        return backend

    lb.register_backend('fake', factory)
    try:
        yield factory, created
    finally:
        lb.BACKENDS.clear()
        lb.BACKENDS.update(old_backends)


def _api_payload(
    content: str | None = 'ok',
    finish_reason: str | None = 'stop',
    usage: dict | None = None,
) -> bytes:
    if content is None:
        message = {}
    else:
        message = {'role': 'assistant', 'content': content}
    payload = {
        'id': 'chatcmpl-test',
        'choices': [{'message': message, 'finish_reason': finish_reason}],
    }
    if usage is not None:
        payload['usage'] = usage
    return json.dumps(payload).encode('utf-8')


def _openai_backend(opener: _CaptureOpener, **overrides: object) -> lb.OpenAICompatibleBackend:
    cfg = {
        'base_url': 'http://host:8000',
        'model': 'test-model',
        'timeout_s': 7.5,
    }
    cfg.update(overrides)
    backend = lb.OpenAICompatibleBackend(cfg)
    backend.opener = opener
    return backend


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        'http://host:8000/v1/chat/completions',
        code,
        f'status {code}',
        hdrs=None,
        fp=None,
    )


def test_openai_url_normalizes_a_service_root() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(opener)
    response = backend.complete([{'role': 'user', 'content': 'hello'}])
    assert response.error is None
    request = opener.requests[0]
    assert request.full_url == 'http://host:8000/v1/chat/completions'


def test_openai_url_keeps_one_v1_suffix() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(opener, base_url='http://host:8000/v1/')
    backend.complete([{'role': 'user', 'content': 'hello'}])
    assert opener.requests[0].full_url == 'http://host:8000/v1/chat/completions'


def test_openai_url_keeps_non_v1_path_prefix() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(opener, base_url='https://gateway.example/api/v1')
    backend.complete([{'role': 'user', 'content': 'hello'}])
    assert (
        opener.requests[0].full_url
        == 'https://gateway.example/api/v1/chat/completions'
    )


def test_auth_header_is_sent_when_named_env_var_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('FOUNDATIONSKILLS_TEST_TOKEN', 'token-value')
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(opener, api_key_env='FOUNDATIONSKILLS_TEST_TOKEN')
    backend.complete([{'role': 'user', 'content': 'hello'}])
    assert opener.requests[0].get_header('Authorization') == 'Bearer token-value'


def test_auth_header_is_absent_without_api_key_env() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(opener)
    backend.complete([{'role': 'user', 'content': 'hello'}])
    assert opener.requests[0].get_header('Authorization') is None


def test_missing_named_auth_env_var_refuses_with_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('FOUNDATIONSKILLS_MISSING_TOKEN', raising=False)
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    with pytest.raises(lb.LLMOpError) as excinfo:
        _openai_backend(opener, api_key_env='FOUNDATIONSKILLS_MISSING_TOKEN')
    message = str(excinfo.value)
    assert 'missing input: env var FOUNDATIONSKILLS_MISSING_TOKEN' in message


def test_openai_config_requires_base_url() -> None:
    with pytest.raises(lb.LLMOpError) as excinfo:
        lb.OpenAICompatibleBackend({'model': 'model'})
    assert 'config.backend.base_url' in str(excinfo.value)


def test_openai_config_requires_model() -> None:
    with pytest.raises(lb.LLMOpError) as excinfo:
        lb.OpenAICompatibleBackend({'base_url': 'http://host:8000'})
    assert 'config.backend.model' in str(excinfo.value)


def test_urllib_errors_retry_with_exponential_sleep_then_succeed() -> None:
    usage = {'prompt_tokens': 11, 'completion_tokens': 3}
    opener = _CaptureOpener(
        [
            urllib.error.URLError('temporary down'),
            urllib.error.URLError('still down'),
            _FakeResponse(_api_payload(content='ready', usage=usage)),
        ]
    )
    backend = _openai_backend(opener, max_retries=2)
    delays: list[float] = []
    backend.sleep = delays.append
    response = backend.complete([{'role': 'user', 'content': 'probe'}], temperature=0.0)
    assert response.content == 'ready'
    assert response.error is None
    assert len(opener.requests) == 3
    assert delays == [1.0, 2.0]


def test_http_429_is_retryable() -> None:
    opener = _CaptureOpener(
        [_http_error(429), _FakeResponse(_api_payload(content='after-retry'))]
    )
    backend = _openai_backend(opener, max_retries=1)
    backend.sleep = lambda seconds: None
    response = backend.complete([{'role': 'user', 'content': 'probe'}])
    assert response.content == 'after-retry'
    assert len(opener.requests) == 2


def test_timeout_is_retryable() -> None:
    opener = _CaptureOpener(
        [TimeoutError('too slow'), _FakeResponse(_api_payload(content='completed'))]
    )
    backend = _openai_backend(opener, max_retries=1)
    delays: list[float] = []
    backend.sleep = delays.append
    response = backend.complete([{'role': 'user', 'content': 'probe'}])
    assert response.content == 'completed'
    assert delays == [1.0]


def test_exhausted_retries_return_error_response_without_raising() -> None:
    opener = _CaptureOpener([_http_error(503), _http_error(503)])
    backend = _openai_backend(opener, max_retries=1)
    delays: list[float] = []
    backend.sleep = delays.append
    response = backend.complete([{'role': 'user', 'content': 'probe'}])
    assert response.content is None
    assert response.error == 'http 503'
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert len(opener.requests) == 2
    assert delays == [1.0]


def test_non_retryable_http_error_is_not_retried() -> None:
    opener = _CaptureOpener([_http_error(400)])
    backend = _openai_backend(opener, max_retries=3)
    calls: list[float] = []
    backend.sleep = calls.append
    response = backend.complete([{'role': 'user', 'content': 'probe'}])
    assert response.error == 'http 400'
    assert len(opener.requests) == 1
    assert calls == []


def test_request_body_contains_required_fields_and_json_mode() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(opener)
    messages = [{'role': 'user', 'content': 'select JSON'}]
    backend.complete(
        messages,
        temperature=0.2,
        max_tokens=17,
        seed=23,
        json_mode=True,
    )
    body = json.loads(opener.requests[0].data.decode('utf-8'))
    assert body['model'] == 'test-model'
    assert body['messages'] == messages
    assert body['temperature'] == 0.2
    assert body['max_tokens'] == 17
    assert body['seed'] == 23
    assert body['response_format'] == {'type': 'json_object'}
    assert opener.timeouts == [7.5]


def test_extra_body_is_merged_without_erasing_required_fields() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload())])
    backend = _openai_backend(
        opener,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}, 'stop': ['END']},
    )
    backend.complete([{'role': 'user', 'content': 'probe'}], json_mode=True)
    body = json.loads(opener.requests[0].data.decode('utf-8'))
    assert body['model'] == 'test-model'
    assert body['chat_template_kwargs'] == {'enable_thinking': False}
    assert body['stop'] == ['END']
    assert body['response_format'] == {'type': 'json_object'}


def test_usage_and_finish_reason_are_parsed() -> None:
    usage = {'prompt_tokens': 19, 'completion_tokens': 7}
    payload = _api_payload(content='answer', finish_reason='length', usage=usage)
    opener = _CaptureOpener([_FakeResponse(payload)])
    response = _openai_backend(opener).complete([{'role': 'user', 'content': 'probe'}])
    assert response == lb.LLMResponse(
        content='answer',
        finish_reason='length',
        prompt_tokens=19,
        completion_tokens=7,
        cache_hit=False,
        error=None,
    )


def test_absent_usage_becomes_zero_not_guessed() -> None:
    opener = _CaptureOpener([_FakeResponse(_api_payload(content='plain'))])
    response = _openai_backend(opener).complete([{'role': 'user', 'content': 'probe'}])
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert response.error is None


def test_non_string_content_and_usage_are_none_or_zero_not_guessed() -> None:
    payload = json.dumps(
        {
            'choices': [{'message': {'content': ['structured']}, 'finish_reason': 4}],
            'usage': {'prompt_tokens': 'bad', 'completion_tokens': -2},
        }
    ).encode('utf-8')
    opener = _CaptureOpener([_FakeResponse(payload)])
    response = _openai_backend(opener).complete([{'role': 'user', 'content': 'probe'}])
    assert response.content is None
    assert response.finish_reason is None
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0


def test_malformed_response_becomes_an_error_response() -> None:
    opener = _CaptureOpener([_FakeResponse(b'not JSON')])
    backend = _openai_backend(opener, max_retries=2)
    backend.sleep = lambda seconds: None
    response = backend.complete([{'role': 'user', 'content': 'probe'}])
    assert response.content is None
    assert response.error is not None
    assert response.error.startswith('response JSON:')
    assert len(opener.requests) == 1


def test_unexpected_opener_error_does_not_raise_from_complete() -> None:
    opener = _CaptureOpener([RuntimeError('mock transport exploded')])
    backend = _openai_backend(opener, max_retries=2)
    backend.sleep = lambda seconds: None
    response = backend.complete([{'role': 'user', 'content': 'probe'}])
    assert response.content is None
    assert 'RuntimeError' in (response.error or '')
    assert len(opener.requests) == 1


def test_make_backend_rejects_missing_backend_with_no_fallback() -> None:
    with pytest.raises(lb.LLMOpError) as excinfo:
        lb.make_backend(None, 'llm_smoke')
    message = str(excinfo.value)
    assert message.startswith('llm_smoke: missing input: config.backend')
    assert 'openai_compatible' in message
    assert 'there is no heuristic fallback' in message


def test_make_backend_rejects_none_kind_with_no_fallback() -> None:
    with pytest.raises(lb.LLMOpError) as excinfo:
        lb.make_backend({'kind': 'none'}, 'llm_score')
    message = str(excinfo.value)
    assert 'llm_score: missing input: config.backend' in message
    assert 'no heuristic fallback' in message


def test_make_backend_rejects_unknown_kind_and_names_known_kinds(fake_factory: tuple) -> None:
    del fake_factory
    with pytest.raises(lb.LLMOpError) as excinfo:
        lb.make_backend({'kind': 'imaginary'}, 'llm_route')
    message = str(excinfo.value)
    assert "unknown LLM backend kind 'imaginary'" in message
    assert 'fake' in message
    assert 'openai_compatible' in message


def test_registered_fake_backend_is_built_and_deterministically_called(fake_factory: tuple) -> None:
    _, created = fake_factory
    backend = lb.make_backend({'kind': 'fake', 'content': 'yes'}, 'llm_test')
    response = backend.complete(
        [{'role': 'user', 'content': 'choose'}],
        temperature=0.4,
        max_tokens=9,
        seed=5,
        json_mode=True,
    )
    assert response.content == 'yes'
    assert created[-1].calls == [([{'role': 'user', 'content': 'choose'}], 0.4, 9, 5, True)]


def test_openai_factory_wraps_cache_dir_in_cached_backend(tmp_path) -> None:
    backend = lb.make_backend(
        {
            'kind': 'openai_compatible',
            'base_url': 'http://host:8000',
            'model': 'cached-model',
            'cache_dir': str(tmp_path),
        },
        'llm_cache',
    )
    assert isinstance(backend, lb.CachedBackend)
    assert backend.kind == 'openai_compatible'
    assert backend.model == 'cached-model'


def _success(content: str = 'stored') -> lb.LLMResponse:
    return lb.LLMResponse(
        content=content,
        finish_reason='stop',
        prompt_tokens=4,
        completion_tokens=5,
        cache_hit=False,
        error=None,
    )


def _failure_response(reason: str = 'request failed') -> lb.LLMResponse:
    return lb.LLMResponse(
        content=None,
        finish_reason=None,
        prompt_tokens=0,
        completion_tokens=0,
        cache_hit=False,
        error=reason,
    )


def test_cache_miss_stores_one_atomic_two_shard_file(tmp_path) -> None:
    inner = _ScriptBackend([_success()])
    cached = lb.CachedBackend(inner, str(tmp_path))
    response = cached.complete([{'role': 'user', 'content': 'cache me'}], seed=3)
    assert response.cache_hit is False
    assert len(inner.calls) == 1
    files = list(tmp_path.rglob('*.json'))
    temporary_files = list(tmp_path.rglob('*.tmp'))
    assert len(files) == 1
    assert temporary_files == []
    assert files[0].parent.name == files[0].stem[:2]
    assert len(files[0].stem) == 64
    stored = json.loads(files[0].read_text(encoding='utf-8'))
    assert stored['content'] == 'stored'
    assert stored['prompt_tokens'] == 4
    assert stored['cache_hit'] is False


def test_cache_hit_returns_stored_response_and_does_not_call_inner(tmp_path) -> None:
    inner = _ScriptBackend([_success('first and only live result')])
    cached = lb.CachedBackend(inner, str(tmp_path))
    messages = [{'role': 'user', 'content': 'same request'}]
    first = cached.complete(messages, temperature=0.0, max_tokens=8, seed=1)
    second = cached.complete(messages, temperature=0.0, max_tokens=8, seed=1)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.content == 'first and only live result'
    assert second.prompt_tokens == 4
    assert second.completion_tokens == 5
    assert len(inner.calls) == 1


def test_cache_does_not_store_error_responses(tmp_path) -> None:
    inner = _ScriptBackend([_failure_response('http 500')])
    cached = lb.CachedBackend(inner, str(tmp_path))
    response = cached.complete([{'role': 'user', 'content': 'fails'}])
    assert response.error == 'http 500'
    assert response.cache_hit is False
    assert list(tmp_path.rglob('*.json')) == []
    assert list(tmp_path.rglob('*')) == []


def test_cache_parameters_are_part_of_the_exact_key(tmp_path) -> None:
    inner = _ScriptBackend([_success('first'), _success('second')])
    cached = lb.CachedBackend(inner, str(tmp_path))
    messages = [{'role': 'user', 'content': 'same words'}]
    one = cached.complete(messages, seed=1)
    two = cached.complete(messages, seed=2)
    repeat = cached.complete(messages, seed=1)
    assert one.content == 'first'
    assert two.content == 'second'
    assert repeat.content == 'first'
    assert repeat.cache_hit is True
    assert len(inner.calls) == 2


def test_cache_none_disables_storage(tmp_path) -> None:
    inner = _ScriptBackend([_success('live')])
    cached = lb.CachedBackend(inner, None)
    cached.complete([{'role': 'user', 'content': 'uncached'}])
    assert inner.calls
    assert list(tmp_path.rglob('*')) == []


def test_malformed_cache_entry_is_a_miss_not_a_fake_hit(tmp_path) -> None:
    inner = _ScriptBackend([_success('live replacement')])
    cached = lb.CachedBackend(inner, str(tmp_path))
    key = cached._key([{'role': 'user', 'content': 'repair'}], temperature=0.0, max_tokens=1024,
                      seed=0, json_mode=False)
    entry = cached._path(key)  # corrupt the REAL entry this request will look up
    entry.parent.mkdir(parents=True)
    entry.write_text('not json', encoding='utf-8')
    response = cached.complete([{'role': 'user', 'content': 'repair'}])
    assert response.cache_hit is False
    assert response.content == 'live replacement'
    assert len(inner.calls) == 1


def test_map_ordered_preserves_order_despite_completion_order() -> None:
    def transform(value: int) -> int:
        if value == 0:
            time.sleep(0.03)
        return value * 10

    result = list(lb.map_ordered(transform, range(8), workers=3, window=2))
    assert result == [0, 10, 20, 30, 40, 50, 60, 70]


def test_map_ordered_streams_a_generator_and_bounds_the_window() -> None:
    pulled = 0

    def source() -> object:
        nonlocal pulled
        for value in range(10):
            pulled += 1
            yield value

    mapped = lb.map_ordered(lambda value: value, source(), workers=3, window=4)
    assert next(mapped) == 0
    assert pulled == 4
    assert next(mapped) == 1
    assert pulled == 5
    mapped.close()
    assert pulled <= 5


def test_map_ordered_with_one_worker_runs_inline() -> None:
    calls: list[int] = []

    def transform(value: int) -> int:
        calls.append(value)
        return -value

    mapped = lb.map_ordered(transform, iter([1, 2, 3]), workers=1, window=10)
    assert calls == []
    assert list(mapped) == [-1, -2, -3]
    assert calls == [1, 2, 3]


def test_map_ordered_nonpositive_window_still_bounded_and_ordered() -> None:
    result = list(lb.map_ordered(lambda value: value * 2, range(6), workers=2, window=0))
    assert result == [0, 2, 4, 6, 8, 10]


def test_prompt_hash_hashes_directly_joined_parts() -> None:
    expected = hashlib.sha256(b'prefix::suffix').hexdigest()[:16]
    assert lb.prompt_hash('prefix', '::', 'suffix') == expected


def test_parse_json_object_accepts_a_json_fence() -> None:
    text = 'Use this result:\n```json\n{"label": "yes", "score": 0.4}\n```'
    assert lb.parse_json_object(text) == {'label': 'yes', 'score': 0.4}


def test_parse_json_object_strips_leading_prose_and_trailing_text() -> None:
    text = 'The JSON is {"answer": "shown } here", "ok": true}. Done.'
    assert lb.parse_json_object(text) == {'answer': 'shown } here', 'ok': True}


def test_parse_json_object_returns_none_for_garbage() -> None:
    assert lb.parse_json_object('no object here') is None
    assert lb.parse_json_object('{"answer": nope}') is None
    assert lb.parse_json_object('[1, 2, 3]') is None
    assert lb.parse_json_object(None) is None


def test_budget_take_reports_exact_exhaustion() -> None:
    calls = lb.budget(2)
    assert not calls.exhausted
    assert calls.take() is True
    assert calls.take() is True
    assert calls.used == 2
    assert calls.exhausted
    assert calls.take() is False
    assert calls.used == 2


def test_budget_none_is_unlimited() -> None:
    calls = lb.budget(None)
    for _ in range(500):
        assert calls.take() is True
    assert calls.used == 500
    assert not calls.exhausted


def test_budget_is_thread_safe_under_contention() -> None:
    calls = lb.budget(13)
    barrier = threading.Barrier(8)

    def worker() -> int:
        barrier.wait()
        return sum(1 for _ in range(5) if calls.take())

    with ThreadPoolExecutor(max_workers=8) as pool:
        successes = list(pool.map(lambda _: worker(), range(8)))
    assert sum(successes) == 13
    assert calls.used == 13
    assert calls.exhausted


def test_invalid_budget_refuses_configuration() -> None:
    with pytest.raises(lb.LLMOpError, match='max_calls'):
        lb.budget(-1)
