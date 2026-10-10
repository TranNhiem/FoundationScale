"""Tests for ``foundationscale.agentic_rl.gateway.session`` (binding API)."""

from __future__ import annotations

import math
import threading
from dataclasses import replace

import pytest

from foundationscale.agentic_rl.gateway.session import (
    EngineCall,
    Mode,
    SamplingContract,
    Session,
    SessionClosed,
    SessionNotFound,
    SessionRefusal,
    SessionStore,
    SessionStoreFull,
    api_format_for,
    build_episode,
)
from foundationscale.agentic_rl.gateway.wire import CanonicalToolCall, SamplingRequest, WireFormat
from foundationscale.agentic_rl.trajectory_v2 import (
    ApiFormat,
    EnvKind,
    EnvMeta,
    EpisodeStatus,
    Fidelity,
    FidelityStatus,
    HarnessMeta,
    ModelMeta,
    ParseVerdict,
    RewardEvent,
    RewardScope,
    Termination,
    Timings,
    validate_episode,
)


def _request(**kwargs: object) -> SamplingRequest:
    """Build a SamplingRequest with sensible defaults, overriding via kwargs."""
    base: dict[str, object] = {"max_tokens": 128}
    base.update(kwargs)
    return SamplingRequest(**base)  # type: ignore[arg-type]


def _call(
    generation_id: str = "gen-0",
    *,
    api_format: WireFormat = WireFormat.CHAT_COMPLETIONS,
    prompt_ids: tuple[int, ...] = (1, 2, 3),
    output_ids: tuple[int, ...] = (4, 5),
    logprobs: tuple[float, ...] = (-0.1, -0.2),
    policy_version: int = 7,
    finish_reason: str | None = "stop",
    sampling_requested: dict[str, object] | None = None,
    sampling_overrides: dict[str, object] | None = None,
    tool_calls: tuple[CanonicalToolCall, ...] = (),
) -> EngineCall:
    """Build an EngineCall with small hand-made token data."""
    return EngineCall(
        generation_id=generation_id,
        api_format=api_format,
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        logprobs=logprobs,
        policy_version=policy_version,
        finish_reason=finish_reason,
        sampling_requested=sampling_requested if sampling_requested is not None else {},
        sampling_overrides=sampling_overrides if sampling_overrides is not None else {},
        tool_calls=tool_calls,
    )


def _canonical(call_id: str, *, origin: str = "model") -> CanonicalToolCall:
    """Build a CanonicalToolCall with the given call id and origin."""
    return CanonicalToolCall(
        call_id=call_id,
        name="tool",
        arguments_raw="{}",
        call_id_origin=origin,
    )


def _reward(value: float | None = 1.0) -> RewardEvent:
    """Build an EPISODE-scope reward event."""
    return RewardEvent(
        source="verifier",
        verifier_id="v",
        verifier_version="1",
        value=value,
        scope=RewardScope.EPISODE,
        at_ts=1.5,
    )


def _model() -> ModelMeta:
    """Build ModelMeta for episode assembly."""
    return ModelMeta(name="m", version=3, checkpoint_sha="abc")


def _harness() -> HarnessMeta:
    """Build HarnessMeta for episode assembly."""
    return HarnessMeta(name="h", version="0.1")


def _env() -> EnvMeta:
    """Build EnvMeta for episode assembly."""
    return EnvMeta(kind=EnvKind.FS_LOCAL, task_id="task-1")


def _store(max_sessions: int = 4, max_calls_per_session: int = 4) -> SessionStore:
    """Build a SessionStore with the given limits."""
    return SessionStore(max_sessions=max_sessions, max_calls_per_session=max_calls_per_session)


def _session(store: SessionStore, session_id: str = "s-1") -> object:
    """Create a TRAIN session with the given id."""
    return store.create(mode=Mode.TRAIN, group_id="g-1", created_at=1.0, session_id=session_id)


def test_train_forces_engine_params_and_records_only_client_sent_overrides() -> None:
    """TRAIN forces top_p/top_k/min_p/n/temperature and overrides only client-sent fields."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.0, max_tokens_cap=None)
    req = _request(temperature=0.9, top_p=0.7, max_tokens=64, seed=11)
    engine, overrides = contract.enforce(req)
    assert engine["temperature"] == 0.0
    assert engine["top_p"] == 1.0
    assert engine["top_k"] == -1
    assert engine["min_p"] == 0.0
    assert engine["n"] == 1
    assert engine["max_tokens"] == 64
    assert engine["seed"] == 11
    assert set(overrides) == {"temperature", "top_p"}
    assert overrides["temperature"] == {"requested": 0.9, "applied": 0.0}
    assert overrides["top_p"] == {"requested": 0.7, "applied": 1.0}


def test_train_records_no_override_for_unsent_fields() -> None:
    """TRAIN records no override entry for fields the client never sent."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.3)
    engine, overrides = contract.enforce(_request(max_tokens=32))
    assert engine["temperature"] == 0.3
    assert overrides == {}


def test_eval_passes_client_values_through() -> None:
    """EVAL keeps client temperature/top_p/top_k/n and records no overrides when equal."""
    contract = SamplingContract(mode=Mode.EVAL, temperature=0.5)
    req = _request(temperature=0.2, top_p=0.8, top_k=5, n=2, max_tokens=16)
    engine, overrides = contract.enforce(req)
    assert engine["temperature"] == 0.2
    assert engine["top_p"] == 0.8
    assert engine["top_k"] == 5
    assert engine["n"] == 2
    assert engine["max_tokens"] == 16
    assert overrides == {}


def test_eval_defaults_when_client_omits_fields() -> None:
    """EVAL falls back to contract temperature and neutral defaults when fields are absent."""
    contract = SamplingContract(mode=Mode.EVAL, temperature=0.5)
    engine, overrides = contract.enforce(_request(max_tokens=16))
    assert engine["temperature"] == 0.5
    assert engine["top_p"] == 1.0
    assert engine["top_k"] == -1
    assert engine["n"] == 1
    assert overrides == {}


def test_max_tokens_capped_in_both_modes() -> None:
    """max_tokens is min(client, cap) in TRAIN and EVAL and never raised above the client value."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.0, max_tokens_cap=10)
    engine, _ = contract.enforce(_request(max_tokens=64))
    assert engine["max_tokens"] == 10
    engine, _ = contract.enforce(_request(max_tokens=4))
    assert engine["max_tokens"] == 4
    eval_contract = SamplingContract(mode=Mode.EVAL, temperature=0.0, max_tokens_cap=10)
    engine, _ = eval_contract.enforce(_request(max_tokens=64))
    assert engine["max_tokens"] == 10


def test_seed_omitted_when_none() -> None:
    """seed is absent from engine_params when the client sent no seed."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.0)
    engine, _ = contract.enforce(_request(max_tokens=8))
    assert "seed" not in engine
    engine, _ = contract.enforce(_request(max_tokens=8, seed=3))
    assert engine["seed"] == 3


def test_to_generation_maps_tool_calls_to_engine_parser_verdicts() -> None:
    """Engine tool calls become ToolCallV2 with ENGINE_PARSER verdicts and kept origins."""
    call = _call(
        tool_calls=(_canonical("c1", origin="model"), _canonical("c2", origin="synthesized")),
    )
    gen = call.to_generation()
    assert len(gen.tool_calls) == 2
    for tc, expected_id, expected_origin in zip(
        gen.tool_calls, ("c1", "c2"), ("model", "synthesized"), strict=True
    ):
        assert tc.call_id == expected_id
        assert tc.parse_verdict is ParseVerdict.ENGINE_PARSER
        assert tc.call_id_origin == expected_origin


def test_to_generation_carries_token_data() -> None:
    """to_generation preserves prompt/output ids, logprobs, policy version and sampling maps."""
    call = _call(
        prompt_ids=(1, 2),
        output_ids=(3, 4, 5),
        logprobs=(-0.1, -0.2, -0.3),
        policy_version=9,
        sampling_requested={"temperature": 0.9},
        sampling_overrides={"temperature": {"requested": 0.9, "applied": 0.0}},
    )
    gen = call.to_generation()
    assert gen.prompt_ids == (1, 2)
    assert gen.output_ids == (3, 4, 5)
    assert gen.logprobs == (-0.1, -0.2, -0.3)
    assert gen.policy_version == 9
    assert dict(gen.sampling_requested) == {"temperature": 0.9}
    assert dict(gen.sampling_overrides)["temperature"]["applied"] == 0.0


def test_store_create_get_and_by_api_key() -> None:
    """create returns a session retrievable by id and by its api key."""
    store = _store()
    session = _session(store)
    assert store.get(session.session_id) is session
    assert store.by_api_key(session.api_key) is session
    assert session.task_prompt_ids is None
    assert len(store) == 1


def test_store_duplicate_session_id_refused() -> None:
    """Creating a session with an existing id is refused."""
    store = _store()
    _session(store, "dup")
    with pytest.raises(SessionRefusal):
        _session(store, "dup")


def test_store_capacity_refused() -> None:
    """Creating beyond max_sessions raises SessionStoreFull naming both counts."""
    store = _store(max_sessions=2)
    _session(store, "a")
    _session(store, "b")
    with pytest.raises(SessionStoreFull):
        _session(store, "c")


def test_record_call_sets_task_prompt_ids_from_first_call() -> None:
    """The first recorded call fixes the session's task_prompt_ids."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call("g1", prompt_ids=(9, 8, 7)))
    assert session.task_prompt_ids == (9, 8, 7)
    store.record_call(session.session_id, _call("g2", prompt_ids=(9, 8, 7, 6)))
    assert session.task_prompt_ids == (9, 8, 7)


def test_record_call_capacity_refused() -> None:
    """Recording beyond max_calls_per_session raises SessionStoreFull naming both counts."""
    store = _store(max_calls_per_session=2)
    session = _session(store)
    store.record_call(session.session_id, _call("g1"))
    store.record_call(session.session_id, _call("g2"))
    with pytest.raises(SessionStoreFull):
        store.record_call(session.session_id, _call("g3"))


def test_record_call_duplicate_generation_id_refused() -> None:
    """A duplicate generation_id within a session is refused."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call("same"))
    with pytest.raises(SessionRefusal):
        store.record_call(session.session_id, _call("same"))


def test_record_call_after_close_refused() -> None:
    """Recording a call on a closed session raises SessionClosed."""
    store = _store()
    session = _session(store)
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    with pytest.raises(SessionClosed):
        store.record_call(session.session_id, _call("late"))


def test_add_reward_allowed_after_close() -> None:
    """Delayed rewards may be added to a closed session."""
    store = _store()
    session = _session(store)
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    store.add_reward(session.session_id, _reward(1.0))
    assert len(session.reward_events) == 1


def test_close_twice_refused() -> None:
    """Closing an already closed session raises SessionClosed."""
    store = _store()
    session = _session(store)
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    with pytest.raises(SessionClosed):
        store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)


def test_api_key_revoked_after_close() -> None:
    """by_api_key raises SessionNotFound for a key revoked by close."""
    store = _store()
    session = _session(store)
    key = session.api_key
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    with pytest.raises(SessionNotFound):
        store.by_api_key(key)


def test_remove_frees_capacity() -> None:
    """remove drops the session and frees a capacity slot."""
    store = _store(max_sessions=1)
    session = _session(store, "only")
    with pytest.raises(SessionStoreFull):
        _session(store, "other")
    store.remove(session.session_id)
    assert len(store) == 0
    _session(store, "other")
    with pytest.raises(SessionNotFound):
        store.get("only")


def test_concurrent_record_call_records_each_call_once() -> None:
    """Eight threads recording calls concurrently each land exactly once."""
    store = _store(max_calls_per_session=8)
    session = _session(store)
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        barrier.wait()
        store.record_call(session.session_id, _call(f"gen-{index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [c.generation_id for c in session.calls]
    assert sorted(ids) == sorted(f"gen-{i}" for i in range(8))
    assert len(set(ids)) == 8


def test_build_episode_refuses_open_session() -> None:
    """build_episode refuses a session that is not closed."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call())
    with pytest.raises(SessionRefusal):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_zero_calls() -> None:
    """build_episode refuses a closed session with zero recorded calls."""
    store = _store()
    session = _session(store)
    store.close(session.session_id, status=EpisodeStatus.INFRA_ERROR, termination=Termination.ERROR)
    with pytest.raises(SessionRefusal):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_mixed_api_formats() -> None:
    """build_episode refuses a session whose calls use mixed wire formats."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call("g1", api_format=WireFormat.CHAT_COMPLETIONS))
    store.record_call(session.session_id, _call("g2", api_format=WireFormat.RESPONSES))
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    with pytest.raises(SessionRefusal):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_merges_extending_prompt_into_one_segment() -> None:
    """A second call whose prompt extends the first merges into a single segment."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call("g1", prompt_ids=(1, 2), output_ids=(3, 4)))
    store.record_call(
        session.session_id,
        _call("g2", prompt_ids=(1, 2, 3, 4, 5), output_ids=(6,), logprobs=(-0.3,)),
    )
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    ep = build_episode(session, model=_model(), harness=_harness(), env=_env())
    assert len(ep.segments) == 1
    seg = ep.segments[0]
    assert seg.prompt_ids == (1, 2)
    assert seg.response_ids == (3, 4, 5, 6)
    assert seg.loss_mask == (1, 1, 0, 1)
    assert [span[2] for span in seg.generation_spans] == ["g1", "g2"]


def test_build_episode_rewritten_prompt_opens_second_segment() -> None:
    """A rewritten prompt that does not extend the prior stream opens a second segment."""
    store = _store()
    session = _session(store)
    store.record_call(
        session.session_id, _call("g1", prompt_ids=(1, 2), output_ids=(3,), logprobs=(-0.1,))
    )
    store.record_call(
        session.session_id, _call("g2", prompt_ids=(9, 9), output_ids=(4,), logprobs=(-0.2,))
    )
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    ep = build_episode(session, model=_model(), harness=_harness(), env=_env())
    assert len(ep.segments) == 2
    assert [seg.segment_index for seg in ep.segments] == [0, 1]
    assert ep.segments[1].prompt_ids == (9, 9)


def test_build_episode_carries_rewards_and_gateway_metadata() -> None:
    """Reward events and gateway metadata reach the episode and it passes validate_episode."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call("g1", api_format=WireFormat.RESPONSES))
    store.add_reward(session.session_id, _reward(1.0))
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    ep = build_episode(
        session,
        model=_model(),
        harness=_harness(),
        env=_env(),
        timings=Timings(run_ms=2.0),
        fidelity=Fidelity(FidelityStatus.VERIFIED),
    )
    assert ep.gateway.api_format is ApiFormat.RESPONSES
    assert ep.gateway.session_id == session.session_id
    assert ep.task_prompt_ids == session.task_prompt_ids
    assert len(ep.reward_events) == 1
    assert ep.reward_events[0].value == 1.0
    validate_episode(ep)


def test_freeze_mapping_refuses_non_mapping_requested() -> None:
    """EngineCall.sampling_requested must be a Mapping."""
    with pytest.raises(SessionRefusal, match="sampling_requested to be a Mapping"):
        _call(sampling_requested=[1, 2])  # type: ignore[arg-type]


def test_freeze_mapping_refuses_non_mapping_overrides() -> None:
    """EngineCall.sampling_overrides must be a Mapping."""
    with pytest.raises(SessionRefusal, match="sampling_overrides to be a Mapping"):
        _call(sampling_overrides=3)  # type: ignore[arg-type]


def test_as_tuple_refuses_non_sequence_prompt_ids() -> None:
    """EngineCall.prompt_ids must be a tuple, list, set or frozenset."""
    with pytest.raises(SessionRefusal, match="prompt_ids to be a sequence"):
        _call(prompt_ids=5)  # type: ignore[arg-type]


def test_as_tuple_refuses_non_sequence_output_ids() -> None:
    """EngineCall.output_ids must be a sequence."""
    with pytest.raises(SessionRefusal, match="output_ids to be a sequence"):
        _call(output_ids="ab")  # type: ignore[arg-type]


def test_as_tuple_refuses_non_sequence_logprobs() -> None:
    """EngineCall.logprobs must be a sequence."""
    with pytest.raises(SessionRefusal, match="logprobs to be a sequence"):
        _call(logprobs=1.0)  # type: ignore[arg-type]


def test_as_tuple_refuses_non_sequence_tool_calls() -> None:
    """EngineCall.tool_calls must be a sequence."""
    with pytest.raises(SessionRefusal, match="tool_calls to be a sequence"):
        _call(tool_calls="x")  # type: ignore[arg-type]


def test_as_tuple_accepts_list_and_frozenset() -> None:
    """_as_tuple normalizes lists and frozensets to tuples."""
    call = _call(prompt_ids=[1, 2], output_ids=frozenset({3}), logprobs=[-0.5])
    assert call.prompt_ids == (1, 2)
    assert call.output_ids == (3,)
    assert call.logprobs == (-0.5,)


def test_sampling_contract_refuses_bad_mode() -> None:
    """SamplingContract.mode must be a Mode."""
    with pytest.raises(SessionRefusal, match="SamplingContract.mode to be a Mode"):
        SamplingContract(mode="train", temperature=0.5)  # type: ignore[arg-type]


def test_sampling_contract_refuses_negative_temperature() -> None:
    """SamplingContract.temperature must be finite and >= 0."""
    with pytest.raises(SessionRefusal, match="temperature to be a finite float >= 0"):
        SamplingContract(mode=Mode.TRAIN, temperature=-0.1)


def test_sampling_contract_refuses_nan_temperature() -> None:
    """SamplingContract.temperature must not be NaN or infinite."""
    with pytest.raises(SessionRefusal, match="temperature to be a finite float >= 0"):
        SamplingContract(mode=Mode.TRAIN, temperature=math.nan)


def test_sampling_contract_refuses_bool_temperature() -> None:
    """SamplingContract.temperature must not be a bool."""
    with pytest.raises(SessionRefusal, match="temperature to be a finite float >= 0"):
        SamplingContract(mode=Mode.TRAIN, temperature=True)  # type: ignore[arg-type]


def test_sampling_contract_refuses_zero_max_tokens_cap() -> None:
    """SamplingContract.max_tokens_cap must be None or an int >= 1."""
    with pytest.raises(SessionRefusal, match="max_tokens_cap to be None or an int >= 1"):
        SamplingContract(mode=Mode.TRAIN, temperature=0.5, max_tokens_cap=0)


def test_sampling_contract_refuses_bool_max_tokens_cap() -> None:
    """SamplingContract.max_tokens_cap must not be a bool."""
    with pytest.raises(SessionRefusal, match="max_tokens_cap to be None or an int >= 1"):
        SamplingContract(mode=Mode.TRAIN, temperature=0.5, max_tokens_cap=True)  # type: ignore[arg-type]


def test_sampling_contract_enforce_refuses_non_request() -> None:
    """SamplingContract.enforce requires a SamplingRequest."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.5)
    with pytest.raises(SessionRefusal, match="enforce.req to be a SamplingRequest"):
        contract.enforce({"max_tokens": 1})  # type: ignore[arg-type]


def test_engine_call_refuses_empty_generation_id() -> None:
    """EngineCall.generation_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="generation_id to be a non-empty str"):
        _call("")


def test_engine_call_refuses_non_str_generation_id() -> None:
    """EngineCall.generation_id must be a str."""
    with pytest.raises(SessionRefusal, match="generation_id to be a non-empty str"):
        _call(7)  # type: ignore[arg-type]


def test_engine_call_refuses_bad_api_format() -> None:
    """EngineCall.api_format must be a WireFormat."""
    with pytest.raises(SessionRefusal, match="api_format to be a WireFormat"):
        _call(api_format="chat_completions")  # type: ignore[arg-type]


def test_engine_call_refuses_empty_prompt_ids() -> None:
    """EngineCall.prompt_ids must be non-empty."""
    with pytest.raises(SessionRefusal, match="prompt_ids to be a non-empty tuple"):
        _call(prompt_ids=())


def test_engine_call_refuses_negative_prompt_id() -> None:
    """EngineCall.prompt_ids entries must be ints >= 0."""
    with pytest.raises(SessionRefusal, match="prompt_ids: 1 of 3 tokens to be ints >= 0"):
        _call(prompt_ids=(1, -2, 3))


def test_engine_call_refuses_bool_prompt_id() -> None:
    """EngineCall.prompt_ids entries must not be bools."""
    with pytest.raises(SessionRefusal, match="prompt_ids: 1 of 2 tokens to be ints >= 0"):
        _call(prompt_ids=(1, True))


def test_engine_call_refuses_empty_output_ids() -> None:
    """EngineCall.output_ids must be non-empty."""
    with pytest.raises(SessionRefusal, match="output_ids to be a non-empty tuple"):
        _call(output_ids=(), logprobs=())


def test_engine_call_refuses_negative_output_id() -> None:
    """EngineCall.output_ids entries must be ints >= 0."""
    with pytest.raises(SessionRefusal, match="output_ids: 2 of 2 tokens to be ints >= 0"):
        _call(output_ids=(-1, -2))


def test_engine_call_refuses_logprobs_length_mismatch() -> None:
    """len(logprobs) must equal len(output_ids)."""
    with pytest.raises(SessionRefusal, match="logprobs: 1 of 2 entries"):
        _call(output_ids=(4, 5), logprobs=(-0.1,))


def test_engine_call_refuses_non_finite_logprob() -> None:
    """EngineCall.logprobs entries must be finite floats."""
    with pytest.raises(SessionRefusal, match="logprobs: 1 of 2 entries to be finite floats"):
        _call(logprobs=(-0.1, math.inf))


def test_engine_call_refuses_bool_policy_version() -> None:
    """EngineCall.policy_version must be an int >= 0, never a bool."""
    with pytest.raises(SessionRefusal, match="policy_version to be an int >= 0"):
        _call(policy_version=True)  # type: ignore[arg-type]


def test_engine_call_refuses_negative_policy_version() -> None:
    """EngineCall.policy_version must be >= 0."""
    with pytest.raises(SessionRefusal, match="policy_version to be an int >= 0"):
        _call(policy_version=-1)


def test_engine_call_refuses_non_str_finish_reason() -> None:
    """EngineCall.finish_reason must be None or str."""
    with pytest.raises(SessionRefusal, match="finish_reason to be None or str"):
        _call(finish_reason=5)  # type: ignore[arg-type]


def test_engine_call_refuses_non_canonical_tool_call() -> None:
    """EngineCall.tool_calls entries must be CanonicalToolCall."""
    with pytest.raises(SessionRefusal, match="tool_calls: 1 of 2 entries to be CanonicalToolCall"):
        _call(tool_calls=(_canonical("c1"), "nope"))  # type: ignore[arg-type]


def test_engine_call_refuses_negative_latency_ms() -> None:
    """EngineCall.latency_ms must be None or a finite float >= 0."""
    with pytest.raises(SessionRefusal, match="latency_ms to be None or a finite float >= 0"):
        EngineCall(
            generation_id="g",
            api_format=WireFormat.CHAT_COMPLETIONS,
            prompt_ids=(1,),
            output_ids=(2,),
            logprobs=(-0.1,),
            policy_version=1,
            finish_reason=None,
            sampling_requested={},
            sampling_overrides={},
            latency_ms=-1.0,
        )


def test_engine_call_refuses_nan_latency_ms() -> None:
    """EngineCall.latency_ms must be finite."""
    with pytest.raises(SessionRefusal, match="latency_ms to be None or a finite float >= 0"):
        EngineCall(
            generation_id="g",
            api_format=WireFormat.CHAT_COMPLETIONS,
            prompt_ids=(1,),
            output_ids=(2,),
            logprobs=(-0.1,),
            policy_version=1,
            finish_reason=None,
            sampling_requested={},
            sampling_overrides={},
            latency_ms=math.nan,
        )


def test_engine_call_accepts_valid_latency_ms() -> None:
    """EngineCall.latency_ms accepts a finite non-negative float."""
    call = EngineCall(
        generation_id="g",
        api_format=WireFormat.CHAT_COMPLETIONS,
        prompt_ids=(1,),
        output_ids=(2,),
        logprobs=(-0.1,),
        policy_version=1,
        finish_reason=None,
        sampling_requested={},
        sampling_overrides={},
        latency_ms=12.5,
    )
    assert call.latency_ms == 12.5


def test_store_refuses_bool_max_sessions() -> None:
    """SessionStore.max_sessions must be an int >= 1, never a bool."""
    with pytest.raises(SessionRefusal, match="max_sessions to be an int >= 1"):
        SessionStore(max_sessions=True, max_calls_per_session=1)  # type: ignore[arg-type]


def test_store_refuses_zero_max_calls_per_session() -> None:
    """SessionStore.max_calls_per_session must be an int >= 1."""
    with pytest.raises(SessionRefusal, match="max_calls_per_session to be an int >= 1"):
        SessionStore(max_sessions=1, max_calls_per_session=0)


def test_store_create_refuses_bad_mode() -> None:
    """SessionStore.create.mode must be a Mode."""
    with pytest.raises(SessionRefusal, match="create.mode to be a Mode"):
        _store().create(mode="train", group_id="g", created_at=1.0)  # type: ignore[arg-type]


def test_store_create_refuses_empty_group_id() -> None:
    """SessionStore.create.group_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="create.group_id to be a non-empty str"):
        _store().create(mode=Mode.TRAIN, group_id="", created_at=1.0)


def test_store_create_refuses_negative_created_at() -> None:
    """SessionStore.create.created_at must be a finite float >= 0."""
    with pytest.raises(SessionRefusal, match="create.created_at to be a finite float >= 0"):
        _store().create(mode=Mode.TRAIN, group_id="g", created_at=-1.0)


def test_store_create_refuses_empty_session_id() -> None:
    """SessionStore.create.session_id must be None or a non-empty str."""
    with pytest.raises(SessionRefusal, match="create.session_id to be None or a non-empty str"):
        _store().create(mode=Mode.TRAIN, group_id="g", created_at=1.0, session_id="")


def test_store_get_refuses_empty_session_id() -> None:
    """SessionStore.get.session_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="get.session_id to be a non-empty str"):
        _store().get("")


def test_store_get_missing_session_not_found() -> None:
    """SessionStore.get raises SessionNotFound for an unknown id."""
    with pytest.raises(SessionNotFound, match="to exist"):
        _store().get("nope")


def test_store_by_api_key_refuses_empty_key() -> None:
    """SessionStore.by_api_key.api_key must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="by_api_key.api_key to be a non-empty str"):
        _store().by_api_key("")


def test_store_by_api_key_unknown_key_not_found() -> None:
    """SessionStore.by_api_key raises SessionNotFound for an unknown key."""
    with pytest.raises(SessionNotFound, match="to resolve to a live session"):
        _store().by_api_key("fs-sess-unknown")


def test_store_by_api_key_dangling_key_not_found() -> None:
    """SessionStore.by_api_key raises SessionNotFound when the key maps to a missing session."""
    store = _store()
    session = _session(store)
    store._by_api_key[session.api_key] = "ghost"
    with pytest.raises(SessionNotFound, match="to resolve to a live session"):
        store.by_api_key(session.api_key)


def test_record_call_refuses_empty_session_id() -> None:
    """SessionStore.record_call.session_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="record_call.session_id to be a non-empty str"):
        _store().record_call("", _call())


def test_record_call_refuses_non_engine_call() -> None:
    """SessionStore.record_call.call must be an EngineCall."""
    with pytest.raises(SessionRefusal, match="record_call.call to be an EngineCall"):
        _store().record_call("s", "nope")  # type: ignore[arg-type]


def test_record_call_missing_session_not_found() -> None:
    """SessionStore.record_call raises SessionNotFound for an unknown session."""
    with pytest.raises(SessionNotFound, match="to exist"):
        _store().record_call("ghost", _call())


def test_add_reward_refuses_empty_session_id() -> None:
    """SessionStore.add_reward.session_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="add_reward.session_id to be a non-empty str"):
        _store().add_reward("", _reward())


def test_add_reward_refuses_non_reward_event() -> None:
    """SessionStore.add_reward.event must be a RewardEvent."""
    with pytest.raises(SessionRefusal, match="add_reward.event to be a RewardEvent"):
        _store().add_reward("s", 1.0)  # type: ignore[arg-type]


def test_add_reward_missing_session_not_found() -> None:
    """SessionStore.add_reward raises SessionNotFound for an unknown session."""
    with pytest.raises(SessionNotFound, match="to exist"):
        _store().add_reward("ghost", _reward())


def test_close_refuses_empty_session_id() -> None:
    """SessionStore.close.session_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="close.session_id to be a non-empty str"):
        _store().close("", status=EpisodeStatus.OK, termination=Termination.STOP)


def test_close_refuses_bad_status() -> None:
    """SessionStore.close.status must be an EpisodeStatus."""
    with pytest.raises(SessionRefusal, match="close.status to be an EpisodeStatus"):
        _store().close("s", status="ok", termination=Termination.STOP)  # type: ignore[arg-type]


def test_close_refuses_bad_termination() -> None:
    """SessionStore.close.termination must be a Termination."""
    with pytest.raises(SessionRefusal, match="close.termination to be a Termination"):
        _store().close("s", status=EpisodeStatus.OK, termination="stop")  # type: ignore[arg-type]


def test_close_missing_session_not_found() -> None:
    """SessionStore.close raises SessionNotFound for an unknown session."""
    with pytest.raises(SessionNotFound, match="to exist"):
        _store().close("ghost", status=EpisodeStatus.OK, termination=Termination.STOP)


def test_remove_refuses_empty_session_id() -> None:
    """SessionStore.remove.session_id must be a non-empty str."""
    with pytest.raises(SessionRefusal, match="remove.session_id to be a non-empty str"):
        _store().remove("")


def test_remove_missing_session_not_found() -> None:
    """SessionStore.remove raises SessionNotFound for an unknown session."""
    with pytest.raises(SessionNotFound, match="to exist"):
        _store().remove("ghost")


def test_api_format_for_refuses_non_wire_format() -> None:
    """api_format_for.fmt must be a WireFormat."""
    with pytest.raises(SessionRefusal, match="api_format_for.fmt to be a WireFormat"):
        api_format_for("chat_completions")  # type: ignore[arg-type]


def test_api_format_for_maps_all_wire_formats() -> None:
    """api_format_for maps every WireFormat value name to its ApiFormat."""
    pairs = (
        (WireFormat.CHAT_COMPLETIONS, ApiFormat.CHAT_COMPLETIONS),
        (WireFormat.RESPONSES, ApiFormat.RESPONSES),
        (WireFormat.ANTHROPIC_MESSAGES, ApiFormat.ANTHROPIC_MESSAGES),
        (WireFormat.GEMINI, ApiFormat.GEMINI),
    )
    for fmt, expected in pairs:
        assert api_format_for(fmt) is expected


def test_build_episode_refuses_non_session() -> None:
    """build_episode.session must be a Session."""
    with pytest.raises(SessionRefusal, match="build_episode.session to be a Session"):
        build_episode("s", model=_model(), harness=_harness(), env=_env())  # type: ignore[arg-type]


def test_build_episode_refuses_session_without_status_after_close() -> None:
    """build_episode requires both status and termination after close."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call())
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    session.status = None
    with pytest.raises(SessionRefusal, match="to carry both status and termination"):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_session_without_termination_after_close() -> None:
    """build_episode requires a termination alongside the status after close."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call())
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    session.termination = None
    with pytest.raises(SessionRefusal, match="to carry both status and termination"):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_unset_task_prompt_ids() -> None:
    """build_episode refuses a session whose task_prompt_ids was never fixed."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call())
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    session.task_prompt_ids = None
    with pytest.raises(SessionRefusal, match="task_prompt_ids to be fixed by its first call"):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_manually_open_session_with_calls() -> None:
    """build_episode refuses an open session even when calls are recorded."""
    session = Session(
        session_id="s",
        api_key="k",
        mode=Mode.TRAIN,
        created_at=1.0,
        group_id="g",
        task_prompt_ids=(1,),
        calls=[_call()],
    )
    with pytest.raises(SessionRefusal, match="to be closed before episode assembly"):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_manually_closed_zero_call_session() -> None:
    """build_episode refuses a manually closed session with zero calls."""
    session = Session(
        session_id="s",
        api_key="k",
        mode=Mode.TRAIN,
        created_at=1.0,
        group_id="g",
        task_prompt_ids=None,
        closed=True,
        status=EpisodeStatus.INFRA_ERROR,
        termination=Termination.ERROR,
    )
    with pytest.raises(SessionRefusal, match="at least 1 of 1 engine calls"):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_build_episode_refuses_mixed_api_formats_count_message() -> None:
    """build_episode names both the format count and the call count for mixed formats."""
    store = _store()
    session = _session(store)
    store.record_call(session.session_id, _call("g1", api_format=WireFormat.CHAT_COMPLETIONS))
    store.record_call(session.session_id, _call("g2", api_format=WireFormat.GEMINI))
    store.close(session.session_id, status=EpisodeStatus.OK, termination=Termination.STOP)
    with pytest.raises(SessionRefusal, match="exactly 1 of 2 api formats across its 2 calls"):
        build_episode(session, model=_model(), harness=_harness(), env=_env())


def test_engine_call_accepts_none_finish_reason_and_latency() -> None:
    """EngineCall accepts finish_reason=None and latency_ms=None."""
    call = _call(finish_reason=None)
    assert call.finish_reason is None
    assert call.latency_ms is None


def test_sampling_contract_cap_none_keeps_client_max_tokens() -> None:
    """A None max_tokens_cap keeps the client max_tokens and a None client max yields the cap."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.0, max_tokens_cap=None)
    engine, _ = contract.enforce(_request(max_tokens=64))
    assert engine["max_tokens"] == 64
    capped = SamplingContract(mode=Mode.TRAIN, temperature=0.0, max_tokens_cap=10)
    engine, _ = capped.enforce(_request(max_tokens=None))
    assert engine["max_tokens"] == 10


def test_eval_records_overrides_for_differing_client_values() -> None:
    """EVAL records overrides only for client-sent fields whose applied value differs."""
    contract = SamplingContract(mode=Mode.EVAL, temperature=0.5, max_tokens_cap=4)
    req = _request(temperature=0.2, max_tokens=64, n=3)
    engine, overrides = contract.enforce(req)
    assert engine["max_tokens"] == 4
    assert set(overrides) == {"max_tokens"}
    assert overrides["max_tokens"] == {"requested": 64, "applied": 4}


def test_train_records_stop_override_when_client_sends_stop() -> None:
    """TRAIN records a stop override when the client-sent stop list differs from the applied one."""
    contract = SamplingContract(mode=Mode.TRAIN, temperature=0.0)
    req = _request(stop=["a", "b"])
    engine, overrides = contract.enforce(req)
    assert engine["stop"] == ["a", "b"]
    assert "stop" not in overrides


def test_replace_engine_call_revalidates() -> None:
    """dataclasses.replace on a frozen EngineCall re-runs __post_init__ validation."""
    call = _call()
    with pytest.raises(SessionRefusal, match="generation_id to be a non-empty str"):
        replace(call, generation_id="")


def test_store_create_generates_default_session_id() -> None:
    """SessionStore.create generates a 32-hex session id when none is supplied."""
    store = _store()
    session = store.create(mode=Mode.EVAL, group_id="g", created_at=0.0)
    assert len(session.session_id) == 32
    assert session.api_key.startswith("fs-sess-")
    assert session.mode is Mode.EVAL
    assert session.created_at == 0.0
