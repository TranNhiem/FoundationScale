"""Tests for ``foundationscale.agentic_rl.verl_adapter`` (binding API)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from foundationscale.agentic_rl.trajectory_v2 import (
    ApiFormat,
    EnvKind,
    EnvMeta,
    GatewayMeta,
    ModelMeta,
    RewardScope,
    Termination,
)
from foundationscale.agentic_rl.verl_adapter import (
    VERL_PIN,
    VerlAdapterRefusal,
    episode_from_agent_loop_output,
    require_pinned_verl,
    samples_from_rollout_batch,
)

MODEL = ModelMeta(name="m", version=0, checkpoint_sha="sha")
ENV = EnvMeta(kind=EnvKind.FS_LOCAL, task_id="t0")
GATEWAY = GatewayMeta(api_format=ApiFormat.WHITE_BOX, session_id="s0")


def _out(**kw):
    """Build a SimpleNamespace AgentLoopOutput with sane defaults."""
    base = dict(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5, 6],
        response_mask=[1, 0, 0, 1],
        response_logprobs=[-0.5, -0.25, -0.125, -0.0625],
        routed_experts=None,
        reward_score=None,
        num_turns=2,
        extra_fields={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _dict_out(**kw):
    """Build a mapping-variant AgentLoopOutput with sane defaults."""
    base = dict(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5, 6],
        response_mask=[1, 0, 0, 1],
        response_logprobs=[-0.5, -0.25, -0.125, -0.0625],
        routed_experts=None,
        reward_score=None,
        num_turns=2,
        extra_fields={},
    )
    base.update(kw)
    return base


def _ep(out, **kw):
    """Adapt ``out`` into a validated Episode."""
    args = dict(
        episode_id="e0", group_id="g0", policy_version=7, model=MODEL, env=ENV, gateway=GATEWAY
    )
    args.update(kw)
    return episode_from_agent_loop_output(out, **args)


def test_multi_turn_mask_produces_spans_g0_g1():
    """Mask runs 1,0,0,1 yield spans g0=(0,1) and g1=(3,4)."""
    ep = _ep(_out())
    seg = ep.segments[0]
    assert seg.generation_spans == ((0, 1, "g0"), (3, 4, "g1"))
    assert seg.loss_mask == (1, 0, 0, 1)


def test_dict_variant_matches_namespace_variant():
    """Mapping access yields the same spans and masks as attribute access."""
    ns = _ep(_out())
    dm = _ep(_dict_out())
    assert dm.segments[0].generation_spans == ns.segments[0].generation_spans
    assert dm.segments[0].loss_mask == ns.segments[0].loss_mask


def test_mask0_tokens_get_nan_logprobs_and_minus_one_versions():
    """Mask-0 positions carry NaN logprobs and policy_version -1."""
    seg = _ep(_out()).segments[0]
    assert math.isnan(seg.logprobs[1]) and math.isnan(seg.logprobs[2])
    assert seg.policy_version == (7, -1, -1, 7)
    assert seg.logprobs[0] == -0.5 and seg.logprobs[3] == -0.0625


def test_reward_score_becomes_one_episode_reward_event():
    """A non-None reward_score yields exactly one EPISODE-scope RewardEvent."""
    ep = _ep(_out(reward_score=1.5))
    assert len(ep.reward_events) == 1
    ev = ep.reward_events[0]
    assert ev.value == 1.5 and ev.scope is RewardScope.EPISODE
    assert ev.source == "verl_reward_score"


def test_none_reward_score_is_abstention_not_zero():
    """reward_score None yields zero reward events, never a 0.0 event."""
    ep = _ep(_out(reward_score=None))
    assert ep.reward_events == ()


def test_routed_experts_converted_to_nested_tuples():
    """routed_experts passes through as nested tuples when length matches."""
    rows = [[(0, 1)], [(2,)], [(3, 4)], [(5,)], [(6, 7)], [(8,)]]  # 2 prompt + 4 response
    seg = _ep(_out(routed_experts=rows)).segments[0]
    assert seg.routed_experts == (((0, 1),), ((2,),), ((3, 4),), ((5,),), ((6, 7),), ((8,),))


def test_routed_experts_length_mismatch_refused():
    """routed_experts rows must cover prompt + response tokens exactly."""
    rows = [[(0, 1)], [(2,)]]
    with pytest.raises(VerlAdapterRefusal, match="routed_experts"):
        _ep(_out(routed_experts=rows))


def test_missing_field_refused():
    """A missing required field is refused by name."""
    out = _dict_out()
    del out["response_mask"]
    with pytest.raises(VerlAdapterRefusal, match="response_mask"):
        _ep(out)


def test_none_logprobs_with_trainable_tokens_refused():
    """response_logprobs None with any mask-1 token is refused with the config hint."""
    with pytest.raises(VerlAdapterRefusal, match="calculate_log_probs"):
        _ep(_out(response_logprobs=None))


def test_logprobs_length_mismatch_refused():
    """response_logprobs must have one entry per response token."""
    with pytest.raises(VerlAdapterRefusal, match="response_logprobs"):
        _ep(_out(response_logprobs=[-0.5, -0.25]))


def test_response_mask_length_mismatch_refused():
    """response_mask must have one entry per response token."""
    with pytest.raises(VerlAdapterRefusal, match="response_mask"):
        _ep(_out(response_mask=[1, 0]))


def test_response_ids_length_mismatch_refused():
    """response_ids length must agree with response_mask length."""
    with pytest.raises(VerlAdapterRefusal, match="response_ids"):
        _ep(_out(response_ids=[3, 4, 5], response_mask=[1, 0, 0, 1]))


def test_mask_values_outside_zero_one_refused():
    """Mask values must be exactly 0 or 1 ints."""
    with pytest.raises(VerlAdapterRefusal, match="response_mask"):
        _ep(_out(response_mask=[1, 2, 0, 1]))


def test_nonfinite_logprob_on_mask1_refused():
    """A non-finite logprob on a mask-1 token is refused."""
    with pytest.raises(VerlAdapterRefusal, match="response_logprobs"):
        _ep(_out(response_logprobs=[float("nan"), -0.25, -0.125, -0.0625]))


def test_empty_response_refused():
    """An empty response is refused."""
    with pytest.raises(VerlAdapterRefusal, match="response_ids"):
        _ep(_out(response_ids=[], response_mask=[], response_logprobs=[]))


def test_policy_version_bool_refused():
    """policy_version must be a real int >= 0; bool is refused."""
    with pytest.raises(VerlAdapterRefusal, match="policy_version"):
        _ep(_dict_out(), policy_version=True)


def test_policy_version_negative_refused():
    """A negative policy_version is refused."""
    with pytest.raises(VerlAdapterRefusal, match="policy_version"):
        _ep(_dict_out(), policy_version=-1)


def test_episode_status_ok_and_termination_stop():
    """The adapted episode is OK with the normal stop termination."""
    ep = _ep(_out())
    assert ep.status.value == "ok" and ep.termination is Termination.STOP


def test_harness_meta_names_verl_agent_loop():
    """HarnessMeta carries name verl_agent_loop and the harness version."""
    ep = _ep(_out(), harness_version="abc123")
    assert ep.harness.name == "verl_agent_loop" and ep.harness.version == "abc123"


def test_task_prompt_ids_equal_prompt_ids():
    """task_prompt_ids is the verl prompt_ids tuple."""
    ep = _ep(_out())
    assert ep.task_prompt_ids == (1, 2)


def test_require_pinned_verl_match_returns_commit():
    """A matching direct_url.json commit id is returned."""
    text = json.dumps({"vcs_info": {"commit_id": VERL_PIN}})
    got = require_pinned_verl(read_text=lambda d, f: text)
    assert got == VERL_PIN


def test_require_pinned_verl_mismatch_names_both_shas():
    """A mismatched commit names both the expected and found SHAs."""
    other = "f" * 40
    text = json.dumps({"vcs_info": {"commit_id": other}})

    def reader(dist, filename):
        return text

    with pytest.raises(VerlAdapterRefusal) as exc:
        require_pinned_verl(read_text=reader)
    msg = str(exc.value)
    assert VERL_PIN in msg and other in msg


def test_require_pinned_verl_missing_file_refused():
    """An absent direct_url.json (PyPI install) is refused."""
    with pytest.raises(VerlAdapterRefusal, match="direct_url.json"):
        require_pinned_verl(read_text=lambda d, f: None)


def test_require_pinned_verl_missing_commit_id_refused():
    """A direct_url.json without vcs_info.commit_id is refused."""
    with pytest.raises(VerlAdapterRefusal, match="commit_id"):
        require_pinned_verl(read_text=lambda d, f: '{"url": "https://example/x"}')


def test_require_pinned_verl_not_installed_refused():
    """A missing verl distribution is refused."""
    with pytest.raises(VerlAdapterRefusal, match="verl"):
        require_pinned_verl(distribution="verl-not-installed-xyz", read_text=lambda d, f: None)


def test_require_pinned_verl_real_metadata_lookup_refuses_absent_distribution():
    """The default importlib.metadata path refuses a distribution that is not installed."""
    with pytest.raises(VerlAdapterRefusal, match="fs-no-such-dist-xyz"):
        require_pinned_verl(distribution="fs-no-such-dist-xyz")


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        json.dumps({"vcs_info": "git"}),
        json.dumps({"vcs_info": {"commit_id": ""}}),
    ],
)
def test_require_pinned_verl_refuses_malformed_direct_url(text):
    """Malformed PEP 610 metadata is refused, never read as a pin match."""
    with pytest.raises(VerlAdapterRefusal):
        require_pinned_verl(read_text=lambda d, f: text)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", True])
def test_non_finite_or_non_numeric_reward_refused(bad):
    """reward_score must be a finite number or None."""
    with pytest.raises(VerlAdapterRefusal, match="reward_score"):
        _ep(_out(reward_score=bad))


def test_non_sequence_ids_refused():
    """prompt_ids given as a string is refused, not split into characters."""
    with pytest.raises(VerlAdapterRefusal, match="prompt_ids"):
        _ep(_out(prompt_ids="12"))


def _padded_batch():
    """Build a B=2, P=4, R=5 padded rollout batch as plain nested lists."""
    return {
        "prompts": [
            [0, 0, 11, 12],
            [0, 21, 22, 23],
        ],
        "responses": [
            [31, 32, 33, 34, 35],
            [41, 42, 43, 44, 45],
        ],
        "response_mask": [
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],
        ],
        "attention_mask": [
            [0, 0, 1, 1, 1, 1, 1, 1, 0],  # 4 real response tokens (one is a tool token), 1 pad
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
        ],
        "rollout_log_probs": [
            [-0.5, -0.25, -0.125, -0.0625, -0.03],
            [-0.7, -0.35, -0.17, -0.08, -0.04],
        ],
        "rm_scores": [
            [0.0, 0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    }


def test_samples_exact_unpadding_of_prompts_and_responses():
    """Left-padded prompts and right-padded responses are stripped exactly."""
    samples = samples_from_rollout_batch(_padded_batch(), {"__num_turns__": [2, 1]})
    assert samples[0]["prompt_ids"] == [11, 12]
    assert samples[0]["response_ids"] == [31, 32, 33, 34]
    assert samples[0]["response_mask"] == [1, 1, 0, 1]
    assert samples[1]["prompt_ids"] == [21, 22, 23]
    assert samples[1]["response_ids"] == [41]


def test_samples_logprobs_follow_response_mask_selection():
    """rollout_log_probs is sliced at the same valid response indices as the ids."""
    samples = samples_from_rollout_batch(_padded_batch(), {"__num_turns__": [2, 1]})
    assert samples[0]["response_logprobs"] == [-0.5, -0.25, -0.125, -0.0625]
    assert samples[1]["response_logprobs"] == [-0.7]


def test_samples_reward_score_at_last_valid_response_index():
    """rm_scores is read at the last valid response index, not at the padded tail."""
    samples = samples_from_rollout_batch(_padded_batch(), {"__num_turns__": [2, 1]})
    assert samples[0]["reward_score"] == 1.5
    assert samples[1]["reward_score"] == 0.0


def test_samples_rm_scores_absent_gives_none_reward():
    """Without rm_scores every row reports reward_score None (abstention)."""
    batch = _padded_batch()
    del batch["rm_scores"]
    samples = samples_from_rollout_batch(batch, {"__num_turns__": [2, 1]})
    assert samples[0]["reward_score"] is None
    assert samples[1]["reward_score"] is None


def test_samples_num_turns_read_from_non_tensor_batch():
    """num_turns comes from non_tensor['__num_turns__'] per row."""
    samples = samples_from_rollout_batch(_padded_batch(), {"__num_turns__": [3, 1]})
    assert samples[0]["num_turns"] == 3
    assert samples[1]["num_turns"] == 1


def test_samples_num_turns_defaults_to_zero_when_absent():
    """num_turns is 0 when the non_tensor batch is missing or lacks the key."""
    batch = _padded_batch()
    assert samples_from_rollout_batch(batch)[0]["num_turns"] == 0
    assert samples_from_rollout_batch(batch, {})[0]["num_turns"] == 0


def test_samples_row_with_no_valid_response_tokens():
    """A row whose response mask is all padding yields empty response fields and None reward."""
    batch = _padded_batch()
    batch["attention_mask"][1] = [0, 1, 1, 1, 0, 0, 0, 0, 0]
    samples = samples_from_rollout_batch(batch, {"__num_turns__": [2, 1]})
    assert samples[1]["response_ids"] == []
    assert samples[1]["response_mask"] == []
    assert samples[1]["response_logprobs"] == []
    assert samples[1]["reward_score"] is None
    assert samples[1]["prompt_ids"] == [21, 22, 23]


def test_samples_every_refusal():
    """Each documented refusal is raised by name with both counts where applicable."""
    with pytest.raises(VerlAdapterRefusal, match="prompts"):
        samples_from_rollout_batch({k: v for k, v in _padded_batch().items() if k != "prompts"})
    with pytest.raises(VerlAdapterRefusal, match="responses"):
        samples_from_rollout_batch({k: v for k, v in _padded_batch().items() if k != "responses"})
    with pytest.raises(VerlAdapterRefusal, match="response_mask"):
        samples_from_rollout_batch(
            {k: v for k, v in _padded_batch().items() if k != "response_mask"}
        )
    with pytest.raises(VerlAdapterRefusal, match="attention_mask"):
        samples_from_rollout_batch(
            {k: v for k, v in _padded_batch().items() if k != "attention_mask"}
        )
    batch = _padded_batch()
    batch["responses"] = batch["responses"][:1]
    with pytest.raises(VerlAdapterRefusal, match="responses"):
        samples_from_rollout_batch(batch)
    batch = _padded_batch()
    batch["attention_mask"] = [[0, 0, 1, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 1, 0, 0, 0]]
    with pytest.raises(VerlAdapterRefusal, match="attention_mask"):
        samples_from_rollout_batch(batch)
    batch = _padded_batch()
    batch["attention_mask"][0] = [0, 1, 0, 1, 1, 1, 0, 1, 0]
    with pytest.raises(VerlAdapterRefusal, match="prompt"):
        samples_from_rollout_batch(batch)
    batch = _padded_batch()
    batch["attention_mask"][0] = [
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
    ][:9]
    batch["attention_mask"][0][4:9] = [1, 0, 1, 0, 0]
    with pytest.raises(VerlAdapterRefusal, match="response"):
        samples_from_rollout_batch(batch)


def test_samples_accepts_tolist_wrapping_object():
    """Arrays exposing .tolist() are converted before slicing."""
    batch = _padded_batch()
    batch["prompts"] = _ListWrapper(batch["prompts"])
    batch["responses"] = _ListWrapper(batch["responses"])
    batch["response_mask"] = _ListWrapper(batch["response_mask"])
    batch["attention_mask"] = _ListWrapper(batch["attention_mask"])
    batch["rollout_log_probs"] = _ListWrapper(batch["rollout_log_probs"])
    batch["rm_scores"] = _ListWrapper(batch["rm_scores"])
    samples = samples_from_rollout_batch(batch, {"__num_turns__": _ListWrapper([2, 1])})
    assert samples[0]["prompt_ids"] == [11, 12]
    assert samples[0]["response_ids"] == [31, 32, 33, 34]
    assert samples[0]["response_logprobs"] == [-0.5, -0.25, -0.125, -0.0625]
    assert samples[0]["reward_score"] == 1.5
    assert samples[0]["num_turns"] == 2


def test_samples_end_to_end_through_episode_adapter():
    """Every unpacked sample adapts into a validated Episode via _ep."""
    samples = samples_from_rollout_batch(_padded_batch(), {"__num_turns__": [2, 1]})
    for i, sample in enumerate(samples):
        ep = _ep(sample, episode_id=f"e{i}", group_id="g0")
        assert ep.segments[0].prompt_ids == tuple(sample["prompt_ids"])
        assert ep.segments[0].response_ids == tuple(sample["response_ids"])
        assert ep.segments[0].loss_mask == tuple(sample["response_mask"])
        if sample["reward_score"] is None:
            assert ep.reward_events == ()
        else:
            assert len(ep.reward_events) == 1
            assert ep.reward_events[0].value == sample["reward_score"]


def test_samples_end_to_end_row_without_response_tokens_is_refused_by_adapter():
    """An all-padding response row unpacks fine but the adapter refuses the empty response."""
    batch = _padded_batch()
    batch["attention_mask"][1] = [0, 1, 1, 1, 0, 0, 0, 0, 0]
    samples = samples_from_rollout_batch(batch, {"__num_turns__": [2, 1]})
    with pytest.raises(VerlAdapterRefusal, match="response_ids"):
        _ep(samples[1], episode_id="e1", group_id="g0")


class _ListWrapper:
    """Tiny stand-in for a tensor-like object exposing .tolist()."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


def test_samples_missing_required_keys_refused():
    """Each required key's absence is refused by name."""
    for key in ("prompts", "responses", "response_mask", "attention_mask"):
        batch = _padded_batch()
        del batch[key]
        with pytest.raises(VerlAdapterRefusal):
            samples_from_rollout_batch(batch)


def test_samples_empty_prompts_refused():
    """An empty prompts batch is refused."""
    batch = _padded_batch()
    batch["prompts"] = []
    batch["responses"] = []
    batch["response_mask"] = []
    batch["attention_mask"] = []
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


@pytest.mark.parametrize("key", ["prompts", "responses", "response_mask", "attention_mask"])
def test_samples_batch_size_mismatch_refused(key):
    """Every key must carry exactly B rows."""
    batch = _padded_batch()
    batch[key] = batch[key][:1]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


@pytest.mark.parametrize("key", ["prompts", "responses", "response_mask", "attention_mask"])
def test_samples_ragged_rows_refused(key):
    """Every key must have constant width across rows."""
    batch = _padded_batch()
    batch[key][1] = batch[key][1][:-1]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


def test_samples_rollout_log_probs_refusals():
    """rollout_log_probs row and column counts must agree with the batch."""
    batch = _padded_batch()
    batch["rollout_log_probs"] = batch["rollout_log_probs"][:1]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)
    batch = _padded_batch()
    batch["rollout_log_probs"][0] = batch["rollout_log_probs"][0][:-1]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


def test_samples_rm_scores_refusals():
    """rm_scores row and column counts must agree with the batch."""
    batch = _padded_batch()
    batch["rm_scores"] = batch["rm_scores"][:1]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)
    batch = _padded_batch()
    batch["rm_scores"][0] = batch["rm_scores"][0][:-1]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


def test_samples_num_turns_length_mismatch_refused():
    """non_tensor.__num_turns__ must carry exactly B entries."""
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(_padded_batch(), {"__num_turns__": [2]})


def test_samples_non_numeric_attention_mask_refused():
    """Non-numeric attention_mask entries are refused in both prompt and response halves."""
    batch = _padded_batch()
    batch["attention_mask"][0][2] = "1"
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)
    batch = _padded_batch()
    batch["attention_mask"][0][4] = "1"
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


def test_samples_prompt_mask_not_left_padded_refused():
    """A 1 followed by a 0 in the prompt half is refused."""
    batch = _padded_batch()
    batch["attention_mask"][0] = [0, 1, 0, 1, 1, 1, 1, 1, 0]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)


def test_samples_response_mask_not_right_padded_refused():
    """A 0 followed by a 1 in the response half is refused."""
    batch = _padded_batch()
    batch["attention_mask"][0] = [0, 0, 1, 1, 1, 0, 1, 1, 0]
    with pytest.raises(VerlAdapterRefusal):
        samples_from_rollout_batch(batch)
