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
