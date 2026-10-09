"""M5b runtime tests: ``select_llm`` gates/records and ``make_transport`` response mapping.

No network anywhere: ``urllib.request.urlopen`` is monkeypatched to raise and every
transport, backend factory and fixture is pure data.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from foundationskills.skills.auto_research import proposers, proposers_llm
from foundationskills.skills.auto_research.ledger import sha256_hex
from foundationskills.skills.auto_research.proposers_llm import (
    NAME,
    PARSE_SPEC_VERSION,
    make_transport,
    select_llm,
)

REGISTRY: dict[str, Any] = {
    "schema": "fs.model_registry/1",
    "models": {
        "nightly-llm": {
            "status": "active",
            "endpoint": "http://unit.invalid:8443",
            "model_id": "fsm-1",
            "auth": {"env": "FS_LLM_KEY"},
        }
    },
}
NORMALISED_CARD = {
    "idea": "cut lr",
    "domain": "optimizer",
    "delta": {"optim.lr": 0.001},
    "watch": ["val"],
    "score": None,
    "rationale": "val loss up",
}
CLEAN_TEXT = json.dumps([NORMALISED_CARD])
TAINT_TEXT = json.dumps([NORMALISED_CARD, {"command": "scancel 4242"}])
HALLUCINATED_TEXT = json.dumps(
    [
        NORMALISED_CARD,
        {"idea": "invent axis", "domain": "optimizer", "delta": {"optim.beta1": 0.5}, "watch": [], "score": None, "rationale": "x"},
        {"idea": "too big", "domain": "optimizer", "delta": {"optim.lr": 10.0}, "watch": [], "score": None, "rationale": "y"},
    ]
)
RESULTS: list[dict[str, Any]] = [
    {"trial": 0, "role": "root", "seed": 0, "metrics": {"val_loss": 1.5}, "delta": {}, "status": "ok"}
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: Any) -> None:
    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unit tests never touch the network")

    monkeypatch.setattr("urllib.request.urlopen", _forbidden)


def make_spec(
    *,
    pool_key: str = "nightly-llm",
    model: str = "fsm-1",
    max_calls: Any = None,
    require_model: bool = False,
    axes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    llm: dict[str, Any] = {"pool_key": pool_key, "model": model, "max_cards": 3}
    if max_calls is not None:
        llm["max_calls"] = max_calls
    return {
        "axes": axes
        if axes is not None
        else [{"key": "optim.lr", "type": "float", "min": 0.00001, "max": 0.01, "values": [0.00001, 0.001, 0.01]}],
        "objective": {"metric": "val_loss", "direction": "min"},
        "budget": {"max_runs": 4},
        "proposer": {"name": NAME, "seed": 7, "require_model": require_model, "llm": llm},
    }


def ok_transport(text: Any = CLEAN_TEXT, calls: list[tuple] | None = None) -> Any:
    def transport(messages: Any, model: Any, sampling: Any) -> dict[str, Any]:
        if calls is not None:
            calls.append((messages, model, sampling))
        return {"status": "ok", "text": text, "kind": None, "latency_s": 0.25, "tokens_in": 11, "tokens_out": 5}

    return transport


def _one_catalog_card(spec: Any, results: Any, launches: Any, current: Any, symptoms: Any, k: Any) -> tuple[list[Any], list[str]]:
    return [{"idea": "catalog"}], ["catalog_drop", "llm_response_tainted:0"]


def run(spec: dict[str, Any] | None = None, symptoms: Any = None, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("k", 1)
    kwargs.setdefault("pool", REGISTRY)
    kwargs.setdefault("transport", ok_transport())
    return select_llm(spec or make_spec(), RESULTS, [], {}, {} if symptoms is None else symptoms, **kwargs)


def test_happy_path_records_cards_request_and_response() -> None:
    calls: list[tuple] = []
    out = run(k=2, transport=ok_transport(calls=calls))
    assert out["refusal"] is None
    assert out["drops"] == []
    assert out["cards"] == [NORMALISED_CARD]
    provenance = out["provenance"]
    assert provenance["requested"] == NAME
    assert provenance["proposer"] == {"name": NAME, "version": proposers_llm.VERSION}
    assert provenance["package_version"] == proposers_llm.package_version()
    assert provenance["seed"] == 7 and provenance["k"] == 2 and provenance["fallback"] is None
    assert provenance["rows_digest"] is not None
    block = out["llm"]
    assert sorted(block) == ["calls_used", "max_calls", "parse", "request", "response"]
    assert block["calls_used"] == 1 and block["max_calls"] == 4
    request = block["request"]
    assert request["model"] == "fsm-1" and request["pool_key"] == "nightly-llm"
    assert request["endpoint_fingerprint"].startswith("sha256:")
    assert request["sampling"] == {"temperature": 0.2, "max_tokens": 1200}
    assert request["prompt_spec_version"] == proposers_llm.PROMPT_SPEC_VERSION
    assert request["prompt_hash"] == "sha256:" + sha256_hex(proposers_llm.canonical(request["messages"]))
    assert request["evidence_truncated"] == []
    body = json.dumps(block)
    assert "unit.invalid" not in body and "base_url" not in body and "api_key_env" not in body
    response = block["response"]
    assert response["status"] == "ok" and response["text"] == CLEAN_TEXT
    assert response["response_hash"] == "sha256:" + sha256_hex(CLEAN_TEXT.encode("utf-8"))
    assert response["excerpt"] == CLEAN_TEXT and response["latency_s"] == 0.25
    assert response["tokens_in"] == 11 and response["tokens_out"] == 5
    parse = block["parse"]
    assert parse["spec_version"] == PARSE_SPEC_VERSION and parse["max_cards"] == 3
    assert parse["cards"] == out["cards"] and parse["cards_total"] == 1
    assert parse["drops"] == [] and parse["tainted"] == []
    assert calls and calls[0][1] == "fsm-1"


def test_unpinned_endpoint_refuses_without_calling() -> None:
    calls: list[tuple] = []
    out = run(make_spec(pool_key="gpt-x"), transport=ok_transport(calls=calls))
    assert out["refusal"] == ("AR-PR-005", "AR-PR-005_unpinned_endpoint:pool_key")
    assert out["cards"] == [] and out["llm"] is None and not calls
    assert out["drops"] == [] and out["provenance"]["fallback"] is None


def test_secret_in_symptoms_refuses_before_any_call() -> None:
    calls: list[tuple] = []
    out = run(symptoms={"note": "sk-live-0123456789abcdefghij"}, transport=ok_transport(calls=calls))
    assert out["refusal"] == ("AR-PR-005", "AR-PR-005_secret_in_payload:prompt")
    assert out["cards"] == [] and out["llm"] is None and not calls


def test_secret_in_response_refuses_with_the_dead_record() -> None:
    out = run(transport=ok_transport(text=CLEAN_TEXT + "\nsk-live-0123456789abcdefghij"))
    assert out["refusal"] == ("AR-PR-005", "AR-PR-005_secret_in_payload:response")
    assert out["cards"] == [] and out["provenance"]["fallback"] is None
    assert out["llm"]["calls_used"] == 1 and out["llm"]["parse"]["cards"] == []


def test_strict_failure_is_an_ar_pr_001_refusal() -> None:
    out = run(make_spec(require_model=True, max_calls=1), calls_used=1)
    assert out["refusal"] == ("AR-PR-001", "proposer_unavailable:no_calls_left")
    assert out["cards"] == [] and out["provenance"]["fallback"] is None
    assert out["drops"] == ["llm_call_budget_exhausted"]


def test_strict_tainted_card_is_an_ar_pr_003_refusal() -> None:
    out = run(make_spec(require_model=True), transport=ok_transport(text=TAINT_TEXT))
    assert out["refusal"] == ("AR-PR-003", "AR-PR-003_unsafe_card:1")
    assert out["cards"] == [] and out["llm"]["parse"]["tainted"] == [1]


def test_tainted_response_taints_every_card_and_falls_back(monkeypatch: Any) -> None:
    monkeypatch.setattr(proposers, "_catalog_cards", _one_catalog_card)
    out = run(transport=ok_transport(text=TAINT_TEXT))
    assert out["refusal"] is None and out["provenance"]["fallback"] == "tainted"
    assert out["cards"] == [{"idea": "catalog"}]
    assert out["drops"] == [
        "llm_response_tainted:0",
        "llm_response_tainted:1",
        "proposer_fallback_catalog:tainted",
        "catalog_drop",
    ]
    assert out["llm"]["parse"]["tainted"] == [1] and out["llm"]["parse"]["cards_total"] == 2


def test_hallucinated_axes_and_values_drop_and_never_clamp() -> None:
    out = run(transport=ok_transport(text=HALLUCINATED_TEXT))
    assert out["refusal"] is None and out["cards"] == [NORMALISED_CARD]
    assert out["llm"]["parse"]["drops"] == [
        "llm_out_of_axes:1:optim.beta1",
        "llm_card_empty:1",
        "llm_value_out_of_range:2:optim.lr",
        "llm_card_empty:2",
    ]
    assert out["drops"] == out["llm"]["parse"]["drops"]


def test_missing_transport_metadata_falls_back_without_calling(monkeypatch: Any) -> None:
    def _boom(resolved: Any, factory: Any = None) -> Any:
        raise RuntimeError("no client for this host")

    monkeypatch.setattr(proposers_llm, "make_transport", _boom)
    out = run(transport=None)
    assert out["refusal"] is None and out["provenance"]["fallback"] == "no_transport"
    assert "proposer_fallback_catalog:no_transport" in out["drops"]
    assert out["llm"] is None


def test_unstable_parse_falls_back(monkeypatch: Any) -> None:
    seen = {"n": 0}

    def _parse(text: Any, spec: Any, max_cards: Any) -> dict[str, Any]:
        seen["n"] += 1
        cards = [NORMALISED_CARD] if seen["n"] == 1 else []
        return {"cards": cards, "cards_total": 1, "drops": [], "tainted": []}

    monkeypatch.setattr(proposers_llm, "parse_cards", _parse)
    out = run()
    assert out["refusal"] is None and out["provenance"]["fallback"] == "parse_unstable"
    assert "proposer_fallback_catalog:parse_unstable" in out["drops"]


def _raising(message: str) -> Any:
    def transport(messages: Any, model: Any, sampling: Any) -> dict[str, Any]:
        raise RuntimeError(message)

    return transport


@pytest.mark.parametrize(
    "case,reason",
    [("budget", "no_calls_left"), ("axes", "no_axes"), ("raises", "transport_error"),
     ("status", "transport_error"), ("unparseable", "parse_error"), ("empty", "no_model_cards")],
)
def test_each_failure_reason_is_named_and_counted(case: str, reason: str) -> None:
    spec = make_spec()
    used = 0
    if case == "budget":
        spec["proposer"]["llm"]["max_calls"] = 1
        used = 1
        transport = ok_transport()
    elif case == "axes":
        spec["axes"] = []
        transport = ok_transport()
    elif case == "raises":
        transport = _raising("refused at the socket")
    elif case == "status":
        def transport(messages: Any, model: Any, sampling: Any) -> dict[str, Any]:
            return {"status": "error", "text": None, "kind": "timeout", "latency_s": 9.0, "tokens_in": None, "tokens_out": None}

    elif case == "unparseable":
        transport = ok_transport(text="no cards in here")
    else:
        transport = ok_transport(text="[]")
    out = run(spec, transport=transport, calls_used=used)
    assert out["refusal"] is None and out["provenance"]["fallback"] == reason
    assert f"proposer_fallback_catalog:{reason}" in out["drops"]
    if case == "budget":
        assert out["drops"][0] == "llm_call_budget_exhausted" and out["llm"] is None
    if case in ("raises", "status"):
        assert out["llm"]["calls_used"] == 1
        assert out["llm"]["response"]["status"] == "error" and out["llm"]["response"]["text"] is None


def test_make_transport_maps_ok_and_error_responses() -> None:
    seen: dict[str, Any] = {}

    class _Backend:
        def __init__(self) -> None:
            self.left = [
                SimpleNamespace(content="hi", error=None, prompt_tokens=3, completion_tokens=0, finish_reason="stop", cache_hit=False),
                SimpleNamespace(content=None, error="slow_upstream", prompt_tokens=None, completion_tokens=None, finish_reason=None, cache_hit=False),
            ]
            self.calls: list[tuple] = []

        def complete(self, messages: Any, **kwargs: Any) -> Any:
            self.calls.append((messages, kwargs))
            return self.left.pop(0)

    backend = _Backend()

    def factory(cfg: Any, op_name: Any) -> Any:
        seen["cfg"] = cfg
        seen["op"] = op_name
        return backend

    resolved = {"pool_key": "nightly-llm", "model": "fsm-1", "base_url": "http://unit.invalid:8443", "api_key_env": "FS_LLM_KEY"}
    transport = make_transport(resolved, backend_factory=factory)
    assert seen["cfg"] == {
        "kind": "openai_compatible",
        "base_url": "http://unit.invalid:8443",
        "model": "fsm-1",
        "api_key_env": "FS_LLM_KEY",
    }
    assert seen["op"] == "auto_research_llm"
    ok = transport([{"role": "user", "content": "hi"}], "fsm-1", {"temperature": 0.7, "max_tokens": 55})
    assert ok["status"] == "ok" and ok["text"] == "hi" and ok["kind"] is None
    assert ok["tokens_in"] == 3 and ok["tokens_out"] is None and isinstance(ok["latency_s"], float)
    assert backend.calls[0][1] == {"temperature": 0.7, "max_tokens": 55, "seed": 0, "json_mode": False}
    bad = transport([], "fsm-1", {})
    assert bad["status"] == "error" and bad["text"] is None and bad["kind"] == "slow_upstream"
    assert bad["tokens_in"] is None and bad["tokens_out"] is None
    assert backend.calls[1][1] == {"temperature": 0.2, "max_tokens": 1200, "seed": 0, "json_mode": False}
    bare = make_transport({"base_url": "http://unit.invalid:2", "model": "fsm-2", "api_key_env": None}, backend_factory=factory)
    assert seen["cfg"] == {"kind": "openai_compatible", "base_url": "http://unit.invalid:2", "model": "fsm-2"}
    assert callable(bare)
