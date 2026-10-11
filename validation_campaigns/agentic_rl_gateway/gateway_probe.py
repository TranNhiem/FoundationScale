#!/usr/bin/env python3
"""Standalone end-to-end probe for the FoundationScale gateway.

Runs on a GB200 node inside a container with ``transformers`` and the
FoundationScale ``src/`` on ``PYTHONPATH``.  A vLLM server must already be
running.  Prints exactly one JSON report line to stdout and writes the same
JSON to ``--out``.  Exits 0 when every check passes, 5 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import transformers

import foundationscale.agentic_rl.cli as fs_cli
import foundationscale.agentic_rl.engines.vllm as fs_vllm
import foundationscale.agentic_rl.tools as fs_tools
from foundationscale.agentic_rl.gateway.server import (
    GatewayConfig,
    GatewayCore,
    serve,
)
from foundationscale.agentic_rl.gateway.session import Mode, SamplingContract

# --------------------------------------------------------------------------- #
# HTTP helpers (urllib only)
# --------------------------------------------------------------------------- #


def _http(method: str, url: str, headers: dict | None = None, body=None, timeout: float = 300.0):
    """Return (status, headers_dict, raw_bytes, elapsed_seconds)."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
            rh = dict(resp.headers.items())
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
        rh = dict(e.headers.items()) if e.headers is not None else {}
    except Exception as e:  # noqa: BLE001
        return -1, {}, repr(e).encode("utf-8"), time.perf_counter() - t0
    return status, rh, raw, time.perf_counter() - t0


def _json_body(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _post_json(url: str, headers: dict, body, timeout: float = 300.0):
    status, rh, raw, dt = _http("POST", url, headers, body, timeout)
    return status, _json_body(raw), dt


def _sse_frames(raw: bytes) -> list[str]:
    """Split a raw SSE byte stream into complete ``data:`` payloads."""
    out: list[str] = []
    for block in raw.decode("utf-8", "replace").split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(line[len("data:") :].strip())
    return out


# --------------------------------------------------------------------------- #
# Wire-format helpers: build requests + extract assistant text / tool calls
# --------------------------------------------------------------------------- #

TOOL_SCHEMA = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def _chat_tools():
    return [{"type": "function", "function": TOOL_SCHEMA}]


def _responses_tools():
    return [{"type": "function", **TOOL_SCHEMA}]


def _anthropic_tools():
    return [
        {
            "name": TOOL_SCHEMA["name"],
            "description": TOOL_SCHEMA["description"],
            "input_schema": TOOL_SCHEMA["parameters"],
        }
    ]


def _gemini_tools():
    return [{"functionDeclarations": [TOOL_SCHEMA]}]


def _build_turn1(fmt: str, model: str) -> dict:
    text = "What is the weather in Paris? Use the tool."
    if fmt == "chat_completions":
        return {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "tools": _chat_tools(),
        }
    if fmt == "responses":
        return {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            ],
            "tools": _responses_tools(),
        }
    if fmt == "anthropic_messages":
        return {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": text}],
            "tools": _anthropic_tools(),
        }
    # gemini
    return {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "tools": _gemini_tools(),
        "generationConfig": {"maxOutputTokens": 1024},
    }


def _extract_assistant(fmt: str, body: dict) -> tuple[str, list[dict]]:
    """Return (assistant_text, tool_calls) where each tool call is
    {"id": str|None, "name": str, "arguments": str}."""
    calls: list[dict] = []
    text = ""
    if body is None:
        return text, calls
    if fmt == "chat_completions":
        msg = (body.get("choices") or [{}])[0].get("message", {}) or {}
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            calls.append(
                {
                    "id": tc.get("id"),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", ""),
                }
            )
    elif fmt == "responses":
        for item in body.get("output") or []:
            t = item.get("type")
            if t == "message":
                for p in item.get("content") or []:
                    if isinstance(p, dict) and p.get("type") in ("output_text", "text"):
                        text += p.get("text", "")
            elif t == "function_call":
                calls.append(
                    {
                        "id": item.get("call_id"),
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    }
                )
    elif fmt == "anthropic_messages":
        for blk in body.get("content") or []:
            bt = blk.get("type")
            if bt == "text":
                text += blk.get("text", "")
            elif bt == "tool_use":
                calls.append(
                    {
                        "id": blk.get("id"),
                        "name": blk.get("name", ""),
                        "arguments": json.dumps(
                            blk.get("input", {}), separators=(",", ":"), ensure_ascii=False
                        ),
                    }
                )
    else:  # gemini
        cands = body.get("candidates") or [{}]
        content = cands[0].get("content", {}) or {}
        for part in content.get("parts") or []:
            if "text" in part:
                text += part.get("text", "")
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(
                    {
                        "id": None,
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(
                            fc.get("args", {}), separators=(",", ":"), ensure_ascii=False
                        ),
                    }
                )
    return text, calls


def _build_turn2(
    fmt: str,
    model: str,
    turn1_body: dict,  # noqa: ARG001 - kept for call-site symmetry
    assistant_text: str,
    tool_calls: list[dict],
) -> dict:
    """Build turn 2 preserving the assistant turn as returned."""
    if tool_calls:
        tc = tool_calls[0]
        result = "18C, sunny"
        if fmt == "chat_completions":
            msgs = [
                {"role": "user", "content": "What is the weather in Paris? Use the tool."},
                {
                    "role": "assistant",
                    "content": assistant_text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc["id"], "content": result},
            ]
            return {"model": model, "messages": msgs, "tools": _chat_tools()}
        if fmt == "responses":
            inp = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "What is the weather in Paris? Use the tool.",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
                {"type": "function_call_output", "call_id": tc["id"], "output": result},
            ]
            return {"model": model, "input": inp, "tools": _responses_tools()}
        if fmt == "anthropic_messages":
            blocks = []
            if assistant_text:
                blocks.append({"type": "text", "text": assistant_text})
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": json.loads(tc["arguments"]) if tc["arguments"] else {},
                }
            )
            msgs = [
                {"role": "user", "content": "What is the weather in Paris? Use the tool."},
                {"role": "assistant", "content": blocks},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
                    ],
                },
            ]
            return {
                "model": model,
                "max_tokens": 1024,
                "messages": msgs,
                "tools": _anthropic_tools(),
            }
        # gemini
        contents = [
            {"role": "user", "parts": [{"text": "What is the weather in Paris? Use the tool."}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": tc["name"],
                            "args": json.loads(tc["arguments"]) if tc["arguments"] else {},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": tc["name"], "response": {"result": result}}}
                ],
            },
        ]
        return {
            "contents": contents,
            "tools": _gemini_tools(),
            "generationConfig": {"maxOutputTokens": 1024},
        }
    # No tool call -> follow-up user turn.
    follow = "Answer in one word."
    if fmt == "chat_completions":
        msgs = [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."},
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": follow},
        ]
        return {"model": model, "messages": msgs, "tools": _chat_tools()}
    if fmt == "responses":
        inp = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is the weather in Paris? Use the tool."}
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": follow}],
            },
        ]
        return {"model": model, "input": inp, "tools": _responses_tools()}
    if fmt == "anthropic_messages":
        msgs = [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            {"role": "user", "content": follow},
        ]
        return {"model": model, "max_tokens": 1024, "messages": msgs, "tools": _anthropic_tools()}
    contents = [
        {"role": "user", "parts": [{"text": "What is the weather in Paris? Use the tool."}]},
        {"role": "model", "parts": [{"text": assistant_text}]},
        {"role": "user", "parts": [{"text": follow}]},
    ]
    return {
        "contents": contents,
        "tools": _gemini_tools(),
        "generationConfig": {"maxOutputTokens": 1024},
    }


# --------------------------------------------------------------------------- #
# Trajectory inspection helpers
# --------------------------------------------------------------------------- #


def _spans(trajectory: dict) -> list[dict]:
    """Collect generation spans from a trajectory dict (tolerant to shape)."""
    spans: list[dict] = []
    if not isinstance(trajectory, dict):
        return spans

    def walk(node):
        if isinstance(node, dict):
            keys = set(node.keys())
            if {"response_ids", "start", "end"} <= keys or (
                "response_ids" in keys and "prompt_len" in keys
            ):
                spans.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(trajectory)
    return spans


def _span_fields(span: dict):
    ids = span.get("response_ids") or span.get("output_ids") or []
    start = span.get("start")
    end = span.get("end")
    if start is None:
        start = span.get("prompt_len", 0)
    if end is None:
        end = len(ids)
    masks = span.get("mask") or span.get("train_mask") or span.get("loss_mask") or []
    lps = span.get("logprobs") or span.get("logprob") or []
    return list(ids), int(start), int(end), list(masks), list(lps)


def _check_trajectory(
    trajectory: dict,
    decode_plain,  # noqa: ARG001 - signature shared with the strict checker
    decode_stripped,
    expected_texts: list[str],
) -> dict:
    spans = _spans(trajectory)
    n_segments = len(spans)
    trainable = 0
    mask1_finite = True
    mask0_null = True
    fidelity: list[bool] = []
    for i, span in enumerate(spans):
        ids, start, end, masks, lps = _span_fields(span)
        seg = ids[start:end]
        trainable += sum(1 for m in masks if m == 1)
        for j, m in enumerate(masks):
            lp = lps[j] if j < len(lps) else None
            if m == 1:
                if lp is None or not isinstance(lp, (int, float)) or not math.isfinite(float(lp)):
                    mask1_finite = False
            elif m == 0 and lp is not None:
                mask0_null = False
        got = decode_stripped(seg).strip()
        want = (expected_texts[i] if i < len(expected_texts) else "").strip()
        fidelity.append(got == want)
    return {
        "n_segments": n_segments,
        "fragmentation": n_segments - 1,
        "trainable_tokens": trainable,
        "mask1_logprobs_finite": mask1_finite,
        "mask0_logprobs_null": mask0_null,
        "fidelity_per_turn": fidelity,
    }


def _policy_versions(trajectory: dict) -> list:
    out: list = []
    if not isinstance(trajectory, dict):
        return out

    def walk(node):
        if isinstance(node, dict):
            if "policy_version" in node:
                out.append(node["policy_version"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(trajectory)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="FS gateway end-to-end probe")
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name = args.served_model_name

    tok = transformers.AutoTokenizer.from_pretrained(args.model_path)
    chat_tok = fs_cli._HFChatTokenizer(tokenizer=tok)
    client = fs_vllm.VLLMClient(base_url=args.vllm_url, timeout_s=600.0, served_model_name=name)
    parser = fs_tools.QwenXmlToolCallParser()
    decode = lambda ids: tok.decode(list(ids), skip_special_tokens=False)  # noqa: E731

    config = GatewayConfig(
        served_model_name=name,
        sampling=SamplingContract(mode=Mode.TRAIN, temperature=1.0, max_tokens_cap=1024),
        admin_key="probe-admin",
        base_url="http://127.0.0.1:0",
    )
    core = GatewayCore(config, client=client, tokenizer=chat_tok, parser=parser, decode=decode)

    server = serve(core, "127.0.0.1", 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    admin_h = {"Authorization": "Bearer probe-admin"}

    report: dict = {"formats": {}, "streaming_chat": {}, "ok": True}
    all_ok = True

    def fail():
        nonlocal all_ok
        all_ok = False

    formats = ["chat_completions", "responses", "anthropic_messages", "gemini"]

    for fmt in formats:
        frec: dict = {"http_statuses": [], "timings_s": {}}
        # --- create session ---
        st, body, dt = _post_json(
            base + "/fs/v1/sessions", admin_h, {"mode": "train", "group_id": fmt}
        )
        frec["http_statuses"].append(st)
        frec["timings_s"]["create_session"] = round(dt, 4)
        if st != 201 or not isinstance(body, dict):
            frec["error"] = "session_create_failed"
            report["formats"][fmt] = frec
            fail()
            continue
        sid = body["session_id"]
        api_key = body["api_key"]
        bu = body["base_urls"][fmt].replace("127.0.0.1:0", f"127.0.0.1:{port}")

        if fmt == "chat_completions":
            url = bu + "/chat/completions"
            hdrs = {"Authorization": f"Bearer {api_key}"}
        elif fmt == "responses":
            url = bu + "/responses"
            hdrs = {"Authorization": f"Bearer {api_key}"}
        elif fmt == "anthropic_messages":
            url = bu + "/messages"
            hdrs = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        else:
            url = bu + "/v1beta/models/" + urllib.parse.quote(name, safe="") + ":generateContent"
            hdrs = {}

        # --- turn 1 ---
        req1 = _build_turn1(fmt, name)
        st1, b1, dt1 = _post_json(url, hdrs, req1)
        frec["http_statuses"].append(st1)
        frec["timings_s"]["turn1"] = round(dt1, 4)
        text1, calls1 = _extract_assistant(fmt, b1)
        frec["turn1_tool_call"] = bool(calls1)

        # --- turn 2 ---
        req2 = _build_turn2(fmt, name, b1, text1, calls1)
        st2, b2, dt2 = _post_json(url, hdrs, req2)
        frec["http_statuses"].append(st2)
        frec["timings_s"]["turn2"] = round(dt2, 4)
        text2, calls2 = _extract_assistant(fmt, b2)
        frec["turn2_tool_call"] = bool(calls2)

        expected = [text1, text2]

        # --- end + trajectory ---
        st_e, b_e, dt_e = _post_json(
            f"{base}/fs/v1/sessions/{sid}/end", admin_h, {"status": "ok", "termination": "stop"}
        )
        frec["http_statuses"].append(st_e)
        frec["timings_s"]["end"] = round(dt_e, 4)

        meta = {
            "model": {"name": name, "version": 0, "checkpoint_sha": "probe"},
            "harness": {"name": "probe", "version": "0"},
            "env": {"kind": "fs_local", "task_id": "probe"},
        }
        st_t, traj, dt_t = _post_json(f"{base}/fs/v1/sessions/{sid}/trajectory", admin_h, meta)
        with Path(args.out.replace("report.json", f"episode_{fmt}.json")).open("w") as _fh:
            json.dump(traj, _fh)
        frec["http_statuses"].append(st_t)
        frec["timings_s"]["trajectory"] = round(dt_t, 4)

        chk = _check_trajectory(
            traj if isinstance(traj, dict) else {},
            decode,
            lambda ids: tok.decode(list(ids), skip_special_tokens=True),
            expected,
        )
        frec.update(chk)
        frec["policy_versions"] = _policy_versions(traj if isinstance(traj, dict) else {})

        ok = (
            st1 == 200
            and st2 == 200
            and st_e == 200
            and st_t == 200
            and chk["mask1_logprobs_finite"]
            and chk["mask0_logprobs_null"]
            and all(chk["fidelity_per_turn"])
            and chk["fragmentation"] == 0
        )
        frec["ok"] = bool(ok)
        if not ok:
            fail()
        report["formats"][fmt] = frec

    # ---------------- streaming chat (own session) ---------------- #
    srec: dict = {"http_statuses": [], "timings_s": {}}
    st, body, dt = _post_json(
        base + "/fs/v1/sessions", admin_h, {"mode": "train", "group_id": "stream_chat"}
    )
    srec["http_statuses"].append(st)
    srec["timings_s"]["create_session"] = round(dt, 4)
    sok = st == 201 and isinstance(body, dict)
    if sok:
        sid = body["session_id"]
        api_key = body["api_key"]
        url = (
            body["base_urls"]["chat_completions"].replace("127.0.0.1:0", f"127.0.0.1:{port}")
            + "/chat/completions"
        )
        hdrs = {"Authorization": f"Bearer {api_key}"}
        req = _build_turn1("chat_completions", name)
        req["stream"] = True
        st_s, rh, raw, dt_s = _http("POST", url, hdrs, req)
        srec["http_statuses"].append(st_s)
        srec["timings_s"]["stream"] = round(dt_s, 4)
        frames = _sse_frames(raw)
        srec["sse_frames"] = len(frames)
        srec["sse_parsed"] = all(
            _json_body(f.encode("utf-8")) is not None for f in frames if f and f != "[DONE]"
        )
        sok = st_s == 200 and len(frames) > 0 and srec["sse_parsed"]

        st_e, _, dt_e = _post_json(
            f"{base}/fs/v1/sessions/{sid}/end", admin_h, {"status": "ok", "termination": "stop"}
        )
        srec["http_statuses"].append(st_e)
        srec["timings_s"]["end"] = round(dt_e, 4)
        sok = sok and st_e == 200
    srec["ok"] = bool(sok)
    if not sok:
        fail()
    report["streaming_chat"] = srec

    report["ok"] = bool(all_ok)
    line = json.dumps(report, separators=(",", ":"), ensure_ascii=False)
    with Path(args.out).open("w", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return 0 if all_ok else 5


if __name__ == "__main__":
    sys.exit(main())
