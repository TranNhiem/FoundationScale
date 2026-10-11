"""Tests for foundationscale.agentic_rl.gateway.wire (binding API only)."""

import dataclasses
import json
import unittest

import pytest

from foundationscale.agentic_rl.gateway.wire import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResult,
    CanonicalTool,
    CanonicalToolCall,
    SamplingRequest,
    WireFormat,
    WireRefusal,
    detect_format,
    parse_request,
    render_response,
    render_stream,
)

CHAT = WireFormat.CHAT_COMPLETIONS
RESPONSES = WireFormat.RESPONSES
ANTHROPIC = WireFormat.ANTHROPIC_MESSAGES
GEMINI = WireFormat.GEMINI


def chat_body():
    """Build a realistic multi-turn tool-use Chat Completions request body."""
    return {
        "model": "m1",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C, sunny"},
            {"role": "user", "content": "And in Berlin?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    }


def responses_body():
    """Build a realistic multi-turn tool-use Responses request body."""
    return {
        "model": "m1",
        "instructions": "You are helpful.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Weather in Paris?"}],
            },
            {"type": "reasoning", "summary": [{"text": "I should call the tool."}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "18C, sunny"},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "And Berlin?"}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    }


def anthropic_body():
    """Build a realistic multi-turn tool-use Anthropic Messages request body."""
    return {
        "model": "m1",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need the tool."},
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "18C, sunny"},
                    {"type": "text", "text": "And Berlin?"},
                ],
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    }


def gemini_body():
    """Build a realistic multi-turn tool-use Gemini request body."""
    return {
        "systemInstruction": {"parts": [{"text": "You are helpful."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"result": "18C, sunny"},
                        }
                    },
                    {"text": "And Berlin?"},
                ],
            },
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            }
        ],
    }


def canonical_request(fmt=CHAT):
    """Build a CanonicalRequest matching the helper bodies."""
    return CanonicalRequest(
        format=fmt,
        model="m1",
        messages=(
            CanonicalMessage(role="system", content="You are helpful."),
            CanonicalMessage(role="user", content="Weather in Paris?"),
            CanonicalMessage(
                role="assistant",
                content="Let me check.",
                tool_calls=(
                    CanonicalToolCall(
                        call_id="call_1", name="get_weather", arguments_raw='{"city":"Paris"}'
                    ),
                ),
            ),
            CanonicalMessage(
                role="tool", content="18C, sunny", tool_call_id="call_1", name="get_weather"
            ),
            CanonicalMessage(role="user", content="And Berlin?"),
        ),
        tools=(
            CanonicalTool(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
        tool_choice=None,
        sampling=SamplingRequest(),
        stream=False,
        has_images=False,
        extras_dropped=(),
    )


def canonical_result():
    """Build a CanonicalResult with text and one tool call."""
    return CanonicalResult(
        text="It is 18C and sunny.",
        tool_calls=(
            CanonicalToolCall(
                call_id="call_2", name="get_weather", arguments_raw='{"city":"Berlin"}'
            ),
        ),
        reasoning="Call the tool.",
        finish_reason="tool_calls",
        prompt_tokens=11,
        completion_tokens=7,
        response_id="resp_1",
        created=1700000000,
    )


def sse_data(payloads):
    """Render JSON payloads as SSE data lines."""
    return ["data: " + json.dumps(p) + "\n\n" for p in payloads]


def parse_sse_data(events):
    """Extract and json.loads every data payload, skipping [DONE]."""
    out = []
    for ev in events:
        for line in ev.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                if payload.strip() == "[DONE]":
                    continue
                out.append(json.loads(payload))
    return out


class TestDetectFormat(unittest.TestCase):
    def test_detect_format_by_path(self):
        """Paths ending in the four route suffixes select the matching format."""
        self.assertEqual(detect_format("/v1/chat/completions", {}, {}), CHAT)
        self.assertEqual(detect_format("/v1/responses", {}, {}), RESPONSES)
        self.assertEqual(detect_format("/v1/messages", {}, {}), ANTHROPIC)
        self.assertEqual(detect_format("/v1beta/models/x:generateContent", {}, {}), GEMINI)
        self.assertEqual(detect_format("/v1beta/models/x:streamGenerateContent", {}, {}), GEMINI)

    def test_detect_format_by_anthropic_version_header(self):
        """The anthropic-version header selects ANTHROPIC regardless of case."""
        self.assertEqual(detect_format("/v1/x", {"anthropic-version": "2023-06-01"}, {}), ANTHROPIC)
        self.assertEqual(detect_format("/v1/x", {"Anthropic-Version": "2023-06-01"}, {}), ANTHROPIC)
        self.assertEqual(detect_format("/v1/x", {"ANTHROPIC-VERSION": "2023-06-01"}, {}), ANTHROPIC)

    def test_detect_format_by_body_shape(self):
        """Body shape (contents, input, messages) selects the format when path and header are
        silent.
        """
        self.assertEqual(detect_format("/v1/x", {}, {"contents": []}), GEMINI)
        self.assertEqual(
            detect_format("/v1/x", {}, {"input": "hi", "instructions": "sys"}), RESPONSES
        )
        self.assertEqual(detect_format("/v1/x", {}, {"input": "hi"}), RESPONSES)
        self.assertEqual(detect_format("/v1/x", {}, {"messages": []}), CHAT)

    def test_detect_format_refusal(self):
        """An undetectable request is refused."""
        with self.assertRaises(WireRefusal):
            detect_format("/v1/x", {}, {"foo": 1})


class TestParseRequest(unittest.TestCase):
    def test_parse_request_chat(self):
        """Chat Completions parses the multi-turn tool conversation into canonical form."""
        req = parse_request(CHAT, chat_body())
        self.assertEqual(
            [m.role for m in req.messages], ["system", "user", "assistant", "tool", "user"]
        )
        self.assertEqual(req.messages[0].content, "You are helpful.")
        self.assertEqual(req.messages[1].content, "What is the weather in Paris?")
        self.assertEqual(req.messages[2].content, "Let me check.")
        self.assertEqual(len(req.messages[2].tool_calls), 1)
        self.assertEqual(req.messages[2].tool_calls[0].call_id, "call_1")
        self.assertEqual(req.messages[2].tool_calls[0].name, "get_weather")
        self.assertEqual(req.messages[2].tool_calls[0].arguments_raw, '{"city":"Paris"}')
        self.assertEqual(req.messages[3].tool_call_id, "call_1")
        self.assertEqual(req.messages[3].content, "18C, sunny")
        self.assertEqual(req.messages[4].content, "And in Berlin?")
        self.assertEqual(req.tools[0].name, "get_weather")
        self.assertEqual(
            req.tools[0].parameters, {"type": "object", "properties": {"city": {"type": "string"}}}
        )

    def test_parse_request_responses(self):
        """Responses parses instructions, items, reasoning attachment and function_call_output."""
        req = parse_request(RESPONSES, responses_body())
        self.assertEqual(
            [m.role for m in req.messages], ["system", "user", "assistant", "tool", "user"]
        )
        self.assertEqual(req.messages[0].content, "You are helpful.")
        self.assertEqual(req.messages[1].content, "Weather in Paris?")
        self.assertEqual(req.messages[2].reasoning, "I should call the tool.")
        self.assertEqual(req.messages[2].tool_calls[0].call_id, "call_1")
        self.assertEqual(req.messages[2].tool_calls[0].name, "get_weather")
        self.assertEqual(req.messages[2].tool_calls[0].arguments_raw, '{"city":"Paris"}')
        self.assertEqual(req.messages[3].tool_call_id, "call_1")
        self.assertEqual(req.messages[3].content, "18C, sunny")
        self.assertEqual(req.messages[4].content, "And Berlin?")

    def test_parse_request_anthropic(self):
        """Anthropic parses system, thinking, tool_use and tool_result blocks to canonical form."""
        req = parse_request(ANTHROPIC, anthropic_body())
        self.assertEqual(
            [m.role for m in req.messages], ["system", "user", "assistant", "tool", "user"]
        )
        self.assertEqual(req.messages[0].content, "You are helpful.")
        self.assertEqual(req.messages[1].content, "Weather in Paris?")
        self.assertEqual(req.messages[2].reasoning, "Need the tool.")
        self.assertEqual(req.messages[2].content, "Let me check.")
        self.assertEqual(req.messages[2].tool_calls[0].call_id, "call_1")
        self.assertEqual(req.messages[2].tool_calls[0].name, "get_weather")
        self.assertEqual(req.messages[2].tool_calls[0].arguments_raw, '{"city":"Paris"}')
        self.assertEqual(req.messages[2].tool_calls[0].call_id_origin, "model")
        self.assertEqual(req.messages[3].tool_call_id, "call_1")
        self.assertEqual(req.messages[3].content, "18C, sunny")
        self.assertEqual(req.messages[4].content, "And Berlin?")

    def test_parse_request_gemini(self):
        """Gemini synthesizes call ids and matches functionResponse to the most recent unmatched
        call.
        """
        req = parse_request(GEMINI, gemini_body())
        self.assertEqual(
            [m.role for m in req.messages], ["system", "user", "assistant", "tool", "user"]
        )
        self.assertEqual(req.messages[0].content, "You are helpful.")
        self.assertEqual(req.messages[1].content, "Weather in Paris?")
        self.assertEqual(req.messages[2].tool_calls[0].call_id, "gemini:0")
        self.assertEqual(req.messages[2].tool_calls[0].call_id_origin, "synthesized")
        self.assertEqual(req.messages[2].tool_calls[0].name, "get_weather")
        self.assertEqual(req.messages[3].tool_call_id, "gemini:0")
        self.assertEqual(req.messages[4].content, "And Berlin?")

    def test_parse_request_gemini_unmatched_function_response_refused(self):
        """A Gemini functionResponse with no matching call is refused."""
        body = {
            "contents": [
                {"role": "user", "parts": [{"functionResponse": {"name": "nope", "response": {}}}]},
            ]
        }
        with self.assertRaises(WireRefusal):
            parse_request(GEMINI, body)

    def test_parse_request_responses_consecutive_function_calls_merge(self):
        """Consecutive Responses function_call items merge into one assistant message."""
        body = {
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            ]
        }
        req = parse_request(RESPONSES, body)
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, "assistant")
        self.assertEqual([c.call_id for c in req.messages[0].tool_calls], ["c1", "c2"])

    def test_parse_request_responses_reasoning_attaches_to_next_assistant(self):
        """A Responses reasoning item attaches as reasoning of the next assistant message."""
        body = {
            "input": [
                {"type": "reasoning", "summary": [{"text": "think"}]},
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
            ]
        }
        req = parse_request(RESPONSES, body)
        self.assertEqual(req.messages[0].reasoning, "think")

    def test_parse_request_images_set_has_images(self):
        """Image or file parts set has_images while text is kept."""
        chat = chat_body()
        chat["messages"][1]["content"] = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        self.assertTrue(parse_request(CHAT, chat).has_images)
        anth = anthropic_body()
        anth["messages"][0]["content"] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
            },
            {"type": "text", "text": "look"},
        ]
        self.assertTrue(parse_request(ANTHROPIC, anth).has_images)
        gem = gemini_body()
        gem["contents"][0]["parts"] = [
            {"inlineData": {"mimeType": "image/png", "data": "AAA"}},
            {"text": "look"},
        ]
        self.assertTrue(parse_request(GEMINI, gem).has_images)

    def test_parse_request_sampling_fields(self):
        """Sampling fields map per format, including Gemini candidateCount to n."""
        chat = chat_body()
        chat.update(
            {
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 128,
                "stop": ["x"],
                "n": 2,
                "seed": 7,
            }
        )
        s = parse_request(CHAT, chat).sampling
        self.assertEqual(
            (s.temperature, s.top_p, s.top_k, s.max_tokens, s.stop, s.n, s.seed),
            (0.5, 0.9, 40, 128, ("x",), 2, 7),
        )
        resp = responses_body()
        resp.update({"temperature": 0.4, "top_p": 0.8, "max_output_tokens": 64})
        s = parse_request(RESPONSES, resp).sampling
        self.assertEqual((s.temperature, s.top_p, s.max_tokens), (0.4, 0.8, 64))
        anth = anthropic_body()
        anth.update(
            {
                "temperature": 0.3,
                "top_p": 0.7,
                "top_k": 10,
                "max_tokens": 32,
                "stop_sequences": ["y"],
            }
        )
        s = parse_request(ANTHROPIC, anth).sampling
        self.assertEqual(
            (s.temperature, s.top_p, s.top_k, s.max_tokens, s.stop), (0.3, 0.7, 10, 32, ("y",))
        )
        gem = gemini_body()
        gem["generationConfig"] = {
            "temperature": 0.2,
            "topP": 0.6,
            "topK": 5,
            "maxOutputTokens": 16,
            "stopSequences": ["z"],
            "candidateCount": 3,
        }
        s = parse_request(GEMINI, gem).sampling
        self.assertEqual(
            (s.temperature, s.top_p, s.top_k, s.max_tokens, s.stop, s.n),
            (0.2, 0.6, 5, 16, ("z",), 3),
        )

    def test_parse_request_unknown_role_refused(self):
        """An unknown role is refused with the role named."""
        with self.assertRaises(WireRefusal) as ctx:
            parse_request(CHAT, {"messages": [{"role": "wizard", "content": "hi"}]})
        self.assertIn("wizard", str(ctx.exception))

    def test_parse_request_empty_messages_refused(self):
        """An empty message list after parsing is refused."""
        with self.assertRaises(WireRefusal):
            parse_request(CHAT, {"messages": []})
        with self.assertRaises(WireRefusal):
            parse_request(RESPONSES, {"input": []})
        with self.assertRaises(WireRefusal):
            parse_request(ANTHROPIC, {"messages": []})
        with self.assertRaises(WireRefusal):
            parse_request(GEMINI, {"contents": []})


class TestRenderResponse(unittest.TestCase):
    def test_render_response_chat(self):
        """Chat render uses chat.completion field names and maps finish_reason and usage."""
        out = render_response(CHAT, canonical_request(CHAT), canonical_result())
        self.assertEqual(out["object"], "chat.completion")
        self.assertEqual(out["id"], "resp_1")
        self.assertEqual(out["created"], 1700000000)
        self.assertEqual(out["model"], "m1")
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "It is 18C and sunny.")
        self.assertEqual(choice["message"]["reasoning_content"], "Call the tool.")
        tc = choice["message"]["tool_calls"][0]
        self.assertEqual(tc["id"], "call_2")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(tc["function"]["arguments"], '{"city":"Berlin"}')
        self.assertEqual(
            out["usage"], {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        )

    def test_render_response_responses(self):
        """Responses render emits reasoning, message and function_call output items with token
        usage.
        """
        out = render_response(RESPONSES, canonical_request(RESPONSES), canonical_result())
        self.assertEqual(out["object"], "response")
        self.assertEqual(out["created_at"], 1700000000)
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["output"][0]["type"], "reasoning")
        self.assertEqual(out["output"][0]["summary"][0]["text"], "Call the tool.")
        msg = out["output"][1]
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"][0], {"type": "output_text", "text": "It is 18C and sunny."})
        fc = out["output"][2]
        self.assertEqual(fc["type"], "function_call")
        self.assertEqual(fc["call_id"], "call_2")
        self.assertEqual(fc["name"], "get_weather")
        self.assertEqual(fc["arguments"], '{"city":"Berlin"}')
        self.assertEqual(out["usage"], {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})

    def test_render_response_responses_incomplete(self):
        """Responses render marks status incomplete with max_output_tokens reason on length."""
        res = CanonicalResult(
            text="partial",
            tool_calls=(),
            reasoning=None,
            finish_reason="length",
            prompt_tokens=1,
            completion_tokens=2,
            response_id="r",
            created=0,
        )
        out = render_response(RESPONSES, canonical_request(RESPONSES), res)
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["incomplete_details"], {"reason": "max_output_tokens"})

    def test_render_response_anthropic(self):
        """Anthropic render emits message content blocks and maps stop_reason and usage."""
        out = render_response(ANTHROPIC, canonical_request(ANTHROPIC), canonical_result())
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["model"], "m1")
        self.assertEqual(out["content"][0], {"type": "thinking", "thinking": "Call the tool."})
        self.assertEqual(out["content"][1], {"type": "text", "text": "It is 18C and sunny."})
        tool_use = out["content"][2]
        self.assertEqual(tool_use["type"], "tool_use")
        self.assertEqual(tool_use["id"], "call_2")
        self.assertEqual(tool_use["name"], "get_weather")
        self.assertEqual(tool_use["input"], {"city": "Berlin"})
        self.assertEqual(out["stop_reason"], "tool_use")
        self.assertIsNone(out["stop_sequence"])
        self.assertEqual(out["usage"], {"input_tokens": 11, "output_tokens": 7})

    def test_render_response_anthropic_non_json_arguments_refused(self):
        """Anthropic render refuses arguments_raw that is not valid JSON."""
        res = CanonicalResult(
            text=None,
            tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw="not json"),),
            reasoning=None,
            finish_reason="tool_calls",
            prompt_tokens=1,
            completion_tokens=1,
            response_id="r",
            created=0,
        )
        with self.assertRaises(WireRefusal):
            render_response(ANTHROPIC, canonical_request(ANTHROPIC), res)

    def test_render_response_gemini(self):
        """Gemini render emits candidates with parts, finishReason and usageMetadata."""
        out = render_response(GEMINI, canonical_request(GEMINI), canonical_result())
        cand = out["candidates"][0]
        self.assertEqual(cand["content"]["role"], "model")
        self.assertEqual(cand["content"]["parts"][0], {"text": "It is 18C and sunny."})
        self.assertEqual(
            cand["content"]["parts"][1],
            {"functionCall": {"name": "get_weather", "args": {"city": "Berlin"}}},
        )
        self.assertEqual(cand["finishReason"], "STOP")
        self.assertEqual(cand["index"], 0)
        self.assertEqual(
            out["usageMetadata"],
            {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
        )
        self.assertEqual(out["modelVersion"], "m1")

    def test_render_response_finish_reason_mapping(self):
        """finish_reason stop/length/tool_calls map to each format's native values."""
        for reason, chat_val, resp_status, anth_val, gem_val in [
            ("stop", "stop", "completed", "end_turn", "STOP"),
            ("length", "length", "incomplete", "max_tokens", "MAX_TOKENS"),
            ("tool_calls", "tool_calls", "completed", "tool_use", "STOP"),
        ]:
            res = CanonicalResult(
                text="hi",
                tool_calls=(),
                reasoning=None,
                finish_reason=reason,
                prompt_tokens=1,
                completion_tokens=1,
                response_id="r",
                created=0,
            )
            self.assertEqual(
                render_response(CHAT, canonical_request(CHAT), res)["choices"][0]["finish_reason"],
                chat_val,
            )
            self.assertEqual(
                render_response(RESPONSES, canonical_request(RESPONSES), res)["status"], resp_status
            )
            self.assertEqual(
                render_response(ANTHROPIC, canonical_request(ANTHROPIC), res)["stop_reason"],
                anth_val,
            )
            self.assertEqual(
                render_response(GEMINI, canonical_request(GEMINI), res)["candidates"][0][
                    "finishReason"
                ],
                gem_val,
            )


class TestRenderStream(unittest.TestCase):
    def test_render_stream_chat(self):
        """Chat stream reassembles to the canonical text and tool arguments and ends with [DONE]."""
        req, res = canonical_request(CHAT), canonical_result()
        events = render_stream(CHAT, req, res)
        self.assertTrue(events[-1].endswith("data: [DONE]\n\n"))
        text, args = "", ""
        for payload in parse_sse_data(events):
            delta = payload["choices"][0].get("delta", {})
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls", []) or []:
                args += tc["function"]["arguments"]
        self.assertEqual(text, res.text)
        self.assertEqual(args, res.tool_calls[0].arguments_raw)

    def test_render_stream_responses(self):
        """Responses stream reassembles text and function arguments and ends with
        response.completed.
        """
        req, res = canonical_request(RESPONSES), canonical_result()
        events = render_stream(RESPONSES, req, res)
        payloads = parse_sse_data(events)
        self.assertEqual(payloads[-1]["type"], "response.completed")
        text, args = "", ""
        for p in payloads:
            if p.get("type") == "response.output_text.delta":
                text += p["delta"]
            elif p.get("type") == "response.function_call_arguments.delta":
                args += p["delta"]
        self.assertEqual(text, res.text)
        self.assertEqual(args, res.tool_calls[0].arguments_raw)

    def test_render_stream_anthropic(self):
        """Anthropic stream reassembles text and input_json deltas in message_start..message_stop
        order.
        """
        req, res = canonical_request(ANTHROPIC), canonical_result()
        events = render_stream(ANTHROPIC, req, res)
        kinds = []
        for ev in events:
            for line in ev.splitlines():
                if line.startswith("event: "):
                    kinds.append(line[len("event: ") :])
        self.assertEqual(kinds[0], "message_start")
        self.assertEqual(kinds[-1], "message_stop")
        self.assertIn("message_delta", kinds)
        text, args = "", ""
        for p in parse_sse_data(events):
            delta = p.get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta["text"]
            elif delta.get("type") == "input_json_delta":
                args += delta["partial_json"]
        self.assertEqual(text, res.text)
        self.assertEqual(args, res.tool_calls[0].arguments_raw)

    def test_render_stream_gemini(self):
        """Gemini stream reassembles text and functionCall args from GenerateContentResponse data
        lines.
        """
        req, res = canonical_request(GEMINI), canonical_result()
        events = render_stream(GEMINI, req, res)
        text, args = "", ""
        for p in parse_sse_data(events):
            for part in p["candidates"][0]["content"]["parts"]:
                text += part.get("text") or ""
                if "functionCall" in part:
                    args += json.dumps(part["functionCall"]["args"], separators=(",", ":"))
        self.assertEqual(text, res.text)
        self.assertEqual(json.loads(args), json.loads(res.tool_calls[0].arguments_raw))


if __name__ == "__main__":
    unittest.main()


def test_anthropic_stream_payloads_carry_their_event_type() -> None:
    """Every Anthropic SSE data payload repeats its event name as "type" (SDKs dispatch on it)."""
    from foundationscale.agentic_rl.gateway.wire import render_stream

    events = render_stream(ANTHROPIC, canonical_request(ANTHROPIC), canonical_result())
    for frame in events:
        name_line, data_line = frame.strip().split("\n")
        assert name_line.startswith("event: ")
        assert json.loads(data_line[len("data: ") :])["type"] == name_line[len("event: ") :]


def test_parse_request_accepts_a_body_without_model() -> None:
    """Gemini carries the model in the URL, so a body without "model" parses with model None."""
    from foundationscale.agentic_rl.gateway.wire import parse_request

    req = parse_request(GEMINI, {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert req.model is None


def test_require_mapping_refuses_non_mapping() -> None:
    """detect_format refuses headers that are not a mapping."""
    with pytest.raises(WireRefusal, match="headers to be a mapping"):
        detect_format("/v1/x", ["not", "a", "mapping"], {})


def test_require_mapping_refuses_body_non_mapping() -> None:
    """parse_request refuses a body that is not a mapping."""
    with pytest.raises(WireRefusal, match="body to be a mapping"):
        parse_request(CHAT, 42)


def test_require_str_refuses_non_string() -> None:
    """A chat message role that is not a string is refused."""
    with pytest.raises(WireRefusal, match="message 'role' to be a string"):
        parse_request(CHAT, {"messages": [{"role": 5, "content": "hi"}]})


def test_optional_str_refuses_non_string() -> None:
    """A chat tool_call_id that is not a string is refused."""
    body = chat_body()
    body["messages"][3]["tool_call_id"] = 7
    with pytest.raises(WireRefusal, match="tool_call_id to be a string"):
        parse_request(CHAT, body)


def test_require_int_refuses_bool() -> None:
    """A bool is refused where an integer is required (bool subclasses int)."""
    body = chat_body()
    body["max_tokens"] = True
    with pytest.raises(WireRefusal, match="max_tokens/max_completion_tokens to be an integer"):
        parse_request(CHAT, body)


def test_require_int_refuses_non_int() -> None:
    """A non-integer max_tokens is refused."""
    body = chat_body()
    body["max_tokens"] = "128"
    with pytest.raises(WireRefusal, match="max_tokens/max_completion_tokens to be an integer"):
        parse_request(CHAT, body)


def test_optional_int_refuses_non_int() -> None:
    """A non-integer top_k is refused."""
    body = chat_body()
    body["top_k"] = 1.5
    with pytest.raises(WireRefusal, match="top_k to be an integer"):
        parse_request(CHAT, body)


def test_require_float_refuses_bool() -> None:
    """A bool is refused where a number is required."""
    body = chat_body()
    body["temperature"] = True
    with pytest.raises(WireRefusal, match="temperature to be a number"):
        parse_request(CHAT, body)


def test_require_float_refuses_non_number() -> None:
    """A non-numeric temperature is refused."""
    body = chat_body()
    body["temperature"] = "hot"
    with pytest.raises(WireRefusal, match="temperature to be a number"):
        parse_request(CHAT, body)


def test_optional_float_refuses_non_number() -> None:
    """A non-numeric top_p is refused."""
    body = chat_body()
    body["top_p"] = "wide"
    with pytest.raises(WireRefusal, match="top_p to be a number"):
        parse_request(CHAT, body)


def test_require_bool_refuses_non_bool() -> None:
    """A non-boolean stream flag is refused."""
    body = chat_body()
    body["stream"] = "yes"
    with pytest.raises(WireRefusal, match="stream to be a boolean"):
        parse_request(CHAT, body)


def test_optional_bool_refuses_non_bool() -> None:
    """A non-boolean stream flag is refused in the Responses parser."""
    body = responses_body()
    body["stream"] = 1
    with pytest.raises(WireRefusal, match="stream to be a boolean"):
        parse_request(RESPONSES, body)


def test_require_sequence_refuses_string() -> None:
    """A string is refused where a sequence of messages is required."""
    with pytest.raises(WireRefusal, match="messages to be a list"):
        parse_request(CHAT, {"messages": "hello"})


def test_require_sequence_refuses_non_sequence() -> None:
    """A non-sequence messages value is refused."""
    with pytest.raises(WireRefusal, match="messages to be a list"):
        parse_request(CHAT, {"messages": 5})


def test_stop_tuple_refuses_non_string_entry() -> None:
    """A non-string stop entry is refused."""
    body = chat_body()
    body["stop"] = ["ok", 3]
    with pytest.raises(WireRefusal, match="each stop entry to be a string"):
        parse_request(CHAT, body)


def test_check_call_id_origin_refuses_bad_origin() -> None:
    """A tool call with an unknown call_id_origin is refused."""
    msg = dataclasses.replace(
        canonical_request().messages[2],
        tool_calls=(
            CanonicalToolCall(call_id="c", name="f", arguments_raw="{}", call_id_origin="invented"),
        ),
    )
    req = dataclasses.replace(canonical_request(), messages=(msg,))
    with pytest.raises(WireRefusal, match="call_id_origin to be 'model' or 'synthesized'"):
        render_response(CHAT, req, canonical_result())


def test_check_finish_reason_refuses_bad_reason() -> None:
    """An unknown finish_reason is refused before rendering."""
    res = dataclasses.replace(canonical_result(), finish_reason="exploded")
    with pytest.raises(WireRefusal, match="finish_reason to be one of"):
        render_response(CHAT, canonical_request(), res)


def test_validate_message_tool_role_requires_tool_call_id() -> None:
    """A role 'tool' message without tool_call_id is refused."""
    req = dataclasses.replace(
        canonical_request(), messages=(CanonicalMessage(role="tool", content="x"),)
    )
    with pytest.raises(WireRefusal, match="tool_call_id to be present for role 'tool'"):
        render_response(CHAT, req, canonical_result())


def test_validate_message_tool_role_rejects_tool_calls() -> None:
    """A role 'tool' message carrying tool_calls is refused."""
    msg = CanonicalMessage(
        role="tool",
        content="x",
        tool_call_id="c1",
        tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw="{}"),),
    )
    req = dataclasses.replace(canonical_request(), messages=(msg,))
    with pytest.raises(WireRefusal, match="no tool_calls on a role 'tool' message"):
        render_response(CHAT, req, canonical_result())


def test_validate_message_tool_call_id_absent_for_non_tool() -> None:
    """A non-tool message carrying tool_call_id is refused."""
    msg = CanonicalMessage(role="user", content="x", tool_call_id="c1")
    req = dataclasses.replace(canonical_request(), messages=(msg,))
    with pytest.raises(WireRefusal, match="tool_call_id to be absent for role 'user'"):
        render_response(CHAT, req, canonical_result())


def test_validate_message_tool_calls_only_on_assistant() -> None:
    """A user message carrying tool_calls is refused."""
    msg = CanonicalMessage(
        role="user",
        content="x",
        tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw="{}"),),
    )
    req = dataclasses.replace(canonical_request(), messages=(msg,))
    with pytest.raises(WireRefusal, match="tool_calls only on role 'assistant' messages"):
        render_response(CHAT, req, canonical_result())


def test_optional_model_refuses_empty_string() -> None:
    """An empty model string is refused as not a non-empty string."""
    body = chat_body()
    body["model"] = ""
    with pytest.raises(WireRefusal, match="non-empty string or absent"):
        parse_request(CHAT, body)


def test_optional_model_refuses_non_string() -> None:
    """A non-string model is refused."""
    body = chat_body()
    body["model"] = 3
    with pytest.raises(WireRefusal, match="non-empty string or absent"):
        parse_request(CHAT, body)


def test_detect_format_refuses_non_string_path() -> None:
    """detect_format refuses a path that is not a string."""
    with pytest.raises(WireRefusal, match="path to be a string"):
        detect_format(None, {}, {})


def test_parse_request_refuses_non_wire_format() -> None:
    """parse_request refuses a fmt that is not a WireFormat."""
    with pytest.raises(WireRefusal, match="fmt to be a WireFormat"):
        parse_request("chat_completions", chat_body())


def test_render_response_refuses_non_wire_format() -> None:
    """render_response refuses a fmt that is not a WireFormat."""
    with pytest.raises(WireRefusal, match="fmt to be a WireFormat"):
        render_response("chat_completions", canonical_request(), canonical_result())


def test_render_stream_refuses_non_wire_format() -> None:
    """render_stream refuses a fmt that is not a WireFormat."""
    with pytest.raises(WireRefusal, match="fmt to be a WireFormat"):
        render_stream("chat_completions", canonical_request(), canonical_result())


def test_chat_null_content_is_kept_as_none() -> None:
    """A chat message with null content parses with content None."""
    body = chat_body()
    body["messages"] = [{"role": "user", "content": None}]
    req = parse_request(CHAT, body)
    assert req.messages[0].content is None


def test_chat_content_part_bad_type_refused() -> None:
    """A chat content part with an unknown type is refused."""
    body = chat_body()
    body["messages"][1]["content"] = [{"type": "audio", "text": "x"}]
    with pytest.raises(WireRefusal, match="content part 'type' to be 'text'"):
        parse_request(CHAT, body)


def test_chat_content_bad_shape_refused() -> None:
    """A chat message content that is neither string, list nor null is refused."""
    body = chat_body()
    body["messages"][1]["content"] = 42
    with pytest.raises(WireRefusal, match="message 'content' to be a string"):
        parse_request(CHAT, body)


def test_chat_tool_calls_only_on_assistant() -> None:
    """A chat user message carrying tool_calls is refused."""
    body = chat_body()
    body["messages"][1]["tool_calls"] = [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]
    with pytest.raises(WireRefusal, match="tool_calls only on assistant messages"):
        parse_request(CHAT, body)


def test_chat_tool_call_function_arguments_non_string_refused() -> None:
    """A chat tool_call function arguments that is not a JSON string is refused."""
    body = chat_body()
    body["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris"}
    with pytest.raises(WireRefusal, match="tool_call function 'arguments' to be a JSON string"):
        parse_request(CHAT, body)


def test_chat_max_tokens_and_max_completion_tokens_both_refused() -> None:
    """Sending both max_tokens and max_completion_tokens is refused."""
    body = chat_body()
    body["max_tokens"] = 10
    body["max_completion_tokens"] = 20
    with pytest.raises(WireRefusal, match="only one of 'max_tokens' or 'max_completion_tokens'"):
        parse_request(CHAT, body)


def test_chat_extras_dropped_records_documented_keys() -> None:
    """Documented droppable chat fields are recorded in extras_dropped."""
    body = chat_body()
    body.update({"user": "u", "metadata": {}, "logprobs": True, "store": False})
    req = parse_request(CHAT, body)
    assert req.extras_dropped == ("user", "metadata", "logprobs", "store")


def test_chat_tool_type_must_be_function() -> None:
    """A chat tool with a non-function type is refused."""
    body = chat_body()
    body["tools"][0]["type"] = "code_interpreter"
    with pytest.raises(WireRefusal, match="tool 'type' to be 'function'"):
        parse_request(CHAT, body)


def test_chat_tool_parameters_required() -> None:
    """A chat tool function without parameters is refused."""
    body = chat_body()
    del body["tools"][0]["function"]["parameters"]
    with pytest.raises(WireRefusal, match="tool function 'parameters' to be a JSON schema object"):
        parse_request(CHAT, body)


def test_chat_tool_choice_bad_type_refused() -> None:
    """A chat tool_choice that is not a string, object or null is refused."""
    body = chat_body()
    body["tool_choice"] = 5
    with pytest.raises(WireRefusal, match="tool_choice to be a string, an object, or null"):
        parse_request(CHAT, body)


def test_chat_tool_choice_string_and_mapping_accepted() -> None:
    """A chat tool_choice string or mapping is recorded verbatim."""
    body = chat_body()
    body["tool_choice"] = "auto"
    assert parse_request(CHAT, body).tool_choice == "auto"
    body["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
    assert parse_request(CHAT, body).tool_choice == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_responses_missing_input_refused() -> None:
    """A Responses body without input is refused."""
    with pytest.raises(WireRefusal, match="input to be a string or a list of input items"):
        parse_request(RESPONSES, {"model": "m1"})


def test_responses_string_input_becomes_user_message() -> None:
    """A Responses string input becomes a single user message."""
    req = parse_request(RESPONSES, {"input": "hello"})
    assert req.messages == (CanonicalMessage(role="user", content="hello"),)


def test_responses_input_part_bad_type_refused() -> None:
    """A Responses input message content part with an unknown type is refused."""
    body = responses_body()
    body["input"][0]["content"] = [{"type": "audio", "text": "x"}]
    with pytest.raises(WireRefusal, match="input message content part 'type' to be 'input_text'"):
        parse_request(RESPONSES, body)


def test_responses_tool_type_must_be_function() -> None:
    """A Responses tool with a non-function type is refused."""
    body = responses_body()
    body["tools"][0]["type"] = "web_search"
    with pytest.raises(WireRefusal, match="tool 'type' to be 'function'"):
        parse_request(RESPONSES, body)


def test_responses_tool_parameters_required() -> None:
    """A Responses tool without parameters is refused."""
    body = responses_body()
    del body["tools"][0]["parameters"]
    with pytest.raises(WireRefusal, match="tool 'parameters' to be a JSON schema object"):
        parse_request(RESPONSES, body)


def test_responses_tool_choice_bad_type_refused() -> None:
    """A Responses tool_choice that is not a string, object or null is refused."""
    body = responses_body()
    body["tool_choice"] = 3
    with pytest.raises(WireRefusal, match="tool_choice to be a string, an object, or null"):
        parse_request(RESPONSES, body)


def test_responses_tool_choice_string_and_mapping_accepted() -> None:
    """A Responses tool_choice string or mapping is recorded verbatim."""
    body = responses_body()
    body["tool_choice"] = "required"
    assert parse_request(RESPONSES, body).tool_choice == "required"
    body["tool_choice"] = {"type": "function"}
    assert parse_request(RESPONSES, body).tool_choice == {"type": "function"}


def test_responses_extras_dropped_records_documented_keys() -> None:
    """Documented droppable Responses fields are recorded in extras_dropped."""
    body = responses_body()
    body.update({"metadata": {}, "user": "u", "store": True, "truncation": "auto"})
    req = parse_request(RESPONSES, body)
    assert req.extras_dropped == ("metadata", "user", "store", "truncation")


def test_anthropic_system_blocks_parsed() -> None:
    """An Anthropic system block list is joined into one system message."""
    body = anthropic_body()
    body["system"] = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    req = parse_request(ANTHROPIC, body)
    assert req.messages[0].content == "ab"


def test_anthropic_system_block_bad_type_refused() -> None:
    """An Anthropic system block with a non-text type is refused."""
    body = anthropic_body()
    body["system"] = [{"type": "image", "text": "x"}]
    with pytest.raises(WireRefusal, match="system block 'type' to be 'text'"):
        parse_request(ANTHROPIC, body)


def test_anthropic_message_role_must_be_user_or_assistant() -> None:
    """An Anthropic message role outside user/assistant is refused."""
    body = anthropic_body()
    body["messages"][0]["role"] = "system"
    with pytest.raises(WireRefusal, match="message 'role' to be 'user' or 'assistant'"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_use_only_on_assistant() -> None:
    """An Anthropic tool_use block on a user message is refused."""
    body = anthropic_body()
    body["messages"][0]["content"] = [{"type": "tool_use", "id": "c", "name": "f", "input": {}}]
    with pytest.raises(WireRefusal, match="tool_use blocks only on assistant messages"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_use_input_required() -> None:
    """An Anthropic tool_use block without input is refused."""
    body = anthropic_body()
    body["messages"][1]["content"][2] = {"type": "tool_use", "id": "c", "name": "f"}
    with pytest.raises(WireRefusal, match="tool_use 'input' to be a JSON object"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_result_content_blocks_parsed() -> None:
    """An Anthropic tool_result content block list is joined into one tool message."""
    body = anthropic_body()
    body["messages"][2]["content"][0]["content"] = [
        {"type": "text", "text": "18C"},
        {"type": "text", "text": ", sunny"},
    ]
    req = parse_request(ANTHROPIC, body)
    assert req.messages[3].content == "18C, sunny"


def test_anthropic_tool_result_block_bad_type_refused() -> None:
    """An Anthropic tool_result content block with a non-text type is refused."""
    body = anthropic_body()
    body["messages"][2]["content"][0]["content"] = [{"type": "image", "text": "x"}]
    with pytest.raises(WireRefusal, match="tool_result content block 'type' to be 'text'"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_result_block_text_must_be_string() -> None:
    """An Anthropic tool_result block text that is not a string is refused."""
    body = anthropic_body()
    body["messages"][2]["content"][0]["content"] = [{"type": "text", "text": 5}]
    with pytest.raises(WireRefusal, match="tool_result block 'text' to be a string"):
        parse_request(ANTHROPIC, body)


def test_anthropic_thinking_falls_back_to_text_key() -> None:
    """An Anthropic thinking block without 'thinking' falls back to its 'text' key."""
    body = anthropic_body()
    body["messages"][1]["content"][0] = {"type": "thinking", "text": "fallback"}
    req = parse_request(ANTHROPIC, body)
    assert req.messages[2].reasoning == "fallback"


def test_anthropic_content_block_bad_type_refused() -> None:
    """An Anthropic content block with an unknown type is refused."""
    body = anthropic_body()
    body["messages"][0]["content"] = [{"type": "audio", "text": "x"}]
    with pytest.raises(WireRefusal, match="content block 'type' to be 'text'"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_input_schema_required() -> None:
    """An Anthropic tool without input_schema is refused."""
    body = anthropic_body()
    del body["tools"][0]["input_schema"]
    with pytest.raises(WireRefusal, match="tool 'input_schema' to be a JSON schema object"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_choice_bad_type_refused() -> None:
    """An Anthropic tool_choice that is not a string, object or null is refused."""
    body = anthropic_body()
    body["tool_choice"] = 9
    with pytest.raises(WireRefusal, match="tool_choice to be a string, an object, or null"):
        parse_request(ANTHROPIC, body)


def test_anthropic_tool_choice_string_and_mapping_accepted() -> None:
    """An Anthropic tool_choice string or mapping is recorded verbatim."""
    body = anthropic_body()
    body["tool_choice"] = "any"
    assert parse_request(ANTHROPIC, body).tool_choice == "any"
    body["tool_choice"] = {"type": "tool", "name": "get_weather"}
    assert parse_request(ANTHROPIC, body).tool_choice == {"type": "tool", "name": "get_weather"}


def test_anthropic_extras_dropped_records_documented_keys() -> None:
    """Documented droppable Anthropic fields are recorded in extras_dropped."""
    body = anthropic_body()
    body.update({"metadata": {}, "user": "u"})
    req = parse_request(ANTHROPIC, body)
    assert req.extras_dropped == ("metadata", "user")


def test_gemini_content_role_missing_refused() -> None:
    """A Gemini content item without role is refused."""
    body = gemini_body()
    body["contents"][0] = {"parts": [{"text": "hi"}]}
    with pytest.raises(WireRefusal, match="content 'role' to be 'user' or 'model'"):
        parse_request(GEMINI, body)


def test_gemini_content_role_bad_value_refused() -> None:
    """A Gemini content role outside user/model is refused."""
    body = gemini_body()
    body["contents"][0]["role"] = "system"
    with pytest.raises(WireRefusal, match="content 'role' to be 'user' or 'model'"):
        parse_request(GEMINI, body)


def test_gemini_function_call_only_on_model() -> None:
    """A Gemini functionCall part on a user content is refused."""
    body = gemini_body()
    body["contents"][0]["parts"] = [{"functionCall": {"name": "f", "args": {}}}]
    with pytest.raises(WireRefusal, match="functionCall parts only on 'model' contents"):
        parse_request(GEMINI, body)


def test_gemini_function_call_args_default_empty_object() -> None:
    """A Gemini functionCall without args defaults to an empty JSON object."""
    body = gemini_body()
    body["contents"][1]["parts"][0]["functionCall"] = {"name": "get_weather"}
    req = parse_request(GEMINI, body)
    assert req.messages[2].tool_calls[0].arguments_raw == "{}"


def test_gemini_function_response_response_required() -> None:
    """A Gemini functionResponse without response is refused."""
    body = gemini_body()
    body["contents"][2]["parts"][0]["functionResponse"] = {"name": "get_weather"}
    with pytest.raises(WireRefusal, match="functionResponse 'response' to be a JSON object"):
        parse_request(GEMINI, body)


def test_gemini_part_bad_keys_refused() -> None:
    """A Gemini content part with none of the known keys is refused."""
    body = gemini_body()
    body["contents"][0]["parts"] = [{"audio": {}}]
    with pytest.raises(WireRefusal, match="content part to contain 'text'"):
        parse_request(GEMINI, body)


def test_gemini_function_declaration_parameters_required() -> None:
    """A Gemini function declaration without parameters is refused."""
    body = gemini_body()
    del body["tools"][0]["functionDeclarations"][0]["parameters"]
    with pytest.raises(WireRefusal, match="function declaration 'parameters' to be a JSON schema"):
        parse_request(GEMINI, body)


def test_gemini_tool_choice_bad_type_refused() -> None:
    """A Gemini tool_choice that is not a string, object or null is refused."""
    body = gemini_body()
    body["tool_choice"] = 2
    with pytest.raises(WireRefusal, match="tool_choice to be a string, an object, or null"):
        parse_request(GEMINI, body)


def test_gemini_tool_choice_string_and_mapping_accepted() -> None:
    """A Gemini tool_choice string or mapping is recorded verbatim."""
    body = gemini_body()
    body["tool_choice"] = "auto"
    assert parse_request(GEMINI, body).tool_choice == "auto"
    body["tool_choice"] = {"functionCallingConfig": {"mode": "ANY"}}
    assert parse_request(GEMINI, body).tool_choice == {"functionCallingConfig": {"mode": "ANY"}}


def test_gemini_extras_dropped_records_documented_keys() -> None:
    """Documented droppable Gemini fields are recorded in extras_dropped."""
    body = gemini_body()
    body.update({"safetySettings": [], "cachedContent": "c"})
    req = parse_request(GEMINI, body)
    assert req.extras_dropped == ("safetySettings", "cachedContent")


def test_check_result_refuses_non_canonical_result() -> None:
    """render_response refuses a res that is not a CanonicalResult."""
    with pytest.raises(WireRefusal, match="res to be a CanonicalResult"):
        render_response(CHAT, canonical_request(), {"text": "x"})


def test_check_result_refuses_non_string_text() -> None:
    """A CanonicalResult with non-string text is refused."""
    res = dataclasses.replace(canonical_result(), text=5)
    with pytest.raises(WireRefusal, match="text to be a string or None"):
        render_response(CHAT, canonical_request(), res)


def test_check_result_refuses_non_string_reasoning() -> None:
    """A CanonicalResult with non-string reasoning is refused."""
    res = dataclasses.replace(canonical_result(), reasoning=5)
    with pytest.raises(WireRefusal, match="reasoning to be a string or None"):
        render_response(CHAT, canonical_request(), res)


def test_check_request_refuses_non_canonical_request() -> None:
    """render_response refuses a req that is not a CanonicalRequest."""
    with pytest.raises(WireRefusal, match="req to be a CanonicalRequest"):
        render_response(CHAT, {"messages": []}, canonical_result())


def test_anthropic_tool_input_invalid_json_refused() -> None:
    """Anthropic render refuses arguments_raw that is not valid JSON."""
    res = dataclasses.replace(
        canonical_result(),
        tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw="{oops"),),
    )
    with pytest.raises(WireRefusal, match="valid JSON so it can be rendered as an Anthropic"):
        render_response(ANTHROPIC, canonical_request(ANTHROPIC), res)


def test_anthropic_tool_input_non_object_refused() -> None:
    """Anthropic render refuses arguments_raw that decodes to a non-object."""
    res = dataclasses.replace(
        canonical_result(),
        tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw="[1,2]"),),
    )
    with pytest.raises(WireRefusal, match="decode to a JSON object for Anthropic"):
        render_response(ANTHROPIC, canonical_request(ANTHROPIC), res)


def test_gemini_function_args_invalid_json_refused() -> None:
    """Gemini render refuses arguments_raw that is not valid JSON."""
    res = dataclasses.replace(
        canonical_result(),
        tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw="{oops"),),
    )
    with pytest.raises(WireRefusal, match="valid JSON so it can be rendered as a Gemini"):
        render_response(GEMINI, canonical_request(GEMINI), res)


def test_gemini_function_args_non_object_refused() -> None:
    """Gemini render refuses arguments_raw that decodes to a non-object."""
    res = dataclasses.replace(
        canonical_result(),
        tool_calls=(CanonicalToolCall(call_id="c", name="f", arguments_raw='"str"'),),
    )
    with pytest.raises(WireRefusal, match="decode to a JSON object for Gemini"):
        render_response(GEMINI, canonical_request(GEMINI), res)


def test_responses_stream_length_sets_incomplete_details() -> None:
    """A Responses stream with finish_reason length carries incomplete_details."""
    res = dataclasses.replace(canonical_result(), finish_reason="length")
    events = render_stream(RESPONSES, canonical_request(RESPONSES), res)
    payloads = parse_sse_data(events)
    assert payloads[-1]["incomplete_details"] == {"reason": "max_output_tokens"}


def test_gemini_stream_empty_result_emits_empty_part() -> None:
    """A Gemini stream with no text and no tool calls emits one empty part chunk."""
    res = dataclasses.replace(canonical_result(), text=None, tool_calls=())
    events = render_stream(GEMINI, canonical_request(GEMINI), res)
    payloads = parse_sse_data(events)
    assert payloads[0]["candidates"][0]["content"]["parts"] == [{}]
