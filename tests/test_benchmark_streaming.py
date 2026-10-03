"""Byte-stream fakes over httpx.MockTransport; no real HTTP service."""

import asyncio
import json

import httpx
import pytest

from qwen_vllm_production.benchmark.client import StreamingClient
from qwen_vllm_production.benchmark.sse import SSEDecoder
from qwen_vllm_production.benchmark.workload import Prompt


class FakeTokenizer:
    def encode(self, text: str, **kwargs) -> list[str]:
        return text.split()


class FakeBytes(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self.chunks, self.error = chunks, error

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


def event(payload) -> bytes:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ("data: " + text + "\n\n").encode()


def measure(chunks: list[bytes], *, status: int = 200, error: Exception | None = None):
    async def invoke():
        tick = -1

        def clock():
            nonlocal tick
            tick += 1
            return tick * 0.1

        def handle(request):
            payload = json.loads(request.content)
            assert payload["stream_options"]["include_usage"] is True
            return httpx.Response(status, headers={"content-type": "text/event-stream"}, stream=FakeBytes(chunks, error))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            client = StreamingClient(http, FakeTokenizer(), base_url="http://fake", model="model", clock=clock)
            return await client.send(Prompt(0, "prompt", 1000, "hash"), 0, 0, 1)

    return asyncio.run(invoke())


def test_role_empty_content_usage_and_done_timing() -> None:
    result = measure([
        event({"model": "model", "choices": [{"delta": {"role": "assistant"}}]}),
        event({"choices": [{"delta": {"content": ""}}]}),
        event({"choices": [{"delta": {"content": "hello "}}]}),
        event({"choices": [{"delta": {"content": "world"}}]}),
        event({"choices": [], "usage": {"completion_tokens": 4, "prompt_tokens": 1000}}),
        event("[DONE]"),
    ])
    assert result.success
    assert result.ttft_ms == pytest.approx(300)
    assert result.e2e_ms == pytest.approx(600)
    assert result.tpot_ms_per_token == pytest.approx(100)
    assert result.generated_text == "hello world"
    assert result.output_token_count_source == "api_usage"
    assert result.first_content_timestamp is not None


def test_partial_bytes_and_tokenizer_fallback() -> None:
    wire = event({"choices": [{"text": "hello world"}]}) + event("[DONE]")
    result = measure([wire[index:index + 7] for index in range(0, len(wire), 7)])
    assert result.success
    assert result.output_tokens == 2
    assert result.output_token_count_source == "tokenizer_fallback"


@pytest.mark.parametrize("chunks", [
    [event("{malformed"), event("[DONE]")],
    [event({"choices": [{"text": "hello"}]})],
    [event({"choices": [{"delta": {"role": "assistant"}}]}), event("[DONE]")],
    [event({"error": {"message": "server failed"}})],
    [event({"unexpected": True})],
    [event({"model": "wrong", "choices": [{"text": "hello"}]}), event("[DONE]")],
])
def test_stream_failures_are_preserved(chunks: list[bytes]) -> None:
    result = measure(chunks)
    assert not result.success
    assert result.error_type == "StreamProtocolError"
    assert result.error_message
    assert not result.slo_compliant


def test_http_error_and_timeout_are_results() -> None:
    assert measure([], status=500).error_type == "HTTPStatusError"
    timeout = measure([], error=httpx.ReadTimeout("mocked timeout"))
    assert not timeout.success
    assert timeout.error_type == "ReadTimeout"


def test_total_timeout_keeps_a_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTimeout:
        async def __aenter__(self):
            raise TimeoutError()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(asyncio, "timeout", lambda seconds: FakeTimeout())
    result = measure([])
    assert not result.success
    assert result.error_type == "TimeoutError"
    assert "Request timeout" in result.error_message


def test_incremental_utf8_comments_crlf_and_multiline_events() -> None:
    decoder = SSEDecoder()
    wire = ": keepalive\r\ndata: 你好\r\ndata: second line\r\n\r\n".encode()
    events = []
    for byte in wire:
        events.extend(decoder.feed(bytes([byte])))
    assert events == ["你好\nsecond line"]
