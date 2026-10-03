"""HTTP streaming measurements; one failed request remains a raw result."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from .metrics import RequestResult, latency_metrics
from .slo import request_compliant
from .sse import SSEDecoder, StreamProtocolError
from .workload import Prompt


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StreamingClient:
    def __init__(
        self, http: httpx.AsyncClient, tokenizer: Any, *, base_url: str,
        model: str, output_tokens: int = 256, timeout: float = 120.0,
        request_path: str = "/v1/completions", ignore_eos: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if request_path != "/v1/completions":
            raise ValueError("Phase 3A uses /v1/completions with locally rendered chat prompts.")
        self.http, self.tokenizer = http, tokenizer
        self.base_url, self.model = base_url.rstrip("/"), model
        self.output_tokens, self.timeout = output_tokens, timeout
        self.request_path, self.ignore_eos, self.clock = request_path, ignore_eos, clock

    async def send(self, prompt: Prompt, request_id: int, worker_id: int, concurrency: int) -> RequestResult:
        result = RequestResult(
            request_id=request_id, model=self.model, concurrency=concurrency,
            worker_id=worker_id, start_timestamp=utc_now(),
            prompt_tokens=prompt.prompt_tokens, prompt_sha256=prompt.sha256,
            requested_output_tokens=self.output_tokens,
        )
        start = self.clock()
        result.request_start_monotonic_seconds = start
        first: float | None = None
        completion: float | None = None
        usage: dict | None = None
        parts: list[str] = []
        done = False
        decoder = SSEDecoder()

        def consume(data: str, received_at: float) -> None:
            nonlocal first, completion, usage, done
            if not data.strip():
                return
            if data.strip() == "[DONE]":
                done, completion = True, received_at
                result.stream_complete_monotonic_seconds = received_at
                result.end_timestamp = utc_now()
                return
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise StreamProtocolError("Malformed JSON in SSE event.") from exc
            if not isinstance(payload, dict):
                raise StreamProtocolError("SSE payload must be an object.")
            if payload.get("error"):
                raise StreamProtocolError(f"Server stream error: {payload['error']}")
            if payload.get("model") not in (None, self.model):
                raise StreamProtocolError("Stream model does not match the requested served model.")
            if payload.get("usage") is not None:
                if not isinstance(payload["usage"], dict):
                    raise StreamProtocolError("Usage payload must be an object.")
                usage = payload["usage"]
            choices = payload.get("choices")
            if not isinstance(choices, list):
                raise StreamProtocolError("SSE event is missing a choices array.")
            if len(choices) > 1:
                raise StreamProtocolError("Only one generated choice is supported.")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise StreamProtocolError("Unexpected choice payload.")
                # The final choice often has empty content but carries these reasons.
                if choice.get("finish_reason") is not None:
                    result.finish_reason = choice["finish_reason"]
                if choice.get("stop_reason") is not None:
                    result.stop_reason = choice["stop_reason"]
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise StreamProtocolError("Unexpected delta payload.")
                content = choice.get("text", delta.get("content"))
                if content is None or content == "":
                    continue
                if not isinstance(content, str):
                    raise StreamProtocolError("Generated content must be text.")
                if first is None:
                    first = received_at
                    result.first_content_monotonic_seconds = received_at
                    result.first_content_timestamp = utc_now()
                parts.append(content)

        try:
            # httpx timeouts are per operation; asyncio.timeout also bounds the whole request.
            async with asyncio.timeout(self.timeout):
                async with self.http.stream(
                    "POST", self.base_url + self.request_path,
                    json={
                        "model": self.model, "prompt": prompt.text,
                        "temperature": 0.0, "max_tokens": self.output_tokens,
                        "stream": True, "stream_options": {"include_usage": True},
                        "ignore_eos": self.ignore_eos,
                        "add_special_tokens": False,
                    },
                ) as response:
                    result.http_status = response.status_code
                    response.raise_for_status()
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise StreamProtocolError("Expected a text/event-stream response.")
                    async for chunk in response.aiter_bytes():
                        received_at = self.clock()
                        for data in decoder.feed(chunk):
                            consume(data, received_at)
                            if done:
                                break
                        if done:
                            break
                    if not done:
                        for data in decoder.feed(b"", final=True):
                            consume(data, self.clock())
                            if done:
                                break
            if not done:
                raise StreamProtocolError("Connection closed before the [DONE] marker.")
            result.generated_text = "".join(parts)
            if first is None or not result.generated_text.strip():
                raise StreamProtocolError("Stream completed without generated content.")
            completion_tokens = usage.get("completion_tokens") if usage else None
            prompt_tokens = usage.get("prompt_tokens") if usage else None
            if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens > 0:
                result.output_tokens = completion_tokens
                result.output_token_count_source = "api_usage"
            else:
                result.output_tokens = len(self.tokenizer.encode(result.generated_text, add_special_tokens=False))
                result.output_token_count_source = "tokenizer_fallback"
            result.output_length_complete = result.output_tokens == result.requested_output_tokens
            if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0:
                result.prompt_tokens = prompt_tokens
                result.prompt_token_count_source = "api_usage"
            if result.output_tokens <= 0:
                raise StreamProtocolError("Generated content has no countable tokens.")
            timing = latency_metrics(start, first, completion, result.output_tokens)
            for key, value in timing.items():
                setattr(result, key, value)
            result.success = True
            result.slo_compliant = request_compliant(True, result.ttft_ms, result.tpot_ms_per_token, result.e2e_ms)
        except Exception as exc:
            result.error_type = type(exc).__name__
            if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
                result.error_message = f"Request timeout (limit {self.timeout}s): {str(exc) or type(exc).__name__}"
            else:
                result.error_message = str(exc) or type(exc).__name__
            result.generated_text = "".join(parts)
            # Failure latency is diagnostic only and is excluded from aggregate percentiles.
            result.e2e_ms = (self.clock() - start) * 1000.0
            if first is not None:
                result.ttft_ms = (first - start) * 1000.0
        result.end_timestamp = result.end_timestamp or utc_now()
        return result
