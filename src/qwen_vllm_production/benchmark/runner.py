"""Manual benchmark entrypoints; no workload, network or models at import time."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..evaluation.common import package_versions, save_json
from .client import StreamingClient
from .metrics import aggregate
from .sweep import closed_loop, summarize_points
from .workload import build_workload, load_local_tokenizer, validate_workload


def concurrency_levels(value: str) -> list[int]:
    try:
        levels = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer concurrency levels.") from exc
    if not levels or any(level <= 0 for level in levels) or len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("Concurrency levels must be positive and unique.")
    return levels


def build_parser(*, sweep: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure streaming client latency with closed-loop workers.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="Exact served model alias from /v1/models.")
    parser.add_argument("--variant", required=True, choices=("bf16", "awq_w4a16"))
    parser.add_argument("--tokenizer-path", required=True, help="Local checkpoint/snapshot or already cached HF ID.")
    parser.add_argument("--request-path", choices=("/v1/completions",), default="/v1/completions")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--concurrency-levels", type=concurrency_levels, required=sweep)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192, help="Record the value used by the serving launcher.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Record the server's configured value.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Record the server's configured value; Phase 3A uses 1.")
    parser.add_argument("--checkpoint-path", help="Record the server's Hub ID or checkpoint path; not loaded by the client.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-early-eos", action="store_true", help="Change the fixed-output workload; use identically for both variants.")
    parser.add_argument("--workload-file", type=Path, help="Reuse and validate an already saved prompt sequence.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/serving"))
    parser.add_argument("--run-id", help="Unique output subdirectory; existing runs are never overwritten.")
    parser.add_argument("--monitoring-enabled", action="store_true")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--grafana-url", default="http://127.0.0.1:3000")
    parser.add_argument("--prometheus-scrape-interval-seconds", type=float, default=5.0)
    return parser


async def run_point(client: StreamingClient, prompts: list, *, concurrency: int, warmup_requests: int, metadata: dict, clock=time.perf_counter) -> tuple[list, dict]:
    warmup = await closed_loop(prompts[:warmup_requests], concurrency, client.send, model=client.model)
    # The measured interval spans worker scheduling through the final measured request.
    # Workload/tokenizer setup, warmup, file writes and monitoring checks are outside it.
    start = clock()
    requests = await closed_loop(prompts[warmup_requests:], concurrency, client.send, model=client.model)
    duration = clock() - start
    point_metadata = {
        **metadata,
        "warmup_requests": warmup_requests,
        "warmup_successful": sum(result.success for result in warmup),
        "warmup_failed": sum(not result.success for result in warmup),
        "duration_boundary": "before measured workers start until all measured workers finish",
    }
    return requests, aggregate(
        requests, model=client.model, concurrency=concurrency,
        duration_seconds=duration, metadata=point_metadata,
    )


async def run(args: argparse.Namespace, *, sweep: bool = False) -> Path:
    levels = args.concurrency_levels if sweep else [args.concurrency]
    if not sweep and args.concurrency_levels is not None:
        raise ValueError("Use benchmark_sweep.py for multiple concurrency levels.")
    if args.num_requests <= 0 or args.warmup_requests < 0 or any(level <= 0 for level in levels):
        raise ValueError("Measured count and concurrency must be positive; warmup must be non-negative.")
    if args.num_requests < max(levels):
        raise ValueError("num-requests must be at least the largest tested concurrency.")
    if not all(math.isfinite(value) and value > 0 for value in (args.timeout, args.prometheus_scrape_interval_seconds)):
        raise ValueError("Timeout and scrape interval must be positive.")
    if args.max_model_len <= 0 or args.input_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("Token targets and max-model-len must be positive.")
    if args.tensor_parallel_size != 1 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("Phase 3A requires tensor-parallel-size=1 and GPU utilization in (0, 1].")
    parsed = urlparse(args.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
        raise ValueError("base-url must be an HTTP(S) server origin, without /v1.")
    tokenizer = load_local_tokenizer(args.tokenizer_path)
    count = args.num_requests + args.warmup_requests
    workload = (
        json.loads(args.workload_file.read_text(encoding="utf-8")) if args.workload_file
        else build_workload(
            tokenizer, count=count, seed=args.seed, target_input_tokens=args.input_tokens,
            target_output_tokens=args.output_tokens, tokenizer_path=args.tokenizer_path,
        )
    )
    prompts = validate_workload(
        workload, tokenizer, count=count, seed=args.seed,
        target_input_tokens=args.input_tokens, target_output_tokens=args.output_tokens,
    )
    if any(prompt.prompt_tokens + args.output_tokens > args.max_model_len for prompt in prompts):
        raise ValueError("Prompt plus target output exceeds the recorded server context length.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if run_id in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run-id must be a safe single directory name.")
    destination = args.output_dir / run_id
    destination.mkdir(parents=True, exist_ok=False)
    save_json(workload, destination / "workload.json")
    metadata = {
        "variant": args.variant, "backend": "vllm", "seed": args.seed,
        "target_input_tokens": args.input_tokens, "target_output_tokens": args.output_tokens,
        "actual_prompt_tokens": [prompt.prompt_tokens for prompt in prompts[args.warmup_requests:]],
        "max_model_len": args.max_model_len, "tokenizer_path": args.tokenizer_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype_configuration": "auto", "checkpoint_path": args.checkpoint_path,
        "prefix_caching": False,
        "timeout_seconds": args.timeout,
        "serving_config_source": "operator-declared; launch with the same values",
        "workload_sha256": workload["workload_sha256"],
        "base_url": args.base_url, "request_path": args.request_path,
        "temperature": 0.0, "ignore_eos": not args.allow_early_eos,
        "chat_template_applied_locally": True, "enable_thinking": False,
        "monitoring_enabled": args.monitoring_enabled,
        "prometheus_url": args.prometheus_url if args.monitoring_enabled else None,
        "grafana_url": args.grafana_url if args.monitoring_enabled else None,
        "prometheus_scrape_interval_seconds": args.prometheus_scrape_interval_seconds if args.monitoring_enabled else None,
        "package_versions": package_versions("vllm", "torch", "transformers", "compressed-tensors", "httpx"),
    }
    points = []
    async with httpx.AsyncClient(
        timeout=args.timeout, trust_env=False,
        limits=httpx.Limits(max_connections=max(levels), max_keepalive_connections=max(levels)),
    ) as http:
        client = StreamingClient(
            http, tokenizer, base_url=args.base_url, model=args.model,
            output_tokens=args.output_tokens, timeout=args.timeout,
            request_path=args.request_path, ignore_eos=not args.allow_early_eos,
        )
        for concurrency in levels:
            requests, point = await run_point(
                client, prompts, concurrency=concurrency,
                warmup_requests=args.warmup_requests, metadata=metadata,
            )
            point_dir = destination / f"c{concurrency}"
            save_json({"requests": [asdict(result) for result in requests]}, point_dir / "raw.json")
            save_json(point, point_dir / "aggregate.json")
            points.append(point)
    save_json(summarize_points(points), destination / "sweep_summary.json")
    return destination


def main(*, sweep: bool = False) -> None:
    parser = build_parser(sweep=sweep)
    args = parser.parse_args()
    try:
        path = asyncio.run(run(args, sweep=sweep))
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(1, f"Benchmark configuration/output error: {exc}\n")
    print(f"Saved measured results: {path}")


def summarize_main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing aggregate JSONs from one experiment.")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    points = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    save_json(summarize_points(points), args.output)
