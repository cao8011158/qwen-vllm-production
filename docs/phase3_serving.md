# Phase 3A: serving, performance and observability

This implementation compares BF16 with W4A16 AWQ. W8A8 is excluded from
serving experiments. Phase 2 established the quality decision: AWQ recovery
approximately 97.16%, W8A8 approximately 74.87%, quality threshold 97%.
Phase 3 neither reruns quality tasks nor computes a combined quality score.

## Serving

One launcher builds an argument vector for the official vLLM CLI:

~~~text
vllm serve MODEL_PATH --served-model-name qwen3-14b
  --max-model-len 8192 --gpu-memory-utilization 0.90
  --tensor-parallel-size 1 --host 0.0.0.0 --port 8000 --dtype bfloat16 --seed 42
  --no-enable-prefix-caching
~~~

CLI flags override MODEL_PATH, SERVED_MODEL_NAME, MAX_MODEL_LEN,
GPU_MEMORY_UTILIZATION, TENSOR_PARALLEL_SIZE, HOST, PORT, DTYPE, SEED and
MODEL_REVISION environment variables. DTYPE defaults to bfloat16 and SEED to
42. --model-revision adds --revision to vLLM only when non-empty; local
checkpoints can leave it empty. MODEL_PATH is required. No quantization override is added:
vLLM reads the checkpoint's quantization configuration. Only one GPU is
accepted in this phase. The launcher replaces itself with the official CLI;
it adds no inference layer, exporter or HTTP server.

Use the same alias and configuration for both checkpoints. The benchmark
records the declared serving settings; it does not remotely change or prove
those settings. Confirm the actual model path, dtype, quantization and engine
configuration in the server startup log.
Record matching --dtype, --serving-seed and --model-revision in the benchmark
CLI. Its --seed controls workload generation, separately from the vLLM seed.

## Workload and endpoint

The primary workload is synthetic concurrent interactive chat: seed 42,
approximately 1000 input tokens, target 256 output tokens, context 8192.
There is no remote dataset. A deterministic PRNG creates planning notes,
with distinct early identifiers to reduce long shared prompt prefixes.

The local Qwen tokenizer renders the complete system/user/assistant chat
template with enable_thinking=False. A binary search sizes the body to the
target input length with a maximum deviation of 16 tokens. The actual length
is counted after rendering, using encode(add_special_tokens=False), rather
than estimated from character counts.

The client sends that fully rendered text to the official **/v1/completions**
endpoint. This is an explicit choice among the official OpenAI-compatible
generation endpoints: chat messages are rendered locally, and the server
must not render them a second time. The stream parser also understands
delta.content chunks, but Phase 3A CLI deliberately selects completions.
Every request explicitly sets add_special_tokens=False so the server does
not add special tokens to the already rendered prompt.

The shared launcher fixes prefix caching OFF. A sweep deliberately reuses
the same saved prompt sequence, so leaving prefix caching enabled could make
later concurrency points hit a fully cached prefill from earlier points.
This fixed policy is recorded in aggregate JSON and is identical for BF16
and AWQ; it is not a cache-tuning experiment.

Tokenizer loading uses local_files_only=True and trust_remote_code=False.
Missing local/cached tokenizer files are an error; the benchmark cannot
download them. Use the same local tokenizer snapshot for both variants.
workload.json saves the complete ordered prompt sequence, exact lengths,
seed, targets, template settings, per-prompt SHA256 and workload digest.
--workload-file validates hashes, lengths, ordering and protocol settings.

Temperature is 0. Default ignore_eos=True makes this a controlled fixed-output
performance workload. --allow-early-eos changes the experiment and must be
identical for both checkpoints; it should not be mixed into a formal sweep.
This synthetic fixed-output workload is not evidence of quality or the output
length distribution of real production conversations.

## Client measurements

The async client uses **httpx** for streaming, connection pooling, HTTP
validation and mock transports. There is one HTTP stack and no OpenAI or
Grafana SDK. httpx operation timeouts are supplemented by asyncio.timeout
for a total per-request deadline.

All latency calculations use a monotonic clock. Wall-clock UTC timestamps
and monotonic receive times are saved in each measured raw record.

| Quantity | Definition |
| --- | --- |
| TTFT, ms | First non-empty generated content receipt minus request start |
| E2E, ms | Receipt of the normal SSE [DONE] completion marker minus request start |
| TPOT, ms/token | (E2E - TTFT) / (output_tokens - 1), only for successful requests with output_tokens > 1 |
| Success rate | Successful measured requests / all measured requests |
| SLO attainment | Successful requests meeting all three latency SLOs / successful requests |
| Throughput, req/s | Successful measured requests / measured elapsed seconds (monotonic) |
| Output tokens/s | Sum of successful output tokens / measured elapsed seconds (monotonic) |
| Goodput, req/s | Successful measured requests meeting all latency SLOs / measured elapsed seconds (monotonic) |

Empty events, role-only events, metadata-only events and empty content do not
start TTFT. Byte boundaries may split UTF-8, SSE frames or JSON. Multiple
content chunks are concatenated. A malformed event, unexpected payload,
server error, wrong model alias, HTTP error, timeout, premature disconnect
without [DONE], or zero-content stream is a failed request and remains in
raw.json. One failed request does not cancel other workers.

Usage completion_tokens is preferred when it is a positive integer;
otherwise the same local tokenizer counts the concatenated generated text.
The source is explicitly api_usage or tokenizer_fallback. Fallback
retokenization is deterministic but may differ from the original generated
token sequence; prefer API usage for formal results. Prompt counts also
record whether they came from API usage or the local tokenizer.
Raw records also retain finish_reason and stop_reason when supplied, including
on final choices with empty content, and requested_output_tokens.
output_length_complete compares the counted output_tokens with the requested
target; it is null when no count is available. Inspect incomplete outputs
against the 256-token target. Short outputs remain recorded and do not change
the existing request-success or latency-SLO definitions.

TPOT is null for output_tokens <= 1 and excluded from TPOT percentiles.
Such requests are conservatively non-compliant. If no requests succeed,
attainment and latency percentiles are null, success rate and rates are zero,
and the operating point fails. JSON writers reject NaN and infinity.

## SLOs, scheduling and output

Per-request compliance requires success AND TTFT <= 1000 ms AND
TPOT <= 50 ms/token AND E2E <= 15000 ms. All metrics must be finite,
non-negative and calculable.

Request-level SLO attainment is the number of successful requests meeting
all three latency thresholds divided by the number of successful requests.
Request success rate is successful measured requests divided by all measured
requests. An operating point passes if and only if request success rate >= 0.99
AND request-level SLO attainment >= 0.95; both rates must be present and finite.
The result field remains slo_compliant_operating_point.

Percentiles use successful requests only, linear interpolation at rank
(n - 1) * q (type 7). p50 and p95 TTFT/TPOT/E2E values are still recorded and
reported for descriptive and tail-latency analysis, model comparison and
concurrency degradation analysis, but do not independently determine
operating-point compliance. p95 latency is not an independent pass/fail gate.
Throughput, output tokens/s and goodput remain reported. Goodput is the number
of request-level SLO-compliant requests divided by measured duration, regardless
of the operating-point boolean.

configs/config.yaml names the request-level latency thresholds ttft_seconds,
tpot_ms_per_token and e2e_seconds; these are not percentile thresholds.

Concurrency C means C async workers. Each worker awaits its current request,
then claims the next prompt. The number of outstanding requests never exceeds
C. No request-rate arrival process is implied, and the run does not end after
only one batch of C requests. Prompt allocation follows the saved sequence;
completion order and worker assignment naturally depend on scheduling.

Warmup uses the same workload type, runs first, and is excluded from measured
raw records, latency statistics, rates and measured duration. The measured
interval begins immediately before creating measured workers and ends when
all measured workers finish. Workload preparation and file writes are outside
that interval. Warmup attempt/success/failure counts are separate metadata.
Aggregate measurement_start_utc and measurement_end_utc mark that interval
in wall-clock UTC for Prometheus/Grafana alignment. Duration and all latency
calculations continue to use the monotonic clock.

Each manually named run creates a new directory; an existing run is rejected:

~~~text
results/serving/<output-subdirectory>/<run-id>/
  workload.json
  c1/raw.json
  c1/aggregate.json
  c4/raw.json                 # only when selected for a sweep
  c4/aggregate.json
  sweep_summary.json
~~~

The sweep launcher requires explicit --concurrency-levels. It runs each point
sequentially with the same workload and warmup count. The summarizer rejects
mixed protocols or duplicate concurrency values. Protocol checks include base
URL, package versions, tokenizer, temperature, local
template/thinking/special-token settings, dtype, serving seed and revision.
Missing metadata cannot be mixed with a present field.
Maximum compliant concurrency
and maximum compliant goodput are computed independently. Their locations
can differ; ties in goodput choose lower concurrency. If nothing passes, all
three maximum/location fields are null. A small Colab sample is insufficient
to establish production capacity.

## Monitoring

The chain is vLLM /metrics -> Prometheus -> Grafana, with no custom exporter.
Prometheus scrapes job=vllm at 127.0.0.1:8000 every 5 seconds. Change that
target in observability/prometheus/prometheus.yml for another host.
If changing the interval, also update Grafana datasource timeInterval and
the benchmark's recorded --prometheus-scrape-interval-seconds.

Grafana provisions datasource UID vllm-prometheus and dashboard UID
vllm-serving-overview. Datasource URL and dashboard filesystem path come
from PROMETHEUS_URL and VLLM_GRAFANA_DASHBOARDS_PATH. API authentication uses
GRAFANA_TOKEN or GRAFANA_USER plus GRAFANA_PASSWORD; no secret is embedded
in application code.

### Runtime metric verification

No network or vLLM 0.26.0 source inspection was available during implementation.
It would be incorrect to claim that every optional family below was verified
for that version. The checked-in dashboard is a bootstrap with the real
Prometheus scrape-health metric up{job="vllm"}. **Before provisioning the full
dashboard, capture /metrics after a generation request and run
prepare_dashboard.py.** It includes a serving panel only when the snapshot
contains its real TYPE descriptor with the correct metric type. Missing
families are omitted and listed in the manifest, not invented.

| Panel | Family considered, included only if actually exported |
| --- | --- |
| Running Requests | vllm:num_requests_running |
| Waiting / Queue Pressure | vllm:num_requests_waiting |
| KV Cache Utilization | vllm:kv_cache_usage_perc |
| TTFT | vllm:time_to_first_token_seconds |
| E2E | vllm:e2e_request_latency_seconds |
| Request TPOT | vllm:request_time_per_output_token_seconds |
| Inter-token latency | vllm:inter_token_latency_seconds |
| Prompt token rate | vllm:prompt_tokens_total |
| Generation token rate | vllm:generation_tokens_total |
| Completed request activity | vllm:request_success_total |
| Queue time | vllm:request_queue_time_seconds |
| Prefill time | vllm:request_prefill_time_seconds |
| Decode time | vllm:request_decode_time_seconds |

Latency panels use histogram_quantile(0.50/0.95, sum by (le)(rate(..._bucket)))
and convert seconds to ms. Counter panels use rate; KV-cache fraction is
converted to percent. Raw bucket counts are never graphed as latency.
Server completed-request counts are not the client's success rate.

Prometheus checks health, readiness, a matching active target with health=up
and no lastError, then a non-empty finite instant-query vector. Grafana checks
database health, datasource UID/type, dashboard UID, and a non-empty
Prometheus query through its datasource proxy. All failures produce diagnostics
and a non-zero exit code. --query can select another confirmed real metric.

Client JSON is authoritative for the formal SLOs. Monitoring observes
running/waiting requests, cache pressure, server latencies and token activity;
it does not replace client latency measurement. Aggregate JSON stores only
monitoring configuration, not time-series samples. No ON/OFF overhead
experiment is added.

## Reproducibility and boundaries

RunPod uses the official vllm/vllm-openai:v0.26.0 runtime unchanged.
GPU bootstrap checks vLLM 0.26.0, compressed-tensors 0.17.0, torch 2.11.0 and
CUDA 13.0 exactly, and checks transformers >=5.5.3 with packaging.version.Version
in both runtime validation blocks. The operator-reported transformers 5.14.1
meets that minimum. runtime-packages.json records the actual transformers version;
bootstrap does not install, upgrade or downgrade the serving stack.

The separately provisioned project environments retain the supplied pins in
pyproject.toml and uv.lock for vLLM 0.26.0,
compressed-tensors 0.17.0, transformers 5.17.0, lm-eval 0.4.13 and datasets
4.8.5. Torch's public version remains constrained to 2.11.0. For Linux x86_64
project environments such as Colab, tool.uv.sources maps torch to the explicit pytorch-cu130 index
at https://download.pytorch.org/whl/cu130; other packages retain their normal
sources. The public-version constraint permits the +cu130 local version.
For those project environments, regenerate uv.lock with uv lock, then install
with uv sync --locked. Do not use this installation flow inside the official RunPod image.
No lock was generated or edited here. Manually verify that resolution selected
torch 2.11.0+cu130, CUDA runtime 13.0 and a compatible driver before integration.
Wheel availability and dependency compatibility were not checked over the
network. Preserve the resolved HTTP/test/tool versions in the lock.

configs/config.yaml now agrees with the 97% quality threshold and the
1000/256/8192 primary workload, GPU utilization 0.90 and TP=1. It no longer
contains structured-output settings or the unused long_context workload.
The launcher and runner still take their settings from their existing CLI
(and, for the launcher, environment), rather than loading this YAML.

The repository currently declares model.revision=main, not an immutable SHA.
Before formal RunPod benchmarking, recover the real immutable Hugging Face
revision used as the AWQ source and pin BF16 to it with MODEL_REVISION.
Use the matching tokenizer snapshot and record --model-revision in benchmark
metadata. A missing revision is saved as null, with the BF16 formal-run
requirement also recorded; no commit SHA has been invented.

Formal BF16/AWQ experiments must reuse the physical A100 SXM 80GB, session
where practical, future image, CUDA/vLLM versions, serving arguments, local
tokenizer, saved prompts, output target, request counts, warmup count, timeout,
concurrency levels 1,2,4,8,16,32,64, SLOs and monitoring configuration.
Other vLLM defaults can influence this workload; retain
the same fixed prefix-cache policy and document the startup log rather than interpreting
Colab smoke results as final performance data.

Docker, RunPod deployment, Terraform, Kubernetes, autoscaling, W8A8 serving,
structured output, tuning sweeps and speculative decoding are outside Phase
3A. Follow phase3_colab_smoke.md for the manual integration sequence.
