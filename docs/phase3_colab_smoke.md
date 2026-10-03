# Phase 3A manual Colab integration smoke test

These are **operator-run instructions**. No tests, servers, requests, downloads
or benchmarks were run by Codex. Follow the steps in order on a Linux Colab
runtime with one NVIDIA A100. Start with AWQ. This produces development /
integration validation data, not final report performance results.

Use a Colab terminal for the Bash blocks. If using separate notebook
%%bash cells, shell variables and PATH do not persist between cells: repeat
the environment setup in each cell. A directory change in one Bash cell also
does not persist into another. Run all steps from the repository root.

The AWQ checkpoint must already exist at
/content/models/qwen3-14b/awq_w4a16 and include its Qwen tokenizer/chat
template files. No benchmark downloads models, tokenizers or datasets.
The BF16 Hub model may be downloaded by the server on its first manual
startup. Copy/mount/cache that model beforehand if the serving environment
must remain offline.

## 1. Repository, real lock and installation

~~~bash
cd /content/qwen-vllm-production
git pull --ff-only
git rev-parse HEAD
git status --short

# uv is an operator's environment tool, not a benchmark runtime dependency.
# Skip this if uv is already installed.
pip install uv

# Resolve on Linux/Colab. Codex has not generated or modified uv.lock.
uv lock
uv sync --extra serving --extra benchmark --extra test

export PATH="$PWD/.venv/bin:$PATH"
export TOKENIZER_PATH=/content/models/qwen3-14b/awq_w4a16
export SERVED_MODEL_NAME=qwen3-14b
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTILIZATION=0.90
export TENSOR_PARALLEL_SIZE=1
export HOST=0.0.0.0
export PORT=8000
export CUDA_VISIBLE_DEVICES=0
export LOG_DIR="$PWD/results/serving/colab/logs"
mkdir -p "$LOG_DIR"
~~~

The source of Python dependencies is pyproject.toml plus the real uv.lock
you just resolved. Preserve the resulting lock and commit hash with the
experiment. Never hand-write the lock. To reproduce a checked-in lock in a
later environment use:

~~~bash
uv sync --locked --extra serving --extra benchmark --extra test
~~~

To include Phase 2 dependencies as well:

~~~bash
uv sync --locked --extra evaluation --extra serving --extra benchmark --extra test
~~~

An editable pip alternative in an already compatible Colab environment is:

~~~bash
pip install -e ".[serving,benchmark,test]"
~~~

Do not mix pip and uv installations within a formal run. After installation,
manually confirm the supplied validated GPU stack before starting integration:

~~~bash
python - <<'PY'
from importlib.metadata import version
for package in ("vllm", "compressed-tensors", "torch", "transformers", "httpx"):
    print(package, version(package))
import torch
print("torch CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
~~~

Expected supplied versions include vllm 0.26.0, compressed-tensors 0.17.0,
transformers 5.17.0 and torch 2.11.0+cu130. Torch's local +cu130 suffix and
CUDA wheel/index selection are not guaranteed by a public-version pin:
verify the resolver's actual result and compatible driver. Stop and resolve
version/driver mismatches before collecting experimental data. The serving
extra obtains torch through vLLM's dependency constraints.

## 2. New CPU/mock unit tests first

~~~bash
python -m pytest -q \
  tests/test_serving_config.py \
  tests/test_serving_health.py \
  tests/test_benchmark_metrics.py \
  tests/test_benchmark_workload.py \
  tests/test_benchmark_streaming.py \
  tests/test_benchmark_concurrency.py \
  tests/test_benchmark_sweep.py \
  tests/test_slo_performance.py \
  tests/test_observability_prometheus.py \
  tests/test_observability_grafana.py \
  tests/test_observability_config.py
~~~

These tests use fake tokenizers, fake streaming bytes, fake clocks and
httpx.MockTransport. They do not start services, load models or access the
network. Only proceed to the manual integration steps after they pass.
Optional existing regression suite, also operator-run:

~~~bash
python -m pytest -q tests
~~~

## 3. Start AWQ using the shared launcher

~~~bash
export MODEL_PATH=/content/models/qwen3-14b/awq_w4a16
setsid python scripts/serve_vllm.py \
  > "$LOG_DIR/vllm-awq.log" 2>&1 &
echo $! > "$LOG_DIR/vllm-awq.pid"

# Inspect actual engine, quantization, dtype and context settings.
tail -n 60 "$LOG_DIR/vllm-awq.log"
~~~

The launcher fixes prefix caching OFF so later sweep points cannot reuse
earlier points' complete prompt prefills. Verify that the vLLM 0.26.0 CLI
accepts --no-enable-prefix-caching and that the startup log reflects it.
setsid creates a process group for explicit later shutdown. No HTTP server
other than the official vLLM server is involved. Wait for startup:

~~~bash
ready=0
for attempt in $(seq 1 120); do
  if python scripts/check_serving.py \
    --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME"; then
    ready=1
    break
  fi
  sleep 5
done
test "$ready" -eq 1 || { tail -n 100 "$LOG_DIR/vllm-awq.log"; exit 1; }
~~~

The checker verifies /health, /v1/models with the expected alias, and a
Prometheus-compatible /metrics response. A failed check returns non-zero;
do not treat a running process alone as readiness.

## 4. First real streaming generation

This one-request run verifies the selected official /v1/completions endpoint,
locally rendered Qwen chat template, non-empty generated content, expected
model alias, streaming [DONE], and usage/fallback token counting.

~~~bash
python scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME" \
  --variant awq_w4a16 --checkpoint-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --concurrency 1 --num-requests 1 --warmup-requests 0 \
  --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --timeout 120 \
  --output-dir results/serving/colab --run-id awq-single

python - <<'PY'
import json
from pathlib import Path
record = json.loads(Path("results/serving/colab/awq-single/c1/raw.json").read_text())["requests"][0]
assert record["success"], record
assert record["model"] == "qwen3-14b"
assert record["generated_text"].strip()
assert record["ttft_ms"] is not None and record["output_tokens"] > 0
print("First streaming request:", record["output_token_count_source"], record["ttft_ms"], record["e2e_ms"])
PY

# Capture after generation so lazily exported histogram families can be present.
python scripts/check_serving.py \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME" \
  --metrics-output results/serving/colab/vllm-metrics.txt

python scripts/prepare_dashboard.py \
  --metrics-file results/serving/colab/vllm-metrics.txt \
  --output observability/grafana/dashboards/vllm-serving-overview.json \
  --manifest-output results/serving/colab/dashboard-metric-manifest.json
~~~

Read the manifest. It records which real metric TYPE descriptors were found
and which optional panels were omitted. The shipped JSON starts as a bootstrap
dashboard using Prometheus up; this command generates the full runtime-verified
dashboard. The Grafana checker rejects the bootstrap without serving panels.
If names in vLLM 0.26.0 differ from the candidate catalogue, inspect the actual
snapshot, adjust the catalogue to real names and repeat preparation. Do not
replace missing series with invented names.

## 5. Install and start Prometheus without Docker

These example external binary versions are separate from the validated Python
GPU stack. Their availability/download URLs and Grafana APIs still require
manual confirmation; no binary or URL was fetched during implementation.
Select the matching Linux amd64 artifacts from the official release pages if
an archive name changes:
https://github.com/prometheus/prometheus/releases and
https://grafana.com/grafana/download .

~~~bash
export TOOLS=/content/phase3-tools
export PROMETHEUS_VERSION=3.5.0
export GRAFANA_VERSION=12.1.0
mkdir -p "$TOOLS/prometheus" "$TOOLS/grafana"

curl -fL \
  "https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" \
  -o "$TOOLS/prometheus.tar.gz"
tar -xzf "$TOOLS/prometheus.tar.gz" --strip-components=1 -C "$TOOLS/prometheus"

# Verify the checked-in scrape configuration before starting.
"$TOOLS/prometheus/promtool" check config observability/prometheus/prometheus.yml

setsid "$TOOLS/prometheus/prometheus" \
  --config.file="$PWD/observability/prometheus/prometheus.yml" \
  --storage.tsdb.path="$TOOLS/prometheus-data" \
  --web.listen-address=127.0.0.1:9090 \
  > "$LOG_DIR/prometheus.log" 2>&1 &
echo $! > "$LOG_DIR/prometheus.pid"
~~~

The default target is 127.0.0.1:8000/metrics with job=vllm and interval 5s.
For another host, edit only the static target address before starting.
For another interval, keep scrape_timeout below it and update both the
Grafana datasource timeInterval and benchmark metadata.

## 6. Validate scrape target and real data

~~~bash
ready=0
for attempt in $(seq 1 30); do
  if python scripts/check_prometheus.py \
    --base-url http://127.0.0.1:9090 \
    --query 'vllm:num_requests_running{job="vllm"}'; then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" -eq 1 || { tail -n 100 "$LOG_DIR/prometheus.log"; exit 1; }

curl -fsS http://127.0.0.1:9090/-/healthy
curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:9090/api/v1/targets
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=vllm:num_requests_running{job="vllm"}'

curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=sum(vllm:generation_tokens_total{job="vllm"})' \
  -o results/serving/colab/token-counter-before.json
~~~

Require targets to show job=vllm health=up with no lastError, query status
success, and a non-empty finite vector. A zero running-request value is valid;
an empty vector is not. If a requested metric is absent, use a confirmed real
metric from the runtime snapshot and update the commands accordingly.

## 7. Install, provision and validate Grafana without Docker

~~~bash
curl -fL \
  "https://dl.grafana.com/oss/release/grafana-$GRAFANA_VERSION.linux-amd64.tar.gz" \
  -o "$TOOLS/grafana.tar.gz"
tar -xzf "$TOOLS/grafana.tar.gz" --strip-components=1 -C "$TOOLS/grafana"

export PROMETHEUS_URL=http://127.0.0.1:9090
export VLLM_GRAFANA_DASHBOARDS_PATH="$PWD/observability/grafana/dashboards"
export GRAFANA_URL=http://127.0.0.1:3000
export GRAFANA_USER=admin
# Development-only example: replace before exposing or using a persistent host.
export GRAFANA_PASSWORD=colab-development-only-change-me
export GF_SECURITY_ADMIN_USER="$GRAFANA_USER"
export GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_PASSWORD"
export GF_SERVER_HTTP_ADDR=127.0.0.1
export GF_SERVER_HTTP_PORT=3000
export GF_PATHS_PROVISIONING="$PWD/observability/grafana/provisioning"
export GF_PATHS_DATA="$TOOLS/grafana-data"
export GF_PATHS_LOGS="$LOG_DIR"
export GF_PATHS_PLUGINS="$TOOLS/grafana-plugins"
mkdir -p "$GF_PATHS_DATA" "$GF_PATHS_PLUGINS"

setsid "$TOOLS/grafana/bin/grafana" server \
  --homepath "$TOOLS/grafana" \
  > "$LOG_DIR/grafana.log" 2>&1 &
echo $! > "$LOG_DIR/grafana.pid"

ready=0
for attempt in $(seq 1 30); do
  if python scripts/check_grafana.py \
    --base-url "$GRAFANA_URL" \
    --query 'vllm:num_requests_running{job="vllm"}'; then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" -eq 1 || { tail -n 100 "$LOG_DIR/grafana.log"; exit 1; }
~~~

Verify /api/health, datasource UID vllm-prometheus with type=prometheus,
dashboard UID vllm-serving-overview with generated serving panels, and a
non-empty successful query through the datasource proxy. A provisioned
datasource URL alone is insufficient. Code reads credentials from environment
and does not hardcode them.

To view the dashboard, use the Colab port proxy if available. In a notebook
Python cell (not executed by Codex):

~~~python
from google.colab import output
output.serve_kernel_port_as_window(3000)
~~~

Open dashboard vllm-serving-overview in folder Phase 3A and log in with the
development credentials above. Colab proxy/reverse-proxy URL handling is
environment-dependent; if the proxy does not render Grafana, use the API
checks first and configure Grafana root_url for the actual proxy URL.
No public tunnel or UI automation is created by this project.

## 8. Measured smoke points C=1 and C=4

Monitoring remains on throughout both points. Existing run directories are
never overwritten: choose new run IDs for a rerun. Use identical seeds,
tokenizer, prompts, output policy, warmup and request counts for both variants.

~~~bash
python scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME" \
  --variant awq_w4a16 --checkpoint-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --concurrency 1 --num-requests 20 --warmup-requests 3 \
  --timeout 120 --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --monitoring-enabled --prometheus-url "$PROMETHEUS_URL" --grafana-url "$GRAFANA_URL" \
  --prometheus-scrape-interval-seconds 5 \
  --output-dir results/serving/colab --run-id awq-c1

python scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME" \
  --variant awq_w4a16 --checkpoint-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --workload-file results/serving/colab/awq-c1/workload.json \
  --concurrency 4 --num-requests 20 --warmup-requests 3 \
  --timeout 120 --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --monitoring-enabled --prometheus-url "$PROMETHEUS_URL" --grafana-url "$GRAFANA_URL" \
  --prometheus-scrape-interval-seconds 5 \
  --output-dir results/serving/colab --run-id awq-c4

python scripts/summarize_benchmarks.py \
  results/serving/colab/awq-c1/c1/aggregate.json \
  results/serving/colab/awq-c4/c4/aggregate.json \
  --output results/serving/colab/awq-smoke-summary.json
~~~

Optional C=8 repeats the second command with concurrency 8 and a new run ID,
keeping the same 20 requests and saved workload. Do not run the formal
1,2,4,8,16,32,64 sweep during Colab smoke.

## 9. Validate saved results and formulas

~~~bash
python - <<'PY'
import json, math
from pathlib import Path

def reject_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)

def load(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)

def finite_tree(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)

root = Path("results/serving/colab")
digests = []
for name, concurrency in (("awq-c1", 1), ("awq-c4", 4)):
    directory = root / name / f"c{concurrency}"
    raw = load(directory / "raw.json")["requests"]
    aggregate = load(directory / "aggregate.json")
    finite_tree(raw)
    finite_tree(aggregate)
    assert len(raw) == aggregate["num_requests"] == 20
    assert aggregate["monitoring_enabled"] is True
    assert aggregate["prometheus_scrape_interval_seconds"] == 5
    assert aggregate["warmup_requests"] == 3
    assert aggregate["max_model_len"] == 8192
    digests.append(aggregate["workload_sha256"])
    successful = [record for record in raw if record["success"]]
    compliant = [record for record in successful if record["slo_compliant"]]
    assert aggregate["num_successful"] == len(successful)
    assert math.isclose(aggregate["request_success_rate"], len(successful) / 20)
    duration = aggregate["benchmark_duration_seconds"]
    assert duration > 0
    assert math.isclose(aggregate["throughput_requests_per_second"], len(successful) / duration)
    assert math.isclose(aggregate["goodput_requests_per_second"], len(compliant) / duration)
    for record in raw:
        for key in ("request_id", "http_status", "prompt_tokens", "output_tokens",
                    "output_token_count_source", "ttft_ms", "tpot_ms_per_token",
                    "e2e_ms", "slo_compliant", "error_type", "error_message", "start_timestamp"):
            assert key in record
        if record["success"]:
            assert record["output_token_count_source"] in ("api_usage", "tokenizer_fallback")
            assert record["ttft_ms"] >= 0 and record["e2e_ms"] >= record["ttft_ms"]
            if record["output_tokens"] > 1:
                expected = (record["e2e_ms"] - record["ttft_ms"]) / (record["output_tokens"] - 1)
                assert math.isclose(record["tpot_ms_per_token"], expected)
            else:
                assert record["tpot_ms_per_token"] is None and not record["slo_compliant"]
        else:
            assert record["error_type"] and record["error_message"]
    if successful:
        assert math.isclose(aggregate["slo_attainment_rate"], len(compliant) / len(successful))
    else:
        assert aggregate["slo_attainment_rate"] is None
    print(name, "successes:", len(successful), "goodput req/s:", aggregate["goodput_requests_per_second"])
assert digests[0] == digests[1]
print("Smoke result structure and formulas checked; this is not formal SLO certification.")
PY
~~~

Inspect failures rather than discarding them. Compare API prompt token
counts against the locally counted lengths in workload.json. Unexpected
differences mean tokenizer or endpoint special-token handling needs review.

## 10. Observe load in Prometheus and Grafana

Run these queries while C=4 (or optional C=8) is executing from another
terminal/cell, and again after it completes. Keep only names listed as
included in dashboard-metric-manifest.json:

~~~bash
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=vllm:num_requests_running{job="vllm"}'
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=vllm:num_requests_waiting{job="vllm"}'
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=vllm:kv_cache_usage_perc{job="vllm"}'
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=sum(rate(vllm:generation_tokens_total{job="vllm"}[1m]))'
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95,sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{job="vllm"}[1m])))'

# Allow at least one new scrape after the measured run.
sleep 6
curl -fsSG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=sum(vllm:generation_tokens_total{job="vllm"})' \
  -o results/serving/colab/token-counter-after.json

python - <<'PY'
import json
from pathlib import Path
root = Path("results/serving/colab")
before = json.loads((root / "token-counter-before.json").read_text())
after = json.loads((root / "token-counter-after.json").read_text())
assert before["status"] == after["status"] == "success"
assert before["data"]["result"] and after["data"]["result"]
first = float(before["data"]["result"][0]["value"][1])
last = float(after["data"]["result"][0]["value"][1])
assert last > first, (first, last)
print("Generation counter increased:", first, "->", last)
PY

python scripts/check_prometheus.py --base-url "$PROMETHEUS_URL"
python scripts/check_grafana.py --base-url "$GRAFANA_URL"
~~~

Zero waiting requests can be legitimate at low concurrency; do not force an
increase in every gauge. Short bursts may fall between scrapes. Token counters
should increase, and histogram/rate panels need enough scrapes and requests
before displaying meaningful data. Inspect Grafana running/waiting requests,
cache utilization, token rates and supported latency/queue/prefill/decode
panels during load. Monitoring data stays in Prometheus storage rather than
being copied into aggregate JSON.

## 11. Stop AWQ and do minimal BF16 smoke

Only use the PID/group recorded by this session's launcher. Confirm it belongs
to this AWQ server before sending the signal; do not reuse an old PID file.

~~~bash
ps -p "$(cat "$LOG_DIR/vllm-awq.pid")" -o pid,pgid,args
kill -TERM -- "-$(cat "$LOG_DIR/vllm-awq.pid")"
for attempt in $(seq 1 60); do
  if ! kill -0 "$(cat "$LOG_DIR/vllm-awq.pid")" 2>/dev/null; then break; fi
  sleep 2
done
# Do not start BF16 until the AWQ process has exited and released its GPU.
if kill -0 "$(cat "$LOG_DIR/vllm-awq.pid")" 2>/dev/null; then
  echo "AWQ is still running; inspect its log before proceeding."
  exit 1
fi

export MODEL_PATH=Qwen/Qwen3-14B
setsid python scripts/serve_vllm.py \
  > "$LOG_DIR/vllm-bf16.log" 2>&1 &
echo $! > "$LOG_DIR/vllm-bf16.pid"

ready=0
for attempt in $(seq 1 120); do
  if python scripts/check_serving.py \
    --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME"; then
    ready=1
    break
  fi
  sleep 5
done
test "$ready" -eq 1 || { tail -n 100 "$LOG_DIR/vllm-bf16.log"; exit 1; }

python scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME" \
  --variant bf16 --checkpoint-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
  --concurrency 1 --num-requests 1 --warmup-requests 0 \
  --output-dir results/serving/colab --run-id bf16-single

python scripts/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --model "$SERVED_MODEL_NAME" \
  --variant bf16 --checkpoint-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
  --workload-file results/serving/colab/awq-c1/workload.json \
  --concurrency 1 --num-requests 20 --warmup-requests 3 \
  --timeout 120 --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --monitoring-enabled --prometheus-url "$PROMETHEUS_URL" --grafana-url "$GRAFANA_URL" \
  --prometheus-scrape-interval-seconds 5 \
  --output-dir results/serving/colab --run-id bf16-c1

python scripts/check_prometheus.py --base-url "$PROMETHEUS_URL"
python scripts/check_grafana.py --base-url "$GRAFANA_URL"
~~~

Repeat the raw/aggregate assertions from Step 9 for bf16-single and bf16-c1;
require generated content for the single request and compare the BF16 C1
workload hash with AWQ C1. Do not interpret differences between these smoke
runs as final report results, and do not combine different variants into one
sweep summary.

If stopping the development stack after validation, use only this session's
recorded process groups:

~~~bash
kill -TERM -- "-$(cat "$LOG_DIR/vllm-bf16.pid")"
kill -TERM -- "-$(cat "$LOG_DIR/grafana.pid")"
kill -TERM -- "-$(cat "$LOG_DIR/prometheus.pid")"
~~~

## Future formal RunPod experiment

Only after integration succeeds, use the same A100 SXM 80GB, session,
serving implementation, future image, CUDA environment, versions, engine
settings, tokenizer, saved workload, output policy, request counts, warmup,
concurrency levels and monitoring configuration for BF16 and AWQ.
Prometheus/Grafana remain enabled. Docker/deployment are intentionally absent.

The following command is a future formal example, **not a Colab smoke step**:

~~~bash
python scripts/benchmark_sweep.py \
  --base-url http://127.0.0.1:8000 --model qwen3-14b \
  --variant awq_w4a16 --checkpoint-path /workspace/models/qwen3-14b/awq_w4a16 \
  --tokenizer-path /workspace/models/qwen3-14b/awq_w4a16 \
  --concurrency-levels 1,2,4,8,16,32,64 \
  --num-requests 1000 --warmup-requests 5 \
  --input-tokens 1000 --output-tokens 256 --seed 42 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --monitoring-enabled \
  --output-dir results/serving/formal --run-id awq-formal
~~~

The request count is an example to be finalized in the formal experimental
protocol, not a statistically validated minimum. Repeat BF16 with the same
saved workload and all controls. Interpret maximum SLO-compliant concurrency
separately from maximum SLO-compliant goodput.
