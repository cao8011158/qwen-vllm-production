#!/usr/bin/env bash
# CPU-only Pod startup. Image must already contain Bash, Python and Prometheus.
set -euo pipefail
umask 077

: "${DEPLOYMENT_REVISION:?Supply the actual repository commit used in the image}"
: "${VLLM_METRICS_URL:?Supply the GPU Pod's https proxy URL ending in /metrics}"
REPO_DIR="${REPO_DIR:-/opt/qwen-vllm-production}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
PROMETHEUS_BIN="${PROMETHEUS_BIN:-/usr/local/bin/prometheus}"
PROMETHEUS_CONFIG_DIR="${PROMETHEUS_CONFIG_DIR:-/workspace/prometheus/config}"
PROMETHEUS_DATA_DIR="${PROMETHEUS_DATA_DIR:-/workspace/prometheus/data}"

[[ "$DEPLOYMENT_REVISION" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid repository revision." >&2; exit 1; }
[[ -f "$REPO_DIR/.deployment-revision" ]] || { echo "Image repository revision marker is missing." >&2; exit 1; }
[[ "$(cat "$REPO_DIR/.deployment-revision")" == "$DEPLOYMENT_REVISION" ]] || { echo "Image repository revision mismatch." >&2; exit 1; }
[[ -x "$PYTHON_BIN" && -x "$PROMETHEUS_BIN" ]] || { echo "Image must contain Python and Prometheus." >&2; exit 1; }
version_text="$("$PROMETHEUS_BIN" --version 2>&1)"
[[ "$version_text" =~ version[[:space:]]3\.5\.0([^0-9]|$) ]] || { echo "Expected Prometheus 3.5.0." >&2; exit 1; }
mkdir -p "$PROMETHEUS_CONFIG_DIR" "$PROMETHEUS_DATA_DIR"

# Write a separate RunPod configuration; preserve both checked-in configs.
"$PYTHON_BIN" - "$VLLM_METRICS_URL" "$PROMETHEUS_CONFIG_DIR/prometheus.yml" <<'PY'
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required.")
url = sys.argv[1]
if not re.fullmatch(r"https://[a-z0-9]+-8000\.proxy\.runpod\.net/metrics", url):
    raise SystemExit("VLLM_METRICS_URL must be the GPU Pod HTTP proxy /metrics URL, not localhost.")
target = urlsplit(url).netloc
config = (
    "global:\n"
    "  scrape_interval: 5s\n"
    "  evaluation_interval: 5s\n"
    "scrape_configs:\n"
    "  - job_name: vllm\n"
    "    scheme: https\n"
    "    metrics_path: /metrics\n"
    "    scrape_timeout: 4s\n"
    "    follow_redirects: false\n"
    "    static_configs:\n"
    f"      - targets: [{json.dumps(target)}]\n"
    "        labels:\n"
    "          experiment: runpod\n"
)
path = Path(sys.argv[2])
temporary = path.with_suffix(".tmp")
temporary.write_text(config, encoding="utf-8")
temporary.replace(path)
PY

exec "$PROMETHEUS_BIN" \
  --config.file="$PROMETHEUS_CONFIG_DIR/prometheus.yml" \
  --storage.tsdb.path="$PROMETHEUS_DATA_DIR" \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=0.0.0.0:9090
