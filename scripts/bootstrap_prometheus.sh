#!/usr/bin/env bash
# External config generator: run on the GPU Pod or an operator machine.
# Do not run this script inside prom/prometheus:v3.5.0.
set -euo pipefail
umask 077
: "${VLLM_METRICS_URL:?Supply https://<GPU_POD_ID>-8000.proxy.runpod.net/metrics}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${PROMETHEUS_CONFIG_OUTPUT_DIR:-results/serving/runpod/config}"

"$PYTHON_BIN" - "$VLLM_METRICS_URL" "$OUTPUT_DIR" <<'PY'
import json
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import urlsplit

url = sys.argv[1]
if not re.fullmatch(r"https://[a-z0-9]+-8000\.proxy\.runpod\.net/metrics", url):
    raise SystemExit("Use the actual GPU Pod HTTPS proxy /metrics URL.")
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
    f"      - targets: [{json.dumps(urlsplit(url).netloc)}]\n"
    "        labels:\n"
    "          experiment: runpod\n"
)
# RunPod Console's documented JSON start-command format overrides ENTRYPOINT.
# BusyBox /bin/sh and /bin/prometheus are available in the official image.
command = (
    "umask 077; printf '%s' " + shlex.quote(config) + " > /tmp/qwen-prometheus.yml"
    " && exec /bin/prometheus"
    " --config.file=/tmp/qwen-prometheus.yml"
    " --storage.tsdb.path=/prometheus"
    " --storage.tsdb.retention.time=7d"
    " --web.listen-address=0.0.0.0:9090"
)
start = {"entrypoint": ["/bin/sh", "-c"], "cmd": [command]}
directory = Path(sys.argv[2])
directory.mkdir(parents=True, exist_ok=True)
for name, content in (
    ("prometheus.yml", config),
    ("prometheus-start-command.json", json.dumps(start) + "\n"),
):
    path = directory / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
print(f"Scrape config: {directory / 'prometheus.yml'}")
print(f"Paste the entire {directory / 'prometheus-start-command.json'} into RunPod Console's Container Start Command.")
print("No Pod or network API was called.")
PY
