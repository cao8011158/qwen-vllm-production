#!/usr/bin/env bash
# CPU-only Pod startup. Provision the existing runtime-verified dashboard.
set -euo pipefail
umask 077

: "${DEPLOYMENT_REVISION:?Supply the actual repository commit used in the image}"
: "${PROMETHEUS_URL:?Supply the Prometheus CPU Pod's https proxy origin}"
: "${GRAFANA_PUBLIC_URL:?Supply this Grafana CPU Pod's https proxy origin}"
: "${GRAFANA_PASSWORD:?Supply a private administrator password outside Terraform}"
: "${VLLM_DASHBOARD_FILE:?Provide the dashboard generated from this GPU Pod's actual metrics snapshot}"
REPO_DIR="${REPO_DIR:-/opt/qwen-vllm-production}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
GRAFANA_HOME="${GRAFANA_HOME:-/usr/share/grafana}"
GRAFANA_BIN="${GRAFANA_BIN:-$GRAFANA_HOME/bin/grafana}"
GRAFANA_WORK_DIR="${GRAFANA_WORK_DIR:-/workspace/grafana}"

[[ "$DEPLOYMENT_REVISION" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid repository revision." >&2; exit 1; }
[[ -f "$REPO_DIR/.deployment-revision" ]] || { echo "Image repository revision marker is missing." >&2; exit 1; }
[[ "$(cat "$REPO_DIR/.deployment-revision")" == "$DEPLOYMENT_REVISION" ]] || { echo "Image repository revision mismatch." >&2; exit 1; }
[[ -x "$PYTHON_BIN" && -x "$GRAFANA_BIN" ]] || { echo "Image must contain Python and Grafana." >&2; exit 1; }
version_text="$("$GRAFANA_BIN" --version 2>&1)"
[[ "$version_text" =~ (^|[^0-9])12\.1\.0([^0-9]|$) ]] || { echo "Expected Grafana 12.1.0." >&2; exit 1; }

export GRAFANA_USER="${GRAFANA_USER:-admin}"
export GF_SECURITY_ADMIN_USER="$GRAFANA_USER"
export GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_PASSWORD"
export GF_USERS_ALLOW_SIGN_UP=false
export GF_SERVER_HTTP_ADDR=0.0.0.0
export GF_SERVER_HTTP_PORT=3000
export GF_SERVER_ROOT_URL="$GRAFANA_PUBLIC_URL/"
export GF_PATHS_PROVISIONING="$GRAFANA_WORK_DIR/provisioning"
export GF_PATHS_DATA="$GRAFANA_WORK_DIR/data"
export GF_PATHS_LOGS="$GRAFANA_WORK_DIR/logs"
export GF_PATHS_PLUGINS="$GRAFANA_WORK_DIR/plugins"
export VLLM_GRAFANA_DASHBOARDS_PATH="$GRAFANA_WORK_DIR/dashboards"

"$PYTHON_BIN" - "$PROMETHEUS_URL" "$GRAFANA_PUBLIC_URL" <<'PY'
import re
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required.")
for url, port in ((sys.argv[1], 9090), (sys.argv[2], 3000)):
    if not re.fullmatch(rf"https://[a-z0-9]+-{port}\.proxy\.runpod\.net", url):
        raise SystemExit(f"Expected the corresponding RunPod HTTP proxy origin on port {port}.")
PY

mkdir -p "$GF_PATHS_PROVISIONING" "$GF_PATHS_DATA" "$GF_PATHS_LOGS" \
  "$GF_PATHS_PLUGINS" "$VLLM_GRAFANA_DASHBOARDS_PATH"
cp -R "$REPO_DIR/observability/grafana/provisioning/." "$GF_PATHS_PROVISIONING/"
"$PYTHON_BIN" - "$VLLM_DASHBOARD_FILE" "$VLLM_GRAFANA_DASHBOARDS_PATH/vllm-serving-overview.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
dashboard = json.loads(source.read_text(encoding="utf-8"))
if dashboard.get("uid") != "vllm-serving-overview":
    raise SystemExit("Unexpected dashboard UID.")
if not any(
    "vllm:" in target.get("expr", "")
    for panel in dashboard.get("panels", [])
    for target in panel.get("targets", [])
):
    raise SystemExit("Bootstrap dashboard rejected: run existing prepare_dashboard.py with an actual metrics snapshot.")
path = Path(sys.argv[2])
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY

exec "$GRAFANA_BIN" server --homepath "$GRAFANA_HOME"
