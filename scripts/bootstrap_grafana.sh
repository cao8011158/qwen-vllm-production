#!/usr/bin/env bash
# External configuration client: run on the GPU Pod or an operator machine.
# Grafana itself uses grafana/grafana:12.1.0 and its unchanged /run.sh entrypoint.
# This script sends API requests only when the operator explicitly executes it.
set -euo pipefail
umask 077
: "${GRAFANA_URL:?Supply https://<GRAFANA_POD_ID>-3000.proxy.runpod.net}"
: "${PROMETHEUS_URL:?Supply https://<PROMETHEUS_POD_ID>-9090.proxy.runpod.net}"
: "${VLLM_DASHBOARD_FILE:?Supply the dashboard generated from an actual GPU metrics snapshot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import base64
import json
import os
import re
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

grafana = os.environ["GRAFANA_URL"]
prometheus = os.environ["PROMETHEUS_URL"]
for url, port in ((grafana, 3000), (prometheus, 9090)):
    if not re.fullmatch(rf"https://[a-z0-9]+-{port}\.proxy\.runpod\.net", url):
        raise SystemExit(f"Expected the actual HTTPS RunPod proxy origin on port {port}.")
dashboard = json.loads(Path(os.environ["VLLM_DASHBOARD_FILE"]).read_text(encoding="utf-8"))
if dashboard.get("uid") != "vllm-serving-overview" or not any(
    "vllm:" in target.get("expr", "")
    for panel in dashboard.get("panels", [])
    for target in panel.get("targets", [])
):
    raise SystemExit("Use prepare_dashboard.py output from actual /metrics, not the bootstrap up-only dashboard.")
for panel in dashboard.get("panels", []):
    if panel.get("datasource", {}).get("uid") != "vllm-prometheus":
        raise SystemExit("Unexpected panel datasource UID.")
token = os.environ.get("GRAFANA_TOKEN")
if token:
    authorization = "Bearer " + token
else:
    user, password = os.environ.get("GRAFANA_USER"), os.environ.get("GRAFANA_PASSWORD")
    if not user or not password:
        raise SystemExit("Set GRAFANA_TOKEN or GRAFANA_USER + GRAFANA_PASSWORD outside Terraform.")
    authorization = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

opener = build_opener(ProxyHandler({}), HTTPSHandler(context=ssl.create_default_context()), NoRedirect())

def request(method, path, payload=None, *, allow_missing=False):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(grafana + path, data=body, method=method, headers={
        "Authorization": authorization, "Accept": "application/json", "Content-Type": "application/json",
    })
    try:
        with opener.open(req, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if allow_missing and error.code == 404:
            return None
        raise SystemExit(f"Grafana {method} {path} failed: HTTP {error.code}; inspect permissions and service logs.") from None
    except (URLError, ValueError) as error:
        raise SystemExit(f"Grafana {method} {path} failed: {type(error).__name__}.") from None

datasource = {
    "uid": "vllm-prometheus", "name": "vLLM Prometheus", "type": "prometheus",
    "access": "proxy", "url": prometheus, "basicAuth": False, "isDefault": True,
    "jsonData": {"httpMethod": "GET", "timeInterval": "5s"},
}
existing = request("GET", "/api/datasources/uid/vllm-prometheus", allow_missing=True)
if existing is None:
    request("POST", "/api/datasources", datasource)
else:
    if existing.get("type") != "prometheus" or existing.get("readOnly"):
        raise SystemExit("The datasource UID is occupied by a different/provisioned datasource; review it explicitly.")
    datasource["id"] = existing["id"]
    request("PUT", "/api/datasources/uid/vllm-prometheus", datasource)

existing_dashboard = request("GET", "/api/dashboards/uid/vllm-serving-overview", allow_missing=True)
dashboard["id"] = None if existing_dashboard is None else existing_dashboard["dashboard"]["id"]
dashboard["version"] = 0 if existing_dashboard is None else existing_dashboard["dashboard"]["version"]
payload = {"dashboard": dashboard, "overwrite": True, "message": "RunPod runtime metrics dashboard"}
if existing_dashboard is not None and existing_dashboard.get("meta", {}).get("folderUid"):
    payload["folderUid"] = existing_dashboard["meta"]["folderUid"]
request("POST", "/api/dashboards/db", payload)
print("Configured datasource vllm-prometheus and dashboard vllm-serving-overview through Grafana HTTP API.")
print("Run the existing check_prometheus.py/check_grafana.py manually to verify actual data.")
PY
