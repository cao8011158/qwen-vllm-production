"""Prometheus metric inventory and dashboards gated by an actual /metrics snapshot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..evaluation.common import save_json


DATASOURCE_UID = "vllm-prometheus"
DASHBOARD_UID = "vllm-serving-overview"
DEFAULT_QUERY = "vllm:num_requests_running"

# These are vLLM metric families, not a guarantee that every family is exported
# by every release/configuration. prepare_dashboard requires the runtime TYPE
# descriptors and drops any absent or differently typed family.
PANELS = (
    ("Running Requests", "vllm:num_requests_running", "gauge", "short"),
    ("Waiting Requests / Queue Pressure", "vllm:num_requests_waiting", "gauge", "short"),
    ("KV Cache Utilization", "vllm:kv_cache_usage_perc", "gauge", "percent"),
    ("TTFT", "vllm:time_to_first_token_seconds", "histogram", "ms"),
    ("E2E Latency", "vllm:e2e_request_latency_seconds", "histogram", "ms"),
    ("Request Time Per Output Token", "vllm:request_time_per_output_token_seconds", "histogram", "ms"),
    ("Inter-token Latency", "vllm:inter_token_latency_seconds", "histogram", "ms"),
    ("Prompt Token Throughput", "vllm:prompt_tokens_total", "counter", "ops"),
    ("Generation Token Throughput", "vllm:generation_tokens_total", "counter", "ops"),
    ("Server Completed Request Activity", "vllm:request_success_total", "counter", "reqps"),
    ("Queue Time", "vllm:request_queue_time_seconds", "histogram", "ms"),
    ("Prefill Time", "vllm:request_prefill_time_seconds", "histogram", "ms"),
    ("Decode Time", "vllm:request_decode_time_seconds", "histogram", "ms"),
)


def metric_types(text: str) -> dict[str, str]:
    return dict(re.findall(r"^# TYPE ([a-zA-Z_:][a-zA-Z0-9_:]*) (counter|gauge|histogram|summary|untyped)\s*$", text, flags=re.MULTILINE))


def build_dashboard(snapshot: str) -> tuple[dict, dict]:
    """Only reference families present with the expected type in this snapshot."""
    available = metric_types(snapshot)
    selector = '{job="vllm",instance=~"$instance"}'
    panels, omitted, selected = [], [], []
    for title, family, kind, unit in PANELS:
        if available.get(family) != kind:
            omitted.append({"title": title, "metric": family, "expected_type": kind, "observed_type": available.get(family)})
            continue
        if kind == "histogram":
            targets = [
                {
                    "refId": label, "legendFormat": label,
                    "expr": f"1000 * histogram_quantile({q}, sum by (le) (rate({family}_bucket{selector}[$__rate_interval])))",
                }
                for label, q in (("p50", 0.50), ("p95", 0.95))
            ]
        else:
            expr = (
                f"sum(rate({family}{selector}[$__rate_interval]))" if kind == "counter"
                else f"{'100 * max' if unit == 'percent' else 'sum'}({family}{selector})"
            )
            targets = [{"refId": "A", "expr": expr, "legendFormat": title}]
        panel_id = len(panels) + 1
        panels.append({
            "id": panel_id, "title": title, "type": "timeseries",
            "datasource": {"type": "prometheus", "uid": DATASOURCE_UID},
            "gridPos": {"h": 8, "w": 12, "x": ((panel_id - 1) % 2) * 12, "y": ((panel_id - 1) // 2) * 8},
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
            "targets": targets,
        })
        selected.append({"title": title, "metric": family, "type": kind})
    if not panels:
        raise ValueError("No supported vLLM metric families found in this runtime snapshot.")
    dashboard = {
        "uid": DASHBOARD_UID, "title": "vLLM Serving Overview",
        "schemaVersion": 39, "version": 1, "editable": True,
        "tags": ["vllm", "phase3a"], "timezone": "utc",
        "refresh": "5s", "time": {"from": "now-15m", "to": "now"},
        "description": "Server observability. Client-side benchmark JSON is authoritative for SLOs.",
        "templating": {"list": [{
            "name": "instance", "type": "query", "label": "vLLM target",
            "datasource": {"type": "prometheus", "uid": DATASOURCE_UID},
            "query": 'label_values(up{job="vllm"}, instance)',
            "refresh": 1, "multi": False, "includeAll": False,
        }]},
        "panels": panels,
    }
    return dashboard, {"included_panels": selected, "omitted_panels": omitted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Grafana dashboard from real vLLM /metrics TYPE descriptors.")
    parser.add_argument("--metrics-file", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("observability/grafana/dashboards/vllm-serving-overview.json"))
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    snapshot = args.metrics_file.read_text(encoding="utf-8")
    dashboard, manifest = build_dashboard(snapshot)
    save_json(dashboard, args.output)
    if args.manifest_output:
        save_json(manifest, args.manifest_output)
    print(f"Prepared {len(dashboard['panels'])} panels from the actual metrics snapshot.")
    for panel in manifest["omitted_panels"]:
        print(f"Omitted unavailable/type-mismatched metric: {panel['metric']}")
