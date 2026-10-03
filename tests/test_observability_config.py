"""Static provisioning and runtime metric selection; no Prometheus/Grafana."""

import json
from pathlib import Path

import pytest
import yaml

from qwen_vllm_production.observability.metrics import PANELS, build_dashboard, metric_types


ROOT = Path(__file__).resolve().parents[1]


def test_prometheus_and_grafana_provisioning() -> None:
    prom = yaml.safe_load((ROOT / "observability/prometheus/prometheus.yml").read_text(encoding="utf-8"))
    assert prom["global"]["scrape_interval"] == "5s"
    assert prom["scrape_configs"][0]["job_name"] == "vllm"
    assert prom["scrape_configs"][0]["metrics_path"] == "/metrics"
    ds = yaml.safe_load((ROOT / "observability/grafana/provisioning/datasources/prometheus.yml").read_text(encoding="utf-8"))
    assert ds["datasources"][0]["type"] == "prometheus"
    assert ds["datasources"][0]["uid"] == "vllm-prometheus"
    provider = yaml.safe_load((ROOT / "observability/grafana/provisioning/dashboards/vllm.yml").read_text(encoding="utf-8"))
    assert provider["providers"][0]["options"]["path"] == "${VLLM_GRAFANA_DASHBOARDS_PATH}"
    bootstrap = json.loads((ROOT / "observability/grafana/dashboards/vllm-serving-overview.json").read_text(encoding="utf-8"))
    assert bootstrap["uid"] == "vllm-serving-overview"


def test_runtime_dashboard_only_uses_available_metric_types() -> None:
    # This fixture verifies selection logic; it is not proof of a release's exports.
    snapshot = "\n".join(f"# TYPE {family} {kind}" for _, family, kind, _ in PANELS)
    dashboard, manifest = build_dashboard(snapshot)
    assert len(dashboard["panels"]) == len(PANELS)
    assert not manifest["omitted_panels"]
    assert {panel["title"] for panel in dashboard["panels"]} == {row[0] for row in PANELS}
    for panel, selected in zip(dashboard["panels"], manifest["included_panels"]):
        if selected["type"] == "histogram":
            assert all("histogram_quantile(" in target["expr"] for target in panel["targets"])
            assert panel["fieldConfig"]["defaults"]["unit"] == "ms"
    filtered, manifest = build_dashboard("# TYPE vllm:num_requests_running gauge\n")
    assert len(filtered["panels"]) == 1
    assert manifest["omitted_panels"]
    assert metric_types("# TYPE vllm:num_requests_running gauge\n")["vllm:num_requests_running"] == "gauge"
    with pytest.raises(ValueError):
        build_dashboard("# TYPE unknown gauge\n")
