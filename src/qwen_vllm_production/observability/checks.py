"""Mockable, minimal HTTP checks; no monitoring processes or SDKs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from urllib.parse import quote

import httpx

from .metrics import DASHBOARD_UID, DATASOURCE_UID, DEFAULT_QUERY, metric_types


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    checks: list[Check]
    metrics_snapshot: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


async def get(client: httpx.AsyncClient, url: str, *, params: dict | None = None) -> httpx.Response:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response


def query_has_data(payload: dict) -> bool:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return False
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "vector":
        return False
    results = data.get("result")
    if not isinstance(results, list) or not results:
        return False
    try:
        return all(isinstance(entry, dict) and math.isfinite(float(entry["value"][1])) for entry in results)
    except (KeyError, IndexError, TypeError, ValueError):
        return False


async def check_serving(client: httpx.AsyncClient, base_url: str, model: str) -> Report:
    base = base_url.rstrip("/")
    report = Report([])
    for name, path in (("health", "/health"), ("models", "/v1/models"), ("metrics", "/metrics")):
        try:
            response = await get(client, base + path)
            if name == "health":
                ok, detail = True, f"HTTP {response.status_code}"
            elif name == "models":
                payload = response.json()
                models = payload.get("data", []) if isinstance(payload, dict) else []
                ok = isinstance(models, list) and any(isinstance(item, dict) and item.get("id") == model for item in models)
                detail = f"Expected served model: {model}"
            else:
                text = response.text
                families = metric_types(text)
                ok = any(key.startswith("vllm:") for key in families)
                detail = f"{sum(key.startswith('vllm:') for key in families)} vLLM TYPE descriptors"
                if ok:
                    report.metrics_snapshot = text
            report.checks.append(Check(name, ok, detail))
        except Exception as exc:
            report.checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
    return report


async def check_prometheus(
    client: httpx.AsyncClient, base_url: str, *, job: str = "vllm", query: str = DEFAULT_QUERY,
) -> Report:
    base = base_url.rstrip("/")
    report = Report([])
    for name, path in (("healthy", "/-/healthy"), ("ready", "/-/ready"), ("targets", "/api/v1/targets"), ("query", "/api/v1/query")):
        try:
            response = await get(client, base + path, params={"query": query} if name == "query" else None)
            if name in ("healthy", "ready"):
                ok, detail = True, f"HTTP {response.status_code}"
            elif name == "targets":
                payload = response.json()
                data = payload.get("data", {})
                active = data.get("activeTargets", [])
                targets = [target for target in active if target.get("labels", {}).get("job") == job]
                ok = payload.get("status") == "success" and bool(targets) and all(target.get("health") == "up" and not target.get("lastError") for target in targets)
                detail = f"job={job}: {len(targets)} target(s); health={[target.get('health') for target in targets]}"
            else:
                ok = query_has_data(response.json())
                detail = f"Query must succeed with a non-empty finite vector: {query}"
            report.checks.append(Check(name, ok, detail))
        except Exception as exc:
            report.checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
    return report


async def check_grafana(
    client: httpx.AsyncClient, base_url: str, *,
    datasource_uid: str = DATASOURCE_UID, dashboard_uid: str = DASHBOARD_UID,
    query: str = DEFAULT_QUERY,
) -> Report:
    base = base_url.rstrip("/")
    ds = quote(datasource_uid, safe="")
    dash = quote(dashboard_uid, safe="")
    report = Report([])
    endpoints = (
        ("health", "/api/health"),
        ("datasource", f"/api/datasources/uid/{ds}"),
        ("dashboard", f"/api/dashboards/uid/{dash}"),
        ("datasource_query", f"/api/datasources/proxy/uid/{ds}/api/v1/query"),
    )
    for name, path in endpoints:
        try:
            response = await get(client, base + path, params={"query": query} if name == "datasource_query" else None)
            payload = response.json()
            if name == "health":
                ok, detail = payload.get("database") == "ok", "Grafana database must be ok"
            elif name == "datasource":
                ok = payload.get("uid") == datasource_uid and payload.get("type") == "prometheus"
                detail = f"Expected Prometheus datasource UID: {datasource_uid}"
            elif name == "dashboard":
                dashboard = payload.get("dashboard", {})
                panels = dashboard.get("panels", [])
                has_serving_panels = any(
                    "vllm:" in target.get("expr", "")
                    for panel in panels for target in panel.get("targets", [])
                )
                ok = dashboard.get("uid") == dashboard_uid and has_serving_panels
                detail = f"Expected dashboard UID {dashboard_uid} with runtime-prepared vLLM panels"
            else:
                ok, detail = query_has_data(payload), f"Prometheus query through Grafana: {query}"
            report.checks.append(Check(name, ok, detail))
        except Exception as exc:
            report.checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
    return report
