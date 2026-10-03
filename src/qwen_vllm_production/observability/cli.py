"""Manual HTTP validation entrypoints with non-zero failure exit status."""

import argparse
import asyncio
import os
from pathlib import Path

import httpx

from .checks import check_grafana, check_prometheus, check_serving
from .metrics import DASHBOARD_UID, DATASOURCE_UID, DEFAULT_QUERY


def main(kind: str) -> None:
    parser = argparse.ArgumentParser(description=f"Validate {kind} via its HTTP APIs.")
    defaults = {
        "serving": ("VLLM_URL", "http://127.0.0.1:8000"),
        "prometheus": ("PROMETHEUS_URL", "http://127.0.0.1:9090"),
        "grafana": ("GRAFANA_URL", "http://127.0.0.1:3000"),
    }
    env_name, default_url = defaults[kind]
    parser.add_argument("--base-url", default=os.environ.get(env_name, default_url))
    parser.add_argument("--timeout", type=float, default=10.0)
    if kind == "serving":
        parser.add_argument("--model", required=True)
        parser.add_argument("--metrics-output", type=Path)
    else:
        parser.add_argument("--query", default=DEFAULT_QUERY)
    if kind == "prometheus":
        parser.add_argument("--job", default="vllm")
    if kind == "grafana":
        parser.add_argument("--datasource-uid", default=DATASOURCE_UID)
        parser.add_argument("--dashboard-uid", default=DASHBOARD_UID)
        parser.add_argument("--user", default=os.environ.get("GRAFANA_USER"))
        # Password/token come only from environment, avoiding command-line secrets.
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive.")

    async def validate():
        kwargs = {}
        if kind == "grafana":
            token = os.environ.get("GRAFANA_TOKEN")
            password = os.environ.get("GRAFANA_PASSWORD")
            if token:
                kwargs["headers"] = {"Authorization": f"Bearer {token}"}
            elif args.user and password:
                kwargs["auth"] = (args.user, password)
            else:
                parser.error("Configure GRAFANA_TOKEN or GRAFANA_USER + GRAFANA_PASSWORD.")
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False, **kwargs) as client:
            if kind == "serving":
                return await check_serving(client, args.base_url, args.model)
            if kind == "prometheus":
                return await check_prometheus(client, args.base_url, job=args.job, query=args.query)
            return await check_grafana(
                client, args.base_url, datasource_uid=args.datasource_uid,
                dashboard_uid=args.dashboard_uid, query=args.query,
            )

    report = asyncio.run(validate())
    for check in report.checks:
        print(f"{'OK' if check.ok else 'FAIL'} {check.name}: {check.detail}")
    if kind == "serving" and report.ok and args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(report.metrics_snapshot, encoding="utf-8")
    raise SystemExit(0 if report.ok else 1)
