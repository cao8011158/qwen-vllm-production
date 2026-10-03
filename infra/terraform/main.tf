# Verified official resource, with its actual v1.0.8 field names.
# No Pod resource is fabricated: GPU selection/startup are not fully
# transmitted by v1.0.8, and CPU creation fields are absent from its schema.
resource "runpod_network_volume" "workspace" {
  count = var.network_volume_id == null ? 1 : 0

  name           = "qwen-vllm-production"
  size           = var.network_volume_size_gb
  data_center_id = var.data_center_id

  lifecycle {
    prevent_destroy = true
  }
}

locals {
  network_volume_id = var.network_volume_id == null ? runpod_network_volume.workspace[0].id : var.network_volume_id
  gpu_proxy_url     = var.gpu_pod_id == null ? null : "https://${var.gpu_pod_id}-8000.proxy.runpod.net"
  prometheus_url    = var.prometheus_pod_id == null ? null : "https://${var.prometheus_pod_id}-9090.proxy.runpod.net"
  grafana_url       = var.grafana_pod_id == null ? null : "https://${var.grafana_pod_id}-3000.proxy.runpod.net"

  repository_base_url = var.repository_url == null ? null : trimsuffix(var.repository_url, ".git")
  gpu_bootstrap_url   = var.repository_url == null || var.repository_revision == null ? null : "${replace(local.repository_base_url, "https://github.com/", "https://raw.githubusercontent.com/")}/${var.repository_revision}/scripts/bootstrap_gpu.sh"

  external_pod_ids = compact([
    var.gpu_pod_id == null ? "" : var.gpu_pod_id,
    var.prometheus_pod_id == null ? "" : var.prometheus_pod_id,
    var.grafana_pod_id == null ? "" : var.grafana_pod_id,
  ])
}
