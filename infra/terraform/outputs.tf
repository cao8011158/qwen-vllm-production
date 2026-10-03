output "network_volume_id" {
  value       = local.network_volume_id
  description = "Manually attach this volume to the GPU Pod at /workspace during Pod creation."
}

output "network_volume_managed" {
  value       = var.network_volume_id == null
  description = "True when Terraform creates and tracks the Network Volume."
}

output "benchmark_base_url" {
  value       = "http://127.0.0.1:8000"
  description = "Use only from the benchmark client on the same GPU Pod as vLLM."
}

output "monitoring_urls" {
  value = {
    vllm_metrics = local.gpu_proxy_url == null ? null : "${local.gpu_proxy_url}/metrics"
    prometheus   = local.prometheus_url
    grafana      = local.grafana_url
  }
  description = "Monitoring-only HTTPS proxy URLs, derived from external Pod IDs."
}

output "gpu_start_command" {
  value = jsonencode({
    entrypoint = ["/bin/bash", "-lc"]
    cmd        = ["curl --fail --location \"$GPU_BOOTSTRAP_URL\" --output /tmp/qwen-bootstrap-gpu.sh && exec /bin/bash /tmp/qwen-bootstrap-gpu.sh"]
  })
  description = "Paste into RunPod Console Container Start Command for a public, published GitHub commit. Prepare Drive credentials first, or use the documented manual first-start mode."
}

output "deployment_contract" {
  value = {
    terraform_manages_pods = false
    gpu = {
      id                = var.gpu_pod_id
      image             = "vllm/vllm-openai:v0.26.0"
      compute_type      = "GPU"
      gpu_target        = "1 x NVIDIA A100 SXM 80GB"
      cloud             = "SECURE"
      data_center       = var.data_center_id
      network_volume_id = local.network_volume_id
      mount_path        = "/workspace"
      container_disk_gb = 50
      exposed_http_port = 8000
    }
    prometheus = {
      id                       = var.prometheus_pod_id
      image                    = "prom/prometheus:v3.5.0"
      compute_type             = "CPU"
      gpu_count                = 0
      container_disk_gb        = 20
      volume_disk_gb           = 0
      data_path                = "/prometheus"
      exposed_http_port        = 9090
      external_config_tool     = "scripts/bootstrap_prometheus.sh"
      console_start_command    = "Paste the generated prometheus-start-command.json"
    }
    grafana = {
      id                    = var.grafana_pod_id
      image                 = "grafana/grafana:12.1.0"
      compute_type          = "CPU"
      gpu_count             = 0
      container_disk_gb     = 20
      volume_disk_gb        = 0
      data_path             = "/var/lib/grafana"
      exposed_http_port     = 3000
      console_start_command = "Leave blank to preserve the official /run.sh entrypoint"
      external_config_tool  = "scripts/bootstrap_grafana.sh"
    }
  }
  description = "Settings for manual Console deployment with official images. CPU data is on container disk in this minimal configuration."

  precondition {
    condition     = length(distinct(local.external_pod_ids)) == length(local.external_pod_ids)
    error_message = "GPU, Prometheus, and Grafana must be three distinct Pods."
  }
}

output "bootstrap_environment" {
  value = {
    gpu = {
      REPOSITORY_URL     = var.repository_url
      REPOSITORY_REVISION = var.repository_revision
      GPU_BOOTSTRAP_URL  = local.gpu_bootstrap_url
      AWQ_SOURCE_REVISION = var.awq_source_revision
      AWQ_GDRIVE_SOURCE  = var.awq_gdrive_source
      NETWORK_VOLUME_ID = local.network_volume_id
      MODEL_VARIANT     = "awq_w4a16"
      RCLONE_CONFIG     = "/workspace/.secrets/rclone.conf"
    }
    prometheus = {}
    grafana = {
      GF_SECURITY_ADMIN_USER = "admin"
      GF_USERS_ALLOW_SIGN_UP = "false"
      GF_SERVER_HTTP_ADDR    = "0.0.0.0"
      GF_SERVER_HTTP_PORT    = "3000"
      GF_SERVER_ROOT_URL     = local.grafana_url == null ? null : "${local.grafana_url}/"
    }
  }
  description = "Non-secret Console environment inputs. Supply GF_SECURITY_ADMIN_PASSWORD privately in Grafana Console settings; rclone file contents stay outside Terraform."
}

output "configuration_client_environment" {
  value = {
    prometheus = {
      VLLM_METRICS_URL = local.gpu_proxy_url == null ? null : "${local.gpu_proxy_url}/metrics"
    }
    grafana = {
      PROMETHEUS_URL = local.prometheus_url
      GRAFANA_URL    = local.grafana_url
    }
  }
  description = "Inputs to the external configuration scripts, not environment requirements inside the CPU images. Grafana credentials and dashboard file path are supplied separately."
}
