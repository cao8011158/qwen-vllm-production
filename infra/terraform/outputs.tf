output "network_volume_id" {
  value       = local.network_volume_id
  description = "Attach this volume to the external GPU Pod at /workspace during Pod creation."
}

output "network_volume_managed" {
  value       = var.network_volume_id == null
  description = "True when this configuration creates and tracks the Network Volume."
}

output "benchmark_base_url" {
  value       = "http://127.0.0.1:8000"
  description = "Use only inside the same GPU Pod as vLLM. Never substitute the GPU proxy URL."
}

output "monitoring_urls" {
  value = {
    vllm_metrics = local.gpu_proxy_url == null ? null : "${local.gpu_proxy_url}/metrics"
    prometheus   = local.prometheus_url
    grafana      = local.grafana_url
  }
  description = "Monitoring-only proxy URLs, derived from externally supplied Pod IDs."
}

output "deployment_contract" {
  value = {
    terraform_manages_pods = false
    gpu                    = {
      id                = var.gpu_pod_id
      image             = var.pod_images.gpu
      compute_type      = "GPU"
      gpu_target        = "1 x NVIDIA A100 SXM 80GB"
      cloud             = "SECURE"
      data_center       = var.data_center_id
      network_volume_id = local.network_volume_id
      mount_path        = "/workspace"
      container_disk_gb = 50
      exposed_http_port = 8000
      startup_script    = "scripts/bootstrap_gpu.sh"
    }
    prometheus             = {
      id                = var.prometheus_pod_id
      image             = var.pod_images.prometheus
      compute_type      = "CPU"
      gpu_count         = 0
      container_disk_gb = 20
      exposed_http_port = 9090
      startup_script    = "scripts/bootstrap_prometheus.sh"
    }
    grafana                = {
      id                = var.grafana_pod_id
      image             = var.pod_images.grafana
      compute_type      = "CPU"
      gpu_count         = 0
      container_disk_gb = 20
      exposed_http_port = 3000
      startup_script    = "scripts/bootstrap_grafana.sh"
    }
  }
  description = "Desired settings to apply externally; this output does not create, inspect, or modify Pods."

  precondition {
    condition     = length(distinct(local.external_pod_ids)) == length(local.external_pod_ids)
    error_message = "GPU, Prometheus, and Grafana must be three distinct Pods."
  }
}

output "bootstrap_environment" {
  value = {
    gpu = {
      DEPLOYMENT_REVISION     = var.deployment_revision
      AWQ_SOURCE_REVISION     = var.awq_source_revision
      AWQ_GDRIVE_SOURCE       = var.awq_gdrive_source
      NETWORK_VOLUME_ID      = local.network_volume_id
      MODEL_VARIANT          = "awq_w4a16"
      SERVED_MODEL_NAME      = "qwen3-14b"
      MAX_MODEL_LEN          = "8192"
      GPU_MEMORY_UTILIZATION = "0.90"
      TENSOR_PARALLEL_SIZE    = "1"
      DTYPE                  = "bfloat16"
      SEED                   = "42"
      RCLONE_CONFIG          = "/run/secrets/rclone.conf"
    }
    prometheus = {
      DEPLOYMENT_REVISION  = var.deployment_revision
      VLLM_METRICS_URL     = local.gpu_proxy_url == null ? null : "${local.gpu_proxy_url}/metrics"
    }
    grafana = {
      DEPLOYMENT_REVISION = var.deployment_revision
      PROMETHEUS_URL      = local.prometheus_url
      GRAFANA_PUBLIC_URL  = local.grafana_url
    }
  }
  description = "Non-secret bootstrap inputs. Null inputs must be supplied before starting a Pod. Credentials stay outside Terraform."
}
