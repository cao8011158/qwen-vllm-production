variable "data_center_id" {
  type        = string
  nullable    = false
  description = "Real RunPod data center with Network Volume support and A100 SXM 80GB availability. No ID is assumed."

  validation {
    condition     = can(regex("^[A-Z0-9]+(-[A-Z0-9]+)+$", var.data_center_id))
    error_message = "Supply the actual RunPod data center ID selected by the operator."
  }
}

variable "network_volume_id" {
  type        = string
  default     = null
  description = "Optional existing external Network Volume ID. Null creates the volume with the official resource."

  validation {
    condition     = var.network_volume_id == null ? true : can(regex("^[A-Za-z0-9_-]+$", var.network_volume_id))
    error_message = "Use null or a non-empty existing Network Volume ID."
  }
}

variable "network_volume_size_gb" {
  type        = number
  nullable    = false
  default     = 100
  description = "Network Volume capacity in GB; 100 by default. Existing external volumes are not resized."

  validation {
    condition     = var.network_volume_size_gb >= 100 && floor(var.network_volume_size_gb) == var.network_volume_size_gb
    error_message = "Capacity must be an integer of at least 100 GB for this deployment."
  }
}

variable "gpu_pod_id" {
  type        = string
  default     = null
  description = "Externally created GPU Pod ID. The official Pod implementation cannot reliably enforce this deployment."

  validation {
    condition     = var.gpu_pod_id == null ? true : can(regex("^[a-z0-9]+$", var.gpu_pod_id))
    error_message = "Use null before deployment, or the actual GPU Pod ID."
  }
}

variable "prometheus_pod_id" {
  type        = string
  default     = null
  description = "Externally created CPU-only Prometheus Pod ID; not a Terraform-managed resource."

  validation {
    condition     = var.prometheus_pod_id == null ? true : can(regex("^[a-z0-9]+$", var.prometheus_pod_id))
    error_message = "Use null before deployment, or the actual Prometheus Pod ID."
  }
}

variable "grafana_pod_id" {
  type        = string
  default     = null
  description = "Externally created CPU-only Grafana Pod ID; not a Terraform-managed resource."

  validation {
    condition     = var.grafana_pod_id == null ? true : can(regex("^[a-z0-9]+$", var.grafana_pod_id))
    error_message = "Use null before deployment, or the actual Grafana Pod ID."
  }
}

variable "repository_url" {
  type        = string
  default     = null
  description = "Actual GitHub HTTPS URL cloned at runtime in the GPU Pod. No repository is baked into an image."

  validation {
    condition     = var.repository_url == null ? true : can(regex("^https://github\\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\\.git)?$", var.repository_url))
    error_message = "Use null or the actual GitHub HTTPS repository URL without embedded credentials."
  }
}

variable "repository_revision" {
  type        = string
  default     = null
  description = "Actual repository commit cloned by GPU bootstrap. CPU Pods do not use the repository."

  validation {
    condition     = var.repository_revision == null ? true : can(regex("^[a-f0-9]{40}$", var.repository_revision))
    error_message = "Use null or the real full repository commit SHA."
  }
}

variable "awq_source_revision" {
  type        = string
  default     = null
  description = "Actual immutable Hugging Face revision used to produce AWQ; BF16 must match it."

  validation {
    condition     = var.awq_source_revision == null ? true : can(regex("^[a-f0-9]{40}$", var.awq_source_revision))
    error_message = "Use null or the actual full source-model commit SHA."
  }
}

variable "awq_gdrive_source" {
  type        = string
  default     = null
  description = "rclone Google Drive remote:folder for the checkpoint contents. Never place credentials or sharing URLs here."

  validation {
    condition     = var.awq_gdrive_source == null ? true : can(regex("^[A-Za-z0-9_-]+:.+$", var.awq_gdrive_source))
    error_message = "Use null or a configured rclone remote:folder, not an HTTP URL."
  }
}
