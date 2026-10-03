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

variable "pod_images" {
  type = object({
    gpu        = optional(string)
    prometheus = optional(string)
    grafana    = optional(string)
  })
  default     = {}
  nullable    = false
  description = "Operator-supplied immutable image references for the external Pods. Image contracts are in docs/runpod_iac.md."

  validation {
    condition = alltrue([
      for image in [var.pod_images.gpu, var.pod_images.prometheus, var.pod_images.grafana] :
      image == null ? true : can(regex("^.+@sha256:[a-f0-9]{64}$", image))
    ])
    error_message = "Leave unknown images null; record real image@sha256:digest references when available."
  }
}

variable "deployment_revision" {
  type        = string
  default     = null
  description = "Actual 40-character repository commit used in all three images. Required by bootstrap, not invented by IaC."

  validation {
    condition     = var.deployment_revision == null ? true : can(regex("^[a-f0-9]{40}$", var.deployment_revision))
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
