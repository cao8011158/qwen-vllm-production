terraform {
  required_version = ">= 1.5.0, < 2.0.0"

  backend "local" {}

  required_providers {
    runpod = {
      source  = "runpod/runpod"
      version = "= 1.0.8"
    }
  }
}

# Official latest release at review time: 1.0.9 (2026-10-04).
# Pin 1.0.8: its Network Volume resource and provider client agree.
# See docs/runpod_iac.md for the 1.0.9 client mismatch and Pod limitations.
# Authentication comes only from RUNPOD_API_KEY, never from tfvars.
provider "runpod" {
  base_url = "https://rest.runpod.io/v1"
}
