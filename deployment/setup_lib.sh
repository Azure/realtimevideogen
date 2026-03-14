#!/usr/bin/env bash
# Shared utility functions for setup_image.sh scripts

ensure_acr_login() {
  local acr_name="$1"
  if ! az acr login --name "$acr_name" >/dev/null 2>&1; then
    echo "ERROR: Failed to log into ACR '$acr_name': az acr login --name $acr_name"
    exit 1
  fi
}

# Detect the current platform architecture for docker buildx
detect_platform() {
  if [[ "$(uname -m)" == "aarch64" ]]; then
    echo "linux/arm64"
  else
    echo "linux/amd64"
  fi
}
