#!/usr/bin/env bash
set -euo pipefail

# Argument parsing
usage() {
  echo "Usage: $0 <IMAGE_NAME> [--push] [--platform linux/amd64|linux/arm64]"
  exit 1
}

ensure_acr_login() {
  local acr_name="$1"
  if ! az acr login --name "$acr_name" >/dev/null 2>&1; then
    echo "ERROR: Failed to log into ACR '$acr_name': az acr login --name $acr_name"
    exit 1
  fi
}

IMAGE_NAME="${1:-}"
[[ -z "$IMAGE_NAME" ]] && usage

PUSH_IMAGE=false

PLATFORM="linux/amd64"
if [[ "$(uname -m)" == "aarch64" ]]; then
  PLATFORM="linux/arm64"
fi

shift || true

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH_IMAGE=true
      shift
      ;;
    --platform)
      shift
      PLATFORM="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Main script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAIN_DIR=$(realpath "$SCRIPT_DIR/..")
DEPLOYMENT_DIR=$MAIN_DIR/deployment
IMAGE_DIR=$DEPLOYMENT_DIR/$IMAGE_NAME

# Source set_properties.sh only if DOCKER_REPO is not already provided in the environment
if [[ -z "${DOCKER_REPO:-}" ]] && [[ -f "$DEPLOYMENT_DIR/set_properties.sh" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "$DEPLOYMENT_DIR/set_properties.sh"
fi

TAG=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")
BASE_TAG=$(jq -r '.base.dockerImage.tag' "$MAIN_DIR/services.json")

mkdir -p "$IMAGE_DIR/docker_files"

# Copy Dockerfile and base requirements into the build context
cp "$IMAGE_DIR/Dockerfile" "$IMAGE_DIR/docker_files/"
cp "$MAIN_DIR/requirements.txt" "$IMAGE_DIR/docker_files/base_requirements.txt"

# Image-specific file preparation
if [[ "$IMAGE_NAME" == "streamwise" ]]; then
  APP_DIR="$MAIN_DIR/$IMAGE_NAME"
  cp "$APP_DIR/requirements.txt" "$IMAGE_DIR/docker_files/"
  cp "$MAIN_DIR"/*.py "$IMAGE_DIR/docker_files/"
  cp "$APP_DIR"/*.py "$IMAGE_DIR/docker_files/"
  cp "$APP_DIR"/*.bash "$IMAGE_DIR/docker_files/"
  [[ -d "$APP_DIR/static" ]] && cp -R "$APP_DIR/static" "$IMAGE_DIR/docker_files/"
  [[ -d "$APP_DIR/templates" ]] && cp -R "$APP_DIR/templates" "$IMAGE_DIR/docker_files/"
  cp "$MAIN_DIR/services.json" "$IMAGE_DIR/docker_files/"
fi

BUILD_ARGS=(
  docker buildx build
  --build-arg "DOCKER_REPO=${DOCKER_REPO}"
  --build-arg "BASE_TAG=${BASE_TAG}"
  --build-arg "TARGETARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
  --platform "$PLATFORM"
  -t "${IMAGE_NAME}:${TAG}"
  "$IMAGE_DIR/docker_files"
)

# Allow CI to override Dockerfile ARGs via environment variables
if [[ -n "${BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi
if [[ -n "${TORCH_PACKAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "TORCH_PACKAGE=${TORCH_PACKAGE}")
fi
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  BUILD_ARGS+=(--build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}")
fi

if [[ "$PUSH_IMAGE" == true ]]; then
  ensure_acr_login "$DOCKER_REPO"
fi

# Build
"${BUILD_ARGS[@]}"

# Tag final image for pushing
docker tag "${IMAGE_NAME}:${TAG}" "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"

if [[ "$PUSH_IMAGE" == true ]]; then
  docker push "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"
fi
