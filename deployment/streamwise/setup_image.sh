#!/usr/bin/env bash
set -euo pipefail

ensure_acr_login() {
  local acr_name="$1"
  if ! az acr login --name "$acr_name" >/dev/null 2>&1; then
    echo "ERROR: Failed to log into ACR '$acr_name': az acr login --name $acr_name"
    exit 1
  fi
}

IMAGE_NAME="streamwise"

PUSH_IMAGE=false

PLATFORM="linux/amd64"  # "linux/arm64"
if [[ "$(uname -m)" == "aarch64" ]]; then
  PLATFORM="linux/arm64"
fi

# Main script
MAIN_DIR=$(realpath ../..)
DEPLOYMENT_DIR=$MAIN_DIR/deployment
APP_DIR=$MAIN_DIR/$IMAGE_NAME

# Source set_properties.sh only if DOCKER_REPO is not already provided in the environment
if [[ -z "${DOCKER_REPO:-}" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "$DEPLOYMENT_DIR/set_properties.sh"
fi

TAG=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")

mkdir -p docker_files

# Copy necessary files from the app directory to the current directory
cp "$DEPLOYMENT_DIR/streamwise/Dockerfile" ./docker_files/

cp "$APP_DIR/requirements.txt" ./docker_files/

cp "$MAIN_DIR"/*.py ./docker_files/
cp "$APP_DIR"/*.py ./docker_files/
cp "$APP_DIR"/*.bash ./docker_files/

cp -R "$APP_DIR"/static ./docker_files/
cp -R "$APP_DIR"/templates ./docker_files/

cp "$MAIN_DIR/services.json" ./docker_files/

BASE_TAG=$(jq -r '.base.dockerImage.tag' "$MAIN_DIR/services.json")

if [[ "$PUSH_IMAGE" == true ]]; then
  ensure_acr_login "$DOCKER_REPO"
fi

# Build
docker buildx build \
  --build-arg DOCKER_REPO="${DOCKER_REPO}" \
  --build-arg BASE_TAG="${BASE_TAG}" \
  --platform "$PLATFORM" \
  -t "${IMAGE_NAME}:${TAG}" \
  ./docker_files/

# Tag final image for pushing
docker tag "${IMAGE_NAME}:${TAG}" "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"

if [[ "$PUSH_IMAGE" == true ]]; then
  docker push "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"
fi
