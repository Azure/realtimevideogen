#!/usr/bin/env bash
set -euo pipefail

ensure_acr_login() {
  local acr_name="$1"
  if ! az acr login --name "$acr_name" >/dev/null 2>&1; then
    echo "ERROR: Failed to log into ACR '$acr_name': az acr login --name $acr_name"
    exit 1
  fi
}

IMAGE_NAME="base"
PUSH_IMAGE=false

PLATFORM="linux/amd64"
if [[ "$(uname -m)" == "aarch64" ]]; then
  PLATFORM="linux/arm64"
fi

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
      exit 1
      ;;
  esac
done

MAIN_DIR=$(realpath ../../..)
DEPLOYMENT_DIR=$MAIN_DIR/deployment

# shellcheck disable=SC1090,SC1091
source "$DEPLOYMENT_DIR/set_properties.sh"

TAG=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")

mkdir -p ./docker_files
cp "$MAIN_DIR/requirements.txt" ./docker_files/base_requirements.txt

ensure_acr_login "$DOCKER_REPO"

docker buildx build \
  --build-arg "DOCKER_REPO=${DOCKER_REPO}" \
  --platform "$PLATFORM" \
  -t "${IMAGE_NAME}:${TAG}" \
  .

docker tag "${IMAGE_NAME}:${TAG}" "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"

if [[ "$PUSH_IMAGE" == true ]]; then
  docker push "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"
fi
