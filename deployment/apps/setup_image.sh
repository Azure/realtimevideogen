#!/usr/bin/env bash
set -euo pipefail

# Argument parsing
usage() {
  echo "Usage: $0 <IMAGE_NAME> [--push]"
  exit 1
}

# Shared utilities
# shellcheck source=deployment/setup_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../setup_lib.sh"

IMAGE_NAME="${1:-}"
[[ -z "$IMAGE_NAME" ]] && usage

PUSH_IMAGE=false

PLATFORM=$(detect_platform)

shift || true

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH_IMAGE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Main script
MAIN_DIR=$(realpath ../../..)
DEPLOYMENT_DIR=$MAIN_DIR/deployment
APPS_DIR=$MAIN_DIR/apps
APP_DIR=$APPS_DIR/$IMAGE_NAME

# Source set_properties.sh only if DOCKER_REPO is not already provided in the environment
if [[ -z "${DOCKER_REPO:-}" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "$DEPLOYMENT_DIR/set_properties.sh"
fi

TAG=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")

mkdir -p docker_files

# Copy necessary files from the app directory to the current directory
cp "$DEPLOYMENT_DIR/apps/Dockerfile" ./docker_files/

cp "$APPS_DIR/requirements.txt" ./docker_files/requirements_streamwise.txt
cp "$APPS_DIR/$IMAGE_NAME/requirements.txt" ./docker_files/requirements_"$IMAGE_NAME".txt

cp "$MAIN_DIR"/*.py ./docker_files/
cp "$APPS_DIR"/*.py ./docker_files/
cp "$APPS_DIR"/*.bash ./docker_files/

cp "$APP_DIR"/*.py ./docker_files/

cp -R "$APPS_DIR"/static ./docker_files/
cp -R "$APPS_DIR"/templates ./docker_files/
cp -R "$APP_DIR"/templates/* ./docker_files/templates/

cp "$MAIN_DIR/services.json" ./docker_files/

if [[ "$PUSH_IMAGE" == true ]]; then
  ensure_acr_login "$DOCKER_REPO"
fi

BASE_TAG=$(jq -r '.base.dockerImage.tag' "$MAIN_DIR/services.json")

# Build
docker buildx build \
  --load \
  --build-arg DOCKER_REPO="${DOCKER_REPO}" \
  --build-arg APP_NAME="${IMAGE_NAME}" \
  --build-arg BASE_TAG="${BASE_TAG}" \
  --platform "${PLATFORM}" \
  -t "${IMAGE_NAME}:${TAG}" \
  ./docker_files/

# Tag final image for pushing
docker tag "${IMAGE_NAME}:${TAG}" "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"

if [[ "$PUSH_IMAGE" == true ]]; then
  docker push "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"
fi
