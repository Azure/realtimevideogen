#!/usr/bin/env bash
set -euo pipefail

# Argument parsing
usage() {
  echo "Usage: $0 <IMAGE_NAME> [--push] [--certfile <path>] [--keyfile <path>]"
  exit 1
}

# Shared utilities
# shellcheck source=deployment/setup_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../setup_lib.sh"

IMAGE_NAME="${1:-}"
[[ -z "$IMAGE_NAME" ]] && usage

PUSH_IMAGE=false
CERT_FILE=""
KEY_FILE=""

PLATFORM=$(detect_platform)

shift || true

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH_IMAGE=true
      shift
      ;;
    --certfile)
      shift
      CERT_FILE="$1"
      shift
      ;;
    --keyfile)
      shift
      KEY_FILE="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate cert flags: both must be provided together
if [[ -n "$CERT_FILE" ]] && [[ -z "$KEY_FILE" ]]; then
  echo "ERROR: --certfile requires --keyfile to be specified as well"
  exit 1
fi
if [[ -z "$CERT_FILE" ]] && [[ -n "$KEY_FILE" ]]; then
  echo "ERROR: --keyfile requires --certfile to be specified as well"
  exit 1
fi

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

# Certs directory (empty by default; populated with --certfile/--keyfile for embedded HTTPS)
mkdir -p docker_files/certs
if [[ -n "$CERT_FILE" ]] && [[ -n "$KEY_FILE" ]]; then
  [[ -f "$CERT_FILE" ]] || { echo "ERROR: Certificate file not found: $CERT_FILE"; exit 1; }
  [[ -f "$KEY_FILE" ]] || { echo "ERROR: Key file not found: $KEY_FILE"; exit 1; }
  cp "$CERT_FILE" docker_files/certs/cert.pem
  cp "$KEY_FILE" docker_files/certs/key.pem
  echo "Embedding TLS certificate: $CERT_FILE"
fi

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
