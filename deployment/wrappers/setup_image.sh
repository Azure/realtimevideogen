#!/usr/bin/env bash
set -euo pipefail

# Argument parsing
usage() {
  echo "Usage: $0 <IMAGE_NAME> [--hf_token] [--push] [--folders folder1,folder2,...] [--platform linux/amd64|linux/arm64]"
  exit 1
}

# shellcheck source=../setup_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../setup_lib.sh"

IMAGE_NAME="${1:-}"
[[ -z "$IMAGE_NAME" ]] && usage

WRAPPER_SUBFOLDERS=()
PARENT_MODEL=""
USE_HF_TOKEN=false
PUSH_IMAGE=false

PLATFORM=$(detect_platform)

shift || true

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf_token)
      USE_HF_TOKEN=true
      shift
      ;;
    --push)
      PUSH_IMAGE=true
      shift
      ;;
    --folders)
      shift
      IFS=',' read -r -a WRAPPER_SUBFOLDERS <<< "$1"
      shift
      ;;
    --parent)
      shift
      PARENT_MODEL="$1"
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
MAIN_DIR=$(realpath ../../..)
DEPLOYMENT_DIR=$MAIN_DIR/deployment
WRAPPERS_DIR=$MAIN_DIR/wrapper
WRAPPER_DIR=$WRAPPERS_DIR/$IMAGE_NAME

# Source set_properties.sh only if DOCKER_REPO is not already provided in the environment
if [[ -z "${DOCKER_REPO:-}" ]]; then
  if [[ -f "$DEPLOYMENT_DIR/set_properties.sh" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$DEPLOYMENT_DIR/set_properties.sh"
  else
    echo "ERROR: DOCKER_REPO is not set and $DEPLOYMENT_DIR/set_properties.sh does not exist"
    exit 1
  fi
fi

TAG=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")

mkdir -p ./docker_files

# Copy parent model files if specified
if [[ -n "$PARENT_MODEL" ]]; then
  PARENT_DIR="$WRAPPERS_DIR/$PARENT_MODEL"
  cp "$PARENT_DIR"/*.py ./docker_files/
  cp "$PARENT_DIR"/requirements.txt ./docker_files/
fi

# Copy necessary files to the current directory
cp "$MAIN_DIR"/requirements.txt ./docker_files/base_requirements.txt
cp "$MAIN_DIR"/services.json ./docker_files/
cp "$MAIN_DIR"/*.py ./docker_files/

cp -R "$WRAPPERS_DIR"/static ./docker_files/
cp -R "$WRAPPERS_DIR"/templates ./docker_files/

cp "$WRAPPERS_DIR"/*.bash ./docker_files/
cp "$WRAPPERS_DIR"/*.py ./docker_files/

# Copy wrapper-specific files if they exist (not all images have their own Python files)
if compgen -G "$WRAPPER_DIR/*.py" > /dev/null 2>&1; then
  cp "$WRAPPER_DIR"/*.py ./docker_files/
fi
if [[ -f "$WRAPPER_DIR/requirements.txt" ]]; then
  cp "$WRAPPER_DIR"/requirements.txt ./docker_files/
fi

# Copy wrapper subfolders to docker_files
for subfolder in "${WRAPPER_SUBFOLDERS[@]}"; do
  SRC_DIR="$WRAPPER_DIR/$subfolder"
  DEST_DIR="./docker_files/$subfolder"

  if [[ -d "$SRC_DIR" ]]; then
    mkdir -p "$DEST_DIR"
    cp -R "$SRC_DIR"/* "$DEST_DIR"/
  else
    echo "WARNING: Subfolder '$SRC_DIR' does not exist, skipping"
  fi
done

# Construct docker build command with optional token
BASE_TAG=$(jq -r '.base.dockerImage.tag' "$MAIN_DIR/services.json")

BUILD_ARGS=(
  docker buildx build
  --build-arg "DOCKER_REPO=${DOCKER_REPO}"
  --build-arg "BASE_TAG=${BASE_TAG}"
  --platform "$PLATFORM"
  -t "${IMAGE_NAME}:${TAG}"
)

if [[ "$USE_HF_TOKEN" == true ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: --hf_token specified but HF_TOKEN is not set"
    exit 1
  fi
  HF_TOKEN_FILE="docker_files/hf_token.txt"
  echo "$HF_TOKEN" > "$HF_TOKEN_FILE"
  BUILD_ARGS+=(
    --secret "id=hf_token,src=$HF_TOKEN_FILE"
  )
fi

if [[ "$PUSH_IMAGE" == true ]]; then
  ensure_acr_login "$DOCKER_REPO"
fi

# Build
"${BUILD_ARGS[@]}" .

# Cleanup secret
if [[ "$USE_HF_TOKEN" == true && -f "$HF_TOKEN_FILE" ]]; then
  rm -f "$HF_TOKEN_FILE"
fi

# Tag final image for pushing
docker tag "${IMAGE_NAME}:${TAG}" "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"

if [[ "$PUSH_IMAGE" == true ]]; then
  docker push "${DOCKER_REPO}/${IMAGE_NAME}:${TAG}"
fi
