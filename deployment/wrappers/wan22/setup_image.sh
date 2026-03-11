#!/usr/bin/env bash

IMAGE_NAME="wan22"

# TODO use generic setup_image.sh
# bash ../setup_image.sh "$IMAGE_NAME" "$@"

# Argument parsing
PUSH_IMAGE=false

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
WRAPPERS_DIR=$MAIN_DIR/wrapper
WRAPPER_DIR=$WRAPPERS_DIR/$IMAGE_NAME

# shellcheck disable=SC1090,SC1091 
source "$DEPLOYMENT_DIR/set_properties.sh"

REPOSITORY=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.repository' "$MAIN_DIR/services.json")
TAG=$(jq -r --arg name "$IMAGE_NAME" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")

mkdir -p ./docker_files

# Copy necessary files to the current directory
cp "$MAIN_DIR"/requirements.txt ./docker_files/base_requirements.txt
cp "$MAIN_DIR"/services.json ./docker_files/
cp "$MAIN_DIR"/*.py ./docker_files/

cp -R "$WRAPPERS_DIR"/static ./docker_files/
cp -R "$WRAPPERS_DIR"/templates ./docker_files/

cp "$WRAPPERS_DIR"/*.bash ./docker_files/
cp "$WRAPPERS_DIR"/*.py ./docker_files/

cp "$WRAPPER_DIR"/*.py ./docker_files/
cp "$WRAPPER_DIR"/requirements.txt ./docker_files/

# TODO this is the blocker to move to the generic wrapper setup_image.sh
cp "$WRAPPERS_DIR"/wan/*.py ./docker_files/
cp "$WRAPPERS_DIR"/wan22/wrapper_wan22.py ./docker_files/

docker buildx build \
  --build-arg "DOCKER_REPO=${REPOSITORY}" \
  -t "${IMAGE_NAME}:${TAG}" \
  .

docker tag "${IMAGE_NAME}:${TAG}" "${REPOSITORY}/${IMAGE_NAME}:${TAG}"

if [[ "$PUSH_IMAGE" == true ]]; then
  docker push "${REPOSITORY}/${IMAGE_NAME}:${TAG}"
fi
