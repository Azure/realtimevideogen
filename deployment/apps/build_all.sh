#!/usr/bin/env bash

# Generate the Docker images for the StreamWise applications.

# shellcheck disable=SC1091
DEPLOYMENT_DIR=../
source "$DEPLOYMENT_DIR/set_properties.sh"

az acr login --name "$ACR_NAME"

for IMAGE in stream*; do
  (
    cd "$IMAGE" || exit
    if [ -f setup_image.sh ]; then
      # Copies files, builds the image, and tags it
      bash setup_image.sh
    fi
  )
done
