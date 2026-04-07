#!/usr/bin/env bash

# Generate the Docker images for the StreamWise applications.

DEPLOYMENT_DIR=../
# shellcheck source=deployment/set_properties.sh.template
source "$DEPLOYMENT_DIR/set_properties.sh"
# shellcheck source=deployment/setup_lib.sh
source "$DEPLOYMENT_DIR/setup_lib.sh"

ensure_acr_login "$ACR_NAME"

for IMAGE in stream*; do
  (
    cd "$IMAGE" || exit
    if [ -f setup_image.sh ]; then
      # Copies files, builds the image, and tags it
      bash setup_image.sh
    fi
  )
done
