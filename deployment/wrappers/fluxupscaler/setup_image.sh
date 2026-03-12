#!/usr/bin/env bash

IMAGE_NAME="fluxupscaler"

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  --hf_token \
  "$@"
