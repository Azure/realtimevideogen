#!/usr/bin/env bash

IMAGE_NAME="flux2klein"

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  --hf_token \
  --parent "flux" \
  "$@"
