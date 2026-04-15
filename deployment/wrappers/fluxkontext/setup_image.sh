#!/usr/bin/env bash

IMAGE_NAME="fluxkontext"

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  --hf_token \
  --parent "flux" \
  "$@"
