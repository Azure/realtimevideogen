#!/usr/bin/env bash

IMAGE_NAME="kokoro"

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  "$@"
