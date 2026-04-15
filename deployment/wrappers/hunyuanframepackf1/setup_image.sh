#!/usr/bin/env bash

IMAGE_NAME="hunyuanframepackf1"

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  --parent "hunyuanframepack" \
  "$@"
