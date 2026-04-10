#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_img.sh" \
  --path realesrgan \
  --img "$SCRIPT_DIR/../../../benchmark/samples/sample_256x192.png" \
  "$@"

bash "$SCRIPT_DIR/../../run_video.sh" \
  --path realesrgan \
  --video "$SCRIPT_DIR/../../../benchmark/samples/sample_320x240.mp4" \
  "$@"
