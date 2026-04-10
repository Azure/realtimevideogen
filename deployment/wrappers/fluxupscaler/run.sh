#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../.."
SAMPLES_DIR="$SCRIPT_DIR/../../../benchmark/samples"

bash "$DEPLOY_DIR/run_img.sh" \
  --path fluxupscaler \
  --img "$SAMPLES_DIR/sample_256x192.png" \
  "$@"

bash "$DEPLOY_DIR/run_video.sh" \
  --path fluxupscaler \
  --video "$SAMPLES_DIR/sample_320x240.mp4" \
  "$@"
