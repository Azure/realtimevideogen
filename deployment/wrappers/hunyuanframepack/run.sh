#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_video.sh" \
  --path hunyuanframepack \
  --width 768 \
  --height 512 \
  --img "$SCRIPT_DIR/../../../benchmark/samples/sample.png" \
  "$@"
