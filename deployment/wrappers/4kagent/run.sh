#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_img.sh" \
  --path 4kagent \
  --img "$SCRIPT_DIR/../../../benchmark/samples/sample_256x192.png" \
  "$@"
