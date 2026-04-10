#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_video.sh" \
  --path ltx \
  --img "$SCRIPT_DIR/../../../benchmark/samples/sample.png" \
  "$@"
