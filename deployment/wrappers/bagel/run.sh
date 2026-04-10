#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_img.sh" \
  --path bagel \
  --img "$SCRIPT_DIR/../../../benchmark/samples/sample.png" \
  "$@"
