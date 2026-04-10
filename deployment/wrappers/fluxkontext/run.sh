#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_img.sh" \
    --path fluxkontext \
    --img "$SCRIPT_DIR/../../../benchmark/samples/sample_256x192.png" \
    --width 2048 \
    --height 1536 \
    "$@"
