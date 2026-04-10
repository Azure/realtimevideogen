#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/../../run_video.sh" \
    --path hunyuanavatar \
    --audio_seconds 2.4 \
    --img "$SCRIPT_DIR/../../../benchmark/samples/single_character_sample.png" \
    --width 640 \
    --height 480 \
    --steps 10 \
    "$@"

bash "$SCRIPT_DIR/../../run_video.sh" \
    --path hunyuanavatar \
    --audio_seconds 2.4 \
    --video "$SCRIPT_DIR/../../../benchmark/samples/sample_320x240.mp4" \
    --width 640 \
    --height 480 \
    --steps 10 \
    "$@"
