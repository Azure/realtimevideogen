#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../.."
SAMPLES_DIR="$SCRIPT_DIR/../../../benchmark/samples"

bash "$DEPLOY_DIR/run_video.sh" \
    --path fantasytalking \
    --audio_seconds 2.4 \
    --img "$SAMPLES_DIR/single_character_sample.png" \
    --width 640 \
    --height 480 \
    --steps 10 \
    "$@"

bash "$DEPLOY_DIR/run_video.sh" \
    --path fantasytalking \
    --audio_seconds 2.4 \
    --video "$SAMPLES_DIR/sample_320x240.mp4" \
    --width 640 \
    --height 480 \
    --steps 10 \
    "$@"
