#!/usr/bin/env bash

bash ../run_video.sh \
    --path hunyuanavatar \
    --audio_seconds 2.4 \
    --img ../../benchmark/samples/single_character_sample.png \
    --width 640 \
    --height 480 \
    --steps 10 \
    "$@"

bash ../run_video.sh \
    --path hunyuanavatar \
    --audio_seconds 2.4 \
    --video ../../benchmark/samples/sample_320x240.mp4 \
    --width 640 \
    --height 480 \
    --steps 10 \
    "$@"
