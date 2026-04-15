#!/usr/bin/env bash

bash ../run_img.sh \
    --path fluxkontext \
    --img ../../benchmark/samples/sample_256x192.png \
    --width 2048 \
    --height 1536 \
    "$@"
