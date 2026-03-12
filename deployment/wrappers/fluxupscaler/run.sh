#!/usr/bin/env bash

bash ../run_img.sh \
  --path fluxupscaler \
  --img ../../benchmark/samples/sample_256x192.png
  "$@"

bash ../run_video.sh \
  --path fluxupscaler \
  --video ../../benchmark/samples/sample_320x240.mp4
  "$@"
