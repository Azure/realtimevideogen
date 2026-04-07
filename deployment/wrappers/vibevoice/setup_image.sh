#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="vibevoice"

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  --hf_token \
  --folders schedule,voices \
  "$@"
