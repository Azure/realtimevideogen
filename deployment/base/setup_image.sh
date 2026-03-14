#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="base"

bash ../setup_image.sh "$IMAGE_NAME" "$@"
