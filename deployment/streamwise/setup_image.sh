#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="streamwise"

bash "$(dirname "$0")/../setup_image.sh" "$IMAGE_NAME" "$@"
