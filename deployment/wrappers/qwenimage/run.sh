#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../.."

# git pull && bash setup_image.sh
# docker stop qwenimage_rest
# docker rm qwenimage_rest; docker run -d --gpus '"device=0,1"' -p 18082:8080 --name qwenimage_rest qwenimage

bash "$DEPLOY_DIR/run_img.sh" \
    "$@" \
    --path qwenimage
