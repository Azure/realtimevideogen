#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# git pull && bash setup_image.sh
# docker stop thinksound_rest; docker rm thinksound_rest
# docker run -d --gpus '"device=0,1"' -p 18082:8080 --name thinksound_rest thinksound

bash "$SCRIPT_DIR/../../run_audio.sh" \
    --path thinksound \
    "$@"
