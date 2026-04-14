#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../.."

bash "$DEPLOY_DIR/run_img.sh" \
    --path cogview \
    "$@"

# To access the REST API:
cat > payload_cogview.json <<EOF
{
    "prompt": "A vibrant cherry red sports car sits proudly under the gleaming sun.",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "sampling_steps": 50
}
EOF
