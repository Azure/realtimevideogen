#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../.."

bash "$DEPLOY_DIR/run_img.sh" \
  --path llamagen \
  "$@"

# To access the REST API:
cat > payload_llamagen.json <<EOF
{
    "prompt": "A woman in the left and a man on the right speaking in a podcast.",
    "image_size": 512,
    "seed": 23
}
EOF
