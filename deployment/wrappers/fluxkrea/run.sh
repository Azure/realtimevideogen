#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/../.."

bash "$DEPLOY_DIR/run_img.sh" \
    "$@" \
    --path fluxkrea
