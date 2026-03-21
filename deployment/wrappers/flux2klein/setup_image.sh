#!/usr/bin/env bash

IMAGE_NAME="flux2klein"

# Copy transformer_flux2.py from the flux2 wrapper (shared, not duplicated)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p docker_files
cp "$SCRIPT_DIR/../../../wrapper/flux2/transformer_flux2.py" docker_files/

bash ../setup_image.sh \
  "$IMAGE_NAME" \
  --hf_token \
  --parent "flux" \
  "$@"
