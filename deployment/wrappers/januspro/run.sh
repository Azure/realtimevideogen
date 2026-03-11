#!/usr/bin/env bash

bash ../run_img.sh \
  --path januspro \
  "$@"

# To access the REST API:
cat > payload_januspro.json <<EOF
{
    "prompt": "A woman in the left and a man on the right speaking in a podcast.",
    "height": 720,
    "width": 1280,
    "sampling_steps": 10
}
EOF
