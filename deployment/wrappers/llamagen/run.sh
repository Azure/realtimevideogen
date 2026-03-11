#!/usr/bin/env bash

bash ../run_img.sh \
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
