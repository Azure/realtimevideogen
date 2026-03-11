#!/usr/bin/env bash

# Default arguments
HOST="localhost"
PORT="8080"
URL_PATH="flux"
PROMPT="A woman in the left and a man on the right speaking in a podcast."
WIDTH=1280
HEIGHT=800
STEPS=10
SEED=0
JSON_FILE="payload_tmp.json"
IMG_BASE64=""

# Get the arguments:
# --host <host> --port <port> --width <width> --height <height> --steps <steps> --seed <seed>
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --path)
            URL_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --json)
            JSON_FILE="$2"
            shift 2
            ;;
        --img)
            IMG_FILE="$2"
            IMG_BASE64=$(base64 -w 0 "$IMG_FILE")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# To access the REST API:
if [ "$JSON_FILE" = "payload_tmp.json" ]; then
cat > "$JSON_FILE" <<EOF
{
    "prompt": "$PROMPT",
    "width": $WIDTH,
    "height": $HEIGHT,
    "sampling_steps": $STEPS,
    "seed": $SEED,
    "img": "$IMG_BASE64"
}
EOF
fi
if [ ! -f "$JSON_FILE" ]; then
    echo "JSON file not found: $JSON_FILE"
    exit 1
fi

# Image generation
URL="http://$HOST:$PORT/$URL_PATH"
mkdir -p output
OUTPUT_FILE="output/test_img_${URL_PATH}_${WIDTH}_${HEIGHT}_${STEPS}.png"
http_code=$(curl \
    -s \
    -w "%{http_code}" \
    -X POST \
    "$URL" \
    -H "Content-Type: application/json" \
    -d @"$JSON_FILE" \
    -o "$OUTPUT_FILE")

if [ "$http_code" -ne 200 ]; then
    echo "Request to $URL failed: $http_code"
    if [ -f "$OUTPUT_FILE" ]; then
        jq . "$OUTPUT_FILE"
    fi
    exit 1
else
    echo "Image generated successfully:"
    du -h "$OUTPUT_FILE"
    file "$OUTPUT_FILE"
    if command -v mediainfo > /dev/null 2>&1; then
        mediainfo "$OUTPUT_FILE"
    else
        echo "mediainfo not found: sudo apt install mediainfo."
    fi
fi

curl -s "$URL"/health | jq .
