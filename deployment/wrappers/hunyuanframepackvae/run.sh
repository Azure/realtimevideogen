#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default arguments
HOST="localhost"
PORT="8080"

# Get the arguments:
# --host <host> --port <port> --width <width> --height <height> --steps <steps>
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TENSOR="$SCRIPT_DIR/../../../benchmark/samples/sample_latents.pt"
#TENSOR="$SCRIPT_DIR/../../../benchmark/samples/sample_latents_long.pt"


mkdir -p output

# Uncompressed
JOB_ID="testuncompressedjobid"
URL="http://$HOST:$PORT/hunyuanframepack/vae/$JOB_ID"
OUTPUT_FILE="output/test_hunyuanframepackvae_uncompressed.mp4"
http_code=$(curl \
    -s \
    -w "%{http_code}" \
    -X POST \
    "$URL" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$TENSOR" \
    -o "$OUTPUT_FILE")
if [ "$http_code" -ne 200 ]; then
    echo "Request failed: $http_code"
    jq . "$OUTPUT_FILE"
    exit 1
else
    echo "Video generated successfully:"
    du -h "$OUTPUT_FILE"
    file "$OUTPUT_FILE"
    if command -v mediainfo > /dev/null 2>&1; then
        mediainfo "$OUTPUT_FILE"
    else
        echo "mediainfo not found: sudo apt install mediainfo."
    fi
fi

# Compressed
JOB_ID="testcompressedjobid"
URL="http://$HOST:$PORT/hunyuanframepack/vae/$JOB_ID"
OUTPUT_FILE="output/test_hunyuanframepackvae_compressed.mp4"
http_code=$(gzip -c "$TENSOR" | curl \
    -s \
    -w "%{http_code}" \
    -X POST \
    "$URL" \
    -H "Content-Type: application/octet-stream" \
    -H "Content-Encoding: gzip" \
    --data-binary @- \
    -o "$OUTPUT_FILE")
if [ "$http_code" -ne 200 ]; then
    echo "Request failed: $http_code"
    jq . "$OUTPUT_FILE"
    exit 1
else
    echo "Video generated successfully:"
    du -h "$OUTPUT_FILE"
    file "$OUTPUT_FILE"
    if command -v mediainfo > /dev/null 2>&1; then
        mediainfo "$OUTPUT_FILE"
    else
        echo "mediainfo not found: sudo apt install mediainfo."
    fi
fi

curl -s "http://$HOST:$PORT/health" | jq .
