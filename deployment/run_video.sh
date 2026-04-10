#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="$SCRIPT_DIR/../benchmark/samples"

# Default arguments
HOST="localhost"
PORT="8080"
URL_PATH="wan"
PROMPT="A woman in the left and a man on the right speaking in a podcast."
NEG_PROMPT="Camera movement, color change"
WIDTH=1280
HEIGHT=720
STEPS=10
FRAMES=81
VIDEO_SECONDS=0
AUDIO_SECONDS=0
SEED=0
IMG_PATH="$SAMPLES_DIR/sample.png"
VIDEO_PATH=""
JSON_FILE="payload_tmp.json"

# Get the arguments:
# --host <host>
# --port <port>
# --width <width>
# --height <height>
# --steps <steps>
# --seed <seed>
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
        --neg_prompt)
            NEG_PROMPT="$2"
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
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --video_seconds)
            VIDEO_SECONDS="$2"
            shift 2
            ;;
        --audio_seconds)
            AUDIO_SECONDS="$2"
            shift 2
            ;;
        --img)
            IMG_PATH="$2"
            IMG_BASE64=$(base64 -w 0 "$IMG_PATH")
            shift 2
            ;;
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --json)
            JSON_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

generate_audio_base64() {
    SAMPLE_RATE=24000
    NUM_CHANNELS=1
    BYTES_PER_SAMPLE=2
    WAV_PATH="$SAMPLES_DIR/sample.wav"
    local seconds=$1
    local audio_bytes
    local audio_bytes_int
    # Calculate audio byte size
    audio_bytes=$(echo "$seconds * $SAMPLE_RATE * $NUM_CHANNELS * $BYTES_PER_SAMPLE" | bc)
    audio_bytes_int=$(echo "$audio_bytes / 1" | bc)  # truncate to integer
    # Extract and encode WAV segment (add 44-byte WAV header)
    dd if="$WAV_PATH" bs=1 count=$((audio_bytes_int + 44)) 2>/dev/null | base64 -w 0
}

# Image -> Video
IMG_BASE64=""
if [ -f "$IMG_PATH" ]; then
    IMG_BASE64=$(base64 -w 0 "$IMG_PATH")
fi
VIDEO_BASE64=""
if [ -f "$VIDEO_PATH" ]; then
    VIDEO_BASE64=$(base64 -w 0 "$VIDEO_PATH")
fi
AUDIO_BASE64=""
if awk "BEGIN { exit !(${AUDIO_SECONDS} > 0) }"; then
    AUDIO_BASE64=$(generate_audio_base64 "$AUDIO_SECONDS")
fi

# To access the REST API:
JOB_ID="video_$(date +%Y%m%dT%H%M%S.%3N)_${WIDTH}_${HEIGHT}_${STEPS}_${FRAMES}_${VIDEO_SECONDS}"
if [ "$JSON_FILE" = "payload_tmp.json" ]; then
cat > "$JSON_FILE" <<EOF
{
    "job_id": "$JOB_ID",
    "img": "$IMG_BASE64",
    "video": "$VIDEO_BASE64",
    "audio": "$AUDIO_BASE64",
    "prompt": "$PROMPT",
    "neg_prompt": "$NEG_PROMPT",
    "num_frames": $FRAMES,
    "video_seconds": $VIDEO_SECONDS,
    "width": $WIDTH,
    "height": $HEIGHT,
    "sampling_steps": $STEPS,
    "seed": $SEED
}
EOF
fi
if [ ! -f "$JSON_FILE" ]; then
    echo "JSON file not found: $JSON_FILE"
    exit 1
fi

# Video generation
URL="http://$HOST:$PORT/$URL_PATH"
mkdir -p output
OUTPUT_FILE="output/test_video_${URL_PATH}_${WIDTH}_${HEIGHT}_${STEPS}.mp4"
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
    echo "Video generated successfully:"
    du -h "$OUTPUT_FILE"
    file "$OUTPUT_FILE"
    if command -v mediainfo > /dev/null 2>&1; then
        mediainfo "$OUTPUT_FILE"
    else
        echo "mediainfo not found: sudo apt install mediainfo."
    fi
fi

curl -s "$URL"/health | jq .
