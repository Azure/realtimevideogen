#!/usr/bin/env bash

LLM_IP="10.244.22.14"
LLM_URL="http://$LLM_IP:8000/v1"

cat > payload_transcript.json <<EOF
{
    "llm_url": "$LLM_URL",
    "llm_model": "google/gemma-3-27b-it",
    "multi_modal": true,
    "pdf_url": "https://arxiv.org/pdf/2309.17030.pdf",
    "max_dialogues": 4
}
EOF

HOST="localhost"
PORT="8080"
if [ -n "$1" ]; then
    HOST=$1
fi
if [ -n "$2" ]; then
    PORT=$2
fi

# Streaming
URL="http://$HOST:$PORT/podcasttranscript/stream"
curl -s -X POST -H "Content-Type: application/json" -d @payload_transcript.json "$URL"

# Single request
URL="http://$HOST:$PORT/podcasttranscript"
curl -s -X POST -H "Content-Type: application/json" -d @payload_transcript.json "$URL"

curl -s "http://$HOST:$PORT/podcasttranscript/health" | jq .