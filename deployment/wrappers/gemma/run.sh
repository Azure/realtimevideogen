#!/usr/bin/env bash

# Default arguments
HOST=localhost
PORT=18086
PROMPT="Explain this paper in detail:"
PAPER_URL="https://arxiv.org/pdf/2501.02600" # TAPAS
PAPER_URL="https://arxiv.org/pdf/2408.00741" # DynamoLLM
PAPER_URL="https://arxiv.org/pdf/2502.00937" # ModeServe
PAPER_URL="https://arxiv.org/pdf/2501.16634" # Murakhab
PAPER_URL="https://arxiv.org/pdf/2501.11179" # Coach
PAPER_URL="https://arxiv.org/pdf/2409.17264" # Medha
PAPER_URL="https://arxiv.org/pdf/2403.03377" # Junctiond
PAPER_URL="https://arxiv.org/pdf/2311.18677" # SplitWise
PAPER_URL="https://arxiv.org/pdf/2308.12908" # POLCA
PAPER_URL="https://arxiv.org/pdf/2104.13869" # Faa$T
PAPER_URL="all"
MAX_OUT_TOKENS=$(( 48 * 1024 ))
JSON_FILE="payload_gemma.json"

# How much of the input we handle
MAX_INPUT_CHARS=$(( 32 * 1024 ))
MAX_INPUT_CHARS=$(( 64 * 1024 ))
MAX_INPUT_CHARS=$(( 96 * 1024 ))
MAX_INPUT_CHARS=$(( 128 * 1024 ))
MAX_INPUT_CHARS=$(( 256 * 1024 ))

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
        --paper_url)
            PAPER_URL="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --max_out_tokens)
            MAX_OUT_TOKENS="$2"
            shift 2
            ;;
        --max_input_chars)
            MAX_INPUT_CHARS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# sudo apt install poppler-utils
if [[ $PAPER_URL == "all" ]]; then
  RAW_TEXT=$(
    for url in \
      https://arxiv.org/pdf/2408.00741 \
      https://arxiv.org/pdf/2501.16634 \
      https://arxiv.org/pdf/2104.13869 \
      https://arxiv.org/pdf/2311.18677 \
      https://arxiv.org/pdf/2403.03377 \
      https://arxiv.org/pdf/2501.11179 \
      https://arxiv.org/pdf/2308.12908
    do
      wget -qO- "$url" | pdftotext -enc ASCII7 - -
    done
  )
else
  RAW_TEXT=$(wget -qO- "$PAPER_URL" | pdftotext -enc ASCII7 - -)
fi

RAW_TEXT=$(echo "$RAW_TEXT" | head -c "$MAX_INPUT_CHARS")
NUM_CHARS=${#RAW_TEXT}
echo "Trimmed text contains $NUM_CHARS characters."

PROMPT="Explain these papers: $RAW_TEXT <PAPERS_END> Now start explaining it with 10,000 words:"
cat > "$JSON_FILE" <<EOF
{
  "model": "google/gemma-3-27b-it",
  "max_tokens": $MAX_OUT_TOKENS,
  "temperature": 0,
  "prompt": $(jq -Rs '.' <<<"$PROMPT")
}
EOF


# Number of words in the prompt
NUM_WORDS=$(jq .prompt "$JSON_FILE" | wc -w)
NUM_CHARS=$(jq .prompt "$JSON_FILE" | wc -c)
echo "Prompt contains $NUM_WORDS words and $NUM_CHARS characters."

# Execute the request
curl -s "http://$HOST:$PORT/v1/completions" \
-H "Content-Type: application/json" \
-d @$JSON_FILE | jq .
#| jq .choices[0].text

# Check metrics
#curl -s http://$HOST:$PORT/metrics
