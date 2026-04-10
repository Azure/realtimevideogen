#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPT="[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on GitHub or Hugging Face. [S3] That is great."

bash "$SCRIPT_DIR/../../run_audio.sh" \
  --path kokoro \
  --text  "$PROMPT" \
  "$@"
