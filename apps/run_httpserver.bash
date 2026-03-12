#!/usr/bin/env bash

echo "Storage:"
df -h

echo ""
echo "Memory:"
free -h

echo "CPU:"
lscpu

echo ""
echo "Environment variables:"
printenv

echo "FFmpeg:"
ffmpeg -version
dpkg -s ffmpeg

APP_NAME="$1"
shift

/opt/conda/bin/conda run -n streamwise \
python3 -u "$APP_NAME.py" \
"$@"
