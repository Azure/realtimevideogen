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

/opt/conda/bin/conda run -n streamwise \
python3 streamwise.py \
"$@"
