#!/bin/bash
# FaceLocker — 双击启动

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "Starting FaceLocker..."
echo ""

# Use venv python directly (bypasses shell activation issues)
"$DIR/venv/bin/python" -u "$DIR/main.py"
EXIT_CODE=$?

echo ""
echo "FaceLocker exited with code: $EXIT_CODE"
echo "Press any key to close this window..."
read -r
