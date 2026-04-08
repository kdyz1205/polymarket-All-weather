#!/bin/bash
echo "=========================================="
echo "  Polymarket Auto Trader - Starting"
echo "=========================================="

# Load from .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$POLYMARKET_PRIVATE_KEY" ]; then
    echo "ERROR: POLYMARKET_PRIVATE_KEY not set."
    echo "Create a .env file with: POLYMARKET_PRIVATE_KEY=0x..."
    exit 1
fi

# Install deps
pip install py-clob-client requests python-dotenv -q 2>/dev/null

echo ""
echo "[1/2] Starting Market Maker (background)..."
python market_maker.py --auto --spread 0.02 --size 5 &
MM_PID=$!

sleep 3

echo "[2/2] Starting Signal Scanner (foreground)..."
echo "      Press Ctrl+C to stop both."
echo ""

trap "kill $MM_PID 2>/dev/null; echo 'Stopped.'; exit" INT TERM
python run_auto.py --loop --interval 300

kill $MM_PID 2>/dev/null
