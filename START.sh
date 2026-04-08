#!/bin/bash
echo "=========================================="
echo "  Polymarket Auto Trader - Starting"
echo "=========================================="

# Load private key from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$POLYMARKET_PRIVATE_KEY" ]; then
    echo "ERROR: POLYMARKET_PRIVATE_KEY not set!"
    echo "Create a .env file with:"
    echo "  POLYMARKET_PRIVATE_KEY=0xyour_key_here"
    exit 1
fi

echo "  Wallet loaded: ${POLYMARKET_PRIVATE_KEY:0:6}...${POLYMARKET_PRIVATE_KEY: -4}"

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
