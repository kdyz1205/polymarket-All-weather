#!/bin/bash
echo "=========================================="
echo "  Polymarket Auto Trader - Starting"
echo "=========================================="

export POLYMARKET_PRIVATE_KEY="0xc41ec26f11c9c1ec8fadd73ae990386ab386585dcad1ab89ff0d2a3e41887970"

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
