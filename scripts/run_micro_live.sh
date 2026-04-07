#!/usr/bin/env bash
#
# Micro-live daily routine — run this on a machine with Polymarket API access.
#
# Prerequisites:
#   1. Python venv with all deps: source .venv/bin/activate
#   2. At least one market mapping in config/market_mappings.json
#      (run: python execution_cli.py add-market)
#   3. Network access to clob.polymarket.com
#
# Usage:
#   ./scripts/run_micro_live.sh
#
# What it does:
#   Step 1: Paper sanity check (5 games)
#   Step 2: Generate candidate signals (1 game sim)
#   Step 3: Fetch real Polymarket prices & deviation check
#   Step 4: Interactive order review (confirm/reject each)
#   Step 5: Print daily status
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=============================================="
echo "  MICRO-LIVE DAILY ROUTINE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: python -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# Check network access to Polymarket
echo "  Checking Polymarket API access..."
if curl -s --max-time 5 -o /dev/null -w "%{http_code}" https://clob.polymarket.com/markets | grep -q "200\|404"; then
    echo "  ✓ Polymarket API reachable"
else
    echo "  ✗ Cannot reach Polymarket API"
    echo "  Running in offline mode (no real price comparison)"
fi

echo ""

# Run the full pipeline
python execution_cli.py run 1

echo ""
echo "=============================================="
echo "  DONE. Next steps:"
echo "  1. If you confirmed orders above, go place them on polymarket.com"
echo "  2. Record fills in micro_live_logs/"
echo "  3. After market close: python live_observer.py status"
echo "  4. Tomorrow: ./scripts/run_micro_live.sh"
echo "=============================================="
