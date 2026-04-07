@echo off
echo ==========================================
echo   Polymarket Auto Trader - Starting
echo ==========================================

REM Set your private key
set POLYMARKET_PRIVATE_KEY=0xc41ec26f11c9c1ec8fadd73ae990386ab386585dcad1ab89ff0d2a3e41887970

REM Install dependencies (first time only)
pip install py-clob-client requests python-dotenv >nul 2>&1

echo.
echo [1/2] Starting Market Maker (spread capture)...
echo      Earns bid-ask spread, no directional risk
echo.
start "Market Maker" cmd /k "python market_maker.py --auto --spread 0.02 --size 5"

echo [2/2] Starting Signal Scanner (edge finder)...
echo      Finds mispriced games, places directional bets
echo.
timeout /t 5 >nul
start "Signal Scanner" cmd /k "python run_auto.py --loop --interval 300"

echo.
echo Both bots running in separate windows.
echo.
echo Commands:
echo   python execution_cli.py portfolio   - View portfolio
echo   python market_maker.py --dry --auto - Test market maker
echo   python run_auto.py --scan           - Scan for edge
echo.
pause
