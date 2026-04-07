@echo off
REM Micro-live daily routine for Windows
REM Usage: scripts\run_micro_live.bat

cd /d %~dp0\..
echo.
echo ==============================================
echo   MICRO-LIVE DAILY ROUTINE
echo   %date% %time%
echo ==============================================
echo.

python execution_cli.py run 1

echo.
echo ==============================================
echo   DONE. Next steps:
echo   1. If you confirmed orders, go place them on polymarket.com
echo   2. Record fills in micro_live_logs\
echo   3. After close: python live_observer.py status
echo   4. Tomorrow: scripts\run_micro_live.bat
echo ==============================================
