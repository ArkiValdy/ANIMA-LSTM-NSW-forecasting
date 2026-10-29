@echo off
rem Daily 21:00 job: forecast tomorrow's NSW price, then update outputs\live_predicted.csv
rem with the new predictions and the actual prices for earlier forecast days.
rem Output goes to outputs\logs\daily_forecast.log.

cd /d "%~dp0"
if not exist outputs\logs mkdir outputs\logs
set LOG=outputs\logs\daily_forecast.log
set TF_CPP_MIN_LOG_LEVEL=2

echo ===== %date% %time% ===== >> "%LOG%"
.venv\Scripts\python.exe dashboard.py --refresh >> "%LOG%" 2>&1
if errorlevel 1 echo [bat] dashboard.py failed - recording actuals only >> "%LOG%"
.venv\Scripts\python.exe record_live.py >> "%LOG%" 2>&1
echo ===== done %date% %time% ===== >> "%LOG%"
