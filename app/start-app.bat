@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Python environment .venv was not found.
  echo Create it using the project setup instructions first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app\server.py --open
if errorlevel 1 pause
