@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -s -B run_ui.py
) else (
  echo Create the project environment and install requirements_windows.txt first.
  echo py -m venv .venv
  echo .venv\Scripts\python.exe -m pip install -r requirements_windows.txt
  pause
  exit /b 1
)
