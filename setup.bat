@echo off
REM One-time setup (Windows): creates .venv and installs requirements.txt into it.
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv || (echo Python 3 not found & exit /b 1) )
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Done.  Run demo.bat
