@echo off
REM Demo (Windows): runs inside .venv, makes the replay video + HTML, then opens the dashboard.
cd /d "%~dp0"
if not exist .venv (echo Run setup.bat first & exit /b 1)
call .venv\Scripts\activate
python -m pytest -q
python scripts\make_replay.py --simulate E2 --speed 2 --out results\demo_replay.mp4 --html results\demo_replay.html
echo Created results\demo_replay.mp4 and results\demo_replay.html
streamlit run app\dashboard.py
