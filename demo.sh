#!/usr/bin/env bash
# Demo (Linux / macOS): runs inside .venv, makes the replay video + HTML, then opens the dashboard.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first"; exit 1; }
source .venv/bin/activate
python -m pytest -q
python scripts/make_replay.py --simulate E2 --speed 2 --out results/demo_replay.mp4 --html results/demo_replay.html
echo "Created results/demo_replay.mp4 and results/demo_replay.html"
streamlit run app/dashboard.py
