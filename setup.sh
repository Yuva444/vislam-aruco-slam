#!/usr/bin/env bash
# One-time setup (Linux / macOS): creates .venv and installs requirements.txt into it.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python) || { echo "Python 3 not found"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "Done.  Activate with:  source .venv/bin/activate   then run ./demo.sh"
