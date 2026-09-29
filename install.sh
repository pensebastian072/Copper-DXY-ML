#!/usr/bin/env bash
# One-step install for Copper DXY ML (macOS / Linux):  ./install.sh   (then ./start.sh next time)
set -euo pipefail
export PYTHONUTF8=1
cd "$(dirname "$0")"
REQ="requirements.txt"
[ "${1:-}" = "--full" ] && REQ="requirements.txt"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
  else
    PY="$(command -v python3.11 || command -v python3 || true)"
    [ -n "$PY" ] || { echo "Python 3.11+ not found - install it from https://www.python.org/downloads/"; exit 1; }
    "$PY" -m venv .venv
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r "$REQ"
else
  .venv/bin/python -m pip install --disable-pip-version-check -q -r "$REQ"
fi
if [ -f .env.example ] && [ ! -f .env ]; then cp .env.example .env; fi
if [ ! -f outputs/final_model.pkl ]; then
  echo "First run: downloading market data and training the model. This takes a few minutes..."
  .venv/bin/python copper_brain_v2.py
fi
echo "Installed. Next time just run ./start.sh"
exec ./start.sh
