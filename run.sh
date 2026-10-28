#!/usr/bin/env bash
# QuantAgent-Desk one-click launcher (macOS / Linux)
# Starts API (port 8000) + Streamlit UI (port 8501)

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Activate venv if it exists
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "✓ Activated .venv"
else
  echo "⚠ No .venv found. Creating one..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  echo "✓ .venv created and dependencies installed"
fi

export PYTHONPATH="$ROOT/src"

# Kill any previous instances on these ports (optional, quiet)
(lsof -ti:8000 | xargs kill -9 2>/dev/null) || true
(lsof -ti:8501 | xargs kill -9 2>/dev/null) || true

echo ""
echo "Starting QuantAgent-Desk..."
echo "  API  → http://localhost:8000"
echo "  UI   → http://localhost:8501"
echo ""

# Start API in background
uvicorn quantagent_desk.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Give API a moment to start
sleep 2

# Start Streamlit (foreground so Ctrl+C stops both)
cleanup() {
  echo ""
  echo "Shutting down..."
  kill $API_PID 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

streamlit run src/quantagent_desk/ui/app.py --server.port 8501 --server.headless true

# If streamlit exits, also stop API
kill $API_PID 2>/dev/null || true
