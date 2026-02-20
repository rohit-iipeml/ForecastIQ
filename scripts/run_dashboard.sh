#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

PORT="${1:-8501}"

if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
  echo "Dashboard is already running on http://localhost:${PORT}"
  exit 0
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

exec streamlit run src/ui/app.py \
  --server.headless true \
  --server.address 127.0.0.1 \
  --server.port "${PORT}" \
  --server.runOnSave false \
  --browser.serverAddress localhost
