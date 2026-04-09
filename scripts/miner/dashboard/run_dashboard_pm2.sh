#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DASHBOARD_NAME="${DASHBOARD_NAME:-poker44_dashboard}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10298}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
PM2_APP_NAME="${PM2_APP_NAME:-poker44_miner}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "python not found at $PYTHON_BIN"
  echo "Run: ./scripts/miner/setup.sh"
  exit 1
fi

cd "$REPO_ROOT"

pm2 delete "$DASHBOARD_NAME" >/dev/null 2>&1 || true
pm2 start "$PYTHON_BIN" \
  --name "$DASHBOARD_NAME" \
  -- scripts/miner/dashboard/miner_dashboard.py \
  --host "$HOST" \
  --port "$PORT"

if ! pm2 save; then
  echo "Warning: pm2 save failed in current environment."
  echo "Run 'pm2 save' manually on your host shell if needed."
fi

echo "Dashboard started: $DASHBOARD_NAME"
echo "URL: http://$HOST:$PORT"
echo "Watch: pm2 logs $DASHBOARD_NAME"
echo "Target miner app for health: $PM2_APP_NAME"
