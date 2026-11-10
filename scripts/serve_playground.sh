#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3000}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-gs1_local}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-30}"
BACKENDS_CONFIG="${BACKENDS_CONFIG:-$REPO_ROOT/playground/backends.yaml}"
cd "$REPO_ROOT/playground"
exec npm run server -- --host "$HOST" --port "$PORT" --default-backend "$DEFAULT_BACKEND" \
  --deadline-seconds "$DEADLINE_SECONDS" \
  --backends-config "$BACKENDS_CONFIG" --env-file "$REPO_ROOT/.env" "$@"
