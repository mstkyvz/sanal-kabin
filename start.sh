#!/bin/bash
# Sanal Kabin'i başlatır: http://localhost:7861
cd "$(dirname "$0")"
PORT=${PORT:-7861}
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
[ -z "$NO_OPEN" ] && (sleep 4 && open http://localhost:$PORT) &
exec uvicorn server:app --app-dir app --host 127.0.0.1 --port $PORT
