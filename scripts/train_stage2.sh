#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG_PATH="${1:-configs/rl_qwen25_3b.yaml}"
NUM_PROCESSES="${SIEVE_NUM_GPUS:-2}"
REQUESTED_MAIN_PROCESS_PORT="${SIEVE_MAIN_PROCESS_PORT:-}"
find_free_port() {
  "$PYTHON_BIN" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}
if [[ -n "$REQUESTED_MAIN_PROCESS_PORT" ]]; then
  if "$PYTHON_BIN" - "$REQUESTED_MAIN_PROCESS_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
    except OSError as error:
        raise SystemExit(
            f"requested main process port is unavailable: {port} ({error})"
        )
PY
  then
    MAIN_PROCESS_PORT="$REQUESTED_MAIN_PROCESS_PORT"
  else
    MAIN_PROCESS_PORT="$(find_free_port)"
    echo "[stage2] requested port unavailable; using free port=${MAIN_PROCESS_PORT}"
  fi
else
  MAIN_PROCESS_PORT="$(find_free_port)"
fi
export MASTER_PORT="$MAIN_PROCESS_PORT"
echo "[stage2] using main_process_port=${MAIN_PROCESS_PORT}"

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  -m sieve.cli.train_rl \
  --root "$ROOT_DIR" \
  --config "$CONFIG_PATH" \
  --num-processes "$NUM_PROCESSES"
