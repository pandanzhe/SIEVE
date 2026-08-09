#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_DIR="${SIEVE_ENV_DIR:-$ROOT_DIR/.venv-stage1}"
"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements/train.txt"
python -m pip install -e "$ROOT_DIR"
