#!/usr/bin/env bash
set -euo pipefail

RELDEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${RELDEC_DIR}/.." && pwd)"

DEFAULT_PYTHON=""
if [[ -x "/root/.pyenv/versions/myenv/bin/python" ]]; then
  DEFAULT_PYTHON="/root/.pyenv/versions/myenv/bin/python"
elif [[ -x "${WORKSPACE_DIR}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
else
  DEFAULT_PYTHON="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

LOCAL_LDPC_SRC="${WORKSPACE_DIR}/ldpc/src_python"
if [[ -d "${LOCAL_LDPC_SRC}" ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${LOCAL_LDPC_SRC}:${PYTHONPATH}"
  else
    export PYTHONPATH="${LOCAL_LDPC_SRC}"
  fi
fi

now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

ensure_run_dirs() {
  local run_root="$1"
  mkdir -p "${run_root}" "${run_root}/state" "${run_root}/logs" "${run_root}/pids" "${run_root}/plots"
}

write_state_json() {
  local state_file="$1"
  local status="$2"
  local code="$3"
  local pid="$4"
  local message="$5"
  local extra_json="${6:-{}}"

  "${PYTHON_BIN}" - "$state_file" "$status" "$code" "$pid" "$message" "$extra_json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_file = Path(sys.argv[1])
status = sys.argv[2]
code = sys.argv[3]
pid = int(sys.argv[4]) if sys.argv[4].isdigit() else None
message = sys.argv[5]
extra_raw = sys.argv[6]

if state_file.exists():
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}
else:
    data = {}

try:
    extra = json.loads(extra_raw)
except Exception:
    extra = {}

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

if "created_at" not in data:
    data["created_at"] = now

data["updated_at"] = now
data["status"] = status
data["code"] = code
if pid is not None:
    data["pid"] = pid
if message:
    data["message"] = message

for k, v in extra.items():
    data[k] = v

state_file.parent.mkdir(parents=True, exist_ok=True)
state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
}

is_pid_alive() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  kill -0 "${pid}" >/dev/null 2>&1
}

read_manifest_field() {
  local manifest="$1"
  local field="$2"
  "${PYTHON_BIN}" - "$manifest" "$field" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
field = sys.argv[2]
obj = json.loads(manifest.read_text(encoding="utf-8"))
parts = field.split('.')
cur = obj
for p in parts:
    cur = cur[p]
if cur is None:
    print("null")
elif isinstance(cur, (dict, list)):
    print(json.dumps(cur))
else:
    print(cur)
PY
}
