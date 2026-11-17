#!/usr/bin/env bash
# One-time setup on the Linux GPU machine. Run from the repo root
# (/workspace/relevance-judge-distill). Everything lives under /workspace so
# it survives a pod stop. Exact versions are frozen into requirements-gpu.lock
# on the first install; later installs reuse the lock file.
set -euo pipefail
export HF_HOME=/workspace/hf
export PIP_CACHE_DIR=/workspace/pip-cache
VENV=/workspace/venv

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
if [ -f requirements-gpu.lock ]; then
  "$VENV/bin/pip" install -q -r requirements-gpu.lock
else
  "$VENV/bin/pip" install -q torch transformers peft accelerate matplotlib \
      numpy==2.5.5 pandas==3.1.1 pyarrow==26.0.1 PyYAML==6.0.3 pytest==9.2.0
  "$VENV/bin/pip" freeze > requirements-gpu.lock
fi

"$VENV/bin/python" -c "import torch, peft, transformers; print('versions:', torch.__version__, peft.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$VENV/bin/python" scripts/download_data.py
"$VENV/bin/python" -m pytest -q
echo SETUP_DONE
