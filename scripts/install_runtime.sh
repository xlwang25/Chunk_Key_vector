#!/usr/bin/env bash
set -euo pipefail
artifact_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_target="$artifact_root/vendor/vllm"

# Source and model hooks are already bundled and integrated.
python "$artifact_root/scripts/verify_runtime.py"
SETUPTOOLS_SCM_PRETEND_VERSION="0.10.1.1" \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.10.1.1" \
  python -m pip install -e "$runtime_target"
python -m pip install -e "$artifact_root"
python -m keychunk.runtime.rope
