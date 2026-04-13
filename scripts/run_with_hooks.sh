#!/usr/bin/env bash
set -euo pipefail

# Add your project to PYTHONPATH for hooks and component modules
export PYTHONPATH="/home/ecnu/disk/wzq:/home/ecnu/disk/wzq/moe-inference/src:${PYTHONPATH:-}"

# Default hook config (can override)
export MOE_HOOK_ENABLE="${MOE_HOOK_ENABLE:-1}"
export MOE_HOOK_CONFIG="${MOE_HOOK_CONFIG:-/home/ecnu/disk/wzq/moe_hook_config.yaml}"

# Unbuffered stdout for immediate hook日志输出
export PYTHONUNBUFFERED=1

# Helpful CUDA alloc conf to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Run the server through the Python wrapper that imports hooks first
# Use absolute path to Python to support sudo execution
exec /home/ecnu/disk/envs/wzq-kf/bin/python /home/ecnu/disk/wzq/run_with_hooks.py "$@"
