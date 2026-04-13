#!/usr/bin/env bash
set -e

chmod +x /home/ecnu/disk/wzq/scripts/run_with_hooks.sh

# ==========================================
# 动态专家调度启动脚本
# ==========================================

# 环境变量配置
export CUDA_VISIBLE_DEVICES=3
export SGLANG_LOG_LEVEL=debug

# MOE Hook 配置
export MOE_HOOK_ENABLE=1
export MOE_HOOK_CONFIG=/home/ecnu/disk/wzq/configs/moe_hook_load_aware.yaml
export MOE_HOOK_LOG_PATH=/home/ecnu/disk/wzq/logs/moe_hook.log

# sglang输出保存配置
# export SGLANG_SAVE_LOGITS_DIR=/home/ecnu/disk/wzq/output/sglang_saved_logits/gsm8k/idx5
# export SGLANG_SAVE_LOGITS_FILENAME_TEMPLATE="req_tp{tp_rank}_pass{forward_pass_id}_{timestamp}_tl_0.15.pt"

# 动态调度配置 (设为1启用)
export MOE_DYNAMIC_SCHEDULING=1

# 启动服务
# 注意: 
# - --mem-fraction-static 0.9: 预留更多 GPU 内存给模型 + GPU expert cache
# - --disable-cuda-graph: 必须禁用，否则动态专家调度无法生效
# - --kt-num-gpu-experts x: 使用原生GPU专家以获得最佳性能
# - --ep-dispatch-algorithm none: 禁用 EP 分发，避免物理 ID 映射问题
/home/ecnu/disk/wzq/scripts/run_with_hooks.sh \
  --model-path /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct \
  --tp 1 \
  --port 30001 \
  --kt-method LLAMAFILE \
  --kt-weight-path /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct-gguf \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 10 \
  --kt-max-deferred-experts-per-token 0 \
  --dtype bfloat16 \
  --max-total-tokens 10240 \
  --max-prefill-tokens 5120 \
  --mem-fraction-static 0.9 \
  --disable-cuda-graph \
  --ep-dispatch-algorithm none \
  --enable-metrics \
  --log-level info \
  --meta-info-save-path output/gsm8k_max_new_512/qwen2-57B-A14B/la_nolimit_v1.jsonl
    