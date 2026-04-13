#!/usr/bin/env bash
set -euo pipefail

# Run send_infer_request.py for data-idx 0..299 in batch mode.
# Usage: ./scripts/run_requests_0_99.sh

PORT=30001
BRANCH='train'
DATA_DIR="/home/ecnu/disk/wzq/moe-inference/data/gsm8k"
MAX_INPUT=5000
MAX_NEW=512  # 降低以适应batch模式，GSM8K答案通常不需要4096 tokens
TEMPERATURE=0.8
START_IDX=0
REQUEST_NUM=100  # 总请求数，可根据需要调整
START_IDX=${START_IDX:-0}  # 从哪个 data-idx 开始提交（0-based），用于续跑场景
PYTHON=python
LOGDIR="output/accuracy_loss/qwen2-57B-A14B"
DELAY=1

BATCH_SIZE=1  # 批量大小

LAST_IDX=$((REQUEST_NUM - 1))

mkdir -p "${LOGDIR}"

echo "Starting batch requests ${START_IDX}..${LAST_IDX} (total=${REQUEST_NUM}, batch_size=${BATCH_SIZE}) -> logs in ${LOGDIR}"

# 批量处理：分成多个batch
for start_idx in $(seq ${START_IDX} ${BATCH_SIZE} ${LAST_IDX}); do
  # 构建batch的data-idx列表
  batch_indices=()
  for offset in $(seq 0 $((BATCH_SIZE - 1))); do
    idx=$((start_idx + offset))
    if [ ${idx} -le ${LAST_IDX} ]; then
      batch_indices+=(${idx})
    fi
  done
  
  echo "==> Running batch request indices: ${batch_indices[@]}"
  ${PYTHON} send_infer_request.py \
    --port ${PORT} \
    --branch ${BRANCH} \
    --data-dir "${DATA_DIR}" \
    --data-idx ${batch_indices[@]} \
    --max-input-tokens ${MAX_INPUT} \
    --max-new-tokens ${MAX_NEW} \
    --temperature ${TEMPERATURE} \
    --use-requests \
    --output "${LOGDIR}/result_tl_rq${REQUEST_NUM}.jsonl"
    > /dev/null 2>&1 || {
      echo "Batch ${start_idx} failed"
    }
  sleep ${DELAY}
done

echo "Finished all batch requests. Check ${LOGDIR} for logs and /home/ecnu/disk/wzq/inference_results.json for results."
