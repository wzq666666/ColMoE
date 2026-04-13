#!/usr/bin/env bash
# 启动推理并同时监控GPU利用率

set -e

# 配置
GPU_ID=3  # 与 start_SGLang_MoE_Col_with_Hook.sh 中的 CUDA_VISIBLE_DEVICES 一致
MONITOR_INTERVAL=0.1  # 采样间隔(秒)
LOG_DIR="/home/ecnu/disk/wzq/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志文件
GPU_UTIL_LOG="${LOG_DIR}/gpu_util_${TIMESTAMP}.log"
INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"

echo "========================================"
echo "Inference with GPU Monitoring"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "Monitoring Interval: ${MONITOR_INTERVAL}s"
echo "GPU Utilization Log: ${GPU_UTIL_LOG}"
echo "Inference Log: ${INFERENCE_LOG}"
echo "========================================"

# 启动GPU监控（后台运行）
echo "Starting GPU monitor..."
python tool/monitor_gpu_utilization.py \
  --gpu-id ${GPU_ID} \
  --interval ${MONITOR_INTERVAL} \
  --output "${GPU_UTIL_LOG}" &

MONITOR_PID=$!
echo "GPU Monitor PID: ${MONITOR_PID}"

# 等待监控器初始化
sleep 2

# 运行推理
echo "Starting inference..."
python send_infer_request.py \
  --port 30001 \
  --data-dir /home/ecnu/disk/wzq/moe-inference/data/gsm8k \
  --data-idx 5 \
  --max-input-tokens 512 \
  --max-new-tokens 4096 \
  --temperature 0.6 \
  --use-requests 2>&1 | tee "${INFERENCE_LOG}"

INFERENCE_EXIT=$?

# 等待一点时间确保监控捕获完整过程
sleep 2

# 停止GPU监控
echo "Stopping GPU monitor..."
kill -SIGINT ${MONITOR_PID} 2>/dev/null || true
wait ${MONITOR_PID} 2>/dev/null || true

echo ""
echo "========================================"
echo "Completed!"
echo "========================================"
echo "GPU Utilization Log: ${GPU_UTIL_LOG}"
echo "Inference Log: ${INFERENCE_LOG}"
echo ""
echo "To view statistics, check the end of GPU log:"
echo "  tail -50 ${GPU_UTIL_LOG}"
echo "========================================"

exit ${INFERENCE_EXIT}
