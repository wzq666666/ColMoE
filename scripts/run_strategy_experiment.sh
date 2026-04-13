#!/usr/bin/env bash
set -e

# ================================================================
# 4种重路由策略对比实验
# ================================================================
#
# 策略说明：
#   1. static        - 不做重路由，按当前GPU/CPU分布直接执行
#   2. dynamic       - 动态调度+权重迁移（IO开销重，退化为static）
#   3. io_free       - IO-free专家级替换，整个CPU专家统一替换为相近GPU专家
#   4. token_reroute - Token级重路由（当前默认），逐token逐位置决策
#
# 实验流程：
#   对每个策略：启动sglang服务 → 等待就绪 → 启动GPU监控 → 发送推理请求 → 停止监控 → 收集结果
#
# 使用方法：
#   bash scripts/run_strategy_experiment.sh [--strategies "static io_free token_reroute"] [--num-requests 5]
# ================================================================

# ========== 可配置参数 ==========
STRATEGIES="${STRATEGIES:-static io_free token_reroute}"  # 要测试的策略列表
NUM_REQUESTS="${NUM_REQUESTS:-3}"                         # 每个策略发送的请求数
PORT="${PORT:-30001}"
GPU_ID="${GPU_ID:-3}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"                   # 预热请求数（不计入统计）
HOT_SWITCH="${HOT_SWITCH:-1}"                             # 热切换策略（1=开启，0=重启服务）

# 路径配置
WORKSPACE="/home/ecnu/disk/wzq"
CONFIG_FILE="${WORKSPACE}/moe_hook_config.yaml"
CONFIG_BACKUP="${CONFIG_FILE}.bak"
LOG_DIR="${WORKSPACE}/logs/experiments/$(date +%Y%m%d_%H%M%S)"
MONITOR_SCRIPT="${WORKSPACE}/tool/monitor_gpu_utilization.py"
INFER_SCRIPT="${WORKSPACE}/send_infer_request.py"
DATA_DIR="${WORKSPACE}/moe-inference/data/gsm8k"
SWITCH_TOOL="${WORKSPACE}/tool/switch_strategy.py"

# sglang启动脚本
SGLANG_SCRIPT="${WORKSPACE}/scripts/start_SGLang_MoE_Col_with_Hook.sh"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategies) STRATEGIES="$2"; shift 2 ;;
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --warmup) WARMUP_REQUESTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ========== 工具函数 ==========

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/experiment.log"
}

# 修改配置文件中的reroute_strategy
set_strategy() {
    local strategy="$1"
    log "Setting reroute_strategy to: ${strategy}"
    
    if [ "${HOT_SWITCH}" = "1" ]; then
        # 热切换：使用工具动态切换策略
        log "Hot-switching strategy using switch_strategy.py..."
        python "${SWITCH_TOOL}" --config "${CONFIG_FILE}" --strategy "${strategy}"
        if [ $? -eq 0 ]; then
            log "Strategy hot-switched successfully"
            sleep 2  # 给调度器一点时间重新加载
        else
            log "Hot-switch failed, falling back to config file edit"
            # 回退到传统方法
            if grep -q "^reroute_strategy:" "${CONFIG_FILE}"; then
                sed -i "s/^reroute_strategy:.*/reroute_strategy: ${strategy}/" "${CONFIG_FILE}"
            else
                echo "reroute_strategy: ${strategy}" >> "${CONFIG_FILE}"
            fi
        fi
    else
        # 传统方法：直接修改配置文件
        if grep -q "^reroute_strategy:" "${CONFIG_FILE}"; then
            sed -i "s/^reroute_strategy:.*/reroute_strategy: ${strategy}/" "${CONFIG_FILE}"
        else
            echo "reroute_strategy: ${strategy}" >> "${CONFIG_FILE}"
        fi
    fi
}

# 等待sglang服务就绪
wait_for_server() {
    local max_wait=600  # 最多等待10分钟
    local waited=0
    log "Waiting for sglang server on port ${PORT} (SGLANG_PID=${SGLANG_PID})..."
    
    while [ $waited -lt $max_wait ]; do
        # 检查sglang进程是否还活着
        if ! kill -0 ${SGLANG_PID} 2>/dev/null; then
            log "ERROR: sglang process ${SGLANG_PID} died during startup (waited ${waited}s)"
            return 1
        fi
        
        if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            log "Server is ready! (waited ${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    
    log "ERROR: Server did not start within ${max_wait}s"
    return 1
}

# 停止sglang服务
stop_server() {
    log "Stopping sglang server..."
    
    # 1. 通过端口找到监听进程并杀掉整个进程组
    local pids
    pids=$(lsof -ti :${PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Found processes on port ${PORT}: ${pids}"
        # 直接杀掉进程，避免复杂的进程组逻辑可能导致阻塞
        kill ${pids} 2>/dev/null || true
        sleep 3
        
        # 强制杀掉残留
        pids=$(lsof -ti :${PORT} 2>/dev/null || true)
        if [ -n "$pids" ]; then
            log "Force killing remaining processes: ${pids}"
            kill -9 ${pids} 2>/dev/null || true
        fi
    fi
    
    # 2. 兜底：匹配 run_with_hooks / sglang 相关进程
    pkill -f "run_with_hooks.py.*--port ${PORT}" 2>/dev/null || true
    pkill -f "python.*sglang.*--port ${PORT}" 2>/dev/null || true
    sleep 3
    pkill -9 -f "run_with_hooks.py.*--port ${PORT}" 2>/dev/null || true
    pkill -9 -f "python.*sglang.*--port ${PORT}" 2>/dev/null || true
    
    # 3. 简化端口等待逻辑（最多10秒，避免长时间阻塞）
    local wait_count=0
    while [ $wait_count -lt 10 ]; do
        if ! lsof -ti :${PORT} > /dev/null 2>&1; then
            log "Port ${PORT} is now free (waited ${wait_count}s)"
            break
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    # 4. 如果仍占用，警告但继续执行（避免整个实验卡住）
    if lsof -ti :${PORT} > /dev/null 2>&1; then
        log "WARNING: Port ${PORT} still occupied, but continuing..."
    fi
    
    log "Server stopped."
}

# 发送推理请求并收集response中的moe_stats
send_request() {
    local idx="$1"
    local output_file="$2"
    
    python "${INFER_SCRIPT}" \
        --port "${PORT}" \
        --data-dir "${DATA_DIR}" \
        --data-idx "${idx}" \
        --max-input-tokens 512 \
        --max-new-tokens 256 \
        --temperature 0.6 \
        --use-requests \
        2>&1 | tee -a "${output_file}"
}

# ========== 主流程 ==========

mkdir -p "${LOG_DIR}"
log "========================================"
log "Strategy Comparison Experiment"
log "========================================"
log "Strategies: ${STRATEGIES}"
log "Requests per strategy: ${NUM_REQUESTS}"
log "Warmup requests: ${WARMUP_REQUESTS}"
log "GPU ID: ${GPU_ID}"
log "Port: ${PORT}"
log "Hot Switch Mode: ${HOT_SWITCH}"
log "Log dir: ${LOG_DIR}"
log "========================================"

# 备份配置文件
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
log "Config backed up to ${CONFIG_BACKUP}"

# 创建结果汇总文件
SUMMARY_FILE="${LOG_DIR}/summary.csv"
echo "strategy,request_idx,avg_gpu_compute_ms,avg_cpu_compute_ms,avg_layer_time_ms,reroute_rate,avg_gpu_load_change,avg_cpu_load_change,total_layers" > "${SUMMARY_FILE}"

# 热切换模式：只启动一次服务
SGLANG_PID=""
FIRST_STRATEGY=true

for strategy in ${STRATEGIES}; do
    log ""
    log "========================================"
    log "Testing strategy: ${strategy}"
    log "========================================"
    
    STRATEGY_DIR="${LOG_DIR}/${strategy}"
    mkdir -p "${STRATEGY_DIR}"
    
    # Step 1: 设置策略
    # 对于 "dynamic" 策略，使用 static（因为实际动态迁移的IO开销会让它退化为static）
    # 但我们可以通过启用prefetch来模拟dynamic
    if [ "${strategy}" = "dynamic" ]; then
        set_strategy "static"  # dynamic退化为static（IO >> compute）
    else
        set_strategy "${strategy}"
    fi
    
    # Step 2: 启动或切换服务
    if [ "${HOT_SWITCH}" = "1" ]; then
        if [ "${FIRST_STRATEGY}" = "true" ]; then
            # 第一个策略：启动服务
            log "Starting sglang server for hot-switch mode..."
            stop_server  # 确保端口清空
            
            export CUDA_VISIBLE_DEVICES="${GPU_ID}"
            export MOE_HOOK_ENABLE=1
            export MOE_HOOK_CONFIG="${CONFIG_FILE}"
            export MOE_HOOK_LOG_PATH="${STRATEGY_DIR}/moe_hook.log"
            export MOE_DYNAMIC_SCHEDULING=1

            # 后台启动sglang
            bash "${SGLANG_SCRIPT}" > "${STRATEGY_DIR}/sglang_stdout.log" 2>&1 &
            SGLANG_PID=$!
            log "sglang started (PID=${SGLANG_PID})"
            
            # 等待服务就绪
            if ! wait_for_server; then
                log "SKIP all strategies (server failed to start)"
                log "=== sglang stdout (last 30 lines) ==="
                tail -30 "${STRATEGY_DIR}/sglang_stdout.log" | tee -a "${LOG_DIR}/experiment.log"
                stop_server
                break
            fi
            
            # 验证进程存活
            if ! kill -0 ${SGLANG_PID} 2>/dev/null; then
                log "ERROR: sglang process ${SGLANG_PID} is dead but port ${PORT} is responding"
                log "SKIP all strategies"
                stop_server
                break
            fi
            
            FIRST_STRATEGY=false
        else
            # 后续策略：仅热切换
            log "Hot-switching to strategy=${strategy} (keeping server running)"
        fi
    else
        # 传统模式：每个策略重启服务
        log "Starting sglang server with strategy=${strategy} (restart mode)..."
        stop_server
        
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        export MOE_HOOK_ENABLE=1
        export MOE_HOOK_CONFIG="${CONFIG_FILE}"
        export MOE_HOOK_LOG_PATH="${STRATEGY_DIR}/moe_hook.log"
        export MOE_DYNAMIC_SCHEDULING=1

        # 后台启动sglang
        bash "${SGLANG_SCRIPT}" > "${STRATEGY_DIR}/sglang_stdout.log" 2>&1 &
        SGLANG_PID=$!
        log "sglang started (PID=${SGLANG_PID})"
        
        # 等待服务就绪
        if ! wait_for_server; then
            log "SKIP strategy=${strategy} (server failed to start)"
            log "=== sglang stdout (last 30 lines) ==="
            tail -30 "${STRATEGY_DIR}/sglang_stdout.log" | tee -a "${LOG_DIR}/experiment.log"
            stop_server
            continue
        fi
        
        # 验证进程存活
        if ! kill -0 ${SGLANG_PID} 2>/dev/null; then
            log "ERROR: sglang process ${SGLANG_PID} is dead but port ${PORT} is responding"
            log "SKIP strategy=${strategy}"
            stop_server
            continue
        fi
    fi
    
    # Step 4: 预热
    if [ "${WARMUP_REQUESTS}" -gt 0 ]; then
        log "Sending ${WARMUP_REQUESTS} warmup requests..."
        for ((w=0; w<WARMUP_REQUESTS; w++)); do
            send_request $((w + 100)) "${STRATEGY_DIR}/warmup.log" > /dev/null 2>&1 || true
        done
        log "Warmup done."
    fi
    
    # Step 5: 启动GPU监控
    GPU_UTIL_LOG="${STRATEGY_DIR}/gpu_utilization.log"
    python "${MONITOR_SCRIPT}" \
        --gpu-id "${GPU_ID}" \
        --interval 0.05 \
        --output "${GPU_UTIL_LOG}" &
    MONITOR_PID=$!
    log "GPU monitor started (PID=${MONITOR_PID})"
    sleep 1
    
    # Step 6: 发送推理请求
    for ((i=0; i<NUM_REQUESTS; i++)); do
        log "Request ${i}/${NUM_REQUESTS} for strategy=${strategy}"
        REQUEST_LOG="${STRATEGY_DIR}/request_${i}.log"
        send_request "${i}" "${REQUEST_LOG}"
        sleep 2  # 间隔，避免请求重叠
    done
    
    # Step 7: 停止GPU监控
    sleep 2
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    log "GPU monitor stopped."
    
    # Step 8: 策略清理
    if [ "${HOT_SWITCH}" = "1" ]; then
        # 热切换模式：不停止服务，留到最后统一停止
        log "Strategy ${strategy} completed (hot-switch mode). Logs in ${STRATEGY_DIR}/"
    else
        # 传统模式：每个策略后停止服务
        stop_server
        log "Strategy ${strategy} completed. Logs in ${STRATEGY_DIR}/"
    fi
done

# 热切换模式的最终清理
if [ "${HOT_SWITCH}" = "1" ] && [ -n "${SGLANG_PID}" ]; then
    log "Stopping sglang server (end of hot-switch experiment)..."
    stop_server
fi

# 恢复配置
cp "${CONFIG_BACKUP}" "${CONFIG_FILE}"
log "Config restored from backup."

log ""
log "========================================"
log "Experiment Complete!"
log "========================================"
log "Results dir: ${LOG_DIR}"
log "Summary CSV: ${SUMMARY_FILE}"
log ""
log "To analyze results:"
log "  python scripts/analyze_strategy_experiment.py --log-dir ${LOG_DIR}"
