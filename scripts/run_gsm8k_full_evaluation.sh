#!/usr/bin/env bash
# ==========================================
# GSM8K 完整评估脚本 - Token粒度 vs 专家粒度
# ==========================================

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GSM8K 精度评估 - 专家替换策略对比${NC}"
echo -e "${BLUE}========================================${NC}"

# 配置路径
WORKSPACE=/home/ecnu/disk/wzq
OPENCOMPASS_DIR=${WORKSPACE}/opencompass
OUTPUT_DIR=${WORKSPACE}/output/gsm8k_evaluation_$(date +%Y%m%d_%H%M%S)
CONFIGS_DIR=${WORKSPACE}/configs
SCRIPTS_DIR=${WORKSPACE}/scripts

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CONFIGS_DIR}

# 检查 OpenCompass 是否安装
if [ ! -d "${OPENCOMPASS_DIR}" ]; then
    echo -e "${RED}错误: OpenCompass 未安装${NC}"
    echo -e "${GREEN}正在克隆 OpenCompass...${NC}"
    cd ${WORKSPACE}
    git clone https://github.com/open-compass/opencompass.git
    cd opencompass
    pip install -e .
fi

# ==========================================
# 函数定义
# ==========================================

# 停止现有的 SGLang 服务
stop_sglang() {
    echo -e "${GREEN}停止现有的 SGLang 服务...${NC}"
    pkill -f "python.*sglang" || true
    sleep 5
}

# 等待服务就绪
wait_for_service() {
    local port=$1
    local max_wait=120
    local waited=0
    
    echo -e "${GREEN}等待服务在端口 ${port} 就绪...${NC}"
    while ! curl -s http://localhost:${port}/health > /dev/null 2>&1; do
        if [ $waited -ge $max_wait ]; then
            echo -e "${RED}错误: 服务启动超时${NC}"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  等待中... ${waited}s / ${max_wait}s"
    done
    echo -e "${GREEN}服务就绪!${NC}"
}

# 运行 OpenCompass 评估
run_opencompass_eval() {
    local strategy_name=$1
    local work_dir=$2
    
    echo -e "${GREEN}运行 OpenCompass 评估: ${strategy_name}${NC}"
    
    cd ${OPENCOMPASS_DIR}
    
    python run.py \
        --datasets gsm8k_gen \
        --hf-type chat \
        --model-type api \
        --url http://localhost:30001/v1/chat/completions \
        --model-name "qwen2-57b-moe-${strategy_name}" \
        --work-dir "${work_dir}" \
        --max-num-workers 4 \
        --max-out-len 512 \
        2>&1 | tee "${OUTPUT_DIR}/${strategy_name}_eval.log"
    
    # 复制结果
    if [ -d "${work_dir}/summary" ]; then
        cp -r "${work_dir}/summary" "${OUTPUT_DIR}/${strategy_name}_summary"
        echo -e "${GREEN}结果已保存到: ${OUTPUT_DIR}/${strategy_name}_summary${NC}"
    fi
}

# 使用自定义评估脚本（支持 arrow 格式）
run_custom_eval() {
    local strategy_name=$1
    local output_file=$2
    
    echo -e "${GREEN}运行自定义评估: ${strategy_name}${NC}"
    
    python3 ${WORKSPACE}/scripts/evaluate_gsm8k.py \
        --mode evaluate \
        --dataset ${WORKSPACE}/moe-inference/data/gsm8k/test \
        --api-url http://localhost:30001/v1/chat/completions \
        --model "qwen2-57b-moe-${strategy_name}" \
        --output "${output_file}" \
        --num-samples 200 \
        2>&1 | tee "${OUTPUT_DIR}/${strategy_name}_custom_eval.log"
}

# ==========================================
# 创建配置文件
# ==========================================

echo -e "${BLUE}创建配置文件...${NC}"

# Token 粒度配置
cat > ${CONFIGS_DIR}/moe_hook_token_level.yaml <<EOF
enable: true
dynamic_scheduling: true
reroute_strategy: token_low_score
reroute_alpha: 0.35
max_gpu_experts_per_layer: 10
cpu_cache_max_experts: 64
enable_cpu_weight_cache: true
enable_pinned_memory: false
disable_deferral: true
num_transfer_streams: 2
pinned_pool_size: 64

gate:
  enable: true
  device: null
  format: auto
  model_path: /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct
  next_layer_offset: 1
  patterns:
  - model.layers.{idx}.mlp.gate.weight
  total_layers: null

predict:
  enable: false
  mode: fate

prefetch:
  enable: false
  mode: layer

preload:
  enable: true
  mode: average
  cache_ratio: 0.2

hf_model_path: /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct
model_path: /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct
log_path: ${OUTPUT_DIR}/token_level_moe_hook.log
log_level: 3
scheduling_backend: native
capture_bs: [1, 2, 4, 8]
EOF

# 专家粒度配置
cat > ${CONFIGS_DIR}/moe_hook_expert_level.yaml <<EOF
enable: true
dynamic_scheduling: true
reroute_strategy: expert_reroute
reroute_alpha: 0.35
max_gpu_experts_per_layer: 10
cpu_cache_max_experts: 64
enable_cpu_weight_cache: true
enable_pinned_memory: false
disable_deferral: true
num_transfer_streams: 2
pinned_pool_size: 64

gate:
  enable: true
  device: null
  format: auto
  model_path: /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct
  next_layer_offset: 1
  patterns:
  - model.layers.{idx}.mlp.gate.weight
  total_layers: null

predict:
  enable: false
  mode: fate

prefetch:
  enable: false
  mode: layer

preload:
  enable: true
  mode: average
  cache_ratio: 0.2

hf_model_path: /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct
model_path: /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct
log_path: ${OUTPUT_DIR}/expert_level_moe_hook.log
log_level: 3
scheduling_backend: native
capture_bs: [1, 2, 4, 8]
EOF

echo -e "${GREEN}配置文件创建完成${NC}"

# ==========================================
# 评估 1: Token 粒度替换
# ==========================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  评估 1: Token 粒度替换${NC}"
echo -e "${BLUE}========================================${NC}"

stop_sglang

export CUDA_VISIBLE_DEVICES=3
export SGLANG_LOG_LEVEL=info
export MOE_HOOK_ENABLE=1
export MOE_HOOK_CONFIG=${CONFIGS_DIR}/moe_hook_token_level.yaml
export MOE_DYNAMIC_SCHEDULING=1

echo -e "${GREEN}启动 SGLang 服务 (Token 粒度)...${NC}"

${SCRIPTS_DIR}/run_with_hooks.sh \
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
  --max-prefill-tokens 2560 \
  --mem-fraction-static 0.9 \
  --disable-cuda-graph \
  --ep-dispatch-algorithm none \
  --enable-metrics \
  --log-level info \
  > ${OUTPUT_DIR}/token_level_sglang.log 2>&1 &

SGLANG_PID=$!
echo "SGLang PID: ${SGLANG_PID}"

# 等待服务就绪
if wait_for_service 30001; then
    # 运行评估（使用自定义脚本，支持 arrow 格式）
    run_custom_eval "token_level" "${OUTPUT_DIR}/token_level_results.json"
else
    echo -e "${RED}服务启动失败，跳过评估${NC}"
fi

# 停止服务
stop_sglang

# ==========================================
# 评估 2: 专家粒度替换
# ==========================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  评估 2: 专家粒度替换${NC}"
echo -e "${BLUE}========================================${NC}"

export MOE_HOOK_CONFIG=${CONFIGS_DIR}/moe_hook_expert_level.yaml

echo -e "${GREEN}启动 SGLang 服务 (专家粒度)...${NC}"

${SCRIPTS_DIR}/run_with_hooks.sh \
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
  --max-prefill-tokens 2560 \
  --mem-fraction-static 0.9 \
  --disable-cuda-graph \
  --ep-dispatch-algorithm none \
  --enable-metrics \
  --log-level info \
  > ${OUTPUT_DIR}/expert_level_sglang.log 2>&1 &

SGLANG_PID=$!
echo "SGLang PID: ${SGLANG_PID}"

# 等待服务就绪
if wait_for_service 30001; then
    # 运行评估（使用自定义脚本，支持 arrow 格式）
    run_custom_eval "expert_level" "${OUTPUT_DIR}/expert_level_results.json"
else
    echo -e "${RED}服务启动失败，跳过评估${NC}"
fi

# 停止服务
stop_sglang

# ==========================================
# 生成对比报告
# ==========================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  生成对比报告${NC}"
echo -e "${BLUE}========================================${NC}"

cat > ${OUTPUT_DIR}/comparison_report.md <<EOF
# GSM8K 精度评估报告 - 专家替换策略对比

**评估时间**: $(date)
**评估数据集**: GSM8K
**模型**: Qwen2-57B-A14B-Instruct

## 评估配置

### Token 粒度替换
- **策略**: token_low_score
- **Alpha**: 0.35
- **GPU 专家数**: 10
- **特点**: 允许重复专家，细粒度控制

### 专家粒度替换
- **策略**: expert_reroute
- **Alpha**: 0.35
- **GPU 专家数**: 10
- **特点**: 不允许重复，粗粒度控制

## 结果对比

### Token 粒度替换结果
\`\`\`
EOF

# 添加 Token 粒度结果
if [ -f "${OUTPUT_DIR}/token_level_summary/summary.csv" ]; then
    cat ${OUTPUT_DIR}/token_level_summary/summary.csv >> ${OUTPUT_DIR}/comparison_report.md
else
    echo "结果文件未找到" >> ${OUTPUT_DIR}/comparison_report.md
fi

cat >> ${OUTPUT_DIR}/comparison_report.md <<EOF
\`\`\`

### 专家粒度替换结果
\`\`\`
EOF

# 添加专家粒度结果
if [ -f "${OUTPUT_DIR}/expert_level_summary/summary.csv" ]; then
    cat ${OUTPUT_DIR}/expert_level_summary/summary.csv >> ${OUTPUT_DIR}/comparison_report.md
else
    echo "结果文件未找到" >> ${OUTPUT_DIR}/comparison_report.md
fi

cat >> ${OUTPUT_DIR}/comparison_report.md <<EOF
\`\`\`

## 日志文件

- Token 粒度 SGLang 日志: token_level_sglang.log
- Token 粒度 Hook 日志: token_level_moe_hook.log
- 专家粒度 SGLang 日志: expert_level_sglang.log
- 专家粒度 Hook 日志: expert_level_moe_hook.log

## 分析建议

1. 比较两种策略的 **Accuracy** 指标
2. 检查 Hook 日志中的专家替换统计
3. 分析性能差异（推理延迟、吞吐量）

EOF

echo -e "${GREEN}对比报告已生成: ${OUTPUT_DIR}/comparison_report.md${NC}"

# ==========================================
# 完成
# ==========================================

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  评估完成!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "结果目录: ${OUTPUT_DIR}"
echo -e "对比报告: ${OUTPUT_DIR}/comparison_report.md"
echo -e ""
echo -e "查看对比报告:"
echo -e "  cat ${OUTPUT_DIR}/comparison_report.md"
