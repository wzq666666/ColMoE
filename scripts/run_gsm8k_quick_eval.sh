#!/usr/bin/env bash
# ==========================================
# 简化版 GSM8K 评估脚本
# 使用自定义 Python 脚本而非 OpenCompass
# ==========================================

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GSM8K 精度评估 - 快速版${NC}"
echo -e "${BLUE}========================================${NC}"

# 配置
WORKSPACE=/home/ecnu/disk/wzq
OUTPUT_DIR=${WORKSPACE}/output/gsm8k_simple_eval_$(date +%Y%m%d_%H%M%S)
DATASET=${WORKSPACE}/moe-inference/data/gsm8k/test  # arrow 格式数据集目录
API_URL="http://localhost:30001/v1/chat/completions"
NUM_SAMPLES=300  # 只评估前 300 个样本，可根据需要调整

mkdir -p ${OUTPUT_DIR}

# 检查数据集
if [ ! -d "${DATASET}" ] && [ ! -f "${DATASET}" ]; then
    echo -e "${RED}错误: 数据集未找到: ${DATASET}${NC}"
    echo -e "${GREEN}请先准备 GSM8K 数据集${NC}"
    echo -e "  支持格式: arrow目录 或 jsonl文件"
    exit 1
fi

# 检查服务是否运行
if ! curl -s http://localhost:30001/health > /dev/null 2>&1; then
    echo -e "${RED}错误: SGLang 服务未运行${NC}"
    echo -e "${GREEN}请先启动服务:${NC}"
    echo -e "  bash ${WORKSPACE}/scripts/start_SGLang_MoE_Col_with_Hook.sh"
    exit 1
fi

# 获取当前配置
CURRENT_CONFIG=${MOE_HOOK_CONFIG:-/home/ecnu/disk/wzq/configs/moe_hook_expert_level.yaml}
STRATEGY=$(grep "reroute_strategy:" ${CURRENT_CONFIG} | awk '{print $2}')

echo -e "${GREEN}当前配置:${NC}"
echo -e "  配置文件: ${CURRENT_CONFIG}"
echo -e "  替换策略: ${STRATEGY}"
echo -e "  数据集: ${DATASET}"
echo -e "  评估样本数: ${NUM_SAMPLES}"
echo -e ""

# 运行评估
echo -e "${BLUE}开始评估...${NC}"

python3 ${WORKSPACE}/scripts/evaluate_gsm8k.py \
    --mode evaluate \
    --dataset ${DATASET} \
    --api-url ${API_URL} \
    --model "qwen2-moe-${STRATEGY}" \
    --output ${OUTPUT_DIR}/results_${STRATEGY}.json \
    --num-samples ${NUM_SAMPLES}

echo -e "${GREEN}评估完成!${NC}"
echo -e "结果文件: ${OUTPUT_DIR}/results_${STRATEGY}.json"
echo -e ""
echo -e "${BLUE}查看结果:${NC}"
echo -e "  python3 -c \"import json; d=json.load(open('${OUTPUT_DIR}/results_${STRATEGY}.json')); print('准确率:', d['statistics']['accuracy'])\""
