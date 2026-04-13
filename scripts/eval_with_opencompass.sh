#!/usr/bin/env bash
# 使用 OpenCompass 评估 GSM8K

set -e

cd /home/ecnu/disk/wzq

echo "==================================="
echo "  OpenCompass GSM8K 评估"
echo "==================================="

# 确保服务在运行
if ! curl -s http://localhost:30001/health > /dev/null 2>&1; then
    echo "错误: SGLang 服务未运行"
    exit 1
fi

# 获取当前策略
CURRENT_CONFIG=${MOE_HOOK_CONFIG:-/home/ecnu/disk/wzq/configs/moe_hook_expert_level.yaml}
STRATEGY=$(grep "reroute_strategy:" ${CURRENT_CONFIG} 2>/dev/null | awk '{print $2}' || echo "unknown")

echo "当前策略: ${STRATEGY}"
echo "配置文件: ${CURRENT_CONFIG}"
echo ""

# 运行 OpenCompass
cd /home/ecnu/disk/wzq/opencompass

python run.py \
    /home/ecnu/disk/wzq/opencompass_eval_config.py \
    --work-dir /home/ecnu/disk/wzq/output/opencompass_${STRATEGY}_$(date +%Y%m%d_%H%M%S)

echo ""
echo "评估完成！"
