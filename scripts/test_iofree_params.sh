#!/bin/bash
# 测试不同io_free参数对重路由率的影响

WORKSPACE="/home/ecnu/disk/wzq"
CONFIG_FILE="${WORKSPACE}/moe_hook_config.yaml"
CONFIG_BACKUP="${CONFIG_FILE}.param_test_bak"

# 备份当前配置
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"

echo "Testing io_free strategy with different parameters..."
echo "=========================================="

# 测试不同的score_threshold_ratio值
for threshold in 0.5 0.7 0.9; do
    for alpha in 0.2 0.3 0.4; do
        echo ""
        echo "Testing: threshold=${threshold}, alpha=${alpha}"
        
        # 修改配置
        python "${WORKSPACE}/tool/switch_strategy.py" \
            --config "${CONFIG_FILE}" \
            --strategy io_free \
            --score-threshold-ratio ${threshold} \
            --alpha ${alpha}
        
        # 发送一个测试请求
        response=$(python "${WORKSPACE}/send_infer_request.py" \
            --data-dir "${WORKSPACE}/moe-inference/data/gsm8k" \
            --num-requests 1 2>/dev/null | tail -1)
        
        # 提取重路由率
        reroute_rate=$(echo "$response" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    rate = data.get('meta_info', {}).get('moe_reroute_rate', 0) * 100
    print(f'{rate:.1f}%')
except:
    print('N/A')
")
        
        echo "  Reroute rate: ${reroute_rate}"
        sleep 2
    done
done

# 恢复配置
cp "${CONFIG_BACKUP}" "${CONFIG_FILE}"
rm "${CONFIG_BACKUP}"

echo ""
echo "=========================================="
echo "Test completed. Config restored."
