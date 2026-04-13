import torch
from ..logger import append_log
log_path = "/home/ecnu/disk/wzq/logs/moe_hook.log"

def analyze_expert_load_vs_score(routing_weights: torch.Tensor, selected_experts: torch.Tensor, min_cnt: int = 10):
    """
    分析高负载专家的得分情况：验证高负载专家是否一定是高得分专家。
    有些专家虽然被选中，但得分并非最高，可能在top-k中得分较低。

    Args:
        routing_weights: 路由权重/logits [num_tokens, num_experts]
        selected_experts: 选中的专家ID [num_tokens, k]
        min_cnt: 最少选中次数阈值，用于过滤低频专家

    Returns:
        dict: 包含相关性分析结果
    """
    if routing_weights.dim() > 2:
        routing_weights = routing_weights.view(-1, routing_weights.size(-1))

    _, num_features = routing_weights.shape
    num_experts = num_features

    # 计算每个专家的平均gate score（logits/概率）
    avg_score_per_expert = routing_weights.mean(dim=0)  # [num_experts]

    # 计算每个专家的负载（被选中的次数）
    load_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=selected_experts.device)
    for t in range(selected_experts.size(0)):
        for k in range(selected_experts.size(1)):
            expert_id = selected_experts[t, k].item()
            if expert_id >= 0:
                load_per_expert[expert_id] += 1

    # 过滤低频专家
    valid_mask = load_per_expert >= min_cnt
    if valid_mask.sum() < 2:
        return {
            'correlation': None,
            'reason': f'Only {valid_mask.sum()} experts have >= {min_cnt} selections'
        }

    # 计算相关性
    valid_scores = avg_score_per_expert[valid_mask]
    valid_loads = load_per_expert[valid_mask].float()

    # 标准化数据以获得更好的相关性估计
    scores_norm = (valid_scores - valid_scores.mean()) / (valid_scores.std() + 1e-6)
    loads_norm = (valid_loads - valid_loads.mean()) / (valid_loads.std() + 1e-6)

    # 计算皮尔逊相关系数
    correlation = torch.corrcoef(torch.stack([loads_norm, scores_norm]))[0, 1]

    # 统计平均得分接近未激活专家但负载量较高的专家
    # 未激活专家：负载为0的专家
    inactive_mask = load_per_expert == 0
    if inactive_mask.sum() > 0:
        inactive_avg_score = avg_score_per_expert[inactive_mask].mean().item()
        # 负载较高阈值：高于平均负载
        high_load_threshold_for_stat = valid_loads.mean()
        low_score_high_load_mask = (valid_scores < inactive_avg_score) & (valid_loads > high_load_threshold_for_stat)
        low_score_high_load_count = low_score_high_load_mask.sum().item()
        low_score_high_load_ratio = low_score_high_load_count / valid_mask.sum().item() if valid_mask.sum().item() > 0 else 0
    else:
        low_score_high_load_count = 0
        low_score_high_load_ratio = 0

    return {
        'correlation': correlation.item(),
        'correlation_interpretation': _interpret_correlation(correlation.item()),
        'low_score_high_load_ratio': low_score_high_load_ratio
    }


def _interpret_correlation(corr: float) -> str:
    """解释相关系数的含义"""
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        strength = "极弱"
    elif abs_corr < 0.3:
        strength = "弱"
    elif abs_corr < 0.5:
        strength = "中等"
    elif abs_corr < 0.7:
        strength = "强"
    else:
        strength = "极强"

    if corr > 0:
        direction = "正相关"
    else:
        direction = "负相关"

    return f"{strength}{direction}"

