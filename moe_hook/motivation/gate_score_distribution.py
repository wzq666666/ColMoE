import torch

def gate_score_distribution(
    topk_ids: torch.Tensor,        # [T, K]
    router_probs: torch.Tensor,    # [T, E]
    eps_ratio: float = 0.1          # 等价区间阈值（10%）
):
    """
    返回当前层 gate 分数分布的统计特征，用于判断是否存在“等价专家区间”
    """

    T, K = topk_ids.shape
    _, E = router_probs.shape

    # 1. 统计当前层所有被激活过的专家
    activated_experts = torch.unique(topk_ids.flatten())  # [E_active]

    # 2. 计算这些专家在所有 token 上的平均 gate 得分
    expert_scores = router_probs[:, activated_experts]    # [T, E_active]
    mean_scores = expert_scores.mean(dim=0)                # [E_active]

    # 3. 排序（从高到低）
    sorted_scores, _ = torch.sort(mean_scores, descending=True)

    # ========== 指标 1：分布集中程度 ==========
    mean = sorted_scores.mean()
    std = sorted_scores.std()
    cv = std / (mean + 1e-9)   # 变异系数（越小越平坦）

    # ========== 指标 2：Top-ratio ==========
    total_mass = sorted_scores.sum()
    top10_ratio = sorted_scores[:max(1, int(0.1 * len(sorted_scores)))].sum() / total_mass
    top20_ratio = sorted_scores[:max(1, int(0.2 * len(sorted_scores)))].sum() / total_mass
    top50_ratio = sorted_scores[:max(1, int(0.5 * len(sorted_scores)))].sum() / total_mass

    # ========== 指标 3：等价区间检测 ==========
    max_score = sorted_scores[0]
    equiv_mask = sorted_scores >= (1 - eps_ratio) * max_score
    equiv_ratio = equiv_mask.float().mean()

    # ========== 指标 4：Gini 系数（可选但很有说服力） ==========
    n = len(sorted_scores)
    index = torch.arange(1, n + 1, device=sorted_scores.device)
    gini = (2 * (index * sorted_scores).sum() / (n * total_mass)) - (n + 1) / n

    return {
        "num_activated_experts": len(activated_experts),
        "cv": cv.item(),
        "top10_ratio": top10_ratio.item(),
        "top20_ratio": top20_ratio.item(),
        "top50_ratio": top50_ratio.item(),
        "equiv_ratio": equiv_ratio.item(),
        "gini": gini.item(),
        "sorted_scores": sorted_scores.detach().cpu()
    }