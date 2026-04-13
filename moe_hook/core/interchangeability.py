import torch
import torch.nn.functional as F
from typing import Any, Dict

@torch.no_grad()
def expert_interchangeability(
    router_logits: torch.Tensor, # [T, E]
    e_src: int,
    e_dst: int,
    topk: int,
    *,
    min_tokens: int = 10,
    similarity_threshold: float = 0.8, # Cosine similarity threshold
    dominance_threshold: float = 0.5,  # Dst should be confident enough
    eps: float = 1e-6,
) -> Dict[str, Any]:

    assert router_logits.dim() == 2, "router_logits must be [T, E]"
    T, E = router_logits.shape
    assert 0 <= e_src < E and 0 <= e_dst < E
    
    # ---- router probabilities ----
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    
    # 1. 路由指纹相似度 (Routing Fingerprint Similarity)
    # 不看它们是否同时激活，看它们在整个数据集上的概率分布趋势是否一致
    # 如果两个专家的 Logits 向量余弦相似度极高，说明路由器对它们的看法是一致的
    
    probs_src = router_probs[:, e_src] # [T]
    probs_dst = router_probs[:, e_dst] # [T]
    
    # 计算余弦相似度 (Cosine Similarity of Probability Distributions)
    dot_product = (probs_src * probs_dst).sum()
    norm_src = probs_src.norm(p=2)
    norm_dst = probs_dst.norm(p=2)
    cos_sim = dot_product / (norm_src * norm_dst + eps)
    
    # 2. 目标覆盖能力 (Target Coverage)
    # 在 src 很重要的时候 (Top-K)，dst 是否也具备足够的置信度？
    topk_idx = torch.topk(router_logits, k=topk, dim=-1).indices  # [T, k]
    topk_mask = torch.zeros((T, E), dtype=torch.bool, device=router_logits.device)
    topk_mask.scatter_(1, topk_idx, True)

    src_mask = topk_mask[:, e_src]
    if src_mask.sum() < min_tokens:
        return {'can_replace': False, 'reason': 'low load on src expert'}

    # 在 src 激活的样本上，dst 的平均概率是否足够高？
    # 哪怕 dst 没进 Top-K，只要概率不低，说明它也能处理
    relative_confidence = (probs_dst[src_mask] / (probs_src[src_mask] + eps)).mean().item()
    
    # 相似性
    interchangeability_score = cos_sim * relative_confidence

    # 判定逻辑
    can_replace = (
        cos_sim.item() > similarity_threshold 
        and relative_confidence > dominance_threshold
    )

    return {
        'interchangeability_score': interchangeability_score,
        'can_replace': can_replace
    }