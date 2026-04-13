import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

def token_level_gpu_preferred_reroute(
    router_probs: torch.Tensor,   # [T, E]
    topk_ids: torch.Tensor,        # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,    # [T, k] - 对应的权重（可选，用于后续计算）
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    alpha: float = 0.2,
    eps: float = 1e-6,
):
    """
    对top-k中的每个CPU专家，尝试用GPU专家替换（向量化版本）。
    
    关键约束：每个 GPU 专家在每个 token 的 topk 中最多只能出现一次。
    这避免了多个 CPU 专家被重路由到同一个 GPU 专家导致的信息损失。
    
    Returns:
        new_topk_ids: [T, k] - 重路由后的专家ID
        reroute_mask: [T, k] - 每个位置是否被重路由
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device
    
    new_topk_ids = topk_ids.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)
    
    if len(gpu_expert_ids) == 0 or len(cpu_expert_ids) == 0:
        return new_topk_ids, reroute_mask
    
    # ========== Phase 1: 向量化计算最佳 GPU 替换候选 ==========
    gpu_tensor = torch.tensor(gpu_expert_ids, dtype=torch.long, device=device)  # [G]
    cpu_tensor = torch.tensor(cpu_expert_ids, dtype=torch.long, device=device)  # [C]
    
    # 标记 CPU 位置
    cpu_pos_mask = torch.isin(topk_ids, cpu_tensor)  # [T, k]
    
    # S_A[t, j] = router_probs[t, topk_ids[t, j]]
    S_A = torch.gather(router_probs, 1, topk_ids.long())  # [T, k]
    
    # gpu_probs[t, g] = router_probs[t, gpu_tensor[g]]
    gpu_probs = router_probs[:, gpu_tensor]  # [T, G]
    
    # rel_gap[t, j, g] = |gpu_probs[t,g] - S_A[t,j]| / (S_A[t,j] + eps)
    rel_gap = torch.abs(gpu_probs.unsqueeze(1) - S_A.unsqueeze(2)) / (S_A.unsqueeze(2) + eps)  # [T, k, G]
    rel_gap[~cpu_pos_mask] = float('inf')
    rel_gap[rel_gap > alpha] = float('inf')
    
    # 排除已在 topk 中的 GPU 专家（维持唯一性约束）
    # gpu_in_topk[t, j, g] = True if gpu_tensor[g] 已出现在 topk_ids[t] 中
    gpu_in_topk = (topk_ids.unsqueeze(2) == gpu_tensor.unsqueeze(0).unsqueeze(0))  # [T, k, G]
    gpu_already_used = gpu_in_topk.any(dim=1, keepdim=True).expand_as(rel_gap)  # [T, 1, G] -> [T, k, G]
    rel_gap[gpu_already_used] = float('inf')
    
    # 找每个 (token, position) 的最佳 GPU 替换
    best_gap, best_gpu_idx = rel_gap.min(dim=2)  # [T, k]
    best_gpu_expert = gpu_tensor[best_gpu_idx]  # [T, k]
    valid_candidate = (best_gap <= alpha) & cpu_pos_mask  # [T, k]
    
    # ========== Phase 2: 应用唯一性约束（每个 GPU 专家只能用一次）==========
    # 需要逐 token 顺序处理，但使用批量 CPU 数据避免逐元素 GPU 同步
    if not valid_candidate.any().item():
        return new_topk_ids, reroute_mask
    
    # 单次 GPU→CPU 传输
    best_gap_cpu = best_gap.cpu().numpy()
    best_gpu_expert_cpu = best_gpu_expert.cpu().numpy()
    valid_cpu = valid_candidate.cpu().numpy()
    topk_ids_cpu = new_topk_ids.cpu().numpy()
    
    result_ids = topk_ids_cpu.copy()
    result_mask = np.zeros((T, k), dtype=bool)
    
    for i in range(T):
        valid_positions = np.where(valid_cpu[i])[0]
        if len(valid_positions) == 0:
            continue
        
        # 按 gap 排序
        gaps = best_gap_cpu[i, valid_positions]
        sorted_positions = valid_positions[np.argsort(gaps)]
        
        used_experts = set(result_ids[i].tolist())
        
        for pos in sorted_positions:
            from_expert = topk_ids_cpu[i, pos]
            to_expert = best_gpu_expert_cpu[i, pos]
            
            # 唯一性约束：该 GPU 专家不能已在当前 topk 中
            if to_expert in used_experts:
                continue
            
            result_ids[i, pos] = to_expert
            result_mask[i, pos] = True
            used_experts.discard(from_expert)
            used_experts.add(to_expert)
    
    # 单次 CPU→GPU 传输
    new_topk_ids = torch.tensor(result_ids, dtype=topk_ids.dtype, device=device)
    reroute_mask = torch.tensor(result_mask, dtype=torch.bool, device=device)
    
    return new_topk_ids, reroute_mask


def token_level_gpu_preferred_reroute_with_duplicate(
    router_probs: torch.Tensor,   # [T, E]
    topk_ids: torch.Tensor,        # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,    # [T, k] - 对应的权重
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    alpha: float = 0.2,
    eps: float = 1e-6,
):
    """
    允许重复专家的GPU优先重路由（向量化版本）。
    
    与 token_level_gpu_preferred_reroute 不同，此方法：
    1. 允许多个 CPU 专家被替换成同一个 GPU 专家（产生重复）
    2. 保持原始权重不变（专家功能相似，权重代表位置重要性）
    3. 后续需要调用 merge_duplicate_experts_with_weights 来合并重复专家
    
    Returns:
        new_topk_ids: [T, k] - 重路由后的专家ID（可能包含重复）
        new_topk_weights: [T, k] - 重路由后的权重（保持原始权重）
        reroute_mask: [T, k] - 每个位置是否被重路由
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device
    
    new_topk_ids = topk_ids.clone()
    new_topk_weights = topk_weights.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)
    
    if len(gpu_expert_ids) == 0 or len(cpu_expert_ids) == 0:
        return new_topk_ids, new_topk_weights, reroute_mask
    
    # ========== 全向量化：允许重复时无状态约束，可完全在 GPU 上完成 ==========
    gpu_tensor = torch.tensor(gpu_expert_ids, dtype=torch.long, device=device)  # [G]
    cpu_tensor = torch.tensor(cpu_expert_ids, dtype=torch.long, device=device)  # [C]
    
    # 标记 CPU 位置
    cpu_pos_mask = torch.isin(topk_ids, cpu_tensor)  # [T, k]
    
    # S_A[t, j] = router_probs[t, topk_ids[t, j]]
    S_A = torch.gather(router_probs, 1, topk_ids.long())  # [T, k]
    
    # gpu_probs[t, g] = router_probs[t, gpu_tensor[g]]
    gpu_probs = router_probs[:, gpu_tensor]  # [T, G]
    
    # rel_gap[t, j, g] = |gpu_probs[t,g] - S_A[t,j]| / (S_A[t,j] + eps)
    rel_gap = torch.abs(gpu_probs.unsqueeze(1) - S_A.unsqueeze(2)) / (S_A.unsqueeze(2) + eps)  # [T, k, G]
    rel_gap[~cpu_pos_mask] = float('inf')
    rel_gap[rel_gap > alpha] = float('inf')
    
    # 找每个 (token, position) 的最佳 GPU 替换（允许重复，无状态约束）
    best_gap, best_gpu_idx = rel_gap.min(dim=2)  # [T, k]
    best_gpu_expert = gpu_tensor[best_gpu_idx]  # [T, k]
    valid_candidate = (best_gap <= alpha) & cpu_pos_mask  # [T, k]
    
    # 直接在 GPU 上执行替换（无需 CPU roundtrip）
    new_topk_ids[valid_candidate] = best_gpu_expert[valid_candidate]
    reroute_mask = valid_candidate
    # 权重保持不变
    
    return new_topk_ids, new_topk_weights, reroute_mask


def token_level_gpu_preferred_reroute_with_limits(
    router_probs: torch.Tensor,   # [T, E]
    topk_ids: torch.Tensor,        # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,    # [T, k] - 对应的权重
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    alpha: float = 0.2,
    eps: float = 1e-6,
    max_duplicates_per_expert: int = 2,  # 每个专家最多重复次数
    min_unique_experts: int = 4,  # 每个token至少保持的唯一专家数
):
    """
    带重复控制的GPU优先重路由（向量化版本）。
    
    关键改进：
    1. 限制每个专家的最大重复次数
    2. 确保每个token保持最小数量的唯一专家
    3. 优先重路由概率差距最小的专家对
    4. 保持原始权重不变（专家功能相似，权重代表位置重要性）
    5. Phase-1 全量向量化：GPU 上批量计算所有候选替换，零 Python 循环
    
    Args:
        max_duplicates_per_expert: 单个专家在一个token中的最大出现次数
        min_unique_experts: 每个token必须保持的最小唯一专家数量
    
    Returns:
        new_topk_ids: [T, k] - 重路由后的专家ID
        new_topk_weights: [T, k] - 重路由后的权重（保持原始）
        reroute_mask: [T, k] - 重路由标记
        reroute_stats: dict - 重路由统计信息
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device
    
    new_topk_ids = topk_ids.clone()
    new_topk_weights = topk_weights.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)
    
    if len(gpu_expert_ids) == 0 or len(cpu_expert_ids) == 0:
        reroute_stats = {
            'total_reroutes': 0,
            'blocked_by_duplicate_limit': 0,
            'blocked_by_unique_limit': 0,
            'reroute_rate': 0.0,
        }
        return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats
    
    # ========== Phase 1: 向量化计算最佳 GPU 替换候选 ==========
    # 全部在 GPU 上完成，零 Python 循环
    
    gpu_tensor = torch.tensor(gpu_expert_ids, dtype=torch.long, device=device)  # [G]
    cpu_tensor = torch.tensor(cpu_expert_ids, dtype=torch.long, device=device)  # [C]
    G = gpu_tensor.shape[0]
    
    # 标记 topk 中哪些位置是 CPU 专家 → 需要重路由的候选
    cpu_pos_mask = torch.isin(topk_ids, cpu_tensor)  # [T, k] bool
    
    # 获取每个 topk 位置对应的路由概率
    # topk_ids: [T, k], router_probs: [T, E]
    # S_A[t, j] = router_probs[t, topk_ids[t, j]]
    S_A = torch.gather(router_probs, 1, topk_ids.long())  # [T, k]
    
    # 获取所有 GPU 专家的路由概率
    # gpu_probs[t, g] = router_probs[t, gpu_tensor[g]]
    gpu_probs = router_probs[:, gpu_tensor]  # [T, G]
    
    # 计算相对差距矩阵: rel_gap[t, j, g] = |gpu_probs[t,g] - S_A[t,j]| / (S_A[t,j] + eps)
    # S_A: [T, k] → [T, k, 1], gpu_probs: [T, G] → [T, 1, G]
    rel_gap = torch.abs(gpu_probs.unsqueeze(1) - S_A.unsqueeze(2)) / (S_A.unsqueeze(2) + eps)  # [T, k, G]
    
    # 对非 CPU 位置，设置 gap 为 inf（不参与替换）
    rel_gap[~cpu_pos_mask] = float('inf')
    
    # 对超过 alpha 阈值的，设置 gap 为 inf
    rel_gap[rel_gap > alpha] = float('inf')
    
    # 找每个 (token, position) 的最佳 GPU 替换
    best_gap, best_gpu_idx = rel_gap.min(dim=2)  # [T, k] gap 值, [T, k] gpu 索引
    best_gpu_expert = gpu_tensor[best_gpu_idx]  # [T, k] 实际专家 ID
    
    # 标记有效候选：gap 在 alpha 内且是 CPU 位置
    valid_candidate = (best_gap <= alpha) & cpu_pos_mask  # [T, k] bool
    
    # ========== Phase 2: 应用约束（重复限制 + 唯一性限制）==========
    # 约束是逐 token 有状态的，需要按 gap 排序后顺序应用
    # 但避免 per-element .item() 调用：批量传输到 CPU 后用 numpy 处理
    
    # 快速路径：无有效候选
    if not valid_candidate.any().item():
        reroute_stats = {
            'total_reroutes': 0,
            'blocked_by_duplicate_limit': 0,
            'blocked_by_unique_limit': 0,
            'reroute_rate': 0.0,
        }
        return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats
    
    # 单次 GPU→CPU 传输所有需要的数据
    # 打包传输避免多次 GPU 同步
    best_gap_cpu = best_gap.cpu().numpy()            # [T, k]
    best_gpu_expert_cpu = best_gpu_expert.cpu().numpy()  # [T, k]
    valid_candidate_cpu = valid_candidate.cpu().numpy()  # [T, k]
    topk_ids_cpu = new_topk_ids.cpu().numpy()           # [T, k]
    
    # 统计信息
    total_reroutes = 0
    blocked_by_duplicate_limit = 0
    blocked_by_unique_limit = 0
    
    # 预分配结果数组（CPU numpy）
    result_ids = topk_ids_cpu.copy()
    result_mask = np.zeros((T, k), dtype=bool)
    
    for i in range(T):
        # 找该 token 所有有效候选位置
        valid_positions = np.where(valid_candidate_cpu[i])[0]
        if len(valid_positions) == 0:
            continue
        
        # 按 gap 排序（小 gap 优先）
        gaps = best_gap_cpu[i, valid_positions]
        sorted_order = np.argsort(gaps)
        sorted_positions = valid_positions[sorted_order]
        
        # 初始化专家计数（使用 numpy bincount 替代 Counter）
        expert_count = Counter()
        for val in result_ids[i]:
            if val >= 0:
                expert_count[val] += 1
        
        for pos in sorted_positions:
            from_expert = topk_ids_cpu[i, pos]
            to_expert = best_gpu_expert_cpu[i, pos]
            
            # 检查重复限制
            if expert_count.get(to_expert, 0) >= max_duplicates_per_expert:
                blocked_by_duplicate_limit += 1
                continue
            
            # 检查唯一性限制：替换后的唯一专家数
            # 如果 from_expert 在其他位置还有出现，唯一数不变
            # 如果 from_expert 只出现一次，唯一数 -1（除非 to_expert 是新的则 ±0）
            from_count = expert_count.get(from_expert, 0)
            to_count = expert_count.get(to_expert, 0)
            unique_delta = 0
            if from_count <= 1:  # from_expert 将消失
                unique_delta -= 1
            if to_count == 0:  # to_expert 是新增
                unique_delta += 1
            current_unique = len(expert_count)
            if current_unique + unique_delta < min_unique_experts:
                blocked_by_unique_limit += 1
                continue
            
            # 执行替换
            result_ids[i, pos] = to_expert
            result_mask[i, pos] = True
            
            # 更新计数
            expert_count[from_expert] -= 1
            if expert_count[from_expert] == 0:
                del expert_count[from_expert]
            expert_count[to_expert] = expert_count.get(to_expert, 0) + 1
            
            total_reroutes += 1
    
    # ========== Phase 3: 批量写回 GPU ==========
    # 单次 CPU→GPU 传输
    new_topk_ids = torch.tensor(result_ids, dtype=topk_ids.dtype, device=device)
    reroute_mask = torch.tensor(result_mask, dtype=torch.bool, device=device)
    
    # 统计信息
    reroute_stats = {
        'total_reroutes': total_reroutes,
        'blocked_by_duplicate_limit': blocked_by_duplicate_limit,
        'blocked_by_unique_limit': blocked_by_unique_limit,
        'reroute_rate': total_reroutes / (T * k) if T * k > 0 else 0.0
    }
    
    return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats


def expert_level_io_free_reroute(
    router_probs: torch.Tensor,   # [T, E]
    topk_ids: torch.Tensor,        # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,    # [T, k] - 对应的权重
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    alpha: float = 0.2,
    eps: float = 1e-6,
    score_threshold_ratio: float = 0.5,
):
    """
    IO-free 专家级替换策略 —— 只替换低分 CPU 专家。
    
    与 token_level 重路由的区别：
    - token_level: 逐 token 逐位置决策，每个 token 独立选择替换目标
    - expert_level (本函数): 逐专家决策，一个 CPU 专家的所有 token 统一替换到同一个 GPU 专家
    
    关键改进（仅替换低分专家）：
    - 高分 CPU 专家对推理结果影响大，不应被替换，宁可走 CPU 计算
    - 低分 CPU 专家影响小，可以用评分相近的 GPU 专家代替，减少 CPU 开销
    - 阈值：score < median(all_activated_expert_scores) * score_threshold_ratio
    
    策略：
    1. 找出所有被激活的 CPU 专家
    2. 计算激活专家的评分中位数，设定低分阈值
    3. 只对低分 CPU 专家寻找评分相近的未激活 GPU 专家（阈值 α 内）
    4. 将该 CPU 专家的所有 token 统一替换到该 GPU 专家
    5. 高分 CPU 专家保留不动（在 CPU 上执行）
    
    Args:
        router_probs: [T, E] 路由概率
        topk_ids: [T, k] 原始 top-k 专家 ID
        topk_weights: [T, k] 对应权重
        gpu_expert_ids: 当前在 GPU 上的专家列表
        cpu_expert_ids: 当前在 CPU 上的专家列表
        alpha: 评分相似性阈值（相对差距）
        eps: 避免除零的小常数
        score_threshold_ratio: 低分阈值比例，score < median * ratio 视为低分专家
    
    Returns:
        new_topk_ids: [T, k] 替换后的专家 ID
        new_topk_weights: [T, k] 权重（保持原始）
        reroute_mask: [T, k] 标记哪些位置被替换
        reroute_stats: dict 统计信息
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device

    gpu_set = set(gpu_expert_ids)
    cpu_set = set(cpu_expert_ids)

    new_topk_ids = topk_ids.clone()
    new_topk_weights = topk_weights.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)

    # Step 1: 找出所有被激活的专家（出现在 topk 中的）
    all_activated = set(topk_ids.view(-1).cpu().tolist())
    all_activated.discard(-1)  # 去除 padding

    # Step 2: 被激活的 CPU 专家 & 未被激活的 GPU 专家
    activated_cpu = sorted(all_activated & cpu_set)
    inactive_gpu = sorted(gpu_set - all_activated)

    if not activated_cpu or not inactive_gpu:
        reroute_stats = {
            'total_reroutes': 0,
            'expert_replacements': 0,
            'activated_cpu_experts': len(activated_cpu),
            'inactive_gpu_experts': len(inactive_gpu),
            'low_score_cpu_experts': 0,
            'high_score_skipped': 0,
            'replacement_details': {},
            'reroute_rate': 0.0,
            'blocked_by_duplicate_limit': 0,
            'blocked_by_unique_limit': 0,
        }
        return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats

    # Step 3: 计算每个激活专家的平均路由概率
    all_scores = {}
    for eid in all_activated:
        all_scores[eid] = router_probs[:, eid].mean().item()

    cpu_scores = {eid: all_scores[eid] for eid in activated_cpu}
    gpu_scores = {eid: router_probs[:, eid].mean().item() for eid in inactive_gpu}

    # Step 4: 计算低分阈值 = 所有激活专家评分中位数 * score_threshold_ratio
    score_values = sorted(all_scores.values())
    median_score = score_values[len(score_values) // 2]
    low_score_threshold = median_score * score_threshold_ratio

    # Step 5: 筛选低分 CPU 专家（只替换这些）
    low_score_cpu = [(eid, s) for eid, s in cpu_scores.items() if s < low_score_threshold]
    low_score_cpu.sort(key=lambda x: x[1])  # 按评分升序，最低分优先替换
    high_score_skipped = len(activated_cpu) - len(low_score_cpu)

    # Step 6: 逐个低分 CPU 专家寻找最佳 GPU 替换
    total_reroutes = 0
    expert_replacements = {}  # cpu_expert -> gpu_expert
    used_gpu = set()

    for cpu_eid, cpu_score in low_score_cpu:
        best_gpu = None
        best_gap = float('inf')

        for gpu_eid, gpu_score in gpu_scores.items():
            if gpu_eid in used_gpu:
                continue

            rel_gap = abs(gpu_score - cpu_score) / (cpu_score + eps)
            if rel_gap <= alpha and rel_gap < best_gap:
                best_gap = rel_gap
                best_gpu = gpu_eid

        if best_gpu is not None:
            expert_replacements[cpu_eid] = best_gpu
            used_gpu.add(best_gpu)

    # Step 7: 批量执行替换
    for cpu_eid, gpu_eid in expert_replacements.items():
        mask = (topk_ids == cpu_eid)
        new_topk_ids[mask] = gpu_eid
        reroute_mask |= mask
        total_reroutes += mask.sum().item()

    # Step 8: 统计
    reroute_stats = {
        'total_reroutes': total_reroutes,
        'expert_replacements': len(expert_replacements),
        'activated_cpu_experts': len(activated_cpu),
        'inactive_gpu_experts': len(inactive_gpu),
        'low_score_cpu_experts': len(low_score_cpu),
        'high_score_skipped': high_score_skipped,
        'low_score_threshold': low_score_threshold,
        'median_score': median_score,
        'replacement_details': {str(k): v for k, v in expert_replacements.items()},
        'reroute_rate': total_reroutes / (T * k) if T * k > 0 else 0.0,
        # 兼容 token_level 统计格式
        'blocked_by_duplicate_limit': 0,
        'blocked_by_unique_limit': 0,
    }

    return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats


def expert_level_low_score_reroute(
    router_probs: torch.Tensor,   # [T, E]
    topk_ids: torch.Tensor,        # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,    # [T, k] - 对应的权重
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    alpha: float = 0.2,
    eps: float = 1e-6,
):
    """
    粗粒度专家级别低分重路由策略（用于对比实验）。
    
    关键特点：
    1. 低分CPU专家：S_{k+1} < Score(e) < (1+α)S_{k+1}
    2. 可替换GPU专家：(1-α)S_{k+1} < Score(e) ≤ S_{k+1}
    3. 专家级别决策：一个CPU专家的所有token统一替换到同一个GPU专家
    4. 允许重复：多个CPU专家可以被替换成同一个GPU专家
    5. 合并重复专家，保持原始权重
    
    低分专家判定：
    - 对每个token计算S_{k+1}（第k+1个专家的路由概率）
    - 如果某个CPU专家在某个token中满足：S_{k+1} < Score(e) < (1+α)S_{k+1}
    - 则该专家在该token中被视为低分专家
    - 统计该专家在所有token中作为低分专家的频率，频率高的优先替换
    
    Args:
        router_probs: [T, E] 路由概率
        topk_ids: [T, k] 原始 top-k 专家 ID
        topk_weights: [T, k] 对应权重
        gpu_expert_ids: 当前在 GPU 上的专家列表
        cpu_expert_ids: 当前在 CPU 上的专家列表
        alpha: 低分阈值参数
        eps: 避免除零的小常数
    
    Returns:
        new_topk_ids: [T, k] 替换后的专家 ID
        new_topk_weights: [T, k] 权重（保持原始）
        reroute_mask: [T, k] 标记哪些位置被替换
        reroute_stats: dict 统计信息
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device

    gpu_set = set(gpu_expert_ids)
    cpu_set = set(cpu_expert_ids)

    new_topk_ids = topk_ids.clone()
    new_topk_weights = topk_weights.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)

    # Step 1: 计算所有专家的平均得分（向量化），单次 GPU→CPU 传输
    expert_scores_gpu = router_probs.mean(dim=0)  # [E], GPU
    expert_scores_cpu = expert_scores_gpu.cpu().numpy()  # [E], CPU — 唯一的 GPU→CPU 传输
    
    # Step 2: 找出所有被激活的专家（GPU 上用 bincount 避免 tolist）
    flat_ids = topk_ids.view(-1)
    activated_mask_vec = torch.zeros(E, dtype=torch.bool, device=device)
    # bincount 只接受非负输入，过滤 -1
    valid_flat = flat_ids[flat_ids >= 0]
    if valid_flat.numel() > 0:
        counts = torch.bincount(valid_flat, minlength=E)
        activated_mask_vec = counts > 0
    all_activated_tensor = torch.nonzero(activated_mask_vec, as_tuple=False).view(-1)
    all_activated = set(all_activated_tensor.cpu().tolist())  # 复用同一次 cpu 传输

    activated_cpu = sorted(all_activated & cpu_set)
    inactive_gpu = sorted(gpu_set - all_activated)

    if not activated_cpu or not inactive_gpu:
        reroute_stats = {
            'total_reroutes': 0,
            'expert_replacements': 0,
            'activated_cpu_experts': len(activated_cpu),
            'inactive_gpu_experts': len(inactive_gpu),
            'low_score_cpu_experts': 0,
            'replaceable_gpu_experts': 0,
            'replacement_details': {},
            'reroute_rate': 0.0,
            'strategy': 'expert_level_low_score',
        }
        return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats

    # Step 3: 使用 CPU 侧 numpy 数组获取 s_kplus1（零 GPU 同步）
    # np.partition 比 torch.topk 更适合 CPU 场景（O(E) vs O(E log k)）
    s_kplus1 = float(np.partition(expert_scores_cpu, -k-1)[-k-1])
    
    threshold_lower = s_kplus1
    threshold_upper = (1 + alpha) * s_kplus1
    replaceable_lower = (1 - alpha) * s_kplus1
    replaceable_upper = s_kplus1

    # Step 4: 筛选低分CPU专家和可替换GPU专家（全 CPU numpy，零 GPU 同步）
    activated_cpu_arr = np.array(activated_cpu)
    activated_cpu_scores = expert_scores_cpu[activated_cpu_arr]
    low_mask = (activated_cpu_scores > threshold_lower) & (activated_cpu_scores < threshold_upper)
    low_cpu_eids = activated_cpu_arr[low_mask]
    low_cpu_scores = activated_cpu_scores[low_mask]
    # 按得分升序排序
    sort_idx = np.argsort(low_cpu_scores)
    low_cpu_eids = low_cpu_eids[sort_idx]
    
    inactive_gpu_arr = np.array(inactive_gpu)
    inactive_gpu_scores = expert_scores_cpu[inactive_gpu_arr]
    repl_mask = (inactive_gpu_scores > replaceable_lower) & (inactive_gpu_scores <= replaceable_upper)
    repl_gpu_eids = inactive_gpu_arr[repl_mask]
    repl_gpu_scores = inactive_gpu_scores[repl_mask]
    sort_idx_gpu = np.argsort(repl_gpu_scores)
    repl_gpu_eids = repl_gpu_eids[sort_idx_gpu]

    # Step 5: 依次分配 GPU 替换（线性指针，O(1) per assignment）
    expert_replacements = {}
    gpu_ptr = 0
    for cpu_eid in low_cpu_eids:
        if gpu_ptr >= len(repl_gpu_eids):
            break
        expert_replacements[int(cpu_eid)] = int(repl_gpu_eids[gpu_ptr])
        gpu_ptr += 1

    # Step 6: 向量化批量替换（构建 lookup table，单次 GPU kernel）
    if expert_replacements:
        # 构建映射表：remap[e] = gpu_replacement if e in replacements, else e
        remap_table = torch.arange(E, dtype=topk_ids.dtype, device=device)
        for cpu_eid, gpu_eid in expert_replacements.items():
            remap_table[cpu_eid] = gpu_eid
        
        # 单次向量化替换（无循环，无逐元素 GPU 同步）
        new_topk_ids = remap_table[topk_ids.long()]
        reroute_mask = (new_topk_ids != topk_ids)
        
        # 统计（GPU reduce 一次）
        total_reroutes = reroute_mask.sum().item()
        rerouted_tokens = reroute_mask.any(dim=1).sum().item()
    else:
        total_reroutes = 0
        rerouted_tokens = 0
    
    # Step 7: 合并重复专家（保持原始权重）
    # new_topk_ids, new_topk_weights = merge_duplicate_experts_with_weights(
    #     new_topk_ids, new_topk_weights, E
    # )

    # Step 8: 统计
    reroute_stats = {
        'total_reroutes': total_reroutes,  # 重路由的位置数量
        'rerouted_tokens': rerouted_tokens,  # 重路由的token数量
        'expert_replacements': len(expert_replacements),
        'activated_cpu_experts': len(activated_cpu),
        'inactive_gpu_experts': len(inactive_gpu),
        'low_score_cpu_experts': len(low_cpu_eids),
        'replaceable_gpu_experts': len(repl_gpu_eids),
        's_kplus1': s_kplus1,
        'threshold_range': (threshold_lower, threshold_upper),
        'replaceable_range': (replaceable_lower, replaceable_upper),
        'replacement_details': {str(k): v for k, v in expert_replacements.items()},
        'position_reroute_rate': total_reroutes / (T * k) if T * k > 0 else 0.0,  # 位置重路由率
        'token_reroute_rate': rerouted_tokens / T if T > 0 else 0.0,  # token重路由率
        'strategy': 'expert_level_low_score',
    }

    return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats


def token_level_low_score_reroute(
    router_probs: torch.Tensor,   # [T, E]
    topk_ids: torch.Tensor,        # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,    # [T, k] - 对应的权重
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    alpha: float = 0.2,
    eps: float = 1e-6,
):
    """
    细粒度token级别低分重路由策略（用于对比实验）。
    
    关键特点：
    1. 低分CPU专家：S_{k+1} < Score(e) < (1+α)S_{k+1}
    2. 可替换GPU专家：(1-α)S_{k+1} < Score(e) ≤ S_{k+1}
    3. token级别决策：每个token独立决策替换目标
    4. 允许重复：同一个GPU专家可以在一个token中多次出现
    5. 合并重复专家，保持原始权重
    
    与粗粒度方法的区别：
    - 粗粒度：一个CPU专家的所有token统一替换到同一个GPU专家
    - 细粒度：每个token独立选择替换目标，同一个CPU专家在不同token可能被替换到不同GPU专家
    - 细粒度方法理论上能迁移更多负载到GPU（更灵活的替换策略）
    
    Args:
        router_probs: [T, E] 路由概率
        topk_ids: [T, k] 原始 top-k 专家 ID
        topk_weights: [T, k] 对应权重
        gpu_expert_ids: 当前在 GPU 上的专家列表
        cpu_expert_ids: 当前在 CPU 上的专家列表
        alpha: 低分阈值参数
        eps: 避免除零的小常数
    
    Returns:
        new_topk_ids: [T, k] 替换后的专家 ID
        new_topk_weights: [T, k] 权重（保持原始）
        reroute_mask: [T, k] 标记哪些位置被替换
        reroute_stats: dict 统计信息
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device

    gpu_set = set(gpu_expert_ids)
    cpu_set = set(cpu_expert_ids)

    new_topk_ids = topk_ids.clone()
    new_topk_weights = topk_weights.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)

    # Step 1: 使用torch.topk获取每个token的第k+1个专家得分（向量化）
    # topk返回top-k，我们需要top-(k+1)
    topk_plus_1_values, _ = torch.topk(router_probs, k + 1, dim=1, sorted=True)
    s_kplus1 = topk_plus_1_values[:, k]  # [T] - 每个token的第k+1个专家得分
    
    # 计算阈值（向量化）
    threshold_lower = s_kplus1  # [T]
    threshold_upper = (1 + alpha) * s_kplus1  # [T]
    replaceable_lower = (1 - alpha) * s_kplus1  # [T]
    replaceable_upper = s_kplus1  # [T]
    
    # Step 2: 向量化判断每个位置的专家得分
    topk_scores = torch.gather(router_probs, 1, topk_ids)  # [T, k]
    
    # 判断每个位置是否为CPU专家 [T, k]
    assert cpu_set.isdisjoint(gpu_set)

    cpu_set = set(cpu_expert_ids)
    is_cpu = torch.isin(topk_ids, torch.tensor(cpu_expert_ids, device=device))
    
    # 判断每个位置是否为低分专家 [T, k]
    is_low_score = is_cpu & (topk_scores > threshold_lower.unsqueeze(1)) & (topk_scores < threshold_upper.unsqueeze(1))
    
    # Step 3: 对每个GPU专家计算其在每个token的得分 [T, num_gpu]
    gpu_tensor = torch.tensor(gpu_expert_ids, device=device).unsqueeze(0).expand(T, -1)  # [T, num_gpu]
    gpu_scores = torch.gather(router_probs, 1, gpu_tensor)  # [T, num_gpu]
    
    # 判断每个GPU专家是否为可替换专家 [T, num_gpu]
    is_replaceable = (gpu_scores > replaceable_lower.unsqueeze(1)) & (gpu_scores <= replaceable_upper.unsqueeze(1))
    
    # Step 4: 批量处理替换（避免逐元素 GPU 同步）
    # 快速路径：无低分候选
    if not is_low_score.any().item():
        rerouted_tokens = 0
        reroute_stats = {
            'total_reroutes': 0,
            'rerouted_tokens': 0,
            'low_score_candidates': 0,
            'successful_reroutes': 0,
            'replaceable_gpu_found': 0,
            'position_reroute_rate': 0.0,
            'token_reroute_rate': 0.0,
            'success_rate': 0.0,
            'strategy': 'token_level_low_score',
        }
        return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats
    
    # 单次 GPU→CPU 批量传输所有需要的数据
    is_low_score_cpu = is_low_score.cpu().numpy()        # [T, k]
    is_replaceable_cpu = is_replaceable.cpu().numpy()     # [T, num_gpu]
    topk_ids_cpu = topk_ids.cpu().numpy()                 # [T, k]
    gpu_expert_ids_arr = np.array(gpu_expert_ids)         # [num_gpu]
    
    # 预分配 CPU 结果数组
    result_ids = topk_ids_cpu.copy()                      # [T, k]
    result_mask = np.zeros((T, k), dtype=bool)            # [T, k]
    
    total_reroutes = 0
    low_score_candidates = 0
    successful_reroutes = 0
    replaceable_gpu_found = 0
    
    for i in range(T):
        # numpy 向量化查找低分位置
        low_positions = np.where(is_low_score_cpu[i])[0]
        if len(low_positions) == 0:
            continue
        
        low_score_candidates += len(low_positions)
        
        # numpy 向量化查找可替换 GPU 专家
        repl_indices = np.where(is_replaceable_cpu[i])[0]
        if len(repl_indices) == 0:
            continue
        
        replaceable_gpu_found += len(repl_indices)
        replaceable_eids = gpu_expert_ids_arr[repl_indices]  # numpy 索引，零拷贝
        
        # 依次分配（每个 GPU 专家最多用一次）
        gpu_ptr = 0  # 线性扫描指针替代 set 查找
        
        for pos in low_positions:
            if gpu_ptr >= len(replaceable_eids):
                break  # GPU 专家用完
            
            result_ids[i, pos] = replaceable_eids[gpu_ptr]
            result_mask[i, pos] = True
            gpu_ptr += 1
            total_reroutes += 1
            successful_reroutes += 1
    
    # 单次 CPU→GPU 批量写回
    new_topk_ids = torch.tensor(result_ids, dtype=topk_ids.dtype, device=device)
    reroute_mask = torch.tensor(result_mask, dtype=torch.bool, device=device)
    
    # 统计被重路由的token数量（复用 CPU 侧 result_mask，无需 GPU roundtrip）
    rerouted_tokens = int(result_mask.any(axis=1).sum())

    # 统计信息
    reroute_stats = {
        'total_reroutes': total_reroutes,  # 重路由的位置数量
        'rerouted_tokens': rerouted_tokens,  # 重路由的token数量
        'low_score_candidates': low_score_candidates,
        'successful_reroutes': successful_reroutes,
        'replaceable_gpu_found': replaceable_gpu_found,
        'position_reroute_rate': total_reroutes / (T * k) if T * k > 0 else 0.0,  # 位置重路由率
        'token_reroute_rate': rerouted_tokens / T if T > 0 else 0.0,  # token重路由率
        'success_rate': successful_reroutes / low_score_candidates if low_score_candidates > 0 else 0.0,
        'strategy': 'token_level_low_score',
    }

    return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats


def analyze_expert_diversity(topk_ids: torch.Tensor):
    """
    分析专家选择的多样性。
    
    Returns:
        diversity_stats: 包含各种多样性指标的字典
    """
    T, k = topk_ids.shape
    
    unique_counts = []
    duplicate_ratios = []
    
    for i in range(T):
        valid_ids = topk_ids[i][topk_ids[i] >= 0]
        if len(valid_ids) > 0:
            unique_experts = len(set(valid_ids.tolist()))
            unique_counts.append(unique_experts)
            
            # 重复率 = (总位置数 - 唯一专家数) / 总位置数
            duplicate_ratio = (len(valid_ids) - unique_experts) / len(valid_ids)
            duplicate_ratios.append(duplicate_ratio)
    
    if unique_counts:
        return {
            'avg_unique_experts': sum(unique_counts) / len(unique_counts),
            'min_unique_experts': min(unique_counts),
            'max_unique_experts': max(unique_counts),
            'avg_duplicate_ratio': sum(duplicate_ratios) / len(duplicate_ratios),
            'max_duplicate_ratio': max(duplicate_ratios),
        }
    else:
        return {
            'avg_unique_experts': 0.0,
            'min_unique_experts': 0,
            'max_unique_experts': 0,
            'avg_duplicate_ratio': 0.0,
            'max_duplicate_ratio': 0.0,
        }


def merge_duplicate_experts_with_weights(new_topk_ids, new_topk_weights, expert_num):
    """
    合并重复专家并正确处理权重。
    
    假设 new_topk_weights 已经被同步更新（与 new_topk_ids 对应）。
    
    处理逻辑：
    1. 找到重复的专家
    2. 累加重复专家的权重
    3. 保留唯一专家，按权重降序排序
    4. 重新归一化权重
    
    Args:
        new_topk_ids: [T, k] 可能包含重复专家的topk IDs
        new_topk_weights: [T, k] 已同步更新的权重（与 new_topk_ids 对应）
        expert_num: 专家总数
    
    Returns:
        result_ids: [T, k] 合并后的topk_ids
        result_weights: [T, k] 合并后的权重（归一化）
    """
    T, k = new_topk_ids.shape
    device = new_topk_ids.device
    
    result_ids = torch.zeros_like(new_topk_ids)
    result_weights = torch.zeros_like(new_topk_weights)
    
    for i in range(T):
        token_ids = new_topk_ids[i]  # [k]
        token_weights = new_topk_weights[i]  # [k] - 已与 token_ids 对应
        
        # 找到唯一的专家ID
        unique_ids, inverse_indices = torch.unique(token_ids, return_inverse=True)
        
        # 为每个唯一的专家累加权重
        merged_weights = torch.zeros(len(unique_ids), dtype=new_topk_weights.dtype, device=device)
        for j in range(len(token_ids)):
            if token_ids[j] >= 0:  # 跳过-1填充
                idx = inverse_indices[j]
                merged_weights[idx] += token_weights[j]
        
        # 按权重降序排序
        sorted_weights, sorted_indices = torch.sort(merged_weights, descending=True)
        sorted_ids = unique_ids[sorted_indices]
        
        # 填充结果（保持k个位置）
        num_unique = min(len(sorted_ids), k)
        result_ids[i, :num_unique] = sorted_ids[:num_unique]
        result_weights[i, :num_unique] = sorted_weights[:num_unique]
        
        # 剩余位置填充-1和0
        if num_unique < k:
            result_ids[i, num_unique:] = -1
            result_weights[i, num_unique:] = 0.0
    
    # 重新归一化权重
    zero_mask = result_ids == -1
    result_weights[zero_mask] = 0.0
    
    weight_sum = result_weights.sum(dim=1, keepdim=True)
    weight_sum = torch.clamp(weight_sum, min=1e-9)
    result_weights = result_weights / weight_sum
    
    result_weights[zero_mask] = 0.0
    
    return result_ids, result_weights


def token_level_load_aware_reroute(
    router_probs: torch.Tensor,      # [T, E]
    topk_ids: torch.Tensor,           # [T, k] - 原始的top-k专家ID
    topk_weights: torch.Tensor,       # [T, k] - 对应的权重
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    expert_load: Dict[int, float],    # 每个专家的当前负载 {expert_id: load_value}
    alpha: float = 0.2,
    eps: float = 1e-6,
    max_gpu_duplicates: int = 2,      # l: 每个GPU专家在单个token中的最大出现次数
    dominance_threshold: float = 0.5, # 主导性阈值：原专家得分/topk总得分 > 此值则跳过
):
    """
    负载感知的GPU优先重路由策略（token级别）— 高效向量化版本。

    相比原始版本的关键优化：
    1. 消除 [T, k, G] 大张量的 GPU→CPU 传输：在 GPU 上用 topk 压缩到 [T, k, L]（L≪G）。
    2. GPU 端用 scatter_add 计算各 token 的 GPU 专家初始使用次数，避免 [T, k, G] 广播。
    3. GPU 端通过优先级偏置编码「未激活优先」顺序，CPU 循环仅需 O(L) 查表。
    4. 一次性合并所有过滤掩码（无效位置、alpha 超限、容量超限），减少大张量遍历次数。
    5. CPU 循环只负责同 token 内的有状态使用计数更新，计算量极小。

    策略细节：
    1. 替换优先级：CPU专家按负载从高到低排序，优先替换负载最高的专家。
    2. 重复限制：每个GPU专家在单个token中的出现总次数不超过 max_gpu_duplicates (l)。
    3. 得分阈值：只替换与GPU专家得分相对差距在 alpha 内的CPU专家。
    4. 主导性检查：若原CPU专家得分占topk总得分比例超过 dominance_threshold，不替换。
    5. 优先选择当前token中未激活的GPU专家（初始计数=0）；若无，再选已激活的
       GPU专家（初始计数>=1），利用剩余的配额。

    Args:
        router_probs:        [T, E] 路由概率矩阵
        topk_ids:            [T, k] 原始top-k专家ID
        topk_weights:        [T, k] 对应权重
        gpu_expert_ids:      当前在GPU显存中的专家ID列表
        cpu_expert_ids:      当前在CPU内存中的专家ID列表
        expert_load:         每个专家的实时负载字典 {expert_id: load_value}
        alpha:               得分相似性阈值（相对差距）
        eps:                 避免除零的小常数
        max_gpu_duplicates:  l，每个GPU专家在单个token的topk中最多出现的总次数
        dominance_threshold: 主导性阈值，原专家得分占topk总得分比例超过此值则放弃替换

    Returns:
        new_topk_ids:    [T, k] 重路由后的专家ID
        new_topk_weights:[T, k] 重路由后的权重（保持原始值不变）
        reroute_mask:    [T, k] 标记哪些位置发生了替换
        reroute_stats:   dict   统计信息
    """
    T, k = topk_ids.shape
    E = router_probs.shape[1]
    device = router_probs.device

    new_topk_ids = topk_ids.clone()
    new_topk_weights = topk_weights.clone()
    reroute_mask = torch.zeros(T, k, dtype=torch.bool, device=device)

    _empty_stats = {
        'total_reroutes': 0,
        'rerouted_tokens': 0,
        'blocked_by_dominance': 0,
        'blocked_by_duplicate_limit': 0,
        'blocked_by_alpha': 0,
        'position_reroute_rate': 0.0,
        'token_reroute_rate': 0.0,
        'strategy': 'token_level_load_aware',
    }
    if not gpu_expert_ids or not cpu_expert_ids:
        return new_topk_ids, new_topk_weights, reroute_mask, _empty_stats

    G = len(gpu_expert_ids)
    gpu_tensor = torch.tensor(gpu_expert_ids, dtype=torch.long, device=device)  # [G]
    cpu_tensor = torch.tensor(cpu_expert_ids, dtype=torch.long, device=device)  # [C]

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: GPU 端全量向量化预计算
    # ══════════════════════════════════════════════════════════════════

    # (a) CPU位置掩码 [T, k]，各位置路由概率及主导性比例 [T, k]
    cpu_pos_mask = torch.isin(topk_ids, cpu_tensor)                         # [T, k]
    S_A          = torch.gather(router_probs, 1, topk_ids.long())            # [T, k]
    topk_sum     = S_A.sum(dim=1, keepdim=True).clamp(min=eps)              # [T, 1]
    dominance    = S_A / topk_sum                                            # [T, k]

    # (b) 无效位置掩码：非CPU位置 或 原专家被强烈偏好（主导性超阈值）[T, k]
    invalid_pos = ~cpu_pos_mask | (dominance > dominance_threshold)
    # 统计被主导性过滤的CPU位置数（单次GPU reduce，无CPU循环）
    blocked_dom_count = int((cpu_pos_mask & (dominance > dominance_threshold)).sum().item())

    # (c) 相对差距矩阵 [T, k, G]
    gpu_probs = router_probs[:, gpu_tensor]                                  # [T, G]
    rel_gap = (torch.abs(gpu_probs.unsqueeze(1) - S_A.unsqueeze(2))
               / (S_A.unsqueeze(2) + eps))                                  # [T, k, G]

    # (d) GPU端计算每个token中各GPU专家的初始使用次数 [T, G]
    #     使用 scatter_add：无需 [T, k, G] 广播，内存友好
    remap = torch.full((E,), -1, dtype=torch.long, device=device)
    remap[gpu_tensor] = torch.arange(G, dtype=torch.long, device=device)
    topk_gpu_idx      = remap[topk_ids.long()]                              # [T, k]，-1=非GPU专家
    valid_gpu_in_topk = topk_gpu_idx >= 0                                   # [T, k]
    topk_gpu_idx_safe = topk_gpu_idx.clamp(min=0)                           # 防scatter越界
    gpu_usage_init    = torch.zeros(T, G, dtype=torch.int32, device=device)
    gpu_usage_init.scatter_add_(
        1, topk_gpu_idx_safe, valid_gpu_in_topk.to(torch.int32)
    )                                                                        # [T, G]

    # (e) 一次性合并所有过滤掩码，减少大张量遍历次数
    #     无效位置 [T, k, 1] | 容量超限 [T, 1, G] → broadcast [T, k, G]
    capacity_exceeded = gpu_usage_init >= max_gpu_duplicates                # [T, G]
    rel_gap = rel_gap.masked_fill(
        invalid_pos.unsqueeze(-1) | capacity_exceeded.unsqueeze(1),
        float('inf')
    )
    rel_gap = rel_gap.masked_fill(rel_gap > alpha, float('inf'))

    # (f) 优先级编码：对已激活GPU专家（initial usage > 0）的gap加偏置
    #     偏置 > alpha，确保任何未激活有效候选排在已激活候选之前
    #     inf + finite = inf，已过滤位置保持 inf
    priority_offset  = alpha + 1.0
    is_active_gpu    = (gpu_usage_init > 0).float()                         # [T, G]
    rel_gap_priority = rel_gap + is_active_gpu.unsqueeze(1) * priority_offset  # [T, k, G]

    # (g) GPU端 topk 压缩 G 维度：每个(token, position)预取 top-L 候选
    #     L 远小于 G，彻底消除 [T, k, G] 的 GPU→CPU 传输瓶颈
    #     inf → -inf，不会进入 topk，自动排除无效候选
    L = min(max(max_gpu_duplicates + 1, 3), G)
    _, top_gidxs    = torch.topk(-rel_gap_priority, L, dim=2)               # [T, k, L]
    top_actual_gaps = rel_gap.gather(2, top_gidxs)                         # [T, k, L] 原始gap

    # (h) 各topk位置的专家实时负载（CPU位置才有意义）[T, k]
    #     在GPU上构建 load 向量，避免CPU端逐位置查字典
    load_arr = np.zeros(E, dtype=np.float32)
    for eid, ld in expert_load.items():
        idx = int(eid)
        if 0 <= idx < E:
            load_arr[idx] = float(ld)
    load_gpu  = torch.from_numpy(load_arr).to(device=device, dtype=torch.float32)  # [E]
    topk_load = load_gpu[topk_ids.long()] * cpu_pos_mask.float()           # [T, k]

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: 单次 GPU→CPU 批量传输
    #   传输量：[T,k]*3 + [T,k,L]*2 + [T,G]，远小于原版的 [T,k,G]
    # ══════════════════════════════════════════════════════════════════
    (cpu_pos_mask_np,
     topk_ids_np,
     topk_load_np,
     top_actual_gaps_np,
     top_gidxs_np,
     gpu_usage_init_np) = (
        cpu_pos_mask.cpu().numpy(),      # [T, k]    bool
        topk_ids.cpu().numpy(),          # [T, k]    int
        topk_load.cpu().numpy(),         # [T, k]    float
        top_actual_gaps.cpu().numpy(),   # [T, k, L] float  ← 原版为 [T, k, G]
        top_gidxs.cpu().numpy(),         # [T, k, L] int
        gpu_usage_init.cpu().numpy(),    # [T, G]    int
    )
    gpu_expert_ids_arr = np.array(gpu_expert_ids, dtype=np.int64)           # [G]

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: CPU 有状态贪心分配（O(T × k_cpu × L)，L 极小）
    #   仅处理同 token 内使用计数的状态更新，无其他 CPU 端计算
    # ══════════════════════════════════════════════════════════════════
    result_ids  = topk_ids_np.copy()
    result_mask = np.zeros((T, k), dtype=bool)
    total_reroutes = 0
    blocked_by_alpha = 0
    blocked_by_dup   = 0

    for i in range(T):
        cpu_positions = np.where(cpu_pos_mask_np[i])[0]
        if not len(cpu_positions):
            continue

        # 按CPU专家负载降序排列（负载最高的优先替换）
        sorted_positions = cpu_positions[
            np.argsort(-topk_load_np[i, cpu_positions])
        ]

        # 该token的GPU专家使用计数（可变副本，追踪同token内分配变化）
        gpu_usage = gpu_usage_init_np[i].copy()  # [G]

        for pos in sorted_positions:
            found           = False
            any_alpha_valid = False

            # 遍历 GPU 端预排序的 L 个候选（未激活优先 → 已激活次之 → invalid排在最后）
            for l in range(L):
                gap = top_actual_gaps_np[i, pos, l]
                if gap > alpha:          # inf 或真正超出阈值，跳过
                    continue

                any_alpha_valid = True
                g_idx = int(top_gidxs_np[i, pos, l])

                if gpu_usage[g_idx] >= max_gpu_duplicates:
                    continue             # 该候选已达容量，尝试下一个

                # 执行分配
                result_ids[i, pos]  = int(gpu_expert_ids_arr[g_idx])
                result_mask[i, pos] = True
                gpu_usage[g_idx]   += 1
                total_reroutes     += 1
                found = True
                break

            if not found:
                if any_alpha_valid:
                    blocked_by_dup  += 1   # 有alpha内候选但均已达容量
                else:
                    blocked_by_alpha += 1  # 无满足alpha的候选

    # ══════════════════════════════════════════════════════════════════
    # Phase 4: 单次 CPU→GPU 批量写回
    # ══════════════════════════════════════════════════════════════════
    new_topk_ids = torch.tensor(result_ids, dtype=topk_ids.dtype,  device=device)
    reroute_mask = torch.tensor(result_mask, dtype=torch.bool,     device=device)
    rerouted_tokens = int(result_mask.any(axis=1).sum())

    reroute_stats = {
        'total_reroutes':             total_reroutes,
        'rerouted_tokens':            rerouted_tokens,
        'blocked_by_dominance':       blocked_dom_count,
        'blocked_by_duplicate_limit': blocked_by_dup,
        'blocked_by_alpha':           blocked_by_alpha,
        'position_reroute_rate':      total_reroutes / (T * k) if T * k > 0 else 0.0,
        'token_reroute_rate':         rerouted_tokens / T if T > 0 else 0.0,
        'strategy':                   'token_level_load_aware',
    }

    return new_topk_ids, new_topk_weights, reroute_mask, reroute_stats
