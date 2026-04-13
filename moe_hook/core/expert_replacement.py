import torch

def find_Replaceable_expert(target_idx, expert_scores, threshold=0.05, exclude_self=True, gpu_set=None):
    """
    在 expert_scores 中寻找与 target_idx 分数相似的替代专家。
    
    Args:
        target_idx: 目标专家的索引。
        expert_scores: 专家得分列表或 Tensor (shape: [num_experts])。
        threshold: 相似度阈值（比例）。例如 0.05 表示分差在目标得分的 5% 以内。
        exclude_self: 是否排除目标专家自身。
        
    Returns:
        replacement_idx: 找到的替代专家索引。如果没有符合阈值的，返回 None。
    """
    
    # 确保是 tensor 格式方便处理
    if not isinstance(expert_scores, torch.Tensor):
        expert_scores = torch.tensor(expert_scores)
    
    target_score = expert_scores[target_idx].item()
    
    # 如果目标专家本身分数为 0，替换可能没有意义或会导致除零错误
    if target_score == 0:
        return None

    # 计算所有专家与目标的绝对分差
    # diffs shape: [num_experts]
    diffs = torch.abs(expert_scores - target_score)
    
    # 定义搜索范围：分差 <= (目标分数 * 阈值)
    allowed_diff = target_score * threshold
    
    # 创建掩码
    mask = diffs <= allowed_diff
    
    if exclude_self:
        mask[target_idx] = False
        
    # 在满足阈值的索引中，寻找分差最小的那一个
    valid_indices = torch.where(mask)[0]

    # 如果提供了 gpu_set，则只保留在 gpu_set 内的候选
    if gpu_set is not None:
        try:
            gpu_lookup = set(int(x) for x in gpu_set)
            valid_indices = [int(x.item()) for x in valid_indices if int(x.item()) in gpu_lookup]
            if len(valid_indices) == 0:
                return None
            # 转回 tensor 以复用原来的选择逻辑
            valid_indices = torch.tensor(valid_indices, dtype=torch.long)
        except Exception:
            # 如果任何错误，退回到不使用 gpu 过滤的逻辑
            import traceback
            traceback.print_exc()

    if len(valid_indices) > 0:
        # 从符合条件的索引中挑出 diff 最小的
        min_diff_idx = torch.argmin(diffs[valid_indices])
        replacement_idx = valid_indices[min_diff_idx].item()
        return replacement_idx

    return None

def _apply_replacements_and_compute_pre_merge(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    load: torch.Tensor,
    router_probs: torch.Tensor,
    cur_cpu_experts: set,
    cur_gpu_experts: set,
    orig_gpu_load: int,
    orig_cpu_load: int,
    N: int,
    layer_idx: int,
    alpha: float = 0.15,
):
    """
    对高负载 CPU 专家尝试替换为 GPU 专家（按得分相似度），
    并在合并重复专家之前计算替换后的负载统计。

    Returns:
        (topk_ids, topk_weights, load, post_total_assignments, duplicates_removed, actual_experts, replacements)
    """
    replacements = {}
    try:
        expert_scores = router_probs.mean(dim=0)

        # 识别高负载专家
        load_mean = float(load.float().mean().item()) if load.numel() > 0 else 0.0
        default_thresh = max(2, int(load_mean))
        load_threshold = default_thresh

        actual_experts_pre = set(torch.nonzero(load > 0).view(-1).cpu().tolist())
        cpu_high_load = [int(e) for e in actual_experts_pre if e in cur_cpu_experts and load[e] >= load_threshold]

        for high_e in cpu_high_load:
            try:
                repl = self.find_Replaceable_expert(high_e, expert_scores, threshold=alpha, exclude_self=True, gpu_set=cur_gpu_experts)
            except Exception:
                repl = None
            if repl is not None and repl in cur_gpu_experts:
                topk_ids[topk_ids == high_e] = repl
                replacements[high_e] = int(repl)

        # 计算替换后的（但未去重）负载
        valid_post = topk_ids.view(-1)
        valid_post = valid_post[valid_post >= 0]
        if valid_post.numel() > 0:
            new_load_before_merge = torch.bincount(valid_post, minlength=router_probs.size(-1))
            post_total_assignments = int(new_load_before_merge.sum().item())
        else:
            new_load_before_merge = torch.zeros(router_probs.size(-1), dtype=torch.long, device=topk_ids.device)
            post_total_assignments = 0

        # 基于替换后的（未去重）负载重新计算实际激活专家
        actual_experts = set(torch.nonzero(new_load_before_merge > 0).view(-1).cpu().tolist())

        # 记录被去重/改变的数量
        try:
            orig_total = int((orig_gpu_load + orig_cpu_load))
        except Exception:
            orig_total = None
        duplicates_removed = (orig_total - post_total_assignments) if (orig_total is not None) else None

        # 覆盖 load，使后续统计使用替换后的（去重前）负载
        load = new_load_before_merge

        return topk_ids, topk_weights, load, post_total_assignments, duplicates_removed, actual_experts, replacements

    except Exception:
        import traceback
        traceback.print_exc()
        return topk_ids, topk_weights, load, None, None, set(), replacements