"""
Native Cache IO 优化使用示例

展示如何使用优化后的 NativeGPUCacheManager 实现高效的专家迁移。
"""

import torch
from moe_hook.native import (
    init_native_cache,
    get_native_cache,
    get_native_migration_manager,
)

# ============================================================================
# 示例 1: 初始化 (启用所有 IO 优化)
# ============================================================================

def example_init():
    """初始化 Native Cache，启用所有 IO 优化特性。"""
    cache = init_native_cache(
        num_layers=28,
        num_experts=64,
        num_gpu_slots=16,
        log_path="native_cache.log",
        
        # === IO 优化参数 ===
        enable_pinned_memory=True,   # ✅ Pinned memory staging (3-4x 带宽)
        num_transfer_streams=2,       # ✅ 2 个并行传输 stream
    )
    
    print(f"✓ Native cache initialized with IO optimizations")
    print(f"  - Pinned memory staging: enabled")
    print(f"  - Transfer streams: 2")
    return cache


# ============================================================================
# 示例 2: 单专家异步加载 (使用 pinned staging)
# ============================================================================

def example_single_expert_async():
    """单个专家的异步加载，使用 pinned memory staging。"""
    cache = get_native_cache()
    migrator = get_native_migration_manager()
    
    layer_idx = 10
    expert_idx = 25
    slot_idx = 5
    
    # 异步加载 (使用所有优化)
    print(f"Loading expert[{layer_idx}][{expert_idx}] to slot {slot_idx}...")
    
    success = migrator.load_expert_to_slot(
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        slot_idx=slot_idx,
        device="cuda",
    )
    
    if success:
        print(f"✓ Expert loaded asynchronously")
        
        # 在使用前等待传输完成
        print(f"Waiting for transfer to complete...")
        cache.wait_transfer_complete(layer_idx, slot_idx)
        print(f"✓ Transfer complete, ready to use")
    else:
        print(f"✗ Failed to load expert")


# ============================================================================
# 示例 3: 批量专家加载 (Pipeline 优化)
# ============================================================================

def example_batch_experts():
    """批量加载多个专家，使用 pipeline 并行传输。"""
    cache = get_native_cache()
    migrator = get_native_migration_manager()
    
    # 准备批量加载计划
    expert_plan = [
        (10, 25, 5),   # (layer_idx, expert_idx, slot_idx)
        (10, 30, 6),
        (11, 15, 3),
        (11, 42, 7),
    ]
    
    print(f"Batch loading {len(expert_plan)} experts...")
    
    # 准备权重 (这里简化，实际应该从 HF 加载)
    swaps = []
    for layer_idx, expert_idx, slot_idx in expert_plan:
        # 加载权重
        weights = migrator.expert_resolver.load_expert_weights_from_hf(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            device="cpu",
            use_cache=True,
        )
        
        if weights:
            # 转换格式
            from moe_hook.native.native_migration import convert_hf_to_sglang_format_contiguous
            w13, w2 = convert_hf_to_sglang_format_contiguous(
                weights['w1'], weights['w2'], weights['w3']
            )
            swaps.append((layer_idx, slot_idx, expert_idx, w13, w2))
    
    # 批量异步传输 (使用 pipeline)
    results = cache.batch_swap_experts(swaps, use_pipeline=True)
    
    # 检查结果
    success_count = sum(1 for v in results.values() if v)
    print(f"✓ Batch loading complete: {success_count}/{len(swaps)} successful")
    
    return results


# ============================================================================
# 示例 4: 与推理集成 (完整流程)
# ============================================================================

def example_inference_integration(model, input_ids, scheduler):
    """展示如何在推理循环中集成优化后的 Native Cache。"""
    cache = get_native_cache()
    migrator = get_native_migration_manager()
    
    num_layers = model.config.num_hidden_layers
    
    for layer_idx in range(num_layers):
        # 1. 获取当前层需要的专家
        needed_experts = scheduler.predict_experts(layer_idx, input_ids)
        
        # 2. 检查缺失的专家
        missing_experts = []
        for expert_idx in needed_experts:
            if not cache.is_expert_on_gpu(layer_idx, expert_idx):
                missing_experts.append(expert_idx)
        
        # 3. 异步加载缺失的专家
        if missing_experts:
            print(f"Layer {layer_idx}: loading {len(missing_experts)} missing experts...")
            
            # 选择可替换的槽位
            available_slots = cache.get_available_slots(
                layer_idx=layer_idx,
                exclude_experts=set(needed_experts),
                num_slots=len(missing_experts)
            )
            
            # 批量异步加载
            swaps = []
            for expert_idx, slot_idx in zip(missing_experts, available_slots):
                weights = migrator.expert_resolver.load_expert_weights_from_hf(
                    layer_idx, expert_idx, device="cpu", use_cache=True
                )
                if weights:
                    from moe_hook.native.native_migration import convert_hf_to_sglang_format_contiguous
                    w13, w2 = convert_hf_to_sglang_format_contiguous(
                        weights['w1'], weights['w2'], weights['w3']
                    )
                    swaps.append((layer_idx, slot_idx, expert_idx, w13, w2))
            
            cache.batch_swap_experts(swaps, use_pipeline=True)
        
        # 4. 等待所有专家加载完成
        for expert_idx in needed_experts:
            slot_idx = cache.get_slot_for_expert(layer_idx, expert_idx)
            if slot_idx is not None:
                cache.wait_transfer_complete(layer_idx, slot_idx)
        
        # 5. 执行推理 (所有专家已就绪)
        layer_output = model.layers[layer_idx](input_ids)
        
        # 6. 更新访问时间 (用于 LRU)
        cache.update_access_time(layer_idx, set(needed_experts))
        
        print(f"✓ Layer {layer_idx} inference complete")


# ============================================================================
# 示例 5: 性能监控
# ============================================================================

def example_performance_monitoring():
    """监控 Native Cache 的性能指标。"""
    cache = get_native_cache()
    
    # 获取统计信息
    stats = {
        'total_swaps': cache._swap_count,
        'pinned_enabled': cache.enable_pinned_memory,
        'num_streams': len(cache._transfer_streams),
    }
    
    print("=" * 60)
    print("Native Cache Performance Stats")
    print("=" * 60)
    print(f"Total expert swaps: {stats['total_swaps']}")
    print(f"Pinned memory enabled: {stats['pinned_enabled']}")
    print(f"Transfer streams: {stats['num_streams']}")
    
    # 检查每层的状态
    for layer_idx, layer_cache in cache._layer_caches.items():
        gpu_experts = set(layer_cache.expert_to_slot.keys())
        print(f"\nLayer {layer_idx}:")
        print(f"  GPU experts: {sorted(gpu_experts)}")
        print(f"  Slots used: {len(gpu_experts)}/{layer_cache.num_slots}")


# ============================================================================
# 示例 6: 自定义传输路径
# ============================================================================

def example_custom_transfer():
    """展示如何使用自定义 stream 和传输选项。"""
    cache = get_native_cache()
    
    # 创建自定义 stream
    custom_stream = torch.cuda.Stream()
    
    layer_idx = 5
    slot_idx = 3
    expert_idx = 20
    
    # 假设已有权重 (简化示例)
    w13_weight = torch.randn(18944*2, 7680, dtype=torch.float16)  # CPU
    w2_weight = torch.randn(7680, 18944, dtype=torch.float16)     # CPU
    
    # 方案 A: 使用 pinned staging (推荐)
    print("Testing pinned staging transfer...")
    success = cache.swap_expert(
        layer_idx=layer_idx,
        slot_idx=slot_idx,
        new_expert_idx=expert_idx,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        non_blocking=True,
        stream=custom_stream,
        use_pinned_staging=True,  # 使用 pinned buffer
    )
    
    # 等待自定义 stream 完成
    custom_stream.synchronize()
    print(f"✓ Pinned staging transfer: {'success' if success else 'failed'}")
    
    # 方案 B: 直接传输 (源已 pinned)
    w13_pinned = w13_weight.pin_memory()
    w2_pinned = w2_weight.pin_memory()
    
    print("Testing direct pinned transfer...")
    success = cache.swap_expert(
        layer_idx=layer_idx,
        slot_idx=slot_idx+1,
        new_expert_idx=expert_idx+1,
        w13_weight=w13_pinned,
        w2_weight=w2_pinned,
        non_blocking=True,
        stream=custom_stream,
        use_pinned_staging=False,  # 源已 pinned，直接传输
    )
    
    custom_stream.synchronize()
    print(f"✓ Direct pinned transfer: {'success' if success else 'failed'}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Native Cache IO Optimization Examples")
    print("=" * 60)
    
    # 示例 1: 初始化
    print("\n[Example 1] Initialization")
    example_init()
    
    # 示例 2: 单专家异步加载
    print("\n[Example 2] Single Expert Async Load")
    # example_single_expert_async()  # 需要实际的 model
    
    # 示例 3: 批量加载
    print("\n[Example 3] Batch Expert Loading")
    # example_batch_experts()  # 需要实际的 model
    
    # 示例 5: 性能监控
    print("\n[Example 5] Performance Monitoring")
    example_performance_monitoring()
    
    # 示例 6: 自定义传输
    print("\n[Example 6] Custom Transfer Paths")
    # example_custom_transfer()  # 需要注册 layer
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
