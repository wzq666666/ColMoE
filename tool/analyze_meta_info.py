#!/usr/bin/env python3
"""
分析保存的 meta_info 数据

使用方法：
    python analyze_meta_info.py <meta_info_file_path> [--bs BATCH_SIZE]

示例：
    python analyze_meta_info.py logs/meta_info.jsonl
    python analyze_meta_info.py logs/meta_info.jsonl --bs 4

参数说明：
    --bs: 批次大小，如果指定，则按批次分组统计
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def load_meta_info(file_path):
    """从 JSON Lines 文件加载所有 meta_info"""
    meta_infos = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    meta_infos.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
    return meta_infos


def group_by_batch(meta_infos, batch_size):
    """将请求按照批次大小分组"""
    batches = []
    for i in range(0, len(meta_infos), batch_size):
        batch = meta_infos[i:i+batch_size]
        batches.append(batch)
    return batches


def aggregate_batch_metrics(batch):
    """
    聚合一个批次的指标
    
    规则：
    - prompt_tokens: 求和
    - completion_tokens: 求和
    - e2e_latency: 取最大值（批次中最慢的请求）
    - ttft: 取最大值
    - throughput: 总tokens / 最大延迟
    - MoE指标: 按token数量加权平均
    """
    if not batch:
        return None
    
    # 基础token统计
    total_prompt_tokens = sum(m.get('prompt_tokens', 0) for m in batch)
    total_completion_tokens = sum(m.get('completion_tokens', 0) for m in batch)
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    # 时间指标：取最大值（批次完成时间由最慢的请求决定）
    max_e2e_latency = max((m.get('e2e_latency', 0) for m in batch), default=0)
    max_ttft = max((m.get('prefill_launch_latency', 0) for m in batch), default=0)
    max_inference_time = max((m.get('inference_time', 0) for m in batch), default=0)
    
    # 吞吐量：总tokens / 最大延迟
    batch_throughput = total_tokens / max_e2e_latency if max_e2e_latency > 0 else 0
    
    # MoE指标：按token数量加权平均
    moe_metrics = {}
    moe_keys = [
        'moe_avg_layer_ms',
        'moe_avg_total_compute_ms',
        'moe_avg_gpu_compute_ms',
        'moe_avg_cpu_compute_ms',
        'avg_decision_ms',
        'moe_reroute_rate',
        'moe_gpu_load_change_ratio',
        'moe_cpu_load_change_ratio',
        'hs_load_pct_mean',
        'hs_load_pct_max',
        'hs_load_pct_std',
        'hs_load_pct_p90',
        'hs_conc_ratio_mean',
        'hs_conc_ratio_max',
        'hs_conc_ratio_std',
        'random_expected_pct',
    ]
    
    for key in moe_keys:
        weighted_sum = 0
        total_weight = 0
        for m in batch:
            if key in m:
                # 使用该请求的总token数作为权重
                weight = m.get('prompt_tokens', 0) + m.get('completion_tokens', 0)
                weighted_sum += m[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            moe_metrics[key] = weighted_sum / total_weight
        else:
            moe_metrics[key] = 0
    
    # 获取其他通用信息
    moe_total_layers = batch[0].get('moe_total_layers', 0) if batch else 0
    
    return {
        'batch_size': len(batch),
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens': total_tokens,
        'max_e2e_latency': max_e2e_latency,
        'max_ttft': max_ttft,
        'max_inference_time': max_inference_time,
        'batch_throughput': batch_throughput,
        'moe_total_layers': moe_total_layers,
        **moe_metrics
    }


def analyze_meta_info(meta_infos, batch_size=None):
    """分析 meta_info 数据"""
    if not meta_infos:
        print("No data to analyze")
        return
    
    print(f"\n{'='*60}")
    print(f"Meta Info Analysis Report")
    print(f"{'='*60}\n")
    
    print(f"Total Requests: {len(meta_infos)}\n")
    
    # 如果指定了batch_size，进行批次分组统计
    if batch_size:
        print(f"{'='*60}")
        print(f"Batch-Level Analysis (Batch Size = {batch_size})")
        print(f"{'='*60}\n")
        
        batches = group_by_batch(meta_infos, batch_size)
        batch_metrics = [aggregate_batch_metrics(batch) for batch in batches]
        
        print(f"Total Batches: {len(batches)}\n")
        
        # 批次级别的统计
        print("=== Batch-Level Token Statistics ===")
        avg_prompt = statistics.mean([b['total_prompt_tokens'] for b in batch_metrics])
        avg_completion = statistics.mean([b['total_completion_tokens'] for b in batch_metrics])
        avg_total = statistics.mean([b['total_tokens'] for b in batch_metrics])
        
        print(f"Average Prompt Tokens per Batch: {avg_prompt:.2f}")
        print(f"Average Completion Tokens per Batch: {avg_completion:.2f}")
        print(f"Average Total Tokens per Batch: {avg_total:.2f}")
        
        print("\n=== Batch-Level Latency Statistics ===")
        avg_latency = statistics.mean([b['max_e2e_latency'] for b in batch_metrics])
        p50_latency = statistics.median([b['max_e2e_latency'] for b in batch_metrics])
        p90_latency = statistics.quantiles([b['max_e2e_latency'] for b in batch_metrics], n=10)[8] if len(batch_metrics) >= 10 else max([b['max_e2e_latency'] for b in batch_metrics])
        
        print(f"Average Batch E2E Latency (s): {avg_latency:.3f}")
        print(f"P50 Batch E2E Latency (s): {p50_latency:.3f}")
        print(f"P90 Batch E2E Latency (s): {p90_latency:.3f}")
        
        avg_ttft = statistics.mean([b['max_ttft'] for b in batch_metrics])
        print(f"\nAverage Batch TTFT (s): {avg_ttft:.3f}")
        
        print("\n=== Batch-Level Throughput Statistics ===")
        avg_throughput = statistics.mean([b['batch_throughput'] for b in batch_metrics])
        p50_throughput = statistics.median([b['batch_throughput'] for b in batch_metrics])
        
        print(f"Average Batch Throughput (tokens/s): {avg_throughput:.2f}")
        print(f"P50 Batch Throughput (tokens/s): {p50_throughput:.2f}")
        
        print("\n=== Batch-Level MoE Statistics (Weighted Average) ===")
        moe_keys = [
            ('moe_avg_layer_ms', 'Avg Layer Time (ms)'),
            ('moe_avg_total_compute_ms', 'Avg Total Compute Time (ms)'),
            ('moe_avg_gpu_compute_ms', 'GPU Compute Time (ms)'),
            ('moe_avg_cpu_compute_ms', 'CPU Compute Time (ms)'),
            ('avg_decision_ms', 'Decision Time (ms)'),
            ('moe_reroute_rate', 'Reroute Rate'),
            ('moe_gpu_load_change_ratio', 'GPU Load Change Ratio'),
            ('moe_cpu_load_change_ratio', 'CPU Load Change Ratio'),
            ('hs_load_pct_mean', 'HS Load Mean (%)'),
            ('hs_load_pct_max', 'HS Load Max (%)'),
            ('hs_load_pct_std', 'HS Load Std (%)'),
            ('hs_load_pct_p90', 'HS Load P90 (%)'),
            ('hs_conc_ratio_mean', 'HS Conc Mean (x)'),
            ('hs_conc_ratio_max', 'HS Conc Max (x)'),
            ('hs_conc_ratio_std', 'HS Conc Std (x)'),
            ('random_expected_pct', 'Random Expected (%)'),
        ]
        
        for key, label in moe_keys:
            values = [b[key] for b in batch_metrics if key in b and b[key] > 0]
            if values:
                avg_val = statistics.mean(values)
                print(f"{label}: {avg_val:.3f}")
        
        print(f"\nMoE Total Layers: {batch_metrics[0]['moe_total_layers']}")
        
        # 批次级别总结
        print(f"\n{'='*60}")
        print("Batch-Level Summary")
        print(f"{'='*60}")
        print(f"{'Metric':<45} {'Value':>12}")
        print(f"{'-'*60}")
        print(f"{'Total Batches':<45} {len(batches):>12d}")
        print(f"{'Batch Size':<45} {batch_size:>12d}")
        print(f"{'Avg Prompt Tokens/Batch':<45} {avg_prompt:>12.2f}")
        print(f"{'Avg Completion Tokens/Batch':<45} {avg_completion:>12.2f}")
        print(f"{'Avg Total Tokens/Batch':<45} {avg_total:>12.2f}")
        print(f"{'Avg E2E Latency (s)':<45} {avg_latency:>12.3f}")
        print(f"{'P50 E2E Latency (s)':<45} {p50_latency:>12.3f}")
        print(f"{'P90 E2E Latency (s)':<45} {p90_latency:>12.3f}")
        print(f"{'Avg TTFT (s)':<45} {avg_ttft:>12.3f}")
        print(f"{'Avg Throughput (tokens/s)':<45} {avg_throughput:>12.2f}")
        print(f"{'P50 Throughput (tokens/s)':<45} {p50_throughput:>12.2f}")
        
        # MoE指标总结
        for key, label in moe_keys:
            values = [b[key] for b in batch_metrics if key in b]
            if values:
                avg_val = statistics.mean(values)
                print(f"{'Avg ' + label:<45} {avg_val:>12.3f}")
        
        print(f"{'='*60}\n")
        
        print(f"\n{'='*60}\n")
    
    # 原始的per-request统计
    print("=== Per-Request Statistics ===\n")
    
    # 1. 基础统计
    print("--- Basic Token Statistics ---")
    prompt_tokens = [m.get('prompt_tokens', 0) for m in meta_infos]
    completion_tokens = [m.get('completion_tokens', 0) for m in meta_infos if 'completion_tokens' in m]
    
    if prompt_tokens:
        print(f"Prompt Tokens:")
        print(f"  Total: {sum(prompt_tokens)}")
        print(f"  Avg: {statistics.mean(prompt_tokens):.2f}")
        print(f"  Min: {min(prompt_tokens)}")
        print(f"  Max: {max(prompt_tokens)}")
    
    if completion_tokens:
        print(f"\nCompletion Tokens:")
        print(f"  Total: {sum(completion_tokens)}")
        print(f"  Avg: {statistics.mean(completion_tokens):.2f}")
        print(f"  Min: {min(completion_tokens)}")
        print(f"  Max: {max(completion_tokens)}")
    
    # 2. 延迟统计
    print("\n--- Latency Statistics ---")
    e2e_latencies = [m.get('e2e_latency', 0) for m in meta_infos if 'e2e_latency' in m]
    ttfts = [m.get('ttft', 0) for m in meta_infos if 'ttft' in m]
    tpots = [m.get('tpot', 0) for m in meta_infos if 'tpot' in m]
    
    if e2e_latencies:
        print(f"E2E Latency (seconds):")
        print(f"  Avg: {statistics.mean(e2e_latencies):.3f}")
        print(f"  P50: {statistics.median(e2e_latencies):.3f}")
        print(f"  P90: {statistics.quantiles(e2e_latencies, n=10)[8]:.3f}")
        print(f"  P99: {statistics.quantiles(e2e_latencies, n=100)[98]:.3f}")
    
    if ttfts:
        print(f"\nTime to First Token (seconds):")
        print(f"  Avg: {statistics.mean(ttfts):.3f}")
        print(f"  P50: {statistics.median(ttfts):.3f}")
        print(f"  P90: {statistics.quantiles(ttfts, n=10)[8]:.3f}")
    
    if tpots:
        print(f"\nTime per Output Token (seconds):")
        print(f"  Avg: {statistics.mean(tpots):.3f}")
        print(f"  P50: {statistics.median(tpots):.3f}")
        print(f"  P90: {statistics.quantiles(tpots, n=10)[8]:.3f}")
    
    # 3. MoE 专用统计
    print("\n--- MoE Statistics (Per-Request) ---")
    moe_metrics = {
        'moe_total_layers': [],
        'moe_avg_layer_ms': [],
        'moe_avg_total_compute_ms': [],
        'moe_avg_gpu_compute_ms': [],
        'moe_avg_cpu_compute_ms': [],
        'avg_decision_ms': [],
        'moe_reroute_rate': [],
        'moe_gpu_load_change_ratio': [],
        'moe_cpu_load_change_ratio': [],
        'hs_load_pct_mean': [],
        'hs_load_pct_max': [],
        'hs_load_pct_std': [],
        'hs_load_pct_p90': [],
        'hs_conc_ratio_mean': [],
        'hs_conc_ratio_max': [],
        'hs_conc_ratio_std': [],
        'random_expected_pct': [],
    }
    
    for m in meta_infos:
        for key in moe_metrics.keys():
            if key in m:
                moe_metrics[key].append(m[key])
    
    for key, values in moe_metrics.items():
        if values:
            print(f"\n{key}:")
            print(f"  Avg: {statistics.mean(values):.3f}")
            if len(values) > 1:
                print(f"  Min: {min(values):.3f}")
                print(f"  Max: {max(values):.3f}")
                print(f"  StdDev: {statistics.stdev(values):.3f}")
    
    # 4. Finish Reason 统计
    print("\n--- Finish Reason Distribution ---")
    finish_reasons = defaultdict(int)
    for m in meta_infos:
        reason = m.get('finish_reason', 'unknown')
        # finish_reason 可能是字典，提取type字段
        if isinstance(reason, dict):
            reason = reason.get('type', 'unknown')
        finish_reasons[reason] += 1
    
    for reason, count in sorted(finish_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} ({count/len(meta_infos)*100:.1f}%)")
    
    # 5. 吞吐量统计
    print("\n--- Throughput Statistics (Per-Request) ---")
    if completion_tokens and e2e_latencies:
        total_tokens = sum(completion_tokens)
        total_time = sum(e2e_latencies)
        if total_time > 0:
            print(f"Overall Throughput: {total_tokens/total_time:.2f} tokens/sec")
    
    decode_throughputs = [m.get('decode_throughput', 0) for m in meta_infos if 'decode_throughput' in m]
    if decode_throughputs:
        print(f"\nPer-request Decode Throughput (tokens/sec):")
        print(f"  Avg: {statistics.mean(decode_throughputs):.2f}")
        print(f"  Min: {min(decode_throughputs):.2f}")
        print(f"  Max: {max(decode_throughputs):.2f}")


def main():
    parser = argparse.ArgumentParser(description='分析 meta_info 数据')
    parser.add_argument('file_path', type=str, help='meta_info JSON Lines 文件路径')
    parser.add_argument('--bs', '--batch-size', type=int, default=None, 
                        dest='batch_size',
                        help='批次大小，如果指定则按批次分组统计')
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Loading data from: {file_path}")
    meta_infos = load_meta_info(file_path)
    
    analyze_meta_info(meta_infos, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
