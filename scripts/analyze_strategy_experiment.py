#!/usr/bin/env python3
"""
分析4种重路由策略的实验结果。

读取 run_strategy_experiment.sh 产出的日志目录，汇总：
1. GPU利用率（来自 gpu_utilization.log）
2. MoE统计（来自 moe_hook.log / request logs）
3. 推理延迟
4. 生成对比图表

Usage:
    python scripts/analyze_strategy_experiment.py --log-dir logs/experiments/20240101_120000
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_gpu_utilization_log(log_path: str) -> Dict[str, Any]:
    """解析GPU利用率日志文件。"""
    samples = []
    
    if not os.path.exists(log_path):
        return {'samples': [], 'avg': 0.0, 'p50': 0.0, 'p95': 0.0, 'max': 0.0, 'min': 0.0}
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 格式: [HH:MM:SS.mmm] GPU X: YY% | Mem: ZZ%
            # 或者: timestamp, gpu_util, mem_util
            m = re.search(r'GPU\s+\d+:\s+(\d+)%', line)
            if m:
                samples.append(int(m.group(1)))
                continue
            # CSV格式: timestamp,datetime,gpu_util,mem_util,...
            # 跳过header行
            if line.startswith('timestamp,'):
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    gpu_util = int(parts[2])
                    samples.append(gpu_util)
                except (ValueError, IndexError):
                    pass
    
    if not samples:
        return {'samples': [], 'avg': 0.0, 'p50': 0.0, 'p95': 0.0, 'max': 0.0, 'min': 0.0}
    
    arr = np.array(samples)
    return {
        'samples': samples,
        'avg': float(np.mean(arr)),
        'p50': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr)),
        'min': float(np.min(arr)),
        'std': float(np.std(arr)),
        'count': len(samples),
    }


def parse_moe_hook_log(log_path: str) -> Dict[str, Any]:
    """解析 moe_hook.log 提取关键指标。"""
    stats = {
        'reroute_rates': [],
        'gpu_compute_ms': [],
        'cpu_compute_ms': [],
        'layer_times_ms': [],
        'expert_replacements': [],  # for io_free
        'reroute_counts': [],
        'blocked_dup': [],
        'blocked_uniq': [],
    }
    
    if not os.path.exists(log_path):
        return stats
    
    with open(log_path, 'r') as f:
        for line in f:
            # [Reroute] Layer X stats: rate=Y%, blocked_dup=A, blocked_uniq=B
            m = re.search(r'\[Reroute\].*stats:.*rate=([\d.]+)%', line)
            if m:
                stats['reroute_rates'].append(float(m.group(1)))
            
            # [IOFree] Layer X stats: replacements=N, rate=Y%
            m = re.search(r'\[IOFree\].*stats:.*replacements=(\d+).*rate=([\d.]+)%', line)
            if m:
                stats['expert_replacements'].append(int(m.group(1)))
                stats['reroute_rates'].append(float(m.group(2)))
            
            # LX compute: N.NNms (GPU N.NNms, CPU N.NNms)
            m = re.search(r'L\d+\s+compute:\s+([\d.]+)ms\s+\(GPU\s+([\d.]+)ms,\s+CPU\s+([\d.]+)ms\)', line)
            if m:
                stats['layer_times_ms'].append(float(m.group(1)))
                stats['gpu_compute_ms'].append(float(m.group(2)))
                stats['cpu_compute_ms'].append(float(m.group(3)))
    
    return stats


def parse_request_log(log_path: str) -> Optional[Dict[str, Any]]:
    """解析单个请求的返回结果，提取 meta_info 中的 moe 指标。"""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # JSON通常在最后一行
    for line in reversed(lines):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        
        try:
            response = json.loads(line)
            meta_info = response.get('meta_info', {})
            
            # 如果有meta_info，提取MoE相关指标
            if meta_info:
                return {
                    'moe_avg_gpu_compute_ms': meta_info.get('moe_avg_gpu_compute_ms', 0),
                    'moe_avg_cpu_compute_ms': meta_info.get('moe_avg_cpu_compute_ms', 0),
                    'moe_gpu_load_change_ratio': meta_info.get('moe_gpu_load_change_ratio', 0),
                    'moe_cpu_load_change_ratio': meta_info.get('moe_cpu_load_change_ratio', 0),
                    'moe_total_layers': meta_info.get('moe_total_layers', 0),
                    'moe_reroute_rate': meta_info.get('moe_reroute_rate', 0),
                    'e2e_latency': meta_info.get('e2e_latency', 0),
                    'decode_throughput': meta_info.get('decode_throughput', 0),
                    'completion_tokens': meta_info.get('completion_tokens', 0),
                }
        except (json.JSONDecodeError, AttributeError):
            continue
    
    return None


def analyze_strategy(strategy_dir: str, strategy_name: str) -> Dict[str, Any]:
    """分析单个策略的结果。"""
    result = {
        'strategy': strategy_name,
        'gpu_util': parse_gpu_utilization_log(os.path.join(strategy_dir, 'gpu_utilization.log')),
        'moe_stats': parse_moe_hook_log(os.path.join(strategy_dir, 'moe_hook.log')),
        'requests': [],
    }
    
    # 解析每个请求，从meta_info提取MoE统计
    for f in sorted(Path(strategy_dir).glob('request_*.log')):
        req = parse_request_log(str(f))
        if req:
            result['requests'].append(req)
    
    # 如果从moe_hook.log没有数据，从请求中聚合
    if result['requests'] and not result['moe_stats']['gpu_compute_ms']:
        # 从请求中提取并聚合MoE统计
        for req in result['requests']:
            if req.get('moe_avg_gpu_compute_ms', 0) > 0:
                result['moe_stats']['gpu_compute_ms'].append(req['moe_avg_gpu_compute_ms'])
            if req.get('moe_avg_cpu_compute_ms', 0) > 0:
                result['moe_stats']['cpu_compute_ms'].append(req['moe_avg_cpu_compute_ms'])
            if req.get('moe_reroute_rate', 0) > 0:
                result['moe_stats']['reroute_rates'].append(req['moe_reroute_rate'] * 100)  # 转换为百分比
    
    return result


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """打印对比表格。"""
    strategies = list(results.keys())
    
    print("\n" + "=" * 90)
    print("  策略对比实验结果")
    print("=" * 90)
    
    # Header
    header = f"{'指标':<30}"
    for s in strategies:
        header += f"  {s:>15}"
    print(header)
    print("-" * 90)
    
    # GPU Utilization
    row = f"{'GPU利用率 (avg%)':<30}"
    for s in strategies:
        val = results[s]['gpu_util']['avg']
        row += f"  {val:>14.1f}%"
    print(row)
    
    row = f"{'GPU利用率 (p50%)':<30}"
    for s in strategies:
        val = results[s]['gpu_util']['p50']
        row += f"  {val:>14.1f}%"
    print(row)
    
    row = f"{'GPU利用率 (p95%)':<30}"
    for s in strategies:
        val = results[s]['gpu_util']['p95']
        row += f"  {val:>14.1f}%"
    print(row)
    
    row = f"{'GPU利用率 (std)':<30}"
    for s in strategies:
        val = results[s]['gpu_util'].get('std', 0)
        row += f"  {val:>14.1f}%"
    print(row)
    
    print("-" * 90)
    
    # MoE Stats
    for metric_name, key in [
        ('GPU平均计算 (ms)', 'gpu_compute_ms'),
        ('CPU平均计算 (ms)', 'cpu_compute_ms'),
        ('层平均耗时 (ms)', 'layer_times_ms'),
    ]:
        row = f"{metric_name:<30}"
        for s in strategies:
            vals = results[s]['moe_stats'].get(key, [])
            val = np.mean(vals) if vals else 0.0
            row += f"  {val:>14.2f}"
        print(row)
    
    row = f"{'平均重路由率 (%)':<30}"
    for s in strategies:
        vals = results[s]['moe_stats'].get('reroute_rates', [])
        val = np.mean(vals) if vals else 0.0
        row += f"  {val:>14.1f}%"
    print(row)
    
    # Load change metrics
    row = f"{'GPU负载变化 (avg)':<30}"
    for s in strategies:
        vals = [r.get('moe_gpu_load_change_ratio', 0) for r in results[s].get('requests', [])]
        val = np.mean(vals) if vals else 0.0
        row += f"  {val:>14.2f}"
    print(row)
    
    row = f"{'CPU负载变化 (avg)':<30}"
    for s in strategies:
        vals = [r.get('moe_cpu_load_change_ratio', 0) for r in results[s].get('requests', [])]
        val = np.mean(vals) if vals else 0.0
        row += f"  {val:>14.2f}"
    print(row)
    
    # IO-free specific
    has_io_free = 'io_free' in results
    if has_io_free:
        row = f"{'专家替换数 (io_free)':<30}"
        for s in strategies:
            vals = results[s]['moe_stats'].get('expert_replacements', [])
            val = np.mean(vals) if vals else 0.0
            row += f"  {val:>14.1f}"
        print(row)
    
    print("=" * 90)
    
    # IO-free策略诊断
    if 'io_free' in results:
        io_free_reqs = results['io_free'].get('requests', [])
        io_free_rates = [r.get('moe_reroute_rate', 0) * 100 for r in io_free_reqs]
        avg_rate = np.mean(io_free_rates) if io_free_rates else 0
        
        if avg_rate < 10:
            print("\n⚠️  IO-Free策略诊断:")
            print(f"   - 平均重路由率仅 {avg_rate:.1f}%，策略几乎未生效")
            print(f"   - 可能原因: score_threshold_ratio太低(默认0.5)，导致很少CPU专家被判定为\"低分\"")
            print(f"   - 建议: 提高 reroute_score_threshold_ratio 到 0.7-0.9")
            print(f"   - 或者: 增大 reroute_alpha 到 0.3-0.4 以放宽GPU专家相似度要求")
            print("=" * 90)


def save_json_results(results: Dict[str, Dict[str, Any]], output_path: str):
    """保存完整结果为JSON。"""
    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='分析重路由策略对比实验结果')
    parser.add_argument('--log-dir', required=True, help='实验日志目录')
    parser.add_argument('--output', default=None, help='输出JSON文件路径')
    args = parser.parse_args()
    
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"ERROR: Directory not found: {log_dir}")
        sys.exit(1)
    
    # 自动发现策略目录
    results = {}
    for entry in sorted(os.listdir(log_dir)):
        strategy_path = os.path.join(log_dir, entry)
        if os.path.isdir(strategy_path) and entry in ('static', 'dynamic', 'io_free', 'token_reroute'):
            print(f"Analyzing strategy: {entry}...")
            results[entry] = analyze_strategy(strategy_path, entry)
    
    if not results:
        print("No strategy results found in the log directory!")
        sys.exit(1)
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 保存JSON
    output_path = args.output or os.path.join(log_dir, 'analysis_results.json')
    save_json_results(results, output_path)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
