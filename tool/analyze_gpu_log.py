#!/usr/bin/env python3
"""
分析GPU利用率监控日志

Usage:
    python analyze_gpu_log.py logs/gpu_util_20240205_120000.log
    python analyze_gpu_log.py logs/gpu_util_20240205_120000.log --plot
"""

import argparse
import sys
from pathlib import Path


def analyze_log(log_file: str, show_plot: bool = False):
    """分析GPU监控日志"""
    
    # 读取CSV数据
    samples = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # 跳过表头
    if not lines or len(lines) < 2:
        print("Error: Log file is empty or has no data")
        return
    
    header = lines[0].strip().split(',')
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            try:
                sample = {
                    'timestamp': float(parts[0]),
                    'datetime': parts[1],
                    'gpu_util': float(parts[2]),
                    'mem_util': float(parts[3]),
                    'mem_used_gb': float(parts[4]),
                    'mem_total_gb': float(parts[5]),
                    'temperature': float(parts[6]),
                    'power': float(parts[7])
                }
                samples.append(sample)
            except ValueError:
                continue
    
    if not samples:
        print("Error: No valid samples found in log file")
        return
    
    # 过滤掉开头和结尾GPU利用率为0的数据（推理准备和完成阶段）
    # 找到第一个GPU利用率>0的位置
    start_idx = 0
    for i, s in enumerate(samples):
        if s['gpu_util'] > 0:
            start_idx = i
            break
    
    # 找到最后一个GPU利用率>0的位置
    end_idx = len(samples) - 1
    for i in range(len(samples) - 1, -1, -1):
        if samples[i]['gpu_util'] > 0:
            end_idx = i
            break
    
    # 过滤样本
    original_count = len(samples)
    samples = samples[start_idx:end_idx + 1]
    filtered_count = original_count - len(samples)
    
    if not samples:
        print("Error: No samples with GPU utilization > 0 found")
        return
    
    # 计算统计信息
    gpu_utils = [s['gpu_util'] for s in samples]
    mem_utils = [s['mem_util'] for s in samples]
    mem_useds = [s['mem_used_gb'] for s in samples]
    temps = [s['temperature'] for s in samples]
    powers = [s['power'] for s in samples]
    
    duration = samples[-1]['timestamp'] - samples[0]['timestamp']
    
    print(f"="*80)
    print(f"GPU Monitoring Analysis")
    print(f"="*80)
    print(f"Log File:             {log_file}")
    print(f"Total Samples:        {original_count}")
    print(f"Filtered Samples:     {filtered_count} (GPU util = 0 at start/end)")
    print(f"Valid Samples:        {len(samples)}")
    print(f"Start Time:           {samples[0]['datetime']}")
    print(f"End Time:             {samples[-1]['datetime']}")
    print(f"Duration:             {duration:.2f}s")
    print(f"Sampling Rate:        {len(samples) / duration:.2f} samples/s")
    print(f"-"*80)
    
    def print_stats(name, values, unit):
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        # 计算中位数
        sorted_vals = sorted(values)
        median = sorted_vals[len(sorted_vals) // 2]
        # 计算标准差
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        print(f"{name}:")
        print(f"  Average:            {avg:.2f}{unit}")
        print(f"  Median:             {median:.2f}{unit}")
        print(f"  Min:                {min_val:.2f}{unit}")
        print(f"  Max:                {max_val:.2f}{unit}")
        print(f"  Std Dev:            {std_dev:.2f}{unit}")
        print(f"-"*80)
    
    print_stats("GPU Utilization", gpu_utils, "%")
    print_stats("Memory Utilization", mem_utils, "%")
    print_stats("Memory Usage", mem_useds, " GB")
    print_stats("Temperature", temps, "°C")
    print_stats("Power Consumption", powers, "W")
    
    # 计算能耗
    energy_wh = sum(powers) * (duration / 3600) / len(powers)
    print(f"Total Energy Consumption: {energy_wh:.2f} Wh")
    print(f"="*80)
    
    # 可视化
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            timestamps = [(s['timestamp'] - samples[0]['timestamp']) for s in samples]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'GPU Monitoring Analysis\n{Path(log_file).name}', fontsize=14)
            
            # GPU利用率
            axes[0, 0].plot(timestamps, gpu_utils, linewidth=1)
            axes[0, 0].axhline(y=sum(gpu_utils)/len(gpu_utils), color='r', 
                              linestyle='--', label=f'Average: {sum(gpu_utils)/len(gpu_utils):.2f}%')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('GPU Utilization (%)')
            axes[0, 0].set_title('GPU Utilization')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # 显存利用率
            axes[0, 1].plot(timestamps, mem_utils, linewidth=1, color='orange')
            axes[0, 1].axhline(y=sum(mem_utils)/len(mem_utils), color='r', 
                              linestyle='--', label=f'Average: {sum(mem_utils)/len(mem_utils):.2f}%')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Memory Utilization (%)')
            axes[0, 1].set_title('Memory Bandwidth Utilization')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # 显存使用量
            axes[1, 0].plot(timestamps, mem_useds, linewidth=1, color='green')
            axes[1, 0].axhline(y=sum(mem_useds)/len(mem_useds), color='r', 
                              linestyle='--', label=f'Average: {sum(mem_useds)/len(mem_useds):.2f} GB')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Memory Usage (GB)')
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # 功耗
            axes[1, 1].plot(timestamps, powers, linewidth=1, color='red')
            axes[1, 1].axhline(y=sum(powers)/len(powers), color='darkred', 
                              linestyle='--', label=f'Average: {sum(powers)/len(powers):.2f} W')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Power (W)')
            axes[1, 1].set_title('Power Consumption')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # 保存图表
            output_path = Path(log_file).with_suffix('.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
            
            plt.show()
            
        except ImportError:
            print("\nNote: matplotlib not installed. Install with: pip install matplotlib")
        except Exception as e:
            print(f"\nError creating plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU monitoring log")
    parser.add_argument('log_file', type=str, help='Path to GPU monitoring log file')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    analyze_log(args.log_file, args.plot)


if __name__ == '__main__':
    main()
