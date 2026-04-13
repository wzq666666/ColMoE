#!/usr/bin/env python3
"""
可视化对比工具 - 生成评估结果的可视化图表
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_results(file_path: str):
    """加载评估结果"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_accuracy_comparison(results_dict: dict, output_file: str):
    """
    绘制准确率对比图
    
    Args:
        results_dict: {strategy_name: results_data}
        output_file: 输出文件路径
    """
    strategies = list(results_dict.keys())
    accuracies = [results_dict[s]['statistics']['accuracy'] * 100 for s in strategies]
    correct = [results_dict[s]['statistics']['correct'] for s in strategies]
    total = [results_dict[s]['statistics']['total_samples'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率柱状图
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    bars = ax1.bar(strategies, accuracies, color=colors[:len(strategies)], alpha=0.8)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('GSM8K 准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc, cor, tot in zip(bars, accuracies, correct, total):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%\n({cor}/{tot})',
                ha='center', va='bottom', fontsize=10)
    
    # 正确/错误样本数
    correct_counts = [results_dict[s]['statistics']['correct'] for s in strategies]
    incorrect_counts = [results_dict[s]['statistics']['incorrect'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, correct_counts, width, label='正确', 
                    color='#4CAF50', alpha=0.8)
    bars2 = ax2.bar(x + width/2, incorrect_counts, width, label='错误',
                    color='#F44336', alpha=0.8)
    
    ax2.set_ylabel('样本数', fontsize=12)
    ax2.set_title('正确/错误样本数对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ 准确率对比图已保存: {output_file}')
    plt.close()

def plot_accuracy_by_question(results_dict: dict, output_file: str, max_questions: int = 50):
    """
    绘制每个问题的准确性对比
    
    Args:
        results_dict: {strategy_name: results_data}
        output_file: 输出文件路径
        max_questions: 显示的最大问题数
    """
    strategies = list(results_dict.keys())
    
    # 提取每个策略的结果
    results_by_strategy = {}
    for strategy in strategies:
        results = results_dict[strategy]['results']
        results_by_strategy[strategy] = {
            r['index']: r['is_correct'] for r in results
        }
    
    # 找出所有问题的索引
    all_indices = sorted(set().union(*[set(r.keys()) for r in results_by_strategy.values()]))
    all_indices = all_indices[:max_questions]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(all_indices))
    width = 0.8 / len(strategies)
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    for i, strategy in enumerate(strategies):
        correctness = [1 if results_by_strategy[strategy].get(idx, False) else 0 
                      for idx in all_indices]
        offset = (i - len(strategies)/2) * width + width/2
        ax.bar(x + offset, correctness, width, 
               label=strategy, alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_xlabel('问题索引', fontsize=12)
    ax.set_ylabel('正确性 (1=正确, 0=错误)', fontsize=12)
    ax.set_title(f'前 {max_questions} 个问题的准确性对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(idx) for idx in all_indices[::5]])
    ax.set_ylim([-0.1, 1.2])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ 问题级对比图已保存: {output_file}')
    plt.close()

def plot_difference_analysis(results_dict: dict, output_file: str):
    """
    分析两种策略结果差异的问题
    
    Args:
        results_dict: {strategy_name: results_data}
        output_file: 输出文件路径
    """
    if len(results_dict) != 2:
        print("差异分析需要恰好两个策略的结果")
        return
    
    strategies = list(results_dict.keys())
    strategy1, strategy2 = strategies[0], strategies[1]
    
    results1 = {r['index']: r['is_correct'] for r in results_dict[strategy1]['results']}
    results2 = {r['index']: r['is_correct'] for r in results_dict[strategy2]['results']}
    
    # 分类
    both_correct = []
    both_incorrect = []
    only_1_correct = []
    only_2_correct = []
    
    all_indices = set(results1.keys()) & set(results2.keys())
    
    for idx in all_indices:
        r1, r2 = results1[idx], results2[idx]
        if r1 and r2:
            both_correct.append(idx)
        elif not r1 and not r2:
            both_incorrect.append(idx)
        elif r1 and not r2:
            only_1_correct.append(idx)
        else:
            only_2_correct.append(idx)
    
    # 绘制饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：差异分布
    categories = ['都正确', '都错误', f'仅{strategy1}正确', f'仅{strategy2}正确']
    counts = [len(both_correct), len(both_incorrect), len(only_1_correct), len(only_2_correct)]
    colors_pie = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
    
    wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    ax1.set_title('结果差异分布', fontsize=14, fontweight='bold')
    
    # 右图：差异统计
    categories_bar = ['都正确', '都错误', '结果不同']
    counts_bar = [len(both_correct), len(both_incorrect), 
                  len(only_1_correct) + len(only_2_correct)]
    
    bars = ax2.bar(categories_bar, counts_bar, color=['#4CAF50', '#F44336', '#FFC107'], alpha=0.8)
    ax2.set_ylabel('样本数', fontsize=12)
    ax2.set_title('结果一致性分析', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(all_indices)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ 差异分析图已保存: {output_file}')
    print(f'\n差异统计:')
    print(f'  都正确: {len(both_correct)} ({len(both_correct)/len(all_indices)*100:.1f}%)')
    print(f'  都错误: {len(both_incorrect)} ({len(both_incorrect)/len(all_indices)*100:.1f}%)')
    print(f'  仅{strategy1}正确: {len(only_1_correct)}')
    print(f'  仅{strategy2}正确: {len(only_2_correct)}')
    plt.close()

def generate_summary_report(results_dict: dict, output_file: str):
    """生成文本摘要报告"""
    report = f"""
# GSM8K 评估结果摘要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 总体对比

| 策略 | 样本数 | 正确数 | 错误数 | 准确率 |
|-----|-------|-------|-------|-------|
"""
    
    for strategy, data in results_dict.items():
        stats = data['statistics']
        report += f"| {strategy} | {stats['total_samples']} | {stats['correct']} | {stats['incorrect']} | {stats['accuracy']*100:.2f}% |\n"
    
    # 计算相对差异
    if len(results_dict) == 2:
        strategies = list(results_dict.keys())
        acc1 = results_dict[strategies[0]]['statistics']['accuracy']
        acc2 = results_dict[strategies[1]]['statistics']['accuracy']
        
        report += f"""
## 策略对比

- **绝对准确率差异**: {abs(acc1 - acc2)*100:.2f}%
- **相对准确率差异**: {(acc1 - acc2) / acc2 * 100:.2f}%
- **优势策略**: {strategies[0] if acc1 > acc2 else strategies[1]}

## 推荐

"""
        if abs(acc1 - acc2) < 0.01:  # 差异小于1%
            report += "两种策略精度相近，建议选择性能更好的专家粒度策略。\n"
        elif acc1 > acc2:
            report += f"{strategies[0]} 精度显著更高，如果精度是关键需求，推荐使用。\n"
        else:
            report += f"{strategies[1]} 精度显著更高，如果精度是关键需求，推荐使用。\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f'✓ 摘要报告已保存: {output_file}')

def main():
    parser = argparse.ArgumentParser(description='可视化评估结果')
    parser.add_argument('--results', nargs='+', required=True,
                       help='结果文件路径（可多个）')
    parser.add_argument('--labels', nargs='+',
                       help='策略标签（可选，默认使用文件名）')
    parser.add_argument('--output-dir', type=str,
                       default='output/visualizations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载结果
    results_dict = {}
    labels = args.labels if args.labels else [Path(f).stem for f in args.results]
    
    for label, result_file in zip(labels, args.results):
        print(f'加载结果: {label} <- {result_file}')
        results_dict[label] = load_results(result_file)
    
    print(f'\n已加载 {len(results_dict)} 个策略的结果')
    
    # 生成可视化
    print('\n生成可视化...')
    
    # 1. 准确率对比
    plot_accuracy_comparison(
        results_dict,
        output_dir / 'accuracy_comparison.png'
    )
    
    # 2. 问题级对比
    plot_accuracy_by_question(
        results_dict,
        output_dir / 'question_level_comparison.png',
        max_questions=50
    )
    
    # 3. 差异分析（仅两个策略）
    if len(results_dict) == 2:
        plot_difference_analysis(
            results_dict,
            output_dir / 'difference_analysis.png'
        )
    
    # 4. 生成摘要报告
    generate_summary_report(
        results_dict,
        output_dir / 'summary_report.md'
    )
    
    print(f'\n✅ 所有可视化已生成在: {output_dir}')
    print(f'\n查看结果:')
    print(f'  图片: ls {output_dir}/*.png')
    print(f'  报告: cat {output_dir}/summary_report.md')

if __name__ == '__main__':
    main()
