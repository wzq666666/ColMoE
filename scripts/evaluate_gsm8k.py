#!/usr/bin/env python3
"""
GSM8K 评估脚本 - 基于 OpenAI API 的评估
支持不同的专家替换策略对比
支持 jsonl 和 arrow (Hugging Face datasets) 格式
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any
from pathlib import Path
import requests
from tqdm import tqdm
from datetime import datetime

try:
    from datasets import load_from_disk, load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Only jsonl format is supported.")

def extract_answer(text: str) -> str:
    """
    从模型输出中提取最终答案
    GSM8K 答案通常在 #### 后面
    """
    # 查找 #### 标记
    if "####" in text:
        answer = text.split("####")[-1].strip()
    else:
        # 如果没有 ####，尝试提取最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        answer = numbers[-1] if numbers else ""
    
    # 清理答案
    answer = answer.replace(",", "").strip()
    
    return answer

def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    try:
        # 尝试转换为数字
        if '.' in answer:
            return str(float(answer))
        else:
            return str(int(answer))
    except:
        return answer.strip()

def load_gsm8k_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    加载 GSM8K 数据集
    支持 jsonl 和 arrow (Hugging Face datasets) 格式
    
    Args:
        data_path: 数据集路径
                  - jsonl 文件: /path/to/test.jsonl
                  - arrow 目录: /path/to/gsm8k (包含 test/ 子目录)
                  - arrow 测试集: /path/to/gsm8k/test
    
    Returns:
        问题和答案的列表
    """
    dataset = []
    data_path = Path(data_path)
    
    # 检查是否是 arrow 格式 (目录)
    if data_path.is_dir():
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required for arrow format. "
                "Install it with: pip install datasets"
            )
        
        # 尝试直接加载目录 (如果是 test/ 目录)
        try:
            print(f"尝试加载 arrow 数据集: {data_path}")
            ds = load_from_disk(str(data_path))
            print(f"成功加载数据集，共 {len(ds)} 条样本")
        except Exception as e:
            # 如果失败，尝试加载父目录的 test split
            parent_path = data_path.parent if data_path.name in ['test', 'train'] else data_path
            try:
                print(f"尝试从 {parent_path} 加载 test split")
                full_ds = load_from_disk(str(parent_path))
                ds = full_ds['test']
                print(f"成功加载 test split，共 {len(ds)} 条样本")
            except Exception as e2:
                raise RuntimeError(f"无法加载 arrow 数据集: {e}, {e2}")
        
        # 转换为列表格式
        for item in ds:
            question = item['question']
            answer = item['answer'].split('####')[-1].strip()
            answer = normalize_answer(answer)
            
            dataset.append({
                'question': question,
                'answer': answer,
                'full_answer': item['answer']
            })
    
    # jsonl 格式
    else:
        print(f"加载 jsonl 数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # GSM8K 格式: {"question": "...", "answer": "..."}
                # 答案格式: "解题过程\n#### 123"
                question = item['question']
                answer = item['answer'].split('####')[-1].strip()
                answer = normalize_answer(answer)
                
                dataset.append({
                    'question': question,
                    'answer': answer,
                    'full_answer': item['answer']
                })
    
    return dataset

def call_model_api(
    question: str,
    api_url: str = "http://localhost:30001/v1/chat/completions",
    model: str = "qwen2-moe",
    max_tokens: int = 512,
    temperature: float = 0.0
) -> str:
    """
    调用模型 API 获取回答
    
    Args:
        question: 问题文本
        api_url: API 地址
        model: 模型名称
        max_tokens: 最大生成 token 数
        temperature: 温度参数
    
    Returns:
        模型生成的回答
    """
    # 构建提示词
    prompt = f"""Solve the following math problem step by step. Show your work and provide the final answer after ####.

Question: {question}

Answer: Let's solve this step by step:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API 调用失败: {e}")
        return ""

def evaluate_gsm8k(
    dataset: List[Dict[str, Any]],
    api_url: str,
    model: str,
    output_file: str,
    num_samples: int = None
) -> Dict[str, Any]:
    """
    评估 GSM8K 数据集
    
    Args:
        dataset: 数据集
        api_url: API 地址
        model: 模型名称
        output_file: 输出文件路径
        num_samples: 评估样本数（None 表示全部）
    
    Returns:
        评估结果统计
    """
    if num_samples:
        dataset = dataset[:num_samples]
    
    results = []
    correct = 0
    total = len(dataset)
    
    print(f"开始评估 {total} 个样本...")
    
    for idx, item in enumerate(tqdm(dataset)):
        question = item['question']
        ground_truth = item['answer']
        
        # 调用模型
        response = call_model_api(question, api_url, model)
        
        # 提取答案
        predicted_answer = extract_answer(response)
        predicted_answer = normalize_answer(predicted_answer)
        
        # 判断正确性
        is_correct = (predicted_answer == ground_truth)
        if is_correct:
            correct += 1
        
        # 保存结果
        result = {
            'index': idx,
            'question': question,
            'ground_truth': ground_truth,
            'model_response': response,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        }
        results.append(result)
        
        # 实时显示准确率
        if (idx + 1) % 10 == 0:
            current_acc = correct / (idx + 1)
            print(f"\n当前准确率: {current_acc:.2%} ({correct}/{idx+1})")
    
    # 计算统计信息
    accuracy = correct / total
    stats = {
        'total_samples': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy,
        'model': model,
        'api_url': api_url,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存结果
    output_data = {
        'statistics': stats,
        'results': results
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    
    return stats

def compare_strategies(
    token_level_results: str,
    expert_level_results: str,
    output_file: str
):
    """
    比较两种策略的结果
    
    Args:
        token_level_results: Token 粒度结果文件
        expert_level_results: 专家粒度结果文件
        output_file: 对比报告输出文件
    """
    # 加载结果
    with open(token_level_results, 'r') as f:
        token_data = json.load(f)
    
    with open(expert_level_results, 'r') as f:
        expert_data = json.load(f)
    
    token_stats = token_data['statistics']
    expert_stats = expert_data['statistics']
    
    # 生成对比报告
    report = f"""
# GSM8K 评估结果对比

**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 总体统计

| 策略 | 样本数 | 正确数 | 准确率 |
|-----|-------|-------|-------|
| Token 粒度替换 | {token_stats['total_samples']} | {token_stats['correct']} | {token_stats['accuracy']:.2%} |
| 专家粒度替换 | {expert_stats['total_samples']} | {expert_stats['correct']} | {expert_stats['accuracy']:.2%} |

## 准确率差异

- **绝对差异**: {abs(token_stats['accuracy'] - expert_stats['accuracy']):.2%}
- **相对差异**: {(token_stats['accuracy'] - expert_stats['accuracy']) / expert_stats['accuracy'] * 100:.2f}%

## 详细分析

### Token 粒度替换
- 总样本: {token_stats['total_samples']}
- 正确: {token_stats['correct']}
- 错误: {token_stats['incorrect']}
- 准确率: {token_stats['accuracy']:.2%}

### 专家粒度替换
- 总样本: {expert_stats['total_samples']}
- 正确: {expert_stats['correct']}
- 错误: {expert_stats['incorrect']}
- 准确率: {expert_stats['accuracy']:.2%}

## 差异样本分析

"""
    
    # 找出两种策略结果不同的样本
    token_results = {r['index']: r for r in token_data['results']}
    expert_results = {r['index']: r for r in expert_data['results']}
    
    diff_samples = []
    for idx in token_results:
        if idx in expert_results:
            token_correct = token_results[idx]['is_correct']
            expert_correct = expert_results[idx]['is_correct']
            
            if token_correct != expert_correct:
                diff_samples.append({
                    'index': idx,
                    'question': token_results[idx]['question'],
                    'ground_truth': token_results[idx]['ground_truth'],
                    'token_answer': token_results[idx]['predicted_answer'],
                    'expert_answer': expert_results[idx]['predicted_answer'],
                    'token_correct': token_correct,
                    'expert_correct': expert_correct
                })
    
    report += f"找到 {len(diff_samples)} 个结果不同的样本\n\n"
    
    if diff_samples:
        report += "### 样本差异详情\n\n"
        for sample in diff_samples[:10]:  # 只显示前10个
            report += f"""
**样本 {sample['index']}**
- 问题: {sample['question']}
- 正确答案: {sample['ground_truth']}
- Token 粒度答案: {sample['token_answer']} ({'✓' if sample['token_correct'] else '✗'})
- 专家粒度答案: {sample['expert_answer']} ({'✓' if sample['expert_correct'] else '✗'})

"""
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n对比报告已保存到: {output_file}")
    print(report)

def main():
    parser = argparse.ArgumentParser(description='GSM8K 评估脚本')
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'compare'],
                       help='运行模式: evaluate 或 compare')
    parser.add_argument('--dataset', type=str,
                       default='/home/ecnu/disk/wzq/moe-inference/data/gsm8k/test',
                       help='GSM8K 数据集路径 (jsonl文件或arrow目录)')
    parser.add_argument('--api-url', type=str,
                       default='http://localhost:30001/v1/chat/completions',
                       help='模型 API 地址')
    parser.add_argument('--model', type=str, default='qwen2-moe',
                       help='模型名称')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件路径')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='评估样本数（默认全部）')
    parser.add_argument('--token-level-results', type=str,
                       help='Token 粒度结果文件（compare 模式）')
    parser.add_argument('--expert-level-results', type=str,
                       help='专家粒度结果文件（compare 模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        # 加载数据集
        print(f"加载数据集: {args.dataset}")
        dataset = load_gsm8k_dataset(args.dataset)
        print(f"数据集大小: {len(dataset)}")
        
        # 运行评估
        evaluate_gsm8k(
            dataset=dataset,
            api_url=args.api_url,
            model=args.model,
            output_file=args.output,
            num_samples=args.num_samples
        )
    
    elif args.mode == 'compare':
        if not args.token_level_results or not args.expert_level_results:
            print("错误: compare 模式需要指定 --token-level-results 和 --expert-level-results")
            sys.exit(1)
        
        compare_strategies(
            token_level_results=args.token_level_results,
            expert_level_results=args.expert_level_results,
            output_file=args.output
        )

if __name__ == '__main__':
    main()
