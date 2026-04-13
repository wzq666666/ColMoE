#!/usr/bin/env python3
"""
配置文件生成器 - 生成不同专家替换策略的配置
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

def create_base_config() -> Dict[str, Any]:
    """创建基础配置"""
    return {
        'enable': True,
        'dynamic_scheduling': True,
        'max_gpu_experts_per_layer': 10,
        'cpu_cache_max_experts': 64,
        'enable_cpu_weight_cache': True,
        'enable_pinned_memory': False,
        'disable_deferral': True,
        'num_transfer_streams': 2,
        'pinned_pool_size': 64,
        
        'gate': {
            'enable': True,
            'device': None,
            'format': 'auto',
            'model_path': '/home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct',
            'next_layer_offset': 1,
            'patterns': ['model.layers.{idx}.mlp.gate.weight'],
            'total_layers': None
        },
        
        'predict': {
            'enable': False,
            'mode': 'fate'
        },
        
        'prefetch': {
            'enable': False,
            'mode': 'layer'
        },
        
        'preload': {
            'enable': True,
            'mode': 'average',
            'cache_ratio': 0.2
        },
        
        'hf_model_path': '/home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct',
        'model_path': '/home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct',
        'log_level': 3,
        'scheduling_backend': 'native',
        'capture_bs': [1, 2, 4, 8]
    }

def create_token_level_config(alpha: float = 0.35, output_dir: str = None) -> str:
    """
    创建 Token 粒度替换配置
    
    Args:
        alpha: 容忍度参数
        output_dir: 输出目录
    
    Returns:
        配置文件路径
    """
    config = create_base_config()
    config['reroute_strategy'] = 'token_low_score'
    config['reroute_alpha'] = alpha
    
    if output_dir:
        config['log_path'] = f'{output_dir}/token_level_moe_hook.log'
    else:
        config['log_path'] = '/home/ecnu/disk/wzq/logs/token_level_moe_hook.log'
    
    return config

def create_expert_level_config(alpha: float = 0.35, output_dir: str = None) -> str:
    """
    创建专家粒度替换配置
    
    Args:
        alpha: 容忍度参数
        output_dir: 输出目录
    
    Returns:
        配置文件路径
    """
    config = create_base_config()
    config['reroute_strategy'] = 'expert_reroute'
    config['reroute_alpha'] = alpha
    
    if output_dir:
        config['log_path'] = f'{output_dir}/expert_level_moe_hook.log'
    else:
        config['log_path'] = '/home/ecnu/disk/wzq/logs/expert_level_moe_hook.log'
    
    return config

def create_baseline_config(output_dir: str = None) -> str:
    """
    创建基线配置（不进行重路由）
    
    Args:
        output_dir: 输出目录
    
    Returns:
        配置
    """
    config = create_base_config()
    config['reroute_strategy'] = 'none'
    config['reroute_alpha'] = 0.0
    
    if output_dir:
        config['log_path'] = f'{output_dir}/baseline_moe_hook.log'
    else:
        config['log_path'] = '/home/ecnu/disk/wzq/logs/baseline_moe_hook.log'
    
    return config

def main():
    parser = argparse.ArgumentParser(description='生成 MoE Hook 配置文件')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['token_level', 'expert_level', 'baseline', 'all'],
                       help='配置策略类型')
    parser.add_argument('--alpha', type=float, default=0.35,
                       help='容忍度参数 (默认: 0.35)')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ecnu/disk/wzq/configs',
                       help='输出目录')
    parser.add_argument('--log-dir', type=str,
                       help='日志目录（可选）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs_to_create = []
    
    if args.strategy == 'all':
        configs_to_create = [
            ('token_level', create_token_level_config(args.alpha, args.log_dir)),
            ('expert_level', create_expert_level_config(args.alpha, args.log_dir)),
            ('baseline', create_baseline_config(args.log_dir))
        ]
    elif args.strategy == 'token_level':
        configs_to_create = [
            ('token_level', create_token_level_config(args.alpha, args.log_dir))
        ]
    elif args.strategy == 'expert_level':
        configs_to_create = [
            ('expert_level', create_expert_level_config(args.alpha, args.log_dir))
        ]
    elif args.strategy == 'baseline':
        configs_to_create = [
            ('baseline', create_baseline_config(args.log_dir))
        ]
    
    # 保存配置文件
    for name, config in configs_to_create:
        filename = output_dir / f'moe_hook_{name}.yaml'
        
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f'✓ 创建配置文件: {filename}')
        print(f'  策略: {config["reroute_strategy"]}')
        print(f'  Alpha: {config.get("reroute_alpha", "N/A")}')
        print(f'  日志: {config["log_path"]}')
        print()
    
    # 生成使用说明
    print('=' * 60)
    print('使用说明:')
    print('=' * 60)
    print()
    
    for name, _ in configs_to_create:
        config_path = output_dir / f'moe_hook_{name}.yaml'
        print(f'{name.upper()} 配置:')
        print(f'  export MOE_HOOK_CONFIG={config_path}')
        print(f'  bash /home/ecnu/disk/wzq/scripts/start_SGLang_MoE_Col_with_Hook.sh')
        print()

if __name__ == '__main__':
    main()
