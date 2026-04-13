#!/usr/bin/env python3
"""
Simple strategy switching utility via config file hot reload.

Since sglang is complex to patch with custom HTTP routes, 
we use a simpler approach: modify the config file and 
trigger a hot reload.
"""

import argparse
import sys
import os
import yaml
import time
from typing import Optional


def switch_strategy_via_config(
    config_path: str,
    strategy: str,
    alpha: Optional[float] = None,
    score_threshold_ratio: Optional[float] = None,
    allow_duplicate: Optional[bool] = None,
    use_limited_reroute: Optional[bool] = None,
    max_duplicates_per_expert: Optional[int] = None,
    min_unique_experts: Optional[int] = None,
) -> bool:
    """
    Switch strategy by updating config file and triggering scheduler reload.
    
    Args:
        config_path: Path to moe_hook_config.yaml
        strategy: New strategy name
        alpha: Alpha parameter (optional)
        score_threshold_ratio: Score threshold for io_free (optional)
        allow_duplicate: Allow duplicate routing (optional)
        use_limited_reroute: Use limited reroute (optional)
        max_duplicates_per_expert: Max duplicates per expert (optional)
        min_unique_experts: Min unique experts (optional)
        
    Returns:
        True if successful
    """
    try:
        # Load existing config
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        else:
            print(f"Config file {config_path} does not exist")
            return False
        
        # Update strategy and related params
        config['reroute_strategy'] = strategy
        
        if alpha is not None:
            config['reroute_alpha'] = alpha
        if score_threshold_ratio is not None:
            config['reroute_score_threshold_ratio'] = score_threshold_ratio
        if allow_duplicate is not None:
            config['reroute_allow_duplicate'] = allow_duplicate
        if use_limited_reroute is not None:
            config['reroute_use_limited'] = use_limited_reroute
        if max_duplicates_per_expert is not None:
            config['reroute_max_duplicates_per_expert'] = max_duplicates_per_expert
        if min_unique_experts is not None:
            config['reroute_min_unique_experts'] = min_unique_experts
        
        # Write back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        print(f"✓ Updated config file: strategy={strategy}")
        
        # Touch a trigger file to signal scheduler to reload
        trigger_path = config_path + '.reload_trigger'
        with open(trigger_path, 'w') as f:
            f.write(str(time.time()))
        
        print(f"✓ Created reload trigger: {trigger_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Switch MoE routing strategy")
    parser.add_argument('--config', required=True, help='Path to moe_hook_config.yaml')
    parser.add_argument('--strategy', required=True, 
                       choices=['static', 'io_free', 'token_reroute', 'dynamic'],
                       help='Routing strategy to switch to')
    parser.add_argument('--alpha', type=float, help='Alpha parameter (default: keep current)')
    parser.add_argument('--score-threshold-ratio', type=float, 
                       help='Score threshold ratio for io_free strategy (default: keep current)')
    parser.add_argument('--allow-duplicate', type=bool, 
                       help='Allow duplicate routing (default: keep current)')
    parser.add_argument('--use-limited-reroute', type=bool,
                       help='Use limited reroute (default: keep current)')
    parser.add_argument('--max-duplicates-per-expert', type=int,
                       help='Max duplicates per expert (default: keep current)')
    parser.add_argument('--min-unique-experts', type=int,
                       help='Min unique experts (default: keep current)')
    
    args = parser.parse_args()
    
    success = switch_strategy_via_config(
        config_path=args.config,
        strategy=args.strategy,
        alpha=args.alpha,
        score_threshold_ratio=args.score_threshold_ratio,
        allow_duplicate=args.allow_duplicate,
        use_limited_reroute=args.use_limited_reroute,
        max_duplicates_per_expert=args.max_duplicates_per_expert,
        min_unique_experts=args.min_unique_experts,
    )
    
    if success:
        print(f"✓ Strategy switched to: {args.strategy}")
        print("Note: The running sglang server needs to support config hot reload")
        print("      to pick up this change automatically.")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()