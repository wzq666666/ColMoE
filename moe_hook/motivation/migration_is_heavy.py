from moe_hook.config import load_config
from moe_hook.core.expert_resolver import ExpertResolver

_cfg = load_config('/home/ecnu/disk/wzq/moe_hook_config.yaml')    
expert_resolver = ExpertResolver(_cfg, 'test_migration_heavy')
model_path = _cfg.get('model_path')
expert_resolver.set_hf_model_path(model_path)

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch

    print(f"Attempting to load HF model from {model_path} (may be large)")
    # Use device='cpu' to avoid GPU use; adjust dtype/device_map to your environment
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    expert_resolver.set_model_reference(hf_model)
    print("Model reference set from HF model.")
except Exception as e:
    # If HF model is too large or not available, log and keep set_hf_model_path for file-based loads
    print(f"[MOE-HOOK] Could not set model reference from HF model: {e}")
    print("[MOE-HOOK] Using HF model path only; you can call load_expert_weights_from_hf(...)")
    
cpu_expert_ids = []
gpu_expert_ids = []
model = expert_resolver.get_expert_model(1, 1)
print(model)
# cpu_expert_weights = [expert_resolver.get_expert_weights(1, i, use_cache=True) for i in cpu_expert_ids]





