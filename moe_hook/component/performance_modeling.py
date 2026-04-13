

def LLM_Memory_modeling(memory_budget_MB, 
                                B, S, config,
                                expert_size_MB=None):
    
    L, H= config.num_hidden_layers, config.num_key_value_heads
    d_model = config.hidden_size
    D = d_model // H
    # ---- 1. KV cache (float16) ----
    kv_bytes = B * S * L * 2 * H * D * 2
    kv_MB = kv_bytes / (1024 ** 2)

    # ---- 2. Activation/Dense ----
    d_ffn = config.intermediate_size
    activation_MB = L * (B * S * d_ffn) / (1024 ** 2)

    # ---- 3. Expert Size(int 8) ----
    d_ffn = config.moe_intermediate_size
    if expert_size_MB is None:
        expert_MB = (d_model * d_ffn * 2 + d_ffn * d_model) / (1024 ** 2)
    else:
        expert_MB = expert_size_MB

    # ---- 4. Other memory ----
    others_MB = kv_MB + activation_MB

    # ---- 5. Available for expert cache ----
    available_MB = max(0, memory_budget_MB - others_MB)
    cache_total = int(available_MB // expert_MB)

    return {
        "kv_MB": round(kv_MB, 2),
        "activation_MB": round(activation_MB, 2),
        "expert_MB": round(expert_MB, 2),
        "others_MB": round(others_MB, 2),
        "available_MB": round(available_MB, 2),
        "cache_total": cache_total
    }