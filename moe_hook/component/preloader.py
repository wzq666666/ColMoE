from collections import defaultdict
import random
from typing import Callable, Dict, List, Optional, Set
# import onnxruntime as ort
import torch
import torch.nn as nn
import logging

# sess_options = ort.SessionOptions()
# sess_options.log_severity_level = 3

model_cache_on_device: List = []  # 模型缓存列表，存储加载的专家模型


# --- 内置策略：按层浅层优先（FATE风格） ---
def _resolve_num_experts_to_cache(wrapper: Optional[object], cfg: Dict[str, object]) -> Optional[int]:
    """
    获取可缓存的专家总数。
    
    这个值表示我们最多可以在 GPU 上缓存多少个专家（跨所有层）。
    
    优先级:
    1. 配置文件 preload.experts_num
    2. 配置文件 experts_num
    3. 自动计算: num_layers * num_experts_per_layer * cache_ratio
    
    注意: wrapper.num_experts 是每层的专家数，不是可缓存总数！
    """
    # 从配置获取
    preload_cfg = cfg.get("preload", {}) if isinstance(cfg, dict) else {}
    val = preload_cfg.get("experts_num") or cfg.get("experts_num")
    if isinstance(val, int) and val > 0:
        return val
    
    # 尝试自动计算（基于 hf_config）
    hf_config = cfg.get("hf_config")
    if hf_config is not None:
        num_layers = getattr(hf_config, 'num_hidden_layers', None)
        num_experts = getattr(hf_config, 'num_experts', None)
        cache_ratio = preload_cfg.get("cache_ratio", 0.3)  # 默认缓存 30% 的专家
        
        if num_layers and num_experts:
            total = int(num_layers * num_experts * cache_ratio)
            logging.info(f"[PRELOADER] 自动计算可缓存专家数: {total} (layers={num_layers}, experts={num_experts}, ratio={cache_ratio})")
            return total
    
    return None


class _HFConfigBox:
    """确保下游看到hf_config属性。"""

    def __init__(self, hf_config):
        self.hf_config = hf_config


def _resolve_hf_config(cfg: Dict[str, object]):
    """从cfg中抽取hf_config（若存在），并包装成拥有hf_config属性的对象。
    
    现在支持直接使用 cfg['hf_config'] (HFModelConfig dataclass)
    """
    # 优先检查 cfg 中是否已有 hf_config（由 model_config.py 注入）
    hf_config = cfg.get("hf_config")
    if hf_config is not None:
        # 如果是 dataclass，直接包装
        if hasattr(hf_config, 'num_experts') and hasattr(hf_config, 'num_hidden_layers'):
            return _HFConfigBox(hf_config)
        # 如果已经有 hf_config 属性
        if hasattr(hf_config, "hf_config"):
            return hf_config
        return _HFConfigBox(hf_config)

    preload_cfg = cfg.get("preload", {}) if isinstance(cfg, dict) else {}
    for key in ("hf_config", "config"):
        maybe = preload_cfg.get(key) or cfg.get(key)
        if maybe is None:
            continue
        if hasattr(maybe, "hf_config"):
            return maybe
        return _HFConfigBox(maybe)
    return None

def fate_Shallow_Preference_Expert_Preload(experts_num, config, L=3) -> List:
    num_expert_per_layer = config.hf_config.num_experts
    num_layers = config.hf_config.num_hidden_layers
    top_k = config.hf_config.num_experts_per_tok

    # 判断是否合适采用Fate的浅层偏好策略
    if experts_num < num_expert_per_layer + num_layers:
        logging.info(f"[WARN] Fate浅层偏好策略不适合，当前可缓存专家数: {experts_num}, 每层专家数: {num_expert_per_layer}, 层数: {num_layers}")
        return model_cache_on_device
    
    # 可以采用Fate的浅层偏好策略进行专家预加载

    # ---- Step 1: 尝试加载前L层的全部专家 ----
    # 深层每层缓存的专家数
    cache_deep = (experts_num - L * num_expert_per_layer) // (num_layers - L)
    logging.info(f"[INFO] 深层每层缓存的专家数/总专家数: {cache_deep}/{num_expert_per_layer}, top_k: {top_k}")

    # 从深往浅寻找L
    if cache_deep < 1:
        # 当前L不适用，导致深层无法缓存。寻找合适的L
        tmp = L - 1
        while tmp > 0 and cache_deep < 1:
            cache_deep = (experts_num - tmp * num_expert_per_layer) // (num_layers - tmp)
            tmp -= 1
        if cache_deep < 1:
            logging.info("[ERROR] 无法找到合适的L层数，导致无法缓存深层专家")
            return model_cache_on_device
        else:
            L = tmp + 1

    # 从浅往深寻找L（深层的可缓存专家更多，更容易命中）

    if cache_deep < top_k:
        logging.info("[WARN] 能够缓存的每层深层专家数小于topk，缓存不命中风险大")
        # todo: 如何调整策略，使得深层专家数目大于等于top_k
    if cache_deep > num_expert_per_layer:
        cache_deep = num_expert_per_layer
        
    # 正常预加载前L层所有专家
    logging.info(f"[INFO] 缓存前{L}层全部专家")
    for layer in range(L):
        for expert_id in range(num_expert_per_layer):
            model_cache_on_device.append((layer, expert_id))

    # 加载深层专家
    for i in range(L, num_layers):
        expert_ids = random.sample(range(num_expert_per_layer),cache_deep)
        for expert_id in expert_ids:
            model_cache_on_device.append((i,expert_id))

    return model_cache_on_device

def average_Partition_Preload(experts_num, config, experts_per_layer=2) -> List:
    if experts_num < config.hf_config.num_hidden_layers:
        logging.info(f"[WARN] 平均分配策略不适合，当前可缓存专家数: {experts_num}, 层数: {config.hf_config.num_hidden_layers}")
        return model_cache_on_device
    
    # 平均分配策略
    num_expert_per_layer = experts_num // config.hf_config.num_hidden_layers
    num_expert_per_layer = max(num_expert_per_layer, experts_per_layer)
    logging.info(f"[INFO] 平均分配策略，每层缓存专家数: {num_expert_per_layer}")
    for i in range(config.hf_config.num_hidden_layers):
        expert_ids = random.sample(range(config.hf_config.num_experts), num_expert_per_layer)
        for expert_id in expert_ids:
            model_cache_on_device.append((i, expert_id))
    return model_cache_on_device


def custom_Expert_Preload(experts_num, config, custom_experts: Optional[List] = None) -> List:
    """
    自定义专家预加载策略，直接指定要加载的专家位置。
    
    Args:
        experts_num: 可缓存的专家总数（用于验证）
        config: 模型配置
        custom_experts: 自定义专家列表，格式为 [(layer_idx, expert_idx), ...]
                       如果为 None，则从配置文件读取
    
    Returns:
        专家位置列表 [(layer_idx, expert_idx), ...]
    """
    if custom_experts is None:
        logging.info("[WARN] custom模式需要指定custom_experts参数")
        return model_cache_on_device
    
    num_expert_per_layer = config.hf_config.num_experts
    num_layers = config.hf_config.num_hidden_layers
    
    # 验证并添加专家
    valid_count = 0
    for item in custom_experts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            logging.warning(f"[WARN] 无效的专家位置格式: {item}，应为 (layer_idx, expert_idx)")
            continue
        
        layer_idx, expert_idx = item
        
        # 验证范围
        if not (0 <= layer_idx < num_layers):
            logging.warning(f"[WARN] layer_idx {layer_idx} 超出范围 [0, {num_layers})")
            continue
        if not (0 <= expert_idx < num_expert_per_layer):
            logging.warning(f"[WARN] expert_idx {expert_idx} 超出范围 [0, {num_expert_per_layer})")
            continue
        
        # 检查是否超出缓存容量
        if valid_count >= experts_num:
            logging.warning(f"[WARN] 已达到最大缓存容量 {experts_num}，忽略剩余专家")
            break
        
        model_cache_on_device.append((layer_idx, expert_idx))
        valid_count += 1
    
    logging.info(f"[INFO] 自定义模式加载了 {valid_count} 个专家")
    return model_cache_on_device


# --- 标准入口：preload(wrapper, cfg) ---
_STRATEGIES: Dict[str, Callable[[int, object], List]] = {
    "fate_shallow": fate_Shallow_Preference_Expert_Preload,
    "average": average_Partition_Preload,
    "custom": custom_Expert_Preload,
}


def register_preload_strategy(name: str, fn: Callable[[int, object], List]) -> None:
    """对外暴露的策略注册接口，便于扩展自定义策略。"""

    if not name:
        return
    _STRATEGIES[name] = fn


def preload(wrapper: Optional[object], cfg: Dict[str, object]) -> List:
    """统一的预加载入口，供hook直接调用。

    参数:
        wrapper: KT的wrapper实例（若存在，可用于读取专家数）
        cfg: hook的全局配置字典
    返回:
        已选专家列表 (layer_idx, expert_id)
    
    配置示例 (moe_hook_config.yaml):
        preload:
          enable: true
          mode: custom  # 可选: fate_shallow, average, custom
          experts_num: 100  # 可缓存的专家总数
          custom_experts:   # custom 模式专用
            - [0, 0]  # layer 0, expert 0
            - [0, 1]  # layer 0, expert 1
            - [1, 0]  # layer 1, expert 0
    """
    # 清空之前的缓存（每次调用重新计算）
    global model_cache_on_device
    model_cache_on_device = []
    
    experts_num = _resolve_num_experts_to_cache(wrapper, cfg)
    hf_config = _resolve_hf_config(cfg)
    
    preload_cfg = cfg.get("preload", {}) if isinstance(cfg, dict) else {}
    mode = preload_cfg.get("mode", "fate_shallow")
    
    # 对于 custom 模式，即使没有 hf_config 也可以工作（如果提供了足够信息）
    if mode == "custom":
        custom_experts = preload_cfg.get("custom_experts", [])
        if custom_experts:
            # 尝试获取模型参数，如果没有则使用默认值
            if hf_config is None:
                # 创建一个简单的配置对象
                class _SimpleConfig:
                    pass
                class _SimpleHFConfig:
                    num_experts = preload_cfg.get("num_experts_per_layer", 8)
                    num_hidden_layers = preload_cfg.get("num_layers", 32)
                    num_experts_per_tok = preload_cfg.get("top_k", 2)
                hf_config = _SimpleConfig()
                hf_config.hf_config = _SimpleHFConfig()
            
            if experts_num is None:
                experts_num = len(custom_experts)  # 默认允许加载所有指定的专家
            
            return custom_Expert_Preload(experts_num, hf_config, custom_experts)
    
    # 其他模式需要完整的配置
    if experts_num is None or hf_config is None:
        logging.info("[PRELOADER] experts_num或hf_config缺失，跳过预加载")
        return model_cache_on_device

    fn = _STRATEGIES.get(mode)
    if fn is None:
        logging.info(f"[PRELOADER] 未知预加载模式: {mode}, 可选: {list(_STRATEGIES.keys())}")
        return model_cache_on_device

    return fn(experts_num, hf_config)



