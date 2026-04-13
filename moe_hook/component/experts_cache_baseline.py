from collections import defaultdict, deque, Counter
from abc import ABC, abstractmethod
from queue import Queue
from typing import Dict, Set, List, Tuple, Any, Optional
import torch.nn as nn
import numpy as np
from IO import IOManager, ExpertIOTask
import logging, time, sys, math, threading
from enum import Enum

logging.basicConfig(
    filename="cache.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 线程安全的缓存类，管理模型的缓存状态
class Cache:
    def __init__(self, config, model_cache:Optional[Dict[str, nn.Module]] = None):
        self.IOManager = IOManager(config=config)
        self.config = config
        ExpertIOTask.start_processor()
        
        self.lock = threading.RLock()
        
        self.model_cache = model_cache if model_cache is not None else {}
        self.experts_cache = self.__get_experts_cache(model_cache) # 用于记录每层的专家 ID 集合
        self.experts_cpumem_cache = {}
    
    def load_experts2cpumem(self):
        for i in range(self.config.hf_config.num_hidden_layers):
            for j in range(self.config.hf_config.num_experts):
                key = self._make_key("expert", i, j)
                if key in self.experts_cpumem_cache:
                    continue
                expert_cpu = self.IOManager.load2Device("expert", i, j, tar_device="cpu")
                self.experts_cpumem_cache[key] = expert_cpu
                
    def __get_experts_cache(self, model_cache:Dict[str, nn.Module]) -> Dict[int, Set[int]]:
        initial_experts_cache = defaultdict(set)
        if not model_cache:
            return initial_experts_cache
        for key in model_cache.keys():
            if key.startswith("expert"):
                layer, expert_id = key.split("_")[1:3]
                initial_experts_cache[int(layer)].add(int(expert_id))
        return initial_experts_cache
    
    def _make_key(self, type: str, layer_idx: Optional[int], expert_id: Optional[int]) -> str:
        key = type
        if layer_idx is not None:
            key += f"_{layer_idx}"
        if expert_id is not None:
            key += f"_{expert_id}"
        return key
    
    def load_model(self, type: str, layer_idx: Optional[int] = None, expert_id: Optional[int] = None) -> nn.Module:
        """加载模型到缓存，并返回模型引用。"""
        if type == "expert":
            raise Exception(f"未识别的key{type}")
        
        model = self.get_model(type, layer_idx, expert_id)
        if model is not None:
            return model
        
        load_model = self.IOManager.load2Device(type, layer_idx, expert_id)
        with self.lock:
            # 再次检查，防止在你加载期间，其他线程已经加载并放入了同一个模型
            model = self.get_model(type, layer_idx, expert_id)
            if model is not None:
                return model
            # 如果确实还没有，才把自己加载的模型放进去
            key = self._make_key(type, layer_idx, expert_id)
            self.model_cache[key] = load_model
            return self.model_cache[key]
        
    def get_model(self, type: str, layer_idx: Optional[int] = None, expert_id: Optional[int] = None) -> Optional[nn.Module]:
        """获取缓存中的模型，如果不存在返回 None。"""
        with self.lock:
            key = self._make_key(type, layer_idx, expert_id)
            return self.model_cache.get(key, None)
        
    def someInCache(self, layer_id: int, experts_id: List[int]) -> List[int]:
        """获取experts_id中在缓存中的模型，如果都不在则返回[]。"""
        in_cache = []
        for eid in experts_id:
            model = self.get_model("expert", layer_id, eid)
            if model is not None:
                in_cache.append(eid)
        return in_cache
            
    def get_cpumem_expert(self, layer_idx, expert_idx):
        key = self._make_key("expert", layer_idx, expert_idx)
        return self.experts_cpumem_cache.get(key, None)
        
    def get_cached_experts(self, layer_idx: int) -> Tuple[List[nn.Module], List[int]]:
        """
        获取缓存中的某层的专家
        返回模型引用和专家id列表
        """
        with self.lock:
            cached_experts_id = self.experts_cache.get(layer_idx, set())
            if not cached_experts_id:
                return [], []
            experts = []
            valid_ids = []
            for eid in cached_experts_id:
                expert = self.get_model("expert", layer_idx, eid)
                if expert is None:
                    continue
                experts.append(expert)
                valid_ids.append(eid)
            return experts, valid_ids
    
    def add_expert(self, layer_id: int, expert_id: int, expert_model: nn.Module):
        """实际添加某个专家模型到缓存"""
        with self.lock:
            self.model_cache[self._make_key("expert", layer_id, expert_id)] = expert_model
            self.experts_cache[layer_id].add(expert_id)
            
            del self.experts_cpumem_cache[self._make_key("expert", layer_id, expert_id)]
        
    def evict_expert(self, layer_id: int, expert_id: int) -> bool:
        """从缓存实际驱逐某个专家模型到cpu内存"""
        with self.lock:
            key = f"expert_{layer_id}_{expert_id}"
            if key in self.model_cache:
                expert = self.model_cache[key]
                self.IOManager.unloadfCuda(expert)
                del self.model_cache[key]
                if expert_id in self.experts_cache.get(layer_id, set()):
                    self.experts_cache[layer_id].remove(expert_id)
                    if not self.experts_cache[layer_id]:
                        del self.experts_cache[layer_id]
                self.experts_cpumem_cache[self._make_key("expert", layer_id, expert_id)] = expert
                return True
            logging.warning(f"[Eviction] Warning: {key} was marked for eviction but not found in cache.")
            return False
          
    def get_model_cache(self) -> Dict[str, nn.Module]:
        with self.lock:
            return self.model_cache.copy()
    
    # 清空当前缓存
    def reset(self):
        with self.lock:
            self.model_cache.clear()
            self.experts_cache.clear()
        
class Cache_Type(Enum):
    LRU = 0
    LFU = 1
    ARC = 2 
    SCORE = 3      
    
# ARCCache 类实现 ARC 算法的核心逻辑
class Standard_Cache(Cache):
    """
    baseline的缓存机制 用于测试LRU LFU ARC等传统方法，以及基于专家打分的方法
    """

    def __init__(self, config,
                 max_experts: Optional[int] = 0, 
                 type: Cache_Type = Cache_Type.LRU):
        """
        初始化 Standard_Cache

        Args:
            max_experts (int): 缓存中可以容纳的最大专家数量 (c)。
            type: 0-LRU, 1-LFU, 2-ARC, 3-SCORE
        """
        super().__init__(config=config)
        self.max_experts = max_experts if max_experts is not None else 0
        self.cur_infer_layer_idx = 0
        self.cur_compute_expert = ()
        
        if type == Cache_Type.LRU.value:
            self.cache = LRU(self.max_experts)
        elif type == Cache_Type.LFU.value:
            self.cache = LFU(self.max_experts)
        elif type == Cache_Type.ARC.value:
            self.cache = ARC(self.max_experts)
        elif type == Cache_Type.SCORE.value:
            self.cache = SCORE(self.max_experts) 
        else:
            raise Exception(f"The given cache type is not be defined")
            
    def prefetch_experts(self, layer_id: int, experts_id: List[int]) -> List[Tuple[int, int]]:
        """预取专家到缓存，将专家加入队列 T"""
        evict_list = []
        for eid in experts_id:
            expert = (layer_id, eid)
        
            # 如果已在任何队列中，则不操作
            with self.lock:
                if self.cache.in_cache(expert):
                    continue
        
            model = self.get_cpumem_expert(layer_id, eid)
            if model is None:
                # 既不在缓存，也不在cpu内存
                logging.error(f"expert{expert} neither in cpu mem nor in gpu cache")
                raise
            
            ExpertIOTask(self.IOManager, f"p_{layer_id}_{eid}", 1, model=model).start()
        
            if ExpertIOTask.wait_for_task(f"p_{layer_id}_{eid}", 1):
                with self.lock:
                    expert_model = ExpertIOTask.get_result(f"p_{layer_id}_{eid}")
                    self.add_expert(layer_id, eid, expert_model)
                    
                    self.cache.set_banned_item(self.cur_compute_expert)
                    evicted_expert = self.cache.add_item(expert)
                    if evicted_expert is not None:
                        self.evict_expert(*evicted_expert)
                        logging.info(f"[cache] evict experts {evicted_expert}")
                        evict_list.append(evicted_expert)
                        
            else:
               logging.error(f"task p_{layer_id}_{eid} timeout")           
        return evict_list
    
    def access_experts(self, layer_id: int, expert_ids: List[int]) -> Tuple[List[nn.Module],List[Tuple[int, int]]]:
        """
        根据一次推理中某个层实际使用的专家列表，来批量访问并更新缓存。返回访问专家的引用和驱逐专家信息
        若触发替换策略，则执行专家驱逐操作
        Args:
            layer_id (int): 发生访问的层 ID。
            expert_ids (List[int]): 该层在这次推理中被激活的专家 ID 列表。
        """
        expert_list = []
        evicted_list = []
        # t1 = time.time()
        for expert_id in expert_ids:
            # 依次处理每个被访问的专家
            expert = (layer_id, expert_id)
            with self.lock:
                # --- Case 1: 命中 ---
                if self.cache.in_cache(expert):
                    expert_list.append(self.get_model("expert", layer_id, expert_id))
                    continue
                
            model = self.get_cpumem_expert(layer_id, expert_id)
            if model is None:
                # --- Case 2:  既不在缓存，也不在cpu内存 ---
                logging.error(f"expert{expert} neither in cpu mem nor in gpu cache")
                raise
            
            # --- Case 3:  在cpu内存 ---
            ExpertIOTask(self.IOManager, f"m_{layer_id}_{expert_id}", 0, model=model).start()
            
            if ExpertIOTask.wait_for_task(f"m_{layer_id}_{expert_id}", timeout=1):
                with self.lock:
                    expert_model = ExpertIOTask.get_result(f"m_{layer_id}_{expert_id}")
                    expert_list.append(expert_model)
                    # 加入缓存
                    self.add_expert(layer_id, expert_id, expert_model)
                    # 加入缓存结构
                    self.cache.set_banned_item(self.cur_compute_expert)
                    evicted_expert = self.cache.add_item(expert)
                    if evicted_expert is not None:
                        self.evict_expert(*evicted_expert)
                        logging.info(f"[cache] evict experts {evicted_expert}")
                        evicted_list.append(evicted_expert)
            else:
               logging.error(f"task m_{layer_id}_{expert_id} timeout")
        return expert_list, evicted_list  
                
    def set_max_experts(self, max_experts: int) -> None:
        self.max_experts = max_experts
        self.cache.set_max_experts(max_experts)

    def set_p(self, p: float) -> None:
        self.p = p
        
    def get_experts_cache(self) -> Dict[int, Set[int]]:
        """
        获取当前缓存的专家状态。

        Returns:
            Dict[int, Set[int]]: 每层的专家 ID 集合。
        """
        return self.experts_cache

class Baseline_Cache(ABC):
    @abstractmethod
    def add_item(self, *args, **kwargs):
        """
        添加/访问一个元素。
        如果因缓存满而发生驱逐，则返回被驱逐的元素。
        否则，返回 None (或一个空元组，如下所示)。
        """
        pass
    
    @abstractmethod
    def in_cache(self, element: Tuple[int, int]):
        pass
    
    @abstractmethod
    def set_max_experts(self, maxsize):
        pass
    
    
class LRU(Baseline_Cache):
    """
        约定：deque 的左侧 (index 0) 是 LRU (最近最少使用)
        deque 的右侧 (index -1) 是 MRU (最近刚使用)
    """
    def __init__(self, maxsize):
        self.T = deque()
        self.capacity = maxsize
        self.banned = ()
    
    def _get_cache_size(self) -> int:
        return len(self.T)
    
    def _remove_item(self, element: Tuple[int, int]):
        try:
            self.T.remove(element)
        except ValueError:
            raise Exception(f"元素{element}不在缓存中")
    
    def _evict_item(self) -> Tuple[int, int]:
        # 找到淘汰的项
        if not self.T:
            return None
        
        victim = None
        # 从缓存队列中移除
        if self.T[0] == self.banned:
            if self._get_cache_size() > 1:
                victim = self.T[1] 
                self._remove_item(victim)
            else:
                logging.error("error in evict, can't evict anyone")
                return None
        else:
            victim = self.T.popleft()
        return victim
        
    def set_max_experts(self, maxsize):
        self.capacity = maxsize
       
    def set_banned_item(self, element: Tuple[int, int]) :
        self.banned = element
        
    def in_cache(self, element: Tuple[int, int]):
        return element in self.T
            
    def add_item(self, element: Tuple[int, int]):
        """
        添加/访问元素。
        - 如果元素已在缓存中 (hit)，将其移动到 MRU (右侧)。
        - 如果元素不在缓存中 (miss)，将其添加到 MRU (右侧)。
        - 如果添加后超出容量，从 LRU (左侧) 驱逐一个元素。
        """
        victim = None
        # 检查是否命中 (hit)
        if element in self.T:
            # Hit: 移动到 MRU (队尾)
            # O(N) 操作
            self.T.remove(element)
            self.T.append(element)
        else:
            # Miss: 添加到 MRU (队尾)
            # O(1) 操作
            self.T.append(element)
            
            # 判断是否缓存满
            if self._get_cache_size() > self.capacity:
                # 若满执行驱逐 (从 LRU 队首驱逐)
                victim = self._evict_item()
        return victim
        

class LFU(Baseline_Cache):
    def __init__(self, maxsize):
        self.T = deque()
        self.capacity = maxsize
        # freq_count: 用于存储每个元素的访问频率
        self.freq_count = defaultdict(int)
        self.banned = ()
        
    def _get_cache_size(self) -> int:
        return len(self.T)
    
    def _remove_item(self, element: Tuple[int, int]):
        """从缓存和频率计数中移除指定元素 (O(N))"""
        try:
            self.T.remove(element)
            del self.freq_count[element]
        except ValueError:
            raise ValueError(f"{element}不在 T 中")
        except KeyError:
            raise ValueError(f"{element}不在 freq_count 中")
    
    def _evict_item(self) -> Tuple[int, int]:
        """
        驱逐 LFU 元素。
        如果频率相同，则驱逐 LRU 元素 (即在 T 中最靠左的)。
        (O(N) 操作)
        """
        if not self.T:
            return None
        
        # 1. 创建所有候选者的列表，附带它们的驱逐优先级
        # 优先级 = (频率, 在T中的LRU位置索引)
        candidates_with_priority = []
        for i, item in enumerate(self.T):
            priority = (self.freq_count[item], i)
            candidates_with_priority.append((priority, item))
        
        # 2. 按优先级排序 (O(N log N))
        # 列表的第一个元素是 LFU/LRU (最佳受害者)
        candidates_with_priority.sort(key=lambda x: x[0])

        # 3. 寻找第一个 *未被禁止* 的候选者
        victim = None
        for priority, candidate in candidates_with_priority:
            if candidate != self.banned:
                victim = candidate
                break # 找到了
        
        if victim:
            # 从缓存中移除
            self._remove_item(victim) # O(N)
        else:
            # 缓存中所有项都是 'banned' 项 (或缓存为空，但已在开头检查)
            logging.error("error in LFU evict, can't evict anyone (only banned item(s) left)")
            
        return victim
    
    def set_banned_item(self, element: Tuple[int, int]) :
        self.banned = element
        
    def set_max_experts(self, maxsize):
        self.capacity = maxsize
        
    def in_cache(self, element: Tuple[int, int]):
        return element in self.T
    
    def add_item(self, element: Tuple[int, int]):
        """
        添加/访问元素。
        - 如果命中 (hit)，增加频率，并将其视为“最近使用”（移动到 T 的右侧）。
        - 如果未命中 (miss)，添加新元素，频率设为1。
        - 如果超出容量，执行 LFU 驱逐。
        """
        victim = None
        
        # 检查是否命中 (hit)
        if element in self.freq_count:
            # Hit: 增加频率
            self.freq_count[element] += 1
            # 更新其“最近使用”状态 (移动到 T 的队尾)
            self.T.remove(element) # O(N)
            self.T.append(element)
        else:
            # Miss
            # 判断是否缓存满 (在添加 之前 检查)
            if self._get_cache_size() >= self.capacity:
                # 若满执行驱逐 (LFU + LRU tie-breaking)
                victim = self._evict_item()

            # 加入新元素
            self.T.append(element)
            self.freq_count[element] = 1 # 初始频率为 1

        return victim
    
class ARC(Baseline_Cache):
    def __init__(self, maxsize, p = 0):
        self.T1 = deque()
        self.T2 = deque()
        self.B1 = deque()
        self.B2 = deque()
        self.capacity = max(maxsize, 1)
        self.p = p
        self.banned = ()
        
    def _get_cache_size(self) -> int:
        return len(self.T1) + len(self.T2)
    
    def _remove_item(self, element: Tuple[int, int]):
        """(辅助函数，O(N)) 从所有列表中尝试移除"""
        try: self.T1.remove(element)
        except ValueError as v: logging.error(v)
        try: self.T2.remove(element)
        except ValueError as v: logging.error(v)
        try: self.B1.remove(element)
        except ValueError as v: logging.error(v)
        try: self.B2.remove(element)
        except ValueError as v: logging.error(v)
    
    def _replace(self, element_is_in_B2: bool):
        """
        ARC 的核心驱逐逻辑 (REPLACE) - 使用健壮的辅助函数
        """
        L1_size = len(self.T1)
        victim = None
        
        # 1. 决定首选目标是 T1 还是 T2
        evict_from_T1_first = L1_size > 0 and (L1_size > self.p or (element_is_in_B2 and L1_size == self.p))

        # 2. 尝试从目标列表驱逐
        if evict_from_T1_first:
            victim = self._find_and_evict_from_deque(self.T1, self.B1)
        else:
            victim = self._find_and_evict_from_deque(self.T2, self.B2)

        # 3. 如果目标列表驱逐失败，尝试从备用列表驱逐
        if victim is None:
            if evict_from_T1_first:
                victim = self._find_and_evict_from_deque(self.T2, self.B2)
            else:
                victim = self._find_and_evict_from_deque(self.T1, self.B1)

        # 4. 如果两边都失败了
        if victim is None:
            logging.error("error in ARC replace, can't evict anyone (only banned item(s) left)")
            return None
            
        return victim
    
    def _find_and_evict_from_deque(self, dq: deque, ghost_dq: deque):
        """
        健壮的驱逐辅助函数：
        从 dq 的 LRU 端 (index 0) 开始查找第一个未被禁止的项，
        驱逐它，并将其放入 ghost_dq。
        """
        victim = None
        victim_index = -1

        # 1. 查找第一个可驱逐的项 (O(N))
        for i, item in enumerate(dq):
            if item != self.banned:
                victim = item
                victim_index = i
                break
        
        if victim is None:
            # 这个 deque 中没有可驱逐的项 (要么为空，要么全是 banned 项)
            return None
            
        # 2. 驱逐该项
        if victim_index == 0:
            # 运气好，是 LRU 项，O(1)
            dq.popleft()
        else:
            # 运气不好，是其他项，O(N)
            # 我们假设 T1/T2 中没有重复项
            try:
                dq.remove(victim)
            except ValueError:
                logging.error(f"ARC evict logic error: item {victim} not in deque.")
                return None
        
        # 3. 放入 ghost list
        ghost_dq.append(victim)
        return victim
    
    def _evict_item(self) -> Tuple[int, int]:
        """
        此方法在 ARC 的标准流程中不被 add_item 直接调用。
        add_item 会在需要时调用 _replace。
        我们让它模拟一次 "REPLACE" (假设 B2 未命中)。
        """
        if self._get_cache_size() == 0:
            return None
        return self._replace(element_is_in_B2=False)
    
    def set_banned_item(self, element: Tuple[int, int]) :
        self.banned = element
        
    def set_max_experts(self, maxsize):
        self.capacity = maxsize
        
    def in_cache(self, element: Tuple[int, int]):
        return (element in self.T1 or element in self.T2)
    
    def add_item(self, element: Tuple[int, int]):
        """
        添加/访问元素 (ARC 的完整算法逻辑) - Case 4 已修复
        """
        victim = None
        L1_size = len(self.T1)
        L2_size = len(self.T2)
        L1_plus_L2 = L1_size + L2_size

        # Case 1: 命中 T1 或 T2 (x in T1 U T2)
        if element in self.T1:
            self.T1.remove(element)
            self.T2.append(element)
            return None
        
        if element in self.T2:
            self.T2.remove(element)
            self.T2.append(element)
            return None

        # Case 2: 命中 B1 (x in B1)
        if element in self.B1:
            delta = 1 if len(self.B1) == 0 else max(len(self.B2) / len(self.B1), 1)
            self.p = min(self.capacity, self.p + delta)
            victim = self._replace(element_is_in_B2=False)
            
            # 驱逐失败，无法添加 (因为 B1 命中意味着缓存已满)
            if victim is None and L1_plus_L2 >= self.capacity:
                 logging.error("error in ARC add (Case 2), _replace failed.")
                 return None

            self.B1.remove(element)
            self.T2.append(element)
            return victim

        # Case 3: 命中 B2 (x in B2)
        if element in self.B2:
            delta = 1 if len(self.B2) == 0 else max(len(self.B1) / len(self.B2), 1)
            self.p = max(0, self.p - delta)
            victim = self._replace(element_is_in_B2=True)

            # 驱逐失败，无法添加
            if victim is None and L1_plus_L2 >= self.capacity:
                 logging.error("error in ARC add (Case 3), _replace failed.")
                 return None

            self.B2.remove(element)
            self.T2.append(element)
            return victim

        # Case 4: 完全未命中 (x not in T1 U T2 U B1 U B2)
        
        # 4a: 如果缓存已满 (L1 + L2 == c)
        if L1_plus_L2 >= self.capacity:
            # 需要驱逐
            if L1_size < self.capacity:
                # 正常 REPLACE
                victim = self._replace(element_is_in_B2=False)
            else:
                # --- START: 修复的逻辑 ---
                # T1 占满了整个缓存 (L2=0)，必须从 T1 驱逐
                # 我们必须调用健壮的驱逐函数
                victim = self._find_and_evict_from_deque(self.T1, self.B1)
                # --- END: 修复的逻辑 ---

            # 检查驱逐是否失败 (在任一分支中)
            if victim is None:
                logging.error("error in ARC add (Case 4), eviction failed (only banned item?)")
                return None # 无法驱逐，因此也无法添加
        
        # 4b: (无论是否驱逐) 将新元素 x 添加到 T1 (MRU)
        self.T1.append(element)

        # 确保 ghost list (B1, B2) 的总大小不超过 c
        while (len(self.B1) + len(self.B2)) > self.capacity:
            if len(self.B1) > 0:
                self.B1.popleft()
            else:
                self.B2.popleft()
                
        while len(self.B1) > self.capacity:
            self.B1.popleft()
        while len(self.B2) > self.capacity:
            self.B2.popleft()

        return victim
    
class SCORE(Baseline_Cache):
    def __init__(self, maxsize):
        self.T = deque()
        self.capacity = maxsize
        
        # 感知当前推理状态
        self.cur_infer_layer_idx = 0
        # 存储推理中，专家的激活次数。区分冷和热专家
        self.expert_activation_matrix = np.zeros((self.config.hf_config.num_hidden_layers, self.config.hf_config.num_experts), dtype=np.int32) 
        # 时间窗口
        self.w = 5
        # 存储每个专家的最近激活情况(最近5步)
        self.recent_activation_matrix = {j: {i: deque(maxlen=self.w) for i in range(self.config.hf_config.num_experts)} for j in range(self.config.hf_config.num_hidden_layers)}
        # 存储最近激活的专家id（最近5步）
        self.recent_active_experts = {i: deque(maxlen=self.w) for i in range(self.config.hf_config.num_hidden_layers)}
        # 存储每个专家的激活位置（第几个推理步激活，只保留最后两个）
        self.activation_position = {j: {i: deque(maxlen=2) for i in range(self.config.hf_config.num_experts)} for j in range(self.config.hf_config.num_hidden_layers)}
        # 存储每个专家的缓存驻留得分
        self.experts_score = np.zeros((self.config.hf_config.num_hidden_layers, self.config.hf_config.num_experts), dtype=float)
    
    def _get_cache_size(self) -> int:
        return len(self.T)
        
    def _remove_item(self, element: Tuple[int, int]):
        pass
    
    def _evict_item(self) -> Tuple[int, int]:
        # 找到淘汰的项
        
        # 从缓存队列中移除
        self._remove_item(victim)
        return victim
    
    def set_max_experts(self, maxsize):
        self.capacity = maxsize
        
    def add_item(self, element: Tuple[int, int]):
        """添加元素"""
        # 加入
        
        # 判断是否缓存满
        
        if self._get_cache_size() > self.capacity:
            # 若满执行驱逐
            victim = self._evict_item(self)
            return victim
        return ()
    
class ACCESS_PATTERN(Enum):
    RA = 0
    TLA = 1
    Normal = 2
    
class SCORE_AP_AWARE(Baseline_Cache):
    def __init__(self, maxsize):
        self.T = deque()
        self.capacity = maxsize
        
        # 感知当前推理状态
        self.cur_infer_layer_idx = 0
        # 存储推理中，专家的激活次数。区分冷和热专家
        self.expert_activation_matrix = np.zeros((self.config.hf_config.num_hidden_layers, self.config.hf_config.num_experts), dtype=np.int32) 
        # 时间窗口
        self.w = 5
        # 专家重合率阈值
        self.TLA_thredhold = 0.6
        self.RA_thredhold = 0.3
        # 存储每个专家的最近激活情况(最近5步)
        self.recent_activation_matrix = {j: {i: deque(maxlen=self.w) for i in range(self.config.hf_config.num_experts)} for j in range(self.config.hf_config.num_hidden_layers)}
        # 存储最近激活的专家id（最近5步）
        self.recent_active_experts = {i: deque(maxlen=self.w) for i in range(self.config.hf_config.num_hidden_layers)}
        # 存储每个专家的激活位置（第几个推理步激活，只保留最后两个）
        self.activation_position = {j: {i: deque(maxlen=2) for i in range(self.config.hf_config.num_experts)} for j in range(self.config.hf_config.num_hidden_layers)}
        # 存储每个专家的缓存驻留得分
        self.experts_score = np.zeros((self.config.hf_config.num_hidden_layers, self.config.hf_config.num_experts), dtype=float)
    
    def _get_cache_size(self) -> int:
        return len(self.T)
        
    def _remove_item(self, element: Tuple[int, int]):
        pass
    
    def _evict_item(self) -> Tuple[int, int]:
        # 找到淘汰的项
        
        # 从缓存队列中移除
        self._remove_item(victim)
        return victim
    
    def set_max_experts(self, maxsize):
        self.capacity = maxsize

    def add_item(self, element: Tuple[int, int]):
        """添加元素"""
        # 加入
        
        # 判断是否缓存满
        
        if self._get_cache_size() > self.capacity:
            # 若满执行驱逐
            victim = self._evict_item(self)
            return victim
        return ()
    
    def _experts_Status_Awareness_Update(self, layer_idx, infer_step, activate_experts_id):
        # 更新专家状态
        for id in activate_experts_id:
            self.expert_activation_matrix[layer_idx][id] += 1
  
        self.recent_active_experts[layer_idx].append(list(activate_experts_id))

        for i in range(self.config.hf_config.num_experts):
            recent_queue = self.recent_activation_matrix[layer_idx][i]
            act_pos_queue = self.activation_position[layer_idx][i]
            if i in activate_experts_id:
                recent_queue.append(1)
                act_pos_queue.append(infer_step)
            else:
                recent_queue.append(0)
      
        # 获取当前层所处的激活模式  
        overlap_ratio = self._get_active_overlap_ratio(layer_idx)
        logging.info(f"overlap_ratio: {overlap_ratio}")
        if overlap_ratio >= self.TLA_thredhold:
            ptn = ACCESS_PATTERN.TLA
        elif overlap_ratio <= self.RA_thredhold:
            ptn = ACCESS_PATTERN.RA
        else:
            ptn = ACCESS_PATTERN.Normal
            
        # 基于当前激活模式和更新后的专家状态，更新update_layers的专家的分数
        self._update_experts_score(update_layers=[layer_idx], pattern=ptn)
        logging.info(f"scores for layer{layer_idx}: {self.experts_score[layer_idx]}")
        # import os
        # os._exit(0)
            
    # 驱逐时考虑，专家分数越高，代表其越有可能在近期被使用
    def _update_experts_score(self, update_layers: List[int], pattern: ACCESS_PATTERN) -> List[float]:
        """
            更新update_layers中的专家分数
            专家分数是以层为单位归一化的，只能在层内比较
        """
        def min_max_normalize(lst):
            if not lst:  # 处理空列表
                return []
            min_val = min(lst)
            max_val = max(lst)
            if min_val == max_val:  # 所有元素相同，避免除零错误
                return [0.0] * len(lst)
            return [(x - min_val) / (max_val - min_val) for x in lst]
        
        for layer_idx in update_layers:
            popularity_l, recent_active_frequency_l, freshness_l = [], [], []
        
            for expert_idx in range(self.config.hf_config.num_experts):
                # 专家流行度，热专家总体上有更高概率使用，因为与当前请求适配
                popularity = self.expert_activation_matrix[layer_idx][expert_idx]
                popularity_l.append(popularity)
                
                # 专家最近的激活频率，反映最近的该专家激活的时间局部性水平
                recent_queue = self.recent_activation_matrix[layer_idx][expert_idx]
                recent_active_frequency = sum(recent_queue)/len(recent_queue) if len(recent_queue) != 0 else 0
                recent_active_frequency_l.append(recent_active_frequency)
                
                # 专家的激活间隔（新鲜度）
                pos_queue = self.activation_position[layer_idx][expert_idx]
                if len(pos_queue) < 2:
                    freshness = 0.0
                else:
                    interval = abs(pos_queue[1] - pos_queue[0])
                    freshness = 1.0 / (1.0 + interval)
                freshness_l.append(freshness)
            
            popularity_l = min_max_normalize(popularity_l)
            recent_active_frequency_l = min_max_normalize(recent_active_frequency_l)
            freshness_l = min_max_normalize(freshness_l)
            logging.info(f"layer{layer_idx}:\npopularity_l:{popularity_l}\nrecent_active_frequency_l:{recent_active_frequency_l}\nfreshness_l:{freshness_l}")    
            
            if pattern == ACCESS_PATTERN.RA:
                w_pop = 0.6
                w_recent = 0.3
                w_fresh = 0.1
            elif pattern == ACCESS_PATTERN.TLA:
                w_pop = 0.2
                w_recent = 0.5
                w_fresh = 0.3
            else:
                # 默认模式下均匀权重
                w_pop = w_recent = w_fresh = 1/3
            
            self.experts_score[layer_idx] = [
                w_pop * i + w_recent * j + w_fresh * k 
                for i, j, k in zip(popularity_l, recent_active_frequency_l, freshness_l)
            ] 
        
    def _get_active_overlap_ratio(self, layer_idx) -> float:
        # layer_idx层最近的激活重合率
        recent_active_queue = self.recent_active_experts[layer_idx]
        n = len(recent_active_queue)
        recent_active_experts = [recent_active_queue[i] for i in range(min(n, self.w))]
        all_recent_active_experts = []
        for lst in recent_active_experts:
            all_recent_active_experts.extend(lst)
        element_counts = Counter(all_recent_active_experts)
        total_occurrences = sum(element_counts.values())
        repeated_occurrences = sum(count for elem, count in element_counts.items() if count >= 2)
        overlap_rate = repeated_occurrences / total_occurrences
        return overlap_rate