import math
import threading
from collections import defaultdict, deque, Counter
from queue import Queue
from typing import Dict, Set, List, Tuple, Optional
import torch.nn as nn
import numpy as np
from IO import IOManager, ExpertIOTask
import logging
import time
from enum import Enum

logging.basicConfig(
    filename="cache.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ARC_QUEUE(Enum):
    TP = 0
    T1 = 1
    T2 = 2

class ACCESS_PATTERN(Enum):
    RA = 0
    TLA = 1
    Normal = 2

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
                
                key = self._make_key("expert_cpu", i, j)
                if key in self.experts_cpumem_cache:
                    continue
                # expert_cpu = self.IOManager.load2Device("expert_ov", i, j)
                expert_cpu = self.IOManager.load2Device("expert_ori", i, j, tar_device="cpu")
                self.experts_cpumem_cache[key] = expert_cpu
        
        logging.info(self.experts_cpumem_cache.keys())
                
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
        # 一次性获取锁，读取所有状态，而不是循环调用 get_model
        with self.lock:
             # 直接访问内部结构，避免反复加锁
             layer_cache = self.experts_cache.get(layer_id, set())
             for eid in experts_id:
                 if eid in layer_cache:
                     in_cache.append(eid)
        return in_cache 
        # for eid in experts_id:
        #     model = self.get_model("expert", layer_id, eid)
        #     if model is not None:
        #         in_cache.append(eid)
        # return in_cache
            
    def get_cpumem_expert(self, layer_idx, expert_idx):
        key = self._make_key("expert", layer_idx, expert_idx)
        return self.experts_cpumem_cache.get(key, None)
    
    def get_cpumem_cpu_expert(self, layer_idx, expert_idx, type: str):
        key = self._make_key(type, layer_idx, expert_idx)
        return self.experts_cpumem_cache.get(key, None)
    
    def get_cached_experts(self, layer_idx: int) -> Tuple[List[nn.Module], List[int]]:
        """
        获取缓存中的某层的专家
        返回模型引用和专家id列表(升序)
        """
        with self.lock:
            cached_experts_id = self.experts_cache.get(layer_idx, set())
            if not cached_experts_id:
                return [], []
            experts = []
            valid_ids = []
            # wzq
            sorted_cached_experts_id = sorted(cached_experts_id)
            for eid in cached_experts_id:
                expert = self.get_model("expert", layer_idx, eid)
                if expert is None:
                    continue
                experts.append(expert)
                valid_ids.append(eid)
            return experts, valid_ids
        
    
    def evict_expert(self, layer_id: int, expert_id: int) -> bool:
        """驱逐某个专家模型。"""
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
        
            
# ARCCache 类实现 ARC 算法的核心逻辑
class ARCCache(Cache):
    """
    实现了自适应替换缓存（ARC）算法。
    该算法用于管理一个缓存系统，例如在混合专家模型（MoE）中缓存专家模块。
    它通过维护四个队列来动态地平衡访问的“近因性”和“频率”：
    - TP: 预取进程访问的专家（基于预测结果，具备不确定性，这些专家需要进一步考察，即实际是否会用到）
    - T1: 最近访问过的专家（缓存中，偏向近因性）。
    - T2: 频繁访问过的专家（缓存中，偏向频率）。
    - B1: 从 T1 中被驱逐的专家记录（“幽灵”列表，记录近因历史）。
    - B2: 从 T2 中被驱逐的专家记录（“幽灵”列表，记录频率历史）。
    算法会根据访问模式动态调整 T1 和 T2 的目标大小，以实现高效的缓存命中率。
    """

    def __init__(self, config,
                 max_experts: int = 0, 
                 p: int = 0):
        """
        初始化 ARCCache。

        Args:
            max_experts (int): 缓存中可以容纳的最大专家数量 (c)。
            p (int): T1 队列的目标大小，会根据访问模式在 [0, max_experts] 之间动态调整。
        """
        super().__init__(config=config)
        self.max_experts = max_experts
        self.p: float = float(p)
        # 使用 (layer_id, expert_id) 元组来唯一标识一个专家
        # TP: 预取的专家（代考察）
        self.TP: deque[Tuple[int, int]] = deque()
        # T1: 最近访问过的缓存项 (近因)
        self.T1: deque[Tuple[int, int]] = deque()
        # T2: 频繁访问过的缓存项 (频率)
        self.T2: deque[Tuple[int, int]] = deque()
        # B1: 最近从 T1 驱逐的项 (近因的幽灵列表)
        self.B1: deque[Tuple[int, int]] = deque()
        # B2: 最近从 T2 驱逐的项 (频率的幽灵列表)
        self.B2: deque[Tuple[int, int]] = deque()
        
        # 提高查找效率的辅助结构
        self.TP_set = set()
        self.T1_set = set()
        self.T2_set = set()
        self.B1_set = set()
        self.B2_set = set()
        
        # p: T1 的目标大小，动态调整
        self.p: float = float(p)
        
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
        """返回当前所有队列的总大小"""
        return len(self.T1) + len(self.T2) + len(self.TP)
    
    def _add_to_queue(self, queue: deque, s: set, element: Tuple[int, int]):
        """添加元素到队列和对应的集合（线程安全）"""
        with self.lock:
            if element not in s:  # 先查集合（O(1)），避免重复添加
                queue.append(element)
                s.add(element)
    
    def _remove_from_queue(self, queue: deque, s: set, element: Tuple[int, int]) -> bool:
        """从队列和集合中移除元素（线程安全），返回是否成功"""
        with self.lock:
            if element not in s:  # 先查集合（O(1)），不存在则直接返回
                return False
            # 从队列中移除（deque.remove是O(n)，但通过集合提前过滤无效操作）
            queue.remove(element)
            s.remove(element)
            return True
    
    def _remove_from_specific_queue(self, queue_type: ARC_QUEUE, element: Tuple[int, int]) -> bool:
        """从指定类型的队列和集合中移除元素返回是否成功"""
        if queue_type == ARC_QUEUE.TP:
            return self._remove_from_queue(self.TP, self.TP_set, element)
        if queue_type == ARC_QUEUE.T1:
            return self._remove_from_queue(self.T1, self.T1_set, element)
        if queue_type == ARC_QUEUE.T2:
            return self._remove_from_queue(self.T2, self.T2_set, element) 
                
    def isInWhitelist(self, victim: Tuple[int, int]):
        if victim[0] in (self.cur_infer_layer_idx, (self.cur_infer_layer_idx + 1) % self.config.hf_config.num_hidden_layers):
            return True
        return False
    
    # 根据驱逐策略进行驱逐
    def changeVictim(self, protected_victim: Tuple[int, int], queue_type: ARC_QUEUE) -> Tuple[Tuple[int, int], ARC_QUEUE]:
        """
        queue_type 
        |TP                       |T1                      |T2
        ___________________________________________________________________
        TP（预取进程访问的专家） > T1（最近访问过的专家） > T2（频繁访问过的专家）
        驱逐previous层的专家  > 驱逐subsequent层专家 【后续层与当前层时间间隔更短，重新IO读取的时间压力更大】
        在有先验经验时，驱逐低可能性激活专家 > 驱逐高可能性激活专家
        """
        with self.lock:
            cache_queue = [self.TP_set, self.T1_set, self.T2_set]
            for k in range(queue_type.value, 3):
                q = cache_queue[k]
                total_layers = self.config.hf_config.num_hidden_layers
                for i in [(self.cur_infer_layer_idx - 1 - k) % total_layers for k in range(total_layers - 2)]:
                    layer_experts_in_cache = self.get_cached_experts(i)[1]
                    for e in layer_experts_in_cache:
                        # todo: 进一步按照专家得分，驱逐
                        potential_victim = (i, e)
                        if potential_victim in q:
                            logging.info(f"[cache] {protected_victim} is protected by whitelist, new victim: {potential_victim}")
                            logging.info(f"[cache] evicted from {ARC_QUEUE(k).name}")
                            return potential_victim, ARC_QUEUE(k)
                        
            # 遍历完后还是没有找到合适的驱逐目标
            logging.info(f"protected_victim:{protected_victim}, queue type: {queue_type}")
            logging.info(f"TP: {self.TP}\nT1: {self.T1}\nT2: {self.T2}")
            return None, None
        
    # 根据驱逐策略进行驱逐
    def getVictim(self, queue_type: ARC_QUEUE) -> Tuple[Tuple[int, int], ARC_QUEUE]:
        """
        获取驱逐的专家id
        queue_type: 需要驱逐的队列
        |TP                       |T1                      |T2
        ___________________________________________________________________
        TP（预取进程访问的专家） > T1（最近访问过的专家） > T2（频繁访问过的专家）
        驱逐previous层的专家  > 驱逐subsequent层专家 【后续层与当前层时间间隔更短，重新IO读取的时间压力更大】
        驱逐低分数专家 > 驱逐高分数专家
        """
        with self.lock:
            cache_queue = [self.TP_set, self.T1_set, self.T2_set]
            for k in range(queue_type.value, 3):
                q = cache_queue[k]
                total_layers = self.config.hf_config.num_hidden_layers
                for i in [(self.cur_infer_layer_idx - 1 - k) % total_layers for k in range(total_layers - 2)]:
                    layer_experts_in_cache = self.get_cached_experts(i)[1]
                    experts_in_queue = [e for e in layer_experts_in_cache if (i, e) in q]
                    # logging.info(f"[cache] experts_in_queue: {experts_in_queue}")
                    if experts_in_queue == []: 
                        # 缓存中的layer i的专家都不在当前queue_type中
                        continue
                    
                    layer_experts_score = list(self.experts_score[i])
                    experts_in_queue_score = [layer_experts_score[idx] for idx in experts_in_queue]
                    # logging.info(f"experts_in_queue_score: {experts_in_queue_score}")
                    assert len(experts_in_queue) == len(experts_in_queue_score), logging.error(f"error in changeVictim")
                    
                    min_idx = experts_in_queue_score.index(min(experts_in_queue_score))
                    victim = (i, experts_in_queue[min_idx])
                    
                    logging.info(f"[cache] victim: {victim}")
                    logging.info(f"[cache] evicted from {ARC_QUEUE(k).name}")
                    return victim, ARC_QUEUE(k)
                        
            # 遍历完后还是没有找到合适的驱逐目标
            logging.info(f"[cache] Can't find any victim!\nTP: {self.TP}\nT1: {self.T1}\nT2: {self.T2}")
            return None, None
    
    # def _evict_victim_pro(self) -> Optional[Tuple[int, int]]:
    #     """
    #     新的统一驱逐方法，采用了白名单保护机制。
    #     """
    #     try:
    #         with self.lock:
    #         # 优先级1：从预取队列 TP 的头部 (LRU) 驱逐
    #             if self.TP:
    #                 possible_victim = self.TP[0]
    #                 possible_queue_type = ARC_QUEUE.TP
    #                 if self.isInWhitelist(possible_victim):
    #                     possible_victim, possible_queue_type = self.changeVictim(possible_victim, ARC_QUEUE.TP)
    #                     if possible_victim is None:
    #                         raise Exception(f"cannot find a new victim")
    #                 victim = possible_victim
    #                 queue_type = possible_queue_type
    #                 self._remove_from_specific_queue(queue_type, victim)
    #                 self.evict_expert(*victim)    
    #                 return victim
                
    #             # 优先级2：如果 TP 为空，则回退到 ARC 的标准驱逐逻辑
    #             if self.T1 and len(self.T1) > self.p:
    #                 possible_victim = self.T1[0]
    #                 possible_queue_type = ARC_QUEUE.T1
    #                 if self.isInWhitelist(possible_victim):
    #                     possible_victim, possible_queue_type = self.changeVictim(possible_victim, ARC_QUEUE.T1)
    #                     if possible_victim is None:
    #                         raise Exception(f"cannot find a new victim")
    #                 victim = possible_victim
    #                 queue_type = possible_queue_type
    #                 self._remove_from_specific_queue(queue_type, victim)
                    
    #                 self.evict_expert(*victim)
    #                 self._add_to_queue(self.B1, self.B1_set, victim)
    #                 if len(self.B1) > self.max_experts: 
    #                     b1_victim = self.B1.popleft()
    #                     if b1_victim in self.B1_set:
    #                         self.B1_set.remove(b1_victim)
    #                 return victim
                
    #             elif self.T2:
    #                 possible_victim = self.T2[0]
    #                 possible_queue_type = ARC_QUEUE.T2
    #                 if self.isInWhitelist(possible_victim):
    #                     possible_victim, possible_queue_type = self.changeVictim(possible_victim, ARC_QUEUE.T2)
    #                     if possible_victim is None:
    #                         raise Exception(f"cannot find a new victim")
    #                 victim = possible_victim
    #                 queue_type = possible_queue_type
    #                 self._remove_from_specific_queue(queue_type, victim)
                    
    #                 self.evict_expert(*victim)
    #                 self._add_to_queue(self.B2, self.B2_set, victim)
    #                 if len(self.B2) > self.max_experts: 
    #                     b2_victim = self.B2.popleft()
    #                     if b2_victim in self.B2_set:
    #                         self.B2_set.remove(b2_victim)
    #                 return victim
    #             return None
    #     except Exception as e:
    #         logging.error(e)
    
    def _evict_victim_pro(self) -> Optional[Tuple[int, int]]:
        """
        新的统一驱逐方法，采用了白名单保护机制，并考虑专家得分，综合进行驱逐。
        不再是从队列头部驱逐，替换原生ARC缓存驱逐逻辑
        """
        try:
            with self.lock:
            # 优先级1：从预取队列 TP 的头部 (LRU) 驱逐
                if self.TP:
                    victim, victim_queue_type = self.getVictim(ARC_QUEUE.TP)
                    if victim is None:
                        raise Exception(f"cannot find a new victim")
                    self._remove_from_specific_queue(victim_queue_type, victim)
                    self.evict_expert(*victim)    
                    return victim
                
                # 优先级2：如果 TP 为空，则回退到 ARC 的标准驱逐逻辑
                if self.T1 and len(self.T1) > self.p:
                    victim, victim_queue_type = self.getVictim(ARC_QUEUE.T1)
                    if victim is None:
                        raise Exception(f"cannot find a new victim")
                    self._remove_from_specific_queue(victim_queue_type, victim)
                    self.evict_expert(*victim)
                    
                    self._add_to_queue(self.B1, self.B1_set, victim)
                    if len(self.B1) > self.max_experts: 
                        b1_victim = self.B1.popleft() # todo：B1\B2队列暂时未考虑专家得分，只是从队头驱逐
                        if b1_victim in self.B1_set:
                            self.B1_set.remove(b1_victim)
                    return victim
                
                elif self.T2:
                    victim, victim_queue_type = self.getVictim(ARC_QUEUE.T2)
                    if victim is None:
                        raise Exception(f"cannot find a new victim")
                    self._remove_from_specific_queue(victim_queue_type, victim)
                    self.evict_expert(*victim)
                    
                    self._add_to_queue(self.B2, self.B2_set, victim)
                    if len(self.B2) > self.max_experts: 
                        b2_victim = self.B2.popleft()
                        if b2_victim in self.B2_set:
                            self.B2_set.remove(b2_victim)
                    return victim
                return None
        except Exception as e:
            logging.error(e)
            
    # def _evict_victim(self) -> Optional[Tuple[int, int]]:
    #     """
    #     朴素的统一驱逐方法
    #     """
    #     with self.lock:
    #         # 优先级1：从预取队列 TP 的头部 (LRU) 驱逐
    #         if self.TP:
    #             victim = self.TP[0]
    #             self._remove_from_specific_queue(ARC_QUEUE.TP, victim)
    #             # 注意：从 TP 驱逐的专家不进入幽灵列表，因为它们从未被证实有用
    #             self.evict_expert(*victim)
    #             logging.info(f"evict from TP")
    #             return victim
            
    #         # 优先级2：如果 TP 为空，则回退到 ARC 的标准驱逐逻辑
    #         # (_replace_arc 内部处理 T1/T2 到 B1/B2 的逻辑)
    #         if self.T1 and len(self.T1) > self.p:
    #             victim = self.T1[0]
    #             self._remove_from_specific_queue(ARC_QUEUE.T1, victim)
    #             self.evict_expert(*victim)
    #             logging.info(f"evict from T1")
    #             self.B1.append(victim)
    #             if len(self.B1) > self.max_experts: self.B1.popleft()
    #             return victim
    #         elif self.T2:
    #             victim = self.T2[0]
    #             self._remove_from_specific_queue(ARC_QUEUE.T2, victim)
    #             self.evict_expert(*victim)
    #             logging.info(f"evict from T2")
    #             self.B2.append(victim)
    #             if len(self.B2) > self.max_experts: self.B2.popleft()
    #             return victim
    #         return None
    
    def prefetch_experts(self, layer_id: int, experts_id: List[int]) -> List[Tuple[int, int]]:
        """预取专家到缓存，将专家加入预取队列 TP"""
        evict_list = []
        # experts_list = []
        for eid in experts_id:
            expert = (layer_id, eid)
        
            # 如果已在任何队列中，则不操作
            with self.lock:
                if expert in self.T1_set or expert in self.T2_set or expert in self.TP_set:
                    continue
        
            model = self.get_cpumem_expert(layer_id, eid)
            if model is None:
                # model已经迁移到了gpu中
                continue
            ExpertIOTask(self.IOManager, f"p_{layer_id}_{eid}", 1, model=model, target_addr=self.config.device).start()
        
            if ExpertIOTask.wait_for_task(f"p_{layer_id}_{eid}", 1):
                with self.lock:
                    expert_model = ExpertIOTask.get_result(f"p_{layer_id}_{eid}")
                    # experts_list.append(expert_model)
                    self.model_cache[self._make_key("expert", layer_id, eid)] = expert_model
                    self.experts_cache[layer_id].add(eid)
                    del self.experts_cpumem_cache[self._make_key("expert", layer_id, eid)]
                    evicted_expert = None
                    if self._get_cache_size() >= self.max_experts:
                        evicted_expert = self._evict_victim_pro()
                        logging.info(f"[cache] evict experts {evicted_expert}")
                        evict_list.append(evicted_expert)
                        
                    # self._add_to_queue(self.T1, self.T1_set, (layer_id, eid)) # 验证无TP队列，在具备预取的场景下，缓存命中率的不足
                    self._add_to_queue(self.TP, self.TP_set, expert)
                    
                    # logging.info(f"[cache] cur {self.experts_cache[layer_id]}")
            else:
               logging.error(f"task p_{layer_id}_{eid} timeout")       
        return evict_list
    
    def prefetch_experts_terminable(self, layer_id: int, experts_id: List[int]) -> List[Tuple[int, int]]:
        """
        预取专家到缓存，将专家加入预取队列 TP
        先批量提交IO任务，再批量回收结果，以便支持中途取消
        """
        evict_list = []
        submitted_tasks = [] # 用于记录本次循环提交成功的任务 (eid, task_id)

        # --- 第一阶段：批量提交任务 (Producer Phase) ---
        for eid in experts_id:
            expert = (layer_id, eid)
        
            # 1. 状态检查
            with self.lock:
                if expert in self.T1_set or expert in self.T2_set or expert in self.TP_set:
                    logging.info(f"[cache] {expert} already in cuda")
                    continue
        
            model = self.get_cpumem_expert(layer_id, eid)
            if model is None:
                # model已经迁移到了gpu中或者不存在
                logging.info(f"[cache] {expert} not exist")
                continue
            
            task_id = f"p_{layer_id}_{eid}"
            
            # 2. 提交任务到队列 (非阻塞)
            # 这样所有任务会尽快进入 PriorityQueue，方便外部进行 cancel_tasks_by_prefix
            ExpertIOTask(self.IOManager, task_id, 1, model=model, target_addr=self.config.device).start()
            submitted_tasks.append((eid, task_id))

        # --- 第二阶段：批量处理结果 (Consumer Phase) ---
        for eid, task_id in submitted_tasks:
            # 等待任务完成。如果此时外部调用了取消，wait_for_task 应返回 (False或True视实现而定)
            # 或者任务被取消后，get_result 会返回 None
            if ExpertIOTask.wait_for_task(task_id, timeout=2):
                # 尝试获取结果
                expert_model = ExpertIOTask.get_result(task_id)
                
                # 如果 expert_model 为 None，说明任务执行失败或被"取消逻辑"拦截并删除了
                if expert_model is not None:
                    with self.lock:
                        # Double Check: 再次检查集合，防止并发导致重复添加
                        if (layer_id, eid) in self.TP_set:
                             continue

                        # 更新缓存
                        self.model_cache[self._make_key("expert", layer_id, eid)] = expert_model
                        self.experts_cache[layer_id].add(eid)
                        
                        # 从CPU记录移除
                        key = self._make_key("expert", layer_id, eid)
                        if key in self.experts_cpumem_cache:
                            del self.experts_cpumem_cache[key]
                        
                        # 驱逐逻辑 (Eviction)
                        if self._get_cache_size() >= self.max_experts:
                            evicted_expert = self._evict_victim_pro()
                            if evicted_expert:
                                logging.info(f"[cache] evict experts {evicted_expert}")
                                evict_list.append(evicted_expert)
                        
                        # 加入 TP 队列
                        self._add_to_queue(self.TP, self.TP_set, (layer_id, eid))
                else:
                    # 任务完成但没有结果，通常意味着任务被取消了
                    logging.info(f"Task {task_id} result is None (likely canceled)")
            else:
                logging.error(f"task {task_id} timeout or canceled waiting")
                
        return evict_list
    
    def terminate_prefetch(self, layer_id):
        """
            终止prefetch任务
            
            layer_id：预取层id
        """
        ExpertIOTask.cancel_tasks_by_prefix(f'p_{layer_id}_')
        
    def reset(self):
        super().reset()
        # 重置所有内部状态
        for q in (self.TP, self.T1, self.T2, self.B1, self.B2):
            q.clear()
        for q in (self.TP_set, self.T1_set, self.T2_set, self.B1_set, self.B2_set):
            q.clear()
        self.p = 0.0      
    
    def access_experts(self, layer_id: int, expert_ids: List[int]) -> Tuple[List[nn.Module],List[Tuple[int, int]]]:
        """
        根据一次推理中某个层实际使用的专家列表，来批量访问并更新 ARC 缓存。返回访问专家的引用和驱逐专家信息
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
                # --- Case 1: 命中 T1 或 T2 (标准ARC命中) ---
                if expert in self.T1_set:
                    self._remove_from_queue(self.T1, self.T1_set, expert)
                    self._add_to_queue(self.T2, self.T2_set, expert)
                    continue
                if expert in self.T2_set:
                    self._remove_from_queue(self.T2, self.T2_set, expert)
                    self._add_to_queue(self.T2, self.T2_set, expert)
                    continue
                # --- Case 2: 命中 TP (预取命中，执行“晋升”) ---
                if expert in self.TP_set:
                    self._remove_from_queue(self.TP, self.TP_set, expert)
                    self._add_to_queue(self.T1, self.T1_set, expert)
                    continue
                
            model = self.get_cpumem_expert(layer_id, expert_id)
            if model is None:
                # model已经迁移到了gpu中
                continue
            ExpertIOTask(self.IOManager, f"m_{layer_id}_{expert_id}", 0, model=model, target_addr=self.config.device).start()
            
            with self.lock:
                evicted_expert = None
                # --- Case 3: 完全未命中 (幽灵列表或全新专家) ---
                # (这部分逻辑与原版ARC类似, 但需要使用新的驱逐方法)
                if expert in self.B1_set:
                    delta = max(len(self.B2) / len(self.B1), 1.0) if self.B1 else 1.0
                    self.p = min(self.max_experts, self.p + delta)
                    self._remove_from_queue(self.B1, self.B1_set, expert)
                    if self._get_cache_size() >= self.max_experts: evicted_expert = self._evict_victim_pro()
                    self._add_to_queue(self.T2, self.T2_set, expert)
                elif expert in self.B2_set:
                    delta = max(len(self.B1) / len(self.B2), 1.0) if self.B2 else 1.0
                    self.p = max(0.0, self.p - delta)
                    self._remove_from_queue(self.B2, self.B2_set, expert)
                    if self._get_cache_size() >= self.max_experts: evicted_expert = self._evict_victim_pro()
                    self._add_to_queue(self.T2, self.T2_set, expert)
                else: # 全新专家
                    if self._get_cache_size() >= self.max_experts:
                        evicted_expert = self._evict_victim_pro()
                    self._add_to_queue(self.T1, self.T1_set, expert)
                if evicted_expert is not None:
                    evicted_list.append(evicted_expert)
            
        ExpertIOTask.wait_for_priority(0) 
        
        for expert_id in expert_ids:
            expert = ExpertIOTask.get_result(f"m_{layer_id}_{expert_id}")
            if expert is None:
                # 没有该专家IO任务，该专家在缓存中
                expert_list.append(self.get_model("expert", layer_id, expert_id))
                continue
            # 缓存基本状态更新
            self.model_cache[self._make_key("expert", layer_id, expert_id)] = expert
            self.experts_cache[layer_id].add(expert_id)
            del self.experts_cpumem_cache[self._make_key("expert", layer_id, expert_id)]
            expert_list.append(expert)
             
        if evicted_list:
            logging.info(f"[Cache] evicted experts: {evicted_list}")
        # logging.info(f"load missing experts I/O cost {time.time()-t1}")
        return expert_list, evicted_list

    # 兼容 Hook 调用的状态更新接口
    def update_status(self, layer_idx, active_ids, infer_step=None):
        try:
            step = infer_step if infer_step is not None else getattr(self, "cur_step", 0)
            self.experts_Status_Awareness_Update(layer_idx=layer_idx, infer_step=step, activate_experts_id=set(active_ids))
        except Exception as exc:
            logging.warning(f"update_status failed: layer {layer_idx}, active_ids {active_ids}, err: {exc}")

    def experts_Status_Awareness_Update(self, layer_idx, infer_step, activate_experts_id):
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
        overlap_ratio = self.get_active_overlap_ratio(layer_idx)
        logging.info(f"overlap_ratio: {overlap_ratio}")
        if overlap_ratio >= self.TLA_thredhold:
            ptn = ACCESS_PATTERN.TLA
        elif overlap_ratio <= self.RA_thredhold:
            ptn = ACCESS_PATTERN.RA
        else:
            ptn = ACCESS_PATTERN.Normal
            
        # 基于当前激活模式和更新后的专家状态，更新update_layers的专家的分数
        self.update_experts_score(update_layers=[layer_idx], pattern=ptn)
        logging.info(f"scores for layer{layer_idx}: {self.experts_score[layer_idx]}")
        # import os
        # os._exit(0)
            
    # 驱逐时考虑，专家分数越高，代表其越有可能在近期被使用
    def update_experts_score(self, update_layers: List[int], pattern: ACCESS_PATTERN) -> List[float]:
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
        
    def get_active_overlap_ratio(self, layer_idx) -> float:
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
                
    def set_max_experts(self, max_experts: int) -> None:
        self.max_experts = max_experts

    def set_p(self, p: float) -> None:
        self.p = p
        
    def get_experts_cache(self) -> Dict[int, Set[int]]:
        """
        获取当前缓存的专家状态。

        Returns:
            Dict[int, Set[int]]: 每层的专家 ID 集合。
        """
        return self.experts_cache


        
    