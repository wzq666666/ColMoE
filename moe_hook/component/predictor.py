from typing import Set, Callable, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from collections import deque
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    def cosine_similarity(a, b):
        """Lightweight cosine similarity fallback to avoid sklearn dependency."""
        a2 = np.atleast_2d(a)
        b2 = np.atleast_2d(b)
        a_norm = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-8)
        b_norm = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        return a_norm @ b_norm.T
import logging


def predict(top_k=None, cfg=None, gate=None, input_gate_i=None, mode=None, **kwargs):
    """统一预测入口，根据配置选择策略。

    支持的 mode:
      - fate: 使用下一层 gate 做 FATE 预测（需 gate 和 input_gate_i）
      - topk: 直接返回当前层 topk_ids 作为预测结果
      - none: 不做预测，返回 None
    """

    mode = mode or (cfg.get('predict', {}).get('mode') if isinstance(cfg, dict) else None) or 'fate'

    if mode == 'none':
        return None

    if mode == 'fate':
        # 需要 gate 和 gate 输入（通常是上一层 combine_input）
        if gate is None:
            logging.warning("[PREDICT] fate mode requires gate; got None, skip")
            return None
        if input_gate_i is None:
            logging.warning("[PREDICT] fate mode requires input_gate_i; got None, skip")
            return None
        topk_ids, topk_weights, routing_probs = fate_predictor(top_k, gate, input_gate_i)
        return topk_ids, topk_weights, routing_probs

    logging.warning(f"[PREDICT] unknown mode {mode}, skip")
    return None

def fate_predictor(topk, gate, input_gate_i) -> Tuple[Set[int], torch.Tensor, Dict[int, float]]:
    """
    模拟FATE预测器的函数，将当前层gate的输入送到下一层gate，获取下一层的预取专家。
    
    :param gate: 下一层的gate
    :param input_gate_i: 当前层gate的输入 [T, D] 或 [B, T, D]
    :return: (unique_expert_ids, selected_experts, expert_scores)
             expert_scores: 每个专家的平均激活概率
    """
    with torch.no_grad():
        # 确保 gate 权重和输入数据类型一致
        if hasattr(gate, "weight"):
            input_gate_i = input_gate_i.to(gate.weight.dtype)
        router_logits = gate(input_gate_i)
    
    # 处理不同维度的情况
    # router_logits 可能是 [T, num_experts] 或 [B, T, num_experts]
    if router_logits.dim() == 3:
        # [B, T, num_experts] -> [B*T, num_experts]
        router_logits = router_logits.view(-1, router_logits.size(-1))
    
    # softmax 在最后一个维度（专家维度）上
    routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    
    # 确保 topk 不超过专家数
    num_experts = routing_probs.size(-1)
    actual_topk = min(topk, num_experts)
    
    topk_weights, topk_ids = torch.topk(routing_probs, actual_topk, dim=-1)
    
    return topk_ids, topk_weights, routing_probs

def neuron_predictor(config, hidden_states, selected_experts, predict_res, threshold=1e-2):
    """
        采用类似Fate的思想，利用hs各层之间的高度相似性，直接把当前层的hs
        喂给后续层的专家，来预测后续层的激活神经元
        返回（专家id， 激活的神经元索引, 专家的激活函数，专家的gate proj）
    """
    expert_mask = F.one_hot(
        selected_experts, num_classes=config.hf_config.num_experts
    ).permute(2, 1, 0)
    res = []
    for i, (expert_idx, act_fn, gate_proj) in enumerate(predict_res):
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[top_x] # [N, intermediate_size]
        activations = act_fn(gate_proj(current_state))  # [N, intermediate_size]
        # mask = torch.abs(activations) > threshold # 得到[intermediate_size]的布尔掩码
        # active_indices = torch.where(mask)[0]  # 活跃神经元的索引
        
        # 聚合N维度（token维度），判断神经元是否在任一token中活跃
        # 对每个神经元，取所有token的激活值绝对值的最大值，再与阈值比较
        max_activations = torch.max(torch.abs(activations), dim=0).values  # 形状：[intermediate_size]
        mask = max_activations > threshold  # 形状：[intermediate_size]（全局活跃判断）
        active_indices = torch.where(mask)[0]
        res.append((expert_idx, active_indices, gate_proj, act_fn))
        # logging.info(f"专家{expert_idx}的全局活跃神经元索引: {active_indices.tolist()}")
    return res

class DoubleSparsityPredictor:
    def __init__(self, config, expert_predictor: Callable, neuron_predictor: Callable):
        self.config = config
        self.expert_predictor = expert_predictor
        self.neuron_predictor = neuron_predictor
        
    def predict_experts(self, gate, input_gate_i):
        """
        模拟FATE预测器的函数，将当前层gate的输入送到下一层gate，获取下一层的预取专家。
        
        :param layer_idx: 当前层的索引
        :param input_gate_i: 当前层gate的输入 [T, D]
        :param model_cache_on_device: device模型缓存
        """
        return self.expert_predictor(self.config, gate, input_gate_i)
        
    def predict_neuron(self, hidden_states, selected_experts, predict_res, threshold=1e-3):
        """
            采用类似Fate的思想，利用hs各层之间的高度相似性，直接把当前层的hs
            喂给后续层的专家，来预测后续层的激活神经元
            返回（专家id， 激活的神经元索引）
        """
        return self.neuron_predictor(self.config, hidden_states, selected_experts, predict_res, threshold)
    
class EAMPredictor:
    def __init__(self, capacity, layer_num, expert_num_per_layer, topk, similarity_threshold=0.8):
        self.capacity = capacity
        self.layer_num = layer_num
        self.expert_num_per_layer = expert_num_per_layer
        self.topk = topk
        self.similarity_threshold = similarity_threshold
        
        self.EAMC = deque(maxlen=capacity)
        
        # 用于跟踪当前请求的 rEAM 和 iEAM
        self.rEAM = np.zeros((layer_num, expert_num_per_layer), dtype=np.int32)
        self.iEAM = np.zeros((layer_num, expert_num_per_layer), dtype=np.int32)
        
    def start_new_request(self):
        """在一个新的请求开始时调用，重置rEAM"""
        self.rEAM = np.zeros((self.layer_num, self.expert_num_per_layer), dtype=np.int32)

    def end_of_request(self):
        """在一个请求结束后调用，将最终的rEAM存入EAMC"""
        if not self.EAMC.maxlen or len(self.EAMC) < self.EAMC.maxlen:
            self.EAMC.append(self.rEAM.copy())
        else:
            # 实现“替换最相似”的逻辑
            new_eam_flat = self.rEAM.flatten()
            min_similarity = -1
            most_similar_idx = -1
            
            # 找到与新rEAM最相似的旧rEAM
            for i, old_eam in enumerate(self.EAMC):
                old_eam_flat = old_eam.flatten()
                # sklearn的cosine_similarity返回[[value]], 需要索引
                sim = cosine_similarity(new_eam_flat.reshape(1, -1), old_eam_flat.reshape(1, -1))[0][0]
                if sim > min_similarity:
                    min_similarity = sim
                    most_similar_idx = i
            
            # 替换掉最相似的那个
            if most_similar_idx != -1:
                self.EAMC[most_similar_idx] = self.rEAM.copy()
                
    def normalize_eam(self, eam):
        """对EAM按层进行归一化处理"""
        normalized = np.zeros_like(eam, dtype=np.float32)
        for layer in range(self.layer_num):
            total = eam[layer].sum()
            if total > 0:
                normalized[layer] = eam[layer] / total
        return normalized
    
    def update(self, layer_idx, selected_experts: np.ndarray):
        """
        更新iEAM，并返回下一层的预测专家
        """
        if layer_idx == 0: # 第一层，先重置iEAM
            self.iEAM = np.zeros((self.layer_num, self.expert_num_per_layer), dtype=np.int32)
        for experts_id in selected_experts:
            for expert_id in experts_id:
                if 0 <= expert_id < self.expert_num_per_layer:
                    self.iEAM[layer_idx, expert_id] += 1
        
        predict_experts = self.matchAndPredict(layer_idx)
        # logging.info(f"predict_experts is {predict_experts}")
        # 如果是最后一层，将本次迭代的iEAM累加到rEAM中
        if layer_idx == self.layer_num - 1: # 最后一层，将iEAM加入EAMC
            self.rEAM += self.iEAM
            
        # 返回下一层的预测结果
        next_layer_idx = layer_idx + 1
        if next_layer_idx < self.layer_num:
            return predict_experts[next_layer_idx]
        else:
            return [] # 没有下一层了
    
    def matchAndPredict(self, layer_idx) -> np.ndarray:
        if not self.EAMC or len(self.EAMC) == 0:
            return np.array([[] for _ in range(self.layer_num)])
        # 匹配iEAM和rEAM，选出匹配的rEAMs
        iEAM = self.iEAM[:layer_idx+1]
        logging.info(f'cur iEAM :{iEAM}')
        flatten_iEAM = iEAM.flatten().reshape(1, -1)
        # logging.info(f'flatten_iEAM :{flatten_iEAM}')
        
        historical_rEAMs_flat = np.array([r[:layer_idx+1].flatten() for r in self.EAMC])
        # logging.info(f"historical_rEAMs_flat[0] is {historical_rEAMs_flat[0]}")
        similarities = cosine_similarity(flatten_iEAM, historical_rEAMs_flat)[0]
        logging.info(similarities)
        matched_indices = np.where(similarities >= self.similarity_threshold)[0]
        
        if matched_indices.size == 0:
            logging.info("[PREDICTOR] No valid rEAMs found for prediction.")
            return np.array([[] for _ in range(self.layer_num)])
        
        valid_rEAMs = [self.EAMC[i] for i in matched_indices]
        # 聚合选出的rEAMs，得到预测结果
        aggregated_rEAM = np.sum(valid_rEAMs, axis=0, dtype=np.float32)
        aggregated_rEAM = self.normalize_eam(aggregated_rEAM)
        # 基于层邻近性调整预测结果
        for i in range(layer_idx + 1, self.layer_num):
            proximity_multiplier = 1.0 - (float(i - layer_idx) / self.layer_num)
            aggregated_rEAM[i] *= proximity_multiplier

        topk_experts_indices = np.argsort(aggregated_rEAM, axis=1)[:, -self.topk:]
        return np.flip(topk_experts_indices, axis=1)
    
    def save_EAMC_to_disk(self, save_dir):
        try:
            with open(save_dir, 'wb') as f:
                # 保存deque中的所有数据
                pickle.dump(list(self.EAMC), f)
            print(f"EAMC已成功保存到 {save_dir}")
        except Exception as e:
            print(f"保存失败: {e}")
            
    def load_EAMC_from_disk(self, save_dir):
        try:
            with open(save_dir, 'rb') as f:
                data_list = pickle.load(f)
                # 重新初始化deque并恢复数据
                self.EAMC = deque(data_list, maxlen=self.capacity)
            print(f"EAMC已从 {save_dir} 加载成功")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {save_dir}")
        except Exception as e:
            print(f"加载失败: {e}")
    
class wzq_predictor:
    def __init__(self, num_layers, num_experts_per_layer, history_window=10):
        self.num_layers = num_layers
        self.num_experts = num_experts_per_layer
        self.history_window = history_window
        
        #
        # self.
class LayerRelationModel:
    def __init__(self, num_layers, num_experts_per_layer, history_window=10):
        self.num_layers = num_layers
        self.num_experts = num_experts_per_layer
        self.history_window = history_window
        
        # 核心数据结构
        self.transition_matrices = {}  # 层间转移概率矩阵
        self.similarity_models = {}    # 层间相似度模型
        self.expert_cooccurrence = {}  # 专家共现统计
        self.performance_history = {}  # 预测性能历史
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化所有层间关系模型"""
        for src_layer in range(self.num_layers - 1):
            # 转移概率矩阵：src_layer -> dst_layer 的专家激活转移
            self.transition_matrices[src_layer] = {
                'target_layer': src_layer + 1,
                'matrix': torch.ones(self.num_experts, self.num_experts) / self.num_experts,
                'update_count': 0
            }
            
            # 相似度模型：预测层间特征相似度
            self.similarity_models[src_layer] = {
                'current_similarity': 0.5,
                'similarity_trend': [],  # 相似度变化趋势
                'stability_score': 1.0   # 关系稳定性得分
            }
            
            # 专家共现统计
            self.expert_cooccurrence[src_layer] = torch.zeros(
                self.num_experts, self.num_experts
            )
    
    def update_transition_matrix(self, src_layer, src_experts, dst_experts, 
                           step_idx, weight=1.0):
        """
        更新层间专家转移概率矩阵
        
        Args:
            src_layer: 源层索引
            src_experts: 源层激活的专家(one-hot或概率分布)
            dst_experts: 目标层激活的专家
            step_idx: 时间步(用于衰减权重)
            weight: 更新权重
        """
        if src_layer not in self.transition_matrices:
            return
        
        matrix_info = self.transition_matrices[src_layer]
        
        # 时间衰减因子：越近的数据权重越高
        time_decay = np.exp(-step_idx / self.history_window)
        effective_weight = weight * time_decay
        
        # 更新转移矩阵
        if len(src_experts.shape) == 1:
            # one-hot向量
            src_exp = src_experts.unsqueeze(1)
            dst_exp = dst_experts.unsqueeze(0)
            update = src_exp @ dst_exp
        else:
            # 概率分布
            update = src_experts.unsqueeze(2) @ dst_experts.unsqueeze(1)
            update = update.mean(dim=0)  # 批次平均
        
        # 指数移动平均更新
        old_matrix = matrix_info['matrix']
        new_matrix = (1 - effective_weight) * old_matrix + effective_weight * update
        
        # 行归一化(保证每行是概率分布)
        row_sums = new_matrix.sum(dim=1, keepdim=True)
        new_matrix = new_matrix / row_sums.clamp(min=1e-8)
        
        matrix_info['matrix'] = new_matrix
        matrix_info['update_count'] += 1
        
        # 同时更新专家共现统计
        self._update_expert_cooccurrence(src_layer, src_experts, dst_experts)
        
    def update_similarity_model(self, src_layer, current_similarity, features_src, features_dst):
        """
        更新层间相似度模型
        
        Args:
            src_layer: 源层索引
            current_similarity: 当前层间特征相似度
            features_src: 源层特征(用于分析变化模式)
            features_dst: 目标层特征
        """
        if src_layer not in self.similarity_models:
            return
        
        model_info = self.similarity_models[src_layer]
        
        # 更新相似度趋势
        model_info['similarity_trend'].append(current_similarity)
        if len(model_info['similarity_trend']) > self.history_window:
            model_info['similarity_trend'] = model_info['similarity_trend'][-self.history_window:]
        
        # 计算稳定性得分(基于相似度变化的标准差)
        if len(model_info['similarity_trend']) >= 3:
            trend_array = np.array(model_info['similarity_trend'])
            stability = 1.0 / (1.0 + np.std(trend_array))
            model_info['stability_score'] = stability
        
        model_info['current_similarity'] = current_similarity
        
        # 分析特征变化模式
        self._analyze_feature_dynamics(src_layer, features_src, features_dst)

    def _analyze_feature_dynamics(self, src_layer, features_src, features_dst):
        """分析特征动态变化模式"""
        # 计算特征变化率
        if hasattr(features_src, 'shape') and hasattr(features_dst, 'shape'):
            feature_change = torch.norm(features_dst - features_src) / torch.norm(features_src)
            
            # 记录特征变化模式
            if 'feature_changes' not in self.similarity_models[src_layer]:
                self.similarity_models[src_layer]['feature_changes'] = []
            
            self.similarity_models[src_layer]['feature_changes'].append(
                feature_change.item()
            )
    
    def _calculate_prediction_accuracy(self, predict, real):
        predict = set(predict)
        real = set(real)
        right_predict = real.intersection(predict)
        return len(right_predict) / len(real) if len(real) != 0 else 1
        
    def update_with_performance_feedback(self, src_layer, predicted_experts, 
                                   actual_experts, prediction_confidence):
        """
        根据预测性能反馈更新模型
        """
        # 计算预测准确率
        accuracy = self._calculate_prediction_accuracy(predicted_experts, actual_experts)
        
        # 记录性能历史
        if src_layer not in self.performance_history:
            self.performance_history[src_layer] = []
        
        self.performance_history[src_layer].append({
            'accuracy': accuracy,
            'confidence': prediction_confidence,
            'timestamp': len(self.performance_history[src_layer])
        })
        
        # 如果性能持续不佳，调整学习率
        recent_performance = self.performance_history[src_layer][-5:]
        if len(recent_performance) >= 5:
            avg_accuracy = np.mean([p['accuracy'] for p in recent_performance])
            if avg_accuracy < 0.7:  # 性能阈值
                # 增加学习率，更快适应变化
                self._adjust_learning_rate(src_layer, increase=True)

    def _adjust_learning_rate(self, src_layer, increase=True):
        """动态调整学习率"""
        # 这里可以实现在转移矩阵更新时使用动态学习率
        pass
    
    
    def predict_experts(self, src_layer, current_experts, method='combined'):
        """
        预测下一层激活的专家
        
        Args:
            src_layer: 当前层索引
            current_experts: 当前层专家激活情况
            method: 预测方法('transition', 'similarity', 'combined')
        """
        if src_layer >= self.num_layers - 1:
            return None  # 最后一层没有下一层
        
        if method == 'transition':
            return self._predict_by_transition(src_layer, current_experts)
        elif method == 'similarity':
            return self._predict_by_similarity(src_layer, current_experts)
        else:  # combined
            return self._predict_combined(src_layer, current_experts)

    def _predict_by_transition(self, src_layer, current_experts):
        """基于转移矩阵的预测"""
        transition_matrix = self.transition_matrices[src_layer]['matrix']
        
        if len(current_experts.shape) == 1:
            # one-hot向量：矩阵乘法
            predicted_probs = current_experts @ transition_matrix
        else:
            # 概率分布：加权平均
            predicted_probs = (current_experts.unsqueeze(1) * transition_matrix.unsqueeze(0))
            predicted_probs = predicted_probs.sum(dim=1)
        
        return predicted_probs

    def _predict_by_similarity(self, src_layer, current_experts):
        """基于相似度模型的预测"""
        similarity_info = self.similarity_models[src_layer]
        stability = similarity_info['stability_score']
        
        # 稳定性高的层间关系更可靠
        if stability > 0.8:
            # 使用历史模式增强预测
            return self._enhanced_prediction(src_layer, current_experts, stability)
        else:
            # 不稳定关系，使用保守预测
            return self._conservative_prediction(src_layer, current_experts)

    def _predict_combined(self, src_layer, current_experts):
        """组合预测"""
        trans_pred = self._predict_by_transition(src_layer, current_experts)
        sim_pred = self._predict_by_similarity(src_layer, current_experts)
        
        # 动态权重：基于模型置信度
        trans_confidence = self._calculate_transition_confidence(src_layer)
        sim_confidence = self.similarity_models[src_layer]['stability_score']
        
        total_confidence = trans_confidence + sim_confidence
        trans_weight = trans_confidence / total_confidence
        sim_weight = sim_confidence / total_confidence
        
        combined_pred = trans_weight * trans_pred + sim_weight * sim_pred
        return combined_pred
    
    
    def evaluate_prediction_confidence(self, src_layer, predicted_experts):
        """
        评估预测结果的置信度
        """
        confidence_metrics = {}
        
        # 1. 转移矩阵置信度
        transition_info = self.transition_matrices[src_layer]
        update_count = transition_info['update_count']
        matrix_confidence = min(1.0, update_count / 100.0)  # 基于更新次数
        
        # 2. 预测分布的集中度
        entropy = -torch.sum(predicted_experts * torch.log(predicted_experts + 1e-8))
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float))
        concentration = 1.0 - (entropy / max_entropy)  # 分布集中度
        
        # 3. 历史性能置信度
        if src_layer in self.performance_history:
            recent_acc = [p['accuracy'] for p in self.performance_history[src_layer][-5:]]
            if recent_acc:
                performance_confidence = np.mean(recent_acc)
            else:
                performance_confidence = 0.5
        else:
            performance_confidence = 0.5
        
        # 综合置信度
        overall_confidence = (matrix_confidence * 0.3 + 
                            concentration * 0.4 + 
                            performance_confidence * 0.3)
        
        confidence_metrics = {
            'overall': overall_confidence,
            'matrix_confidence': matrix_confidence,
            'distribution_concentration': concentration,
            'performance_confidence': performance_confidence
        }
        
        return confidence_metrics
        
        