import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class FusionEngine:
    def __init__(self):
        self.default_weights = {
            "reconstruction_error": 0.4,
            "distance_score": 0.3,
            "isolation_forest": 0.3
        }
    
    async def fuse(
        self,
        scores: Dict[str, float],
        method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        融合多个异常检测算法的评分
        
        Args:
            scores: 各算法的评分字典
            method: 融合方法
            weights: 融合权重，None则使用默认或均等权重
            
        Returns:
            融合结果字典
        """
        if not scores:
            raise ValueError("评分字典不能为空")
        
        score_values = np.array(list(scores.values()))
        score_keys = list(scores.keys())
        
        # 归一化评分到[0, 1]范围
        if score_values.max() - score_values.min() > 1e-8:
            score_values = (score_values - score_values.min()) / (score_values.max() - score_values.min())
        else:
            score_values = np.ones_like(score_values) * 0.5
        
        normalized_scores = dict(zip(score_keys, score_values))
        
        # 根据方法选择融合策略
        if method == "weighted_average":
            fused_score = self._weighted_average(normalized_scores, weights, score_keys)
        elif method == "average":
            fused_score = self._average(score_values)
        elif method == "max":
            fused_score = self._max(score_values)
        elif method == "min":
            fused_score = self._min(score_values)
        elif method == "median":
            fused_score = self._median(score_values)
        elif method == "rank_fusion":
            fused_score = self._rank_fusion(normalized_scores, weights, score_keys)
        elif method == "confidence_weighted":
            fused_score = self._confidence_weighted(normalized_scores, weights, score_keys)
        elif method == "dynamic_weight":
            fused_score = self._dynamic_weight(normalized_scores, score_keys)
        else:
            raise ValueError(f"不支持的融合方法: {method}")
        
        # 计算置信度（基于评分的一致性）
        confidence = self._calculate_confidence(score_values)
        
        # 确定异常标签（阈值0.5）
        anomaly_label = fused_score > 0.5
        
        return {
            "fused_score": float(fused_score),
            "confidence": float(confidence),
            "anomaly_label": bool(anomaly_label),
            "method": method
        }
    
    def _weighted_average(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]],
        score_keys: List[str]
    ) -> float:
        """加权平均融合"""
        if weights is None:
            # 使用均等权重
            weights = {k: 1.0 / len(scores) for k in score_keys}
        
        # 归一化权重
        total_weight = sum(weights.get(k, 0) for k in score_keys)
        if total_weight == 0:
            weights = {k: 1.0 / len(scores) for k in score_keys}
            total_weight = 1.0
        
        weighted_sum = sum(scores[k] * weights.get(k, 0) for k in score_keys)
        return weighted_sum / total_weight
    
    def _average(self, scores: np.ndarray) -> float:
        """简单平均融合"""
        return float(np.mean(scores))
    
    def _max(self, scores: np.ndarray) -> float:
        """最大值融合（保守策略）"""
        return float(np.max(scores))
    
    def _min(self, scores: np.ndarray) -> float:
        """最小值融合（宽松策略）"""
        return float(np.min(scores))
    
    def _median(self, scores: np.ndarray) -> float:
        """中位数融合"""
        return float(np.median(scores))
    
    def _rank_fusion(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]],
        score_keys: List[str]
    ) -> float:
        """排序融合"""
        # 计算每个算法的排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if weights is None:
            weights = {k: 1.0 for k in score_keys}
        
        # 加权排序分数
        rank_sum = 0.0
        total_weight = 0.0
        
        for rank, (key, score) in enumerate(sorted_scores):
            weight = weights.get(key, 1.0)
            # 排名越高（rank越小），权重越大
            rank_score = (len(scores) - rank) / len(scores)
            rank_sum += rank_score * weight * score
            total_weight += weight
        
        if total_weight == 0:
            return self._average(np.array(list(scores.values())))
        
        return rank_sum / total_weight
    
    def _confidence_weighted(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]],
        score_keys: List[str]
    ) -> float:
        """基于置信度的加权融合"""
        score_values = np.array([scores[k] for k in score_keys])
        
        # 使用评分的标准差作为置信度的反向指标
        std_dev = np.std(score_values)
        confidence_scores = 1.0 / (1.0 + std_dev)  # 标准差越小，置信度越高
        
        # 归一化置信度作为权重
        if weights is None:
            weights = {k: confidence_scores for k in score_keys}
        else:
            # 结合用户权重和置信度
            for i, k in enumerate(score_keys):
                weights[k] = weights.get(k, 1.0) * confidence_scores
        
        return self._weighted_average(scores, weights, score_keys)
    
    def _dynamic_weight(
        self,
        scores: Dict[str, float],
        score_keys: List[str]
    ) -> float:
        """动态权重融合（根据评分分布自动调整）"""
        score_values = np.array([scores[k] for k in score_keys])
        
        # 计算每个评分相对于均值的偏差
        mean_score = np.mean(score_values)
        deviations = np.abs(score_values - mean_score)
        
        # 偏差越大，权重越大（表示该算法更确信）
        if np.sum(deviations) > 1e-8:
            weights = deviations / np.sum(deviations)
        else:
            weights = np.ones_like(score_values) / len(score_values)
        
        weight_dict = dict(zip(score_keys, weights))
        return self._weighted_average(scores, weight_dict, score_keys)
    
    def _calculate_confidence(self, scores: np.ndarray) -> float:
        """
        计算融合置信度
        基于评分的一致性（标准差的反函数）
        """
        if len(scores) < 2:
            return 1.0
        
        std_dev = np.std(scores)
        # 标准差越小，置信度越高
        confidence = 1.0 / (1.0 + std_dev * 2)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def list_methods(self) -> List[str]:
        """列出所有支持的融合方法"""
        return [
            "weighted_average",
            "average",
            "max",
            "min",
            "median",
            "rank_fusion",
            "confidence_weighted",
            "dynamic_weight"
        ]

