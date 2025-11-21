import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
from typing import List, Dict, Optional


class StatsCalculator:
    async def calculate(
        self,
        data,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算时间序列统计指标
        Args:
            data: TimeSeriesData对象
            metrics: 要计算的指标列表，None则计算所有
        Returns:
            统计指标字典
        """
        values = np.array(data.values)
        
        if len(values) < 2:
            raise ValueError("时间序列数据至少需要2个数据点")
        
        # 所有可用的指标
        all_metrics = {
            "mean": self._mean,
            "median": self._median,
            "std": self._std,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "range": self._range,
            "skewness": self._skewness,
            "kurtosis": self._kurtosis,
            "q1": self._q1,
            "q3": self._q3,
            "iqr": self._iqr,
            "autocorr": self._autocorr,
            "trend": self._trend,
            "volatility": self._volatility,
            "entropy": self._entropy
        }
        
        # 如果指定了指标，只计算指定的
        if metrics:
            metrics_to_calc = {k: v for k, v in all_metrics.items() if k in metrics}
        else:
            metrics_to_calc = all_metrics
        
        results = {}
        for metric_name, metric_func in metrics_to_calc.items():
            try:
                results[metric_name] = metric_func(values)
            except Exception as e:
                results[metric_name] = None  # 计算失败则返回None
        
        return {"metrics": results}
    
    def _mean(self, values: np.ndarray) -> float:
        """均值"""
        return float(np.mean(values))
    
    def _median(self, values: np.ndarray) -> float:
        """中位数"""
        return float(np.median(values))
    
    def _std(self, values: np.ndarray) -> float:
        """标准差"""
        return float(np.std(values))
    
    def _variance(self, values: np.ndarray) -> float:
        """方差"""
        return float(np.var(values))
    
    def _min(self, values: np.ndarray) -> float:
        """最小值"""
        return float(np.min(values))
    
    def _max(self, values: np.ndarray) -> float:
        """最大值"""
        return float(np.max(values))
    
    def _range(self, values: np.ndarray) -> float:
        """极差"""
        return float(np.max(values) - np.min(values))
    
    def _skewness(self, values: np.ndarray) -> float:
        """偏度"""
        if len(values) < 3:
            return 0.0
        return float(stats.skew(values))
    
    def _kurtosis(self, values: np.ndarray) -> float:
        """峰度"""
        if len(values) < 4:
            return 0.0
        return float(stats.kurtosis(values))
    
    def _q1(self, values: np.ndarray) -> float:
        """第一四分位数"""
        return float(np.percentile(values, 25))
    
    def _q3(self, values: np.ndarray) -> float:
        """第三四分位数"""
        return float(np.percentile(values, 75))
    
    def _iqr(self, values: np.ndarray) -> float:
        """四分位距"""
        return float(np.percentile(values, 75) - np.percentile(values, 25))
    
    def _autocorr(self, values: np.ndarray) -> float:
        """一阶自相关"""
        if len(values) < 2:
            return 0.0
        try:
            autocorr_values = acf(values, nlags=1, fft=False)
            return float(autocorr_values[1]) if len(autocorr_values) > 1 else 0.0
        except:
            # 如果计算失败，手动计算
            if len(values) < 2:
                return 0.0
            mean = np.mean(values)
            shifted = values[1:] - mean
            original = values[:-1] - mean
            if np.std(original) == 0:
                return 0.0
            return float(np.corrcoef(original, shifted)[0, 1])
    
    def _trend(self, values: np.ndarray) -> float:
        """趋势（线性回归斜率）"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    
    def _volatility(self, values: np.ndarray) -> float:
        """波动率（收益率的标准差）"""
        if len(values) < 2:
            return 0.0
        returns = np.diff(values) / values[:-1]
        return float(np.std(returns))
    
    def _entropy(self, values: np.ndarray) -> float:
        """信息熵（离散化后计算）"""
        if len(values) < 2:
            return 0.0
        try:
            # 将连续值离散化为10个区间
            hist, _ = np.histogram(values, bins=10)
            hist = hist[hist > 0]  # 移除0值
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            return float(entropy)
        except:
            return 0.0
    
    def list_metrics(self) -> List[str]:
        """列出所有支持的统计指标"""
        return [
            "mean",
            "median",
            "std",
            "variance",
            "min",
            "max",
            "range",
            "skewness",
            "kurtosis",
            "q1",
            "q3",
            "iqr",
            "autocorr",
            "trend",
            "volatility",
            "entropy"
        ]

