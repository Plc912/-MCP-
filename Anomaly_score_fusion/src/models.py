from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class TimeSeriesData(BaseModel):
    """时间序列数据"""
    timestamps: List[str] = Field(..., description="时间戳列表")
    values: List[float] = Field(..., description="数值列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class AnomalyDetectionRequest(BaseModel):
    """异常检测请求"""
    data: TimeSeriesData = Field(..., description="时间序列数据")
    method: str = Field("isolation_forest", description="异常检测方法")
    params: Optional[Dict[str, Any]] = Field(None, description="算法参数")


class AnomalyDetectionResponse(BaseModel):
    """异常检测响应"""
    scores: List[float] = Field(..., description="异常评分列表")
    labels: List[bool] = Field(..., description="异常标签列表")
    method: str = Field(..., description="使用的检测方法")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class StatsCalculationRequest(BaseModel):
    """统计指标计算请求"""
    data: TimeSeriesData = Field(..., description="时间序列数据")
    metrics: Optional[List[str]] = Field(None, description="要计算的指标列表，为空则计算所有")


class StatsCalculationResponse(BaseModel):
    """统计指标计算响应"""
    metrics: Dict[str, float] = Field(..., description="统计指标结果")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FusionRequest(BaseModel):
    """评分融合请求"""
    algorithm_scores: Dict[str, float] = Field(..., description="各算法的评分字典")
    fusion_method: str = Field("weighted_average", description="融合方法")
    weights: Optional[Dict[str, float]] = Field(None, description="融合权重")


class FusionResponse(BaseModel):
    """评分融合响应"""
    fused_score: float = Field(..., description="融合后的评分")
    confidence: float = Field(..., description="融合置信度")
    anomaly_label: bool = Field(..., description="异常标签")
    method: str = Field(..., description="使用的融合方法")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    data_point_id: Optional[int] = Field(None, description="数据点ID（批量处理时使用）")


class DataPoint(BaseModel):
    """数据点"""
    id: int = Field(..., description="数据点ID")
    algorithm_scores: Dict[str, float] = Field(..., description="各算法的评分字典")


class BatchFusionRequest(BaseModel):
    """批量融合请求"""
    data_points: List[DataPoint] = Field(..., description="数据点列表")
    fusion_method: str = Field("weighted_average", description="融合方法")
    weights: Optional[Dict[str, float]] = Field(None, description="融合权重")

