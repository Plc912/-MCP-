"""
时间序列异常检测MCP服务器
使用FastMCP框架封装，支持SSE协议和MCP标准接口
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastmcp import FastMCP

from src.anomaly_detector import AnomalyDetector
from src.stats_calculator import StatsCalculator
from src.fusion_engine import FusionEngine
from src.csv_loader import load_timeseries_from_csv

# 创建FastMCP实例
mcp = FastMCP("anomaly-fusion-mcp", debug=True, log_level="INFO")

# 全局实例（延迟初始化）
anomaly_detector = None
stats_calculator = None
fusion_engine = None


def _init_engines():
    """延迟初始化引擎"""
    global anomaly_detector, stats_calculator, fusion_engine
    if anomaly_detector is None:
        anomaly_detector = AnomalyDetector()
        stats_calculator = StatsCalculator()
        fusion_engine = FusionEngine()


@mcp.tool()
async def detect_anomaly(
    timestamps: List[str],
    values: List[float],
    method: str = "isolation_forest",
    contamination: Optional[float] = None,
    n_estimators: Optional[int] = None,
    threshold: Optional[float] = None,
    n_neighbors: Optional[int] = None,
) -> Dict[str, Any]:
    """
    对已有的时间序列数据（时间戳列表和数值列表）执行异常检测。
    
    注意：如果数据在CSV文件中，建议使用 detect_anomaly_from_csv 工具，它会自动读取CSV文件。
    
    参数:
    - timestamps: List[str] - 时间戳列表（例如：["2024-01-01T00:00:00", "2024-01-01T01:00:00", ...]）
    - values: List[float] - 对应的数值列表（例如：[1.2, 1.5, 2.8, 1.3, ...]）
    - method: str - 检测方法，可选: isolation_forest, lof, knn, hbos, z_score, iqr, statistical
    - contamination: Optional[float] - 异常比例（某些算法使用）
    - n_estimators: Optional[int] - 树的数量（Isolation Forest使用）
    - threshold: Optional[float] - 阈值（Z-score、Statistical使用）
    - n_neighbors: Optional[int] - 邻居数量（LOF、KNN使用）
    
    返回:
    - Dict包含: scores (List[float]), labels (List[bool]), method (str), anomaly_count (int), total_points (int)
    """
    _init_engines()
    
    params = {}
    if contamination is not None:
        params["contamination"] = contamination
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if threshold is not None:
        params["threshold"] = threshold
    if n_neighbors is not None:
        params["n_neighbors"] = n_neighbors
    
    # 构造TimeSeriesData对象
    from src.models import TimeSeriesData
    data = TimeSeriesData(timestamps=timestamps, values=values)
    
    # 执行检测
    result = await anomaly_detector.detect(data, method, params)
    
    return {
        "status": "ok",
        "method": method,
        "scores": result["scores"],
        "labels": result["labels"],
        "anomaly_count": sum(result["labels"]),
        "total_points": len(result["labels"])
    }


@mcp.tool()
async def calculate_stats(
    timestamps: List[str],
    values: List[float],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    计算时间序列统计指标
    
    参数:
    - timestamps: List[str] - 时间戳列表
    - values: List[float] - 数值列表
    - metrics: Optional[List[str]] - 要计算的指标列表，为空则计算所有
        可选: mean, median, std, variance, min, max, range, skewness, kurtosis,
             q1, q3, iqr, autocorr, trend, volatility, entropy
    
    返回:
    - Dict包含: metrics (Dict[str, float])
    """
    _init_engines()
    
    from src.models import TimeSeriesData
    data = TimeSeriesData(timestamps=timestamps, values=values)
    
    result = await stats_calculator.calculate(data, metrics)
    
    return {
        "status": "ok",
        "metrics": result["metrics"],
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool()
async def fuse_scores(
    algorithm_scores: Dict[str, float],
    fusion_method: str = "weighted_average",
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    融合多个异常检测算法的评分
    
    参数:
    - algorithm_scores: Dict[str, float] - 各算法的评分字典，如 {"reconstruction_error": 0.85, "distance_score": 0.72}
    - fusion_method: str - 融合方法，可选: weighted_average, average, max, min, median, rank_fusion, confidence_weighted, dynamic_weight
    - weights: Optional[Dict[str, float]] - 融合权重（用于weighted_average等方法）
    
    返回:
    - Dict包含: fused_score (float), confidence (float), anomaly_label (bool), method (str)
    """
    _init_engines()
    
    result = await fusion_engine.fuse(algorithm_scores, fusion_method, weights)
    
    return {
        "status": "ok",
        **result
    }


@mcp.tool()
async def batch_fuse_scores(
    data_points: List[Dict[str, Any]],
    fusion_method: str = "weighted_average",
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    批量融合多个数据点的异常评分
    
    参数:
    - data_points: List[Dict[str, Any]] - 数据点列表，每个包含id和algorithm_scores
        示例: [{"id": 1, "algorithm_scores": {"algo1": 0.8, "algo2": 0.7}}, ...]
    - fusion_method: str - 融合方法
    - weights: Optional[Dict[str, float]] - 融合权重
    
    返回:
    - Dict包含: results (List[Dict])
    """
    _init_engines()
    
    results = []
    for item in data_points:
        result = await fusion_engine.fuse(
            item["algorithm_scores"],
            fusion_method,
            weights
        )
        results.append({
            "id": item.get("id"),
            **result
        })
    
    return {
        "status": "ok",
        "count": len(results),
        "results": results
    }


@mcp.tool()
async def load_timeseries_csv(
    path: str,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
    limit: Optional[int] = None,
    dropna: bool = True,
) -> Dict[str, Any]:
    """
    从CSV文件读取并加载时间序列数据。
    此工具可以直接读取CSV文件，将数据转换为时间序列格式（时间戳列表和数值列表）。
    
    支持的文件路径格式：
    - 仅文件名（如 "sample_timeseries.csv"）会自动在data文件夹中查找
    - 相对路径（如 "data/sample_timeseries.csv"）
    - 绝对路径
    
    CSV文件格式要求：
    - 必须有表头行
    - 包含时间戳列（默认列名为"timestamp"）
    - 包含数值列（默认列名为"value"）
    
    参数:
    - path: str - CSV文件路径（例如："sample_timeseries.csv" 或 "data/sample_timeseries.csv"）
    - timestamp_column: str - CSV中时间戳列的名称，默认 "timestamp"
    - value_column: str - CSV中数值列的名称，默认 "value"
    - limit: Optional[int] - 仅返回前N条记录（可选，用于快速测试）
    - dropna: bool - 是否跳过空值行，默认 True
    
    返回:
    - Dict包含:
      - timestamps: List[str] - 时间戳列表
      - values: List[float] - 数值列表
      - path: str - 实际读取的文件路径
      - total_rows: int - 文件总行数
      - returned_rows: int - 返回的数据行数
    
    使用示例:
    - load_timeseries_csv("sample_timeseries.csv")  # 读取data文件夹中的sample_timeseries.csv
    - load_timeseries_csv("data/sample_timeseries.csv")  # 明确指定路径
    """
    return load_timeseries_from_csv(
        path=path,
        timestamp_column=timestamp_column,
        value_column=value_column,
        limit=limit,
        dropna=dropna
    )


@mcp.tool()
async def detect_anomaly_from_csv(
    path: str,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
    method: str = "isolation_forest",
    contamination: Optional[float] = None,
    n_estimators: Optional[int] = None,
    threshold: Optional[float] = None,
    n_neighbors: Optional[int] = None,
    limit: Optional[int] = None,
    dropna: bool = True,
) -> Dict[str, Any]:
    """
    从CSV文件直接读取数据并执行异常检测。
    这是一个便捷的一站式工具：自动读取CSV文件 → 执行异常检测 → 返回详细结果和建议。
    
    此工具可以：
    1. 自动读取CSV文件（支持多种路径格式）
    2. 使用指定的异常检测算法进行分析
    3. 返回检测结果、异常点详情和专家建议
    
    支持的文件路径格式：
    - 仅文件名（如 "sample_timeseries.csv"）会自动在data文件夹中查找
    - 相对路径（如 "data/sample_timeseries.csv"）
    - 绝对路径
    
    CSV文件格式要求：
    - 必须有表头行
    - 包含时间戳列（默认列名为"timestamp"）
    - 包含数值列（默认列名为"value"）
    
    参数:
    - path: str - CSV文件路径（例如："sample_timeseries.csv" 或 "data/sample_timeseries.csv"）
    - timestamp_column: str - CSV中时间戳列的名称，默认 "timestamp"
    - value_column: str - CSV中数值列的名称，默认 "value"
    - method: str - 异常检测方法，可选值：
      * "isolation_forest" (默认) - 孤立森林算法，适合高维数据
      * "lof" - 局部异常因子，适合局部异常
      * "knn" - K近邻算法
      * "hbos" - 基于直方图的方法
      * "z_score" - Z分数方法，适合正态分布数据
      * "iqr" - 四分位距方法，适合有偏分布
      * "statistical" - 统计方法
    - contamination: Optional[float] - 预期的异常比例（0-1之间），某些算法使用
    - n_estimators: Optional[int] - Isolation Forest的树数量，默认100
    - threshold: Optional[float] - Z-score或Statistical方法的阈值，默认3.0
    - n_neighbors: Optional[int] - LOF或KNN的邻居数量，默认10
    - limit: Optional[int] - 仅处理前N条记录（可选，用于快速测试）
    - dropna: bool - 是否跳过空值行，默认 True
    
    返回:
    - Dict包含:
      - source: 数据源信息（文件路径、行列数等）
      - detection: 检测结果（方法、异常数量、评分、标签等）
      - anomaly_points: 详细的异常点列表（时间戳、数值、评分）
      - suggestions: 基于检测结果的专家建议
    
    使用示例:
    - detect_anomaly_from_csv("sample_timeseries.csv")  # 使用默认方法检测
    - detect_anomaly_from_csv("data/sample_timeseries.csv", method="z_score")  # 指定检测方法
    - detect_anomaly_from_csv("sample_timeseries.csv", method="isolation_forest", contamination=0.1)  # 指定异常比例
    """
    _init_engines()
    
    # 加载CSV数据
    csv_data = load_timeseries_from_csv(
        path=path,
        timestamp_column=timestamp_column,
        value_column=value_column,
        limit=limit,
        dropna=dropna
    )
    
    # 准备检测参数
    params = {}
    if contamination is not None:
        params["contamination"] = contamination
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if threshold is not None:
        params["threshold"] = threshold
    if n_neighbors is not None:
        params["n_neighbors"] = n_neighbors
    
    # 构造TimeSeriesData对象
    from src.models import TimeSeriesData
    data = TimeSeriesData(
        timestamps=csv_data["timestamps"],
        values=csv_data["values"]
    )
    
    # 执行异常检测
    result = await anomaly_detector.detect(data, method, params)
    
    # 找出异常点的详细信息
    anomaly_points = []
    for i, (is_anomaly, score, timestamp, value) in enumerate(zip(
        result["labels"],
        result["scores"],
        csv_data["timestamps"],
        csv_data["values"]
    )):
        if is_anomaly:
            anomaly_points.append({
                "index": i,
                "timestamp": timestamp,
                "value": value,
                "score": round(score, 4)
            })
    
    # 生成建议
    suggestions = []
    anomaly_count = sum(result["labels"])
    total_points = len(result["labels"])
    anomaly_ratio = anomaly_count / total_points if total_points > 0 else 0
    
    if anomaly_count == 0:
        suggestions.append("✅ 未检测到异常数据，数据看起来正常")
    else:
        suggestions.append(f"⚠️ 检测到 {anomaly_count} 个异常点（占比 {anomaly_ratio*100:.2f}%）")
        
        if anomaly_ratio > 0.1:
            suggestions.append("⚠️ 异常点比例较高（>10%），建议检查数据质量或调整检测参数")
        
        if anomaly_ratio < 0.01:
            suggestions.append("✅ 异常点比例很低，数据质量良好")
        
        # 找出最异常的5个点
        if anomaly_points:
            top_anomalies = sorted(anomaly_points, key=lambda x: x["score"], reverse=True)[:5]
            suggestions.append(f"🔍 最异常的5个点：")
            for idx, point in enumerate(top_anomalies, 1):
                suggestions.append(
                    f"   {idx}. 时间: {point['timestamp']}, 值: {point['value']}, "
                    f"异常评分: {point['score']:.4f}"
                )
    
    return {
        "status": "ok",
        "source": {
            "path": csv_data["path"],
            "timestamp_column": timestamp_column,
            "value_column": value_column,
            "total_rows": csv_data["total_rows"],
            "returned_rows": csv_data["returned_rows"]
        },
        "detection": {
            "method": method,
            "anomaly_count": anomaly_count,
            "total_points": total_points,
            "anomaly_ratio": round(anomaly_ratio, 4),
            "scores": [round(s, 4) for s in result["scores"]],
            "labels": result["labels"]
        },
        "anomaly_points": anomaly_points,
        "suggestions": suggestions
    }


@mcp.tool()
async def list_available_tools() -> Dict[str, Any]:
    """
    列出所有可用的工具、方法和指标
    
    返回:
    - Dict包含所有可用的检测方法、融合方法和统计指标
    """
    _init_engines()
    
    return {
        "status": "ok",
        "anomaly_detection_methods": anomaly_detector.list_methods(),
        "fusion_methods": fusion_engine.list_methods(),
        "statistical_metrics": stats_calculator.list_metrics()
    }


if __name__ == "__main__":
    # 启动MCP服务器，使用SSE传输协议
    mcp.run(transport="sse", port=2250)

