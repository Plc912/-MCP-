from pathlib import Path
from typing import Dict, List, Optional, Any
import csv
import pandas as pd


def _resolve_csv_path(path: str) -> Path:
    """
    解析CSV文件路径，支持多种路径格式
    """
    candidate = Path(path).expanduser()
    search_order = []
    
    if candidate.is_absolute():
        search_order.append(candidate)
    else:
        # 相对路径：尝试多个位置
        cwd = Path.cwd()
        base_dir = Path(__file__).parent.parent  # 项目根目录
        
        search_order.append((cwd / candidate).resolve())
        search_order.append((base_dir / candidate).resolve())
        
        # 如果只是文件名，尝试data文件夹
        if candidate.parent == Path("."):
            search_order.append((base_dir / "data" / candidate.name).resolve())
            search_order.append((cwd / "data" / candidate.name).resolve())
    
    # 查找存在的文件
    for option in search_order:
        if option.exists() and option.is_file():
            return option
    
    # 如果都不存在，返回最后一个候选路径（用于错误提示）
    return candidate if candidate.is_absolute() else search_order[-1] if search_order else candidate


def load_timeseries_from_csv(
    path: str,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
    limit: Optional[int] = None,
    dropna: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    从CSV文件加载时间序列数据
    
    参数:
    - path: str - CSV文件路径（支持相对路径、绝对路径，或仅文件名）
    - timestamp_column: str - 时间戳列名，默认 "timestamp"
    - value_column: str - 数值列名，默认 "value"
    - limit: Optional[int] - 仅返回前N条记录（用于快速测试）
    - dropna: bool - 是否跳过空值行，默认 True
    - encoding: str - 文件编码，默认 "utf-8"
    
    返回:
    - Dict包含: timestamps (List[str]), values (List[float]), path (str), total_rows (int)
    """
    target = _resolve_csv_path(path)
    
    if not target.exists():
        raise FileNotFoundError(
            f"CSV文件未找到: {target}\n"
            f"尝试的路径包括:\n"
            f"  - {target}\n"
            f"  - {Path.cwd() / path}\n"
            f"  - {Path(__file__).parent.parent / 'data' / Path(path).name}"
        )
    
    timestamps = []
    values = []
    total_rows = 0
    
    try:
        with target.open("r", encoding=encoding, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            
            if reader.fieldnames is None:
                raise ValueError("CSV文件没有表头行")
            
            header = reader.fieldnames
            
            # 检查必需的列
            if timestamp_column not in header:
                raise ValueError(
                    f"时间戳列 '{timestamp_column}' 未在CSV文件中找到。"
                    f"可用列: {list(header)}"
                )
            
            if value_column not in header:
                raise ValueError(
                    f"数值列 '{value_column}' 未在CSV文件中找到。"
                    f"可用列: {list(header)}"
                )
            
            # 读取数据
            for row in reader:
                total_rows += 1
                
                timestamp = row.get(timestamp_column)
                value_str = row.get(value_column)
                
                # 处理空值
                if dropna and (not timestamp or not value_str or timestamp.strip() == "" or value_str.strip() == ""):
                    continue
                
                try:
                    value = float(value_str)
                    timestamps.append(timestamp.strip() if timestamp else "")
                    values.append(value)
                    
                    # 限制返回数量
                    if limit is not None and len(values) >= limit:
                        break
                        
                except (ValueError, TypeError) as e:
                    if not dropna:
                        raise ValueError(
                            f"第 {total_rows} 行的数值列 '{value_column}' 无法转换为浮点数: {value_str!r}"
                        ) from e
    
    except Exception as e:
        raise ValueError(f"读取CSV文件失败: {str(e)}") from e
    
    if len(values) == 0:
        raise ValueError(f"CSV文件中没有有效数据（共读取 {total_rows} 行）")
    
    return {
        "status": "ok",
        "path": str(target),
        "timestamp_column": timestamp_column,
        "value_column": value_column,
        "total_rows": total_rows,
        "returned_rows": len(values),
        "timestamps": timestamps,
        "values": values
    }

