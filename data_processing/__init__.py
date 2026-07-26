"""数据处理模块"""

from .data_cleaner import DataCleaner
from .graph_builder import GraphBuilder

__all__ = ["DataCleaner", "GraphBuilder"]


def get_dataset_class():
    """延迟获取 RecommendationDataset 类（需要 torch）"""
    from .dataset import RecommendationDataset
    return RecommendationDataset


def get_split_data_func():
    """延迟获取 split_data 函数"""
    from .dataset import split_data
    return split_data


def get_build_dataloaders_func():
    """延迟获取 build_dataloaders 函数"""
    from .dataset import build_dataloaders
    return build_dataloaders