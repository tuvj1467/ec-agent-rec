"""电商用户行为分析项目 - 模块初始化"""

__version__ = "1.0.0"
__author__ = "E-commerce Analytics Team"

from .data_loader import DataLoader
from .user_analysis import UserAnalysis
from .product_analysis import ProductAnalysis
from .conversion_analysis import ConversionAnalysis
from .visualization import Visualization

__all__ = [
    "DataLoader",
    "UserAnalysis",
    "ProductAnalysis",
    "ConversionAnalysis",
    "Visualization"
]