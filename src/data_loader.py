import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载和预处理模块"""

    def __init__(self, file_path=None):
        self.file_path = file_path or self._find_data_file()
        self.df = None
        self.column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

    def _find_data_file(self):
        """自动查找数据文件"""
        possible_paths = [
            './data/UserBehavior.csv',
            '../data/UserBehavior.csv',
            './UserBehavior.csv/UserBehavior.csv',
            '../UserBehavior.csv/UserBehavior.csv',
            'UserBehavior.csv/UserBehavior.csv'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        logger.warning("未找到数据文件，请确保数据文件存在")
        return None

    def load_data(self, sample_size=None, chunk_size=1000000):
        """
        加载数据，支持采样和分块处理

        :param sample_size: 采样数量，如果为None则加载全部数据
        :param chunk_size: 分块大小，默认为100万条
        :return: 加载后的数据框
        """
        if not self.file_path:
            logger.error("数据文件路径为空")
            return None

        logger.info(f"开始加载数据: {self.file_path}")
        logger.info(f"采样大小: {'全部' if sample_size is None else sample_size}")

        try:
            if sample_size is not None:
                self.df = self._load_with_sampling(sample_size)
            else:
                self.df = self._load_full_data(chunk_size)

            logger.info(f"数据加载完成，共 {len(self.df)} 条记录")
            return self.df

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None

    def _load_with_sampling(self, sample_size):
        """
        使用采样方式加载数据

        :param sample_size: 采样数量
        :return: 采样后的数据框
        """
        total_lines = self._count_lines()
        logger.info(f"文件总行数: {total_lines}")

        if total_lines <= sample_size:
            logger.info("采样数量大于文件行数，加载全部数据")
            return pd.read_csv(
                self.file_path,
                names=self.column_names,
                header=None
            )

        skip_prob = 1 - sample_size / total_lines
        logger.info(f"采样概率: {1 - skip_prob:.4f}")

        return pd.read_csv(
            self.file_path,
            names=self.column_names,
            header=None,
            skiprows=lambda i: i > 0 and np.random.random() < skip_prob
        )

    def _load_full_data(self, chunk_size):
        """
        分块加载全部数据

        :param chunk_size: 分块大小
        :return: 合并后的数据框
        """
        chunks = []
        for i, chunk in enumerate(pd.read_csv(
            self.file_path,
            names=self.column_names,
            header=None,
            chunksize=chunk_size
        )):
            chunks.append(chunk)
            logger.info(f"已加载第 {i + 1} 块，累计 {len(chunks) * chunk_size} 条记录")

        return pd.concat(chunks, ignore_index=True)

    def _count_lines(self):
        """
        计算文件行数

        :return: 文件行数
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"计算文件行数失败: {e}")
            return 120000000

    def preprocess(self):
        """
        数据预处理

        :return: 预处理后的数据框
        """
        if self.df is None:
            logger.error("数据未加载，请先调用 load_data()")
            return None

        logger.info("开始数据预处理")

        self.df = self._convert_timestamp(self.df)
        self.df = self._clean_data(self.df)
        self.df = self._add_time_features(self.df)

        logger.info("数据预处理完成")
        return self.df

    def _convert_timestamp(self, df):
        """
        将时间戳转换为datetime格式

        :param df: 原始数据框
        :return: 转换后的数据框
        """
        logger.info("转换时间戳格式")
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    def _clean_data(self, df):
        """
        数据清洗

        :param df: 原始数据框
        :return: 清洗后的数据框
        """
        logger.info("数据清洗")

        initial_count = len(df)

        df = df.drop_duplicates()
        logger.info(f"去重后: {len(df)} 条")

        df = df.dropna()
        logger.info(f"去除空值后: {len(df)} 条")

        valid_behaviors = ['pv', 'buy', 'cart', 'fav']
        df = df[df['behavior_type'].isin(valid_behaviors)]
        logger.info(f"过滤有效行为类型后: {len(df)} 条")

        df = df[(df['user_id'] > 0) & (df['item_id'] > 0) & (df['category_id'] > 0)]
        logger.info(f"过滤有效ID后: {len(df)} 条")

        # 过滤异常时间戳，天池数据时间范围：2017-11-25 ~ 2017-12-03
        valid_start = pd.Timestamp('2017-11-25').timestamp()
        valid_end = pd.Timestamp('2017-12-04').timestamp()
        df = df[(df['timestamp'] >= valid_start) & (df['timestamp'] <= valid_end)]
        logger.info(f"过滤异常时间戳后: {len(df)} 条")

        removed_count = initial_count - len(df)
        logger.info(f"共移除 {removed_count} 条无效数据")

        return df

    def _add_time_features(self, df):
        """
        添加时间相关特征

        :param df: 原始数据框
        :return: 添加特征后的数据框
        """
        logger.info("添加时间特征")

        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        return df

    def get_data_summary(self):
        """
        获取数据摘要信息

        :return: 摘要信息字典
        """
        if self.df is None:
            return None

        summary = {
            'total_records': len(self.df),
            'total_users': self.df['user_id'].nunique(),
            'total_items': self.df['item_id'].nunique(),
            'total_categories': self.df['category_id'].nunique(),
            'behavior_distribution': self.df['behavior_type'].value_counts().to_dict(),
            'date_range': {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max())
            }
        }

        return summary

    def save_processed_data(self, output_path='./output/processed_data.csv'):
        """
        保存处理后的数据

        :param output_path: 输出路径
        :return: 是否保存成功
        """
        if self.df is None:
            logger.error("数据未加载，请先调用 load_data()")
            return False

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_csv(output_path, index=False)
            logger.info(f"处理后的数据已保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return False