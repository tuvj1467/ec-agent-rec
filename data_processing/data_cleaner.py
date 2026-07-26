import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗器 - 负责天池用户行为数据的清洗和预处理"""

    def __init__(self, input_path=None, output_dir='./output/data'):
        self.input_path = input_path or self._find_data_file()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.df = None

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

    def load_and_clean(self, sample_size=None, chunk_size=1000000):
        """
        加载并清洗数据

        :param sample_size: 采样数量，None表示全部
        :param chunk_size: 分块大小
        :return: 清洗后的数据框
        """
        logger.info("开始数据清洗流程")

        self.df = self._load_data(sample_size, chunk_size)
        if self.df is None:
            return None

        self.df = self._remove_duplicates(self.df)
        self.df = self._filter_valid_behaviors(self.df)
        self.df = self._filter_valid_ids(self.df)
        self.df = self._filter_valid_timestamps(self.df)
        self.df = self._convert_timestamp(self.df)

        logger.info(f"数据清洗完成，最终记录数: {len(self.df):,}")
        return self.df

    def _load_data(self, sample_size, chunk_size):
        """加载数据"""
        if not self.input_path:
            logger.error("数据文件路径为空")
            return None

        logger.info(f"加载数据: {self.input_path}")
        column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

        try:
            if sample_size is not None:
                df = self._load_with_sampling(sample_size, column_names)
            else:
                df = self._load_full_data(chunk_size, column_names)

            logger.info(f"加载完成，共 {len(df):,} 条记录")
            return df

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None

    def _load_with_sampling(self, sample_size, column_names):
        """采样加载"""
        total_lines = self._count_lines()
        if total_lines <= sample_size:
            return pd.read_csv(self.input_path, names=column_names, header=None)

        skip_prob = 1 - sample_size / total_lines
        return pd.read_csv(
            self.input_path,
            names=column_names,
            header=None,
            skiprows=lambda i: i > 0 and np.random.random() < skip_prob
        )

    def _load_full_data(self, chunk_size, column_names):
        """全量分块加载"""
        chunks = []
        for i, chunk in enumerate(pd.read_csv(
            self.input_path,
            names=column_names,
            header=None,
            chunksize=chunk_size
        )):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.info(f"已加载 {len(chunks) * chunk_size:,} 条")
        return pd.concat(chunks, ignore_index=True)

    def _count_lines(self):
        """计算文件行数"""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"计算行数失败: {e}")
            return 100000000

    def _remove_duplicates(self, df):
        """去重"""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        logger.info(f"去重: {before:,} → {after:,} (移除 {before - after:,} 条)")
        return df

    def _filter_valid_behaviors(self, df):
        """过滤有效行为类型"""
        valid_behaviors = ['pv', 'buy', 'cart', 'fav']
        before = len(df)
        df = df[df['behavior_type'].isin(valid_behaviors)]
        after = len(df)
        logger.info(f"过滤行为类型: {before:,} → {after:,} (移除 {before - after:,} 条)")
        return df

    def _filter_valid_ids(self, df):
        """过滤有效ID"""
        before = len(df)
        df = df[(df['user_id'] > 0) & (df['item_id'] > 0) & (df['category_id'] > 0)]
        after = len(df)
        logger.info(f"过滤有效ID: {before:,} → {after:,} (移除 {before - after:,} 条)")
        return df

    def _filter_valid_timestamps(self, df):
        """过滤有效时间戳（天池数据范围: 2017-11-25 ~ 2017-12-03）"""
        valid_start = int(pd.Timestamp('2017-11-25').timestamp())
        valid_end = int(pd.Timestamp('2017-12-04').timestamp())

        before = len(df)
        df = df[(df['timestamp'] >= valid_start) & (df['timestamp'] <= valid_end)]
        after = len(df)
        logger.info(f"过滤时间戳: {before:,} → {after:,} (移除 {before - after:,} 条)")
        return df

    def _convert_timestamp(self, df):
        """转换时间戳格式"""
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        return df

    def get_buy_interactions(self):
        """
        获取购买交互数据（用于推荐模型训练）

        :return: 购买交互数据框 (user_id, item_id)
        """
        if self.df is None:
            logger.error("数据未加载")
            return None

        buy_df = self.df[self.df['behavior_type'] == 'buy'][['user_id', 'item_id']].drop_duplicates()
        logger.info(f"购买交互数: {len(buy_df):,}")
        return buy_df

    def get_all_interactions(self):
        """
        获取所有交互数据（行为类型作为权重）

        :return: 交互数据框 (user_id, item_id, weight)
        """
        if self.df is None:
            logger.error("数据未加载")
            return None

        behavior_weight = {'pv': 1.0, 'fav': 2.0, 'cart': 3.0, 'buy': 4.0}
        interactions = self.df.copy()
        interactions['weight'] = interactions['behavior_type'].map(behavior_weight)
        interactions = interactions.groupby(['user_id', 'item_id'])['weight'].max().reset_index()

        logger.info(f"用户-商品交互对: {len(interactions):,}")
        return interactions

    def get_data_summary(self):
        """获取数据摘要"""
        if self.df is None:
            return None

        return {
            'total_records': int(len(self.df)),
            'total_users': int(self.df['user_id'].nunique()),
            'total_items': int(self.df['item_id'].nunique()),
            'total_categories': int(self.df['category_id'].nunique()),
            'behavior_distribution': {
                k: int(v) for k, v in self.df['behavior_type'].value_counts().items()
            },
            'date_range': {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max())
            },
            'buy_interactions': int(len(self.df[self.df['behavior_type'] == 'buy']))
        }

    def save_cleaned_data(self, filename='clean_interactions.csv'):
        """保存清洗后的数据"""
        if self.df is None:
            return False

        output_path = os.path.join(self.output_dir, filename)
        self.df.to_csv(output_path, index=False)
        logger.info(f"清洗后的数据已保存: {output_path}")
        return True