import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserAnalysis:
    """用户行为分析模块"""

    def __init__(self, df):
        self.df = df
        self.user_stats = None

    def analyze_user_activity(self):
        """
        用户活跃度分析

        :return: 活跃度分析结果
        """
        logger.info("开始用户活跃度分析")

        daily_active = self._calculate_daily_active()
        weekly_active = self._calculate_weekly_active()
        monthly_active = self._calculate_monthly_active()

        result = {
            'daily_active_users': daily_active,
            'weekly_active_users': weekly_active,
            'monthly_active_users': monthly_active,
            'summary': {
                'avg_daily_active': int(daily_active.mean()),
                'max_daily_active': int(daily_active.max()),
                'min_daily_active': int(daily_active.min())
            }
        }

        logger.info("用户活跃度分析完成")
        return result

    def _calculate_daily_active(self):
        """
        计算日活跃用户数

        :return: 日活跃用户序列
        """
        daily_active = self.df[self.df['behavior_type'] == 'pv'] \
            .groupby('date')['user_id'] \
            .nunique() \
            .sort_index()
        return daily_active

    def _calculate_weekly_active(self):
        """
        计算周活跃用户数

        :return: 周活跃用户序列
        """
        df_weekly = self.df.copy()
        df_weekly['week'] = df_weekly['datetime'].dt.to_period('W')
        weekly_active = df_weekly[df_weekly['behavior_type'] == 'pv'] \
            .groupby('week')['user_id'] \
            .nunique() \
            .sort_index()
        return weekly_active

    def _calculate_monthly_active(self):
        """
        计算月活跃用户数

        :return: 月活跃用户序列
        """
        df_monthly = self.df.copy()
        df_monthly['month'] = df_monthly['datetime'].dt.to_period('M')
        monthly_active = df_monthly[df_monthly['behavior_type'] == 'pv'] \
            .groupby('month')['user_id'] \
            .nunique() \
            .sort_index()
        return monthly_active

    def analyze_user_behavior_path(self):
        """
        用户行为路径分析

        :return: 行为路径分析结果
        """
        logger.info("开始用户行为路径分析")

        user_behavior_count = self.df.groupby(['user_id', 'behavior_type'])['item_id'] \
            .count() \
            .unstack(fill_value=0)

        behavior_ratios = self._calculate_behavior_ratios(user_behavior_count)
        user_conversion = self._calculate_user_conversion(user_behavior_count)

        result = {
            'behavior_statistics': {
                'total_users': len(user_behavior_count),
                'avg_pv_per_user': float(user_behavior_count.get('pv', 0).mean()),
                'avg_buy_per_user': float(user_behavior_count.get('buy', 0).mean()),
                'avg_cart_per_user': float(user_behavior_count.get('cart', 0).mean()),
                'avg_fav_per_user': float(user_behavior_count.get('fav', 0).mean())
            },
            'behavior_ratios': behavior_ratios,
            'conversion_metrics': user_conversion
        }

        logger.info("用户行为路径分析完成")
        return result

    def _calculate_behavior_ratios(self, user_behavior_count):
        """
        计算用户行为比例

        :param user_behavior_count: 用户行为计数
        :return: 行为比例字典
        """
        ratios = {}
        behaviors = ['pv', 'buy', 'cart', 'fav']

        for behavior in behaviors:
            count = user_behavior_count.get(behavior, 0)
            ratios[f'{behavior}_users'] = int((count > 0).sum())
            ratios[f'{behavior}_ratio'] = float((count > 0).mean() * 100)

        return ratios

    def _calculate_user_conversion(self, user_behavior_count):
        """
        计算用户转化率

        :param user_behavior_count: 用户行为计数
        :return: 转化率指标字典
        """
        pv_users = (user_behavior_count.get('pv', 0) > 0).sum()
        buy_users = (user_behavior_count.get('buy', 0) > 0).sum()
        cart_users = (user_behavior_count.get('cart', 0) > 0).sum()
        fav_users = (user_behavior_count.get('fav', 0) > 0).sum()

        metrics = {
            'pv_to_buy_conversion': float(buy_users / pv_users * 100) if pv_users > 0 else 0,
            'pv_to_cart_conversion': float(cart_users / pv_users * 100) if pv_users > 0 else 0,
            'pv_to_fav_conversion': float(fav_users / pv_users * 100) if pv_users > 0 else 0,
            'cart_to_buy_conversion': float(buy_users / cart_users * 100) if cart_users > 0 else 0,
            'fav_to_buy_conversion': float(buy_users / fav_users * 100) if fav_users > 0 else 0
        }

        return metrics

    def analyze_user_retention(self):
        """
        用户留存率分析

        :return: 留存率分析结果
        """
        logger.info("开始用户留存率分析")

        retention_data = self._calculate_retention()

        result = {
            'retention_table': retention_data.to_dict(),
            'summary': {
                'day_1_retention': float(retention_data.get(1, 0)),
                'day_3_retention': float(retention_data.get(3, 0)),
                'day_7_retention': float(retention_data.get(7, 0)),
                'day_14_retention': float(retention_data.get(14, 0)),
                'day_30_retention': float(retention_data.get(30, 0))
            }
        }

        logger.info("用户留存率分析完成")
        return result

    def _calculate_retention(self):
        """
        计算用户留存率

        :return: 留存率序列
        """
        first_visit = self.df[self.df['behavior_type'] == 'pv'] \
            .groupby('user_id')['date'] \
            .min() \
            .reset_index() \
            .rename(columns={'date': 'first_date'})

        user_visits = self.df[self.df['behavior_type'] == 'pv'] \
            .groupby(['user_id', 'date'])['item_id'] \
            .count() \
            .reset_index()

        merged = user_visits.merge(first_visit, on='user_id')
        merged['days_since_first'] = (merged['date'] - merged['first_date']).apply(lambda x: x.days)

        retention = {}
        for days in [1, 3, 7, 14, 30]:
            active_users = merged[merged['days_since_first'] == days]['user_id'].nunique()
            total_users = len(first_visit)
            retention[days] = float(active_users / total_users * 100)

        return pd.Series(retention)

    def analyze_user_segmentation(self):
        """
        用户分群分析

        :return: 用户分群结果
        """
        logger.info("开始用户分群分析")

        user_features = self._extract_user_features()
        segments = self._segment_users(user_features)

        result = {
            'user_segments': segments,
            'segment_distribution': segments['segment'].value_counts().to_dict(),
            'segment_profiles': self._describe_segments(user_features, segments)
        }

        logger.info("用户分群分析完成")
        return result

    def _extract_user_features(self):
        """
        提取用户特征

        :return: 用户特征数据框
        """
        user_features = self.df.groupby('user_id').agg(
            total_pv=('behavior_type', lambda x: (x == 'pv').sum()),
            total_buy=('behavior_type', lambda x: (x == 'buy').sum()),
            total_cart=('behavior_type', lambda x: (x == 'cart').sum()),
            total_fav=('behavior_type', lambda x: (x == 'fav').sum()),
            avg_session_duration=('datetime', lambda x: (x.max() - x.min()).total_seconds() / 3600 if len(x) > 1 else 0),
            days_active=('date', 'nunique'),
            first_visit=('date', 'min'),
            last_visit=('date', 'max')
        )

        user_features['buy_ratio'] = user_features['total_buy'] / user_features['total_pv']
        user_features['cart_ratio'] = user_features['total_cart'] / user_features['total_pv']
        user_features['engagement_score'] = (
            user_features['total_pv'] * 1 +
            user_features['total_cart'] * 2 +
            user_features['total_buy'] * 5
        )

        return user_features

    def _segment_users(self, user_features):
        """
        用户分群

        :param user_features: 用户特征数据框
        :return: 分群结果
        """
        segments = user_features.copy()
        segments['segment'] = '普通用户'

        segments.loc[segments['total_buy'] >= 10, 'segment'] = '高价值用户'
        segments.loc[(segments['total_buy'] > 0) & (segments['total_buy'] < 10), 'segment'] = '购买用户'
        segments.loc[(segments['total_pv'] >= 100) & (segments['total_buy'] == 0), 'segment'] = '浏览用户'
        segments.loc[segments['total_pv'] < 10, 'segment'] = '新用户'
        segments.loc[segments['days_active'] >= 7, 'segment'] = '活跃用户'
        segments.loc[segments['buy_ratio'] >= 0.1, 'segment'] = '高转化用户'

        return segments[['segment']]

    def _describe_segments(self, user_features, segments):
        """
        描述各用户群体特征

        :param user_features: 用户特征数据框
        :param segments: 分群结果
        :return: 群体描述字典
        """
        merged = user_features.merge(segments, on='user_id')
        descriptions = {}

        for segment in merged['segment'].unique():
            segment_data = merged[merged['segment'] == segment]
            descriptions[segment] = {
                'count': int(len(segment_data)),
                'avg_pv': float(segment_data['total_pv'].mean()),
                'avg_buy': float(segment_data['total_buy'].mean()),
                'avg_days_active': float(segment_data['days_active'].mean()),
                'avg_engagement_score': float(segment_data['engagement_score'].mean())
            }

        return descriptions

    def analyze_user_time_distribution(self):
        """
        用户活跃时间分布分析

        :return: 时间分布分析结果
        """
        logger.info("开始用户活跃时间分布分析")

        hour_distribution = self._calculate_hour_distribution()
        weekday_distribution = self._calculate_weekday_distribution()

        result = {
            'hour_distribution': hour_distribution.to_dict(),
            'weekday_distribution': weekday_distribution.to_dict(),
            'peak_hours': hour_distribution.nlargest(3).index.tolist(),
            'peak_days': weekday_distribution.nlargest(2).index.tolist()
        }

        logger.info("用户活跃时间分布分析完成")
        return result

    def _calculate_hour_distribution(self):
        """
        计算小时活跃度分布

        :return: 小时活跃度序列
        """
        return self.df[self.df['behavior_type'] == 'pv'] \
            .groupby('hour')['user_id'] \
            .nunique() \
            .sort_index()

    def _calculate_weekday_distribution(self):
        """
        计算周活跃度分布

        :return: 周活跃度序列
        """
        weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
        distribution = self.df[self.df['behavior_type'] == 'pv'] \
            .groupby('dayofweek')['user_id'] \
            .nunique() \
            .sort_index()
        distribution.index = distribution.index.map(weekday_map)
        return distribution

    def get_all_user_analysis(self):
        """
        获取所有用户分析结果

        :return: 完整的用户分析结果字典
        """
        return {
            'user_activity': self.analyze_user_activity(),
            'user_behavior_path': self.analyze_user_behavior_path(),
            'user_retention': self.analyze_user_retention(),
            'user_segmentation': self.analyze_user_segmentation(),
            'user_time_distribution': self.analyze_user_time_distribution()
        }