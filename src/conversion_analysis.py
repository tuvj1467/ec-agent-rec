import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversionAnalysis:
    """转化率和漏斗分析模块"""

    def __init__(self, df):
        self.df = df

    def analyze_conversion_funnel(self):
        """
        分析转化漏斗

        :return: 漏斗分析结果
        """
        logger.info("开始转化漏斗分析")

        funnel_data = self._calculate_funnel()
        funnel_rates = self._calculate_funnel_rates(funnel_data)

        result = {
            'funnel_data': funnel_data,
            'funnel_rates': funnel_rates,
            'overall_conversion': funnel_rates.get('pv_to_buy', 0),
            'step_details': self._get_step_details(funnel_data)
        }

        logger.info("转化漏斗分析完成")
        return result

    def _calculate_funnel(self):
        """
        计算漏斗各环节数据

        :return: 漏斗数据字典
        """
        pv_count = len(self.df[self.df['behavior_type'] == 'pv'])
        cart_count = len(self.df[self.df['behavior_type'] == 'cart'])
        fav_count = len(self.df[self.df['behavior_type'] == 'fav'])
        buy_count = len(self.df[self.df['behavior_type'] == 'buy'])

        pv_users = self.df[self.df['behavior_type'] == 'pv']['user_id'].nunique()
        cart_users = self.df[self.df['behavior_type'] == 'cart']['user_id'].nunique()
        fav_users = self.df[self.df['behavior_type'] == 'fav']['user_id'].nunique()
        buy_users = self.df[self.df['behavior_type'] == 'buy']['user_id'].nunique()

        return {
            'pv': {'count': int(pv_count), 'users': int(pv_users)},
            'cart': {'count': int(cart_count), 'users': int(cart_users)},
            'fav': {'count': int(fav_count), 'users': int(fav_users)},
            'buy': {'count': int(buy_count), 'users': int(buy_users)}
        }

    def _calculate_funnel_rates(self, funnel_data):
        """
        计算漏斗各环节转化率

        :param funnel_data: 漏斗数据
        :return: 漏斗转化率字典
        """
        pv_count = funnel_data['pv']['count']
        cart_count = funnel_data['cart']['count']
        fav_count = funnel_data['fav']['count']
        buy_count = funnel_data['buy']['count']

        pv_users = funnel_data['pv']['users']
        cart_users = funnel_data['cart']['users']
        fav_users = funnel_data['fav']['users']
        buy_users = funnel_data['buy']['users']

        rates = {
            'pv_to_cart': float(cart_users / pv_users * 100) if pv_users > 0 else 0,
            'pv_to_fav': float(fav_users / pv_users * 100) if pv_users > 0 else 0,
            'pv_to_buy': float(buy_users / pv_users * 100) if pv_users > 0 else 0,
            'cart_to_buy': float(buy_users / cart_users * 100) if cart_users > 0 else 0,
            'fav_to_buy': float(buy_users / fav_users * 100) if fav_users > 0 else 0,
            'cart_rate': float(cart_count / pv_count * 100) if pv_count > 0 else 0,
            'fav_rate': float(fav_count / pv_count * 100) if pv_count > 0 else 0,
            'buy_rate': float(buy_count / pv_count * 100) if pv_count > 0 else 0
        }

        return rates

    def _get_step_details(self, funnel_data):
        """
        获取各环节详细信息

        :param funnel_data: 漏斗数据
        :return: 环节详情列表
        """
        steps = [
            {'name': '浏览', 'key': 'pv', 'count': funnel_data['pv']['count'], 'users': funnel_data['pv']['users']},
            {'name': '加购', 'key': 'cart', 'count': funnel_data['cart']['count'], 'users': funnel_data['cart']['users']},
            {'name': '收藏', 'key': 'fav', 'count': funnel_data['fav']['count'], 'users': funnel_data['fav']['users']},
            {'name': '购买', 'key': 'buy', 'count': funnel_data['buy']['count'], 'users': funnel_data['buy']['users']}
        ]
        return steps

    def analyze_user_conversion_paths(self):
        """
        分析用户转化路径

        :return: 用户转化路径分析结果
        """
        logger.info("开始用户转化路径分析")

        user_paths = self._extract_user_paths()
        path_analysis = self._analyze_paths(user_paths)

        result = {
            'total_users': int(len(user_paths)),
            'path_distribution': path_analysis['path_distribution'],
            'conversion_path_efficiency': path_analysis['efficiency'],
            'common_paths': path_analysis['common_paths']
        }

        logger.info("用户转化路径分析完成")
        return result

    def _extract_user_paths(self):
        """
        提取用户行为路径

        :return: 用户行为路径字典
        """
        user_behavior = self.df.sort_values(['user_id', 'timestamp']) \
            .groupby('user_id')['behavior_type'] \
            .apply(list) \
            .reset_index()

        user_paths = {}
        for _, row in user_behavior.iterrows():
            user_paths[row['user_id']] = self._simplify_path(row['behavior_type'])

        return user_paths

    def _simplify_path(self, behaviors):
        """
        简化用户行为路径（去除连续重复行为）

        :param behaviors: 行为列表
        :return: 简化后的路径字符串
        """
        simplified = []
        prev = None
        for behavior in behaviors:
            if behavior != prev:
                simplified.append(behavior)
                prev = behavior
        return ' -> '.join(simplified)

    def _analyze_paths(self, user_paths):
        """
        分析用户路径

        :param user_paths: 用户路径字典
        :return: 路径分析结果
        """
        path_counts = pd.Series(list(user_paths.values())).value_counts()
        total_users = len(user_paths)

        path_distribution = {}
        for path, count in path_counts.head(20).items():
            path_distribution[path] = {
                'count': int(count),
                'percentage': float(count / total_users * 100),
                'converted': 'buy' in path
            }

        converted_paths = {k: v for k, v in path_distribution.items() if v['converted']}
        non_converted_paths = {k: v for k, v in path_distribution.items() if not v['converted']}

        efficiency = {
            'converted_path_ratio': float(len(converted_paths) / len(path_distribution) * 100) if path_distribution else 0,
            'avg_path_length_for_converted': float(np.mean([len(p.split(' -> ')) for p in converted_paths])),
            'avg_path_length_for_non_converted': float(np.mean([len(p.split(' -> ')) for p in non_converted_paths])) if non_converted_paths else 0
        }

        common_paths = []
        for path, stats in list(path_distribution.items())[:10]:
            common_paths.append({
                'path': path,
                'user_count': stats['count'],
                'percentage': stats['percentage'],
                'converted': stats['converted']
            })

        return {
            'path_distribution': path_distribution,
            'efficiency': efficiency,
            'common_paths': common_paths
        }

    def analyze_conversion_by_time(self):
        """
        按时间分析转化率

        :return: 时间维度转化率分析结果
        """
        logger.info("开始按时间分析转化率")

        hourly_conversion = self._calculate_hourly_conversion()
        daily_conversion = self._calculate_daily_conversion()
        weekday_conversion = self._calculate_weekday_conversion()

        result = {
            'hourly_conversion': hourly_conversion,
            'daily_conversion': daily_conversion,
            'weekday_conversion': weekday_conversion,
            'best_hour': hourly_conversion.idxmax(),
            'best_day': weekday_conversion.idxmax()
        }

        logger.info("按时间分析转化率完成")
        return result

    def _calculate_hourly_conversion(self):
        """
        计算小时转化率

        :return: 小时转化率序列
        """
        hourly_pv = self.df[self.df['behavior_type'] == 'pv'].groupby('hour')['user_id'].nunique()
        hourly_buy = self.df[self.df['behavior_type'] == 'buy'].groupby('hour')['user_id'].nunique()

        hourly_conversion = (hourly_buy / hourly_pv * 100).fillna(0)
        return hourly_conversion

    def _calculate_daily_conversion(self):
        """
        计算每日转化率

        :return: 每日转化率序列
        """
        daily_pv = self.df[self.df['behavior_type'] == 'pv'].groupby('date')['user_id'].nunique()
        daily_buy = self.df[self.df['behavior_type'] == 'buy'].groupby('date')['user_id'].nunique()

        daily_conversion = (daily_buy / daily_pv * 100).fillna(0)
        return daily_conversion

    def _calculate_weekday_conversion(self):
        """
        计算周转化率

        :return: 周转化率序列
        """
        weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}

        weekday_pv = self.df[self.df['behavior_type'] == 'pv'].groupby('dayofweek')['user_id'].nunique()
        weekday_buy = self.df[self.df['behavior_type'] == 'buy'].groupby('dayofweek')['user_id'].nunique()

        weekday_conversion = (weekday_buy / weekday_pv * 100).fillna(0)
        weekday_conversion.index = weekday_conversion.index.map(weekday_map)
        return weekday_conversion

    def analyze_conversion_by_category(self):
        """
        按商品分类分析转化率

        :return: 分类转化率分析结果
        """
        logger.info("开始按商品分类分析转化率")

        category_conversion = self._calculate_category_conversion()
        top_categories = category_conversion.sort_values('conversion_rate', ascending=False).head(10)
        bottom_categories = category_conversion.sort_values('conversion_rate').head(10)

        result = {
            'category_conversion': category_conversion.to_dict('records'),
            'top_conversion_categories': top_categories.to_dict('records'),
            'bottom_conversion_categories': bottom_categories.to_dict('records'),
            'avg_conversion_rate': float(category_conversion['conversion_rate'].mean()),
            'median_conversion_rate': float(category_conversion['conversion_rate'].median())
        }

        logger.info("按商品分类分析转化率完成")
        return result

    def _calculate_category_conversion(self):
        """
        计算分类转化率

        :return: 分类转化率数据框
        """
        category_pv = self.df[self.df['behavior_type'] == 'pv'] \
            .groupby('category_id')['user_id'] \
            .nunique() \
            .reset_index() \
            .rename(columns={'user_id': 'pv_users'})

        category_buy = self.df[self.df['behavior_type'] == 'buy'] \
            .groupby('category_id')['user_id'] \
            .nunique() \
            .reset_index() \
            .rename(columns={'user_id': 'buy_users'})

        category_conversion = category_pv.merge(category_buy, on='category_id', how='left').fillna(0)
        category_conversion['conversion_rate'] = category_conversion['buy_users'] / category_conversion['pv_users'] * 100

        category_item_count = self.df.groupby('category_id')['item_id'].nunique().reset_index().rename(columns={'item_id': 'item_count'})
        category_conversion = category_conversion.merge(category_item_count, on='category_id', how='left')

        return category_conversion

    def analyze_conversion_by_user_segment(self, user_segments):
        """
        按用户群体分析转化率

        :param user_segments: 用户分群数据
        :return: 用户群体转化率分析结果
        """
        logger.info("开始按用户群体分析转化率")

        if user_segments is None:
            logger.warning("用户分群数据为空")
            return {'error': '用户分群数据为空'}

        segment_conversion = self._calculate_segment_conversion(user_segments)

        result = {
            'segment_conversion': segment_conversion.to_dict('records'),
            'best_segment': segment_conversion.loc[segment_conversion['conversion_rate'].idxmax()]['segment'],
            'worst_segment': segment_conversion.loc[segment_conversion['conversion_rate'].idxmin()]['segment']
        }

        logger.info("按用户群体分析转化率完成")
        return result

    def _calculate_segment_conversion(self, user_segments):
        """
        计算用户群体转化率

        :param user_segments: 用户分群数据
        :return: 用户群体转化率数据框
        """
        merged = self.df.merge(user_segments, on='user_id')

        segment_pv = merged[merged['behavior_type'] == 'pv'] \
            .groupby('segment')['user_id'] \
            .nunique() \
            .reset_index() \
            .rename(columns={'user_id': 'pv_users'})

        segment_buy = merged[merged['behavior_type'] == 'buy'] \
            .groupby('segment')['user_id'] \
            .nunique() \
            .reset_index() \
            .rename(columns={'user_id': 'buy_users'})

        segment_conversion = segment_pv.merge(segment_buy, on='segment', how='left').fillna(0)
        segment_conversion['conversion_rate'] = segment_conversion['buy_users'] / segment_conversion['pv_users'] * 100

        return segment_conversion

    def get_all_conversion_analysis(self, user_segments=None):
        """
        获取所有转化率分析结果

        :param user_segments: 用户分群数据（可选）
        :return: 完整的转化率分析结果字典
        """
        return {
            'conversion_funnel': self.analyze_conversion_funnel(),
            'user_conversion_paths': self.analyze_user_conversion_paths(),
            'conversion_by_time': self.analyze_conversion_by_time(),
            'conversion_by_category': self.analyze_conversion_by_category(),
            'conversion_by_user_segment': self.analyze_conversion_by_user_segment(user_segments)
        }