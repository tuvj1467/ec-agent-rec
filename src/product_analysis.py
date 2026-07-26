import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductAnalysis:
    """商品分析模块"""

    def __init__(self, df):
        self.df = df

    def analyze_top_products(self, top_n=20):
        """
        热门商品TOP排行榜分析

        :param top_n: TOP数量，默认20
        :return: 热门商品分析结果
        """
        logger.info("开始热门商品TOP排行榜分析")

        pv_rank = self._calculate_product_rank('pv', top_n)
        buy_rank = self._calculate_product_rank('buy', top_n)
        cart_rank = self._calculate_product_rank('cart', top_n)

        result = {
            'top_pv_products': pv_rank,
            'top_buy_products': buy_rank,
            'top_cart_products': cart_rank,
            'summary': {
                'total_items': int(self.df['item_id'].nunique()),
                'avg_pv_per_item': float(self.df[self.df['behavior_type'] == 'pv'].groupby('item_id').size().mean()),
                'avg_buy_per_item': float(self.df[self.df['behavior_type'] == 'buy'].groupby('item_id').size().mean())
            }
        }

        logger.info("热门商品TOP排行榜分析完成")
        return result

    def _calculate_product_rank(self, behavior_type, top_n):
        """
        计算商品排名

        :param behavior_type: 行为类型
        :param top_n: TOP数量
        :return: 商品排名数据框
        """
        rank = self.df[self.df['behavior_type'] == behavior_type] \
            .groupby(['item_id', 'category_id'])['user_id'] \
            .count() \
            .reset_index() \
            .rename(columns={'user_id': 'count'}) \
            .sort_values('count', ascending=False) \
            .head(top_n) \
            .reset_index(drop=True)

        rank['rank'] = rank.index + 1
        return rank

    def analyze_category_hotness(self):
        """
        商品分类热度分析

        :return: 分类热度分析结果
        """
        logger.info("开始商品分类热度分析")

        category_pv = self._calculate_category_stat('pv')
        category_buy = self._calculate_category_stat('buy')
        category_conversion = self._calculate_category_conversion(category_pv, category_buy)

        result = {
            'category_pv_rank': category_pv,
            'category_buy_rank': category_buy,
            'category_conversion': category_conversion,
            'total_categories': int(self.df['category_id'].nunique())
        }

        logger.info("商品分类热度分析完成")
        return result

    def _calculate_category_stat(self, behavior_type):
        """
        计算分类统计

        :param behavior_type: 行为类型
        :return: 分类统计数据框
        """
        stat = self.df[self.df['behavior_type'] == behavior_type] \
            .groupby('category_id') \
            .agg(
                count=('item_id', 'count'),
                unique_users=('user_id', 'nunique'),
                unique_items=('item_id', 'nunique')
            ) \
            .sort_values('count', ascending=False) \
            .reset_index()

        stat['rank'] = stat.index + 1
        return stat

    def _calculate_category_conversion(self, category_pv, category_buy):
        """
        计算分类转化率

        :param category_pv: 分类浏览统计
        :param category_buy: 分类购买统计
        :return: 分类转化率数据框
        """
        merged = category_pv.merge(category_buy, on='category_id', suffixes=('_pv', '_buy'))
        merged['conversion_rate'] = merged['count_buy'] / merged['count_pv'] * 100
        merged = merged.sort_values('conversion_rate', ascending=False)
        return merged

    def analyze_product_conversion(self):
        """
        商品转化率分析

        :return: 商品转化率分析结果
        """
        logger.info("开始商品转化率分析")

        product_stats = self._calculate_product_stats()
        conversion_distribution = self._analyze_conversion_distribution(product_stats)

        result = {
            'product_conversion_stats': {
                'avg_conversion_rate': float(product_stats['conversion_rate'].mean()),
                'median_conversion_rate': float(product_stats['conversion_rate'].median()),
                'max_conversion_rate': float(product_stats['conversion_rate'].max()),
                'min_conversion_rate': float(product_stats['conversion_rate'].min()),
                'products_with_conversion': int((product_stats['conversion_rate'] > 0).sum()),
                'total_products': int(len(product_stats))
            },
            'conversion_distribution': conversion_distribution,
            'top_conversion_products': product_stats.sort_values('conversion_rate', ascending=False).head(10).to_dict('records')
        }

        logger.info("商品转化率分析完成")
        return result

    def _calculate_product_stats(self):
        """
        计算商品统计信息

        :return: 商品统计数据框
        """
        pv_stats = self.df[self.df['behavior_type'] == 'pv'] \
            .groupby('item_id')['user_id'] \
            .count() \
            .reset_index() \
            .rename(columns={'user_id': 'pv_count'})

        buy_stats = self.df[self.df['behavior_type'] == 'buy'] \
            .groupby('item_id')['user_id'] \
            .count() \
            .reset_index() \
            .rename(columns={'user_id': 'buy_count'})

        stats = pv_stats.merge(buy_stats, on='item_id', how='left').fillna(0)
        stats['conversion_rate'] = stats['buy_count'] / stats['pv_count'] * 100

        return stats

    def _analyze_conversion_distribution(self, product_stats):
        """
        分析转化率分布

        :param product_stats: 商品统计数据框
        :return: 转化率分布字典
        """
        bins = [0, 0.1, 0.5, 1, 5, 10, 100]
        labels = ['0-0.1%', '0.1-0.5%', '0.5-1%', '1-5%', '5-10%', '>10%']
        product_stats['conversion_bin'] = pd.cut(product_stats['conversion_rate'], bins=bins, labels=labels, right=False)

        distribution = product_stats['conversion_bin'].value_counts().to_dict()
        total = sum(distribution.values())
        distribution_percent = {k: float(v / total * 100) for k, v in distribution.items()}

        return {
            'count': distribution,
            'percentage': distribution_percent
        }

    def analyze_product_association(self, top_n=10):
        """
        商品关联分析（基于共同购买用户）

        :param top_n: TOP数量，默认10
        :return: 商品关联分析结果
        """
        logger.info("开始商品关联分析")

        buy_data = self.df[self.df['behavior_type'] == 'buy']
        if len(buy_data) == 0:
            logger.warning("没有购买数据，跳过商品关联分析")
            return {'error': '没有购买数据'}

        frequent_items = self._find_frequent_items(buy_data, top_n)
        item_pairs = self._find_item_pairs(buy_data, top_n)

        result = {
            'frequent_items': frequent_items,
            'item_pairs': item_pairs
        }

        logger.info("商品关联分析完成")
        return result

    def _find_frequent_items(self, buy_data, top_n):
        """
        发现频繁购买商品

        :param buy_data: 购买数据
        :param top_n: TOP数量
        :return: 频繁商品列表
        """
        frequent = buy_data.groupby('item_id')['user_id'] \
            .nunique() \
            .sort_values(ascending=False) \
            .head(top_n) \
            .reset_index() \
            .rename(columns={'user_id': 'buyer_count'})

        frequent['rank'] = frequent.index + 1
        return frequent

    def _find_item_pairs(self, buy_data, top_n):
        """
        发现商品关联对

        :param buy_data: 购买数据
        :param top_n: TOP数量
        :return: 商品关联对列表
        """
        user_items = buy_data.groupby('user_id')['item_id'].apply(list).reset_index()

        pairs = []
        for _, row in user_items.iterrows():
            items = row['item_id']
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    pairs.append((min(items[i], items[j]), max(items[i], items[j])))

        if not pairs:
            return []

        pair_counts = pd.DataFrame(pairs, columns=['item1', 'item2']) \
            .groupby(['item1', 'item2']) \
            .size() \
            .reset_index(name='count') \
            .sort_values('count', ascending=False) \
            .head(top_n)

        pair_counts['rank'] = pair_counts.index + 1
        return pair_counts

    def analyze_product_time_trend(self):
        """
        商品时间趋势分析

        :return: 时间趋势分析结果
        """
        logger.info("开始商品时间趋势分析")

        daily_trend = self._calculate_daily_trend()
        hourly_trend = self._calculate_hourly_trend()

        result = {
            'daily_pv_trend': daily_trend['pv'].to_dict(),
            'daily_buy_trend': daily_trend['buy'].to_dict(),
            'hourly_pv_trend': hourly_trend['pv'].to_dict(),
            'hourly_buy_trend': hourly_trend['buy'].to_dict(),
            'peak_time': {
                'day': str(daily_trend['pv'].idxmax()),
                'hour': int(hourly_trend['pv'].idxmax())
            }
        }

        logger.info("商品时间趋势分析完成")
        return result

    def _calculate_daily_trend(self):
        """
        计算每日趋势

        :return: 每日趋势字典（值为 Series）
        """
        trends = {}
        for behavior in ['pv', 'buy']:
            trend = self.df[self.df['behavior_type'] == behavior] \
                .groupby('date')['item_id'] \
                .count() \
                .sort_index()
            trends[behavior] = trend
        return trends

    def _calculate_hourly_trend(self):
        """
        计算小时趋势

        :return: 小时趋势字典（值为 Series）
        """
        trends = {}
        for behavior in ['pv', 'buy']:
            trend = self.df[self.df['behavior_type'] == behavior] \
                .groupby('hour')['item_id'] \
                .count() \
                .sort_index()
            trends[behavior] = trend
        return trends

    def get_all_product_analysis(self):
        """
        获取所有商品分析结果

        :return: 完整的商品分析结果字典
        """
        return {
            'top_products': self.analyze_top_products(),
            'category_hotness': self.analyze_category_hotness(),
            'product_conversion': self.analyze_product_conversion(),
            'product_association': self.analyze_product_association(),
            'product_time_trend': self.analyze_product_time_trend()
        }