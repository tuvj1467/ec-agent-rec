import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import logging

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualization:
    """可视化模块"""

    def __init__(self, output_dir='./output/charts'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_user_activity(self, daily_active, weekly_active):
        """
        绘制用户活跃度图表

        :param daily_active: 日活跃用户序列
        :param weekly_active: 周活跃用户序列
        """
        logger.info("绘制用户活跃度图表")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        axes[0].plot(daily_active.index, daily_active.values, color='#1677ff')
        axes[0].set_title('日活跃用户趋势', fontsize=14)
        axes[0].set_xlabel('日期', fontsize=12)
        axes[0].set_ylabel('活跃用户数', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(weekly_active.index.astype(str), weekly_active.values, color='#52c41a')
        axes[1].set_title('周活跃用户趋势', fontsize=14)
        axes[1].set_xlabel('周', fontsize=12)
        axes[1].set_ylabel('活跃用户数', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'user_activity.png'), dpi=150)
        plt.close()

    def plot_user_time_distribution(self, hour_distribution, weekday_distribution):
        """
        绘制用户活跃时间分布图表

        :param hour_distribution: 小时分布序列
        :param weekday_distribution: 周分布序列
        """
        logger.info("绘制用户活跃时间分布图表")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].bar(hour_distribution.index, hour_distribution.values, color='#722ed1')
        axes[0].set_title('用户活跃小时分布', fontsize=14)
        axes[0].set_xlabel('小时', fontsize=12)
        axes[0].set_ylabel('活跃用户数', fontsize=12)
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(weekday_distribution.index, weekday_distribution.values, color='#eb2f96')
        axes[1].set_title('用户活跃周分布', fontsize=14)
        axes[1].set_xlabel('星期', fontsize=12)
        axes[1].set_ylabel('活跃用户数', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'user_time_distribution.png'), dpi=150)
        plt.close()

    def plot_conversion_funnel(self, funnel_data):
        """
        绘制转化漏斗图

        :param funnel_data: 漏斗数据字典
        """
        logger.info("绘制转化漏斗图")

        steps = ['浏览', '加购', '收藏', '购买']
        counts = [
            funnel_data['pv']['count'],
            funnel_data['cart']['count'],
            funnel_data['fav']['count'],
            funnel_data['buy']['count']
        ]

        colors = ['#1677ff', '#52c41a', '#faad14', '#f5222d']

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (step, count) in enumerate(zip(steps, counts)):
            width = 1 - i * 0.15
            ax.bar(0, count, width=width, color=colors[i], label=f'{step}: {count:,}',
                   align='center', edgecolor='white')

        ax.set_xticks([])
        ax.set_ylabel('行为数量', fontsize=12)
        ax.set_title('转化漏斗', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'conversion_funnel.png'), dpi=150)
        plt.close()

    def plot_top_products(self, top_pv, top_buy):
        """
        绘制热门商品TOP图表

        :param top_pv: TOP浏览商品数据框
        :param top_buy: TOP购买商品数据框
        """
        logger.info("绘制热门商品TOP图表")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].barh(top_pv['item_id'].astype(str), top_pv['count'], color='#1677ff')
        axes[0].set_title('TOP20 浏览商品', fontsize=14)
        axes[0].set_xlabel('浏览次数', fontsize=12)
        axes[0].set_ylabel('商品ID', fontsize=12)
        axes[0].invert_yaxis()

        axes[1].barh(top_buy['item_id'].astype(str), top_buy['count'], color='#f5222d')
        axes[1].set_title('TOP20 购买商品', fontsize=14)
        axes[1].set_xlabel('购买次数', fontsize=12)
        axes[1].set_ylabel('商品ID', fontsize=12)
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_products.png'), dpi=150)
        plt.close()

    def plot_category_hotness(self, category_pv, category_buy):
        """
        绘制商品分类热度图表

        :param category_pv: 分类浏览统计数据框
        :param category_buy: 分类购买统计数据框
        """
        logger.info("绘制商品分类热度图表")

        top_categories_pv = category_pv.head(10)
        top_categories_buy = category_buy.head(10)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].bar(top_categories_pv['category_id'].astype(str), top_categories_pv['count'], color='#1677ff')
        axes[0].set_title('TOP10 浏览分类', fontsize=14)
        axes[0].set_xlabel('分类ID', fontsize=12)
        axes[0].set_ylabel('浏览次数', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(top_categories_buy['category_id'].astype(str), top_categories_buy['count'], color='#f5222d')
        axes[1].set_title('TOP10 购买分类', fontsize=14)
        axes[1].set_xlabel('分类ID', fontsize=12)
        axes[1].set_ylabel('购买次数', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'category_hotness.png'), dpi=150)
        plt.close()

    def plot_conversion_by_time(self, hourly_conversion, weekday_conversion):
        """
        绘制时间维度转化率图表

        :param hourly_conversion: 小时转化率序列
        :param weekday_conversion: 周转化率序列
        """
        logger.info("绘制时间维度转化率图表")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(hourly_conversion.index, hourly_conversion.values, marker='o', color='#1677ff')
        axes[0].set_title('小时转化率趋势', fontsize=14)
        axes[0].set_xlabel('小时', fontsize=12)
        axes[0].set_ylabel('转化率(%)', fontsize=12)
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(weekday_conversion.index, weekday_conversion.values, color='#52c41a')
        axes[1].set_title('周转化率对比', fontsize=14)
        axes[1].set_xlabel('星期', fontsize=12)
        axes[1].set_ylabel('转化率(%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'conversion_by_time.png'), dpi=150)
        plt.close()

    def plot_user_segmentation(self, segment_distribution):
        """
        绘制用户分群分布图表

        :param segment_distribution: 用户分群分布字典
        """
        logger.info("绘制用户分群分布图表")

        segments = list(segment_distribution.keys())
        counts = list(segment_distribution.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#1677ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#eb2f96']

        ax.pie(counts, labels=segments, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title('用户分群分布', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'user_segmentation.png'), dpi=150)
        plt.close()

    def plot_retention(self, retention_data):
        """
        绘制用户留存率图表

        :param retention_data: 留存率序列
        """
        logger.info("绘制用户留存率图表")

        days = list(retention_data.keys())
        rates = list(retention_data.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(days, rates, marker='o', color='#1677ff', linewidth=2)
        ax.bar(days, rates, color='#1677ff', alpha=0.3)

        for day, rate in zip(days, rates):
            ax.text(day, rate, f'{rate:.2f}%', ha='center', va='bottom', fontsize=10)

        ax.set_title('用户留存率', fontsize=14)
        ax.set_xlabel('天数', fontsize=12)
        ax.set_ylabel('留存率(%)', fontsize=12)
        ax.set_xticks(days)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'user_retention.png'), dpi=150)
        plt.close()

    def plot_behavior_distribution(self, behavior_distribution):
        """
        绘制行为类型分布图表

        :param behavior_distribution: 行为分布字典
        """
        logger.info("绘制行为类型分布图表")

        behaviors = list(behavior_distribution.keys())
        counts = list(behavior_distribution.values())

        behavior_names = {'pv': '浏览', 'buy': '购买', 'cart': '加购', 'fav': '收藏'}
        labels = [behavior_names.get(b, b) for b in behaviors]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#1677ff', '#f5222d', '#52c41a', '#faad14']

        ax.bar(labels, counts, color=colors)
        ax.set_title('用户行为类型分布', fontsize=14)
        ax.set_xlabel('行为类型', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)

        for i, count in enumerate(counts):
            ax.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=10)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'behavior_distribution.png'), dpi=150)
        plt.close()

    def plot_conversion_distribution(self, conversion_distribution):
        """
        绘制转化率分布图表

        :param conversion_distribution: 转化率分布字典
        """
        logger.info("绘制转化率分布图表")

        bins = list(conversion_distribution['percentage'].keys())
        percentages = list(conversion_distribution['percentage'].values())

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#f5222d', '#fa8c16', '#faad14', '#a0d911', '#52c41a', '#13c2c2']

        ax.bar(bins, percentages, color=colors)
        ax.set_title('商品转化率分布', fontsize=14)
        ax.set_xlabel('转化率区间', fontsize=12)
        ax.set_ylabel('占比(%)', fontsize=12)

        for i, pct in enumerate(percentages):
            ax.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'conversion_distribution.png'), dpi=150)
        plt.close()

    def generate_all_charts(self, analysis_results):
        """
        根据分析结果生成所有图表

        :param analysis_results: 分析结果字典
        """
        logger.info("开始生成所有可视化图表")

        try:
            if 'user_activity' in analysis_results:
                self.plot_user_activity(
                    analysis_results['user_activity']['daily_active_users'],
                    analysis_results['user_activity']['weekly_active_users']
                )

            if 'user_time_distribution' in analysis_results:
                self.plot_user_time_distribution(
                    pd.Series(analysis_results['user_time_distribution']['hour_distribution']),
                    pd.Series(analysis_results['user_time_distribution']['weekday_distribution'])
                )

            if 'conversion_funnel' in analysis_results:
                self.plot_conversion_funnel(
                    analysis_results['conversion_funnel']['funnel_data']
                )

            if 'top_products' in analysis_results:
                self.plot_top_products(
                    analysis_results['top_products']['top_pv_products'],
                    analysis_results['top_products']['top_buy_products']
                )

            if 'category_hotness' in analysis_results:
                self.plot_category_hotness(
                    analysis_results['category_hotness']['category_pv_rank'],
                    analysis_results['category_hotness']['category_buy_rank']
                )

            if 'conversion_by_time' in analysis_results:
                self.plot_conversion_by_time(
                    analysis_results['conversion_by_time']['hourly_conversion'],
                    analysis_results['conversion_by_time']['weekday_conversion']
                )

            if 'user_segmentation' in analysis_results:
                self.plot_user_segmentation(
                    analysis_results['user_segmentation']['segment_distribution']
                )

            if 'user_retention' in analysis_results:
                self.plot_retention(
                    analysis_results['user_retention']['retention_table']
                )

            if 'data_summary' in analysis_results:
                self.plot_behavior_distribution(
                    analysis_results['data_summary']['behavior_distribution']
                )

            if 'product_conversion' in analysis_results:
                self.plot_conversion_distribution(
                    analysis_results['product_conversion']['conversion_distribution']
                )

            logger.info(f"所有图表已生成，保存在 {self.output_dir}")

        except Exception as e:
            logger.error(f"生成图表时出错: {e}")