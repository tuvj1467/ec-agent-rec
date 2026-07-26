import argparse
import json
import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _json_serialize(obj):
    """JSON序列化辅助函数，处理numpy类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

from src.data_loader import DataLoader
from src.user_analysis import UserAnalysis
from src.product_analysis import ProductAnalysis
from src.conversion_analysis import ConversionAnalysis
from src.visualization import Visualization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='电商用户行为分析项目')
    parser.add_argument('--sample', type=int, default=None, help='采样数量，如 1000000 表示使用100万条数据')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    parser.add_argument('--charts', action='store_true', default=True, help='生成可视化图表')
    parser.add_argument('--report', action='store_true', default=True, help='生成分析报告')
    return parser.parse_args()


def run_analysis(sample_size=None, output_dir='./output'):
    """
    运行完整分析流程

    :param sample_size: 采样数量
    :param output_dir: 输出目录
    :return: 分析结果字典
    """
    logger.info("=" * 60)
    logger.info("电商用户行为分析项目 - 开始执行")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    data_loader = DataLoader()

    logger.info("\n[阶段1] 数据加载")
    df = data_loader.load_data(sample_size=sample_size)
    if df is None:
        logger.error("数据加载失败，退出程序")
        return None

    logger.info("\n[阶段2] 数据预处理")
    df = data_loader.preprocess()

    data_summary = data_loader.get_data_summary()
    logger.info(f"\n数据摘要:")
    logger.info(f"  总记录数: {data_summary['total_records']:,}")
    logger.info(f"  总用户数: {data_summary['total_users']:,}")
    logger.info(f"  总商品数: {data_summary['total_items']:,}")
    logger.info(f"  总分类数: {data_summary['total_categories']:,}")
    logger.info(f"  行为分布: {data_summary['behavior_distribution']}")
    logger.info(f"  时间范围: {data_summary['date_range']['start']} ~ {data_summary['date_range']['end']}")

    logger.info("\n[阶段3] 用户分析")
    user_analysis = UserAnalysis(df)
    user_activity = user_analysis.analyze_user_activity()
    user_behavior_path = user_analysis.analyze_user_behavior_path()
    user_retention = user_analysis.analyze_user_retention()
    user_segmentation = user_analysis.analyze_user_segmentation()
    user_time_distribution = user_analysis.analyze_user_time_distribution()

    logger.info(f"  日均活跃用户: {user_activity['summary']['avg_daily_active']:,}")
    logger.info(f"  浏览→购买转化率: {user_behavior_path['conversion_metrics']['pv_to_buy_conversion']:.2f}%")
    logger.info(f"  7日留存率: {user_retention['summary']['day_7_retention']:.2f}%")

    logger.info("\n[阶段4] 商品分析")
    product_analysis = ProductAnalysis(df)
    top_products = product_analysis.analyze_top_products()
    category_hotness = product_analysis.analyze_category_hotness()
    product_conversion = product_analysis.analyze_product_conversion()
    product_association = product_analysis.analyze_product_association()
    product_time_trend = product_analysis.analyze_product_time_trend()

    logger.info(f"  总商品数: {top_products['summary']['total_items']:,}")
    logger.info(f"  平均商品转化率: {product_conversion['product_conversion_stats']['avg_conversion_rate']:.2f}%")

    logger.info("\n[阶段5] 转化率分析")
    user_segments = user_segmentation['user_segments']
    conversion_analysis = ConversionAnalysis(df)
    conversion_funnel = conversion_analysis.analyze_conversion_funnel()
    user_conversion_paths = conversion_analysis.analyze_user_conversion_paths()
    conversion_by_time = conversion_analysis.analyze_conversion_by_time()
    conversion_by_category = conversion_analysis.analyze_conversion_by_category()
    conversion_by_user_segment = conversion_analysis.analyze_conversion_by_user_segment(user_segments)

    logger.info(f"  整体转化率: {conversion_funnel['overall_conversion']:.2f}%")
    logger.info(f"  最佳转化时段: {conversion_by_time['best_hour']}点")

    analysis_results = {
        'data_summary': data_summary,
        'user_activity': user_activity,
        'user_behavior_path': user_behavior_path,
        'user_retention': user_retention,
        'user_segmentation': user_segmentation,
        'user_time_distribution': user_time_distribution,
        'top_products': top_products,
        'category_hotness': category_hotness,
        'product_conversion': product_conversion,
        'product_association': product_association,
        'product_time_trend': product_time_trend,
        'conversion_funnel': conversion_funnel,
        'user_conversion_paths': user_conversion_paths,
        'conversion_by_time': conversion_by_time,
        'conversion_by_category': conversion_by_category,
        'conversion_by_user_segment': conversion_by_user_segment
    }

    logger.info("\n" + "=" * 60)
    logger.info("电商用户行为分析项目 - 分析完成")
    logger.info("=" * 60)

    return analysis_results


def generate_report(analysis_results, output_dir='./output'):
    """
    生成分析报告

    :param analysis_results: 分析结果字典
    :param output_dir: 输出目录
    """
    logger.info(f"\n生成分析报告到 {output_dir}")

    report = {
        'project': '电商用户行为分析',
        'data_summary': analysis_results['data_summary'],
        'key_metrics': {
            'daily_active_users': analysis_results['user_activity']['summary']['avg_daily_active'],
            'weekly_active_users': int(analysis_results['user_activity']['weekly_active_users'].mean()),
            'pv_to_buy_conversion': analysis_results['user_behavior_path']['conversion_metrics']['pv_to_buy_conversion'],
            'day_7_retention': analysis_results['user_retention']['summary']['day_7_retention'],
            'avg_product_conversion': analysis_results['product_conversion']['product_conversion_stats']['avg_conversion_rate'],
            'total_users': analysis_results['data_summary']['total_users'],
            'total_items': analysis_results['data_summary']['total_items'],
            'total_records': analysis_results['data_summary']['total_records']
        },
        'analysis': {
            'user_analysis': {
                'activity': analysis_results['user_activity']['summary'],
                'conversion': analysis_results['user_behavior_path']['conversion_metrics'],
                'retention': analysis_results['user_retention']['summary'],
                'segments': analysis_results['user_segmentation']['segment_distribution'],
                'time_distribution': analysis_results['user_time_distribution']
            },
            'product_analysis': {
                'top_pv': analysis_results['top_products']['top_pv_products'].head(5).to_dict('records'),
                'top_buy': analysis_results['top_products']['top_buy_products'].head(5).to_dict('records'),
                'category_conversion': analysis_results['category_hotness']['category_conversion'].head(5).to_dict('records'),
                'conversion_stats': analysis_results['product_conversion']['product_conversion_stats']
            },
            'conversion_analysis': {
                'funnel': analysis_results['conversion_funnel']['funnel_rates'],
                'best_time': {
                    'hour': analysis_results['conversion_by_time']['best_hour'],
                    'day': analysis_results['conversion_by_time']['best_day']
                },
                'common_paths': analysis_results['user_conversion_paths']['common_paths'][:5]
            }
        },
        'insights': generate_insights(analysis_results),
        'recommendations': generate_recommendations(analysis_results)
    }

    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_serialize)

    logger.info(f"分析报告已保存到 {report_path}")

    summary_txt = generate_summary_text(report)
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_txt)

    logger.info(f"分析摘要已保存到 {summary_path}")


def generate_insights(analysis_results):
    """
    生成业务洞察

    :param analysis_results: 分析结果字典
    :return: 洞察列表
    """
    insights = []

    conversion_rate = analysis_results['user_behavior_path']['conversion_metrics']['pv_to_buy_conversion']
    if conversion_rate < 5:
        insights.append(f"当前浏览到购买转化率为 {conversion_rate:.2f}%，低于行业平均水平，建议优化转化流程")

    day_7_retention = analysis_results['user_retention']['summary']['day_7_retention']
    if day_7_retention < 30:
        insights.append(f"7日留存率为 {day_7_retention:.2f}%，用户粘性有待提升")

    peak_hours = analysis_results['user_time_distribution']['peak_hours']
    insights.append(f"用户活跃高峰时段为 {peak_hours} 点，建议在这些时段加大营销投入")

    segment_dist = analysis_results['user_segmentation']['segment_distribution']
    if '浏览用户' in segment_dist and segment_dist['浏览用户'] > segment_dist.get('购买用户', 0):
        insights.append("浏览用户占比较高但购买转化率低，存在优化空间")

    return insights


def generate_recommendations(analysis_results):
    """
    生成业务建议

    :param analysis_results: 分析结果字典
    :return: 建议列表
    """
    recommendations = [
        "优化商品详情页，提升用户浏览到购买的转化率",
        "针对高活跃时段制定促销活动策略",
        "加强新用户引导，提升首日留存率",
        "分析低转化分类的原因，优化商品结构",
        "针对高价值用户群体提供专属服务和优惠",
        "优化购物车流程，降低加购到购买的流失率"
    ]
    return recommendations


def generate_summary_text(report):
    """
    生成文本格式的分析摘要

    :param report: 分析报告字典
    :return: 文本摘要
    """
    txt = "=" * 60 + "\n"
    txt += "电商用户行为分析报告\n"
    txt += "=" * 60 + "\n\n"

    txt += "一、数据概览\n"
    txt += "-" * 40 + "\n"
    txt += f"总记录数: {report['data_summary']['total_records']:,}\n"
    txt += f"总用户数: {report['data_summary']['total_users']:,}\n"
    txt += f"总商品数: {report['data_summary']['total_items']:,}\n"
    txt += f"总分类数: {report['data_summary']['total_categories']:,}\n"
    txt += f"行为分布: {report['data_summary']['behavior_distribution']}\n"
    txt += f"时间范围: {report['data_summary']['date_range']['start']} ~ {report['data_summary']['date_range']['end']}\n\n"

    txt += "二、核心指标\n"
    txt += "-" * 40 + "\n"
    txt += f"日均活跃用户: {report['key_metrics']['daily_active_users']:,}\n"
    txt += f"周均活跃用户: {report['key_metrics']['weekly_active_users']:,}\n"
    txt += f"浏览→购买转化率: {report['key_metrics']['pv_to_buy_conversion']:.2f}%\n"
    txt += f"7日留存率: {report['key_metrics']['day_7_retention']:.2f}%\n"
    txt += f"平均商品转化率: {report['key_metrics']['avg_product_conversion']:.2f}%\n\n"

    txt += "三、业务洞察\n"
    txt += "-" * 40 + "\n"
    for i, insight in enumerate(report['insights'], 1):
        txt += f"{i}. {insight}\n"
    txt += "\n"

    txt += "四、优化建议\n"
    txt += "-" * 40 + "\n"
    for i, recommendation in enumerate(report['recommendations'], 1):
        txt += f"{i}. {recommendation}\n"
    txt += "\n"

    txt += "=" * 60 + "\n"
    txt += "报告生成时间: 自动生成\n"
    txt += "=" * 60 + "\n"

    return txt


def main():
    args = parse_args()

    logger.info(f"参数设置:")
    logger.info(f"  采样数量: {'全部' if args.sample is None else args.sample}")
    logger.info(f"  输出目录: {args.output}")
    logger.info(f"  生成图表: {args.charts}")
    logger.info(f"  生成报告: {args.report}")

    analysis_results = run_analysis(sample_size=args.sample, output_dir=args.output)

    if analysis_results is None:
        logger.error("分析失败")
        sys.exit(1)

    if args.report:
        generate_report(analysis_results, output_dir=args.output)

    if args.charts:
        charts_dir = os.path.join(args.output, 'charts')
        visualization = Visualization(output_dir=charts_dir)
        visualization.generate_all_charts(analysis_results)

    logger.info("\n" + "=" * 60)
    logger.info("所有任务完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()