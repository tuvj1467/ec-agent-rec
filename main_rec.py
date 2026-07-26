#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电商推荐 Agent 项目 - 主入口

支持的运行模式：
1. data     - 数据清洗和图构造
2. train    - LightGCN 模型训练
3. index    - 向量入库 Qdrant
4. chat     - 启动电商 Agent 对话
5. pipeline - 完整流程（数据→训练→入库→对话）
"""

import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ec_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_data_pipeline(sample_size=None):
    """运行数据处理流程：数据清洗 + 图构造"""
    from data_processing.data_cleaner import DataCleaner
    from data_processing.graph_builder import GraphBuilder

    logger.info("=" * 60)
    logger.info("阶段1: 数据清洗与图构造")
    logger.info("=" * 60)

    cleaner = DataCleaner(output_dir=Config.DATA_CONFIG['output_dir'])
    df = cleaner.load_and_clean(sample_size=sample_size)
    if df is None:
        logger.error("数据清洗失败")
        return None

    summary = cleaner.get_data_summary()
    logger.info(f"数据摘要: {summary}")

    cleaner.save_cleaned_data()

    graph_builder = GraphBuilder(output_dir=Config.GRAPH_CONFIG['output_dir'])
    adj_matrix = graph_builder.build_graph(df, use_buy_only=Config.GRAPH_CONFIG['use_buy_only'])

    graph_info = graph_builder.get_graph_info()
    logger.info(f"图信息: {graph_info}")

    graph_builder.save_all()

    logger.info("数据处理完成")
    return graph_builder


def run_training(graph_builder=None, sample_size=None):
    """运行模型训练流程"""
    import torch
    from data_processing.dataset import split_data, build_dataloaders
    from model.lightgcn import LightGCN
    from model.trainer import Trainer
    from model.evaluator import Evaluator
    from model.embedding_exporter import EmbeddingExporter

    logger.info("=" * 60)
    logger.info("阶段2: LightGCN 模型训练")
    logger.info("=" * 60)

    if graph_builder is None:
        from data_processing.graph_builder import GraphBuilder
        graph_builder = GraphBuilder(output_dir=Config.GRAPH_CONFIG['output_dir'])
        graph_builder.load_all()

    adj_matrix = graph_builder.adj_matrix
    user_items = graph_builder.get_user_items_dict()

    train_adj, val_adj, test_adj, train_user_items, val_user_items, test_user_items = split_data(
        adj_matrix,
        train_ratio=Config.DATASET_CONFIG['train_ratio'],
        val_ratio=Config.DATASET_CONFIG['val_ratio'],
        test_ratio=Config.DATASET_CONFIG['test_ratio'],
        seed=Config.DATASET_CONFIG['seed']
    )

    train_loader = build_dataloaders(
        train_adj,
        batch_size=Config.TRAIN_CONFIG['batch_size'],
        num_negatives=Config.TRAIN_CONFIG['num_negatives']
    )

    num_users = graph_builder.num_users
    num_items = graph_builder.num_items

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=Config.MODEL_CONFIG['embedding_dim'],
        num_layers=Config.MODEL_CONFIG['num_layers'],
        dropout=Config.MODEL_CONFIG['dropout']
    )

    from data_processing.graph_builder import GraphBuilder
    lightgcn_graph = GraphBuilder(output_dir=Config.GRAPH_CONFIG['output_dir'])
    lightgcn_graph.adj_matrix = train_adj
    lightgcn_graph.num_users = num_users
    lightgcn_graph.num_items = num_items
    norm_adj = lightgcn_graph.build_lightgcn_adj_matrix()
    model.set_adj_matrix(norm_adj)

    evaluator = Evaluator(
        test_user_items=val_user_items,
        train_user_items=train_user_items,
        num_items=num_items,
        k_list=Config.EVAL_CONFIG['k_list']
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        evaluator=evaluator,
        config=Config.TRAIN_CONFIG
    )

    train_history = trainer.train()

    best_model_path = os.path.join(Config.TRAIN_CONFIG['save_dir'], 'best_model.pt')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    test_evaluator = Evaluator(
        test_user_items=test_user_items,
        train_user_items=train_user_items,
        num_items=num_items,
        k_list=Config.EVAL_CONFIG['k_list']
    )
    test_metrics = test_evaluator.evaluate_simple(model, num_sample_users=500)
    logger.info(f"测试集指标: {test_metrics}")

    exporter = EmbeddingExporter(model, output_dir=Config.EMBEDDING_CONFIG['output_dir'])
    user_embeddings, item_embeddings = exporter.export_embeddings()
    exporter.save_embeddings(user_embeddings, item_embeddings)

    stats = exporter.get_embedding_stats(user_embeddings, item_embeddings)
    logger.info(f"向量统计: {stats}")

    logger.info("模型训练完成")
    return model, user_embeddings, item_embeddings


def run_vector_index(item_embeddings=None):
    """运行向量入库流程"""
    from vector_store.qdrant_client import QdrantClientWrapper
    from vector_store.indexer import VectorIndexer

    logger.info("=" * 60)
    logger.info("阶段3: 向量入库 Qdrant")
    logger.info("=" * 60)

    qdrant_client = QdrantClientWrapper(
        host=Config.QDRANT_CONFIG['host'],
        port=Config.QDRANT_CONFIG['port'],
        collection_name=Config.QDRANT_CONFIG['collection_name'],
        embedding_dim=Config.QDRANT_CONFIG['embedding_dim'],
        distance=Config.QDRANT_CONFIG['distance']
    )

    item_mapping_path = os.path.join(Config.GRAPH_CONFIG['output_dir'], 'item_mapping.json')

    indexer = VectorIndexer(
        qdrant_client=qdrant_client,
        item_mapping_path=item_mapping_path if os.path.exists(item_mapping_path) else None,
        output_dir=Config.EMBEDDING_CONFIG['output_dir']
    )

    if item_embeddings is not None:
        indexer.index_item_embeddings(item_embeddings, batch_size=Config.QDRANT_CONFIG['batch_size'])
    else:
        indexer.index_item_embeddings(batch_size=Config.QDRANT_CONFIG['batch_size'])

    stats = indexer.get_index_stats()
    logger.info(f"索引统计: {stats}")

    logger.info("向量入库完成")
    return qdrant_client


def run_chat_mode():
    """启动电商 Agent 对话模式"""
    import numpy as np
    from vector_store.qdrant_client import QdrantClientWrapper
    from vector_store.retriever import VectorRetriever
    from cache.redis_client import RedisClientWrapper
    from cache.recall_cache import RecallCache
    from agent.recommendation_engine import RecommendationEngine
    from agent.ecommerce_agent import EcommerceAgent

    logger.info("=" * 60)
    logger.info("阶段4: 电商 Agent 对话")
    logger.info("=" * 60)

    qdrant_client = QdrantClientWrapper(
        host=Config.QDRANT_CONFIG['host'],
        port=Config.QDRANT_CONFIG['port'],
        collection_name=Config.QDRANT_CONFIG['collection_name'],
        embedding_dim=Config.QDRANT_CONFIG['embedding_dim'],
        distance=Config.QDRANT_CONFIG['distance']
    )

    vector_retriever = VectorRetriever(
        qdrant_client=qdrant_client,
        top_k=Config.QDRANT_CONFIG['top_k']
    )

    redis_client = RedisClientWrapper(
        host=Config.REDIS_CONFIG['host'],
        port=Config.REDIS_CONFIG['port'],
        db=Config.REDIS_CONFIG['db'],
        password=Config.REDIS_CONFIG['password']
    )

    recall_cache = RecallCache(
        redis_client=redis_client,
        ttl=Config.REDIS_CONFIG['ttl'],
        prefix=Config.REDIS_CONFIG['prefix']
    )

    rec_engine = RecommendationEngine(
        vector_retriever=vector_retriever,
        recall_cache=recall_cache if Config.AGENT_CONFIG['use_cache'] else None,
        top_k=Config.AGENT_CONFIG['top_k']
    )

    rec_engine.load_user_embeddings(
        os.path.join(Config.EMBEDDING_CONFIG['output_dir'], 'user_embedding.npy')
    )

    rec_engine.load_user_mapping(
        os.path.join(Config.GRAPH_CONFIG['output_dir'], 'user_mapping.json')
    )

    agent = EcommerceAgent(recommendation_engine=rec_engine)

    logger.info("电商 Agent 已启动！")
    logger.info("输入 'quit' 或 'exit' 退出")
    logger.info("使用方式: <user_id> <你的问题>")
    logger.info("示例: 100 帮我推荐一些商品\n")

    while True:
        try:
            user_input = input("你: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("退出对话")
                break

            if not user_input:
                continue

            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("💡 请输入: <user_id> <问题>")
                print("示例: 100 帮我推荐一些商品")
                continue

            try:
                user_id = int(parts[0])
                message = parts[1]
            except ValueError:
                print("💡 user_id 必须是数字")
                continue

            response, item_ids = agent.chat(user_id, message)

            print(f"\n小荐: {response}\n")
            print(f"📦 推荐商品ID: {', '.join(map(str, item_ids))}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n退出对话")
            break
        except Exception as e:
            logger.error(f"对话出错: {e}")
            print(f"出错了: {e}")


def run_full_pipeline(sample_size=None):
    """运行完整流程"""
    Config.create_output_dirs()

    graph_builder = run_data_pipeline(sample_size=sample_size)
    if graph_builder is None:
        return

    model, user_embeddings, item_embeddings = run_training(graph_builder=graph_builder, sample_size=sample_size)

    qdrant_client = run_vector_index(item_embeddings=item_embeddings)

    logger.info("=" * 60)
    logger.info("完整流程执行完成！")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='电商推荐 Agent 项目')
    parser.add_argument('mode', nargs='?', default='chat',
                        choices=['data', 'train', 'index', 'chat', 'pipeline'],
                        help='运行模式: data(数据处理), train(训练), index(入库), chat(对话), pipeline(完整流程)')
    parser.add_argument('--sample', type=int, default=None,
                        help='数据采样数量（用于快速测试）')
    parser.add_argument('--output', type=str, default='./output',
                        help='输出目录')

    args = parser.parse_args()

    Config.create_output_dirs()

    logger.info(f"运行模式: {args.mode}")
    if args.sample:
        logger.info(f"采样数量: {args.sample}")

    if args.mode == 'data':
        run_data_pipeline(sample_size=args.sample)

    elif args.mode == 'train':
        run_training(sample_size=args.sample)

    elif args.mode == 'index':
        run_vector_index()

    elif args.mode == 'chat':
        run_chat_mode()

    elif args.mode == 'pipeline':
        run_full_pipeline(sample_size=args.sample)


if __name__ == '__main__':
    main()