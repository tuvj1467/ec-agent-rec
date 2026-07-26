"""快速测试脚本 - 验证 Qdrant + Agent 流程"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_store.qdrant_client import QdrantClientWrapper
from vector_store.indexer import VectorIndexer
from vector_store.retriever import VectorRetriever
from cache.redis_client import RedisClientWrapper
from cache.recall_cache import RecallCache
from agent.recommendation_engine import RecommendationEngine
from agent.ecommerce_agent import EcommerceAgent

print("=" * 60)
print("测试1: Qdrant 向量库（本地模式）")
print("=" * 60)

num_items = 1000
embedding_dim = 64

qdrant = QdrantClientWrapper(
    host='localhost',
    port=6333,
    collection_name='test_items',
    embedding_dim=embedding_dim,
    distance='Cosine'
)
print(f"Qdrant 本地模式: {qdrant.is_local_mode()}")

item_embeddings = np.random.randn(num_items, embedding_dim).astype(np.float32)
item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

indexer = VectorIndexer(qdrant_client=qdrant)
indexer.index_item_embeddings(item_embeddings, batch_size=200)

info = qdrant.get_collection_info()
print(f"集合信息: {info}")

print("\n" + "=" * 60)
print("测试2: 向量检索")
print("=" * 60)

retriever = VectorRetriever(qdrant_client=qdrant, top_k=10)

user_embedding = np.random.randn(embedding_dim).astype(np.float32)
results = retriever.retrieve(user_embedding, top_k=5)
print(f"检索结果数量: {len(results)}")
for i, r in enumerate(results[:3]):
    print(f"  Top{i+1}: item_idx={r['item_idx']}, score={r['score']:.4f}")

print("\n" + "=" * 60)
print("测试3: Redis 缓存（本地模式）")
print("=" * 60)

redis_client = RedisClientWrapper(host='localhost', port=6379, db=0)
print(f"Redis 本地模式: {redis_client.is_local_mode()}")

recall_cache = RecallCache(redis_client, ttl=3600)

recall_cache.set(100, results, top_k=10)
cached = recall_cache.get(100, top_k=10)
print(f"缓存命中: {cached is not None}")
print(f"缓存统计: {recall_cache.get_stats()}")

print("\n" + "=" * 60)
print("测试4: 推荐引擎")
print("=" * 60)

user_embeddings = np.random.randn(500, embedding_dim).astype(np.float32)
user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)

rec_engine = RecommendationEngine(
    vector_retriever=retriever,
    user_embeddings=user_embeddings,
    recall_cache=recall_cache,
    top_k=10
)

recs1 = rec_engine.recommend(42, top_k=5)
print(f"推荐结果数量: {len(recs1)}")
print(f"推荐商品ID: {[r['original_item_id'] for r in recs1[:3]]}")

recs2 = rec_engine.recommend(42, top_k=5)
print(f"第二次推荐（缓存命中）: {rec_engine.get_stats()}")

print("\n" + "=" * 60)
print("测试5: 电商 Agent 对话")
print("=" * 60)

agent = EcommerceAgent(recommendation_engine=rec_engine)

response, item_ids = agent.chat(42, "你好，帮我推荐一些商品")
print(f"\nAgent回复:\n{response}")
print(f"\n推荐商品ID: {item_ids}")

response2, item_ids2 = agent.chat(42, "介绍一下商品")
print(f"\nAgent回复2:\n{response2}")
print(f"\n推荐商品ID: {item_ids2}")

print("\n" + "=" * 60)
print("所有测试通过！✅")
print("=" * 60)