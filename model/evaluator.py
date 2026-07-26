import torch
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """评估器 - 评估推荐模型性能"""

    def __init__(self, test_user_items, train_user_items, num_items, k_list=[20, 50]):
        """
        初始化评估器

        :param test_user_items: 测试集用户-商品交互 {user_idx: set(item_idx)}
        :param train_user_items: 训练集用户-商品交互 {user_idx: set(item_idx)}
        :param num_items: 商品总数
        :param k_list: 评估的 K 值列表
        """
        self.test_user_items = test_user_items
        self.train_user_items = train_user_items
        self.num_items = num_items
        self.k_list = k_list

        # 过滤掉测试集中无交互的用户
        self.test_users = [u for u, items in test_user_items.items() if len(items) > 0]

        logger.info(f"评估器初始化: 测试用户 {len(self.test_users)} 人, K={k_list}")

    def evaluate(self, model):
        """
        评估模型

        :param model: LightGCN 模型
        :return: 评估指标字典
        """
        model.eval()
        device = next(model.parameters()).device

        all_metrics = defaultdict(list)

        with torch.no_grad():
            user_embeddings = model.get_user_embeddings()
            item_embeddings = model.get_item_embeddings()

            batch_size = 1024

            for i in range(0, len(self.test_users), batch_size):
                batch_users = self.test_users[i:i + batch_size]

                user_indices = torch.LongTensor(batch_users).to(device)
                batch_user_emb = user_embeddings[user_indices]

                # 计算所有商品的评分
                scores = torch.matmul(batch_user_emb, item_embeddings.t())

                # 去除训练集中已有的商品
                for j, user_idx in enumerate(batch_users):
                    if user_idx in self.train_user_items:
                        train_items = list(self.train_user_items[user_idx])
                        if train_items:
                            scores[j, train_items] = -float('inf')

                # 获取 Top-K 商品
                topk_scores, topk_indices = torch.topk(scores, max(self.k_list), dim=1)
                topk_indices = topk_indices.cpu().numpy()

                # 计算每个用户的指标
                for j, user_idx in enumerate(batch_users):
                    test_items = self.test_user_items[user_idx]
                    if len(test_items) == 0:
                        continue

                    ranking = topk_indices[j]

                    for k in self.k_list:
                        topk = ranking[:k]

                        hits = len(set(topk) & test_items)

                        recall = hits / len(test_items)
                        precision = hits / k
                        ndcg = self._calculate_ndcg(topk, test_items)

                        all_metrics[f'recall@{k}'].append(recall)
                        all_metrics[f'precision@{k}'].append(precision)
                        all_metrics[f'ndcg@{k}'].append(ndcg)

        # 计算平均指标
        avg_metrics = {}
        for k, v in all_metrics.items():
            avg_metrics[k] = float(np.mean(v))

        return avg_metrics

    def _calculate_ndcg(self, ranking, ground_truth):
        """
        计算 NDCG

        :param ranking: 排序后的商品列表
        :param ground_truth: 真实交互商品集合
        :return: NDCG 值
        """
        dcg = 0.0
        for i, item in enumerate(ranking):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)

        # 计算 IDCG（理想 DCG）
        idcg = 0.0
        for i in range(min(len(ground_truth), len(ranking))):
            idcg += 1.0 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_simple(self, model, num_sample_users=1000):
        """
        简化版评估（采样用户，速度更快）

        :param model: LightGCN 模型
        :param num_sample_users: 采样用户数
        :return: 评估指标字典
        """
        if len(self.test_users) <= num_sample_users:
            sample_users = self.test_users
        else:
            sample_users = np.random.choice(self.test_users, num_sample_users, replace=False)

        original_test_users = self.test_users
        self.test_users = list(sample_users)
        metrics = self.evaluate(model)
        self.test_users = original_test_users

        return metrics