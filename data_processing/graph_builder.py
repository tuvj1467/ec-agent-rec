import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """图构造器 - 构建用户-商品二部图和邻接矩阵"""

    def __init__(self, output_dir='./output/graph'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.adj_matrix = None
        self.num_users = 0
        self.num_items = 0

    def build_graph(self, interactions_df, use_buy_only=True):
        """
        构建用户-商品二部图

        :param interactions_df: 交互数据框 (user_id, item_id, weight可选)
        :param use_buy_only: 是否仅使用购买行为
        :return: 邻接矩阵（稀疏）
        """
        logger.info("开始构建用户-商品二部图")

        if use_buy_only and 'behavior_type' in interactions_df.columns:
            interactions_df = interactions_df[interactions_df['behavior_type'] == 'buy']

        self._build_mappings(interactions_df)
        self.adj_matrix = self._build_adjacency_matrix(interactions_df)

        logger.info(f"图构建完成: {self.num_users} 个用户, {self.num_items} 个商品, "
                    f"{self.adj_matrix.nnz} 条边")
        return self.adj_matrix

    def _build_mappings(self, interactions_df):
        """构建用户和商品的ID映射（原始ID → 连续索引）"""
        user_ids = sorted(interactions_df['user_id'].unique())
        item_ids = sorted(interactions_df['item_id'].unique())

        self.user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
        self.reverse_user_mapping = {idx: uid for uid, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: iid for iid, idx in self.item_mapping.items()}

        self.num_users = len(self.user_mapping)
        self.num_items = len(self.item_mapping)

        logger.info(f"ID映射构建完成: {self.num_users} 用户, {self.num_items} 商品")

    def _build_adjacency_matrix(self, interactions_df):
        """
        构建邻接矩阵（用户-商品二部图）

        矩阵形状: (num_users, num_items)
        值: 1表示有交互，0表示无交互
        """
        row = interactions_df['user_id'].map(self.user_mapping).values
        col = interactions_df['item_id'].map(self.item_mapping).values

        if 'weight' in interactions_df.columns:
            data = interactions_df['weight'].values
        else:
            data = np.ones(len(interactions_df), dtype=np.float32)

        adj_matrix = sp.csr_matrix(
            (data, (row, col)),
            shape=(self.num_users, self.num_items)
        )
        return adj_matrix

    def build_lightgcn_adj_matrix(self):
        """
        构建 LightGCN 所需的归一化邻接矩阵

        LightGCN 使用的是对称归一化的拉普拉斯矩阵:
        A_hat = D^(-1/2) * A * D^(-1/2)

        其中 A 是 (N+M) x (N+M) 的大图邻接矩阵:
        [0, R]
        [R^T, 0]

        :return: 归一化后的大图邻接矩阵 (num_users + num_items, num_users + num_items)
        """
        if self.adj_matrix is None:
            logger.error("请先调用 build_graph() 构建邻接矩阵")
            return None

        logger.info("构建 LightGCN 归一化邻接矩阵")

        R = self.adj_matrix.astype(np.float32)

        num_nodes = self.num_users + self.num_items

        # 构建大图邻接矩阵 A = [[0, R], [R^T, 0]]
        adj_top = sp.hstack([sp.csr_matrix((self.num_users, self.num_users)), R])
        adj_bottom = sp.hstack([R.T, sp.csr_matrix((self.num_items, self.num_items))])
        adj_matrix = sp.vstack([adj_top, adj_bottom])

        # 计算度矩阵 D
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()

        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # L = D^(-1/2) * A * D^(-1/2)
        norm_adj = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)
        norm_adj = norm_adj.tocsr().astype(np.float32)

        logger.info(f"LightGCN 邻接矩阵构建完成: {num_nodes} 个节点, {norm_adj.nnz} 条边")
        return norm_adj

    def get_user_items_dict(self):
        """
        获取每个用户的交互商品集合

        :return: dict {user_idx: set(item_idx)}
        """
        if self.adj_matrix is None:
            logger.error("请先调用 build_graph() 构建邻接矩阵")
            return None

        user_items = {}
        for user_idx in range(self.num_users):
            items = set(self.adj_matrix[user_idx].indices)
            user_items[user_idx] = items

        return user_items

    def save_mappings(self):
        """保存ID映射"""
        user_map_path = os.path.join(self.output_dir, 'user_mapping.json')
        item_map_path = os.path.join(self.output_dir, 'item_mapping.json')

        with open(user_map_path, 'w') as f:
            json.dump({str(k): v for k, v in self.user_mapping.items()}, f)

        with open(item_map_path, 'w') as f:
            json.dump({str(k): v for k, v in self.item_mapping.items()}, f)

        logger.info(f"ID映射已保存: {user_map_path}, {item_map_path}")

    def load_mappings(self):
        """加载ID映射"""
        user_map_path = os.path.join(self.output_dir, 'user_mapping.json')
        item_map_path = os.path.join(self.output_dir, 'item_mapping.json')

        if not os.path.exists(user_map_path) or not os.path.exists(item_map_path):
            logger.warning("映射文件不存在")
            return False

        with open(user_map_path, 'r') as f:
            self.user_mapping = {int(k): v for k, v in json.load(f).items()}

        with open(item_map_path, 'r') as f:
            self.item_mapping = {int(k): v for k, v in json.load(f).items()}

        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        self.num_users = len(self.user_mapping)
        self.num_items = len(self.item_mapping)

        logger.info(f"ID映射已加载: {self.num_users} 用户, {self.num_items} 商品")
        return True

    def save_adj_matrix(self, filename='adj_matrix.npz'):
        """保存邻接矩阵"""
        if self.adj_matrix is None:
            logger.error("邻接矩阵未构建")
            return False

        output_path = os.path.join(self.output_dir, filename)
        sp.save_npz(output_path, self.adj_matrix)
        logger.info(f"邻接矩阵已保存: {output_path}")
        return True

    def load_adj_matrix(self, filename='adj_matrix.npz'):
        """加载邻接矩阵"""
        input_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(input_path):
            logger.warning(f"邻接矩阵文件不存在: {input_path}")
            return False

        self.adj_matrix = sp.load_npz(input_path)
        self.num_users = self.adj_matrix.shape[0]
        self.num_items = self.adj_matrix.shape[1]
        logger.info(f"邻接矩阵已加载: {self.num_users} x {self.num_items}")
        return True

    def save_all(self):
        """保存所有图数据"""
        self.save_mappings()
        self.save_adj_matrix()
        return True

    def load_all(self):
        """加载所有图数据"""
        self.load_mappings()
        self.load_adj_matrix()
        return True

    def get_graph_info(self):
        """获取图信息"""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_edges': self.adj_matrix.nnz if self.adj_matrix is not None else 0,
            'density': (self.adj_matrix.nnz / (self.num_users * self.num_items))
            if self.adj_matrix is not None else 0,
            'avg_degree_user': (self.adj_matrix.nnz / self.num_users)
            if self.adj_matrix is not None else 0,
            'avg_degree_item': (self.adj_matrix.nnz / self.num_items)
            if self.adj_matrix is not None else 0
        }