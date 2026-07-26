import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """LightGCN 模型 - 图卷积神经网络推荐模型"""

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, dropout=0.0):
        """
        初始化 LightGCN 模型

        :param num_users: 用户数量
        :param num_items: 商品数量
        :param embedding_dim: 嵌入维度
        :param num_layers: GCN 层数
        :param dropout: dropout 比率
        """
        super(LightGCN, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_embeddings()

        self.norm_adj_matrix = None

        logger.info(f"LightGCN 初始化: {num_users} 用户, {num_items} 商品, "
                    f"embedding_dim={embedding_dim}, num_layers={num_layers}")

    def _init_embeddings(self):
        """初始化嵌入权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def set_adj_matrix(self, norm_adj_matrix):
        """
        设置归一化的邻接矩阵

        :param norm_adj_matrix: 归一化邻接矩阵 (scipy sparse 或 torch sparse)
        """
        if isinstance(norm_adj_matrix, sp.csr_matrix):
            self.norm_adj_matrix = self._convert_sparse_matrix_to_torch(norm_adj_matrix)
        else:
            self.norm_adj_matrix = norm_adj_matrix

        device = next(self.parameters()).device
        self.norm_adj_matrix = self.norm_adj_matrix.to(device)

        logger.info("邻接矩阵已设置")

    def _convert_sparse_matrix_to_torch(self, sparse_matrix):
        """
        将 scipy 稀疏矩阵转换为 torch 稀疏张量

        :param sparse_matrix: scipy sparse matrix
        :return: torch sparse tensor
        """
        sparse_matrix = sparse_matrix.tocoo()
        indices = torch.LongTensor(np.vstack([sparse_matrix.row, sparse_matrix.col]))
        values = torch.FloatTensor(sparse_matrix.data)
        shape = torch.Size(sparse_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, user_indices, item_indices):
        """
        前向传播，计算用户和商品的嵌入

        :param user_indices: 用户索引
        :param item_indices: 商品索引
        :return: 用户嵌入, 商品嵌入
        """
        all_embeddings = self._compute_all_embeddings()
        user_emb = all_embeddings[user_indices]
        item_emb = all_embeddings[self.num_users + item_indices]
        return user_emb, item_emb

    def _compute_all_embeddings(self):
        """
        计算所有节点的嵌入（通过 GCN 传播）

        :return: 所有节点的嵌入 (num_users + num_items, embedding_dim)
        """
        if self.norm_adj_matrix is None:
            raise ValueError("请先调用 set_adj_matrix() 设置邻接矩阵")

        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        device = all_emb.device
        if self.norm_adj_matrix.device != device:
            self.norm_adj_matrix = self.norm_adj_matrix.to(device)

        embeddings_list = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)

            if self.dropout > 0 and self.training:
                all_emb = nn.functional.dropout(all_emb, p=self.dropout)

            embeddings_list.append(all_emb)

        # 层聚合：各层嵌入的均值
        all_emb = torch.stack(embeddings_list, dim=0)
        all_emb = torch.mean(all_emb, dim=0)

        return all_emb

    def get_user_embeddings(self):
        """获取所有用户的最终嵌入"""
        all_embeddings = self._compute_all_embeddings()
        return all_embeddings[:self.num_users]

    def get_item_embeddings(self):
        """获取所有商品的最终嵌入"""
        all_embeddings = self._compute_all_embeddings()
        return all_embeddings[self.num_users:]

    def bpr_loss(self, user_indices, pos_item_indices, neg_item_indices):
        """
        计算 BPR 损失

        :param user_indices: 用户索引 (batch_size, 1)
        :param pos_item_indices: 正样本商品索引 (batch_size, 1)
        :param neg_item_indices: 负样本商品索引 (batch_size, num_negatives)
        :return: BPR 损失值
        """
        user_indices = user_indices.squeeze()
        pos_item_indices = pos_item_indices.squeeze()

        user_emb, pos_item_emb = self.forward(user_indices, pos_item_indices)

        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)

        neg_scores_list = []
        for i in range(neg_item_indices.size(1)):
            neg_items = neg_item_indices[:, i].squeeze()
            _, neg_item_emb = self.forward(user_indices, neg_items)
            neg_score = torch.sum(user_emb * neg_item_emb, dim=1)
            neg_scores_list.append(neg_score)

        neg_scores = torch.stack(neg_scores_list, dim=1)

        # BPR loss: -ln(sigmoid(pos_score - neg_score))
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8)
        loss = torch.mean(loss)

        return loss

    def predict(self, user_indices, item_indices):
        """
        预测用户对商品的评分

        :param user_indices: 用户索引
        :param item_indices: 商品索引
        :return: 预测分数
        """
        user_emb, item_emb = self.forward(user_indices, item_indices)
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

    def get_embedding_dim(self):
        """获取嵌入维度"""
        return self.embedding_dim