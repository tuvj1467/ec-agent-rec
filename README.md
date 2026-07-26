# 电商推荐 Agent 项目

基于阿里云天池 UserBehavior 数据集，构建以 LightGCN 图推荐模型为核心、Qdrant 向量数据库为召回引擎、LLM 为导购交互层的电商推荐 Agent 系统。

## 项目架构

```
天池 UserBehavior 数据集
         ↓
[数据层] 离线清洗、构造交互图
         ↓
[模型层] LightGCN 图推荐模型训练
         ↓
[向量层] 输出 user_embedding、item_embedding
         ↓
[召回层] Qdrant 向量库（物品向量索引）
         ↓
[缓存层] Redis 缓存高频召回结果
         ↓
[Agent层] LLM 电商导购对话
         ↓
[输出层] 推荐商品 item_id 列表 + 导购回复
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 数据处理 | pandas, numpy, scipy | 数据清洗、稀疏矩阵、图构造 |
| 推荐模型 | PyTorch + LightGCN | 图卷积神经网络推荐 |
| 向量数据库 | Qdrant | 稠密向量检索（支持本地内存模式） |
| 缓存 | Redis | 高频召回结果缓存（支持本地内存模式） |
| Agent | Mock LLM / 真实 LLM | 电商导购对话生成 |

## 项目结构

```
ec-agent-rec/
├── doc/                           # 文档目录
│   └── 01-项目规划.md             # 项目规划文档
│
├── data_processing/               # 数据处理模块
│   ├── __init__.py
│   ├── data_cleaner.py            # 数据清洗器
│   ├── graph_builder.py           # 图构造器
│   └── dataset.py                 # 数据集封装
│
├── model/                         # 模型训练模块
│   ├── __init__.py
│   ├── lightgcn.py                # LightGCN 模型定义
│   ├── trainer.py                 # 训练器
│   ├── evaluator.py               # 评估器
│   └── embedding_exporter.py      # 向量导出器
│
├── vector_store/                  # 向量存储模块
│   ├── __init__.py
│   ├── qdrant_client.py           # Qdrant 客户端封装
│   ├── indexer.py                 # 向量索引器
│   └── retriever.py               # 向量检索器
│
├── cache/                         # 缓存模块
│   ├── __init__.py
│   ├── redis_client.py            # Redis 客户端封装
│   └── recall_cache.py            # 召回缓存管理器
│
├── agent/                         # Agent 模块
│   ├── __init__.py
│   ├── ecommerce_agent.py         # 电商 Agent 主类
│   ├── prompt_templates.py        # Prompt 模板
│   ├── recommendation_engine.py   # 推荐引擎
│   └── response_generator.py      # 回复生成器
│
├── config/                        # 配置模块
│   ├── __init__.py
│   └── config.py                  # 全局配置
│
├── output/                        # 输出目录（自动生成）
│   ├── data/                      # 清洗后数据
│   ├── graph/                     # 图数据
│   ├── embeddings/                # 向量文件
│   ├── checkpoints/               # 模型检查点
│   └── logs/                      # 训练日志
│
├── main_rec.py                    # 主入口脚本
├── test_flow.py                   # 流程测试脚本
├── requirements.txt               # 依赖列表
└── README.md                      # 项目说明
```

## 快速开始

### 环境要求

- Python 3.8+
- 依赖库见 requirements.txt

### 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：`torch`、`qdrant-client`、`redis` 为可选依赖，未安装时系统会自动降级为本地内存模式。

### 运行流程

#### 1. 数据处理

```bash
# 使用全量数据
python main_rec.py data

# 使用采样数据（快速测试）
python main_rec.py data --sample 100000
```

输出：
- `output/data/clean_interactions.csv` - 清洗后的数据
- `output/graph/adj_matrix.npz` - 邻接矩阵
- `output/graph/user_mapping.json` - 用户ID映射
- `output/graph/item_mapping.json` - 商品ID映射

#### 2. 模型训练

```bash
python main_rec.py train
```

输出：
- `output/embeddings/user_embedding.npy` - 用户向量
- `output/embeddings/item_embedding.npy` - 商品向量
- `output/checkpoints/best_model.pt` - 最优模型检查点

> 注意：需要安装 PyTorch 才能运行训练。

#### 3. 向量入库

```bash
python main_rec.py index
```

将商品向量批量写入 Qdrant 向量数据库。

#### 4. 启动 Agent 对话

```bash
python main_rec.py chat
```

进入交互式对话模式，输入格式：
```
<user_id> <你的问题>
```

示例：
```
100 帮我推荐一些商品
100 介绍一下商品
100 有什么数码产品推荐
```

#### 5. 完整流程

```bash
python main_rec.py pipeline --sample 100000
```

一键运行：数据处理 → 模型训练 → 向量入库

### 快速测试

不依赖 torch、qdrant、redis 也可以测试 Agent 流程：

```bash
python test_flow.py
```

该脚本使用随机向量和本地内存模式测试完整流程。

## 核心功能

### 数据处理
- 数据清洗：去重、过滤异常值、时间范围过滤
- 图构造：用户-商品二部图、邻接矩阵、LightGCN 归一化矩阵
- 数据集划分：训练/验证/测试集

### LightGCN 推荐模型
- 多层图卷积传播
- BPR 损失函数
- Recall@K、NDCG@K、Precision@K 评估指标
- 早停机制、模型检查点

### 向量检索
- Qdrant 向量数据库集成
- 支持本地内存模式（无需 Docker）
- 余弦相似度检索
- 批量入库

### 缓存优化
- Redis 缓存召回结果
- 支持本地内存模式
- TTL 过期机制
- 缓存命中率统计

### 电商 Agent
- 意图识别（欢迎/推荐/详情/品类）
- Mock LLM 模式（无需 API Key）
- 自然语言导购回复
- 商品 ID 列表输出
- 对话历史管理

## 数据说明

- **数据集来源**：阿里云天池 UserBehavior 数据集
- **数据格式**：用户ID, 商品ID, 分类ID, 行为类型, 时间戳
- **行为类型**：pv（浏览）、buy（购买）、cart（加购）、fav（收藏）
- **时间范围**：2017-11-25 ~ 2017-12-03
- **数据量**：约 1 亿条记录

## 配置说明

所有配置在 `config/config.py` 中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| embedding_dim | 64 | 向量维度 |
| num_layers | 3 | GCN 层数 |
| epochs | 100 | 训练轮数 |
| batch_size | 1024 | 批次大小 |
| learning_rate | 0.001 | 学习率 |
| top_k | 20 | 推荐数量 |
| ttl | 3600 | 缓存过期时间（秒） |

## 项目亮点

1. **完整的推荐系统链路**：从数据处理到 Agent 对话，端到端实现
2. **图神经网络推荐**：LightGCN 模型，业界主流推荐算法
3. **向量检索召回**：Qdrant 向量数据库，高效相似性搜索
4. **缓存优化**：Redis 缓存高频召回，提升性能
5. **LLM 导购交互**：自然语言对话，智能推荐
6. **降级设计**：所有外部依赖（torch/qdrant/redis）均支持本地内存模式
7. **模块化设计**：各模块独立，可单独运行和测试
8. **简历友好**：技术栈全面，覆盖数据工程、算法、工程化、大模型应用