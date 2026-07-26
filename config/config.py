"""配置模块"""

import os

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("已加载 .env 文件")
except ImportError:
    pass

def get_env(key, default=None):
    """获取环境变量，支持 .env 文件"""
    return os.getenv(key, default)



class Config:
    """全局配置"""

    # 数据配置
    DATA_CONFIG = {
        'input_path': None,  # None 表示自动查找
        'output_dir': './output/data',
        'sample_size': None,  # None 表示全量
        'chunk_size': 1000000,
        'valid_start_date': '2017-11-25',
        'valid_end_date': '2017-12-04'
    }

    # 图构造配置
    GRAPH_CONFIG = {
        'output_dir': './output/graph',
        'use_buy_only': True
    }

    # 模型配置
    MODEL_CONFIG = {
        'embedding_dim': 64,
        'num_layers': 3,
        'dropout': 0.0
    }

    # 训练配置
    TRAIN_CONFIG = {
        'epochs': 100,
        'batch_size': 1024,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'num_negatives': 1,
        'eval_every': 10,
        'patience': 10,
        'save_dir': './output/checkpoints',
        'log_dir': './output/logs'
    }

    # 数据划分配置
    DATASET_CONFIG = {
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'seed': 42
    }

    # 评估配置
    EVAL_CONFIG = {
        'k_list': [20, 50]
    }

    # 向量导出配置
    EMBEDDING_CONFIG = {
        'output_dir': './output/embeddings'
    }

    # Qdrant 配置
    QDRANT_CONFIG = {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'ecommerce_items',
        'embedding_dim': 64,
        'distance': 'Cosine',
        'batch_size': 1000,
        'top_k': 20
    }

    # Redis 配置
    REDIS_CONFIG = {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'password': None,
        'ttl': 3600,
        'prefix': 'recall'
    }

    # Agent 配置
    AGENT_CONFIG = {
        'top_k': 20,
        'use_cache': True,
        'use_mock_llm': True
    }

    # LLM 配置
    LLM_CONFIG = {
        'provider': get_env('LLM_PROVIDER', 'mock'),  # mock / openai / qwen / zhipu / deepseek / ollama
        'timeout': int(get_env('LLM_TIMEOUT', 60)),
        'max_retries': int(get_env('LLM_MAX_RETRIES', 3)),
        'retry_delay': int(get_env('LLM_RETRY_DELAY', 1)),
        'model_name': {
            'openai': get_env('OPENAI_MODEL', 'gpt-3.5-turbo'),
            'qwen': get_env('QWEN_MODEL', 'qwen-turbo'),
            'zhipu': get_env('ZHIPU_MODEL', 'glm-4'),
            'deepseek': get_env('DEEPSEEK_MODEL', 'deepseek-chat'),
            'ollama': 'qwen2.5:7b'
        },
        'api_key': {
            'openai': get_env('OPENAI_API_KEY', ''),
            'qwen': get_env('DASHSCOPE_API_KEY', ''),
            'zhipu': get_env('ZHIPU_API_KEY', ''),
            'deepseek': get_env('DEEPSEEK_API_KEY', '')
        },
        'base_url': {
            'openai': get_env('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'qwen': get_env('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            'zhipu': get_env('ZHIPU_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4'),
            'deepseek': get_env('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
            'ollama': 'http://localhost:11434/v1'
        },
        'generation': {
            'temperature': float(get_env('LLM_TEMPERATURE', 0.7)),
            'max_tokens': int(get_env('LLM_MAX_TOKENS', 1024)),
            'top_p': float(get_env('LLM_TOP_P', 0.9))
        }
    }

    @classmethod
    def get_output_dirs(cls):
        """获取所有输出目录"""
        dirs = [
            cls.DATA_CONFIG['output_dir'],
            cls.GRAPH_CONFIG['output_dir'],
            cls.TRAIN_CONFIG['save_dir'],
            cls.TRAIN_CONFIG['log_dir'],
            cls.EMBEDDING_CONFIG['output_dir']
        ]
        return dirs

    @classmethod
    def create_output_dirs(cls):
        """创建所有输出目录"""
        for dir_path in cls.get_output_dirs():
            os.makedirs(dir_path, exist_ok=True)