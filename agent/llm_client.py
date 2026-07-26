"""LLM 客户端模块：封装多提供商的大模型调用"""

import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMClient:
    """LLM 客户端基类"""

    def chat(self, system_prompt, user_prompt):
        """
        对话接口

        :param system_prompt: 系统提示词
        :param user_prompt: 用户提示词
        :return: 模型回复文本
        """
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Mock LLM 客户端：用于无 LLM 时的演示"""

    def chat(self, system_prompt, user_prompt):
        logger.info("[MockLLM] 收到请求，使用模板回复（实际环境中应替换为真实LLM）")
        return "（Mock 模式）这是一个模拟的导购回复，接入真实大模型后会生成自然语言推荐内容。"


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI 兼容接口客户端（支持 OpenAI / 通义千问 / 智谱 / DeepSeek / Ollama）"""

    def __init__(self, api_key, base_url, model_name, temperature=0.7,
                 max_tokens=1024, top_p=0.9, timeout=60, max_retries=3, retry_delay=1):
        """
        初始化 OpenAI 兼容客户端

        :param api_key: API Key
        :param base_url: API 基础地址
        :param model_name: 模型名称
        :param temperature: 温度系数
        :param max_tokens: 最大生成 token 数
        :param top_p: 核采样参数
        :param timeout: 请求超时（秒）
        :param max_retries: 最大重试次数
        :param retry_delay: 重试延迟（秒）
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None

        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"LLM 客户端初始化成功: model={self.model_name}")
        except ImportError:
            logger.warning("openai 库未安装，LLM 功能不可用。请执行: pip install openai")
            self.client = None
        except Exception as e:
            logger.warning(f"LLM 客户端初始化失败: {e}")
            self.client = None

    def chat(self, system_prompt, user_prompt):
        if self.client is None:
            raise RuntimeError("LLM 客户端未初始化，检查 API Key 和网络设置")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )
                content = response.choices[0].message.content
                logger.info(f"LLM 调用成功，生成 {len(content)} 字符")
                return content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"LLM 调用失败（第 {attempt + 1} 次）: {e}，{self.retry_delay}秒后重试")
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"LLM 调用失败（已重试 {self.max_retries} 次）: {e}")

        raise RuntimeError(f"LLM 调用失败: {last_error}")


def create_llm_client(llm_config):
    """
    根据配置创建 LLM 客户端

    :param llm_config: LLM 配置字典
    :return: LLM 客户端实例
    """
    provider = llm_config.get('provider', 'mock')

    if provider == 'mock' or not llm_config.get('api_key', {}).get(provider):
        logger.info(f"使用 Mock LLM（provider={provider}）")
        return MockLLMClient()

    model_name = llm_config.get('model_name', {}).get(provider, '')
    api_key = llm_config.get('api_key', {}).get(provider, '')
    base_url = llm_config.get('base_url', {}).get(provider, '')
    gen_cfg = llm_config.get('generation', {})

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=gen_cfg.get('temperature', 0.7),
        max_tokens=gen_cfg.get('max_tokens', 1024),
        top_p=gen_cfg.get('top_p', 0.9),
        timeout=llm_config.get('timeout', 60),
        max_retries=llm_config.get('max_retries', 3),
        retry_delay=llm_config.get('retry_delay', 1)
    )

    return client
