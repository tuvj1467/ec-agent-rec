import re
with open("config/config.py", "r", encoding="utf-8") as f:
    c = f.read()

c = c.replace("'provider': 'mock'", "'provider': get_env('LLM_PROVIDER', 'mock')")
c = c.replace("'timeout': 60,", "'timeout': int(get_env('LLM_TIMEOUT', 60)),")
c = c.replace("'max_retries': 3,", "'max_retries': int(get_env('LLM_MAX_RETRIES', 3)),")
c = c.replace("'retry_delay': 1,", "'retry_delay': int(get_env('LLM_RETRY_DELAY', 1)),")

c = c.replace("'openai': 'gpt-3.5-turbo',", "'openai': get_env('OPENAI_MODEL', 'gpt-3.5-turbo'),")
c = c.replace("'qwen': 'qwen-turbo',", "'qwen': get_env('QWEN_MODEL', 'qwen-turbo'),")
c = c.replace("'zhipu': 'glm-4',", "'zhipu': get_env('ZHIPU_MODEL', 'glm-4'),")
c = c.replace("'deepseek': 'deepseek-chat',", "'deepseek': get_env('DEEPSEEK_MODEL', 'deepseek-chat'),")
c = c.replace("'ollama': 'qwen2.5:7b',", "'ollama': get_env('OLLAMA_MODEL', 'qwen2.5:7b'),")

c = c.replace("os.getenv('OPENAI_API_KEY', '')", "get_env('OPENAI_API_KEY', '')")
c = c.replace("os.getenv('DASHSCOPE_API_KEY', '')", "get_env('DASHSCOPE_API_KEY', '')")
c = c.replace("os.getenv('ZHIPU_API_KEY', '')", "get_env('ZHIPU_API_KEY', '')")
c = c.replace("os.getenv('DEEPSEEK_API_KEY', '')", "get_env('DEEPSEEK_API_KEY', '')")

c = c.replace("os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')", "get_env('OPENAI_BASE_URL', 'https://api.openai.com/v1')")
c = c.replace("'qwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1',", "'qwen': get_env('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),")
c = c.replace("'zhipu': 'https://open.bigmodel.cn/api/paas/v4',", "'zhipu': get_env('ZHIPU_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4'),")
c = c.replace("'deepseek': 'https://api.deepseek.com/v1',", "'deepseek': get_env('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),")
c = c.replace("'ollama': 'http://localhost:11434/v1',", "'ollama': get_env('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),")

c = c.replace("'temperature': 0.7,", "'temperature': float(get_env('LLM_TEMPERATURE', 0.7)),")
c = c.replace("'max_tokens': 1024,", "'max_tokens': int(get_env('LLM_MAX_TOKENS', 1024)),")
c = c.replace("'top_p': 0.9", "'top_p': float(get_env('LLM_TOP_P', 0.9))")

with open("config/config.py", "w", encoding="utf-8") as f:
    f.write(c)

print("done")
