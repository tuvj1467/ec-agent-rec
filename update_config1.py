import os

with open("config/config.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Insert dotenv code after line 3 (import os)
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if i == 2:  # after "import os"
        new_lines.append("\n")
        new_lines.append("# 尝试加载 .env 文件\n")
        new_lines.append("try:\n")
        new_lines.append("    from dotenv import load_dotenv\n")
        new_lines.append("    load_dotenv()\n")
        new_lines.append("    print(\"已加载 .env 文件\")\n")
        new_lines.append("except ImportError:\n")
        new_lines.append("    pass\n")
        new_lines.append("\n")
        new_lines.append("def get_env(key, default=None):\n")
        new_lines.append("    \"\"\"获取环境变量，支持 .env 文件\"\"\"\n")
        new_lines.append("    return os.getenv(key, default)\n")
        new_lines.append("\n")

with open("config/config.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Step 1: Added dotenv support")
