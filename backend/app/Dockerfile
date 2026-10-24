FROM python:3.11.7-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# 设置Python模块搜索路径
ENV PYTHONPATH=/
ENV QWEN_TOKENIZER_MODEL=/opt/qwen-tokenizer


# 配置pip使用清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 复制代码到容器
COPY . /app/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 镜像构建阶段只下载并校验 Qwen tokenizer，不下载模型权重；运行时完全离线计数。
RUN python scripts/download_qwen_tokenizer.py --output "$QWEN_TOKENIZER_MODEL" \
    && python -c "from transformers import AutoTokenizer; import os; AutoTokenizer.from_pretrained(os.environ['QWEN_TOKENIZER_MODEL'], use_fast=True, local_files_only=True)"

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]
