FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装Python和pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 创建模型目录
RUN mkdir -p /app/models

# 复制应用文件
COPY embedding.py requirements.txt /app/

# 安装依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV MODEL_PATH=/app/models/bge-m3
ENV DEVICE=cuda
ENV PORT=6008

# 暴露端口
EXPOSE 6008

# 创建启动脚本
RUN echo '#!/bin/bash\n\
if [ ! -d "$MODEL_PATH" ]; then\n\
  echo "Downloading model to $MODEL_PATH..."\n\
  mkdir -p "$MODEL_PATH"\n\
  git clone https://huggingface.co/BAAI/bge-m3 "$MODEL_PATH"\n\
fi\n\
python3 embedding.py' > /app/start.sh \
    && chmod +x /app/start.sh

# 启动命令
CMD ["/app/start.sh"]
