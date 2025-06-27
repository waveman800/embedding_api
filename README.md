# 嵌入向量 API 服务

这是一个高性能的文本嵌入向量服务，支持多种预训练模型（如 BGE、M3E、GTE 等），提供 RESTful API 接口。该服务使用 Docker 容器化部署，支持 GPU 加速，并实现了请求批处理和并发控制。

## 功能特性

- 🚀 支持多种预训练模型（Qwen3、BGE、M3E、GTE 等）
- ⚡ 高性能异步处理，支持请求批处理
- 🔒 支持 API 密钥认证
- 📊 内置速率限制和并发控制
- 🐳 容器化部署，支持 GPU 加速
- 🔄 自动下载和管理模型

## 支持的模型

### Qwen 系列
- [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - 轻量级模型，适合资源有限的环境
- [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) - 平衡性能与资源消耗
- [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) - 高性能模型，适合对质量要求高的场景

### BGE 系列
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)

### 其他模型
- [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)
- [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- 其他兼容的 Sentence Transformers 模型

### 模型选择建议
- **轻量级应用**：Qwen3-Embedding-0.6B 或 m3e-base
- **通用场景**：bge-m3 或 Qwen3-Embedding-4B
- **高质量需求**：Qwen3-Embedding-8B 或 bge-large-zh

## 快速开始

### 1. 环境准备

- Docker 19.03+
- Docker Compose 1.29+
- NVIDIA Container Toolkit（如需 GPU 支持）

### 2. 配置

1. 复制环境变量文件：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，根据需求修改配置：
   ```bash
   # API 配置
   API_KEY=your_api_key_here
   PORT=6008
   
   # 模型配置
   MODEL_PATH=./models/bge-m3
   DEVICE=cuda  # 或 cpu
   TARGET_DIM=2560
   
   # 并发控制
   MAX_CONCURRENT_REQUESTS=10
   MAX_BATCH_SIZE=32
   BATCH_TIMEOUT=0.1
   MAX_QUEUE_SIZE=100
   THREAD_POOL_SIZE=8
   
   # 速率限制
   RATE_LIMIT_REQUESTS=100
   RATE_LIMIT_WINDOW=60
   ```

### 3. 构建并启动服务

```bash
# 构建镜像
# 注意：首次构建会自动下载模型，可能需要较长时间
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:6008/health

# 获取 API 文档
# 在浏览器中访问：http://localhost:6008/docs
```

## API 使用示例

### 获取文本嵌入向量

```bash
curl -X 'POST' \
  'http://localhost:6008/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your_api_key_here' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": ["这是一个测试句子", "这是另一个测试句子"],
    "model": "bge-m3"
  }'
```

### 响应示例

```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.4, 0.5, 0.6, ...],
      "index": 1
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## 性能调优

1. **批处理大小**：根据 GPU 内存调整 `MAX_BATCH_SIZE`
2. **线程池大小**：根据 CPU 核心数调整 `THREAD_POOL_SIZE`
3. **并发请求数**：根据服务器性能调整 `MAX_CONCURRENT_REQUESTS`
4. **批处理超时**：根据延迟需求调整 `BATCH_TIMEOUT`

## 开发

### 本地开发

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 启动开发服务器：
   ```bash
   python embedding.py
   ```

### 测试

```bash
# 运行单元测试
pytest

# 运行性能测试
python -m tests.performance_test
```

## 部署

### 生产环境建议

1. 使用 HTTPS 和反向代理（如 Nginx）
2. 配置监控和告警
3. 定期备份模型和配置
4. 使用容器编排工具（如 Kubernetes）进行扩展

## 许可证

MIT

## 致谢

- [BAAI](https://huggingface.co/BAAI) - 提供优秀的预训练模型
- [Hugging Face](https://huggingface.co/) - 模型托管和社区支持
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架

## 贡献

欢迎提交 Issue 和 Pull Request。

## 问题反馈

如有问题，请提交 Issue 或联系维护者。
