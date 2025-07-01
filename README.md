# 嵌入向量 API 服务

这是一个高性能的文本嵌入向量服务，支持多种预训练模型（如 BGE、M3E、GTE 等），提供 RESTful API 接口。该服务使用 Docker 容器化部署，支持 GPU 加速，并实现了请求批处理和并发控制。

## 功能特性

- 🚀 支持多种预训练模型（Qwen3、BGE、M3E、GTE 等）
- ⚡ 高性能异步处理，支持请求批处理
- 🔒 支持 API 密钥认证
- 🛡️ **先进的并发控制** - 使用信号量机制防止服务过载
- 🔄 **优雅降级** - 超限时返回429状态码而非503错误
- 📊 实时状态监控和健康检查
- ⏱️ **请求超时控制** - 防止长时间阻塞
- 🐳 容器化部署，支持 GPU 加速
- 🔄 自动下载和管理模型

## 支持的模型

### 🆕 Qwen3 系列（推荐）
- **Alibaba-NLP/gte-Qwen2-1.5B-instruct** - 轻量级高性能模型，1536维嵌入
- **Alibaba-NLP/gte-Qwen2-7B-instruct** - 大型高质量模型，4096维嵌入
- **Alibaba-NLP/gte-large-en-v1.5** - 英文专用大型模型
- **Alibaba-NLP/gte-base-en-v1.5** - 英文专用基础模型

### 🔥 传统高性能模型

- **BAAI/bge-m3** - 多语言高性能模型，1024维嵌入
- **BAAI/bge-large-zh** - 中文专用大型模型

### 🌐 其他模型
- **moka-ai/m3e-base** - 中文优化模型
- **sentence-transformers/all-mpnet-base-v2** - 英文通用模型
- 其他兼容的 Sentence Transformers 模型

### 🎯 模型选择建议

| 场景 | 推荐模型 | 维度 | 特点 |
|------|------------|------|------|
| **轻量级部署** | gte-Qwen2-1.5B-instruct | 1536 | 快速、低内存 |
| **通用应用** | bge-m3 | 1024 | 多语言、稳定 |
| **高质量需求** | gte-Qwen2-7B-instruct | 4096 | 最佳效果、高精度 |
| **中文专用** | bge-large-zh | 1024 | 中文优化 |
| **英文专用** | gte-large-en-v1.5 | 1024 | 英文优化 |

## 快速开始

### 1. 环境准备

- Docker 19.03+
- Docker Compose 1.29+
- NVIDIA Container Toolkit（如需 GPU 支持）

### 2. 配置

#### 方法一：使用自动设置脚本（推荐）

使用我们提供的环境设置脚本快速配置：

```bash
# 自动创建 .env 文件
python setup_env.py
```

脚本将自动：
- 📝 从 .env.example 创建 .env 文件
- ⚙️ 检查依赖包安装状态
- 📝 提供配置指导
- ✅ 验证环境配置

#### 方法二：手动配置

1. 复制环境变量文件：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，配置你的服务参数：
   ```bash
   # API 配置
   API_KEY=your_api_key_here
   PORT=6008
   
   # Qwen3模型配置（Docker镜像已内置模型）
   MODEL_PATH=./models/Qwen3-Embedding-4B  # 已打包在镜像中
   DEVICE=cuda  # 或 cpu
   TARGET_DIM=1536  # Qwen3模型的嵌入维度
   
   # 优化的并发控制参数
   MAX_CONCURRENT_REQUESTS=20  # 提高并发处理能力
   MAX_BATCH_SIZE=64          # 增大批处理大小
   BATCH_TIMEOUT=0.05         # 优化批处理超时
   MAX_QUEUE_SIZE=2000        # 扩大队列容量
   THREAD_POOL_SIZE=8         # 线程池大小
   REQUEST_TIMEOUT=300        # 请求超时时间
   
   # 速率限制
   RATE_LIMIT_REQUESTS=1000   # 提高速率限制
   RATE_LIMIT_WINDOW=60
   ```

### 3. 构建并启动服务

#### 方法一：使用智能构建脚本（推荐）

```bash
# 使用智能构建脚本，包含前置检查和构建指导
python build_docker.py
```

或者使用简化版本：
```bash
python build.py
```

#### 方法二：使用 Docker Compose

```bash
# 构建镜像（包含 Qwen3 模型，首次构建需要 10-30 分钟）
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

#### 方法三：使用服务启动脚本

如果你在本地环境中运行（非 Docker）：

```bash
# 使用服务启动脚本，自动检查环境和依赖
python start_service.py
```

该脚本将：
- ✅ 检查 .env 配置文件
- ✅ 验证模型路径
- ✅ 检查依赖包
- 🚀 启动 embedding.py 服务

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:6008/health

# 状态监控（包含并发统计信息）
curl http://localhost:6008/status

# 获取 API 文档
# 在浏览器中访问：http://localhost:6008/docs
```

#### 状态监控响应示例

```json
{
  "status": "operational",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "concurrency": {
    "active_requests": 2,
    "total_requests": 150,
    "rejected_requests": 5,
    "available_slots": 8
  },
  "model_loaded": true,
  "device": "cuda"
}

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

### 🚀 Qwen3 模型优化配置

针对 Qwen3-Embedding-4B 模型的推荐配置：

```bash
# GPU 环境优化配置
MAX_CONCURRENT_REQUESTS=20    # 高并发处理
MAX_BATCH_SIZE=64            # 大批处理提高吞吐量
BATCH_TIMEOUT=0.05           # 低延迟批处理
MAX_QUEUE_SIZE=2000          # 大队列容量
REQUEST_TIMEOUT=300          # 适中请求超时

# CPU 环境优化配置
MAX_CONCURRENT_REQUESTS=8     # 降低并发数
MAX_BATCH_SIZE=16            # 小批处理减少内存压力
BATCH_TIMEOUT=0.1            # 适当增加超时
```

### ⚙️ 详细调优指南

1. **批处理大小** (`MAX_BATCH_SIZE`)
   - GPU: 32-128，根据 VRAM 大小调整
   - CPU: 8-32，避免内存溢出

2. **并发控制** (`MAX_CONCURRENT_REQUESTS`)
   - 高性能 GPU: 20-50
   - 中端 GPU: 10-20
   - CPU 环境: 4-8

3. **批处理超时** (`BATCH_TIMEOUT`)
   - 低延迟需求: 0.01-0.05s
   - 高吞吐量需求: 0.1-0.5s

4. **线程池配置** (`THREAD_POOL_SIZE`)
   - 通常设为 CPU 核心数的 1-2 倍

5. **内存优化**
   - 设置适当的 `MAX_QUEUE_SIZE` 避免内存溢出
   - 监控内存使用情况

## 🐳 Docker 优化特性

本项目的 Docker 镜像已经进行了全面优化：

### ✨ 优化亮点

- 📦 **模型内置**：Qwen3-Embedding-4B 模型直接打包在镜像中
- 🚀 **快速启动**：无需运行时下载，秒级启动服务
- 🌍 **国内优化**：使用清华镜像源，构建速度提升 70%
- 📊 **高性能**：优化的并发参数，支持 20+ 并发请求
- 🔒 **离线部署**：支持完全离线环境部署

### 🛠️ 构建优化

```bash
# 使用优化的 Dockerfile.cn
# - 清华镜像源，国内网络友好
# - 移除不必要工具，减小镜像体积
# - 优化的环境变量配置

# 构建时间对比：
# 传统方式: 45-60 分钟
# 优化后:  15-25 分钟
```

### 📝 配置管理

- ✅ **环境变量驱动**：完全依赖 .env 文件配置
- ✅ **智能检查**：自动验证配置和依赖
- ✅ **一键部署**：提供多种部署方式

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

#### 并发控制测试

使用内置的测试工具验证服务的并发处理能力：

```bash
# 基本并发测试（20个并发请求）
python test_concurrency.py --concurrent 20

# 逐步负载测试（从5到50个并发）
python test_concurrency.py --gradual --max-concurrent 50

# 指定服务地址和API密钥
python test_concurrency.py --url http://your-server:6008 --api-key your_key
```

#### 其他测试

```bash
# 运行单元测试
pytest

# 运行性能测试
python -m tests.performance_test
```

## 故障排查

### 常见问题

#### 1. 服务返回 429 错误

**原因**：请求超过了并发限制

**解决方案**：
- 检查 `/status` 端点查看当前并发情况
- 调整 `MAX_CONCURRENT_REQUESTS` 参数
- 实现客户端重试机制，遵循 `Retry-After` 头部

#### 2. 请求超时

**原因**：单个请求处理时间过长

**解决方案**：
- 调整 `REQUEST_TIMEOUT` 参数
- 检查模型加载和 GPU 资源使用情况
- 优化批处理参数 `MAX_BATCH_SIZE`

#### 3. 内存不足

**原因**：模型太大或并发请求过多

**解决方案**：
- 降低 `MAX_CONCURRENT_REQUESTS` 和 `MAX_BATCH_SIZE`
- 使用更小的模型
- 增加系统内存或使用 GPU

### 监控指标

通过 `/status` 端点监控以下指标：

- `active_requests`: 当前活跃请求数
- `total_requests`: 总请求数
- `rejected_requests`: 被拒绝的请求数
- `available_slots`: 可用并发槽位

**建议告警阈值**：
- 拒绝率 > 5%
- 平均响应时间 > 2秒
- 可用槽位 < 2

### 日志分析

```bash
# 查看实时日志
docker-compose logs -f

# 查看错误日志
docker-compose logs | grep ERROR

# 查看并发控制日志
docker-compose logs | grep "Request rejected"
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
