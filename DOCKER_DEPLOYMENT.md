# Docker Compose 部署指南

## 🚀 快速开始

### 1. 准备工作

确保你的系统已安装：
- Docker (>= 20.10)
- Docker Compose (>= 2.0)
- NVIDIA Docker (如果使用GPU)

### 2. 克隆项目并准备模型

```bash
# 克隆项目
git clone <your-repo-url>
cd embedding_api

# 下载Qwen3-4B模型（推荐）
python setup_qwen3.py qwen3-4b

# 或者手动下载模型到 models/Qwen3-Embedding-4B/ 目录
```

### 3. 配置环境变量

复制并编辑环境配置文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 基础配置
API_KEY=your_secure_api_key_here
MODEL_PATH=/app/models/Qwen3-Embedding-4B
DEVICE=cuda  # 或 cpu
PORT=6008
TARGET_DIM=1536

# 批处理控制（重要！）
ENABLE_BATCH_PROCESSING=false  # 推荐：禁用批处理，实现立即响应

# 并发控制
MAX_CONCURRENT_REQUESTS=20
MAX_BATCH_SIZE=8              # 仅在批处理模式下生效
BATCH_TIMEOUT=0.01            # 仅在批处理模式下生效
MAX_QUEUE_SIZE=2000
THREAD_POOL_SIZE=8
REQUEST_TIMEOUT=300
```

### 4. 启动服务

```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f embedding-api

# 检查服务状态
docker-compose ps
```

## 📋 配置选项详解

### 批处理模式控制

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_BATCH_PROCESSING` | `false` | 是否启用批处理模式 |

**推荐配置**：
- **直接处理模式**（推荐）：`ENABLE_BATCH_PROCESSING=false`
  - 每个请求立即处理，无等待时间
  - 响应速度最快
  - 适合单个或少量请求

- **批处理模式**：`ENABLE_BATCH_PROCESSING=true`
  - 多个请求打包处理
  - 适合大量并发请求
  - 可能有轻微延迟

### GPU vs CPU 配置

#### GPU 配置（推荐）：
```yaml
environment:
  - DEVICE=cuda
  - MAX_CONCURRENT_REQUESTS=20
  - THREAD_POOL_SIZE=8
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

#### CPU 配置：
```yaml
environment:
  - DEVICE=cpu
  - MAX_CONCURRENT_REQUESTS=8
  - THREAD_POOL_SIZE=4
# 移除 deploy.resources 部分
```

## 🛠️ 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f embedding-api

# 进入容器
docker-compose exec embedding-api bash
```

### 配置管理

```bash
# 在容器内禁用批处理
docker-compose exec embedding-api python disable_batch.py

# 重新构建镜像（配置更改后）
docker-compose build --no-cache

# 更新并重启
docker-compose up -d --build
```

## 🔧 自定义配置

### 1. 修改端口

编辑 `docker-compose.yml`：

```yaml
ports:
  - "8080:6008"  # 将服务映射到主机的8080端口
environment:
  - PORT=6008    # 容器内部端口保持不变
```

### 2. 使用外部模型目录

如果模型文件很大，可以挂载外部目录：

```yaml
volumes:
  - ./models:/app/models  # 挂载本地models目录
environment:
  - MODEL_PATH=/app/models/Qwen3-Embedding-4B
```

### 3. 多实例部署

创建 `docker-compose.scale.yml`：

```yaml
version: '3.8'

services:
  embedding-api-1:
    extends:
      file: docker-compose.yml
      service: embedding-api
    ports:
      - "6008:6008"
    
  embedding-api-2:
    extends:
      file: docker-compose.yml
      service: embedding-api
    ports:
      - "6009:6008"
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - embedding-api-1
      - embedding-api-2
```

## 🧪 测试部署

### 1. 健康检查

```bash
# 检查服务状态
curl http://localhost:6008/status

# 测试嵌入生成
curl -X POST http://localhost:6008/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "input": ["Hello, world!"],
    "model": "qwen3-4b"
  }'
```

### 2. 性能测试

```bash
# 进入容器运行测试
docker-compose exec embedding-api python test_direct_mode.py
```

## 🐛 故障排除

### 常见问题

1. **GPU不可用**
   ```bash
   # 检查NVIDIA Docker
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **内存不足**
   ```yaml
   # 在docker-compose.yml中限制内存
   deploy:
     resources:
       limits:
         memory: 8G
   ```

3. **模型加载失败**
   ```bash
   # 检查模型文件
   docker-compose exec embedding-api ls -la /app/models/
   
   # 重新下载模型
   docker-compose exec embedding-api python setup_qwen3.py qwen3-4b
   ```

4. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep 6008
   
   # 修改端口映射
   # 编辑 docker-compose.yml 中的 ports 配置
   ```

### 日志分析

```bash
# 查看详细日志
docker-compose logs --tail=100 embedding-api

# 实时监控日志
docker-compose logs -f embedding-api | grep -E "(ERROR|WARNING|批处理|Direct)"
```

## 📊 监控和维护

### 1. 资源监控

```bash
# 查看容器资源使用
docker stats embedding-api_embedding-api_1

# 查看GPU使用情况
nvidia-smi
```

### 2. 定期维护

```bash
# 清理未使用的镜像
docker system prune -f

# 更新镜像
docker-compose pull
docker-compose up -d
```

## 🔒 安全建议

1. **API密钥管理**
   - 使用强密码作为API_KEY
   - 定期轮换API密钥
   - 不要在代码中硬编码密钥

2. **网络安全**
   - 使用反向代理（如Nginx）
   - 启用HTTPS
   - 限制访问IP

3. **容器安全**
   - 定期更新基础镜像
   - 使用非root用户运行
   - 限制容器权限

## 📈 性能优化

### 直接处理模式（推荐）
```bash
ENABLE_BATCH_PROCESSING=false
MAX_CONCURRENT_REQUESTS=20
THREAD_POOL_SIZE=8
```

### 批处理模式（高并发场景）
```bash
ENABLE_BATCH_PROCESSING=true
MAX_BATCH_SIZE=32
BATCH_TIMEOUT=0.05
MAX_CONCURRENT_REQUESTS=10
```

通过Docker Compose，你可以轻松部署和管理嵌入API服务，享受容器化带来的便利和一致性。
