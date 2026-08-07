# Qwen3-VL-Embedding-2B API 服务

基于 **Qwen3-VL-Embedding-2B** 模型的嵌入向量生成服务，提供兼容 OpenAI 风格的 RESTful API 接口。支持文本与图像输入，支持 GPU（CUDA）与 CPU 两种部署方式。

## 功能特性

- 🚀 基于 Qwen3-VL-Embedding-2B 多模态模型
- 🔧 2048 维嵌入向量（L2 归一化）
- ⚡ 批量文本处理
- 🖥️ GPU 加速（NVIDIA CUDA 12.4）与 CPU 双版本
- 🖼️ 支持文本与 base64 图像输入
- 🔐 API 认证机制（Bearer Token）
- 📦 Docker 部署，模型外挂（镜像不含模型）
- 🔄 模型通过 ModelScope 下载

## 快速开始

### 1. 准备模型

```bash
# 使用 ModelScope 下载模型（推荐）
python3 download-model.py

# 模型将下载到 models/Qwen3-VL-Embedding-2B/ 目录
# 注意：modelscope 下载的结构包含 snapshots/master 子目录
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 修改 API_KEY、DEVICE、MODEL_PATH 等配置
```

> ⚠️ **重要**：`.env.example` 中 `MODEL_PATH` 默认指向 `/app/models/Qwen3-VL-Embedding-2B/snapshots/master`，因为 ModelScope 下载的模型结构包含 `snapshots/master` 子目录。请根据实际模型目录结构调整。

### 3. 启动服务

```bash
# CUDA 版本（推荐，有 GPU）
sudo docker compose -f docker-compose.cuda-nomodel.yml up -d

# CPU 版本（无 GPU）
sudo docker compose -f docker-compose.cpu-nomodel.yml up -d
```

### 4. 测试 API

```bash
# 健康检查
curl http://localhost:6008/health

# 嵌入请求
curl -X POST http://localhost:6008/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <你的API_KEY>" \
  -d '{"input": "测试文本", "model": "Qwen3-VL-Embedding-2B"}'
```

## 项目结构

```
embedding_api/
├── embedding.py                        # 主服务代码
├── requirements.txt                    # Python 依赖
├── .env.example                        # 环境变量示例
├── .env                                # 本地环境变量配置
├── .dockerignore                       # Docker 构建忽略文件
├── docker-compose.cuda-nomodel.yml     # CUDA 版本 Compose
├── docker-compose.cuda-nomodel-local.yml # CUDA 版本 Compose（本地开发）
├── docker-compose.cpu-nomodel.yml      # CPU 版本 Compose
├── Dockerfile.cuda-runtime-nomodel     # CUDA 版本 Dockerfile
├── Dockerfile.cpu-nomodel              # CPU 版本 Dockerfile
├── download-model.py                   # 模型下载脚本（ModelScope）
├── package-cuda-nomodel.sh             # CUDA 镜像打包脚本
├── package-cpu-nomodel.sh              # CPU 镜像打包脚本
├── load-image-cuda-nomodel.sh          # CUDA 镜像加载脚本
├── load-image-cpu-nomodel.sh           # CPU 镜像加载脚本
├── cache/                              # 模型缓存目录（持久化）
└── models/                             # 模型目录（挂载）
    └── Qwen3-VL-Embedding-2B/
        └── snapshots/
            └── master/                 # 实际模型文件
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 模型路径（容器内） | `/app/models/Qwen3-VL-Embedding-2B/snapshots/master` |
| `DEVICE` | 计算设备：`cpu` 或 `cuda` | `cuda` |
| `EMBEDDING_DIMENSION` | 嵌入维度 | `2048` |
| `PORT` | 服务端口 | `6008` |
| `HOST` | 监听地址（`127.0.0.1` 仅本地 / `0.0.0.0` 允许外部） | `0.0.0.0` |
| `WORKERS` | 工作进程数 | `1` |
| `API_KEY` | API 认证密钥 | `sk-hv6xtPbK183j3RR306Fe23B6196b4d919a8e854887F6213d` |
| `OMP_NUM_THREADS` | OpenMP 线程数 | `4` |
| `MKL_NUM_THREADS` | MKL 线程数 | `4` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA 内存配置 | `max_split_size_mb:2048` |
| `PYTHONUNBUFFERED` | Python 输出无缓冲 | `1` |
| `MODEL_HOST_PATH` | 宿主机模型路径（Compose 挂载用） | `./models/Qwen3-VL-Embedding-2B` |

> ⚠️ **安全提示**：请务必修改默认的 `API_KEY` 为强密码。

## API 接口

### 健康检查

```http
GET /health
```

返回模型加载状态、设备、版本等信息。

### 根路径

```http
GET /
```

返回 API 基本信息与可用端点。

### 获取嵌入向量

```http
POST /v1/embeddings
Content-Type: application/json
Authorization: Bearer <API_KEY>
```

**单条文本请求**:
```json
{
    "input": "文本内容",
    "model": "Qwen3-VL-Embedding-2B"
}
```

**批量文本请求**:
```json
{
    "input": ["文本1", "文本2", "文本3"],
    "model": "Qwen3-VL-Embedding-2B"
}
```

**图像请求**（base64 编码）:
```json
{
    "input": [
        {"type": "text", "data": "描述这张图片"},
        {"type": "image", "data": "data:image/png;base64,..."}
    ],
    "model": "Qwen3-VL-Embedding-2B"
}
```

**响应格式**（OpenAI 兼容）:
```json
{
    "data": [
        {
            "embedding": [0.123, -0.456, ...],
            "index": 0,
            "object": "embedding"
        }
    ],
    "model": "Qwen3-VL-Embedding-2B",
    "object": "list",
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10
    }
}
```

## 镜像打包与部署

> 镜像均为**模型外挂版本**，不包含模型文件，模型需单独下载并挂载。

### 构建 Docker 镜像

**CUDA 版本:**
```bash
# 使用 Dockerfile.cuda-runtime-nomodel 构建
docker build -f Dockerfile.cuda-runtime-nomodel -t embedding-api:qwen3vl-cuda-nomodel .
```

**CPU 版本:**
```bash
# 使用 Dockerfile.cpu-nomodel 构建
docker build -f Dockerfile.cpu-nomodel -t embedding-api:qwen3vl-cpu-nomodel .
```

> 构建完成后可直接运行，或使用下方打包脚本导出为 tar.gz 以便离线部署。

### CUDA 版本（推荐，有 GPU）

**打包:**
```bash
./package-cuda-nomodel.sh
```
输出：`embedding-api-qwen3vl-cuda-nomodel.tar.gz`（约 4-6GB）

**部署:**
```bash
# 1. 传输镜像包
scp embedding-api-qwen3vl-cuda-nomodel.tar.gz user@target-host:/path/

# 2. 加载镜像（目标机器）
./load-image-cuda-nomodel.sh

# 3. 准备模型目录和 .env 文件
mkdir -p models/Qwen3-VL-Embedding-2B
cp .env.example .env
# 编辑 .env 配置 DEVICE=cuda

# 4. 下载模型
python3 download-model.py

# 5. 启动服务
sudo docker compose -f docker-compose.cuda-nomodel.yml up -d
```

### CPU 版本（无 GPU）

**打包:**
```bash
./package-cpu-nomodel.sh
```
输出：`embedding-api-qwen3vl-cpu-nomodel.tar.gz`

**部署:**
```bash
# 1. 传输镜像包
scp embedding-api-qwen3vl-cpu-nomodel.tar.gz user@target-host:/path/

# 2. 加载镜像（目标机器）
./load-image-cpu-nomodel.sh

# 3. 准备模型目录和 .env 文件
mkdir -p models/Qwen3-VL-Embedding-2B
cp .env.example .env
# 编辑 .env 配置 DEVICE=cpu

# 4. 下载模型
python3 download-model.py

# 5. 启动服务
sudo docker compose -f docker-compose.cpu-nomodel.yml up -d
```

## 常用命令

### CUDA 版本
```bash
# 构建镜像
docker build -f Dockerfile.cuda-runtime-nomodel -t embedding-api:qwen3vl-cuda-nomodel .

# 启动服务
sudo docker compose -f docker-compose.cuda-nomodel.yml up -d

# 停止服务
sudo docker compose -f docker-compose.cuda-nomodel.yml down

# 查看日志
sudo docker logs -f embedding-api-qwen3vl-cuda
```

### CPU 版本
```bash
# 构建镜像
docker build -f Dockerfile.cpu-nomodel -t embedding-api:qwen3vl-cpu-nomodel .

# 启动服务
sudo docker compose -f docker-compose.cpu-nomodel.yml up -d

# 停止服务
sudo docker compose -f docker-compose.cpu-nomodel.yml down

# 查看日志
sudo docker logs -f embedding-api-qwen3vl-cpu
```

## Docker Compose 配置说明

### CUDA 版本（`docker-compose.cuda-nomodel.yml`）

- **镜像**: `embedding-api:qwen3vl-cuda-nomodel`
- **容器名**: `embedding-api-qwen3vl-cuda`
- **GPU**: 通过 `deploy.resources.reservations.devices` 指定，默认使用 `device_ids: ["3"]`（请根据实际 GPU 编号修改）
- **端口**: `0.0.0.0:${PORT:-6008}:6008`
- **挂载**:
  - 模型目录: `${MODEL_HOST_PATH:-./models/Qwen3-VL-Embedding-2B}:/app/models/Qwen3-VL-Embedding-2B:ro`
  - 环境变量: `./.env:/app/.env:ro`
  - 缓存: `./cache:/root/.cache`
- **健康检查**: 每 30s 检查 `/health`，启动等待 60s

### CPU 版本（`docker-compose.cpu-nomodel.yml`）

- **镜像**: `embedding-api:qwen3vl-cpu-nomodel`
- **容器名**: `embedding-api-qwen3vl-cpu`
- **资源限制**: CPU 4 核 / 内存 8G（可调整）
- **健康检查**: 每 60s 检查 `/health`，启动等待 300s（CPU 加载模型较慢）

### 本地开发版本（`docker-compose.cuda-nomodel-local.yml`）

用于本地开发调试，配置与 CUDA 版本类似。

## 镜像说明

### CUDA 版本
- **基础镜像**: `ccr.ccs.tencentyun.com/waveman/cuda:12.4.1-runtime-ubuntu22.04`
- **镜像大小**: 约 4-6GB（不含模型）
- **Python**: 3.10
- **PyTorch**: 2.11.0（CUDA 12.8 / cu128）
- **torchvision**: 0.26.0
- **torchaudio**: 2.11.0

### CPU 版本
- **基础镜像**: `ccr.ccs.tencentyun.com/waveman/python:3.12-slim`
- **镜像大小**: 约 2-3GB（不含模型）
- **Python**: 3.12
- **PyTorch**: CPU 版本

### 模型文件
- **位置**: 外挂挂载到 `models/Qwen3-VL-Embedding-2B/`
- **大小**: 约 4.5GB
- **下载**: `python3 download-model.py`（ModelScope）

## 常见问题

### 1. 模型加载失败 / 找不到模型

容器启动时会检查 `/app/models/Qwen3-VL-Embedding-2B` 目录是否存在且非空。请确认：
- 模型已正确下载到 `models/Qwen3-VL-Embedding-2B/`
- `.env` 中 `MODEL_PATH` 指向正确的模型子目录（如 `snapshots/master`）

### 2. GPU 不可用

- 确认已安装 NVIDIA 驱动和 `nvidia-container-toolkit`
- 确认 `docker-compose.cuda-nomodel.yml` 中 `device_ids` 与实际 GPU 编号一致
- 可通过 `nvidia-smi` 查看 GPU 编号

### 3. 端口被占用

修改 `.env` 中的 `PORT` 变量，或使用 `MODEL_HOST_PATH` 指定模型路径。

### 4. 显存不足

调整 `.env` 中的 `PYTORCH_CUDA_ALLOC_CONF`：
- 8GB 显存: `max_split_size_mb:512-1024`
- 16GB 显存: `max_split_size_mb:1024-2048`
- 24GB+ 显存: `max_split_size_mb:2048-4096`

## 许可证

请遵守 Qwen3-VL-Embedding-2B 模型的开源许可协议。
