# Qwen3-VL-Embedding-2B API 服务

基于 Qwen3-VL-Embedding-2B 模型的嵌入向量生成服务，提供 RESTful API 接口。

## 功能特性

- 🚀 支持 Qwen3-VL-Embedding-2B 模型
- 🔧 2048 维嵌入向量
- ⚡ 批量文本处理
- 🖥️ GPU 加速（NVIDIA CUDA）
- 🔐 API 认证机制
- 📦 Docker 部署

## 快速开始

### 1. 准备模型

```bash
# 下载模型（使用 ModelScope）
python download-model.py

# 或手动下载到 models/Qwen3-VL-Embedding-2B 目录
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 修改 API_KEY 等配置
```

### 3. 启动服务

```bash
# 使用 docker-compose 启动（推荐）
sudo docker compose -f docker-compose.cuda-nomodel.yml up -d

# 查看日志
sudo docker logs -f embedding-api-qwen3vl-cuda
```

### 4. 测试 API

```bash
# 健康检查
curl http://localhost:6008/health

# 嵌入请求
curl -X POST http://localhost:6008/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-hv6xtPbK183j3RR306Fe23B6196b4d919a8e854887F6213d" \
  -d '{"input": "测试文本", "model": "Qwen3-VL-Embedding-2B"}'
```

## 项目结构

```
embedding_api/
├── embedding.py                    # 主服务代码
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量示例
├── .env                            # 本地环境变量配置
├── docker-compose.cuda-nomodel.yml # CUDA版本Docker Compose
├── docker-compose.cpu-nomodel.yml  # CPU版本Docker Compose
├── Dockerfile.cuda-runtime-nomodel # CUDA版本Dockerfile
├── Dockerfile.cpu-nomodel          # CPU版本Dockerfile
├── download-model.py               # 模型下载脚本
├── package-cuda-nomodel.sh         # CUDA镜像打包脚本
├── package-cpu-nomodel.sh          # CPU镜像打包脚本
├── load-image-cuda-nomodel.sh      # CUDA镜像加载脚本
├── load-image-cpu-nomodel.sh       # CPU镜像加载脚本
└── models/                         # 模型目录（挂载）
    └── Qwen3-VL-Embedding-2B/
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 服务端口 | 6008 |
| `HOST` | 监听地址 | 0.0.0.0 |
| `MODEL_PATH` | 模型路径 | /app/models/Qwen3-VL-Embedding-2B |
| `DEVICE` | 计算设备 | cuda |
| `API_KEY` | API 认证密钥 | sk-hv6xtPbK183j3RR306Fe23B6196b4d919a8e854887F6213d |
| `EMBEDDING_DIMENSION` | 嵌入维度 | 2048 |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA 内存配置 | max_split_size_mb:1024 |

## API 接口

### 健康检查

```http
GET /health
```

### 获取嵌入向量

```http
POST /v1/embeddings
Content-Type: application/json
Authorization: Bearer <API_KEY>
```

**请求体**:
```json
{
    "input": "文本内容",
    "model": "Qwen3-VL-Embedding-2B"
}
```

**批量请求**:
```json
{
    "input": ["文本1", "文本2", "文本3"],
    "model": "Qwen3-VL-Embedding-2B"
}
```

## 镜像打包与部署

### CUDA 版本（推荐，有 GPU）

**打包:**
```bash
./package-cuda-nomodel.sh
```
输出：`embedding-api-qwen3vl-cuda-nomodel.tar.gz`

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

# 4. 启动服务
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

# 4. 启动服务
sudo docker compose -f docker-compose.cpu-nomodel.yml up -d
```

## 常用命令

### CUDA 版本
```bash
# 启动服务
sudo docker compose -f docker-compose.cuda-nomodel.yml up -d

# 停止服务
sudo docker compose -f docker-compose.cuda-nomodel.yml down

# 查看日志
sudo docker logs -f embedding-api-qwen3vl-cuda
```

### CPU 版本
```bash
# 启动服务
sudo docker compose -f docker-compose.cpu-nomodel.yml up -d

# 停止服务
sudo docker compose -f docker-compose.cpu-nomodel.yml down

# 查看日志
sudo docker logs -f embedding-api-qwen3vl-cpu
```

## 镜像说明

### CUDA 版本
- **基础镜像**: `ccr.ccs.tencentyun.com/waveman/cuda:12.4.1-runtime-ubuntu22.04`
- **镜像大小**: 约 6-8GB（不含模型）
- **Python**: 3.10
- **PyTorch**: 2.x (CUDA 12.4)

### CPU 版本
- **基础镜像**: `ccr.ccs.tencentyun.com/waveman/python:3.12-slim`
- **镜像大小**: 约 2-3GB（不含模型）
- **Python**: 3.12
- **PyTorch**: CPU 版本

### 模型文件
- **位置**: 外挂挂载到 `models/Qwen3-VL-Embedding-2B/`
- **大小**: 约 4.5GB
- **下载**: `python download-model.py`
