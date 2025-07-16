# Embedding API 服务

基于 Qwen3-Embedding-4B 模型的嵌入向量生成服务，支持 GPU 加速，提供简单的 RESTful API 接口。

## 功能特性

- 🚀 支持 Qwen3-Embedding-4B 模型（默认）
- 🔧 可配置的嵌入维度（默认2560维）
- ⚡ 支持批量处理文本
- 🎯 优化的向量归一化处理
- 🖥️ 支持多GPU张量并行（NVIDIA）
- 🔐 简单的 API 认证机制
- 🧪 完整的测试覆盖
- 🛠️ 代码质量检查与自动格式化

## 快速开始

### 前置要求

- Python 3.8+
- pip 或 uv 包管理工具
- (可选) Docker 19.03+ 和 Docker Compose 1.28+
- (GPU 支持) NVIDIA 驱动 >= 535.86.05
- (多GPU 支持) 至少2个 NVIDIA GPU 卡

### 1. 克隆仓库

```bash
git clone <repository-url>
cd embedding_api
```

### 2. 创建并激活虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows

# 使用 conda
# conda create -n embedding-api python=3.10
# conda activate embedding-api
```

### 3. 安装依赖

使用 `uv`（推荐，更快）：

```bash
uv pip install -r requirements.txt

# 开发环境
uv pip install -r requirements-dev.txt
```

或使用 `pip`：

```bash
pip install -r requirements.txt

# 开发环境
pip install -r requirements-dev.txt
```

### 4. 配置环境变量

复制示例环境文件并修改：

```bash
cp .env.example .env
# 编辑 .env 文件配置您的设置
```

### 5. 启动开发服务器

```bash
# 使用 Makefile
make dev

# 或直接运行
uvicorn embedding_api.main:app --reload --host 0.0.0.0 --port 6008
```

### 6. 使用 Docker 运行（可选）

确保已安装 Docker 和 Docker Compose：

```bash
docker-compose up -d --build
```

## 模型准备

1. 下载 Qwen3-Embedding-4B 模型到 `models/Qwen3-Embedding-4B` 目录
2. 确保模型文件结构如下：
   ```
   models/
   └── Qwen3-Embedding-4B/
       ├── config.json
       ├── pytorch_model.bin
       ├── tokenizer.json
       └── ...
   ```

## 环境变量配置

复制 `.env.example` 到 `.env` 并修改配置：

```env
# API 配置
API_KEY=your-secret-key  # API 认证密钥
PORT=6008               # 服务端口

# 模型配置
MODEL_PATH=./models/Qwen3-Embedding-4B  # 模型路径
DEVICE=cuda              # 使用 cuda 或 cpu
EMBEDDING_DIMENSION=2560 # 嵌入向量维度

# GPU 配置 (多GPU支持)
CUDA_VISIBLE_DEVICES=0,1   # 指定使用的 GPU 设备号，例如 0,1 表示使用前两个GPU
BATCH_SIZE=24             # 批处理大小，根据GPU内存调整

# 高级 GPU 配置 (可选)
NCCL_DEBUG=INFO           # 启用NCCL调试信息
NCCL_IB_DISABLE=0         # 启用InfiniBand支持
OMP_NUM_THREADS=1         # 设置OpenMP线程数

# 日志配置
LOG_LEVEL=INFO          # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 多GPU张量并行支持

### 功能特点

- 🚀 自动将模型张量分布到多个GPU上
- ⚡ 支持动态批处理，优化GPU利用率
- 🔧 可配置的GPU设备选择
- 📊 提供GPU使用情况监控工具

### 使用方法

1. **使用Docker运行** (推荐):
   ```bash
   # 使用前两个GPU
   docker run --gpus all -p 6008:6008 \
     -e CUDA_VISIBLE_DEVICES=0,1 \
     -e BATCH_SIZE=24 \
     -v /path/to/models:/app/models \
     embedding-api:gpu
   ```

2. **验证多GPU配置**:
   ```bash
   # 进入容器
   docker exec -it <container_id> bash
   
   # 运行GPU检查脚本
   python check_gpu.py
   ```

3. **监控GPU使用情况**:
   ```bash
   # 在宿主机上执行
   watch -n 1 nvidia-smi
   ```

## API 文档

### 1. 健康检查

```http
GET /health
```

**响应示例**:
```json
{
    "status": "ok",
    "model": "Qwen3-Embedding-4B",
    "device": "cuda"
}
```

### 2. 获取嵌入向量

```http
POST /v1/embeddings
Content-Type: application/json
Authorization: Bearer your-api-key
```

**请求体**:
```json
{
    "input": "这是一段示例文本",
    "model": "Qwen3-Embedding-4B"
}
```

**批量处理**:
```json
{
    "input": ["文本1", "文本2", "文本3"],
    "model": "Qwen3-Embedding-4B"
}
```

**响应示例**:
```json
{
    "data": [
        {
            "embedding": [0.1, 0.2, 0.3, ...],
            "index": 0,
            "object": "embedding"
        }
    ],
    "model": "Qwen3-Embedding-4B",
    "object": "list",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

## 开发指南

### 项目结构

```
embedding_api/
├── embedding_api/          # 项目源代码
│   ├── __init__.py        # 包初始化文件
│   └── main.py            # FastAPI 主应用
├── tests/                 # 测试代码
├── .env.example           # 环境变量示例
├── .gitignore             # Git 忽略配置
├── Makefile               # 常用命令
├── pyproject.toml         # 项目元数据和依赖
├── README.md              # 项目说明
├── requirements.txt       # 生产依赖
└── requirements-dev.txt   # 开发依赖
```

### 开发命令

```bash
# 安装开发依赖
make install-dev

# 运行测试
make test

# 代码格式化和检查
make format    # 自动格式化代码
make lint      # 运行代码检查
make typecheck # 类型检查

# 清理构建文件
make clean
```

## 部署

### 使用 Docker 部署

1. 构建并启动容器：
   ```bash
   docker-compose up -d --build
   ```

2. 查看日志：
   ```bash
   docker-compose logs -f
   ```

3. 停止服务：
   ```bash
   docker-compose down
   ```

### 生产环境部署建议

1. 使用反向代理（如 Nginx）处理 HTTPS 和负载均衡
2. 配置适当的监控和日志收集
3. 使用进程管理工具（如 systemd 或 supervisor）管理服务
4. 定期备份模型和配置

## 配置说明

### 自定义嵌入维度

在 `.env` 中修改 `EMBEDDING_DIMENSION` 环境变量（默认2560）。

### 使用其他模型

1. 将模型文件放入 `models/` 目录
2. 更新 `.env` 中的 `MODEL_PATH` 和 `EMBEDDING_DIMENSION`

### GPU 配置

- 修改 `docker-compose.yml` 中的 `device_ids` 指定使用的 GPU 设备
- 或设置 `CUDA_VISIBLE_DEVICES` 环境变量

## 常见问题

### 1. 模型加载失败

- 确保模型文件完整且路径正确
- 检查文件权限
- 验证模型是否与代码版本兼容

### 2. GPU 内存不足

- 减小批处理大小
- 使用更小的模型
- 降级到 CPU 模式（设置 `DEVICE=cpu`）
- 检查是否有其他进程占用 GPU 内存

### 3. 性能优化

- 确保使用 GPU 加速
- 增加 `docker-compose.yml` 中的 `shm_size` 如果遇到共享内存问题
- 调整批处理大小 (`BATCH_SIZE`) 以优化吞吐量
- 使用 `uv` 替代 `pip` 加速依赖安装
- 对于多GPU环境，确保 `CUDA_VISIBLE_DEVICES` 正确设置
- 使用 `check_gpu.py` 脚本验证张量并行是否正常工作

### 4. API 认证失败

- 检查 `API_KEY` 环境变量是否设置正确
- 确保请求头中包含正确的 `Authorization: Bearer <API_KEY>`

## 故障排除

### 多GPU相关问题

1. **张量并行未生效**
   - 运行 `python check_gpu.py` 验证模型是否分布在多个GPU上
   - 检查 `nvidia-smi` 确认所有目标GPU都被使用
   - 确保 `CUDA_VISIBLE_DEVICES` 环境变量设置正确

2. **GPU内存不足**
   - 减小 `BATCH_SIZE` 环境变量
   - 检查是否有其他进程占用GPU内存
   - 考虑使用 `torch.cuda.empty_cache()` 清理缓存

3. **NCCL通信问题**
   - 设置 `NCCL_DEBUG=INFO` 查看详细日志
   - 确保所有GPU使用相同型号和驱动版本
   - 检查InfiniBand/RDMA配置（如果使用）

## 贡献指南

欢迎提交 Issue 和 Pull Request。在提交代码前，请确保：

1. 通过所有测试
2. 代码符合 PEP 8 规范
3. 更新相关文档
4. 添加适当的测试用例

## 许可证

[MIT](LICENSE)
