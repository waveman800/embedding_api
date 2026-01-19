# Embedding API 服务

基于 Qwen3-VL-Embedding-2B 模型的嵌入向量生成服务，支持文本和图片输入，提供简单的 RESTful API 接口。

## 功能特性

- 🚀 支持 Qwen3-VL-Embedding-2B 多模态模型
- 🔧 动态嵌入维度检测（自动适配模型实际输出）
- ⚡ 支持批量处理文本和图片
- 🖼️ 支持图片URL输入
- 🎨 支持图片base64格式输入
- 🎯 优化的向量归一化处理
- 🖥️ 支持 GPU 加速（NVIDIA CUDA）
- 🔐 简单的 API 认证机制
- 🧪 完整的输入验证和错误处理
- 🛠️ 灵活的环境变量配置

## 快速开始

### 前置要求

- Python 3.8+
- pip 包管理工具
- (可选) NVIDIA GPU 和 CUDA 环境（如需 GPU 加速）

### 1. 克隆仓库

```bash
git clone https://github.com/waveman800/embedding_api.git
cd embedding_api
```

### 2. 创建并激活虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制示例环境文件并修改：

```bash
cp .env.example .env
# 编辑 .env 文件配置您的设置
```

### 5. 启动开发服务器

```bash
python -m uvicorn embedding_api.main:app --host 0.0.0.0 --port 6008
```

## 模型准备

1. 下载 Qwen3-VL-Embedding-2B 模型到 `models/Qwen3-VL-Embedding-2B` 目录
2. 确保模型文件结构如下：
   ```
   models/
   └── Qwen3-VL-Embedding-2B/
       ├── config.json
       ├── pytorch_model.bin
       ├── tokenizer.json
       └── ...
   ```

## 环境变量配置

复制 `.env.example` 到 `.env` 并修改配置：

```env
# API 配置
API_KEY=sk-embedding-api-secret-key-20260114  # API 认证密钥
PORT=6008                                     # 服务端口

# 模型配置
MODEL_NAME=Qwen3-VL-Embedding-2B              # 模型名称
MODEL_PATH=./models/Qwen3-VL-Embedding-2B     # 模型路径
DEVICE=cuda                                    # 使用 cuda 或 cpu
# EMBEDDING_DIMENSION=2048                   # 嵌入向量维度（可选，默认自动检测）

# GPU 配置 (可选)
CUDA_VISIBLE_DEVICES=1                         # 指定使用的 GPU 设备号

# 图片处理配置
MAX_IMAGE_SIZE=512                             # 图片最大尺寸
MAX_IMAGE_WIDTH=512                            # 图片最大宽度
MAX_IMAGE_HEIGHT=512                           # 图片最大高度

# 日志配置
LOG_LEVEL=INFO                                 # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
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
    "model": "Qwen3-VL-Embedding-2B",
    "model_path": "./models/Qwen3-VL-Embedding-2B",
    "device": "cuda",
    "cuda_available": true,
    "embedding_dimension": 2048,
    "max_image_width": 512,
    "max_image_height": 512
}
```

### 2. 获取嵌入向量

```http
POST /v1/embeddings
Content-Type: application/json
Authorization: Bearer your-api-key
```

**文本输入**:
```json
{
    "input": "这是一段示例文本",
    "model": "Qwen3-VL-Embedding-2B"
}
```

**批量文本处理**:
```json
{
    "input": ["文本1", "文本2", "文本3"],
    "model": "Qwen3-VL-Embedding-2B"
}
```

**图片URL输入**:
```json
{
    "input": {
        "type": "image",
        "data": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=100&h=100&fit=crop"
    },
    "model": "Qwen3-VL-Embedding-2B"
}
```

**图片base64输入**:
```json
{
    "input": {
        "type": "image",
        "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/..."
    },
    "model": "Qwen3-VL-Embedding-2B"
}
```

**混合输入**:
```json
{
    "input": [
        {
            "type": "text",
            "data": "这是一段文本描述，例如：这张图片展示了什么？"
        },
        {
            "type": "image",
            "data": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=100&h=100&fit=crop"
        }
    ],
    "model": "Qwen3-VL-Embedding-2B"
}
```

**响应示例**:
```json
{
    "data": [
        {
            "embedding": [0.00027890555088747076, -0.00029999671793678816, -0.0007192418379596056, ...],
            "index": 0,
            "object": "embedding"
        }
    ],
    "model": "Qwen3-VL-Embedding-2B",
    "object": "list",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

## 项目结构

```
embedding_api/
├── embedding_api/          # 项目源代码
│   ├── __init__.py        # 包初始化文件
│   └── main.py            # FastAPI 主应用
├── models/                # 模型文件目录
├── .env.example           # 环境变量示例
├── .gitignore             # Git 忽略配置
├── Makefile               # 常用命令
├── pyproject.toml         # 项目元数据
├── README.md              # 项目说明
├── requirements.txt       # 生产依赖
└── requirements-dev.txt   # 开发依赖
```

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务器
python -m uvicorn embedding_api.main:app --host 0.0.0.0 --port 6008

# 查看日志
python -m uvicorn embedding_api.main:app --host 0.0.0.0 --port 6008 --log-level info
```

## 部署

### 生产环境部署建议

1. 使用进程管理工具（如 systemd 或 supervisor）管理服务
2. 配置适当的监控和日志收集
3. 考虑使用反向代理（如 Nginx）处理 HTTPS
4. 定期备份模型和配置

## 配置说明

### 自定义嵌入维度

默认情况下，系统会自动检测模型的实际输出维度（Qwen3-VL-Embedding-2B 模型为 2048 维）。

如果需要将嵌入向量扩展到特定维度，可以在 `.env` 中设置 `EMBEDDING_DIMENSION` 环境变量：

```bash
# 示例：将嵌入向量扩展到 2048 维
EMBEDDING_DIMENSION=2048
```

### 使用其他模型

1. 将模型文件放入 `models/` 目录
2. 更新 `.env` 中的 `MODEL_NAME` 和 `MODEL_PATH`

### GPU 配置

- 设置 `CUDA_VISIBLE_DEVICES` 环境变量指定使用的 GPU 设备
- 确保 CUDA 环境配置正确

### 图片处理

- `MAX_IMAGE_SIZE`：设置图片的最大尺寸（宽度或高度）
- `MAX_IMAGE_WIDTH` 和 `MAX_IMAGE_HEIGHT`：单独设置图片的最大宽度和高度

## 常见问题

### 1. 模型加载失败

- 确保模型文件完整且路径正确
- 检查文件权限
- 验证模型是否与代码版本兼容

### 2. GPU 内存不足

- 减小批处理大小
- 使用更小的模型
- 降级到 CPU 模式（设置 `DEVICE=cpu`）

### 3. 图片处理失败

- 确保图片 URL 可访问且格式正确
- 检查 base64 编码是否正确
- 验证图片大小是否在配置范围内

### 4. API 认证失败

- 检查 `API_KEY` 环境变量是否设置正确
- 确保请求头中包含正确的 `Authorization: Bearer <API_KEY>`

## 贡献指南

欢迎提交 Issue 和 Pull Request。在提交代码前，请确保：

1. 通过所有输入验证
2. 代码符合项目的风格和结构
3. 更新相关文档

## 许可证

[MIT](LICENSE)