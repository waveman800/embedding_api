#!/bin/bash
# 打包 Qwen3-VL-Embedding-2B CUDA Docker 镜像（模型外挂版本）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "  打包 Qwen3-VL-Embedding-2B CUDA 镜像"
echo "  [模型外挂版本 - 镜像不含模型]"
echo "========================================"
echo ""

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    exit 1
fi

# 检查 Docker 权限
if ! docker info &> /dev/null 2>&1; then
    echo -e "${RED}错误: 没有 Docker 权限${NC}"
    echo "请尝试: sudo $0"
    exit 1
fi

# 配置
IMAGE_NAME="embedding-api"
IMAGE_TAG="qwen3vl-cuda-nomodel"
OUTPUT_FILE="embedding-api-qwen3vl-cuda-nomodel.tar.gz"
DOCKERFILE="Dockerfile.cuda-runtime-nomodel"

echo "镜像名称: $IMAGE_NAME:$IMAGE_TAG"
echo "输出文件: $OUTPUT_FILE"
echo "基础镜像: ccr.ccs.tencentyun.com/waveman/cuda:12.4.1-runtime-ubuntu22.04"
echo "Dockerfile: $DOCKERFILE"
echo ""
echo -e "${YELLOW}注意: 此版本镜像不包含模型文件${NC}"
echo -e "${YELLOW}      模型需要单独下载并挂载${NC}"
echo ""

# 检查是否已有镜像
echo -e "${BLUE}检查现有镜像...${NC}"
if docker images "$IMAGE_NAME:$IMAGE_TAG" --format "{{.Repository}}:{{.Tag}}" | grep -q "$IMAGE_NAME:$IMAGE_TAG"; then
    echo -e "${YELLOW}发现现有镜像: $IMAGE_NAME:$IMAGE_TAG${NC}"
    read -p "是否重新构建? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "使用现有镜像..."
        SKIP_BUILD=1
    fi
fi

# 构建镜像
if [ -z "$SKIP_BUILD" ]; then
    echo ""
    echo -e "${BLUE}步骤 1/3: 构建 Docker 镜像...${NC}"
    echo "镜像不包含模型，预计大小: 4-6GB"
    echo "预计耗时: 5-15 分钟"
    echo ""
    
    docker build \
        -f "$DOCKERFILE" \
        -t "$IMAGE_NAME:$IMAGE_TAG" \
        .
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 镜像构建失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 镜像构建成功!${NC}"
fi

# 显示镜像信息
echo ""
echo -e "${BLUE}镜像信息:${NC}"
docker images "$IMAGE_NAME:$IMAGE_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.ID}}"

# 导出镜像
echo ""
echo -e "${BLUE}步骤 2/3: 导出镜像...${NC}"
echo "正在打包镜像到 $OUTPUT_FILE ..."

docker save "$IMAGE_NAME:$IMAGE_TAG" | gzip > "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 导出失败${NC}"
    exit 1
fi

# 显示结果
echo -e "${GREEN}✓ 导出成功!${NC}"
echo ""
echo "文件信息:"
ls -lh "$OUTPUT_FILE"

# 计算 MD5
echo ""
echo -e "${BLUE}步骤 3/3: 生成校验文件...${NC}"
md5sum "$OUTPUT_FILE" > "$OUTPUT_FILE.md5"
echo "MD5: $(cat $OUTPUT_FILE.md5 | awk '{print $1}')"

# 生成加载脚本
cat > "load-image-cuda-nomodel.sh" << 'EOF'
#!/bin/bash
# 加载 Qwen3-VL-Embedding-2B CUDA 镜像（模型外挂版本）

IMAGE_FILE="embedding-api-qwen3vl-cuda-nomodel.tar.gz"

echo "Loading Docker image from $IMAGE_FILE..."
docker load < "$IMAGE_FILE"

echo ""
echo "Image loaded successfully!"
echo ""
echo "=================================================="
echo "重要: 此镜像不包含模型文件，需要单独下载模型"
echo "=================================================="
echo ""
echo "步骤 1: 准备 .env 配置文件"
echo "  cp .env.example .env"
echo "  # 编辑 .env 文件，修改必要配置"
echo ""
echo "步骤 2: 下载模型"
echo "  python3 download-model.py"
echo ""
echo "步骤 3: 启动服务（使用 docker-compose）"
echo "  docker-compose -f docker-compose.cuda-nomodel.yml up -d"
echo ""
echo "或直接使用 docker:"
echo "  docker run -d --gpus all -p 6008:6008 \\"
echo "    -v \$(pwd)/.env:/app/.env:ro \\"
echo "    -v \$(pwd)/models/Qwen3-VL-Embedding-2B:/app/models/Qwen3-VL-Embedding-2B:ro \\"
echo "    embedding-api:qwen3vl-cuda-nomodel"
echo ""
EOF
chmod +x load-image-cuda-nomodel.sh

# 生成模型下载脚本
cat > "download-model.py" << 'EOF'
#!/usr/bin/env python3
"""
下载 Qwen3-VL-Embedding-2B 模型
"""

import os
from modelscope import snapshot_download

model_id = "qwen/Qwen3-VL-Embedding-2B"
cache_dir = "./models"

print(f"Downloading {model_id}...")
print(f"Target directory: {cache_dir}")

local_path = snapshot_download(model_id, cache_dir=cache_dir)
print(f"\nModel downloaded to: {local_path}")
print(f"\nYou can now run the container with:")
print(f"  docker run -d --gpus all -p 6008:6008 \\")
print(f"    -v $(pwd)/models/Qwen3-VL-Embedding-2B:/app/models/Qwen3-VL-Embedding-2B \\")
print(f"    embedding-api:qwen3vl-cuda-nomodel")
EOF
chmod +x download-model.py

echo ""
echo "========================================"
echo -e "${GREEN}  打包完成!${NC}"
echo "========================================"
echo ""
echo "输出文件:"
echo "  - $OUTPUT_FILE (镜像包, 约 4-6GB)"
echo "  - $OUTPUT_FILE.md5 (校验文件)"
echo "  - load-image-cuda-nomodel.sh (加载脚本)"
echo "  - download-model.py (模型下载脚本)"
echo ""
echo "模型下载:"
echo "  python3 download-model.py"
echo ""
echo "启动服务:"
echo "  docker run -d --gpus all -p 6008:6008 \\"
echo "    -v \$(pwd)/models/Qwen3-VL-Embedding-2B:/app/models/Qwen3-VL-Embedding-2B \\"
echo "    $IMAGE_NAME:$IMAGE_TAG"
echo ""
