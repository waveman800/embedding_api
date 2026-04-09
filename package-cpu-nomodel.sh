#!/bin/bash
# 打包 Qwen3-VL-Embedding-2B CPU 版本 Docker 镜像（模型外挂版本）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Embedding API CPU 镜像打包脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查必要文件
echo -e "${BLUE}检查必要文件...${NC}"
if [ ! -f "Dockerfile.cpu-nomodel" ]; then
    echo -e "${RED}错误: Dockerfile.cpu-nomodel 不存在${NC}"
    exit 1
fi

if [ ! -f "embedding.py" ]; then
    echo -e "${RED}错误: embedding.py 不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 文件检查通过${NC}"
echo ""

# 构建镜像
IMAGE_NAME="embedding-api:qwen3vl-cpu-nomodel"
IMAGE_FILE="embedding-api-qwen3vl-cpu-nomodel.tar.gz"

echo -e "${BLUE}开始构建镜像...${NC}"
echo "镜像名称: $IMAGE_NAME"
echo "Dockerfile: Dockerfile.cpu-nomodel"
echo ""

docker build -f Dockerfile.cpu-nomodel -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 镜像构建失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 镜像构建成功${NC}"
echo ""

# 保存镜像
echo -e "${BLUE}保存镜像到文件...${NC}"
echo "输出文件: $IMAGE_FILE"
docker save $IMAGE_NAME | gzip > $IMAGE_FILE

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 镜像保存失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 镜像保存成功${NC}"
echo ""

# 计算文件大小和 MD5
FILE_SIZE=$(du -h $IMAGE_FILE | cut -f1)
MD5_SUM=$(md5sum $IMAGE_FILE | cut -d' ' -f1)

echo "$MD5_SUM  $IMAGE_FILE" > ${IMAGE_FILE}.md5

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  打包完成!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "镜像信息:"
echo "  - 镜像名称: $IMAGE_NAME"
echo "  - 文件大小: $FILE_SIZE"
echo "  - 输出文件: $IMAGE_FILE"
echo "  - MD5 校验: $MD5_SUM"
echo "  - 校验文件: ${IMAGE_FILE}.md5"
echo ""
echo -e "${YELLOW}注意: 此镜像不包含模型文件${NC}"
echo "部署时需要:"
echo "  1. 准备 .env 配置文件"
echo "  2. 下载模型到 models/Qwen3-VL-Embedding-2B/"
echo "  3. 使用 docker-compose.cpu-nomodel.yml 启动"
echo ""
