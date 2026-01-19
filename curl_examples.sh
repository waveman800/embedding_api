#!/bin/bash

# Qwen3 Embedding API - Curl请求示例

# API配置
API_URL="http://localhost:6008/v1/embeddings"
API_KEY="sk-embedding-api-secret-key-20260114"

# ===========================================
# 1. 纯文本嵌入请求
# ===========================================
echo "\n=== 1. 纯文本嵌入请求 ==="
curl -X POST "$API_URL" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": ["你好啊我的朋友", "这是一个测试句子", "Qwen3 Embedding API 测试"], "model": "Qwen3-VL-Embedding-2B"}'

# ===========================================
# 2. 纯图片嵌入请求
# 注意：需要先将图片转换为base64编码
# ===========================================
echo "\n\n=== 2. 纯图片嵌入请求 ==="
echo "请先准备一张图片并转换为base64编码，然后替换下面的<base64_image_data>部分"

# 示例（需要替换实际的base64数据）
# curl -X POST "$API_URL" \
#   -H "Authorization: Bearer $API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"input": ["<base64_image_data>"], "model": "Qwen3-VL-Embedding-2B"}'

# 或者使用结构化格式：
# curl -X POST "$API_URL" \
#   -H "Authorization: Bearer $API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"input": [{"type": "image", "data": "<base64_image_data>"}], "model": "Qwen3-VL-Embedding-2B"}'

# 实际使用时，可以通过以下命令将图片转换为base64并发送请求：
echo "\n实际使用示例命令："
echo "base64 image.jpg | curl -X POST '$API_URL' -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' -d '{\"input\": [\"@-\"], \"model\": \"Qwen3-VL-Embedding-2B\"}'"

# ===========================================
# 3. 图文混合嵌入请求
# ===========================================
echo "\n\n=== 3. 图文混合嵌入请求 ==="
echo "请先准备一张图片并转换为base64编码，然后替换下面的<base64_image_data>部分"

# 示例（需要替换实际的base64数据）
# curl -X POST "$API_URL" \
#   -H "Authorization: Bearer $API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"input": [
#     {"type": "text", "data": "这张图片展示了什么？"},
#     {"type": "image", "data": "<base64_image_data>"},
#     {"type": "text", "data": "请描述图片内容"}
#   ], "model": "Qwen3-VL-Embedding-2B"}'

# 实际使用时，可以通过以下方式组合文本和图片：
echo "\n实际使用示例（使用文件中的base64数据）："
echo "curl -X POST '$API_URL' \
  -H 'Authorization: Bearer $API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{\"input\": [
    {\"type\": \"text\", \"data\": \"这张图片展示了什么？\"},
    {\"type\": \"image\", \"data\": \"@image_base64.txt\"},
    {\"type\": \"text\", \"data\": \"请描述图片内容\"}
  ], \"model\": \"Qwen3-VL-Embedding-2B\"}'"

# ===========================================
# 4. 简化的图文混合请求（使用base64前缀）
# ===========================================
echo "\n\n=== 4. 简化的图文混合请求（使用base64前缀） ==="
echo "图片数据可以包含data:image/前缀"

# 示例（需要替换实际的base64数据）
# curl -X POST "$API_URL" \
#   -H "Authorization: Bearer $API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{"input": [
#     "这是图片描述",
#     "data:image/jpeg;base64,<base64_image_data>"
#   ], "model": "Qwen3-VL-Embedding-2B"}'

# ===========================================
# 5. 单文本请求（最小化示例）
# ===========================================
echo "\n\n=== 5. 单文本请求（最小化示例） ==="
curl -X POST "$API_URL" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "单个文本嵌入测试"}'

echo "\n"
