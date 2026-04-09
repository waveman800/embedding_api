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
