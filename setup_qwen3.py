#!/usr/bin/env python3
"""
Qwen3 模型自动下载和配置脚本
支持多种Qwen3嵌入模型的自动下载、配置和测试
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import shutil

# 支持的模型配置
SUPPORTED_MODELS = {
    'qwen3-4b': {
        'model_id': 'Qwen/Qwen3-Embedding-4B',
        'target_dim': 1536,
        'description': 'RECOMMENDED: Qwen3-Embedding-4B, 4B params, best performance balance',
        'local_path': './models/Qwen3-Embedding-4B'
    },
    'qwen2-1.5b': {
        'model_id': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        'target_dim': 1536,
        'description': 'Lightweight high-performance model, 1536-dim embeddings',
        'local_path': './models/gte-Qwen2-1.5B-instruct'
    },
    'qwen2-7b': {
        'model_id': 'Alibaba-NLP/gte-Qwen2-7B-instruct',
        'target_dim': 4096,
        'description': 'Large high-quality model, 4096-dim embeddings',
        'local_path': './models/gte-Qwen2-7B-instruct'
    },
    'qwen2-large-en': {
        'model_id': 'Alibaba-NLP/gte-large-en-v1.5',
        'target_dim': 1024,
        'description': 'English-specific large model',
        'local_path': './models/gte-large-en-v1.5'
    },
    'qwen2-base-en': {
        'model_id': 'Alibaba-NLP/gte-base-en-v1.5',
        'target_dim': 768,
        'description': 'English-specific base model',
        'local_path': './models/gte-base-en-v1.5'
    }
}

def print_models():
    """Display all supported models"""
    print("Supported Qwen3 Models:")
    print("=" * 60)
    for key, config in SUPPORTED_MODELS.items():
        print(f"  {key:15} - {config['description']}")
        print(f"  {'':15}   Model ID: {config['model_id']}")
        print(f"  {'':15}   Dimensions: {config['target_dim']}")
        print()

def check_dependencies():
    """检查必要的依赖"""
    try:
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
        print("SUCCESS: All dependencies installed")
        return True
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Please run: pip install torch transformers sentence-transformers")
        return False

def download_model(model_key, custom_path=None):
    """下载指定的模型"""
    if model_key not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model '{model_key}'")
        print_models()
        return False
    
    config = SUPPORTED_MODELS[model_key]
    model_id = config['model_id']
    local_path = custom_path or config['local_path']
    
    print(f"开始下载模型: {config['description']}")
    print(f"模型ID: {model_id}")
    print(f"本地路径: {local_path}")
    print()
    
    try:
        # 创建本地目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 使用sentence-transformers下载模型
        from sentence_transformers import SentenceTransformer
        print("正在下载模型...")
        model = SentenceTransformer(model_id)
        
        # 保存到本地路径
        model.save(local_path)
        print(f"SUCCESS: 模型已下载到 {local_path}")
        
        # 测试模型
        test_texts = ["这是一个测试句子", "This is a test sentence"]
        embeddings = model.encode(test_texts)
        print(f"SUCCESS: 模型测试通过，嵌入维度: {embeddings.shape[1]}")
        
        return True, local_path, config['target_dim']
        
    except Exception as e:
        print(f"ERROR: 下载失败: {e}")
        return False, None, None

def update_env_file(model_path, target_dim):
    """更新.env文件配置"""
    env_file = '.env'
    env_example = '.env.example'
    
    # 如果.env不存在，从.env.example复制
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print(f"已从 {env_example} 创建 {env_file}")
        else:
            print("WARNING: .env.example 文件不存在，无法自动创建配置")
            return False
    
    # 读取现有配置
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 更新配置
    updated_lines = []
    model_path_updated = False
    target_dim_updated = False
    
    for line in lines:
        if line.startswith('MODEL_PATH='):
            updated_lines.append(f'MODEL_PATH={model_path}\n')
            model_path_updated = True
        elif line.startswith('TARGET_DIM='):
            updated_lines.append(f'TARGET_DIM={target_dim}\n')
            target_dim_updated = True
        else:
            updated_lines.append(line)
    
    # 如果没有找到配置项，添加到文件末尾
    if not model_path_updated:
        updated_lines.append(f'MODEL_PATH={model_path}\n')
    if not target_dim_updated:
        updated_lines.append(f'TARGET_DIM={target_dim}\n')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"SUCCESS: 已更新 {env_file} 配置")
    print(f"  模型路径: {model_path}")
    print(f"  目标维度: {target_dim}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Qwen3模型自动下载和配置工具')
    parser.add_argument('model', nargs='?', help='模型类型 (qwen3-4b, qwen2-1.5b, qwen2-7b, etc.)')
    parser.add_argument('--path', help='自定义模型保存路径')
    parser.add_argument('--list', action='store_true', help='显示所有支持的模型')
    parser.add_argument('--no-env', action='store_true', help='不更新.env文件')
    
    args = parser.parse_args()
    
    if args.list or not args.model:
        print_models()
        if not args.model:
            return
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 下载模型
    success, model_path, target_dim = download_model(args.model, args.path)
    if not success:
        return
    
    # 更新.env文件
    if not args.no_env:
        update_env_file(model_path, target_dim)
    
    print()
    print("=" * 60)
    print("SUCCESS: 设置完成！")
    print()
    print("下一步:")
    print("1. 检查 .env 文件配置")
    print("2. 启动服务: python embedding.py")
    print("3. 测试服务: curl http://localhost:6008/health")

if __name__ == "__main__":
    main()
