#!/usr/bin/env python3
"""
快速禁用批处理脚本
将ENABLE_BATCH_PROCESSING设置为false，实现单个请求立即处理
"""

import os
import shutil
from pathlib import Path

def update_env_disable_batch():
    """更新.env文件，禁用批处理"""
    env_file = '.env'
    env_example = '.env.example'
    
    # 如果.env不存在，从.env.example复制
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print(f"已从 {env_example} 创建 {env_file}")
        else:
            print("ERROR: .env.example 文件不存在")
            return False
    
    # 读取现有配置
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 更新配置
    updated_lines = []
    batch_processing_updated = False
    
    for line in lines:
        if line.startswith('ENABLE_BATCH_PROCESSING='):
            updated_lines.append('ENABLE_BATCH_PROCESSING=false\n')
            batch_processing_updated = True
        else:
            updated_lines.append(line)
    
    # 如果没有找到配置项，添加到文件末尾
    if not batch_processing_updated:
        updated_lines.append('ENABLE_BATCH_PROCESSING=false\n')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print("SUCCESS: 已禁用批处理模式")
    print("配置更新:")
    print("  ENABLE_BATCH_PROCESSING=false")
    print("\n现在请重启服务以应用更改:")
    print("  python embedding.py")
    return True

if __name__ == "__main__":
    print("禁用批处理模式 - 实现单个请求立即处理")
    print("=" * 50)
    
    if update_env_disable_batch():
        print("\nSUCCESS: Batch processing disabled!")
        print("After restarting the service, each request will be processed immediately.")
    else:
        print("\nERROR: Configuration update failed")
