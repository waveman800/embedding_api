#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker Build Script for Embedding API with Qwen3 Model
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_files():
    """Check required files"""
    print("Checking required files...")
    
    # Check model directory
    model_path = Path('./models/Qwen3-Embedding-4B')
    if not model_path.exists():
        print(f"ERROR: Model directory not found: {model_path}")
        return False
    
    # Check Dockerfile
    dockerfile_path = Path('./Dockerfile.cn')
    if not dockerfile_path.exists():
        print("ERROR: Dockerfile.cn not found")
        return False
    
    print("All required files found")
    return True

def build_image():
    """Build Docker image"""
    print("Building Docker image...")
    
    cmd = [
        'docker', 'build',
        '-f', 'Dockerfile.cn',
        '-t', 'embedding-api-qwen3:latest',
        '.'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with exit code: {e.returncode}")
        return False

def main():
    """Main function"""
    print("Docker Build Tool for Embedding API")
    print("=" * 40)
    
    if not check_files():
        sys.exit(1)
    
    if build_image():
        print("\nBuild completed successfully!")
        print("Run with: docker run -d -p 6008:6008 --gpus all embedding-api-qwen3:latest")
    else:
        print("\nBuild failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
