#!/usr/bin/env python3
"""
Docker Compose 快速部署脚本
支持GPU/CPU模式，单实例/多实例部署
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """运行命令并返回结果"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None

def check_requirements():
    """检查部署要求"""
    print("Checking deployment requirements...")
    
    # 检查Docker
    result = run_command("docker --version", check=False)
    if not result or result.returncode != 0:
        print("ERROR: Docker not found. Please install Docker first.")
        return False
    
    # 检查Docker Compose
    result = run_command("docker-compose --version", check=False)
    if not result or result.returncode != 0:
        print("ERROR: Docker Compose not found. Please install Docker Compose first.")
        return False
    
    # 检查NVIDIA Docker（如果需要GPU）
    if sys.platform.startswith('linux'):
        result = run_command("nvidia-docker --version", check=False)
        if result and result.returncode == 0:
            print("NVIDIA Docker detected - GPU support available")
        else:
            print("WARNING: NVIDIA Docker not found - GPU support may not work")
    
    return True

def setup_environment():
    """设置环境配置"""
    print("Setting up environment...")
    
    # 检查.env文件
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("Created .env from .env.example")
        else:
            print("ERROR: .env.example not found")
            return False
    
    # 检查模型目录
    model_path = Path('models/Qwen3-Embedding-4B')
    if not model_path.exists():
        print("WARNING: Qwen3-Embedding-4B model not found")
        print("Please run: python setup_qwen3.py qwen3-4b")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True

def deploy_single(mode='gpu'):
    """部署单实例"""
    print(f"Deploying single instance ({mode} mode)...")
    
    if mode == 'cpu':
        compose_file = 'docker-compose.cpu.yml'
    else:
        compose_file = 'docker-compose.yml'
    
    if not os.path.exists(compose_file):
        print(f"ERROR: {compose_file} not found")
        return False
    
    # 停止现有服务
    run_command(f"docker-compose -f {compose_file} down", check=False)
    
    # 构建并启动
    result = run_command(f"docker-compose -f {compose_file} up -d --build")
    if not result:
        return False
    
    print("Deployment completed!")
    print(f"Service available at: http://localhost:6008")
    print(f"Check logs: docker-compose -f {compose_file} logs -f")
    
    return True

def deploy_scale():
    """部署多实例（负载均衡）"""
    print("Deploying scaled instances with load balancer...")
    
    compose_file = 'docker-compose.scale.yml'
    if not os.path.exists(compose_file):
        print(f"ERROR: {compose_file} not found")
        return False
    
    # 停止现有服务
    run_command(f"docker-compose -f {compose_file} down", check=False)
    
    # 构建并启动
    result = run_command(f"docker-compose -f {compose_file} up -d --build")
    if not result:
        return False
    
    print("Scaled deployment completed!")
    print("Services available at:")
    print("  - Load Balancer: http://localhost:80")
    print("  - Monitoring: http://localhost:3000 (Grafana)")
    print("  - Metrics: http://localhost:9090 (Prometheus)")
    print(f"Check logs: docker-compose -f {compose_file} logs -f")
    
    return True

def test_deployment():
    """测试部署"""
    print("Testing deployment...")
    
    # 等待服务启动
    import time
    print("Waiting for service to start...")
    time.sleep(10)
    
    # 健康检查
    result = run_command("curl -f http://localhost:6008/status", check=False)
    if result and result.returncode == 0:
        print("SUCCESS: Service is healthy!")
        return True
    else:
        print("WARNING: Service health check failed")
        print("Check logs: docker-compose logs -f")
        return False

def main():
    parser = argparse.ArgumentParser(description='Docker Compose Deployment Script')
    parser.add_argument('--mode', choices=['gpu', 'cpu'], default='gpu',
                       help='Deployment mode (default: gpu)')
    parser.add_argument('--scale', action='store_true',
                       help='Deploy multiple instances with load balancer')
    parser.add_argument('--test', action='store_true',
                       help='Test deployment after startup')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip requirement checks')
    
    args = parser.parse_args()
    
    print("Docker Compose Deployment Script")
    print("=" * 50)
    
    # 检查要求
    if not args.skip_checks and not check_requirements():
        sys.exit(1)
    
    # 设置环境
    if not setup_environment():
        sys.exit(1)
    
    # 部署
    success = False
    if args.scale:
        success = deploy_scale()
    else:
        success = deploy_single(args.mode)
    
    if not success:
        print("Deployment failed!")
        sys.exit(1)
    
    # 测试
    if args.test:
        test_deployment()
    
    print("\nDeployment completed successfully!")
    print("\nUseful commands:")
    print("  docker-compose ps                    # Check service status")
    print("  docker-compose logs -f               # View logs")
    print("  docker-compose down                  # Stop services")
    print("  docker-compose restart               # Restart services")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Deployment failed with error: {e}")
        sys.exit(1)
