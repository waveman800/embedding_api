#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service Startup Script
Check environment and start the embedding service
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists"""
    env_file = Path('.env')
    if not env_file.exists():
        print("ERROR: .env file not found")
        print("Run 'python setup_env.py' first to create .env file")
        return False
    
    print("Found .env file")
    return True

def check_model():
    """Check if model directory exists"""
    # Try to read model path from .env
    model_path = os.getenv('MODEL_PATH', './models/Qwen3-Embedding-4B')
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please ensure the model is downloaded or update MODEL_PATH in .env")
        return False
    
    print(f"Found model at {model_path}")
    return True

def check_dependencies():
    """Check required Python packages"""
    required_packages = [
        'torch',
        'transformers', 
        'flask',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nERROR: Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def start_service():
    """Start the embedding service"""
    print("\nStarting embedding service...")
    
    try:
        # Start the service
        subprocess.run([sys.executable, 'embedding.py'], check=True)
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Service failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"Error starting service: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("Embedding Service Startup Tool")
    print("=" * 35)
    
    # Check prerequisites
    if not check_env_file():
        sys.exit(1)
    
    if not check_model():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\nAll checks passed!")
    
    # Start service
    start_service()

if __name__ == "__main__":
    main()
