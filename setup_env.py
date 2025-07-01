#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Setup Script
Automatically create .env file from .env.example
"""

import os
import shutil
from pathlib import Path

def setup_env():
    """Setup environment configuration"""
    print("Setting up environment configuration...")
    
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if not env_example.exists():
        print("ERROR: .env.example file not found")
        return False
    
    if env_file.exists():
        print("INFO: .env file already exists")
        choice = input("Overwrite existing .env file? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("Setup cancelled")
            return False
    
    # Copy .env.example to .env
    shutil.copy2(env_example, env_file)
    print(f"Created .env file from {env_example}")
    
    print("\nPlease review and modify the .env file as needed:")
    print("- Set your API_KEY")
    print("- Adjust MODEL_PATH if needed")
    print("- Configure other parameters")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("WARNING: PyTorch not installed")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("WARNING: Transformers not installed")
    
    try:
        import flask
        print(f"Flask: {flask.__version__}")
    except ImportError:
        print("WARNING: Flask not installed")

def main():
    """Main function"""
    print("Environment Setup Tool")
    print("=" * 30)
    
    if setup_env():
        check_dependencies()
        print("\nSetup completed!")
        print("Run 'python start_service.py' to start the service")
    else:
        print("Setup failed!")

if __name__ == "__main__":
    main()
