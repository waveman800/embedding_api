#!/usr/bin/env python3
"""
测试直接处理模式
验证单个请求是否立即处理，无需等待批处理
"""

import asyncio
import time
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

async def test_direct_processing():
    """测试直接处理模式的响应速度"""
    try:
        # 导入embedding模块
        import embedding
        
        print("Testing direct processing mode...")
        print("=" * 50)
        
        # 测试单个请求
        test_texts = ["This is a test sentence for embedding."]
        
        print(f"Processing {len(test_texts)} text(s)...")
        start_time = time.time()
        
        # 调用嵌入函数
        result = await embedding.get_embeddings_async(test_texts)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processing completed in {processing_time:.3f} seconds")
        print(f"Result shape: {len(result)} embeddings")
        print(f"Embedding dimension: {len(result[0]) if result else 'N/A'}")
        
        # 测试多个单独请求
        print("\nTesting multiple individual requests...")
        individual_times = []
        
        for i in range(3):
            start_time = time.time()
            result = await embedding.get_embeddings_async([f"Test sentence {i+1}"])
            end_time = time.time()
            individual_times.append(end_time - start_time)
            print(f"Request {i+1}: {individual_times[-1]:.3f}s")
        
        avg_time = sum(individual_times) / len(individual_times)
        print(f"Average individual request time: {avg_time:.3f}s")
        
        print("\nSUCCESS: Direct processing mode is working!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("Direct Processing Mode Test")
    print("=" * 50)
    
    success = await test_direct_processing()
    
    if success:
        print("\nTest completed successfully!")
        print("Each request was processed immediately without batching delay.")
    else:
        print("\nTest failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
