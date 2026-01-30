#!/usr/bin/env python3
import numpy as np
import requests
import json

API_URL = "http://localhost:6008/v1/embeddings"
API_KEY = "sk-embedding-api-secret-key-20260114"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_similarity():
    """Test embedding similarity with similar and dissimilar texts."""
    
    # Test cases
    test_cases = [
        {
            "name": "相似文本测试",
            "text1": "人工智能是计算机科学的一个分支",
            "text2": "AI是计算机科学的重要领域",
            "expected_range": (0.7, 0.95)
        },
        {
            "name": "相同文本测试",
            "text1": "机器学习是人工智能的核心技术",
            "text2": "机器学习是人工智能的核心技术",
            "expected_range": (0.99, 1.0)
        },
        {
            "name": "不相似文本测试",
            "text1": "今天天气很好",
            "text2": "量子计算的基本原理",
            "expected_range": (0.0, 0.3)
        }
    ]
    
    print("=" * 80)
    print("嵌入向量相似度测试")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\n【{test_case['name']}】")
        print(f"文本1: {test_case['text1']}")
        print(f"文本2: {test_case['text2']}")
        
        # Generate embeddings
        payload1 = {
            "input": test_case['text1'],
            "model": "Qwen3-VL-Embedding-2B"
        }
        
        payload2 = {
            "input": test_case['text2'],
            "model": "Qwen3-VL-Embedding-2B"
        }
        
        try:
            response1 = requests.post(API_URL, json=payload1, headers=headers)
            response2 = requests.post(API_URL, json=payload2, headers=headers)
            
            if response1.status_code == 200 and response2.status_code == 200:
                embedding1 = np.array(response1.json()['data'][0]['embedding'])
                embedding2 = np.array(response2.json()['data'][0]['embedding'])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(embedding1, embedding2)
                
                print(f"相似度: {similarity:.4f}")
                print(f"预期范围: {test_case['expected_range'][0]:.2f} - {test_case['expected_range'][1]:.2f}")
                
                # Check if similarity is in expected range
                if test_case['expected_range'][0] <= similarity <= test_case['expected_range'][1]:
                    print("✓ 测试通过")
                else:
                    print("✗ 测试失败")
                    
            else:
                print(f"✗ API请求失败: {response1.status_code}, {response2.status_code}")
                
        except Exception as e:
            print(f"✗ 测试出错: {str(e)}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_similarity()
