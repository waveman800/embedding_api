import requests
import json

# 测试健康检查
print("测试健康检查...")
response = requests.get("http://localhost:6008/health")
print(f"健康检查: {response.status_code} - {response.json()}")

# 测试状态检查
print("\n测试状态检查...")
response = requests.get("http://localhost:6008/status")
print(f"状态检查: {response.status_code} - {response.json()}")

# 测试嵌入API（不使用认证）
print("\n测试嵌入API...")
data = {
    "input": ["hello world"],
    "model": "bge-m3"
}

try:
    response = requests.post(
        "http://localhost:6008/v1/embeddings", 
        json=data,
        timeout=10
    )
    print(f"嵌入API: {response.status_code}")
    if response.status_code != 200:
        print(f"错误响应: {response.text}")
    else:
        result = response.json()
        print(f"成功! 嵌入维度: {len(result['data'][0]['embedding'])}")
except Exception as e:
    print(f"请求异常: {e}")
