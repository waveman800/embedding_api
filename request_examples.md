# Qwen3 Embedding API 请求示例

## 1. 包含 model 参数的文本嵌入请求
```http
POST /v1/embeddings HTTP/1.1
Host: localhost:6008
Authorization: Bearer sk-embedding-api-secret-key-20260114
Content-Type: application/json
Content-Length: 82

{
  "input": ["你好啊我的朋友", "这是一个测试句子"],
  "model": "Qwen3-VL-Embedding-2B"
}
```

## 2. 包含 model 参数的图像嵌入请求
```http
POST /v1/embeddings HTTP/1.1
Host: localhost:6008
Authorization: Bearer sk-embedding-api-secret-key-20260114
Content-Type: application/json
Content-Length: [实际内容长度]

{
  "input": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/..."],
  "model": "Qwen3-VL-Embedding-2B"
}
```

## 3. 使用结构化输入格式的请求
```http
POST /v1/embeddings HTTP/1.1
Host: localhost:6008
Authorization: Bearer sk-embedding-api-secret-key-20260114
Content-Type: application/json
Content-Length: 168

{
  "input": [
    {
      "type": "text",
      "data": "这是文本输入"
    },
    {
      "type": "text",
      "data": "这是另一个文本输入"
    }
  ],
  "model": "Qwen3-VL-Embedding-2B"
}
```

## 4. 混合文本和图像的请求（仅VL模型支持）
```http
POST /v1/embeddings HTTP/1.1
Host: localhost:6008
Authorization: Bearer sk-embedding-api-secret-key-20260114
Content-Type: application/json
Content-Length: [实际内容长度]

{
  "input": [
    {
      "type": "text",
      "data": "这张图片展示了什么？"
    },
    {
      "type": "image",
      "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/..."
    }
  ],
  "model": "Qwen3-VL-Embedding-2B"
}
```

## 5. 图片URL输入请求（仅VL模型支持）
```http
POST /v1/embeddings HTTP/1.1
Host: localhost:6008
Authorization: Bearer sk-embedding-api-secret-key-20260114
Content-Type: application/json
Content-Length: 180

{
  "input": [
    {
      "type": "image",
      "data": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=100&h=100&fit=crop"
    }
  ],
  "model": "Qwen3-VL-Embedding-2B"
}
```

## 6. 不包含 model 参数的请求（使用默认模型）
```http
POST /v1/embeddings HTTP/1.1
Host: localhost:6008
Authorization: Bearer sk-embedding-api-secret-key-20260114
Content-Type: application/json
Content-Length: 36

{"input": ["你好啊我的朋友"]}
```

## 响应示例
```http
HTTP/1.1 200 OK
Date: [当前日期]
Content-Type: application/json
Content-Length: [实际内容长度]

{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, 0.0789, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.1011, -0.1213, 0.1415, ...],
      "index": 1
    }
  ],
  "model": "Qwen3-VL-Embedding-2B",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15
  }
}
```

## 注意事项
1. **model 参数说明**：
   - 可以在请求中指定 `model` 参数，也可以不指定
   - 如果不指定，会使用环境变量中配置的默认模型名称 (`MODEL_NAME`)
   - 当前环境变量中配置的默认模型是：`Qwen3-VL-Embedding-2B`

2. **API 密钥**：
   - 固定为：`sk-embedding-api-secret-key-20260114`
   - 必须以 `Bearer ` 开头

3. **图像输入**：
   - 支持三种格式：
     - Base64 编码字符串（带或不带 `data:image/[格式];base64,` 前缀）
     - 图片 URL（以 http:// 或 https:// 开头）
     - 直接的 Base64 数据
   - 仅支持 VL (Vision-Language) 模型

4. **模型限制**：
   - 当前服务端只加载了一个模型（环境变量中配置的那个）
   - 即使请求中指定了不同的模型名称，服务端仍然会使用已加载的模型
   - 要使用不同的模型，需要修改环境变量并重启服务
