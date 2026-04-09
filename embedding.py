# coding=utf-8
import time
import torch
import uvicorn
import os
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from starlette.status import HTTP_401_UNAUTHORIZED
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from typing import Union
from transformers import AutoProcessor, AutoModel

# 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    model: str = "Qwen3-VL-Embedding-2B"


class EmbeddingResponse(BaseModel):
    data: List[Dict[str, Any]]
    model: str
    object: str
    usage: Dict[str, int]


# 全局模型变量
processor = None
model = None


def load_model():
    """加载 Qwen3-VL-Embedding-2B 模型"""
    global processor, model
    
    model_path = os.environ.get('MODEL_PATH', '/app/models/Qwen3-VL-Embedding-2B')
    device = os.environ.get('DEVICE', 'cpu')
    
    print(f"Loading Qwen3-VL-Embedding-2B from {model_path}")
    print(f"Device: {device}")
    
    # 加载处理器和模型
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.eval()
    
    print("Model loaded successfully!")
    return processor, model


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    try:
        # Remove data URI prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_str)
        
        # Convert to PIL Image
        return Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        print(f"Failed to decode image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")


def process_input_item(item):
    """处理单个输入项，返回 (text, image) 元组"""
    text = None
    image = None
    
    if isinstance(item, str):
        # 纯文本
        text = item
    elif isinstance(item, dict):
        # 字典格式
        item_type = item.get('type', 'text')
        if item_type == 'text':
            text = item.get('data', '')
        elif item_type == 'image':
            image = decode_base64_image(item.get('data', ''))
    
    return text, image


def get_embedding_for_text(text: str) -> np.ndarray:
    """为文本生成嵌入向量"""
    global processor, model
    
    # 处理输入
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # 移动到设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 生成嵌入
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 mean pooling
        if hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state.mean(dim=1)[0]
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output[0]
        else:
            embedding = outputs[0][0]
    
    # 转换为 numpy
    embedding = embedding.cpu().numpy()
    
    return embedding


async def verify_token(request: Request):
    auth_header = request.headers.get('Authorization')
    api_key = os.environ.get('API_KEY', 'sk-hv6xtPbK183j3RR306Fe23B6196b4d919a8e854887F6213d')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if token_type.lower() == "bearer" and token == api_key:
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": os.environ.get('MODEL_PATH', 'unknown'),
        "device": os.environ.get('DEVICE', 'cpu'),
        "version": "1.0.0",
        "model_type": "Qwen3-VL-Embedding-2B",
        "model_loaded": model is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Embedding API",
        "version": "1.0.0",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "health": "/health"
        },
        "model": "Qwen3-VL-Embedding-2B",
        "device": os.environ.get('DEVICE', 'cpu'),
        "dimension": int(os.environ.get('EMBEDDING_DIMENSION', '2048'))
    }


@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest, authorized: bool = Depends(verify_token)):
    global processor, model
    
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 处理输入数据
        input_data = request.input
        
        # 统一转换为列表
        if isinstance(input_data, str):
            # 单条文本
            texts = [input_data]
        elif isinstance(input_data, list):
            # 列表格式
            texts = []
            for item in input_data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    # 处理 dict 类型，提取文本
                    item_type = item.get('type', 'text')
                    if item_type == 'text':
                        texts.append(item.get('data', ''))
                    else:
                        # 暂时不支持图像，使用空文本
                        texts.append('')
                else:
                    texts.append(str(item))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported input type: {type(input_data)}")
        
        # 生成嵌入
        embeddings = []
        total_tokens = 0
        
        for text in texts:
            # 计算近似 token 数 (简单分词)
            tokens = len(text.split())
            total_tokens += tokens
            
            # 生成嵌入
            embedding = get_embedding_for_text(text)
            embeddings.append(embedding)
        
        # 获取目标维度
        target_dimension = int(os.environ.get('EMBEDDING_DIMENSION', '2048'))
        
        # 调整维度并归一化
        processed_embeddings = []
        for embedding in embeddings:
            # 调整维度
            if len(embedding) != target_dimension:
                if len(embedding) > target_dimension:
                    embedding = embedding[:target_dimension]
                else:
                    embedding = np.pad(embedding, (0, target_dimension - len(embedding)))
            
            # L2 归一化
            norm_val = np.linalg.norm(embedding)
            if norm_val > 0:
                embedding = embedding / norm_val
            
            processed_embeddings.append(embedding.tolist())
        
        # 构建响应
        response_data = {
            "data": [
                {
                    "embedding": emb,
                    "index": idx,
                    "object": "embedding"
                } for idx, emb in enumerate(processed_embeddings)
            ],
            "model": request.model,
            "object": "list",
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
        return response_data
        
    except Exception as e:
        import traceback
        print(f"Error processing request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 从环境变量获取配置
    model_path = os.environ.get('MODEL_PATH', '/app/models/Qwen3-VL-Embedding-2B')
    device = os.environ.get('DEVICE', 'cpu')
    port = int(os.environ.get('PORT', '6008'))
    embedding_dimension = int(os.environ.get('EMBEDDING_DIMENSION', '2048'))
    
    print("=" * 50)
    print("Embedding API with Qwen3-VL-Embedding-2B")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print(f"计算设备: {device}")
    print(f"服务端口: {port}")
    print(f"嵌入维度: {embedding_dimension}")
    print("=" * 50)
    
    # 加载模型
    try:
        load_model()
        print(f"Model loaded successfully!")
        print(f"Starting server on port {port}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 监听所有网络接口，允许外部访问
    host = os.environ.get('HOST', '0.0.0.0')
    uvicorn.run(app, host=host, port=port)
