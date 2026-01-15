# coding=utf-8
import time
import torch
import uvicorn
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageInput(BaseModel):
    type: str = "image"
    data: str  # Base64 encoded image data

class TextInput(BaseModel):
    type: str = "text"
    data: str

class EmbeddingProcessRequest(BaseModel):
    input: Union[List[str], List[Union[ImageInput, TextInput]]]
    model: str


class EmbeddingQuestionRequest(BaseModel):
    input: Union[str, ImageInput, TextInput]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


async def verify_token(request: Request):
    auth_header = request.headers.get('Authorization')
    # 从环境变量获取API密钥，如果未设置则使用默认值
    api_key = os.environ.get('API_KEY', 'sk-hv6xtPbK183j3RR306Fe23B6196b4d919a8e854887F6213d')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if token_type.lower() == "bearer" and token == api_key:
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    try:
        # Remove data URI prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_str)
        
        # Convert to PIL Image
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"Failed to decode image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def process_input(input_data):
    """Process input data and convert to model-compatible format."""
    if isinstance(input_data, str):
        return [input_data]
    elif isinstance(input_data, list):
        processed = []
        for item in input_data:
            if isinstance(item, str):
                processed.append(item)
            elif isinstance(item, dict) or hasattr(item, 'type'):
                if getattr(item, 'type', item.get('type', '')) == 'text':
                    processed.append(getattr(item, 'data', item.get('data', '')))
                elif getattr(item, 'type', item.get('type', '')) == 'image':
                    image = decode_base64_image(getattr(item, 'data', item.get('data', '')))
                    processed.append(image)
        return processed
    elif isinstance(input_data, dict) or hasattr(input_data, 'type'):
        if getattr(input_data, 'type', input_data.get('type', '')) == 'text':
            return [getattr(input_data, 'data', input_data.get('data', ''))]
        elif getattr(input_data, 'type', input_data.get('type', '')) == 'image':
            image = decode_base64_image(getattr(input_data, 'data', input_data.get('data', '')))
            return [image]
    return []

def expand_features(embedding, target_length):  #使用了正则来归一向量
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        expanded_embedding = np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
    normalizer = Normalizer(norm='l2')
    expanded_embedding = normalizer.transform(expanded_embedding.reshape(1, -1)).flatten()
    return expanded_embedding


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: Union[EmbeddingProcessRequest, EmbeddingQuestionRequest]):
    if isinstance(request, EmbeddingProcessRequest):
        print('EmbeddingProcessRequest')
        input_data = request.input
    elif isinstance(request, EmbeddingQuestionRequest):
        print('EmbeddingQuestionRequest')
        input_data = request.input
    else:
        print('Request')
        data = request.json()
        print(data)
        return

    # Process input
    print(f"Processing input...")
    processed_input = process_input(input_data)
    
    # Generate embeddings
    embeddings = [embeddings_model.encode(item) for item in processed_input]
    
    # 从环境变量获取目标维度，默认为2560
    target_dimension = int(os.environ.get('EMBEDDING_DIMENSION', '2560'))
    print(f"Using embedding dimension: {target_dimension}")
    
    # 如果嵌入向量的维度不为目标维度，则扩展至目标维度
    embeddings = [expand_features(embedding, target_dimension) if len(embedding) != target_dimension else embedding for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    
    # Calculate token usage (approximate)
    prompt_tokens = 0
    total_tokens = 0
    for item in processed_input:
        if isinstance(item, str):
            prompt_tokens += len(item.split())
            total_tokens += num_tokens_from_string(item)
        else:  # For images
            prompt_tokens += 100  # Approximate word count for images
            total_tokens += 100  # Approximate token count for images

    response = {
        "data": [
            {
                "embedding": embedding,
                "index": index,
                "object": "embedding"
            } for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
    }

    return response

if __name__ == "__main__":
    # 从环境变量获取配置
    model_path = os.environ.get('MODEL_PATH', '/app/models/bge-large-zh-v1.5')
    device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    port = int(os.environ.get('PORT', '6008'))
    embedding_dimension = int(os.environ.get('EMBEDDING_DIMENSION', '2560'))
    
    print("=== 服务配置 ===")
    print(f"模型路径: {model_path}")
    print(f"计算设备: {device}")
    print(f"服务端口: {port}")
    print(f"嵌入维度: {embedding_dimension}")
    
    print(f"Loading model from {model_path} using device {device}")
    embeddings_model = SentenceTransformer(model_path, device=device)
    print(f"Model loaded successfully, starting server on port {port}")
    
    uvicorn.run(app, host='0.0.0.0', port=port)
