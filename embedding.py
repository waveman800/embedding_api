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
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import PolynomialFeatures
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingProcessRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingQuestionRequest(BaseModel):
    input: str
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
    # data = await request.body()
    # print(data)
    # return
    if isinstance(request, EmbeddingProcessRequest):
        print('EmbeddingProcessRequest')
        payload = request.input
    elif isinstance(request, EmbeddingQuestionRequest):
        print('EmbeddingQuestionRequest')
        payload = [request.input]
    else:
        print('Request')
        data = request.json()
        print(data)
        return

    print(payload)
    # Process embeddings in batches for better performance
    batch_size = 24  # Adjust based on your GPU memory
    embeddings = []
    for i in range(0, len(payload), batch_size):
        batch = payload[i:i + batch_size]
        batch_embeddings = embeddings_model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    # 从环境变量获取目标维度，默认为2560
    target_dimension = int(os.environ.get('EMBEDDING_DIMENSION', '2560'))
    print(f"Using embedding dimension: {target_dimension}")
    # 如果嵌入向量的维度不为目标维度，则扩展至目标维度
    embeddings = [expand_features(embedding, target_dimension) if len(embedding) != target_dimension else embedding for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in payload)
    total_tokens = sum(num_tokens_from_string(text) for text in payload)

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
    model_path = os.environ.get('MODEL_PATH', '/app/models/bge-m3')
    device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    port = int(os.environ.get('PORT', '6008'))
    embedding_dimension = int(os.environ.get('EMBEDDING_DIMENSION', '2560'))
    
    print("=== 服务配置 ===")
    print(f"模型路径: {model_path}")
    print(f"计算设备: {device}")
    print(f"服务端口: {port}")
    print(f"嵌入维度: {embedding_dimension}")
    
    print(f"Loading model from {model_path} with tensor parallelism")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) found. For best performance, use at least 2 GPUs.")
    
    # Load model with device_map='auto' for tensor parallelism
    print(f"Using {num_gpus} GPU(s) for tensor parallelism")
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize SentenceTransformer with the parallelized model
    embeddings_model = SentenceTransformer(model_path)
    embeddings_model[0].auto_model = model
    embeddings_model[0].tokenizer = tokenizer
    
    print(f"Model loaded successfully across {num_gpus} GPU(s), starting server on port {port}")
    
    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)
