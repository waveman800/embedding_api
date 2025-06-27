# coding=utf-8
import time
import torch
import uvicorn
import os
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from datetime import datetime, timedelta

app = FastAPI(title="Embedding API", 
             description="高性能的文本嵌入向量服务",
             version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# 限流中间件
@app.middleware("http")
async def rate_limiter_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    async with rate_limit_lock:
        now = datetime.utcnow()
        
        # 清理过期的请求记录
        if client_ip in rate_limit_requests:
            rate_limit_requests[client_ip] = [
                t for t in rate_limit_requests[client_ip] 
                if now - t < rate_limit_window
            ]
        else:
            rate_limit_requests[client_ip] = []
        
        # 检查是否超过限制
        if len(rate_limit_requests[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"}
            )
        
        # 添加当前请求
        rate_limit_requests[client_ip].append(now)
    
    response = await call_next(request)
    return response


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
    # 跳过健康检查的认证
    if request.url.path == "/health":
        return True
        
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if token_type.lower() == "bearer" and token == API_KEY:
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

def expand_features(embedding, target_length=None):
    """
    扩展特征维度并归一化
    :param embedding: 输入嵌入向量
    :param target_length: 目标维度，如果为None则保持原维度
    :return: 处理后的嵌入向量
    """
    # 如果不需要扩展维度，直接返回归一化后的向量
    if target_length is None or len(embedding) >= target_length:
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        normalizer = Normalizer(norm='l2')
        return normalizer.transform(embedding).flatten()
    
    # 使用多项式特征扩展
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1)).flatten()
    
    # 如果扩展后的维度仍然小于目标维度，使用插值填充
    if len(expanded_embedding) < target_length:
        # 使用线性插值扩展维度
        x_original = np.linspace(0, 1, len(expanded_embedding))
        x_new = np.linspace(0, 1, target_length)
        expanded_embedding = np.interp(x_new, x_original, expanded_embedding)
    # 如果扩展后的维度大于目标维度，截断
    elif len(expanded_embedding) > target_length:
        expanded_embedding = expanded_embedding[:target_length]
    
    # L2归一化
    normalizer = Normalizer(norm='l2')
    expanded_embedding = normalizer.transform(expanded_embedding.reshape(1, -1)).flatten()
    return expanded_embedding


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: Union[EmbeddingProcessRequest, EmbeddingQuestionRequest]):
    """获取文本的嵌入向量（入口点）"""
    if isinstance(request, EmbeddingQuestionRequest):
        input_texts = [request.input]
    else:
        input_texts = request.input

    # 计算token数量
    prompt_tokens = sum(num_tokens_from_string(text) for text in input_texts)
    total_tokens = prompt_tokens

    # 获取嵌入向量
    try:
        embeddings = await get_embeddings_async(input_texts)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute embeddings: {str(e)}"
        )

    # 构建响应
    response = {
        "object": "list",
        "model": request.model,
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": i
            }
            for i, embedding in enumerate(embeddings)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
    }
    return response


async def process_batch(batch: List[Tuple[str, asyncio.Future]]) -> None:
    """处理一批文本的嵌入向量计算"""
    try:
        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        
        # 在线程池中执行计算密集型操作
        loop = asyncio.get_event_loop()
        with model_lock:  # 确保模型访问是线程安全的
            embeddings = await loop.run_in_executor(
                executor,
                lambda: embeddings_model.encode(texts, convert_to_numpy=True)
            )
        
        # 处理嵌入向量维度
        for i, emb in enumerate(embeddings):
            processed_emb = expand_features(emb, TARGET_DIM)
            futures[i].set_result(processed_emb.tolist())
            
    except Exception as e:
        # 设置所有future的异常
        for future in futures:
            if not future.done():
                future.set_exception(e)


async def batch_processor():
    """批量处理请求的协程"""
    while True:
        batch = []
        start_time = time.time()
        
        try:
            # 等待第一个请求
            item = await request_queue.get()
            if item is None:  # 退出信号
                break
                
            batch.append(item)
            
            # 收集一批请求或超时
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    timeout = BATCH_TIMEOUT - (time.time() - start_time)
                    if timeout <= 0:
                        break
                        
                    item = await asyncio.wait_for(
                        asyncio.shield(request_queue.get()),
                        timeout=timeout
                    )
                    if item is None:  # 退出信号
                        break
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                    
            # 处理批次
            if batch:
                await process_batch(batch)
                
        except Exception as e:
            print(f"Error in batch processor: {e}")
            # 设置所有future的异常
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)


async def get_embeddings_async(texts: List[str]) -> List[List[float]]:
    """异步获取文本嵌入向量"""
    # 创建future列表
    loop = asyncio.get_event_loop()
    futures = [loop.create_future() for _ in texts]
    
    # 将任务放入队列
    async with queue_semaphore:
        for text, future in zip(texts, futures):
            await request_queue.put((text, future))
    
    # 等待所有future完成
    results = await asyncio.gather(*futures, return_exceptions=True)
    
    # 处理异常
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise result
            
    return results


# 从环境变量加载配置
API_KEY = os.environ.get('API_KEY', '')
MODEL_PATH = os.environ.get('MODEL_PATH', './models/bge-m3')
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
PORT = int(os.environ.get('PORT', '6008'))
TARGET_DIM = int(os.environ.get('TARGET_DIM', '2560'))

# 并发控制配置
MAX_CONCURRENT_REQUESTS = int(os.environ.get('MAX_CONCURRENT_REQUESTS', '10'))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '32'))
BATCH_TIMEOUT = float(os.environ.get('BATCH_TIMEOUT', '0.1'))  # 秒
MAX_QUEUE_SIZE = int(os.environ.get('MAX_QUEUE_SIZE', '100'))
THREAD_POOL_SIZE = int(os.environ.get('THREAD_POOL_SIZE', '4'))

# 请求队列和批处理
request_queue = asyncio.Queue()
queue_semaphore = asyncio.Semaphore(MAX_QUEUE_SIZE)
model_lock = threading.Lock()  # 模型访问锁

# 速率限制
RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', '100'))
RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', '60'))  # 秒
rate_limit_window = timedelta(seconds=RATE_LIMIT_WINDOW)
rate_limit_requests: Dict[str, List[datetime]] = {}
rate_limit_lock = asyncio.Lock()

# 线程池
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE, thread_name_prefix="embedding_worker")

# 打印配置信息
print("\n=== 配置信息 ===")
print(f"API_KEY: {'*' * 8}{API_KEY[-4:] if API_KEY else 'None'}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"DEVICE: {DEVICE}")
print(f"PORT: {PORT}")
print(f"TARGET_DIM: {TARGET_DIM}")
print("================\n")

# 全局模型实例
embeddings_model = None

def load_model():
    """加载模型"""
    global embeddings_model
    if embeddings_model is None:
        print(f"Loading model from {MODEL_PATH} using device {DEVICE}")
        embeddings_model = SentenceTransformer(MODEL_PATH, device=DEVICE)
        print("Model loaded successfully")
    return embeddings_model

async def startup():
    """启动时初始化"""
    # 加载模型
    load_model()
    
    # 启动批处理协程
    global batch_processor_task
    batch_processor_task = asyncio.create_task(batch_processor())
    print(f"Batch processor started with {THREAD_POOL_SIZE} worker threads")

async def shutdown():
    """关闭时清理"""
    # 发送退出信号给批处理协程
    if 'batch_processor_task' in globals():
        await request_queue.put(None)
        await batch_processor_task
    
    # 关闭线程池
    executor.shutdown(wait=True)
    print("Thread pool executor shutdown")

# 添加生命周期事件处理
app.add_event_handler("startup", startup)
app.add_event_handler("shutdown", shutdown)

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Max batch size: {MAX_BATCH_SIZE}")
    print(f"Thread pool size: {THREAD_POOL_SIZE}")
    
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=PORT,
        workers=1,  # 使用单worker，通过异步处理并发
        limit_concurrency=MAX_CONCURRENT_REQUESTS,
        timeout_keep_alive=30,
    )
