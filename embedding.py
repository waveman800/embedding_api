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
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS, HTTP_503_SERVICE_UNAVAILABLE
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 并发控制类
class ConcurrencyControl:
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.total_requests = 0
        self.rejected_requests = 0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """尝试获取信号量"""
        async with self.lock:
            self.total_requests += 1
            
        # 检查是否有可用的信号量
        if self.semaphore._value > 0:
            try:
                # 非阻塞获取信号量
                await asyncio.wait_for(self.semaphore.acquire(), timeout=0.001)
                async with self.lock:
                    self.active_requests += 1
                return True
            except asyncio.TimeoutError:
                async with self.lock:
                    self.rejected_requests += 1
                return False
            except Exception:
                async with self.lock:
                    self.rejected_requests += 1
                return False
        else:
            async with self.lock:
                self.rejected_requests += 1
            return False
    
    async def release(self):
        """释放信号量"""
        async with self.lock:
            self.active_requests -= 1
        self.semaphore.release()
    
    async def get_stats(self):
        """获取统计信息"""
        async with self.lock:
            return {
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "rejected_requests": self.rejected_requests,
                "available_slots": self.semaphore._value
            }

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
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": embeddings_model is not None
    }

# 状态监控端点
@app.get("/status")
async def status_check():
    if 'concurrency_control' in globals():
        stats = await concurrency_control.get_stats()
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "concurrency": stats,
            "model_loaded": embeddings_model is not None,
            "device": DEVICE
        }
    else:
        return {
            "status": "initializing",
            "timestamp": datetime.utcnow().isoformat()
        }

# 并发控制中间件
@app.middleware("http")
async def concurrency_control_middleware(request: Request, call_next):
    # 跳过健康检查和状态检查
    if request.url.path in ["/health", "/status", "/docs", "/openapi.json"]:
        return await call_next(request)
    
    # 尝试获取并发控制信号量
    if 'concurrency_control' in globals():
        acquired = await concurrency_control.acquire()
        if not acquired:
            # 计算重试延迟（基于当前负载）
            stats = await concurrency_control.get_stats()
            retry_after = min(30, max(5, stats["active_requests"] * 2))
            
            logger.warning(f"Request rejected due to concurrency limit. Active: {stats['active_requests']}")
            
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": "服务器当前负载过高，请稍后重试",
                    "retry_after": retry_after,
                    "active_requests": stats["active_requests"]
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        try:
            # 添加请求超时控制
            response = await asyncio.wait_for(
                call_next(request), 
                timeout=float(os.environ.get('REQUEST_TIMEOUT', '60'))
            )
            return response
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            return JSONResponse(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Request Timeout",
                    "message": "请求处理超时，请稍后重试"
                }
            )
        except Exception as e:
            logger.error(f"Request processing error: {str(e)}")
            return JSONResponse(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service Error",
                    "message": "服务处理异常，请稍后重试"
                }
            )
        finally:
            await concurrency_control.release()
    else:
        return await call_next(request)


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
    # 跳过健康检查和状态检查的认证
    if request.url.path in ["/health", "/status"]:
        return True
    
    # 如果API_KEY为空，则跳过认证
    if not API_KEY:
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
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
RETRY_DELAY = float(os.environ.get('RETRY_DELAY', '1.0'))
REQUEST_TIMEOUT = float(os.environ.get('REQUEST_TIMEOUT', '60'))

# 请求队列和批处理
request_queue = asyncio.Queue()
queue_semaphore = asyncio.Semaphore(MAX_QUEUE_SIZE)
model_lock = threading.Lock()  # 模型访问锁

# 线程池
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE, thread_name_prefix="embedding_worker")

# 初始化并发控制
concurrency_control = None

# 打印配置信息
print("\n=== 配置信息 ===")
print(f"API_KEY: {'*' * 8}{API_KEY[-4:] if API_KEY else 'None'}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"DEVICE: {DEVICE}")
print(f"PORT: {PORT}")
print(f"TARGET_DIM: {TARGET_DIM}")
print(f"MAX_CONCURRENT_REQUESTS: {MAX_CONCURRENT_REQUESTS}")
print(f"MAX_BATCH_SIZE: {MAX_BATCH_SIZE}")
print(f"REQUEST_TIMEOUT: {REQUEST_TIMEOUT}s")
print(f"MAX_RETRIES: {MAX_RETRIES}")
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
    global embeddings_model, concurrency_control
    print("正在启动服务...")
    
    # 初始化并发控制
    concurrency_control = ConcurrencyControl(MAX_CONCURRENT_REQUESTS)
    logger.info(f"并发控制初始化完成，最大并发数: {MAX_CONCURRENT_REQUESTS}")
    
    load_model()
    # 启动批处理协程
    asyncio.create_task(batch_processor())
    print("服务启动完成!")

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
        timeout_keep_alive=30,
    )
