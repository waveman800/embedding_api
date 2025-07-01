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

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载 .env 文件
    print("SUCCESS: .env file loaded")
except ImportError:
    print("WARNING: python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"WARNING: Error loading .env file: {e}")

# 配置日志 - 优化输出格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 禁用transformers和sentence_transformers的详细日志
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# 设置环境变量禁用进度条
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 优化的并发控制类 - 支持排队和无错误处理
class ConcurrencyControl:
    def __init__(self, max_concurrent: int, max_queue_size: int = 1000):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.active_requests = 0
        self.total_requests = 0
        self.queued_requests = 0
        self.completed_requests = 0
        self.lock = asyncio.Lock()
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_queue = True
        
    async def acquire_with_queue(self):
        """获取信号量，支持排队等待"""
        async with self.lock:
            self.total_requests += 1
            
        # 直接尝试获取信号量
        try:
            await self.semaphore.acquire()
            async with self.lock:
                self.active_requests += 1
            return True
        except Exception as e:
            logger.error(f"Error acquiring semaphore: {e}")
            return False
    
    async def release(self):
        """释放信号量"""
        async with self.lock:
            if self.active_requests > 0:
                self.active_requests -= 1
            self.completed_requests += 1
        self.semaphore.release()
    
    async def get_stats(self):
        """获取统计信息"""
        async with self.lock:
            return {
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "queued_requests": self.queued_requests,
                "available_slots": self.semaphore._value,
                "max_concurrent": self.max_concurrent,
                "queue_size": self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0
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

# 优化的并发控制中间件 - 支持排队等待，无错误处理
@app.middleware("http")
async def concurrency_control_middleware(request: Request, call_next):
    # 跳过健康检查和状态检查
    if request.url.path in ["/health", "/status", "/docs", "/openapi.json"]:
        return await call_next(request)
    
    # 使用优化的并发控制
    if 'concurrency_control' in globals():
        try:
            # 获取信号量，支持排队等待
            acquired = await concurrency_control.acquire_with_queue()
            if not acquired:
                logger.error("Failed to acquire concurrency control")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal Server Error",
                        "message": "服务器内部错误"
                    }
                )
            
            try:
                # 执行请求，增加超时时间以适应排队等待
                response = await asyncio.wait_for(
                    call_next(request), 
                    timeout=float(os.environ.get('REQUEST_TIMEOUT', '300'))  # 增加到5分钟
                )
                return response
            except asyncio.TimeoutError:
                logger.warning("Request timeout after extended wait")
                return JSONResponse(
                    status_code=408,  # Request Timeout
                    content={
                        "error": "Request Timeout",
                        "message": "请求处理超时，但已尽力处理"
                    }
                )
            except Exception as e:
                logger.error(f"Request processing error: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Processing Error",
                        "message": f"请求处理异常: {str(e)}"
                    }
                )
            finally:
                await concurrency_control.release()
                
        except Exception as e:
            logger.error(f"Concurrency control error: {str(e)}")
            # 即使并发控制失败，也要尝试处理请求
            return await call_next(request)
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
                lambda: embeddings_model.encode(
                    texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False,  # 禁用进度条
                    batch_size=min(len(texts), 32)  # 优化批处理大小
                )
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
    """优化的批量处理协程 - 支持高并发和错误恢复"""
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        batch = []
        start_time = time.time()
        
        try:
            # 等待第一个请求，带有超时保护
            try:
                item = await asyncio.wait_for(request_queue.get(), timeout=30.0)
                if item is None:  # 退出信号
                    break
                batch.append(item)
            except asyncio.TimeoutError:
                # 如果30秒没有请求，继续等待
                continue
            
            # 收集更多请求，优化批处理效率
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    timeout = max(0.01, BATCH_TIMEOUT - (time.time() - start_time))
                    if timeout <= 0:
                        break
                        
                    item = await asyncio.wait_for(
                        request_queue.get(),
                        timeout=timeout
                    )
                    if item is None:  # 退出信号
                        break
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    logger.warning(f"Error collecting batch item: {e}")
                    break
                    
            # 处理批次
            if batch:
                batch_size = len(batch)
                # 只在批次较大时记录日志
                if batch_size > 5:
                    logger.info(f"Processing batch: {batch_size} requests")
                
                try:
                    await process_batch(batch)
                    consecutive_errors = 0  # 重置错误计数
                    
                    # 只记录较大批次的处理时间
                    if batch_size > 10:
                        processing_time = time.time() - start_time
                        logger.info(f"Batch completed: {batch_size} requests in {processing_time:.2f}s")
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Batch error ({consecutive_errors}/{max_consecutive_errors}): {str(e)[:100]}")
                    
                    # 处理当前批次的所有请求
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(e)
                    
                    # 如果连续错误过多，暂停处理
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(f"Too many errors, pausing for 1s...")
                        await asyncio.sleep(1.0)
                        consecutive_errors = 0
                
        except Exception as e:
            logger.error(f"Critical error in batch processor: {e}")
            # 处理当前批次的所有请求
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
            
            # 稍作等待后继续
            await asyncio.sleep(0.1)


async def get_single_embedding(text: str) -> List[float]:
    """单个文本嵌入处理 - 用于批处理失败时的回退"""
    try:
        # 使用线程池处理单个文本
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: embeddings_model.encode(
                [text], 
                normalize_embeddings=True,
                show_progress_bar=False  # 禁用进度条
            )
        )
        embedding = result[0].tolist()
        return expand_features(embedding, TARGET_DIM)
    except Exception as e:
        logger.error(f"Single embedding processing error: {e}")
        raise

async def get_embeddings_async(texts: List[str]) -> List[List[float]]:
    """异步获取文本嵌入向量"""
    # 检查是否启用批处理
    if not ENABLE_BATCH_PROCESSING:
        # 直接处理模式：立即处理所有文本
        try:
            loop = asyncio.get_event_loop()
            with model_lock:
                embeddings = await loop.run_in_executor(
                    executor,
                    lambda: embeddings_model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=min(len(texts), 32)
                    )
                )
            
            # 处理嵌入向量维度
            results = []
            for emb in embeddings:
                processed_emb = expand_features(emb, TARGET_DIM)
                results.append(processed_emb.tolist())
            
            return results
            
        except Exception as e:
            logger.error(f"Direct processing error: {e}")
            raise
    
    else:
        # 批处理模式：使用原有的队列机制
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
def load_config():
    """加载并验证配置参数"""
    config = {}
    
    # 基础配置
    config['API_KEY'] = os.environ.get('API_KEY', '')
    config['MODEL_PATH'] = os.environ.get('MODEL_PATH', './models/bge-m3')
    config['DEVICE'] = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数值配置（带验证）
    try:
        config['PORT'] = int(os.environ.get('PORT', '6008'))
        config['TARGET_DIM'] = int(os.environ.get('TARGET_DIM', '2560'))
        
        # 并发控制配置
        config['ENABLE_BATCH_PROCESSING'] = os.environ.get('ENABLE_BATCH_PROCESSING', 'true').lower() == 'true'
        config['MAX_CONCURRENT_REQUESTS'] = int(os.environ.get('MAX_CONCURRENT_REQUESTS', '10'))
        config['MAX_BATCH_SIZE'] = int(os.environ.get('MAX_BATCH_SIZE', '32'))
        config['BATCH_TIMEOUT'] = float(os.environ.get('BATCH_TIMEOUT', '0.1'))
        config['MAX_QUEUE_SIZE'] = int(os.environ.get('MAX_QUEUE_SIZE', '100'))
        config['THREAD_POOL_SIZE'] = int(os.environ.get('THREAD_POOL_SIZE', '4'))
        config['MAX_RETRIES'] = int(os.environ.get('MAX_RETRIES', '3'))
        config['RETRY_DELAY'] = float(os.environ.get('RETRY_DELAY', '1.0'))
        config['REQUEST_TIMEOUT'] = float(os.environ.get('REQUEST_TIMEOUT', '60'))
        
    except ValueError as e:
        print(f"❌ 配置参数格式错误: {e}")
        print("请检查 .env 文件中的数值配置")
        raise
    
    # 配置验证
    if config['PORT'] < 1 or config['PORT'] > 65535:
        raise ValueError(f"端口号无效: {config['PORT']}，应在 1-65535 范围内")
    
    if config['MAX_CONCURRENT_REQUESTS'] < 1:
        raise ValueError(f"最大并发请求数无效: {config['MAX_CONCURRENT_REQUESTS']}，应大于0")
    
    if config['THREAD_POOL_SIZE'] < 1:
        raise ValueError(f"线程池大小无效: {config['THREAD_POOL_SIZE']}，应大于0")
    
    return config

# 加载配置
try:
    CONFIG = load_config()
    
    # 将配置赋值给全局变量（保持向后兼容）
    API_KEY = CONFIG['API_KEY']
    MODEL_PATH = CONFIG['MODEL_PATH']
    DEVICE = CONFIG['DEVICE']
    PORT = CONFIG['PORT']
    TARGET_DIM = CONFIG['TARGET_DIM']
    ENABLE_BATCH_PROCESSING = CONFIG['ENABLE_BATCH_PROCESSING']
    MAX_CONCURRENT_REQUESTS = CONFIG['MAX_CONCURRENT_REQUESTS']
    MAX_BATCH_SIZE = CONFIG['MAX_BATCH_SIZE']
    BATCH_TIMEOUT = CONFIG['BATCH_TIMEOUT']
    MAX_QUEUE_SIZE = CONFIG['MAX_QUEUE_SIZE']
    THREAD_POOL_SIZE = CONFIG['THREAD_POOL_SIZE']
    MAX_RETRIES = CONFIG['MAX_RETRIES']
    RETRY_DELAY = CONFIG['RETRY_DELAY']
    REQUEST_TIMEOUT = CONFIG['REQUEST_TIMEOUT']
    
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    print("请检查 .env 文件是否存在且格式正确")
    print("可以运行 'python setup_env.py' 来创建配置文件")
    raise

# 请求队列和批处理
request_queue = asyncio.Queue()
queue_semaphore = asyncio.Semaphore(MAX_QUEUE_SIZE)
model_lock = threading.Lock()  # 模型访问锁

# 线程池
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE, thread_name_prefix="embedding_worker")

# 初始化并发控制
concurrency_control = None

# 打印配置信息
def print_config():
    """打印当前配置信息 - 简化版本"""
    print("\n" + "="*40)
    print("Embedding API Service")
    print("="*40)
    
    # 基础配置
    api_key_display = f"{'*' * 8}{API_KEY[-4:]}" if API_KEY else "Not Set"
    print(f"Port: {PORT} | Device: {DEVICE} | Dim: {TARGET_DIM}")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"API Key: {api_key_display}")
    
    # 并发配置
    batch_mode = "Enabled" if ENABLE_BATCH_PROCESSING else "Disabled (Direct)"
    print(f"Concurrency: {MAX_CONCURRENT_REQUESTS} | Batch: {batch_mode} | Threads: {THREAD_POOL_SIZE}")
    
    # 配置来源
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    config_source = ".env file" if os.path.exists(env_file) else "environment variables"
    print(f"Config: {config_source}")
    
    print("="*40 + "\n")

# 打印配置
print_config()

# 全局模型实例
embeddings_model = None

def load_model():
    """加载嵌入模型 - 支持多种模型包括Qwen3"""
    global embeddings_model
    if embeddings_model is None:
        print(f"Loading model: {os.path.basename(MODEL_PATH)} on {DEVICE}...")
        
        try:
            # 尝试加载模型
            if 'qwen' in MODEL_PATH.lower():
                print("Detected Qwen model, using optimized settings...")
                # Qwen模型的特殊设置
                embeddings_model = SentenceTransformer(
                    MODEL_PATH, 
                    device=DEVICE,
                    trust_remote_code=True,  # Qwen模型可能需要这个参数
                    cache_folder=None  # 避免缓存警告
                )
            else:
                # 默认加载方式
                embeddings_model = SentenceTransformer(
                    MODEL_PATH, 
                    device=DEVICE,
                    cache_folder=None  # 避免缓存警告
                )
            
            print(f"Model loaded: {type(embeddings_model).__name__}")
            
            # 打印模型信息
            if hasattr(embeddings_model, 'get_sentence_embedding_dimension'):
                model_dim = embeddings_model.get_sentence_embedding_dimension()
                print(f"Model embedding dimension: {model_dim}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            try:
                # 备用加载方法
                embeddings_model = SentenceTransformer(
                    MODEL_PATH, 
                    device=DEVICE,
                    trust_remote_code=True,
                    use_auth_token=False,
                    cache_folder=None  # 避免缓存警告
                )
                print("Model loaded (alternative method)")
            except Exception as e2:
                print(f"Failed to load model with both methods: {e2}")
                raise e2
                
    return embeddings_model

async def startup():
    """启动时初始化"""
    global embeddings_model, concurrency_control
    print("Starting service...")
    
    # 初始化优化的并发控制
    concurrency_control = ConcurrencyControl(
        max_concurrent=MAX_CONCURRENT_REQUESTS,
        max_queue_size=int(os.environ.get('MAX_QUEUE_SIZE', '1000'))
    )
    logger.info(f"Concurrency control initialized: max={MAX_CONCURRENT_REQUESTS}")
    
    load_model()
    
    # 只在启用批处理时启动批处理协程
    if ENABLE_BATCH_PROCESSING:
        asyncio.create_task(batch_processor())
        print("Service started! (Batch processing mode)")
    else:
        print("Service started! (Direct processing mode)")

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
