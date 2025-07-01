#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发控制测试脚本
用于验证嵌入API服务的并发处理能力和错误处理机制
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
import argparse
import sys

class ConcurrencyTester:
    def __init__(self, base_url: str = "http://localhost:6008", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
        
    async def test_health(self) -> Dict[str, Any]:
        """测试健康检查端点"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    return {
                        "status_code": response.status,
                        "data": await response.json()
                    }
            except Exception as e:
                return {"error": str(e)}
    
    async def test_status(self) -> Dict[str, Any]:
        """测试状态监控端点"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/status") as response:
                    return {
                        "status_code": response.status,
                        "data": await response.json()
                    }
            except Exception as e:
                return {"error": str(e)}
    
    async def send_embedding_request(self, session: aiohttp.ClientSession, text: str, request_id: int) -> Dict[str, Any]:
        """发送单个嵌入请求"""
        payload = {
            "input": [text],
            "model": "bge-m3"
        }
        
        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                
                result = {
                    "request_id": request_id,
                    "status_code": response.status,
                    "response_time": end_time - start_time,
                    "success": response.status == 200
                }
                
                if response.status == 200:
                    data = await response.json()
                    result["embedding_length"] = len(data["data"][0]["embedding"]) if data["data"] else 0
                elif response.status == 429:
                    # 处理限流响应
                    data = await response.json()
                    result["error"] = data.get("message", "Too Many Requests")
                    result["retry_after"] = response.headers.get("Retry-After")
                else:
                    result["error"] = await response.text()
                
                return result
                
        except asyncio.TimeoutError:
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    async def test_concurrency(self, concurrent_requests: int = 20, test_text: str = "测试文本") -> Dict[str, Any]:
        """测试并发处理能力"""
        print(f"开始并发测试：{concurrent_requests} 个并发请求")
        
        async with aiohttp.ClientSession() as session:
            # 创建并发任务
            tasks = [
                self.send_embedding_request(session, f"{test_text} {i}", i)
                for i in range(concurrent_requests)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # 统计结果
            successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
            failed_requests = len(results) - successful_requests
            rate_limited = sum(1 for r in results if isinstance(r, dict) and r.get("status_code") == 429)
            avg_response_time = sum(r.get("response_time", 0) for r in results if isinstance(r, dict)) / len(results)
            
            return {
                "total_requests": concurrent_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "rate_limited_requests": rate_limited,
                "success_rate": successful_requests / concurrent_requests * 100,
                "total_time": total_time,
                "avg_response_time": avg_response_time,
                "requests_per_second": concurrent_requests / total_time,
                "results": results
            }
    
    async def test_gradual_load(self, max_concurrent: int = 50, step: int = 5) -> List[Dict[str, Any]]:
        """逐步增加负载测试"""
        print(f"开始逐步负载测试：从 {step} 到 {max_concurrent} 个并发请求")
        
        results = []
        for concurrent in range(step, max_concurrent + 1, step):
            print(f"测试 {concurrent} 个并发请求...")
            result = await self.test_concurrency(concurrent)
            result["concurrent_level"] = concurrent
            results.append(result)
            
            # 短暂休息，避免对服务器造成过大压力
            await asyncio.sleep(2)
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """打印测试结果"""
        print("\n" + "="*50)
        print("并发测试结果")
        print("="*50)
        print(f"总请求数: {results['total_requests']}")
        print(f"成功请求数: {results['successful_requests']}")
        print(f"失败请求数: {results['failed_requests']}")
        print(f"限流请求数: {results['rate_limited_requests']}")
        print(f"成功率: {results['success_rate']:.2f}%")
        print(f"总耗时: {results['total_time']:.2f}s")
        print(f"平均响应时间: {results['avg_response_time']:.3f}s")
        print(f"请求处理速率: {results['requests_per_second']:.2f} req/s")
        
        # 显示错误详情
        errors = {}
        for result in results['results']:
            if isinstance(result, dict) and not result.get('success', False):
                error = result.get('error', 'Unknown error')
                errors[error] = errors.get(error, 0) + 1
        
        if errors:
            print("\n错误统计:")
            for error, count in errors.items():
                print(f"  {error}: {count}")
    
    def print_gradual_results(self, results: List[Dict[str, Any]]):
        """打印逐步负载测试结果"""
        print("\n" + "="*70)
        print("逐步负载测试结果")
        print("="*70)
        print(f"{'并发数':<8} {'成功率':<8} {'限流数':<8} {'平均响应时间':<12} {'处理速率':<12}")
        print("-"*70)
        
        for result in results:
            print(f"{result['concurrent_level']:<8} "
                  f"{result['success_rate']:<8.1f}% "
                  f"{result['rate_limited_requests']:<8} "
                  f"{result['avg_response_time']:<12.3f}s "
                  f"{result['requests_per_second']:<12.2f}")

async def main():
    parser = argparse.ArgumentParser(description="嵌入API并发测试工具")
    parser.add_argument("--url", default="http://localhost:6008", help="API服务地址")
    parser.add_argument("--api-key", default="", help="API密钥")
    parser.add_argument("--concurrent", type=int, default=20, help="并发请求数")
    parser.add_argument("--gradual", action="store_true", help="执行逐步负载测试")
    parser.add_argument("--max-concurrent", type=int, default=50, help="逐步测试的最大并发数")
    
    args = parser.parse_args()
    
    tester = ConcurrencyTester(args.url, args.api_key)
    
    # 测试服务状态
    print("检查服务状态...")
    health_result = await tester.test_health()
    print(f"健康检查: {health_result}")
    
    status_result = await tester.test_status()
    print(f"状态检查: {status_result}")
    
    if health_result.get("status_code") != 200:
        print("服务不可用，退出测试")
        sys.exit(1)
    
    # 执行并发测试
    if args.gradual:
        print(f"\n执行逐步负载测试...")
        gradual_results = await tester.test_gradual_load(args.max_concurrent)
        tester.print_gradual_results(gradual_results)
    else:
        print(f"\n执行并发测试...")
        test_results = await tester.test_concurrency(args.concurrent)
        tester.print_results(test_results)

if __name__ == "__main__":
    asyncio.run(main())
