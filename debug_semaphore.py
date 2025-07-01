import asyncio

async def test_semaphore():
    # 创建信号量
    semaphore = asyncio.Semaphore(10)
    print(f"初始信号量值: {semaphore._value}")
    
    # 测试acquire_nowait
    try:
        semaphore.acquire_nowait()
        print(f"获取后信号量值: {semaphore._value}")
        print("成功获取信号量")
        
        # 释放
        semaphore.release()
        print(f"释放后信号量值: {semaphore._value}")
    except Exception as e:
        print(f"获取信号量失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_semaphore())
