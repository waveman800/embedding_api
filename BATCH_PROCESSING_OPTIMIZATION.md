# 批处理优化说明

## 问题描述

原始的嵌入API服务强制使用批处理模式，导致以下问题：
- 单个请求需要等待 `BATCH_TIMEOUT` 时间才能处理
- 20个请求的批处理耗时245秒，响应速度严重受影响
- 无法满足需要立即响应的应用场景

## 解决方案

### 1. 新增配置选项

在 `.env` 文件中添加了 `ENABLE_BATCH_PROCESSING` 配置：

```bash
# 批处理控制
ENABLE_BATCH_PROCESSING=false   # false=直接处理, true=批处理
```

### 2. 两种处理模式

**直接处理模式（推荐）**：
- `ENABLE_BATCH_PROCESSING=false`
- 每个请求立即处理，无等待时间
- 适合单个或少量请求的场景
- 响应速度最快

**批处理模式**：
- `ENABLE_BATCH_PROCESSING=true`
- 多个请求打包处理，提高吞吐量
- 适合大量并发请求的场景
- 可能有轻微延迟

### 3. 代码修改

#### embedding.py 主要修改：

1. **配置加载**：
   ```python
   config['ENABLE_BATCH_PROCESSING'] = os.environ.get('ENABLE_BATCH_PROCESSING', 'true').lower() == 'true'
   ```

2. **处理逻辑**：
   ```python
   async def get_embeddings_async(texts):
       if not ENABLE_BATCH_PROCESSING:
           # 直接处理模式：立即处理所有文本
           loop = asyncio.get_event_loop()
           embeddings = await loop.run_in_executor(
               executor, 
               lambda: process_texts_directly(texts)
           )
           return embeddings
       else:
           # 批处理模式：使用原有的队列机制
           # ... 原有批处理逻辑
   ```

3. **启动逻辑**：
   ```python
   async def startup():
       # 只在启用批处理时启动批处理协程
       if ENABLE_BATCH_PROCESSING:
           asyncio.create_task(batch_processor())
           print("Service started! (Batch processing mode)")
       else:
           print("Service started! (Direct processing mode)")
   ```

### 4. 便捷工具

#### disable_batch.py 脚本：
```bash
python disable_batch.py
```
- 自动将 `ENABLE_BATCH_PROCESSING` 设置为 `false`
- 快速切换到直接处理模式

#### test_direct_mode.py 测试脚本：
```bash
python test_direct_mode.py
```
- 验证直接处理模式是否正常工作
- 测试响应时间和处理效果

## 使用方法

### 快速启用直接处理模式（推荐）：

```bash
# 1. 禁用批处理
python disable_batch.py

# 2. 重启服务
python embedding.py

# 3. 测试效果（可选）
python test_direct_mode.py
```

### 手动配置：

编辑 `.env` 文件：
```bash
# 直接处理模式（推荐）
ENABLE_BATCH_PROCESSING=false

# 批处理模式
ENABLE_BATCH_PROCESSING=true
```

## 性能对比

| 模式 | 单个请求延迟 | 适用场景 | 优势 |
|------|-------------|----------|------|
| 直接处理 | 立即响应 | 单个/少量请求 | 响应最快 |
| 批处理 | 有等待时间 | 大量并发请求 | 吞吐量高 |

## 配置建议

### 直接处理模式配置：
```bash
ENABLE_BATCH_PROCESSING=false
MAX_CONCURRENT_REQUESTS=20
THREAD_POOL_SIZE=8
```

### 批处理模式配置：
```bash
ENABLE_BATCH_PROCESSING=true
MAX_BATCH_SIZE=32
BATCH_TIMEOUT=0.1
MAX_CONCURRENT_REQUESTS=10
```

## 注意事项

1. **重启服务**：修改配置后需要重启服务才能生效
2. **资源使用**：直接处理模式可能使用更多线程资源
3. **并发控制**：两种模式都受 `MAX_CONCURRENT_REQUESTS` 限制
4. **编码问题**：已修复Windows控制台的Unicode编码问题

## 验证方法

启动服务后，查看控制台输出：
- `Batch: Disabled (Direct)` - 直接处理模式
- `Batch: Enabled` - 批处理模式

或者运行测试脚本验证响应速度：
```bash
python test_direct_mode.py
```

## 总结

通过添加 `ENABLE_BATCH_PROCESSING` 配置选项，现在可以灵活选择处理模式：
- **默认推荐直接处理模式**，获得最快的响应速度
- **可选批处理模式**，适合高并发场景
- **一键切换**，使用 `disable_batch.py` 脚本快速配置
- **完全向后兼容**，不影响现有功能

这个优化解决了批处理导致的延迟问题，让嵌入API服务能够满足不同场景的需求。
