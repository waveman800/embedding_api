import torch
from transformers import AutoModel, AutoTokenizer
import os

def check_gpu_availability():
    print("=== GPU 信息 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"可用 GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  已分配显存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  缓存分配显存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("未检测到可用的 CUDA 设备")

def test_tensor_parallelism(model_path):
    print("\n=== 测试张量并行 ===")
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("需要至少2个GPU来测试张量并行")
        return
    
    print(f"加载模型: {model_path}")
    
    # 加载模型并启用张量并行
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("\n=== 模型设备分布 ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
    
    # 检查模型是否分布在多个GPU上
    devices = set(param.device for param in model.parameters())
    if len(devices) > 1:
        print("\n✅ 张量并行已启用，模型分布在多个GPU上:")
        for device in devices:
            print(f"  - {device}")
    else:
        print("\n❌ 张量并行未正确配置，模型仅在一个设备上运行:")
        print(f"  - {list(devices)[0]}")
    
    # 测试推理
    print("\n=== 测试推理 ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer("这是一个测试文本", return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"输出张量形状: {outputs.last_hidden_state.shape}")
    print("✅ 推理测试通过")

if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "/app/models/Qwen3-Embedding-4B")
    
    check_gpu_availability()
    
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        test_tensor_parallelism(model_path)
    else:
        print("\n⚠️  需要至少2个GPU来测试张量并行")
