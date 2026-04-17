import torch
import sys

print("=" * 60)
print("CUDA 状态检查")
print("=" * 60)
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    # 尝试创建一个简单的 tensor 在 GPU 上
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        print(f"GPU Tensor 创建成功: {x.device}")
    except Exception as e:
        print(f"GPU Tensor 创建失败: {e}")
else:
    print("CUDA 不可用！")
    print("\n可能原因:")
    print("1. 没有安装 NVIDIA GPU 驱动")
    print("2. PyTorch 安装的是 CPU 版本")
    print("3. CUDA 版本不匹配")

print("=" * 60)
