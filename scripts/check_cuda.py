"""
CUDA 配置檢查工具
診斷 PyTorch 和 CUDA 兼容性問題
"""
import sys
import subprocess

def print_section(title):
    """打印分隔線"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def check_nvidia_smi():
    """檢查 nvidia-smi"""
    print_section("檢查 NVIDIA GPU 驅動")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA 驅動已安裝")
            # 提取 CUDA 版本
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"  {line.strip()}")
                if '|' in line and 'MiB' in line and not 'Processes' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("✗ nvidia-smi 無法運行")
            return False
    except FileNotFoundError:
        print("✗ 未找到 nvidia-smi (可能沒有安裝 NVIDIA 驅動)")
        return False
    except Exception as e:
        print(f"✗ 檢查時發生錯誤: {e}")
        return False

def check_pytorch():
    """檢查 PyTorch 配置"""
    print_section("檢查 PyTorch 配置")
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        
        # CUDA 可用性
        cuda_available = torch.cuda.is_available()
        print(f"{'✓' if cuda_available else '✗'} CUDA 可用: {cuda_available}")
        
        if cuda_available:
            # CUDA 版本
            print(f"  PyTorch CUDA 版本: {torch.version.cuda}")
            
            # cuDNN 版本
            if torch.backends.cudnn.is_available():
                print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
            
            # GPU 數量和名稱
            gpu_count = torch.cuda.device_count()
            print(f"  檢測到 {gpu_count} 個 GPU:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_capability = torch.cuda.get_device_capability(i)
                print(f"    GPU {i}: {gpu_name}")
                print(f"      計算能力: {gpu_capability[0]}.{gpu_capability[1]}")
                
                # 內存信息
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"      顯存: {total_memory:.2f} GB")
                except:
                    pass
        else:
            print("\n⚠️ CUDA 不可用的可能原因:")
            print("  1. PyTorch CPU 版本 (沒有 CUDA 支持)")
            print("  2. PyTorch CUDA 版本與系統 CUDA 不兼容")
            print("  3. 沒有 NVIDIA GPU")
            print("  4. NVIDIA 驅動未正確安裝")
        
        return cuda_available
        
    except ImportError:
        print("✗ PyTorch 未安裝")
        print("  安裝: pip install torch torchvision")
        return False

def test_cuda_operation():
    """測試 CUDA 操作"""
    print_section("測試 CUDA 操作")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⊗ 跳過 (CUDA 不可用)")
            return False
        
        print("測試簡單的張量操作...")
        try:
            # 創建張量並移到 GPU
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            z.cpu()
            print("✓ 基本張量操作成功")
            
            # 測試卷積操作
            print("測試卷積操作...")
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            input_tensor = torch.randn(1, 3, 224, 224).cuda()
            output = conv(input_tensor)
            print("✓ 卷積操作成功")
            
            return True
            
        except RuntimeError as e:
            print(f"✗ CUDA 操作失敗: {e}")
            if "no kernel image is available" in str(e):
                print("\n⚠️ 這是 PyTorch/CUDA 版本不兼容的典型錯誤!")
                print("   需要重新安裝匹配的 PyTorch 版本")
            return False
            
    except Exception as e:
        print(f"✗ 測試時發生錯誤: {e}")
        return False

def recommend_solution(has_gpu, pytorch_cuda):
    """推薦解決方案"""
    print_section("建議的解決方案")
    
    if not has_gpu:
        print("⚠️ 未檢測到 NVIDIA GPU")
        print("\n選項 1: 使用 CPU 訓練")
        print("  python train.py --gpu_id -1 --batch_size 1")
        print("  或運行: train_cpu.bat")
        print("\n選項 2: 使用雲端 GPU (Google Colab, AWS, Azure 等)")
        
    elif not pytorch_cuda:
        print("⚠️ 有 GPU 但 PyTorch CUDA 不可用")
        print("\n可能原因:")
        print("  1. 安裝了 CPU 版本的 PyTorch")
        print("  2. PyTorch CUDA 版本與系統不兼容")
        
        print("\n解決方案: 重新安裝 PyTorch")
        print("\n首先卸載:")
        print("  pip uninstall torch torchvision")
        
        print("\n然後根據您的 CUDA 版本安裝:")
        print("\nCUDA 11.8 (推薦):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        print("\nCUDA 12.1:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        
        print("\nCUDA 12.4+:")
        print("  pip install torch torchvision")
        
    else:
        print("✓ PyTorch 和 CUDA 配置正常")
        print("\n如果仍然遇到錯誤，可能需要:")
        print("  1. 更新 NVIDIA 驅動")
        print("  2. 重新安裝 PyTorch")
        print("  3. 檢查代碼中的錯誤 (如拼寫錯誤)")

def main():
    print("\n" + "="*60)
    print("HDRSL CUDA 配置診斷工具")
    print("="*60)
    
    # 檢查步驟
    has_gpu = check_nvidia_smi()
    pytorch_cuda = check_pytorch()
    
    if pytorch_cuda:
        cuda_works = test_cuda_operation()
    else:
        cuda_works = False
    
    # 總結
    print_section("診斷總結")
    print(f"NVIDIA GPU 驅動:  {'✓ 已安裝' if has_gpu else '✗ 未檢測到'}")
    print(f"PyTorch CUDA:     {'✓ 可用' if pytorch_cuda else '✗ 不可用'}")
    print(f"CUDA 操作測試:    {'✓ 通過' if cuda_works else '✗ 失敗或跳過'}")
    
    # 推薦解決方案
    recommend_solution(has_gpu, pytorch_cuda)
    
    print("\n" + "="*60)
    if cuda_works:
        print("🎉 系統配置正常，可以開始訓練!")
        print("   運行: train.bat")
    else:
        print("⚠️ 需要修復配置後才能使用 GPU 訓練")
        print("   詳細說明請查看: CUDA_FIX_GUIDE.txt")
    print("="*60)

if __name__ == "__main__":
    main()
