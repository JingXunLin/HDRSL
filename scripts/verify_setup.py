"""
HDRSL Setup Verification Script
验证环境配置是否正确
"""
import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_imports():
    """检查所有必要的导入"""
    print("=" * 60)
    print("检查Python包导入...")
    print("=" * 60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('h5py', 'h5py'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
        ('wandb', 'Weights & Biases'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name:20s} - 已安裝")
        except ImportError:
            print(f"✗ {name:20s} - 缺失")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  缺少以下包: {', '.join(missing)}")
        print("請運行: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ 所有必要的包都已安裝")
        return True

def check_models():
    """检查模型定义"""
    print("\n" + "=" * 60)
    print("檢查模型定義...")
    print("=" * 60)

    try:
        from models.unet import UNet, UNet_attention
        print("✓ UNet 模型導入成功")

        from models.ResUNet import ResUNet, ResUNet_attention
        print("✓ ResUNet 模型導入成功")

        from models.Attention_module import CBAMBlock, ChannelAttention_WH, SpatialAttention_WH
        print("✓ 注意力模塊導入成功")

        from models.loss import SSIM, MseDirectionLoss
        print("✓ 損失函數導入成功")

        from utils.data_loading import BasicDataset, BasicDataset_High_Reflect
        print("✓ 數據加載器導入成功")
        
        print("\n✓ 所有模型和模塊導入正常")
        return True
    except ImportError as e:
        print(f"\n✗ 導入錯誤: {e}")
        return False

def check_dataset():
    """检查数据集"""
    print("\n" + "=" * 60)
    print("檢查數據集...")
    print("=" * 60)
    
    dataset_dir = "datasets"
    required_dirs = [
        "images_GT",
        "images_low", 
        "images_4",
        "fenzi_GT_mat_2",
        "fenmu_GT_mat_2"
    ]
    
    if not os.path.exists(dataset_dir):
        print(f"✗ 數據集目錄不存在: {dataset_dir}")
        print("請將數據集下載到 datasets/ 目錄")
        return False
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_dir, dir_name)
        if os.path.exists(dir_path):
            # 检查目录是否有内容
            if dir_name == "images_GT":
                subdirs = [d for d in os.listdir(dir_path) 
                          if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith('.')]
                if subdirs:
                    print(f"✓ {dir_name:20s} - 找到 {len(subdirs)} 個樣本")
                else:
                    print(f"⚠ {dir_name:20s} - 目錄為空")
            else:
                files = os.listdir(dir_path)
                print(f"✓ {dir_name:20s} - 找到 {len(files)} 個文件")
        else:
            print(f"✗ {dir_name:20s} - 不存在")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  缺少以下目錄: {', '.join(missing_dirs)}")
        return False
    else:
        print("\n✓ 數據集結構完整")
        return True

def check_cuda():
    """检查CUDA可用性"""
    print("\n" + "=" * 60)
    print("檢查GPU/CUDA...")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  GPU 數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA 不可用，將使用CPU訓練（速度較慢）")
            return True
    except Exception as e:
        print(f"✗ 檢查CUDA時出錯: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("HDRSL 環境配置驗證")
    print("=" * 60)
    
    results = {
        "包導入": check_imports(),
        "模型定義": check_models(),
        "數據集": check_dataset(),
        "GPU/CUDA": check_cuda()
    }
    
    print("\n" + "=" * 60)
    print("驗證總結")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{check_name:15s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("\n🎉 所有檢查通過！您可以開始訓練了。")
        print("\n運行訓練:")
        print("  Windows: train.bat")
        print("  或: python train.py --dir_img datasets/images_GT --dir_mask datasets --save_checkpoint_path checkpoints")
    else:
        print("\n⚠️  部分檢查失敗，請根據上述提示修復問題。")
        sys.exit(1)

if __name__ == "__main__":
    main()
