"""
数据集调试工具 - 检查数据加载问题
"""
import os
import sys
# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from utils.data_loading import BasicDataset, BasicDataset_High_Reflect

def check_dataset_structure():
    """检查数据集目录结构"""
    print("\n" + "="*60)
    print("检查数据集目录结构")
    print("="*60)
    
    dataset_dir = "datasets"
    
    expected_dirs = {
        "images_GT": "原始GT图像",
        "images_low": "低曝光图像",
        "images_4": "高曝光图像",
        "fenzi_GT_mat_2": "正弦分量 (Numerator)",
        "fenmu_GT_mat_2": "余弦分量 (Denominator)",
        "Phases_GT_mat": "相位GT (可选)",
    }
    
    existing_dirs = os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []
    
    print(f"\n在 {dataset_dir}/ 中找到:")
    for dir_name in existing_dirs:
        path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(path):
            desc = expected_dirs.get(dir_name, "未知")
            print(f"  ✓ {dir_name:20s} - {desc}")
    
    print(f"\n缺少:")
    for dir_name, desc in expected_dirs.items():
        if dir_name not in existing_dirs:
            print(f"  ✗ {dir_name:20s} - {desc}")

def check_single_sample():
    """检查单个样本"""
    print("\n" + "="*60)
    print("检查单个样本数据")
    print("="*60)
    
    dir_img = "datasets/images_GT"
    dir_mask = "datasets"
    
    # 获取第一个样本
    sub_dirs = [d for d in os.listdir(dir_img) 
                if os.path.isdir(os.path.join(dir_img, d)) and not d.startswith('.')]
    
    if not sub_dirs:
        print("✗ 没有找到样本!")
        return
    
    sample_name = sub_dirs[0]
    print(f"\n检查样本: {sample_name}")
    
    # 检查图像文件
    img_dir = os.path.join(dir_img, sample_name)
    expected_imgs = [f"{sample_name}_{i}.bmp" for i in [1, 2, 3, 4]]
    
    print(f"\n图像文件 (在 {img_dir}):")
    for img_name in expected_imgs:
        img_path = os.path.join(img_dir, img_name)
        exists = os.path.exists(img_path)
        print(f"  {'✓' if exists else '✗'} {img_name}")
    
    # 检查mat文件
    print(f"\n正弦分量文件 (fenzi):")
    for i in [1, 2, 3, 4]:
        # 检查fenzi_GT_mat_2目录
        mat_name = f"{sample_name}-{i}.mat"
        path = os.path.join("datasets/fenzi_GT_mat_2", mat_name)
        
        exists = os.path.exists(path)
        
        if exists:
            print(f"  ✓ {mat_name:15s} (在 fenzi_GT_mat_2/)")
        else:
            print(f"  ✗ {mat_name:15s} (不存在)")
    
    print(f"\n余弦分量文件 (fenmu):")
    for i in [1, 2, 3, 4]:
        mat_name = f"{sample_name}-{i}.mat"
        path = os.path.join("datasets/fenmu_GT_mat_2", mat_name)
        
        exists = os.path.exists(path)
        
        if exists:
            print(f"  ✓ {mat_name:15s} (在 fenmu_GT_mat_2/)")
        else:
            print(f"  ✗ {mat_name:15s} (不存在)")

def test_dataloader():
    """测试数据加载器"""
    print("\n" + "="*60)
    print("测试数据加载器")
    print("="*60)
    
    try:
        print("\n尝试创建数据集...")
        dataset = BasicDataset_High_Reflect("datasets/images_GT", "datasets")
        print(f"✓ 数据集创建成功")
        print(f"  样本数量: {len(dataset)}")
        
        print("\n尝试加载第一个样本...")
        sample = dataset[0]
        
        print(f"✓ 第一个样本加载成功")
        print(f"  样本名称: {sample['name']}")
        print(f"  image 形状: {sample['image'].shape}")
        print(f"  image_aug 形状: {sample['image_aug'].shape}")
        print(f"  mask 形状: {sample['mask'].shape}")
        
        print(f"\n数据范围检查:")
        print(f"  image: min={sample['image'].min():.4f}, max={sample['image'].max():.4f}")
        print(f"  image_aug: min={sample['image_aug'].min():.4f}, max={sample['image_aug'].max():.4f}")
        print(f"  mask: min={sample['mask'].min():.4f}, max={sample['mask'].max():.4f}")
        
        # 检查是否有nan或inf
        has_nan = torch.isnan(sample['image']).any() or torch.isnan(sample['image_aug']).any() or torch.isnan(sample['mask']).any()
        has_inf = torch.isinf(sample['image']).any() or torch.isinf(sample['image_aug']).any() or torch.isinf(sample['mask']).any()
        
        if has_nan:
            print(f"  ⚠️  发现 NaN 值!")
        if has_inf:
            print(f"  ⚠️  发现 Inf 值!")
        
        if not has_nan and not has_inf:
            print(f"  ✓ 没有 NaN 或 Inf 值")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_batch():
    """测试批量数据加载"""
    print("\n" + "="*60)
    print("测试批量数据加载")
    print("="*60)
    
    try:
        from torch.utils.data import DataLoader
        
        print("\n创建数据集和数据加载器...")
        dataset = BasicDataset_High_Reflect("datasets/images_GT", "datasets")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        print(f"✓ 数据加载器创建成功")
        print(f"  批次大小: 4")
        print(f"  总批次数: {len(dataloader)}")
        
        print("\n加载第一个批次...")
        batch = next(iter(dataloader))
        
        print(f"✓ 第一个批次加载成功")
        print(f"  image 形状: {batch['image'].shape}")
        print(f"  image_aug 形状: {batch['image_aug'].shape}")
        print(f"  mask 形状: {batch['mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("HDRSL 数据集调试工具")
    print("="*60)
    
    # 检查目录结构
    check_dataset_structure()
    
    # 检查单个样本
    check_single_sample()
    
    # 测试数据加载器
    if test_dataloader():
        # 测试批量加载
        test_dataloader_batch()
    
    print("\n" + "="*60)
    print("调试完成")
    print("="*60)

if __name__ == "__main__":
    main()
