"""
模拟训练过程进行调试
"""
import torch
import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作以获得准确的错误位置

from torch.utils.data import DataLoader, random_split
from models.unet import UNet, UNet_attention
from models.loss import SSIM, MseDirectionLoss
from utils.data_loading import BasicDataset_High_Reflect

print("="*60)
print("模拟训练过程 - 调试 Index Out of Bounds")
print("="*60)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 创建数据集
print("\n创建数据集...")
dataset = BasicDataset_High_Reflect("datasets/images_GT", "datasets")
print(f"数据集大小: {len(dataset)}")

# 分割数据集
n_train = int(len(dataset) * 0.8)
n_val = int(len(dataset) * 0.1)
n_test = len(dataset) - n_val - n_train

train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test], 
    generator=torch.Generator().manual_seed(0)
)
print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

# 创建DataLoader
print("\n创建DataLoader...")
train_loader = DataLoader(train_set, shuffle=True, batch_size=4, num_workers=0)
print(f"批次数: {len(train_loader)}")

# 创建模型
print("\n创建模型...")
model_student = UNet_attention(4, 4, False).to(device)
model_teacher = UNet(4, 4, False).to(device)
model_result = UNet(8, 8, False).to(device)
print("模型创建完成")

# 创建损失函数
print("\n创建损失函数...")
criterion_MSE = torch.nn.MSELoss()
criterion_MAE = torch.nn.L1Loss()
criterion_SSIM = SSIM(device=device)
Distillation_loss = MseDirectionLoss(0.1)
print("损失函数创建完成")

# 测试第一个batch
print("\n"+"="*60)
print("测试第一个batch")
print("="*60)

try:
    print("\n获取第一个batch...")
    batch = next(iter(train_loader))
    
    print(f"Batch 数据形状:")
    print(f"  image: {batch['image'].shape}")
    print(f"  image_aug: {batch['image_aug'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    
    # 移到设备
    print("\n移动数据到设备...")
    imgs = batch['image'].to(device=device, dtype=torch.float32)
    true_masks = batch['mask'].to(device=device, dtype=torch.float32)
    imgs_aug = batch['image_aug'].to(device=device, dtype=torch.float32)
    print("数据已移动到设备")
    
    print(f"\n在设备上的数据形状:")
    print(f"  imgs: {imgs.shape}")
    print(f"  true_masks: {true_masks.shape}")
    print(f"  imgs_aug: {imgs_aug.shape}")
    
    # 前向传播
    print("\n开始前向传播...")
    
    print("  1. Student model...")
    pred, fea_student = model_student(imgs_aug)
    print(f"     pred: {pred.shape}")
    print(f"     fea_student: list of {len(fea_student)} tensors")
    
    print("  2. Teacher model...")
    pred_teacher, fea_teacher = model_teacher(imgs)
    print(f"     pred_teacher: {pred_teacher.shape}")
    print(f"     fea_teacher: list of {len(fea_teacher)} tensors")
    
    print("  3. Concatenate...")
    joined_in = torch.cat([pred, imgs_aug], dim=1)
    print(f"     joined_in: {joined_in.shape}")
    
    print("  4. Result model...")
    out, _ = model_result(joined_in)
    print(f"     out: {out.shape}")
    
    # 计算损失
    print("\n计算损失...")
    
    print("  1. Student losses...")
    l2_loss_student = torch.sqrt(criterion_MSE(pred, imgs))
    print(f"     L2: {l2_loss_student.item():.4f}")
    
    l1_loss_student = criterion_MAE(pred, imgs)
    print(f"     L1: {l1_loss_student.item():.4f}")
    
    print("  2. SSIM loss...")
    ssim_student = criterion_SSIM(pred, imgs)
    print(f"     SSIM: {ssim_student.item():.4f}")
    
    print("  3. Distillation loss...")
    distillation_loss = Distillation_loss(fea_student, fea_teacher)
    print(f"     Distillation: {distillation_loss.item():.4f}")
    
    print("  4. Teacher losses...")
    l2_loss_teacher = torch.sqrt(criterion_MSE(pred_teacher, imgs))
    l1_loss_teacher = criterion_MAE(pred_teacher, imgs)
    ssim_teacher = criterion_SSIM(pred_teacher, imgs)
    print(f"     Teacher total: {(l2_loss_teacher + l1_loss_teacher + ssim_teacher).item():.4f}")
    
    print("  5. Result losses...")
    l2_loss_result = torch.sqrt(criterion_MSE(out, true_masks))
    print(f"     L2: {l2_loss_result.item():.4f}")
    
    l1_loss_result = criterion_MAE(out, true_masks)
    print(f"     L1: {l1_loss_result.item():.4f}")
    
    print("\n✓ 第一个batch测试成功！")
    
except Exception as e:
    print(f"\n✗ 错误发生!")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    
    import traceback
    print("\n完整堆栈:")
    traceback.print_exc()
    
    # 尝试获取更多信息
    if torch.cuda.is_available():
        print(f"\nCUDA 错误检查:")
        try:
            torch.cuda.synchronize()
        except Exception as cuda_e:
            print(f"CUDA同步错误: {cuda_e}")

print("\n"+"="*60)
print("调试完成")
print("="*60)
