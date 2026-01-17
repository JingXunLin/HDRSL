# HDRSL Net for Accurate High Dynamic Range Imaging-based Structure Light 3D Reconstruction

## Overview of HDRSL

<div style="text-align: justify; text-indent: 2em;">
HDRSL (High Dynamic Range Imaging-based Structure Light) is a deep learning approach aimed at high-accuracy 3D reconstruction using structured light, specifically addressing high dynamic range challenges in metallic and reflective surfaces.
</div>

## Environment Setup

To reproduce the results, please follow the steps below to set up your environment.

### 1. Prerequisites
- **OS**: Linux or Windows
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Conda**: Anaconda or Miniconda installed

### 2. Create Conda Environment
```bash
conda create -n hdrsl python=3.8
conda activate hdrsl
```

### 3. Install Dependencies

First, install PyTorch (adjust the CUDA version according to your GPU):
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

```text
HDRSL/
├── models/             # Neural network model definitions (UNet, ResUNet, etc.)
├── scripts/            # Helper scripts (training shells, verification, debug)
├── utils/              # Utility functions (data loading, etc.)
├── datasets/           # Dataset directory (see below)
├── checkpoints/        # Saved model weights
├── train.py            # Main training entry point
├── test.py             # Main testing entry point
└── requirements.txt    # Python dependencies
```

## Dataset Setup

1.  **Download**: Click <a href="https://wangh257.github.io/HDRSL/Data_Download.html" target="_blank">here</a> to download the dataset.
2.  **Organize**: Extract the dataset into the `datasets/` folder in the root of this project.

The expected folder structure is:
```text
HDRSL/
└── datasets/
    ├── images_GT/         # Teacher input images
    │   ├── 000001/
    │   └── ...
    ├── images_low/        # Low exposure images (Student input)
    │   ├── 000001/
    │   └── ...
    ├── images_4/          # High exposure images (Student input for dual mode)
    │   ├── 000001/
    │   └── ...
    ├── fenzi_GT_mat_2/    # Numerator Ground Truth (MAT files)
    ├── fenmu_GT_mat_2/    # Denominator Ground Truth (MAT files)
    └── Phases_GT_mat/     # Phase Ground Truth (MAT files, optional)
```

## Usage

This repository supports two training modes:
1.  **Single Exposure Mode**: Uses only low-exposure images (4 channels input). You can use ```wget```, and then unzip the file. 
2.  **Dual Exposure Mode**: Uses both low and high-exposure images (8 channels input), matching the original paper's configuration.

### Verify Environment

```
python verify_setup.py 
python debug_dataset.py 
python debug_train.py 
```

### Training

**Option 1: Single Exposure Mode (4-channel)**
```bash
python train.py \
    --dir_img datasets/images_GT \
    --dir_mask datasets \
    --save_checkpoint_path checkpoints \
    --batch_size 4 \
    --epochs 100 \
    --gpu_id 0
```
*Or use the provided script (Linux/Bash):*
```bash
bash train.sh
```

**Option 2: Dual Exposure Mode (8-channel)**
```bash
python train.py \
    --dir_img datasets/images_GT \
    --dir_mask datasets \
    --save_checkpoint_path checkpoints_dual \
    --dual_exposure \
    --batch_size 4 \
    --epochs 100 \
    --gpu_id 0
```
*Or use the provided script (Linux/Bash):*
```bash
bash train_dual.sh
```

### Testing / Evaluation

**Option 1: Single Exposure Mode**
```bash
python test.py \
    --dir_img datasets/images_GT \
    --dir_mask datasets \
    --load_checkpoint checkpoints \
    --gpu_id 0
```
*Or use the provided script:*
```bash
bash test.sh
```

**Option 2: Dual Exposure Mode**
```bash
python test.py \
    --dir_img datasets/images_GT \
    --dir_mask datasets \
    --load_checkpoint checkpoints_dual \
    --dual_exposure \
    --gpu_id 0
```
*Or use the provided script:*
```bash
bash test_dual.sh
```

## Experimental Results

The metrics used for evaluation are:
- **L1 Loss (MAE)**: Mean Absolute Error between predicted and ground truth phase numerator/denominator.
- **L2 Loss (RMSE)**: Root Mean Square Error.
- **SSIM**: Structural Similarity Index.

<!-- <div style="text-align: center;">
  <p>Table 1. MAE of sine and cosine components, wrapped phase, and absolute phase.</p>
  <img src="images/table1.png" alt="matal_dataset">
</div>

<div style="text-align: center;">
  <img src="images/fig_metal.png" alt="matal_dataset">
  <p>Fig 3. matal dataset result</p>
</div> -->

## 📁 輸出文件

### 訓練輸出 (checkpoints/)
- `student_checkpoint.pth` - Student網絡權重
- `teacher_checkpoint.pth` - Teacher網絡權重
- `result_checkpoint.pth` - Result網絡權重

### 測試輸出 (results/)
```
results/
├── imgs_aug/           # 增強圖像
├── imgs_GT/            # 原始GT圖像
├── rec/                # 重建圖像
├── GT/                 # Ground Truth
│   ├── fenzi/          # 正弦分量GT
│   └── fenmu/          # 餘弦分量GT
├── pred/               # 預測結果
│   ├── fenzi/          # 正弦分量預測
│   └── fenmu/          # 餘弦分量預測
├── error/              # 誤差圖
│   ├── fenzi/
│   └── fenmu/
├── fenzi_mat/          # 正弦分量.mat文件
├── fenmu_mat/          # 餘弦分量.mat文件
└── loss_scatter.png    # Loss散點圖