# HDRSL 運行指南

## ✅ 最近更新

**2026-01-16**: 
- ✅ 修復了 `model_with_attention.py` 和 `Attention_module.py` 的導入錯誤
- ✅ 添加了缺失的 `CBAMBlock` 注意力模塊
- ✅ 移除了對不存在類的導入引用
- ✅ 所有依賴項現在可以正常導入

## 📋 環境準備

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 數據集結構

確保數據集已下載到 `datasets/` 目錄，結構如下：

```
datasets/
├── images_GT/          # 原始GT圖像（每個樣本一個子目錄）
├── images_low/         # 低曝光增強圖像
├── images_4/           # 高曝光增強圖像
├── fenzi_GT_mat_2/     # 正弦分量GT（.mat格式）
├── fenmu_GT_mat_2/     # 餘弦分量GT（.mat格式）
└── Phases_GT_mat/      # 相位GT（.mat格式）
```

## 🚀 快速開始

### 訓練模型

#### 方法1：使用批處理腳本（Windows）

雙擊運行 `train.bat`

或在命令行中：
```bash
train.bat
```

#### 方法2：直接使用Python

```bash
python train.py ^
    --dir_img "datasets/images_GT" ^
    --dir_mask "datasets" ^
    --save_checkpoint_path "checkpoints" ^
    --gpu_id 0 ^
    --batch_size 4 ^
    --epochs 100 ^
    --learning_rate 1e-5 ^
    --validation 10
```

### 測試模型

#### 方法1：使用批處理腳本（Windows）

雙擊運行 `test.bat`

或在命令行中：
```bash
test.bat
```

#### 方法2：直接使用Python

```bash
python test.py ^
    --dir-img "datasets/images_GT" ^
    --dir-mask "datasets" ^
    --load "checkpoints" ^
    --save-dir "results" ^
    --gpu-id 0 ^
    --scale 1.0
```

## ⚙️ 參數說明

### 訓練參數 (train.py)

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--dir_img` | 輸入圖像目錄 | 必需 |
| `--dir_mask` | Ground Truth目錄 | 必需 |
| `--save_checkpoint_path` | 模型保存路徑 | 必需 |
| `--gpu_id` | GPU編號 | 0 |
| `--batch_size` | 批次大小 | 4 |
| `--epochs` | 訓練輪數 | 5 |
| `--learning_rate` | 學習率 | 1e-5 |
| `--validation` | 驗證集比例(%) | 10 |
| `--load_checkpoint` | 是否加載已有模型 | False |

### 測試參數 (test.py)

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--dir-img` | 輸入圖像目錄 | 必需 |
| `--dir-mask` | Ground Truth目錄 | 必需 |
| `--load` | 模型checkpoint路徑 | 必需 |
| `--save-dir` | 結果保存目錄 | ./ |
| `--gpu-id` | GPU編號 | 0 |
| `--scale` | 圖像縮放比例 | 1.0 |

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
```

## 🏗️ 模型架構

項目使用**Teacher-Student知識蒸餾架構**：

- **Student Model**: UNet_attention (4→4 channels)
  - 處理增強圖像
  - 帶注意力機制
  
- **Teacher Model**: UNet (4→4 channels)
  - 處理原始GT圖像
  - 用於知識蒸餾
  
- **Result Model**: UNet (8→8 channels)
  - 融合student輸出和增強圖像
  - 生成最終結果

## 🔧 常見問題

### 1. CUDA Out of Memory
- 減小 `batch_size` (例如從4改為2或1)
- 降低圖像分辨率 (修改 `--scale` 參數)

### 2. 數據集路徑錯誤
- 確保 `datasets/` 目錄存在
- 檢查子目錄結構是否完整

### 3. 找不到checkpoint
- 訓練前測試：先運行 `train.bat` 生成模型
- 檢查 `checkpoints/` 目錄是否包含 `.pth` 文件

### 4. WandB離線模式
代碼已設置為離線模式（`os.environ['WANDB_MODE'] = 'dryrun'`），不需要WandB賬號。

## 📊 數據集

原始數據集包含1700組金屬結構光數據，涵蓋多種材料、幾何形狀和標準/非標準零件。

下載地址：https://wangh257.github.io/HDRSL/Data_Download.html

## 📝 引用

如使用此代碼，請引用原論文：
```
HDRSL Net for Accurate High Dynamic Range Imaging-based Structure Light 3D Reconstruction
```

## ⚖️ 許可證

請參考原項目的許可證要求。
