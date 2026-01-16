import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from loss import SSIM
# Import model architectures
from utils.data_loading import BasicDataset, BasicDataset_High_Reflect
from scipy.io import savemat
from ResUNet import ResUNet, ResUNet_attention
from unet import UNet, UNet_attention
from torch.cuda.amp import autocast




from evaluate import evaluate

import warnings 
warnings.filterwarnings('ignore')

# 离线运行

# dir_img = Path('/home/wangh20/data/structure/YM_use_dataset/right/images')
# dir_mask = Path('/home/wangh20/data/structure/YM_use_dataset/right/GT')

def normalize_img(img):
    """
    归一化图像数组，使所有值位于[0, 1]范围。
    
    参数:
    img -- 输入的图像数组 (numpy array)。
    
    返回:
    归一化的图像数组。
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return img
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm


def test_model(
        model_student,
        model_result,
        device,
        dir_img,
        dir_mask,
        img_scale=1.0,
        save_dir='.'
):  
    
    # 1. Create dataset
    # Use local datasets directory
    try:
        metal_dataset = BasicDataset(args.dir_img, args.dir_mask)
    except (AssertionError, RuntimeError, IndexError):
        metal_dataset = BasicDataset(args.dir_img, args.dir_mask)

    # 2. Split into train / validation partitions
    n_train = int(len(metal_dataset) * 0.8)
    n_val = int(len(metal_dataset) * 0.1)
    n_test = len(metal_dataset) - n_val - n_train
    
    _, _, test_set = random_split(metal_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))  
    # Create data loaders
    test_loader = DataLoader(metal_dataset, shuffle=False, batch_size=1, drop_last=True, num_workers=16)
    # test_loader = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=True, num_workers=16)
    # save img dir
    save_aug_dir = os.path.join(save_dir, 'imgs_aug')
    save_rec_dir = os.path.join(save_dir, 'rec')
    save_img_dir = os.path.join(save_dir, 'imgs_GT')

    save_gt_dir = os.path.join(save_dir, 'GT')
    save_error_dir = os.path.join(save_dir, 'error')
    save_pred_dir = os.path.join(save_dir, 'pred')
    save_fenzi_GT_dir = os.path.join(save_gt_dir, 'fenzi')
    save_fenzi_pred_dir = os.path.join(save_pred_dir, 'fenzi')
    save_fenzi_error_dir = os.path.join(save_error_dir, 'fenzi')
    save_fenmu_GT_dir = os.path.join(save_gt_dir, 'fenmu')
    save_fenmu_pred_dir = os.path.join(save_pred_dir, 'fenmu')
    save_fenmu_error_dir = os.path.join(save_error_dir, 'fenmu')


    os.makedirs(save_aug_dir, exist_ok=True)
    os.makedirs(save_rec_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    os.makedirs(save_gt_dir, exist_ok=True)
    os.makedirs(save_error_dir, exist_ok=True)
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_fenzi_GT_dir, exist_ok=True)
    os.makedirs(save_fenzi_pred_dir, exist_ok=True)
    os.makedirs(save_fenzi_error_dir, exist_ok=True)
    os.makedirs(save_fenmu_GT_dir, exist_ok=True)
    os.makedirs(save_fenmu_pred_dir, exist_ok=True)
    os.makedirs(save_fenmu_error_dir, exist_ok=True)

    # save mat dir 
    save_fenzi_mat_dir = os.path.join(save_dir, 'fenzi_mat')
    save_fenmu_mat_dir = os.path.join(save_dir, 'fenmu_mat')
    os.makedirs(save_fenzi_mat_dir, exist_ok=True)
    os.makedirs(save_fenmu_mat_dir, exist_ok=True)



    criterion_MAE = torch.nn.L1Loss()
    criterion_MSE = torch.nn.MSELoss()

    model_student.eval().to(device=device)
    model_result.eval().to(device=device)

    loss_scatter = []
    epoch_loss = 0
    epoch_loss_L1 = 0
    epoch_loss_L2 = 0
    # val_score_MSE, val_score_MAE = evaluate(model_student, model_result, test_loader, device)
    # print(val_score_MAE)
    # import pdb; pdb.set_trace()

    name_list = []
    for index, batch in tqdm(enumerate(test_loader), desc="A Processing Bar Test: "):
        images, true_masks = batch['image'], batch['mask']
        images_aug = batch['image_aug']
        name = batch['name'][0]
        name_list.append(int(name))
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        images_aug = images_aug.to(device=device, dtype=torch.float32)
        # eval 
        pred, _ = model_student(images_aug)
        joined_in = torch.cat((pred, images_aug), dim=1)
        masks_pred, _ = model_result(joined_in)

        l2_loss = torch.sqrt(criterion_MSE(masks_pred, true_masks))
        l1_loss = criterion_MAE(masks_pred, true_masks)
        loss = l1_loss + l2_loss    
        # logging.info(f'Validation score L1+L2: {round(loss.item(), 4)} = {round(l1_loss.item(), 4)} + {round(l2_loss.item(), 4)}')
        epoch_loss += loss.item()
        epoch_loss_L1 += l1_loss.item()
        epoch_loss_L2 += l2_loss.item()
        loss_scatter.append(l1_loss.item())
    
        error_map = np.abs((masks_pred - true_masks).cpu().detach().numpy())
        error_map = error_map.squeeze()
        mask_pred_tensor = masks_pred.squeeze().cpu().detach().numpy()
        mask_true_test = true_masks.squeeze().cpu().detach().numpy()
        images_test = images.squeeze().cpu().detach().numpy()
        images_aug_test = images_aug.squeeze().cpu().detach().numpy()
        images_rec = pred.squeeze().cpu().detach().numpy()

        # new_each_img
        save_name_dir = os.path.join(save_dir, 'each_img', name)
        os.makedirs(save_name_dir, exist_ok=True)
        save_name_dir_fenzi_mat = os.path.join(save_name_dir, 'fenzi_mat')
        save_name_dir_fenmu_mat = os.path.join(save_name_dir, 'fenmu_mat')
        save_name_dir_phase_mat = os.path.join(save_name_dir, 'phase_mat')
        os.makedirs(save_name_dir_fenzi_mat, exist_ok=True)
        os.makedirs(save_name_dir_fenmu_mat, exist_ok=True)
        os.makedirs(save_name_dir_phase_mat, exist_ok=True)

        for i in range(4):

            # images and images aug
            images_i = images_test[i]
            images_aug_low_i = images_aug_test[2 * i]
            images_aug_high_i = images_aug_test[2 * i + 1]
            images_rec_i = images_rec[i]

            pred_fenzi_i = mask_pred_tensor[2 * i]
            pred_fenmu_i = mask_pred_tensor[2 * i + 1]
            pred_phase_i = -np.arctan2(pred_fenzi_i, pred_fenmu_i)
            error_fenzi_i = error_map[2 * i]
            error_fenmu_i = error_map[2 * i + 1]
            GT_fenzi_i = mask_true_test[2 * i]
            GT_fenmu_i = mask_true_test[2 * i + 1]

            pred_fenzi_mat = {'numerator': pred_fenzi_i}
            pred_fenmu_mat = {'denominator': pred_fenmu_i}
            pred_phase_mat = {'wrap_phase': pred_phase_i}
            # 归一化
            pred_fenzi_i = normalize_img(pred_fenzi_i)
            pred_fenmu_i = normalize_img(pred_fenmu_i)
            # error_fenzi_i = normalize_img(error_fenzi_i)
            # error_fenmu_i = normalize_img(error_fenmu_i)
            GT_fenzi_i = normalize_img(GT_fenzi_i)
            GT_fenmu_i = normalize_img(GT_fenmu_i)
            # save img
            name = str(name)
            # save AUG image
            Image.fromarray((images_i * 255).astype('uint8')).save(os.path.join(save_img_dir, name + f'_{i}_ori.bmp'))
            Image.fromarray((images_rec_i * 255).astype('uint8')).save(os.path.join(save_rec_dir, name + f'_{i}_rec.bmp'))
            Image.fromarray((images_aug_low_i * 255).astype('uint8')).save(os.path.join(save_aug_dir, name + f'_{i}_low.bmp'))
            Image.fromarray((images_aug_high_i * 255).astype('uint8')).save(os.path.join(save_aug_dir, name + f'_{i}_high.bmp'))

            # save results
            Image.fromarray((GT_fenzi_i * 255).astype('uint8')).save(os.path.join(save_fenzi_GT_dir, name + f'_{i}_GT.bmp'))
            Image.fromarray((GT_fenmu_i * 255).astype('uint8')).save(os.path.join(save_fenmu_GT_dir, name + f'_{i}_GT.bmp'))
            Image.fromarray((pred_fenzi_i * 255).astype('uint8')).save(os.path.join(save_fenzi_pred_dir, name + f'_{i}_pred.bmp'))
            Image.fromarray((pred_fenmu_i * 255).astype('uint8')).save(os.path.join(save_fenmu_pred_dir, name + f'_{i}_pred.bmp'))
            Image.fromarray((error_fenzi_i * 255).astype('uint8')).save(os.path.join(save_fenzi_error_dir, name + f'_{i}_error.bmp'))
            Image.fromarray((error_fenmu_i * 255).astype('uint8')).save(os.path.join(save_fenmu_error_dir, name + f'_{i}_error.bmp'))
            # save mat
            savemat(os.path.join(save_name_dir_fenzi_mat, f'{i+1}.mat'), pred_fenzi_mat)
            savemat(os.path.join(save_name_dir_fenmu_mat, f'{i+1}.mat'), pred_fenmu_mat)
            savemat(os.path.join(save_name_dir_phase_mat, f'{i+1}.mat'), pred_phase_mat)

    print(sorted(name_list))
    logging.info(f'Test score: {round(epoch_loss / len(test_loader), 4)}')
    logging.info(f'Test L1: {round(epoch_loss_L1 / len(test_loader), 4)}')
    logging.info(f'Test L2: {round(epoch_loss_L2 / len(test_loader), 4)}')
    num_blow = len([i for i in loss_scatter if i <= 0.02])
    max_index = np.argmax(loss_scatter)
    plt.figure()
    plt.scatter(range(len(loss_scatter)), loss_scatter, s=1)
    plt.title(f'Loss scatter max index{max_index} num blow {num_blow}')
    plt.savefig(os.path.join(save_dir, f'loss_scatter.png'))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--gpu-id', type=str, default='0', help='choose gpu id')
    parser.add_argument('--dir-img', type=str, required=True)
    parser.add_argument('--dir-mask', type=str, required=True)
    parser.add_argument('--scale', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--save-dir', type=str, default='./')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model_student = UNet(8, 4, False)
    # model_student = UNet_attention(8, 4, False)
    model_result = UNet(12, 8, False)
    # import pdb; pdb.set_trace()

    model_student.load_state_dict(torch.load(os.path.join(args.load, 'student.pth'), map_location=device))
    model_result.load_state_dict(torch.load(os.path.join(args.load, 'result.pth'), map_location=device))
    logging.info(f'Model loaded from {args.load}')
    
    # model_student.to(device=device)
    # model_result.to(device=device)
    try:
        test_model(
            model_student=model_student,
            model_result=model_result,
            device=device,
            dir_img=args.dir_img,
            dir_mask=args.dir_mask,
            img_scale=args.scale,
            save_dir=args.save_dir
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()