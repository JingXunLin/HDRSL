import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast


import warnings 
warnings.filterwarnings('ignore')

# MSE_loss = torch.nn.modules.loss.MSELoss(mode='mean')
MSE_loss = torch.nn.MSELoss()
MAE_loss = torch.nn.modules.loss.L1Loss()
# @torch.inference_mode()
def evaluate(model_student, net, dataloader, device):
    net.eval()
    model_student.eval()
    num_val_batches = len(dataloader)
    # dice_score = 0
    total_loss_MSE = 0
    total_loss_MAE = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, image_aug = batch['image'], batch['mask'], batch['image_aug']
        # move images and labels to correct device and type
        image = image.to(device=device)
        mask_true = mask_true.to(device=device)
        image_aug = image_aug.to(device=device)
        # predict the mask
        pred, _ = model_student(image_aug)
        joined_in = torch.cat((pred, image_aug), dim=1)
        mask_pred, _ = net(joined_in)
        total_loss_MSE += torch.sqrt(MSE_loss(mask_pred, mask_true)).detach().cpu()
        total_loss_MAE += MAE_loss(mask_pred, mask_true).detach().cpu()
    net.train()
    model_student.train()
    return total_loss_MSE / max(num_val_batches, 1), total_loss_MAE / max(num_val_batches, 1)

def evaluate_generation(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    # dice_score = 0
    total_loss_MSE = 0
    total_loss_MAE = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image_aug'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device)
        mask_true = mask_true.to(device=device)
        # predict the mask
        mask_pred, _ = net(image)
        total_loss_MSE += torch.sqrt(MSE_loss(mask_pred, mask_true)).detach().cpu()
        total_loss_MAE += MAE_loss(mask_pred, mask_true).detach().cpu()
    net.train()
    return total_loss_MSE / max(num_val_batches, 1), total_loss_MAE / max(num_val_batches, 1)
