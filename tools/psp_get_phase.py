import numpy as np
import os
from skimage.io import imread, imsave
from skimage import img_as_float
from scipy.io import savemat
# import matlab.engine
# import matlab
from tqdm import tqdm
from skimage.morphology import disk, erosion

crop_area = (380, 150, 1020, 694)  # 调整为所需的裁剪区域 640 * 544
def crop_image(img, crop_area):
        img = img[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]
        return img

def CPSD(mask_path, img_total, thresh):
    steps = len(img_total[0])
    if thresh < 0:
        mask = np.ones_like(img_total[0][0]).astype(np.uint8)
        return mask
    
    mask = np.zeros_like(img_total[0][0])
    for d in range((steps + 1) // 2):
        for j in range(steps):
            img1 = img_total[0][d]
            img2 = img_total[0][j]
            mask = np.maximum(mask, np.abs(img1 - img2))
    # 二值化处理
    if thresh is None:
        thresh_value = 0
    else:
        thresh_value = thresh
    mask = mask > thresh_value
    
    # 腐蚀处理
    selem = disk(3)
    mask = erosion(mask, selem)
    
    # 将二值化图像转换为 0 和 1
    mask = mask.astype(np.uint8)
    # 保存掩码
    # imsave(mask_path, mask * 255)
    return mask
    


def save_img(img, path, data_type=None):
    """
    根据文件后缀保存图像为.mat或.bmp格式
    
    参数:
    img : numpy.ndarray
        要保存的图像
    path : str
        文件保存路径，包含后缀
    """
    _, ext = os.path.splitext(path)
    if ext == '.mat':
        savemat(path, {data_type: img})
    elif ext == '.bmp':
        if np.min(img) == np.max(img):
            print(f"Warning: Image {path} is a constant image.")
            norm_img = img * 255
            imsave(path, norm_img)
            return
        norm_img = (img-np.min(img)) / (np.max(img)-np.min(img)) * 255
        norm_img = norm_img.astype(np.uint8)
        imsave(path, norm_img)
    else:
        raise ValueError("Unsupported file extension. Supported extensions are .mat and .bmp")

def psp_get_phase(img_total, m, n, save_dir=None, crop_area=None):
    """
    计算四频N步相移法的相位
    
    参数:
    img_total : list
        包含所有相移图像的列表，每个元素都是一个包含N步图像的列表
    m : int
        频率数量
    n : int
        每个频率的相移步数
        
    返回:
    phi : numpy.ndarray
        计算得到的相位
    """
    # 初始化相位矩阵
    phi = np.zeros((img_total[0][0].shape[0], img_total[0][0].shape[1], m))
    
    for i in range(m):
        numerator = np.zeros_like(img_total[0][0])
        denominator = np.zeros_like(img_total[0][0])
        for j in range(n):
            numerator += img_total[i][j] * np.sin(2 * (j) * np.pi / n)
            denominator += img_total[i][j] * np.cos(2 * (j) * np.pi / n)
        phi[:, :, i] = -np.arctan2(numerator, denominator)
        wrap_phase = phi[:, :, i]
        if crop_area is not None:
            wrap_phase = crop_image(phi[:, :, i], crop_area)
            numerator = crop_image(numerator, crop_area)
            denominator = crop_image(denominator, crop_area)

        # 保存分子和分母图像
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_fenzi_mat = os.path.join(save_dir, 'fenzi_mat')
            save_fenzi_bmp = os.path.join(save_dir, 'fenzi_bmp')
            save_fenmu_mat = os.path.join(save_dir, 'fenmu_mat')
            save_fenmu_bmp = os.path.join(save_dir, 'fenmu_bmp')
            save_phase_mat = os.path.join(save_dir, 'phase_mat')
            save_phase_bmp = os.path.join(save_dir, 'phase_bmp')
            save_mask_bmp = os.path.join(save_dir, 'mask')
            os.makedirs(save_fenzi_mat, exist_ok=True)
            os.makedirs(save_fenzi_bmp, exist_ok=True)
            os.makedirs(save_fenmu_mat, exist_ok=True)
            os.makedirs(save_fenmu_bmp, exist_ok=True)
            os.makedirs(save_phase_mat, exist_ok=True)
            os.makedirs(save_phase_bmp, exist_ok=True)
            os.makedirs(save_mask_bmp, exist_ok=True)

            numerator_path_mat = os.path.join(save_fenzi_mat, f'{i+1}.mat')
            denominator_path_mat = os.path.join(save_fenmu_mat, f'{i+1}.mat')
            numerator_path_bmp = os.path.join(save_fenzi_bmp, f'{i+1}.bmp')
            denominator_path_bmp = os.path.join(save_fenmu_bmp, f'{i+1}.bmp')
            phase_path_mat = os.path.join(save_phase_mat, f'{i+1}.mat')
            phase_path_bmp = os.path.join(save_phase_bmp, f'{i+1}.bmp')
            # import pdb; pdb.set_trace()
            save_img(numerator, numerator_path_mat, 'numerator')
            save_img(denominator, denominator_path_mat, 'denominator')
            save_img(wrap_phase, phase_path_mat, 'wrap_phase')
            save_img(numerator, numerator_path_bmp)
            save_img(denominator, denominator_path_bmp)
            save_img(wrap_phase, phase_path_bmp)
            # 生成mask
    mask_path = os.path.join(save_mask_bmp, 'mask.bmp')
    mask = CPSD(mask_path, img_total, 0.18)
    # import pdb; pdb.set_trace()
    if crop_area is not None:
        mask = crop_image(mask, crop_area)
    # # 保存掩码
    save_img(mask, mask_path)
    return phi



def read_images(base_dir, m, n, step):
    
    """
    从文件夹中读取图像并构建img_total列表
    
    参数:
    base_dir : str
        包含所有图像的基础目录
    m : int
        频率数量
    n : int
        每个频率的相移步数
    step : int
        采样步长
        
    返回:
    img_total : list
        包含所有相移图像的列表，每个元素都是一个包含N步图像的列表
    """

    img_total = []
    for freq in range(1, m + 1):
        freq_images = []
        for i in range(n):
            img_index = i * step + 1
            # img_path = os.path.join(base_dir, f'Reconstruction_Polar/PSPImg', f'1_{freq}_{img_index}.bmp')
            img_path = os.path.join(base_dir, f'Reconstruction_Polar/PolarPSPImg/Channel4', f'1_{freq}_{img_index}.bmp')
            # for YM DataSet
            # img_path = os.path.join(base_dir, f'{freq}_{img_index}.bmp')
            img = imread(img_path)
            img = img_as_float(img)  # 将图像转换为浮点数
            freq_images.append(img)
        img_total.append(freq_images)
    return img_total

def process_all_subdirs(base_dir, save_base_dir, m, n, step):
    """
    处理base_dir目录下的所有子目录，并将结果保存到save_base_dir目录中
    
    参数:
    base_dir : str
        包含所有子目录的基础目录
    save_base_dir : str
        保存所有结果的基础目录
    m : int
        频率数量
    n : int
        每个频率的相移步数
    step : int
        采样步长
    """
    for subdir in tqdm(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith('.'):
            print(f"Processing directory: {subdir_path}")
            img_total = read_images(subdir_path, m, n, step)
            if img_total is not None:
                save_dir = os.path.join(save_base_dir, subdir)
                psp_get_phase(img_total, m, n, save_dir=save_dir, crop_area=crop_area)

# 调用matlab
# calDir = '/Volumes/PCL_server/data/structure/metal_dataset/4.24_result'
# eng = matlab.engine.start_matlab()
# eng.addpath(os.path.join(calDir, 'data_analysis_code', 'functions'))

# 示例用法
base_dir = '/home/wangh20/data/structure/metal_dataset/4.24_result'  # 替换为图像文件的实际路径
# base_dir = '/Volumes/Wanghao_SSD/wanghao/dataset/struct_light/Metal_dataset/4.24/plat_result'
# base_dir = '/Volumes/Wanghao_SSD/wanghao/dataset/struct_light/YM_dataset/images'
# save_dir = '/Volumes/Wanghao_SSD/wanghao/dataset/struct_light/Metal_dataset/img_phase_num_den/multi_step/steps_3/metal'  # 替换为保存图像的实际路径
# save_dir = '/Users/wanghao/SynologyDrive/data/Metal_dataset/img_phase_num_den/multi_step/steps_12_origin4'  # 替换为保存图像的实际路径
save_dir = '/home/wangh20/Wanghao_paper/Pipeline_SL/img_phase_num_den/multi_step/steps_12_origin4'  # 替换为保存图像的实际路径
os.makedirs(save_dir, exist_ok=True)

# 参数设置
m = 4  # 频率数量
n = 12  # 每个频率的相移步数
step = 1 # 采样步长
process_all_subdirs(base_dir, save_dir, m, n, step)

# # 四频三步相移法
# m = 4  # 频率数量
# n = 3  # 每个频率的相移步数
# step = 4  # 采样步长
# img_total_3_step = read_images(base_dir, m, n, step)
# phi_3_step = psp_get_phase(img_total_3_step, m, n)

# # 四频四步相移法
# n = 4  # 每个频率的相移步数
# step = 3  # 采样步长
# img_total_4_step = read_images(base_dir, m, n, step)
# phi_4_step = psp_get_phase(img_total_4_step, m, n)

# # 四频六步相移法
# n = 6  # 每个频率的相移步数
# step = 2  # 采样步长
# img_total_6_step = read_images(base_dir, m, n, step)
# phi_6_step = psp_get_phase(img_total_6_step, m, n)

# eng.quit()