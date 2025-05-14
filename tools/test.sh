# checkpoint='./checkpoints/fenzi_fenmu_input_1/add_plat_qiu/UNet_4/bs3_lr0.001_400'
checkpoint='./checkpoints/HDR_fenzi_fenmu/Distillation/add_plat_qiu/UNet_14/bs5_lr0.0002_600'
checkpoint='./checkpoints_1/HDR_fenzi_fenmu/Attention/add_plat_qiu/UNet_14/bs2_lr0.001_1000'
checkpoint='./checkpoints_1/HDR_fenzi_fenmu/Distillation_attention/add_plat_qiu/UNet_14/bs3_lr0.001_1000'
checkpoint='./checkpoints_YM/UNet/YM_Data/bs8_lr0.001_600'
# checkpoint='./checkpoints_1/HDR_input_1/Attention_Distillation/add_plat_qiu/UNet_1/bs3_lr0.001_420'
# checkpoint='./checkpoints_1/HDR_input_1/Attention_Distillation/add_plat_qiu/UNet_1/bs3_lr0.001_420'
# checkpoint='./checkpoints_1/HDR_fenzi_fenmu/Attention/add_plat_qiu/UNet_14/bs2_lr0.001_600'
# checkpoint='./checkpoints/fenzi_fenmu/add_plat_qiu/UNet_14/bs3_lr0.001_400'
# save_dir='./results/fenzi_fenmu/multi_model/CBAM_attention/bs8_lr0.001_200'
# save_dir='./results_1/metal/HDR_input_1/Attention_Distillation/add_plat_qiu/UNet_14/bs6_lr0.001_600'
save_dir='./results_1/qiu_plat_jieti/jieti/HDR_fenzi_fenmu/Attention_Distillation/add_plat_qiu/UNet_14/bs3_lr0.001_1000'
save_dir='./results_YM/UNet/YM_Data/bs8_lr0.001_600_all'
# save_dir='./results/fenzi_fenmu/multi_model/plat_qiu/FCN/bs8_lr0.001_1000/qiu_plat/plat'
# save_dir='./results/fenzi_fenmu/multi_model/plat_qiu/FCN/bs8_lr0.001_1000/qiu_plat/qiu1'
# save_dir='./results/fenzi_fenmu/multi_model/plat_qiu/FCN/bs8_lr0.001_1000/qiu_plat/qiu2'
# dir_img='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/images_GT'
dir_img='/home/wangh20/data/structure/YM_dataset/images_GT'
# dir_img='/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/plat/images_GT'
# dir_img='/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_1/images_GT'
# dir_img='/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_2/images_GT'
# dir_mask='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/'
dir_mask='/home/wangh20/data/structure/YM_dataset'
# dir_mask='/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/plat'
# dir_mask='/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_1'
# dir_mask='/home/wangh20/data/structure/metal_dataset/qiu_plat_fenzi_fenmu/qiu_2'
gpu_id=1
scale=1

        # python ../test_input_1.py \
        python ../test.py \
        --load=$checkpoint \
        --gpu-id=$gpu_id \
        --dir-img=$dir_img \
        --dir-mask=$dir_mask \
        --scale=$scale \
        --save-dir=$save_dir