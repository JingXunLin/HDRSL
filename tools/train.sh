load_checkpoint=''
gpu_id=0
lr=0.001
bs=3
epochs=400
validation=10
# dir_img='/home/wangh20/data/structure/YM_use_dataset/right/images'
# dir_img='/home/wangh20/data/structure/val_effect_freq/right/4/train/images'
# dir_img='/home/wangh20/data/structure/SIDO/data_use/grate'
# dir_img='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/images_GT'
dir_img='/home/wangh20/data/structure/YM_dataset/images_GT'
# dir_img='/home/wangh20/data/structure/metal_dataset/metal_1000_cera_800/cera_800/images_GT'
# dir_img='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/images'
# dir_img='/home/wangh20/data/structure/YM_dataset/images_GT'
# dir_mask='/home/wangh20/data/structure/val_effect_freq/right/4/train/GT'
dir_mask='/home/wangh20/data/structure/YM_dataset/'
# dir_mask='/home/wangh20/data/structure/metal_dataset/metal_1000_cera_800/cera_800'
# dir_mask='/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/'
# save_checkpoint_path='./checkpoints/fenzi_fenmu_input_1/add_plat_qiu/UNet_4/bs3_lr0.001_600'
# save_checkpoint_path='./checkpoints/fenzi_fenmu/add_plat_qiu/UNet_14/bs6_lr0.001_600'
# save_checkpoint_path='./checkpoints_1/HDR_fenzi_fenmu/Distillation_attention/add_plat_qiu/UNet_14/bs3_lr0.0006_600'
# save_checkpoint_path='./checkpoints_1/HDR_input_1/Attention_Distillation/add_plat_qiu/UNet_14/bs6_lr0.001_600'
# save_checkpoint_path='./checkpoints_1/HDR_generation/Distillation/add_plat_qiu/UNet_14/bs6_lr0.001_600'
save_checkpoint_path='./checkpoints_YM/UNet/YM_Data_34/bs3_lr0.001_400'
# save_checkpoint_path='./checkpoints_1/HDR_fenzi_fenmu/Distillation/add_plat_qiu/UNet_14/bs3_lr0.001_600'


        python ../train_811_add_data.py \
        --gpu_id=$gpu_id \
        --learning_rate=$lr \
        --batch_size=$bs \
        --epochs=$epochs \
        --validation=$validation \
        --dir_img=$dir_img \
        --dir_mask=$dir_mask \
        --save_checkpoint_path=$save_checkpoint_path \
        --load_checkpoint=$load_checkpoint 