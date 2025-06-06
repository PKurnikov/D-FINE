# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml --use-amp=False --seed=0 -t output/dfine_s_obj365.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml --use-amp=False --seed=0 -r output/dfine_hgnetv2_s_obj2coco/checkpoint0010.pth


# Mean IoU for semantic segmentation: 0.9168
# IOU of class 0 = 0.9662353532318397
# IOU of class 1 = 0.9134037261473047
# IOU of class 2 = 0.934697716700647
# IOU of class 3 = 0
# IOU of class 4 = 0.39914502849905004
# IOU of class 5 = 0.9863127956378323