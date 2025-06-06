import shutil
import os

with open("/data1/Datasets/shared_datasets/layer_0/val.txt", "r") as file:
    for line in file:
        src_path = "src/trk.101.020.rightImage.000270.png" #line.split('\t')
        gt_path = "gt/trk.101.020.rightImage.000270.png"
        
        src_path = os.path.join("/data1/Datasets/shared_datasets/layer_0/", src_path)
        dst_path = os.path.join("/data/kurnikov/D-FINE/data/KROMKA/val/src", os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        gt_path = os.path.join("/data1/Datasets/shared_datasets/layer_0/", gt_path)
        dst_path = os.path.join("/data/kurnikov/D-FINE/data/KROMKA/val/gt/", os.path.basename(gt_path))
        shutil.copy2(gt_path, dst_path)
        print(src_path)