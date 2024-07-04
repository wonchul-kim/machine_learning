import glob 
import os
import os.path as osp
from ultralytics import YOLO

import imgviz
import json
from mlearning.utils.vis.vis_seg import vis_seg

weights_file = "/HDD/etc/best.pt"
model = YOLO(weights_file) 

input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset_yolo_is/images/val'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset/val'
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/results/yolo_results'

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

preds = {}
for img_file in img_files:
    
    result = model(img_file, save=False, imgsz=768, conf=0.5)[0]
    classes = result.boxes.cls.tolist()
    idx2class = result.names
    confs = result.boxes.conf.tolist()
    masks = result.masks
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1]
    idx2masks = {}
    for cls, mask in zip(classes, masks):
        if cls not in idx2masks:
            idx2masks[cls] = []
        idx2masks[cls].append([xy.tolist() for xy in mask.xy])
    
    preds.update({img_file: {'idx2masks': idx2masks, 'idx2class': idx2class}})
    
    vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir)
    
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(preds, json_file, ensure_ascii=False, indent=4)
