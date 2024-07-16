import glob 
import os
import os.path as osp
from ultralytics import YOLO

import imgviz
import json
import pandas as pd
from tqdm import tqdm

from mlearning.utils.vis.vis_obb import vis_obb

weights_file = "/HDD/etc/yolov8/obb/best.pt"
model = YOLO(weights_file) 

input_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/split_dataset/val'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/split_dataset/val'
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/results/yolo_obb_results'

iou_threshold = 0.7
_classes = ['BOX']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

preds = {}
compare = {}
compare_gt = True
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    pred = model(img_file, save=False, imgsz=2048, conf=0.5)[0]
    
    idx2class = pred.names
    obb_result = pred.obb
    classes = obb_result.cls.tolist()
    confs = obb_result.conf.tolist()
    
    idx2xyxys = {}
    for cls, xyxys in zip(classes, obb_result.xyxyxyxy):
        if cls not in idx2xyxys:
            idx2xyxys[cls] = []
        idx2xyxys[cls].append([xy.tolist() for xy in xyxys])
    
    
    if _classes is not None:
        new_idx2xyxys = {}
        for idx, _cls in enumerate(_classes):
            for jdx, cls in enumerate(idx2class.values()):
                if cls == _cls:
                    new_idx2xyxys[idx] = idx2xyxys[jdx]
        
        idx2xyxys = new_idx2xyxys
        idx2class = _idx2class   
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
    preds.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_gt:
        _compare = vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold)
            
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(preds, json_file, ensure_ascii=False, indent=4)

if compare_gt:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))