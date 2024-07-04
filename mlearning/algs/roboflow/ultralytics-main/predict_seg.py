import glob 
import os
import os.path as osp
from ultralytics import YOLO

import imgviz
import json
import pandas as pd
from mlearning.utils.vis.vis_seg import vis_seg

compare_mask = True
weights_file = "/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/outputs/yolo/best.pt"
model = YOLO(weights_file) 

input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset_yolo_is/images/val'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset/val'
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/reports/preds/yolo_results'

_classes = ['ROI', 'FILM', 'MARK']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

preds = {}
compare = {}
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
    
    
    if _classes is not None:
        new_idx2masks = {}
        for idx, _cls in enumerate(_classes):
            for jdx, cls in enumerate(idx2class.values()):
                if cls == _cls:
                    new_idx2masks[idx] = idx2masks[jdx]
        
        idx2masks = new_idx2masks
        idx2class = _idx2class   
    
    
    preds.update({img_file: {'idx2masks': idx2masks, 'idx2class': idx2class}})
    
    if compare_mask:
        _compare = vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir, compare_mask=compare_mask)
        compare.update({img_file: _compare})
    else:
        vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir, compare_mask=compare_mask)
            
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(preds, json_file, ensure_ascii=False, indent=4)


if compare_mask:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
        
    df_labels_by_image = pd.DataFrame(compare).T
    df_labels_by_image.name = 'filename'
    df_labels_by_image.fillna(0, inplace=True)
    df_labels_by_image.to_csv(osp.join(output_dir, 'diff.csv'))