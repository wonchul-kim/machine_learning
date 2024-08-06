import glob 
import os
import os.path as osp
from ultralytics import YOLO

import imgviz
import json
import pandas as pd
from tqdm import tqdm

from mlearning.utils.vis.vis_obb import vis_obb

weights_file = "/DeepLearning/_projects/kt_g/wonchul/240805/obb/train4/weights/best.pt"
model = YOLO(weights_file) 

input_dir = '/DeepLearning/_projects/kt_g/24.07.25/test'
json_dir = None
output_dir = '/DeepLearning/_projects/kt_g/wonchul/240805/test/obb_seen'
compare_gt = False

# input_dir = '/Data/01.Image/kt&g/24.07.26/unseen_img'
# json_dir = None
# output_dir = '/DeepLearning/_projects/kt_g/wonchul/240805/test/obb_unseen'
# compare_gt = False

iou_threshold = 0.7
conf_threshold = 0.25
line_width = 12
font_scale = 10
imgsz = 2048
_classes = ['CIGA']
# _classes = ['MIX_ETC', 'NGP', 'SOO_ETC', 'SOO', 'SOO_0.5', 'SOO_0.1', 'ESSE_0.5', 'FIIT_CHANGE', 'FIIT_UP', 'MIX_BANG', 
#             'BOHEM_3',  'MIX_BLU',  'BOHEM_ETC',  'MIX_ICEAN',  'BOHEM_1',  'RAISON_FB',  'BOHEM_6',  'ETC',  
#             'ESSE_ETC',  'ESSE_GOLD',  'ESSE_1']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

preds = {}
compare = {}
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    pred = model(img_file, save=False, imgsz=imgsz, conf=conf_threshold)[0]
    
    idx2class = pred.names
    obb_result = pred.obb
    classes = obb_result.cls.tolist()
    confs = obb_result.conf.tolist()
    
    idx2xyxys = {}
    for cls, xyxys, conf in zip(classes, obb_result.xyxyxyxy, confs):
        if cls not in idx2xyxys:
            idx2xyxys[cls] = {'polygon': [], 'confidence': []}
        idx2xyxys[cls]['polygon'].append([xy.tolist() for xy in xyxys])
        idx2xyxys[cls]['confidence'].append(conf)
    
    if _classes is not None:
        new_idx2xyxys = {}
        for idx, _cls in enumerate(_classes):
            for jdx, cls in enumerate(idx2class.values()):
                if cls == _cls and jdx in idx2xyxys:
                    new_idx2xyxys[idx] = idx2xyxys[jdx]
        
        idx2xyxys = new_idx2xyxys
        idx2class = _idx2class   
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
    preds.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_gt:
        _compare = vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
            
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(preds, json_file, ensure_ascii=False, indent=4)

if compare_gt:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))