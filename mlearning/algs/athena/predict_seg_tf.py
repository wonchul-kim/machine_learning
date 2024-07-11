import glob 
import os
import os.path as osp

import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from mlearning.utils.vis.vis_seg import vis_seg
from athena.src.tasks.segmentation.frameworks.tensorflow.models.tf_model_v2 import TFModelV2


def get_mask_from_pred(pred, conf=0.5, contour_thres=50):
    mask = (pred > conf).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for contour in contours:
        if len(contour) < contour_thres:
            pass
        else:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].tolist()
            points.append(approx)

    return mask, points

compare_mask = True
weights = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/outputs/SEGMENTATION/2024_07_03_07_57_21/train/weights/last_weights.h5'
model = TFModelV2(model_name='deeplabv3plus', backbone='efficientnetb3', backbone_weights='imagenet', 
                  batch_size=1, width=768, height=512, channel=3, num_classes=4,
                  weights=weights)
classes = ['MARK', 'ROI', 'FILM']
idx2class = {idx: cls for idx, cls in enumerate(classes)}
_classes = ['ROI', 'FILM', 'MARK']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}
input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset/val'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset/val'
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/reports/preds/athena_results'

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

results = {}
compare = {}
for img_file in img_files:
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    preds = model(img[np.newaxis, :])[0].numpy()
    
    idx2masks = {}
    for idx in idx2class.keys():
        mask, points = get_mask_from_pred(preds[:, :, idx + 1], conf=0.2)
    
        idx2masks[idx] = points
        
    if _classes is not None:
        new_idx2masks = {}
        for idx, _cls in enumerate(_classes):
            for jdx, cls in enumerate(classes):
                if cls == _cls:
                    new_idx2masks[idx] = idx2masks[jdx]
        
        idx2masks = new_idx2masks
        idx2class = _idx2class    
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1]
    results.update({filename: {'idx2masks': idx2masks, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_mask:
        _compare = vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir, compare_mask=compare_mask)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir, compare_mask=compare_mask)
    
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

if compare_mask:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
        
    df_compare_pixel = df_compare.loc['diff_pixel'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_pixel.csv'))
    
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))