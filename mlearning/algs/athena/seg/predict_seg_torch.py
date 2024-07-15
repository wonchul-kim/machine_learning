import glob 
import os
import os.path as osp
import torch
import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from mlearning.utils.vis.vis_seg import vis_seg
from athena.src.tasks.segmentation.frameworks.pytorch.models.torch_model_v2 import TorchModelV2


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
ml_framework = 'pytorch'
model_name='segnext'
weights = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/outputs/SEGMENTATION/pytorch_segnext_b/train/weights/last.pth'
model = TorchModelV2(model_name=model_name, backbone='b', model_dict=None, 
                  width=768, height=512, channel=3, num_classes=4,
                  device='cuda:0', aux_loss=False, init_lr=1e-3, device_ids=[0], 
                  weights=weights)
classes = ['FILM', 'ROI', 'MARK']
idx2class = {idx: cls for idx, cls in enumerate(classes)}
_classes = ['ROI', 'FILM', 'MARK']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}
input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset/val'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset/val'
output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/reports/preds/athena_{model_name}'

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

results = {}
compare = {}
for img_file in img_files:
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file).astype(np.float32)
    if ml_framework == 'tensorflow':
        preds = model(img[np.newaxis, :])
        pred = preds[0].numpy()
    elif ml_framework == 'pytorch':
        torch_img = torch.from_numpy(np.transpose(img[np.newaxis, :], (0, 3, 1, 2))).to('cuda')
        preds = model(torch_img)
        pred = preds[0].detach().cpu().numpy()
        pred = np.transpose(pred, (1, 2, 0)).astype(np.uint8)

    idx2masks = {}
    for idx in idx2class.keys():
        mask, points = get_mask_from_pred(pred[:, :, idx + 1], conf=0.2)
    
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