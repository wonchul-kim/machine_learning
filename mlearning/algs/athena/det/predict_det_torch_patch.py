import glob 
import os
import os.path as osp
import torch
import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mlearning.utils.vis.vis_hbb_patch import vis_hbb_patch
from mlearning.utils.functionals import letterbox
from athena.src.tasks.detection.frameworks.pytorch.models.yolov7.models.experimental import attempt_load as attempt_load7
from athena.src.tasks.detection.frameworks.pytorch.models.yolov5.utils.general import check_img_size
from athena.src.tasks.detection.frameworks.pytorch.models.yolov5.utils.general import (non_max_suppression, scale_coords)

compare_mask = True
imgsz = 1024
device = 'cuda'
weights = '/DeepLearning/_projects/sungjin_body/patch/2024_07_28_10_56_16/train/weights/best.pt'

input_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/w_json/기타'
json_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/w_json/기타'
output_dir = f'/DeepLearning/_projects/sungjin_body/tests/patch/winter/w_json/기타'

# input_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/wo_json/기타'
# json_dir = None
# output_dir = f'/DeepLearning/_projects/sungjin_body/tests/patch/winter/wo_json/기타'

patch_overlap_ratio = 0.1
patch_width = imgsz
patch_height = imgsz
dx = int((1. - patch_overlap_ratio) * patch_width)
dy = int((1. - patch_overlap_ratio) * patch_height)

compare_gt = True if json_dir is not None else False
model = attempt_load7(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check image size
if isinstance(imgsz, int):
    imgsz = (imgsz, imgsz)
model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
idx2class = {ii: vv for ii, vv in enumerate(model.classes)}

iou_threshold = 0.5
nms_conf_threshold = 0.25
nms_iou_threshold = 0.1
line_width = 5
font_scale = 2
classes = model.classes

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

results = {}
compare = {}
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file).astype(np.float32)
    img_h, img_w, img_c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    
    num_patches = 0
    idx2xyxys = {}
    for y0 in range(0, img_h, dy):
        for x0 in range(0, img_w, dx):
            num_patches += 1
            if y0 + patch_height > img_h:
                # skip if too much overlap (> 0.6)
                y = img_h - patch_height
            else:
                y = y0

            if x0 + patch_width > img_w:
                x = img_w - patch_width
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x + patch_width, y, y + patch_height
            input_img, ratio, pad_values = letterbox(img[ymin:ymax, xmin:xmax, :], (imgsz[0], imgsz[1]))
            
            torch_img = torch.from_numpy(np.transpose(input_img[np.newaxis, :], (0, 3, 1, 2))).to('cuda')
            with torch.no_grad():
                preds = model(torch_img)
            pred = preds[0].detach().cpu()
            dets = non_max_suppression(pred, nms_conf_threshold, nms_iou_threshold, None, False)
            
            for det in dets:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[:2], det[:, :4], (img_h, img_w)).round()

                    for *xyxy, conf, cls in reversed(det):
                        cls = cls.detach().cpu().item()
                        conf = float(conf.detach().cpu().item())
                        if cls not in idx2xyxys.keys():
                            idx2xyxys[cls] = []
                            idx2xyxys[cls] = {'bbox': [], 'confidence': []}
                            
                        idx2xyxys[cls]['bbox'].append([[int(np.round(xyxy[0].detach().cpu().item()) + xmin), int(np.round(xyxy[1].detach().cpu().item()) + ymin)], 
                                            [int(np.round(xyxy[2].detach().cpu().item()) + xmin), int(np.round(xyxy[3].detach().cpu().item()) + ymin)]])
                        idx2xyxys[cls]['confidence'].append(conf)
        
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
    results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_gt:
        _compare = vis_hbb_patch(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, 
                        compare_gt=compare_gt, iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_hbb_patch(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, 
                compare_gt=compare_gt, iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
    
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

if compare_gt:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))