import glob 
import os
import os.path as osp
from ultralytics import YOLO
import cv2
import imgviz
import json
import pandas as pd
from tqdm import tqdm

from mlearning.utils.vis.vis_hbb import vis_hbb

weights_file = "/DeepLearning/_projects/sungjin_body/yolov8/patch/weights/best.pt"
model = YOLO(weights_file) 

# input_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/w_json/학습'
# json_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/w_json/학습'
# output_dir = '/DeepLearning/_projects/sungjin_body/tests/yolov8_patch_v2/winter/w_json/학습'

# input_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/wo_json/학습'
# json_dir = None
# output_dir = '/DeepLearning/_projects/sungjin_body/tests/yolov8_patch_v2/winter/wo_json/학습'

input_dir = '/DeepLearning/_projects/sungjin_body/benchmark/24.08.12'
json_dir = '/DeepLearning/_projects/sungjin_body/benchmark/24.08.12'
output_dir = '/DeepLearning/_projects/sungjin_body/benchmark/preds'

compare_gt = True if json_dir is not None else False
imgsz = 1024
line_width = 2
font_scale = 0.5
conf_threshold = 0.1
iou_threshold = 0.5
_classes = ['STABBED', 'QR', 'CRACK', 'RUST', 'SCRATCH', 'PRESSED', 'BOTTOM']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

### patch params.
patch_overlap_ratio = 0.1
patch_width = imgsz
patch_height = imgsz
dx = int((1. - patch_overlap_ratio) * patch_width)
dy = int((1. - patch_overlap_ratio) * patch_height)

preds = {}
compare = {}
for img_file in tqdm(img_files):
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    
    image = cv2.imread(img_file)
    img_h, img_w, img_c = image.shape
    
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
            pred = model(image[ymin:ymax, xmin:xmax, :], save=False, imgsz=imgsz, conf=conf_threshold)[0]
    
            idx2class = pred.names
            boxes = pred.boxes
            classes = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            
            for cls, xyxys, conf in zip(classes, boxes.xyxy.tolist(), confs):
                if cls not in idx2xyxys:
                    idx2xyxys[cls] = {'bbox': [], 'confidence': []}
                idx2xyxys[cls]['bbox'].append([[xyxys[0] + xmin, xyxys[1] + ymin], [xyxys[2] + xmin, xyxys[3] + ymin]])
                idx2xyxys[cls]['confidence'].append(conf)
        
    if _classes is not None:
        new_idx2xyxys = {}
        for idx, _cls in enumerate(_classes):
            for jdx, cls in enumerate(idx2class.values()):
                if cls == _cls and jdx in idx2xyxys:
                    new_idx2xyxys[idx] = idx2xyxys[jdx]
        
        idx2xyxys = new_idx2xyxys
        # idx2class = _idx2class   
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
    preds.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_gt:
        _compare = vis_hbb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold,
                           line_width=line_width, font_scale=font_scale)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_hbb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, 
                iou_threshold=iou_threshold, line_width=line_width, font_scale=font_scale)
            
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(preds, json_file, ensure_ascii=False, indent=4)

if compare_gt:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))