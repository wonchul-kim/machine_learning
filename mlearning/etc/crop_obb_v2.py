import os
import glob
import os.path as osp
import json
import cv2
import numpy as np
from tqdm import tqdm

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated, M

input_dir = '/Data/01.Image/kt&g/24.08.05/0725_cls'
output_dir = '/Data/01.Image/kt&g/24.08.05/0725_cls_crop'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

json_files = glob.glob(osp.join(input_dir, '*.json'))

x_offset = 100
y_offset = 100
num_classes = {}
for json_file in tqdm(json_files):
    filename = osp.split(osp.splitext(json_file)[0])[-1]
    img_file = osp.join(osp.splitext(json_file)[0] + '.bmp') 
    
    img = cv2.imread(img_file)
    img_h, img_w, img_c = img.shape
    
    with open(json_file, 'r') as jf:
        anns = json.load(jf)['shapes']
        
    for idx, ann in enumerate(anns):
        label = ann['label']
        
        if label not in num_classes:
            num_classes[label] = 0
        else:
            num_classes[label] += 1
        
        _output_path = osp.join(output_dir, label)
        
        if not osp.exists(_output_path):
            os.mkdir(_output_path)

        points = np.array(ann['points'], dtype='float32')

        # 네 점을 기준으로 변환 행렬 계산
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 이미지 회전
        angle = rect[2]
        rotated_img, M = rotate_image(img, angle)

        # 회전된 이미지에서 변환된 좌표 계산
        rotated_box = cv2.transform(np.array([box]), M)[0]

        x_min, y_min = np.min(rotated_box, axis=0)
        x_max, y_max = np.max(rotated_box, axis=0)

        x_min = max(0, int(x_min) - x_offset)
        y_min = max(0, int(y_min) - y_offset)
        x_max = min(rotated_img.shape[1], int(x_max) + x_offset)
        y_max = min(rotated_img.shape[0], int(y_max) + y_offset)

        cropped_rotated = rotated_img[y_min:y_max, x_min:x_max]

        if cropped_rotated.shape[0] > 1300:
            cropped_rotated = cv2.transpose(cropped_rotated)
            cropped_rotated = cv2.flip(cropped_rotated, flipCode=1)


        cv2.imwrite(osp.join(_output_path, f'{filename}_{label}_{idx}.bmp'), cropped_rotated)
        
        
txt = open(osp.join(output_dir, 'classes.txt'), 'w')
for key, val in num_classes.items():
    txt.write(f"{key}: {val}\n")
    
txt.close()