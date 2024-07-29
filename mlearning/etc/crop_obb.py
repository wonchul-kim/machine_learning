import os
import glob 
import os.path as osp 
import json
from shapely.affinity import rotate, translate
from mlearning.datasets.converters.utils import coords2polygon
import cv2 
import numpy as np
from tqdm import tqdm 

input_dir = '/Data/01.Image/kt&g/24.07.17/bmp'
output_dir = '/DeepLearning/_projects/kt_g/classification/split_dataset'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

json_files = glob.glob(osp.join(input_dir, '*.json'))

for json_file in tqdm(json_files):
    filename = osp.split(osp.splitext(json_file)[0])[-1]
    img_file = osp.join(osp.splitext(json_file)[0] + '.bmp') 
    
    img = cv2.imread(img_file)
    img_h, img_w, img_c = img.shape
    
    with open(json_file, 'r') as jf:
        anns = json.load(jf)['shapes']
        
    for idx, ann in enumerate(anns):
        label = ann['label']
        _output_dir = osp.join(output_dir, label)
        
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)

        points = np.array(ann['points'], dtype='float32')

        # 네 점을 기준으로 변환 행렬 계산
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 회전 변환 행렬 계산
        width = int(rect[1][0]) 
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 이미지의 일부를 자르고 원래의 방향으로 복원
        warped = cv2.warpPerspective(img, M, (width, height))

        cv2.imwrite(osp.join(_output_dir, f'{filename}_{label}_{idx}.png'), warped)