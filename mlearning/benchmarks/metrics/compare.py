import glob 
import os
import os.path as osp
import imgviz
import json
import cv2
import numpy as np

input_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/split_dataset_yolo_is/images/val'
output_dir = '/DeepLearning/etc/_athena_tests/benchmark/sungwoo/u_gap/benchmark'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
img_files = glob.glob(osp.join(input_dir, '*.bmp'))

for img_file in img_files:
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    json_file = osp.splitext(img_file)[0] + '.json'
    img = cv2.imread(img_file)
    img_height, img_width, img_channel = img.shape
    
    origin = 25, 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = np.zeros((50, img_width, 3), np.uint8)
    text2 = np.zeros((50, img_width, 3), np.uint8)
    text3 = np.zeros((50, 300, 3), np.uint8)
    cv2.putText(text1, "(a) original", origin, font, 0.6, (255, 255, 255), 1)
    cv2.putText(text2, "(b) predicted", origin, font, 0.6, (255, 255, 255), 1)
    cv2.putText(text3, "(c) legend", origin, font, 0.6, (255, 255, 255), 1)
    
    gt_mask = np.zeros(img_height, img_width, img_channel)
    
    
    
    
    
    

