import os.path as osp 
import glob 
import cv2 
import numpy as np

input_dir = '/HDD/datasets/public/duts/raw_dataset/DUTS-TR/DUTS-TR-Mask'


img_files = glob.glob(osp.join(input_dir, '*.png'))

img = cv2.imread(img_files[0])
print(np.unique(img))