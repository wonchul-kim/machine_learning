import glob 
import os.path as osp 
import cv2 
import numpy as np
import os
from tqdm import tqdm 

def min_max_normalize(image_array):
    normalized_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    return normalized_array

def get_min_max(input_dir, output_dir, xyxy=None):

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    
    min_val = float('inf')
    max_val = float('-inf')
    num_images = 0
    img_files = glob.glob(osp.join(input_dir, '*.bmp'))
    for img_file in tqdm(img_files):
        
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        if xyxy is not None:
            img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
        
        min_val = min(min_val, np.min(img))
        max_val = max(max_val, np.max(img))
        num_images += 1

    print("* min_val: ", min_val)
    print("* max_val: ", max_val)      
    
    txt = open(osp.join(output_dir, 'min_max.txt'), 'w')
    txt.write(f"min_val: {min_val}\n")
    txt.write(f"max_val: {max_val}\n")
    txt.close()
         
input_dir = '/HDD/datasets/projects/sungjin/body/test'
output_dir = '/HDD/datasets/projects/sungjin/body/test/patches'

patch_overlap_ratio = 0.2
patch_width = 1024
patch_height = 1024

xyxy = [350, 150, 350 + 1664, 150 + 1664]
    
get_min_max(input_dir, output_dir, xyxy)
                        
                    
                    
                



                


