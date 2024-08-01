import glob 
import os.path as osp 
import cv2 
import numpy as np
import os
from tqdm import tqdm 

def get_mean_std(input_dir, output_dir):

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    
    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0
    img_files = glob.glob(osp.join(input_dir, '*.bmp'))
    for img_file in tqdm(img_files):
        
        img = cv2.imread(img_file)
        
        img = img/255.0
        
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
        num_images += 1

    mean /= num_images
    std /= num_images 
    
    print("* Mean: ", mean)
    print("* std: ", std)      
    
    txt = open(osp.join(output_dir, 'mean_std.txt'), 'w')
    txt.write(f"mean: {mean}\n")
    txt.write(f"std: {std}\n")
    txt.close()
         
input_dir = '/HDD/datasets/projects/sungjin/body/test'
output_dir = '/HDD/datasets/projects/sungjin/body/test/patches'

patch_overlap_ratio = 0.2
patch_width = 1024
patch_height = 1024
    
get_mean_std(input_dir, output_dir)
                        
                    
                    
                



                


