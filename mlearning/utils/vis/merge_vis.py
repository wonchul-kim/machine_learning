import os
import os.path as osp 
import glob 
import cv2
import numpy as np
from tqdm import tqdm


input_dir = "/DeepLearning/_projects/sungjin_body/tests"
output_dir = '/DeepLearning/_projects/sungjin_body/summarize_v2'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

datasets = ['winter/w_json/학습', 'winter/w_json/기타', 'winter/wo_json/학습', 'winter/wo_json/기타']
resize_factor = 3

folders = sorted([folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)])
num_folders = len(folders)
for dataset in datasets:
    img_files = glob.glob(osp.join(input_dir, folders[0], dataset, '*.png'))
    
    for img_file in tqdm(img_files, desc=dataset):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        
        img = cv2.imread(img_file)
        img_h, img_w, img_c = img.shape 
        
        text_img = np.zeros((img_h*num_folders, 1500, 3), np.uint8)
        
        total_img = None
        for idx, folder in enumerate(folders):
            cv2.putText(text_img, folder, (50, int(img_h*idx + img_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 10)
            
            if total_img is None:
                total_img = img
            else:
                _img = cv2.imread(osp.join(input_dir, folder, dataset, filename + '.png'))
                total_img = cv2.vconcat([total_img, _img])
                
        
        total_img = cv2.hconcat([text_img, total_img])
        
        _output_dir = osp.join(output_dir, dataset)
        if not osp.exists(_output_dir):
            os.makedirs(_output_dir)
            
        total_img = cv2.resize(total_img, (int(total_img.shape[1]/resize_factor), int(total_img.shape[0]/resize_factor)))
        cv2.imwrite(osp.join(_output_dir, filename + '.jpg'), total_img)
        
            
        
    

    

        