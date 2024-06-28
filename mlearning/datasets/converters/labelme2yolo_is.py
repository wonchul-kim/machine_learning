import glob 
import os
import os.path as osp 
import json 
import numpy as np
from utils import xyxy2xywh

input_dir = '/HDD/datasets/projects/interojo/split_datasets'
output_dir = '/HDD/datasets/projects/interojo/split_datasets_yolo_is'

image_width = 1626
image_height = 1236

copy_image = True
image_ext = 'png'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]


class2idx = {}
for folder in folders:
    _output_labels_dir = osp.join(output_dir, 'labels', folder)
    if not osp.exists(_output_labels_dir):
        os.makedirs(_output_labels_dir)

    json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
    print(f"There are {len(json_files)} json files")

    for json_file in json_files:
        filename = osp.split(osp.splitext(json_file)[0])[-1]
        txt = open(osp.join(_output_labels_dir, filename + '.txt'), 'w')
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        if copy_image:
            import cv2
            from shutil import copyfile
            img_file = osp.join(input_dir, folder, filename + f'.{image_ext}')
            image = cv2.imread(img_file)
            image_width = image.shape[1]
            image_height = image.shape[0]
            _output_image_dir = osp.join(output_dir, 'images', folder)
            if not osp.exists(_output_image_dir):
                os.makedirs(_output_image_dir)
            copyfile(img_file, osp.join(_output_image_dir, filename + f'.{image_ext}'))
            
        if len(anns) != 0:
            for ann in anns:
                shape_type = ann['shape_type']
                label = ann['label']
                if label not in class2idx.keys():
                    class2idx.update({label: len(class2idx)})
                points = ann['points']
                xyxy = []

                txt.write(str(class2idx[label]) + ' ')
                for idx, point in enumerate(points):
                    if idx == len(points) -1:
                        txt.write(f'{round(point[0]/image_width, 3)} {round(point[1]/image_height, 3)}')
                    else:
                        txt.write(f'{round(point[0]/image_width, 3)} {round(point[1]/image_height, 3)} ')
                txt.write('\n')
        
    txt.close()

txt = open(osp.join(output_dir, 'classes.txt'), 'w')
for key, val in class2idx.items():
    txt.write(f'{key}: {val}\n')
txt.close()
        
        