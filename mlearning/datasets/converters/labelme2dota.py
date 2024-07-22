import glob 
import os
import os.path as osp 
import json 
import numpy as np
from tqdm import tqdm
from polygon2dota import polygon2dota_by_rotate

input_dir = '/HDD/datasets/projects/rich/split_dataset'
output_dir = '/HDD/datasets/projects/rich/split_dataset_dota'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

for folder in folders:
    _output_dir = osp.join(output_dir, folder, 'labelTxt')
    if not osp.exists(_output_dir):
        os.makedirs(_output_dir)

    json_files = glob.glob(osp.join(input_dir, folder, '*.json'))
    print(f"There are {len(json_files)} json files")

    for json_file in tqdm(json_files, desc=folder):
        filename = osp.split(osp.splitext(json_file)[0])[-1]
        txt = open(osp.join(_output_dir, filename + '.txt'), 'w')
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        if len(anns) != 0:
            for ann in anns:
                points, _ = polygon2dota_by_rotate(ann['points'])
                data = ''
                for point in points:
                    data += f'{str(int(point[0]))} {str(int(point[1]))} '
                
                data += f'{ann["label"]} 0'
                txt.write(data)
                txt.write("\n")
        
    txt.close()

        
        