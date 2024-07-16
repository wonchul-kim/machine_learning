import os.path as osp 
import glob
from tqdm import tqdm
import json
import numpy as np

input_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/results'
json_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/split_dataset/val'
idx2class = {0: 'BOX'}
json_files = glob.glob(osp.join(json_dir, '*.json'))


for json_file in json_files:
    assert osp.exists(json_file), ValueError(f"There is no such json file: {json_file}")
    
    with open(json_file, 'r') as jf:
        gt_anns = json.load(jf)['shapes']   
        
    filename = osp.split(osp.splitext(json_file)[0])[-1]
        
    gt_infos = {filename: {}}
    for idx, cls in idx2class.items():
        for ann in gt_anns:
            label = ann['label']
            if label != cls:
                continue
            points = ann['points']
            shape_type = ann['shape_type']
            
            if len(points) > 2:
                if idx not in gt_infos[filename]:
                    gt_infos[filename][idx] = []
                gt_infos[filename][idx].append(points)
                
    
folders = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

preds_infos = {}
for folder in folders:
    pred_json_file = osp.join(input_dir, folder, 'preds.json')

    with open(pred_json_file, 'r') as jf:
        preds_infos[folder] = json.load(jf)
    
def get_coordinate_diff(gt_coords, pred_coords):
    pass
    
    
diff_infos = {}
for filename, gt_info in gt_infos.items():
    
    for preds_info in preds_infos:
        preds_info[filename]