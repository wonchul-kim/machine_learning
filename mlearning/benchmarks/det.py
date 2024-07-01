import glob 
import json
import os
import os.path as osp 
import numpy as np
from tqdm import tqdm
import csv

            
def get_distance_two_points(points1, points2, threshold=None):
    dist = np.sqrt(pow(points1[0] - points2[0], 2) + pow(points1[1] - points2[1], 2))
    
    if threshold:
        if dist <= threshold:
            return dist
        else:
            return -1
    else:
        return dist
    
            
THRRESHOLD_CENTER_DIST = 10 # pixel

pred_dir = '/Data/01.Image/리치코리아/LA/24.06.21/test/exp11/labels'
gt_dir = '/Data/01.Image/리치코리아/LA/24.06.21/test/data'
output_dir = '/Data/01.Image/리치코리아/LA/24.06.21/test/results'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    

pred_json_files = glob.glob(osp.join(pred_dir, '*.json'))
gt_json_files = glob.glob(osp.join(pred_dir, '*.json'))

assert len(pred_json_files) == len(gt_json_files), ValueError(f"json files should be same b/t pred({len(pred_json_files)}) and gt({len(gt_json_files)})")


total_results = []
for pred_json_file in tqdm(pred_json_files):
    
    filename = osp.split(osp.splitext(pred_json_file)[0])[-1]
    gt_json_file = osp.join(gt_dir, filename + '.json')
    
    assert osp.exists(gt_json_file), ValueError(f"There is no such gt-json-file: {gt_json_file}")
    
    with open(pred_json_file, 'r') as jf:
        pred_anns = json.load(jf)['shapes']
    with open(gt_json_file, 'r') as jf:
        gt_anns = json.load(jf)['shapes']
        
    print(f"There are {len(pred_anns)} at PRED")
    print(f"There are {len(gt_anns)} at GT")
        
    ignore_gt_idx = []
    results = {"image": filename, 'number of preds': len(pred_anns), 'number of gt': len(gt_anns)}
    max_center_diff = -1
    max_height_diff = -1
    max_width_diff = -1
    for pred_ann in pred_anns:
        
        pred_points = pred_ann['points']
        pred_label = pred_ann['points']
        pred_xs, pred_ys = [], []
        for pred_point in pred_points:
            pred_xs.append(pred_point[0])
            pred_ys.append(pred_point[1])
            
        pred_center = [np.mean(pred_xs), np.mean(pred_ys)]
        pred_width = np.max(pred_xs) - np.min(pred_xs)
        pred_height = np.max(pred_ys) - np.min(pred_ys)
        
        for gt_idx, gt_ann in enumerate(gt_anns):
            if gt_idx in ignore_gt_idx:
                continue
            gt_points = gt_ann['points']
            gt_label = gt_ann['points']
            gt_xs, gt_ys = [], []
            for gt_point in gt_points:
                gt_xs.append(gt_point[0])
                gt_ys.append(gt_point[1])
                
            gt_center = [np.mean(gt_xs), np.mean(gt_ys)]
            gt_width = np.max(gt_xs) - np.min(gt_xs)
            gt_height = np.max(gt_ys) - np.min(gt_ys)
            
            if get_distance_two_points(pred_center, gt_center, THRRESHOLD_CENTER_DIST) >= 0:
                ignore_gt_idx.append(gt_idx)
                results.update({f"box_{gt_idx}": {"center diff": get_distance_two_points(pred_center, gt_center),
                                              "width diff": abs(gt_width - pred_width),
                                              'height diff': abs(gt_height - pred_height)}
                })
                if get_distance_two_points(pred_center, gt_center) > max_center_diff:
                    max_center_diff = get_distance_two_points(pred_center, gt_center)
                if abs(gt_width - pred_width) > max_width_diff:
                    max_width_diff = abs(gt_width - pred_width)
                if abs(gt_height - pred_height) > max_height_diff:
                    max_height_diff = abs(gt_height - pred_height)
            else:
                continue
        
    results.update({"max center diff": max_center_diff,
                    "max height diff": max_height_diff,
                    "max width diff": max_width_diff})

    total_results.append(results)
        
with open(osp.join(output_dir, 'results.json'), "w") as jf:
    json.dump(total_results, jf)
print(total_results)


fieldnames = ["image", "number of preds", "number of gt", 'max center diff', 'max height diff', 'max width diff']

with open(osp.join(output_dir, 'results.csv'), 'w', newline='', encoding='utf-8') as csvfile:
    # CSV writer 객체 생성
    fieldnames = total_results[0].keys()  # 첫 번째 사전의 key들을 필드로 사용
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 첫 행에 헤더를 쓰기
    writer.writeheader()

    # 각 사전을 CSV 파일로 쓰기
    for _row in total_results:
        row = {
            "image": _row["image"],
            "number of preds": _row["number of preds"],
            "number of gt": _row["number of gt"],
            "max center diff": _row["max center diff"],
            "max height diff": _row["max height diff"],
            "max width diff": _row["max width diff"],
        }
        writer.writerow(row)

