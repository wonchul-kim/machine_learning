import os.path as osp
import glob 
import json

def labelme2metrics(input_dir, return_class2idx=True):
    json_files = glob.glob(osp.join(input_dir, '*.json'))

    gts = []
    class2idx = {}
    for json_file in json_files:
        
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        if len(anns) != 0:
            filename = osp.split(osp.splitext(json_file)[0])[-1]
            for ann in anns:
                shape_type = ann['shape_type']
                points = ann['points']
                label = ann['label']
                
                if label not in class2idx:
                    class2idx[label] = len(class2idx)
                
                if shape_type == 'rectangle':
                    assert len(points) == 2 and len(points[0]) == 2, ValueError(f"ERROR: points are wrong: {points} at {json_file}")
                    
                    gt = [filename, class2idx[label], 1, (points[0][0], points[0][1], points[1][0], points[1][1])]
            
            gts.append(gt)                    

    if return_class2idx:
        return gts, class2idx
    else:
        return gts
        
if __name__ == '__main__':
    input_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/w_json/학습'


    ground_truths = labelme2metrics(input_dir)
    print(ground_truths)
