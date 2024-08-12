import os.path as osp
import glob 
import json

def preds2metrics(preds_json, class2idx):
    with open(preds_json, 'r') as jf:
        anns = json.load(jf)
    
    detections = []
    for filename, ann in anns.items():
        for _class, val in ann['idx2xyxys'].items():
            for box, conf in zip(val['bbox'], val['confidence']):
                detections.append([filename, int(class2idx[ann['idx2class'][_class]]), float(conf), (box[0][0], box[0][1], box[1][0], box[1][1])])
                
    return detections
        
    
if __name__ == '__main__':
    preds_json = '/DeepLearning/_projects/sungjin_body/tests/yolov8_patch_v2/winter/w_json/학습/preds.json'
        
    detections = preds2metrics(preds_json)
    print(detections)