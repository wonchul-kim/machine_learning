from mlearning.benchmarks.labelme2metcis import labelme2metrics
from mlearning.benchmarks.preds2metrics import preds2metrics
from mlearning.benchmarks.metrics import get_average_precision

input_dir = '/Data/01.Image/sungjin_yoke/IMAGE/BODY/24.07.29_미검이미지/w_json/학습'
ground_truths, class2idx = labelme2metrics(input_dir)
    
preds_json = '/DeepLearning/_projects/sungjin_body/tests/yolov8_patch_v2/winter/w_json/학습/preds.json'
detections = preds2metrics(preds_json, class2idx)

iou_threshold = 0.1
classes = class2idx.values()
print(classes)

ap = get_average_precision(detections, ground_truths, classes, iou_threshold)
map = ap['map']
ap_by_class = ap['by_class']
ap_by_image = ap['by_image']

print(map)