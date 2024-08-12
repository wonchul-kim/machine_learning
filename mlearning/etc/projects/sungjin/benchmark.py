from mlearning.benchmarks.labelme2metcis import labelme2metrics
from mlearning.benchmarks.preds2metrics import preds2metrics
from mlearning.benchmarks.metrics import get_average_precision

input_dir = '/DeepLearning/_projects/sungjin_body/benchmark/data'
ground_truths, class2idx = labelme2metrics(input_dir)
    
preds_json = '/DeepLearning/_projects/sungjin_body/benchmark/output/preds.json'
detections = preds2metrics(preds_json, class2idx)

iou_threshold = 0.1
classes = class2idx.values()

ap = get_average_precision(detections, ground_truths, classes, iou_threshold)
map = ap['map']
ap_by_class = ap['by_class']
ap_by_image = ap['by_image']

print(class2idx)
print(ap)