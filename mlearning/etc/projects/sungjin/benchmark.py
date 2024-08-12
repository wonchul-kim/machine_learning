from mlearning.benchmarks.labelme2metcis import labelme2metrics
from mlearning.benchmarks.preds2metrics import preds2metrics
from mlearning.benchmarks.metrics import get_average_precision, get_average_g_results


input_dir = '/DeepLearning/_projects/sungjin_body/benchmark/24.08.12'
ground_truths, class2idx = labelme2metrics(input_dir)
print(class2idx)

preds_json = '/DeepLearning/_projects/sungjin_body/benchmark/preds/preds.json'
detections, class2idx = preds2metrics(preds_json, class2idx)
print(class2idx)

iou_threshold = 0.1
classes = class2idx.values()

ap = get_average_precision(detections, ground_truths, classes, iou_threshold)
map = ap['map']
ap_by_class = ap['by_class']
ap_by_image = ap['by_image']
g_results = ap['g_results']
print(ap)

avg_g_results = get_average_g_results(g_results)
print(avg_g_results)


