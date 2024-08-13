from mlearning.benchmarks.labelme2metcis import labelme2metrics
from mlearning.benchmarks.preds2metrics import preds2metrics
from mlearning.benchmarks.metrics import get_performance
from mlearning.benchmarks.save import save_pf_by_image_to_excel
import os.path as osp

output_dir = '/DeepLearning/_projects/sungjin_body/benchmark/'

input_dir = '/DeepLearning/_projects/sungjin_body/benchmark/24.08.12'
ground_truths, class2idx = labelme2metrics(input_dir)
print(class2idx)

preds_json = '/DeepLearning/_projects/sungjin_body/benchmark/preds/preds.json'
detections, class2idx = preds2metrics(preds_json, class2idx)
print(class2idx)

# _detections, _ground_truths = [], []
# for detection in detections:
#     if detection[0] == '124080711064364_54_BODY':
#         _detections.append(detection)
# for ground_truth in ground_truths:
#     if ground_truth[0] == '124080711064364_54_BODY':
#         _ground_truths.append(ground_truth)
# detections = _detections
# ground_truths = _ground_truths


iou_threshold = 0.1
classes = class2idx.values()

pf = get_performance(detections, ground_truths, classes, iou_threshold)
pf_by_image = pf['by_image']
pf_map = pf['map']
pf_by_class = pf['by_class']

print('* by image: ', pf['by_image'])
print('* map: ', pf['map'])
print('* by class: ', pf['by_class'])

save_pf_by_image_to_excel(pf_by_image, osp.join(output_dir, 'pf_by_image.xlsx'))