from mlearning.benchmarks.labelme2metcis import labelme2metrics
from mlearning.benchmarks.preds2metrics import preds2metrics
from mlearning.benchmarks.metrics import get_performance
from mlearning.benchmarks.save import save_pf_by_image_to_excel, save_df_by_class_to_pdf
import os.path as osp

output_dir = '/HDD/etc/obb'

input_dir = '/HDD/etc/obb/images'
ground_truths, class2idx = labelme2metrics(input_dir)
print(class2idx)

preds_json = '/HDD/etc/obb/preds.json'
detections, class2idx = preds2metrics(preds_json, class2idx)
print(class2idx)

_detections, _ground_truths = [], []
for detection in detections:
    if detection[0] == '124071716123313_1_image_1' or detection[0] == '124071716123338_1_image_1': #'124080701581332_54_BODY':
        _detections.append(detection)
for ground_truth in ground_truths:
    if ground_truth[0] == '124071716123313_1_image_1' or ground_truth[0] == '124071716123338_1_image_1': #'124080701581332_54_BODY':
        _ground_truths.append(ground_truth)
detections = _detections
ground_truths = _ground_truths

iou_threshold = 0.1
classes = class2idx.values()
idx2class = {idx: _class for _class, idx in class2idx.items()}

pf = get_performance(detections, ground_truths, classes, iou_threshold, shape_type='polygon')
pf_by_image = pf['by_image']
pf_by_class = pf['by_class']

print('* by image: ', pf['by_image'])
print('* by class: ', pf['by_class'])

save_pf_by_image_to_excel(pf_by_image, osp.join(output_dir, 'pf_by_image.xlsx'), idx2class)
save_df_by_class_to_pdf(pf_by_class, osp.join(output_dir, 'pf_by_class.pdf'), idx2class)