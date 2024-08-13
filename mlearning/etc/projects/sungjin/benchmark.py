from mlearning.benchmarks.labelme2metcis import labelme2metrics
from mlearning.benchmarks.preds2metrics import preds2metrics
from mlearning.benchmarks.metrics import get_ap_by_class, get_average_g_results


input_dir = '/DeepLearning/_projects/sungjin_body/benchmark/24.08.12'
ground_truths, class2idx = labelme2metrics(input_dir)
print(class2idx)

preds_json = '/DeepLearning/_projects/sungjin_body/benchmark/preds/preds.json'
detections, class2idx = preds2metrics(preds_json, class2idx)
print(class2idx)

_detections, _ground_truths = [], []
for detection in detections:
    if detection[0] == '124080711064364_54_BODY':
        _detections.append(detection)
for ground_truth in ground_truths:
    if ground_truth[0] == '124080711064364_54_BODY':
        _ground_truths.append(ground_truth)
        

detections = _detections
ground_truths = _ground_truths


iou_threshold = 0.1
classes = class2idx.values()

ap = get_ap_by_class(detections, ground_truths, classes, iou_threshold)
print(ap)


# import pandas as pd
# import os.path as osp

# df = pd.DataFrame(avg_g_results).T  # .T는 전치(transpose)하여 원하는 형식으로 변환

# # 엑셀 파일로 저장
# df.to_excel(osp.join('/DeepLearning/_projects/sungjin_body/benchmark/', 'output.xlsx'), index_label='filename', engine='openpyxl')
