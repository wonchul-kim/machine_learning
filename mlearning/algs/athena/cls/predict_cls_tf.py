import glob 
import os
import os.path as osp
import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from athena.src.tasks.classification.frameworks.tensorflow.models.modeling import EfficientNet


compare_mask = True
model_height = 320
model_width = 640
model_channel = 3
num_classes = 3
classes = ['rock', 'paper', 'scissors']
device = 'cuda'

weights = '/DeepLearning/etc/_athena_tests/recipes/agent/classification/tensorflow/train_unit/_outputs/CLASSIFICATION/2023_12_19_11_28_56/train/weights/last.h5'

model = EfficientNet(
            model_name='efficientnetb3',
            input_shape=(model_height, model_width, model_channel),
            num_classes=num_classes
        )

model.build(input_shape=(1, model_height, model_width, model_channel))
model.load_weights(weights)


idx2class = {ii: vv for ii, vv in enumerate(classes)}

# compare_gt = True
# iou_threshold = 0.7
# max_dets = 50
# nms_conf_threshold = 0.25
# nms_iou_threshold = 0.1
# classes = model.classes
# input_dir = '/DeepLearning/_athena_tests/datasets/directory1/split_dataset_unit/val'
# output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/interojo/results/yolov7_results'

# if not osp.exists(output_dir):
#     os.makedirs(output_dir)

# img_files = glob.glob(osp.join(input_dir, '*.bmp'))

# results = {}
# compare = {}
# for img_file in tqdm(img_files):
#     filename = osp.split(osp.splitext(img_file)[0])[-1]
#     img = cv2.imread(img_file).astype(np.float32)
#     img_h, img_w, img_c = img.shape
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    
#     with torch.no_grad():
#         preds = model(torch_img)
#     pred = preds[0].detach().cpu()
    
#     idx2xyxys = {}
#     for det in dets:
#         if len(det):
#             det[:, :4] = scale_coords(img.shape[:2], det[:, :4], (img_h, img_w)).round()

#             for *xyxy, conf, cls in reversed(det):
#                 cls = cls.detach().cpu().item()
#                 conf = float(conf.detach().cpu().item())
#                 if cls not in idx2xyxys.keys():
#                     idx2xyxys[cls] = []

#                 idx2xyxys[cls].append([[int(np.round(xyxy[0].detach().cpu().item())), int(np.round(xyxy[1].detach().cpu().item()))], 
#                                        [int(np.round(xyxy[2].detach().cpu().item())), int(np.round(xyxy[3].detach().cpu().item()))]])
  
#     color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
#     results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
#     if compare_gt:
#         _compare = vis_hbb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, 
#                            compare_gt=compare_gt, iou_threshold=iou_threshold)
#         _compare.update({"img_file": img_file})
#         compare.update({filename: _compare})
#     else:
#         vis_hbb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, 
#                 compare_gt=compare_gt, iou_threshold=iou_threshold)
            
# with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
#     json.dump(results, json_file, ensure_ascii=False, indent=4)

# if compare_gt:
#     with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
#         json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
#     df_compare = pd.DataFrame(compare)
#     df_compare_pixel = df_compare.loc['diff_iou'].T
#     df_compare_pixel.fillna(0, inplace=True)
#     df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))