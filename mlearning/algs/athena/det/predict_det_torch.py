import glob 
import os
import os.path as osp
import torch
import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mlearning.utils.vis.vis_obb import vis_obb
from mlearning.utils.functionals import letterbox
from athena.src.tasks.detection.frameworks.pytorch.models.yolov7.models.experimental import attempt_load as attempt_load7
from athena.src.tasks.detection.frameworks.pytorch.models.yolov5.utils.general import check_img_size


compare_mask = True
imgsz = 1664
device = 'cuda'
weights = '/DeepLearning/_projects/interojo/transparent/ver20/outputs/detection/2024_02_19_14_17/train/weights/last.pt'
ckpt = torch.load(weights)

# model = attempt_load7(weights, map_location=device)  # load FP32 model
# imgsz = check_img_size(imgsz, s=model.stride.max())  # check image size
# if isinstance(imgsz, int):
#     imgsz = (imgsz, imgsz)
# model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
# idx2class = {ii: vv for ii, vv in enumerate(model.classes)}

# compare_gt = True
# iou_threshold = 0.7
# max_dets = 50
# nms_conf_threshold = 0.5
# nms_iou_threshold = 0.25
# classes = ['BOX']
# idx2class = {idx: cls for idx, cls in enumerate(classes)}
# _classes = ['BOX']
# _idx2class = {idx: cls for idx, cls in enumerate(_classes)}
# input_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/split_dataset/val'
# json_dir = '/DeepLearning/etc/_athena_tests/benchmark/rich/split_dataset/val'
# output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/rich/results/{model_name}_results'

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
#     img, ratio, pad_values = letterbox(img, (1664, 1664))
    
#     torch_img = torch.from_numpy(np.transpose(img[np.newaxis, :], (0, 3, 1, 2))).to('cuda')
#     model.forward = model.forward_export
#     with torch.no_grad():
#         preds = model(torch_img)
#     pred = preds[0].detach().cpu()
#     dets, _ = nms_rotated(pred[:, :-1], pred[:, -1], iou_threshold=nms_iou_threshold)
#     confs, labels = torch.max(dets[:, 5:], axis=1)

#     idx2xyxys = {}
#     for det, conf, cls in zip(dets[:max_dets].numpy(), confs[:max_dets].numpy(), labels[:max_dets].numpy()):
#         if conf >= nms_conf_threshold:
#             if cls not in idx2xyxys.keys():
#                 idx2xyxys[cls] = []

#             det[0] -= pad_values[0]
#             det[1] -= pad_values[1]
#             det[0] /= ratio[0]
#             det[1] /= ratio[1]
#             det[2] /= ratio[0]
#             det[3] /= ratio[1]

#             poly_points = det[np.newaxis, ...]
#             poly_points = obb2poly_np_le90(poly_points)
#             poly_points = poly_points[:, :8].reshape(4, 2)
#             poly_points = np.round(poly_points, 2)

#             idx2xyxys[cls].append([list(poly_point) for poly_point in poly_points])
  
#     if _classes is not None:
#         new_idx2xyxys = {}
#         for idx, _cls in enumerate(_classes):
#             for jdx, cls in enumerate(idx2class.values()):
#                 if cls == _cls:
#                     new_idx2xyxys[idx] = idx2xyxys[jdx]
        
#         idx2xyxys = new_idx2xyxys
#         idx2class = _idx2class   
    
#     color_map = imgviz.label_colormap()[1:len(idx2class) + 1 + 1]
#     results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
#     if compare_gt:
#         _compare = vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold)
#         _compare.update({"img_file": img_file})
#         compare.update({filename: _compare})
#     else:
#         vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_gt=compare_gt, iou_threshold=iou_threshold)
            
# with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
#     json.dump(results, json_file, ensure_ascii=False, indent=4)

# if compare_gt:
#     with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
#         json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
#     df_compare = pd.DataFrame(compare)
#     df_compare_pixel = df_compare.loc['diff_iou'].T
#     df_compare_pixel.fillna(0, inplace=True)
#     df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))