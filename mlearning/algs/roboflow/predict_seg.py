import glob 
import os
import os.path as osp
from ultralytics import YOLO

import imgviz
import json
import pandas as pd
from tqdm import tqdm
from mlearning.utils.vis.vis_seg import vis_seg


def predict_seg(input_dir, json_dir, output_dir, nms_conf_threshold, nms_iou_threshold, _classes, compare_mask, font_scale=10, draw_rect=True):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    img_files = glob.glob(osp.join(input_dir, f'*.{image_ext}'))

    assert len(img_files) != 0, RuntimeError(f"There is no {image_ext} image at {input_dir}")


    preds = {}
    compare = {}
    for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        result = model(img_file, save=False, imgsz=2048, conf=nms_conf_threshold, iou=nms_iou_threshold, verbose=False)[0]
        classes = result.boxes.cls.tolist()
        idx2class = result.names
        confs = result.boxes.conf.tolist()
        masks = result.masks
        boxes = result.boxes
        
        color_map = imgviz.label_colormap()[1:len(idx2class) + 1]
        idx2masks = {}
        for cls, mask, box, conf in zip(classes, masks, boxes, confs):
            if cls not in idx2masks:
                idx2masks[cls] = {'polygon': [], 'confidence': [], 'box': []}
            idx2masks[cls]['polygon'].append([xy.tolist() for xy in mask.xy][0])
            box = box.xyxy[0].detach().cpu().tolist() 
            idx2masks[cls]['box'].append([[box[0], box[1]], [box[2], box[3]]])
            idx2masks[cls]['confidence'].append(conf)
            
        if _classes is not None:
            _idx2class = {idx: cls for idx, cls in enumerate(_classes)}
            new_idx2masks = {}
            for idx, _cls in enumerate(_classes):
                for jdx, cls in enumerate(idx2class.values()):
                    if cls == _cls and jdx in idx2masks:
                        new_idx2masks[idx] = idx2masks[jdx]
            
            idx2masks = new_idx2masks
            idx2class = _idx2class   
        
        
        preds.update({filename: {'idx2masks': idx2masks, 'idx2class': idx2class, 'img_file': img_file}})
        
        if compare_mask:
            _compare = vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir, 
                               compare_mask=compare_mask, font_scale=font_scale, draw_rect=draw_rect)
            _compare.update({"img_file": img_file})
            compare.update({filename: _compare})
        else:
            vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir, 
                    compare_mask=compare_mask, font_scale=font_scale, draw_rect=draw_rect)
                
    with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
        json.dump(preds, json_file, ensure_ascii=False, indent=4)

    if compare_mask:
        with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
            json.dump(compare, json_file, ensure_ascii=False, indent=4)
        
        df_compare = pd.DataFrame(compare)
            
        df_compare_pixel = df_compare.loc['diff_pixel'].T
        df_compare_pixel.fillna(0, inplace=True)
        df_compare_pixel.to_csv(osp.join(output_dir, 'diff_pixel.csv'))
        
        df_compare_pixel = df_compare.loc['diff_iou'].T
        df_compare_pixel.fillna(0, inplace=True)
        df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))
        
if __name__ == '__main__':
    compare_mask = True
    weights_file = "/DeepLearning/_projects/LX/wonchul/240807/train/train/weights/best.pt"
    model = YOLO(weights_file) 

    image_ext = 'jpg'
    input_dir = '/Data/01.Image/LX/IMAGE/240807/easy/val'
    json_dir = '/Data/01.Image/LX/IMAGE/240807/easy/val'
    output_dir = '/DeepLearning/_projects/LX/wonchul/240807/test/easy/val'

    nms_conf_threshold = 0.1
    nms_iou_threshold = 1
    font_scale = 1
    _classes = ['TIMBER', 'SCREW']
    draw_rect = False

    predict_seg(input_dir, json_dir, output_dir, nms_conf_threshold, nms_iou_threshold, _classes, compare_mask, font_scale=font_scale, draw_rect=draw_rect)