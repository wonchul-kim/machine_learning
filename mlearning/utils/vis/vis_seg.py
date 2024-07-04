import os.path as osp
import cv2 
import numpy as np
import warnings

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir=None, compare_mask=True):
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    height, width, channel = img.shape
    vis_mask = np.zeros((height, width, channel))
    if compare_mask:
        compare = {}
        compr_pred_mask = {cls: np.zeros((height, width)) for cls in idx2class.values()}
    else:
        compr_pred_mask = None
    compr_gt_mask, vis_gt = None, None
    
    origin = 25, 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_ori = np.zeros((50, width, 3), np.uint8)
    text_pred = np.zeros((50, width, 3), np.uint8)
    text_legends = np.zeros((50, 300, 3), np.uint8)
    cv2.putText(text_ori, "Original", origin, font, 0.6, (255, 255, 255), 1)
    cv2.putText(text_pred, "Predicted", origin, font, 0.6, (255, 255, 255), 1)
    cv2.putText(text_legends, "Legends", origin, font, 0.6, (255, 255, 255), 1)
    
    if json_dir is not None:
        import json
        json_file = osp.join(json_dir, filename + '.json')
        vis_gt = np.zeros((height, width, channel))
        if compare_mask:
            compr_gt_mask = {cls: np.zeros((height, width)) for cls in idx2class.values()}
        text_gt = np.zeros((50, width, 3), np.uint8)
        cv2.putText(text_gt, "GT", origin, font, 0.6, (255, 255, 255), 1)
        
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        for idx, cls in idx2class.items():
            for ann in anns:
                label = ann['label']
                if label != cls:
                    continue
                points = ann['points']
                shape_type = ann['shape_type']
                
                if len(points) > 2:
                    cv2.fillConvexPoly(vis_gt, np.array(points, dtype=np.int32), 
                                color=tuple(map(int, color_map[int(get_key_by_value(idx2class, label))])))
                    if compare_mask:
                        cv2.fillConvexPoly(compr_gt_mask[label], np.array(points, dtype=np.int32), 
                                color=1)
                else:
                    warnings.warn(f"The points is {points} with {shape_type}")
                
        vis_gt = vis_gt.astype(np.uint8)
        vis_gt = cv2.addWeighted(img, 0.4, vis_gt, 0.6, 0)
        vis_gt = cv2.vconcat([text_gt, vis_gt])
            
            
    for cls, masks in idx2masks.items():
        for mask in masks:
            points = np.array(mask, dtype=np.int32)
            cv2.fillConvexPoly(vis_mask, points, color=tuple(map(int, color_map[int(cls)])))
            # cv2.fillPoly(vis_mask, [points], color=tuple(map(int, color_map[int(cls)])))
            if compare_mask:
                cv2.fillConvexPoly(compr_pred_mask[idx2class[int(cls)]], points, color=1)

    vis_img = cv2.vconcat([text_ori, img])

    vis_mask = vis_mask.astype(np.uint8)
    vis_mask = cv2.addWeighted(img, 0.4, vis_mask, 0.6, 0)
    vis_mask = cv2.vconcat([text_pred, vis_mask])

    vis_legend = np.zeros((height, 300, channel), dtype="uint8")
    for idx, color in enumerate(color_map):
        color = [int(c) for c in color]
        cv2.putText(vis_legend, idx2class[idx], (5, (idx * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(vis_legend, (150, (idx * 25)), (300, (idx * 25) + 25), tuple(color), -1)
    vis_legend = cv2.vconcat([text_legends, vis_legend])

    if vis_gt is None:
        vis_res = cv2.hconcat([vis_img, vis_mask, vis_legend])
    else:
        vis_res = cv2.hconcat([vis_img, vis_gt, vis_mask, vis_legend])

    cv2.imwrite(osp.join(output_dir, filename + '.png'), vis_res)
    if compare_mask:
        for idx, (key, val) in enumerate(compr_gt_mask.items()):
            origin = 25, 25
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_compr = np.zeros((50, val.shape[1]))
            if idx != 0:
                _vis_compr = val - compr_pred_mask[key]
                diff = np.sum(np.abs(vis_compr))
                cv2.putText(text_compr, f'{key} ({str(diff)})', origin, font, 0.6, (255, 255, 255), 1)
                _vis_compr = cv2.vconcat([text_compr, _vis_compr])
                vis_compr = cv2.hconcat([vis_compr, _vis_compr])
                compare[key] = diff
            else:
                vis_compr = val - compr_pred_mask[key]
                diff = np.sum(np.abs(vis_compr))
                cv2.putText(text_compr, f'{key} ({str(diff)})', origin, font, 0.6, (255, 255, 255), 1)
                vis_compr = cv2.vconcat([text_compr, vis_compr])
                compare[key] = diff
        
        vis_compr = 255*np.abs(vis_compr)
        cv2.imwrite(osp.join(output_dir, filename + '_diff.png'), vis_compr)
            
        return compare