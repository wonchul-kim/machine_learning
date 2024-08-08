import os.path as osp
import cv2 
import numpy as np
import warnings
from mlearning.utils.metrics.iou import get_iou
from shapely.geometry import (GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
                              mapping)
from shapely.ops import polygonize, unary_union

def get_text_coords(points, width, height, offset_w=0, offset_h=10):
    text_coord_x, text_coord_y = int(np.min(points, axis=0)[0]), int(np.min(points, axis=0)[1] - 10)
    if text_coord_x < offset_w:
        text_coord_x = int(np.max(points, axis=0)[0] + offset_w)
        
    if text_coord_x > width - 100:
        text_coord_x = 10
        
    if text_coord_y < offset_h:
        text_coord_y = int(np.max(points, axis=0)[1] + offset_h)
        
    if text_coord_y > height - 100:
        text_coord_y = 10
        
    return (text_coord_x, text_coord_y)
    
def handle_self_intersection(points):
    new_points = []
    line = LineString([[int(x), int(y)] for x, y in points + [points[0]]])

    polygons = list(polygonize(unary_union(line)))

    if len(polygons) > 1:
        print("The line is forming polygons by intersecting itself")
        for polygon in polygons:
            polygon = [list(item) for item in mapping(polygon)['coordinates'][0]]
            new_points.append(polygon[:-1])
    else:
        return [points]

    return new_points


def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def vis_seg(img_file, idx2masks, idx2class, output_dir, color_map, json_dir=None, compare_mask=True, 
            iou_threshold=0.2, font_scale=1, line_width=1, draw_rect=True):
   
    class2idx = {cls: idx for idx, cls in idx2class.items()}
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    height, width, channel = img.shape
    vis_mask = np.zeros((height, width, channel))
    if compare_mask:
        diff_dict = {}
        points_dict = {'gt': {}, 'pred': {}}
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
                    cv2.putText(vis_gt, label, get_text_coords(points, width, height, offset_h=10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[-1])), 3)
                    if compare_mask:
                        cv2.fillConvexPoly(compr_gt_mask[label], np.array(points, dtype=np.int32), 
                                color=1)
                        # cv2.putText(compr_gt_mask[label], label, get_text_coords(points, width, height, offset_h=10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[-1])), 3)
                        if label in points_dict['gt']:
                            points_dict['gt'][label].append(points)
                        else:
                            points_dict['gt'].update({label: [points]})
                else:
                    warnings.warn(f"The points is {points} with {shape_type}")
                
        vis_gt = vis_gt.astype(np.uint8)
        vis_gt = cv2.addWeighted(img, 0.4, vis_gt, 0.6, 0)
        vis_gt = cv2.vconcat([text_gt, vis_gt])
            
            
    for idx, cls in idx2class.items():
        if class2idx[cls] in idx2masks:
            pred = idx2masks[class2idx[cls]]
            for mask, box, conf in zip(pred['polygon'], pred['box'], pred['confidence']):
                points = np.array(mask, dtype=np.int32)
                if points.size != 0:
                    cv2.putText(vis_mask, f"{cls} {conf:.2f}", 
                                get_text_coords(points, width, height, offset_h=10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                tuple(map(int, color_map[int(idx)])), 3)
                    cv2.fillConvexPoly(vis_mask, points, color=tuple(map(int, color_map[int(idx)])))
                    # cv2.fillPoly(vis_mask, [points], color=tuple(map(int, color_map[int(idx)])))
                    if compare_mask:
                        # cv2.putText(compr_pred_mask[idx], f"{idx}", get_text_coords(points, width, height, offset_h=30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[int(cls)])), 3)
                        cv2.fillConvexPoly(compr_pred_mask[cls], points, color=1)

                        if cls in points_dict['pred']:
                            points_dict['pred'][cls].append(points.tolist())
                        else:
                            points_dict['pred'].update({cls: [points.tolist()]})
                
                if draw_rect and len(box) != 0:
                    cv2.rectangle(vis_mask, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), 
                                    tuple(map(int, color_map[int(idx)])), line_width + 1)
                    cv2.putText(vis_mask, label, get_text_coords(box, width, height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                tuple(map(int, color_map[int(idx)])), line_width)

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
                diff_dict[key] = diff
            else:
                vis_compr = val - compr_pred_mask[key]
                diff = np.sum(np.abs(vis_compr))
                cv2.putText(text_compr, f'{key} ({str(diff)})', origin, font, 0.6, (255, 255, 255), 1)
                vis_compr = cv2.vconcat([text_compr, vis_compr])
                diff_dict[key] = diff
        
        vis_compr = 255*np.abs(vis_compr)
        cv2.imwrite(osp.join(output_dir, filename + '_diff.png'), vis_compr)
            
        ious = {}
        for gt_cls, gt_points in points_dict['gt'].items():
            for pred_cls, pred_points in points_dict['pred'].items():
                if gt_cls == pred_cls:
                    for gt_point in gt_points:
                        for pred_point in pred_points:
                            try:
                                iou = get_iou(gt_point, pred_point)
                                if iou > iou_threshold:
                                    if gt_cls in ious:
                                        ious[gt_cls].append(iou)
                                    else:
                                        ious[gt_cls] = [iou]
                                
                            except:
                                _gt_points = handle_self_intersection(gt_point)
                                
                                for _gt_point in _gt_points:
                                    iou = get_iou(_gt_point, pred_point)
                                    if iou > iou_threshold:
                                        if gt_cls in ious:
                                            ious[gt_cls].append(iou)
                                        else:
                                            ious[gt_cls] = [iou]
            
        return {'diff_pixel': diff_dict, 'diff_iou': ious}