import os.path as osp
import cv2 
import numpy as np
import warnings
from mlearning.utils.metrics.iou import get_iou
from shapely.geometry import (GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
                              mapping)
from shapely.ops import polygonize, unary_union
from copy import deepcopy

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

def vis_obb(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir=None, 
            compare_gt=False, iou_threshold=0.2, line_width=2, font_scale=1,
            output_img_ext='png', output_img_size_ratio=1):
    
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    height, width, channel = img.shape
    vis_obb = deepcopy(img)
    if compare_gt:
        diff_dict = {}
        points_dict = {'gt': {}, 'pred': {}}
    vis_gt = None, None
    
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
        vis_gt, vis_compare = deepcopy(img), deepcopy(img)
        text_gt = np.zeros((50, width, 3), np.uint8)
        cv2.putText(text_gt, "GT", origin, font, 0.6, (255, 255, 255), 1)
        text_compare = np.zeros((50, width, 3), np.uint8)
        cv2.putText(text_compare, "Compare", origin, font, 0.6, (255, 255, 255), 1)
        
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
                    cv2.polylines(vis_gt, [np.array(points, dtype=np.int32)], True, 
                                  tuple(map(int, color_map[-1])), line_width + 2)

                    cv2.putText(vis_gt, label, get_text_coords(points, width, height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[-1])), line_width)
                    if compare_gt:
                        cv2.polylines(vis_compare, [np.array(points, dtype=np.int32)], True, 
                                    tuple(map(int, color_map[-1])), line_width + 2)
                        cv2.putText(vis_compare, label, get_text_coords(points, width, height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[-1])), line_width)
                        if label in points_dict['gt']:
                            points_dict['gt'][label].append(points)
                        else:
                            points_dict['gt'].update({label: [points]})
                else:
                    warnings.warn(f"The points is {points} with {shape_type}")
                
        vis_gt = vis_gt.astype(np.uint8)
        vis_gt = cv2.vconcat([text_gt, vis_gt])
            
            
    for cls, pred in idx2xyxys.items():
        for xyxy, conf in zip(pred['polygon'], pred['confidence']):
            points = np.array(xyxy, dtype=np.int32)
            cv2.putText(vis_obb, f"{idx2class[int(cls)]} {conf:.2f}", get_text_coords(points, width, height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[int(cls)])), line_width)
            cv2.polylines(vis_obb, [points], True,
                          tuple(map(int, color_map[int(cls)])), line_width)
            if compare_gt:
                cv2.putText(vis_compare, f"{idx2class[int(cls)]}", get_text_coords(points, width, height, offset_h=30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[int(cls)])), line_width)
                cv2.polylines(vis_compare, [np.array(xyxy, dtype=np.int32)], True,
                          tuple(map(int, color_map[int(cls)])), line_width)
                
                if idx2class[int(cls)] in points_dict['pred']:
                    points_dict['pred'][idx2class[int(cls)]].append(points.tolist())
                else:
                    points_dict['pred'].update({idx2class[int(cls)]: [points.tolist()]})

    vis_img = cv2.vconcat([text_ori, img])

    vis_obb = vis_obb.astype(np.uint8)
    vis_obb = cv2.vconcat([text_pred, vis_obb])

    vis_legend = np.zeros((height, 300, channel), dtype="uint8")
    for idx, color in enumerate(color_map):
        color = [int(c) for c in color]
        cv2.rectangle(vis_legend, (150, (idx * 25)), (300, (idx * 25) + 25), tuple(color), -1)
        if idx < len(color_map) - 1:
            cv2.putText(vis_legend, idx2class[idx], (5, (idx * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(vis_legend, 'GT', (5, (idx * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    vis_legend = cv2.vconcat([text_legends, vis_legend])

    if vis_gt is None:
        vis_res = cv2.hconcat([vis_img, vis_obb, vis_legend])
    else:
        if compare_gt:
            vis_compare = cv2.vconcat([text_compare, vis_compare])
            vis_res = cv2.hconcat([vis_img, vis_gt, vis_obb, vis_compare, vis_legend])
        else:
            vis_res = cv2.hconcat([vis_img, vis_obb, vis_legend])

    if output_img_size_ratio == 1:
        cv2.imwrite(osp.join(output_dir, filename + f'.{output_img_ext}'), vis_res)
    else:
        vis_res = cv2.resize(vis_res, (int(vis_res.shape[1]*output_img_size_ratio), int(vis_res.shape[0]*output_img_size_ratio)))
        cv2.imwrite(osp.join(output_dir, filename + f'.{output_img_ext}'), vis_res)
    if compare_gt:
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
            
        return {'diff_iou': ious}