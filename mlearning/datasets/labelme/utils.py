import json
import math
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SHAPE_TYPE = ['polygon', 'watershed', 'point', 'rectangle', 'image']


def get_roi_info(json_file, anns, roi_info, img_h, img_w):
    assert osp.exists(json_file), ValueError(f"There is no such json-file: {json_file}")

    if osp.exists(json_file) and 'rois' in anns and isinstance(anns['rois'], list) and len(anns['rois']) > 0:
        roi_info = anns['rois']
    else:
        if roi_info is None:
            roi_info = [[0, 0, img_w, img_h]]
        else:
            roi_info = roi_info

    return roi_info


def init_labelme_json(image_fn, imgsz_w, imgsz_h, imageData=None, imageId=None):
    labelme = {}

    labelme['version'] = "2.2.0"
    labelme['flags'] = {}
    labelme['shapes'] = []
    labelme["rois"] = []
    labelme["imageId"] = imageId
    labelme["imagePath"] = image_fn
    labelme["imageData"] = imageData
    labelme["imageHeight"] = imgsz_h
    labelme["imageWidth"] = imgsz_w

    return labelme


def add_labelme_element(labelme,
                        shape_type,
                        label,
                        points,
                        rotationrect=None,
                        group_id=None,
                        bbox=None,
                        flags={},
                        otherData=None):
    anns = {}
    anns['otherData'] = otherData
    anns["label"] = label
    anns["points"] = points
    anns["rotationrect"] = rotationrect
    anns["group_id"] = group_id
    anns["shape_type"] = shape_type
    anns["flags"] = flags
    anns['bbox'] = bbox

    labelme['shapes'].append(anns)

    return labelme


def read_labelme(json_file):

    if isinstance(json_file, str):
        assert osp.exists(json_file), f"There is no labelme json file: {json_file}"
    elif isinstance(json_file, Path):
        assert json_file.exists(), f"There is no labelme json file: {json_file}"
        json_file = str(json_file)

    with open(json_file) as f_json:
        ann = json.load(f_json)

    # version = ann['version']
    # flags = ann['flags']
    # shapes = ann['shapes']
    # imagePath = ann["imagePath"]
    # imageData = ann["imageData"]
    # imageHeight = ann["imageHeight"]
    # imageWidth = ann["imageWidth"]

    return ann


def xyxy_to_polygon(points):
    """This will change rectangle(p[x1, y1], [x2, y2]]) to polygon."""
    assert isinstance(points, list), ValueError(f"The type of points should be list, not {type(points)}")
    assert len(points) == 2, ValueError(f"The length of points({points}) should be 2, not {len(points)}")
    assert len(
        points[0]) == 2, ValueError(f"The length of one of points({points[0]}) should be 2, not {len(points[0])}")

    polygon = [points[0]]
    polygon.append([points[1][0], points[0][1]])
    polygon.append(points[1])
    polygon.append([points[0][0], points[1][1]])
    polygon.append(points[0])

    return polygon


def get_polygon_points_from_labelme_shape(shape, shape_type, include_point_positive, mode):
    """
    This will consider all points including point or line(two points) to check a negative sample for false positive
    cases.

    For negative samples,
        - if mode is 'train': include or not according to includ_point_positive
        - if mode is 'val': must include
    """

    assert shape_type in SHAPE_TYPE, ValueError(f"The shape-type({shape_type}) must be one of {SHAPE_TYPE}")

    points = shape['points']
    # TODO: Need to consider for empty json
    if len(points) == 0:  # handle exception where there is no points
        return []

    if shape_type in ['polygon', 'watershed', "point"]:
        if len(points) > 0 and len(points) <= 2:  # for a negative sample
            if mode in ['train']:
                if include_point_positive:
                    return points
                else:
                    return []
            elif mode in ['test', 'val']:
                return points
            else:
                raise RuntimeError(f"There is no such mode({mode}) for datasets")

    # TODO: How to consider circle as polygon
    # elif shape_type == 'circle':  # two points: center point and one of edge point
    #     points = [points[0]]

    elif shape_type == 'rectangle':
        points = xyxy_to_polygon(points)

    else:
        raise ValueError(f"There is no such shape-type: {shape_type}")

    return points


def get_mask_from_labels(labels, width, height, class2label, format='pil'):

    mask = np.zeros((height, width))
    for label_dict in labels:
        label = label_dict['label']
        points = label_dict['points']

        if len(points) != 0:
            arr = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [arr], color=(class2label[label]))

    if format == 'pil':
        return Image.fromarray(mask)
    elif format == 'opencv':
        return mask


def get_mask_from_labelme(json_file, width, height, class2label, format='pil', metis=None):

    if metis is None:
        with open(json_file) as f:
            anns = json.load(f)
    else:
        anns = {"shapes": metis}
    mask = np.zeros((height, width))
    for shapes in anns['shapes']:
        shape_type = shapes['shape_type'].lower()
        label = shapes['label'].lower()
        if label in class2label.keys():
            _points = shapes['points']
            if shape_type == 'circle':
                cx, cy = _points[0][0], _points[0][1]
                radius = int(math.sqrt((cx - _points[1][0]) ** 2 + (cy - _points[1][1]) ** 2))
                cv2.circle(mask, (int(cx), int(cy)), int(radius), True, -1)
            elif shape_type in ['rectangle']:
                if len(_points) == 2:
                    arr = np.array(_points, dtype=np.int32)
                else:
                    RuntimeError(f"Rectangle labeling should have 2 points")
                cv2.fillPoly(mask, [arr], color=(class2label[label]))
            elif shape_type in ['polygon', 'watershed']:
                if len(_points) > 2:  # handle cases for 1 point or 2 points
                    arr = np.array(_points, dtype=np.int32)
                else:
                    continue
                cv2.fillPoly(mask, [arr], color=(class2label[label]))
            elif shape_type in ['point']:
                pass
            else:
                raise ValueError(f"There is no such shape-type: {shape_type}")

    if format == 'pil':
        return Image.fromarray(mask)
    elif format == 'opencv':
        return mask
    else:
        NotImplementedError(f'There is no such case for {format}')


def get_points_from_image(image, class_list, roi, ltp, _labelme, contour_thres):
    for idx, cls_name in enumerate(class_list):
        new_mask = (image == idx + 1).astype(np.uint8)
        contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < contour_thres:
                pass
            else:
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                for objects in [approx]:
                    polygon = []
                    xs, ys = [], []
                    for jdx in range(0, len(objects)):
                        polygon.append([
                            int(objects[jdx][0][0] + roi[0] + ltp[0]),
                            int(objects[jdx][0][1] + roi[1] + ltp[1])])
                        xs.append(int(objects[jdx][0][0] + roi[0] + ltp[0]))
                        ys.append(int(objects[jdx][0][1] + roi[1] + ltp[1]))

                    bbox = [int(np.max(xs)), int(np.max(ys)), int(np.min(xs)), int(np.min(ys))]
                    _labelme = add_labelme_element(_labelme,
                                                   shape_type="polygon",
                                                   label=cls_name,
                                                   points=polygon,
                                                   bbox=bbox)
    return _labelme


def get_labeled_gt_image(gt_img, json_file, classes, color_map, logger=None):

    assert osp.exists(json_file), ValueError(f"There is no such json-file: {json_file}")

    test_y_offset = 8
    linewidth = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5

    try:
        with open(json_file) as f:
            anns = json.load(f)
    except Exception as error:
        raise Exception(f"There is something wrong in json_file: {json_file} - {error}")

    for shapes in anns['shapes']:
        label = shapes['label'].lower()
        if label in classes or label.lower() in classes:
            shape_type = shapes['shape_type'].lower()
            if shape_type in ['polygon', 'watershed', 'point']:
                if shape_type == 'polygon' or shape_type == 'watershed':
                    points = shapes['points']
                    if len(points) <= 2:
                        continue
                elif shape_type == 'point':
                    points = shapes['points']
                    if len(points) <= 1:
                        continue
                arr = np.array(points, dtype=np.int32)
                color = ([int(c) for c in color_map[classes.index(label) + 1]])
                cv2.fillPoly(gt_img, [arr], color=color)
                cv2.rectangle(gt_img, (int(np.min(arr, axis=0)[0]), int(np.min(arr, axis=0)[1])),
                              (int(np.max(arr, axis=0)[0]), int(np.max(arr, axis=0)[1])), color, linewidth)
                if np.min(arr, axis=0)[1] - 10 < 0:
                    text_y = int(np.min(arr, axis=0)[1] + 5)
                else:
                    text_y = int(np.min(arr, axis=0)[1] - 10)
                cv2.putText(gt_img, label, (int(np.min(arr, axis=0)[0]), text_y), font, font_size, color, 2)

            elif shape_type == 'rectangle':
                points = shapes['points']
                cv2.rectangle(gt_img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])),
                              color, linewidth)
                if points[0][1] - test_y_offset < 0:
                    text_y = int(points[0][1] + 5)
                else:
                    text_y = int(points[0][1] - test_y_offset)
                cv2.putText(gt_img, label, (int(points[0][0]), text_y), font, font_size, color, 2)
            elif shape_type == 'circle':
                _points = shapes['points']
                assert len(_points) == 2, 'Shape of shape_type=circle must have 2 points'
                (cx, cy), (px, py) = _points
                d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                cv2.circle(gt_img, (int(cx), int(cy)),
                           int(d),
                           color=([int(c) for c in color_map[classes.index(label) + 1]]),
                           thickness=-1)
                if cy - d - test_y_offset < 0:
                    text_y = int(cy - d + 5)
                else:
                    text_y = int(cy - d - test_y_offset)
                cv2.putText(gt_img, label, (int(cx - d), text_y), font, font_size, color, 2)
            else:
                if logger is not None:
                    logger.raise_error_log(ValueError, f"There is no such shape-type: {shape_type}")
                else:
                    raise ValueError(f"There is no such shape-type: {shape_type}")

    return gt_img, anns


if __name__ == '__main__':

    # json_file = "/HDD/datasets/projects/samkee/test_90_movingshot/split_dataset/val/20230213_64_Side64_94.json"
    # width = 1920
    # height = 1080
    # class2label = {'bubble': 0, 'dust': 1, 'line': 2, 'crack': 3, 'black': 4, 'peeling': 5, 'burr': 6}

    # mask = get_mask_from_labelme(json_file, width, height, class2label, 'opencv')
    # print(mask.shape)
    # cv2.imwrite("/projects/mask.png", mask)

    json_file = "/HDD/datasets/_unittests/multiple_rois/wo_patches/sungwoo_edge/split_datasets/val/122111520173660_7_EdgeDown.json"
    width = 9344
    height = 7000
    class2label = {'_background_': 0, 'stabbed': 1, 'pop': 2}

    mask = get_mask_from_labelme(json_file, width, height, class2label, 'opencv')
    print(mask.shape)
    cv2.imwrite("/projects/mask.png", mask)