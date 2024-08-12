import numpy as np
from collections import Counter
import cv2
import os.path as osp


# https://github.com/rafaelpadilla/review_object_detection_metrics?tab=readme-ov-file#ap-with-iou-threshold-t05

def is_overlapped(box_1, box_2):
    if box_1[0] > box_2[2]:
        return False
    if box_2[0] > box_1[2]:
        return False
    if box_1[3] < box_2[1]:
        return False
    if box_1[1] > box_2[3]:
        return False
    
    return True

def get_area(box):
    
    area = (box[2] - box[0])*(box[3] - box[1])
    
    return area 

def get_overlap_area(box_1, box_2):
    lt_x = max(box_1[0], box_2[0])
    lt_y = max(box_1[1], box_2[1])
    rb_x = min(box_1[2], box_2[2])
    rb_y = min(box_1[3], box_2[3])

    overlap_area = (rb_x - lt_x)*(rb_y - lt_y)
    
    return overlap_area

def get_iou(box_1, box_2, return_dict=False):
    
    if not is_overlapped(box_1, box_2):
        return 0
    
    area_1 = get_area(box_1)
    area_2 = get_area(box_2)
    
    overlap_area = get_overlap_area(box_1, box_2)
    assert overlap_area > 0, RuntimeError(f"ERROR: overlap-area must be more than 0, not {overlap_area}")
    
    
    iou = overlap_area/float(area_1 + area_2 - overlap_area)
    assert iou >= 0, RuntimeError(f"ERROR: iou must be more than 0, not {iou}")
    
    if return_dict:
        return {'iou': iou, 'area_1': area_1, 'area_2': area_2, 'overlap_area': overlap_area}
    else:
        return iou
    
def ElevenPointInterpolatedAP(rec, prec):
    mrec = [e for e in rec]
    mpre = [e for e in prec]

    # recallValues = [1.0, 0.9, ..., 0.0]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []


    for r in recallValues:
        # r : recall값의 구간
        # argGreaterRecalls : r보다 큰 값의 index
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        print(r, argGreaterRecalls)

        # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11
    
    return [ap, rhoInterp, recallValues, None]

def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']
    mAP = ap / len(result)
    
    return mAP

def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

def get_average_precision(detections, ground_truths, classes, iou_threhold=0.3, method='ap'):
    '''
        detections: ['image filename', class-index, confidence, (x1, y1, x2, y2)]
        ground_truths: ['image filename', class-index, confidence, (x1, y1, x2, y2)]
    '''
    
    results_by_class = []
    results_by_image = {}
    # loop by each class for all images
    for _class in classes:
        
        # Extract info for each class
        dets = [det for det in detections if det[1] == _class]
        gts = [gt for gt in ground_truths if gt[1] == _class]
        
        num_gt = len(gts) # len(tp) + len(tn)
        
        dets = sorted(dets, key=lambda conf: conf[2], reverse=True) # descending by confidence

        g_tp, g_fp = np.zeros(len(dets)), np.zeros(len(dets))
        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        gt_box_detected_map = Counter(c[0] for c in gts) # number of gt-boxes by image
        for key, val in gt_box_detected_map.items():
            gt_box_detected_map[key] = np.zeros(val)
            
        for det_index, det in enumerate(dets):
            
            # match dets and gts by image
            gt = [gt for gt in gts if gt[0] == det[0]]
            
            max_iou = 0
            for gt_index, _gt in enumerate(gt):
                '''
                    Within the same image, compare all gt-boxes for each det-box and then, calculate iou.
                    match the det-box and gt-box by the maximum iou.
                '''
                iou = get_iou(det[3], _gt[3])
                if iou > max_iou:
                    max_iou = iou 
                    max_gt_index = gt_index
                    
                if iou >= iou_threhold:
                    g_tp[det_index] = 1
                else:
                    g_fp[det_index] = 1
            
            g_fp = np.array([max(0, d) for d in g_fp - g_tp])
                    
            if max_iou >= iou_threhold:
                '''
                    * tp: 
                        - bigger than iou-threhold 
                        - gt-box is not matched yet
                    * fp:
                        - otherwise
                '''
                if gt_box_detected_map[det[0]][max_gt_index] == 0:
                    tp[det_index] = 1
                    gt_box_detected_map[det[0]][max_gt_index] = 1
                else:
                    fp[det_index] = 1
            else:
                fp[det_index] = 1
                
        for idx, (image_name, _num_gt) in enumerate(gt_box_detected_map.items()):
            num_gt = len(_num_gt)
            if image_name not in results_by_image:
                results_by_image[image_name] = {}
                
            _tp = np.sum(tp) if len(tp) != 0 else 0
            _fp = np.sum(fp) if len(fp) != 0 else 0
                
            results_by_image[image_name].update({
                                                _class: {
                                                    'precision': _tp/float(_tp + _fp) if _tp + _fp != 0 else 0,
                                                    'recall': _tp/float(num_gt),
                                                    'total gt': num_gt,
                                                    'TP': _tp,
                                                    'FP': _fp,
                                                    'FN': num_gt - _tp,
                                                    'g-TP': np.sum(g_tp) if len(g_tp) != 0 else 0,
                                                    'g-FP': np.sum(g_fp) if len(g_fp) != 0 else 0,
                                                    'g-FN': np.sum(_num_gt==0),
                                                    'g-FPR': np.sum(_num_gt==0)/num_gt if num_gt != 0 else 0
                                                }
                                            })

        accumulated_tp = np.cumsum(tp)
        accumulated_fp = np.cumsum(fp)
        accumulated_precision = np.divide(accumulated_tp, (accumulated_tp + accumulated_fp))
        accumulated_recall = accumulated_tp/num_gt
                    
        if method.lower() == 'ap':
            [ap, mean_precision, mean_recall, ii] = calculateAveragePrecision(accumulated_recall, accumulated_precision)
        else:
            [ap, mean_precision, mean_recall, _] = ElevenPointInterpolatedAP(accumulated_recall, accumulated_precision)

        result_by_class = {
            'class' : _class,
            'accumulated_precision' : accumulated_precision,
            'accumulated_recall' : accumulated_recall,
            'precision': accumulated_precision[-1] if len(accumulated_precision) != 0 else 0,
            'recall': accumulated_recall[-1] if len(accumulated_recall) != 0 else 0,
            'AP' : ap,
            'interpolated precision' : mean_precision,
            'interpolated recall' : mean_recall,
            'total gt' : num_gt,
            'total TP' : np.sum(tp),
            'total FP' : np.sum(fp),
            'total FN' : num_gt - np.sum(tp),
        }
        
        results_by_class.append(result_by_class)
        
    return {'by_class': results_by_class, 'by_image': results_by_image, 'map': mAP(results_by_class)}



if __name__ == '__main__':
    
    # test iou
    box_1 = [0, 0, 5, 5]
    box_2 = [4, 4, 8, 8]
    
    iou = get_iou(box_1, box_2)
    print(iou)
    
    # test ap (class-index, confidence, x1, y1, x2, y2)
    classes = [0, 1, 2]
    detections = [['image1.png', 0, 1, (3, 3, 8, 8)], ['image1.png', 1, 1, (10, 10, 15, 15)], 
                  ['image2.png', 0, 1, (3, 3, 8, 8)], ['image2.png', 1, 1, (10, 10, 15, 15)]]
    ground_truths = [['image1.png', 0, 1, (0, 0, 9, 9)], ['image1.png', 1, 1, (11, 11, 16, 16)], ['image1.png', 1, 1, (30, 30, 36, 36)], ['image1.png', 2, 1, (20, 20, 25, 25)], 
                     ['image2.png', 0, 1, (0, 0, 9, 9)], ['image2.png', 1, 1, (11, 11, 16, 16)], ['image2.png', 1, 1, (30, 30, 36, 36)], ['image2.png', 2, 1, (20, 20, 25, 25)]]
    iou_threshold = 0.3
    
    ap = get_average_precision(detections, ground_truths, classes, iou_threshold)
    print(ap['map'])
    print(ap['by_class'])
    print(ap['by_image'])
