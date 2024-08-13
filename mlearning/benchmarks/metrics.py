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
        ap += r['ap']
    mAP = ap / len(result)
    
    return mAP

def get_average_g_results(g_results):
    result = {}
    total_avg_g_fnr, total_avg_g_fpr = 0, 0
    cnt = 0
    for image_name, g_result in g_results.items():
        avg_g_fnr, avg_g_fpr = 0, 0
        for key, val in g_result.items():
            avg_g_fnr += val['g_fnr']
            avg_g_fpr += val['g_fpr']
            cnt += 1
        total_avg_g_fnr += avg_g_fnr
        total_avg_g_fpr += avg_g_fpr
        result[image_name] = {'avg_g_fnr': avg_g_fnr/len(val), 
                              'avg_g_fpr': avg_g_fpr/len(val)
                             }
        
    result['total_avg_g_fnr'] = total_avg_g_fnr/cnt
    result['total_avg_g_fpr'] = total_avg_g_fpr/cnt
    
    return result
        
        
def update_ap_by_image(results_by_image):
    
    overall_by_class = {}
    if len(results_by_image) != 0:
        for image_name, val in results_by_image.items():
            for _class, results in val.items():
                fpr = results['fp']/(results['tp'] + results['fp'] + 1e-5)
                fnr = results['fn']/(results['tp'] + results['fn'] + 1e-5)
                results.update({'fpr': fpr, 'fnr': fnr})
                
                if _class not in overall_by_class:
                    overall_by_class[_class] = {'fpr': [], 'fnr': [], 'tp': [], 'fp': [], 'fn': [], 'tn': [], 'total_gt': []}
                    
                overall_by_class[_class]['fpr'].append(fpr)
                overall_by_class[_class]['fnr'].append(fnr)
                overall_by_class[_class]['tp'].append(results['tp'])
                overall_by_class[_class]['fp'].append(results['fp'])
                overall_by_class[_class]['fn'].append(results['fn'])
                overall_by_class[_class]['tn'].append(results['tn'])
                overall_by_class[_class]['total_gt'].append(results['total_gt'])
                
        for key, val in overall_by_class.items():
            overall_by_class[key]['fpr'] = np.mean(overall_by_class[key]['fpr'])
            overall_by_class[key]['fnr'] = np.mean(overall_by_class[key]['fnr'])
            overall_by_class[key]['tp'] = np.sum(overall_by_class[key]['tp'])
            overall_by_class[key]['fp'] = np.sum(overall_by_class[key]['fp'])
            overall_by_class[key]['fn'] = np.sum(overall_by_class[key]['fn'])
            overall_by_class[key]['tn'] = np.sum(overall_by_class[key]['tn'])
            overall_by_class[key]['total_gt'] = np.sum(overall_by_class[key]['total_gt'])
                
    results_by_image['overall'] = overall_by_class
    
    return results_by_image
    _

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

def get_performance(detections, ground_truths, classes, iou_threhold=0.3, method='ap'):
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

        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        gt_box_detected_map = Counter(c[0] for c in gts) # number of gt-boxes by image
        for key, val in gt_box_detected_map.items():
            gt_box_detected_map[key] = np.zeros(val)
            
        for det_index, det in enumerate(dets):
            
            # match dets and gts by image
            gt = [gt for gt in gts if gt[0] == det[0]]
            
            if det[0] not in results_by_image:
                results_by_image[det[0]] = {_class: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_gt': len(gt)}}
            
            else: 
                if _class not in results_by_image[det[0]]:
                    results_by_image[det[0]].update({_class: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_gt': len(gt)}})

            max_iou, iou = 0, 0
            for gt_index, _gt in enumerate(gt):
                '''
                    Within the same image, compare all gt-boxes for each det-box and then, calculate iou.
                    match the det-box and gt-box by the maximum iou.
                '''
                iou = get_iou(det[3], _gt[3])
                if iou > max_iou:
                    max_iou = iou 
                    max_gt_index = gt_index
                    
            if iou != 0 and iou >= iou_threshold:
                results_by_image[det[0]][_class]['tp'] += 1
            else:
                results_by_image[det[0]][_class]['fp'] += 1
                    
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
                
                
        
        accumulated_tp = np.cumsum(tp)
        accumulated_fp = np.cumsum(fp)
        accumulated_precision = np.divide(accumulated_tp, (accumulated_tp + accumulated_fp))
        accumulated_recall = accumulated_tp/num_gt if num_gt != 0 else accumulated_tp
                    
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
            'ap' : ap,
            'interpolated_precision' : mean_precision,
            'interpolated_recall' : mean_recall,
            'total_gt' : num_gt,
            'total_tp' : np.sum(tp),
            'total_fp' : np.sum(fp),
            'total_fn' : num_gt - np.sum(tp),
        }
        
        results_by_class.append(result_by_class)
        
    results_by_image = update_ap_by_image(results_by_image)
    results_by_class.append({'map': mAP(results_by_class)})
        
        
    return {'by_class': results_by_class, 'by_image': results_by_image}

