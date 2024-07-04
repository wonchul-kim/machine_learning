from shapely.geometry import Polygon
import numpy as np 

def points2polygon(points):
    
    if isinstance(points, list):
        points = np.array(points)
        
    if isinstance(points, np.ndarray):
        points = Polygon(points)
    
    return points

def get_iou(points1, points2):
    
    polygon1 = points2polygon(points1)
    polygon2 = points2polygon(points2)
    
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    iou = intersection_area / union_area

    return iou