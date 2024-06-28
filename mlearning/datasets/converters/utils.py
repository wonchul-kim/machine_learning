import numpy as np
import copy
import os.path as osp
from typing import Union
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from scipy.spatial import ConvexHull

def polygon2numpy(polygon_coords: Polygon):
    
    return np.array(list(polygon_coords.exterior.coords)[:-1])

def coords2polygon(coords: Union[list, np.ndarray]):
    polygon_coords = copy.deepcopy(coords)
    if isinstance(polygon_coords, list):
        polygon_coords = np.array(polygon_coords)
    
    if isinstance(polygon_coords, np.ndarray):
        polygon_coords = Polygon(polygon_coords)

    assert isinstance(polygon_coords, Polygon), ValueError(f"The type of input coords must be Polygon, not {type(polygon_coords)}")

    return polygon_coords

def coords2numpy(coords: Union[list, Polygon]):
    numpy_coords = copy.deepcopy(coords)
    if isinstance(numpy_coords, list):
        numpy_coords = np.array(numpy_coords)
    
    if isinstance(numpy_coords, Polygon):
        numpy_coords = polygon2numpy(numpy_coords)

    assert isinstance(numpy_coords, np.ndarray), ValueError(f"The type of input coords must be Numpy, not {type(numpy_coords)}")

    return numpy_coords

def get_obb_coord_by_convex(coords: np.ndarray, flatten:bool = False):
    hull = ConvexHull(coords)
    hull_polygon = Polygon(coords[hull.vertices])

    obb_polygon = hull_polygon.minimum_rotated_rectangle
    obb_coords = np.array(list(obb_polygon.exterior.coords)[:-1])

    if flatten:
        obb_coords = obb_coords.flatten().tolist()
        
    return obb_coords, obb_polygon  

def get_obb_coord_by_roatate(polygon: Polygon, rotate_degree:int = 1, flatten:bool = False):
    def get_obb(polygon, rotate_degree):
        min_area = float('inf')
        best_obb = None
        for angle in range(0, 180, rotate_degree):
            rotated_polygon = rotate(polygon, angle, origin='centroid', use_radians=False)
            minx, miny, maxx, maxy = rotated_polygon.bounds
            area = (maxx - minx) * (maxy - miny)
            if area < min_area:
                min_area = area
                best_obb = (minx, miny, maxx, maxy, angle)
        return best_obb

    minx, miny, maxx, maxy, angle = get_obb(polygon, rotate_degree)
    obb_polygon = Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy)
    ])

    obb_polygon = rotate(obb_polygon, -angle, origin='centroid', use_radians=False)
    obb_polygon = translate(obb_polygon, xoff=polygon.centroid.x - obb_polygon.centroid.x, yoff=polygon.centroid.y - obb_polygon.centroid.y)
    obb_coords = polygon2numpy(obb_polygon)
    if flatten:
        obb_coords = obb_coords.flatten().tolist()
    
    return obb_coords, obb_polygon

def xyxy2xywh(imgsz, xyxy):

    if not len(imgsz) >= 2:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")
    elif imgsz[0] <= 3:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")

    if isinstance(xyxy, list):
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    else:
        raise RuntimeError(f"xyxy must be list such as [x1, y1, x2, y2]")

    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin

    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    
    dw = 1./imgsz[1]
    dh = 1./imgsz[0]
    
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    
    return [x, y, w, h]

def xywh2xyxy(imgsz, xywh):

    if not len(imgsz) >= 2:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")
    elif imgsz[0] <= 3:
        raise RuntimeError(f"imgsz should be [height, width, channel] or [height, width]")

    if isinstance(xywh, list):
        x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    else:
        raise RuntimeError(f"xywh must be list such as [x, y, w, h]")

    dh, dw = imgsz[0], imgsz[1]
    l = ((x - w / 2) * dw) # x0
    r = ((x + w / 2) * dw) # x1
    t = ((y - h / 2) * dh) # y0
    b = ((y + h / 2) * dh) # y1
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1   

    return [l, t, r, b] # x0, y0, x1, y1