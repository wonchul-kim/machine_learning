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
