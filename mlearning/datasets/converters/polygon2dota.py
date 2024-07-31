import numpy as np
import os.path as osp
from typing import Union
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from mlearning.datasets.converters.utils import get_obb_coord_by_roatate, get_obb_coord_by_convex, coords2polygon, coords2numpy

def polygon2dota_by_rotate(coords: Union[list, np.ndarray, Polygon], rotate_degree:int = 1, flatten: bool = False, output_dir: str = None):
    polygon = coords2polygon(coords)
    obb_coords, obb_polygon = get_obb_coord_by_roatate(polygon, rotate_degree, flatten)

    if output_dir and osp.exists(output_dir):
        plt.figure(figsize=(10, 10))
        plt.plot(*polygon.exterior.xy, color='blue', label='Original Polygon')
        plt.plot(*obb_polygon.exterior.xy, color='red', linestyle='--', label='Oriented Bounding Box')
        # plt.scatter(coords[:, 0], coords[:, 1], color='blue')
        plt.scatter(obb_coords[:, 0], obb_coords[:, 1], color='red')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon and its Oriented Bounding Box')
        plt.show()
        plt.savefig(osp.join(output_dir, "obb.png"))
        
    return obb_coords, obb_polygon

def polygon2dota_by_convex(coords: Union[list, np.ndarray, Polygon], flatten: bool = False, output_dir: str = None):
    numpy_coords = coords2numpy(coords)
    obb_coords, obb_polygon = get_obb_coord_by_convex(numpy_coords, flatten)

    if output_dir and osp.exists(output_dir):
        plt.figure(figsize=(10, 10))
        closed_numpy_coords = np.vstack([numpy_coords, numpy_coords[0]])
        plt.plot(closed_numpy_coords[:, 0], closed_numpy_coords[:, 1], color='blue', label='Original Polygon')
        plt.plot(*obb_polygon.exterior.xy, color='red', linestyle='--', label='Oriented Bounding Box')
        plt.scatter(numpy_coords[:, 0], numpy_coords[:, 1], color='blue')
        plt.scatter(obb_coords[:, 0], obb_coords[:, 1], color='red')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon and its Oriented Bounding Box')
        plt.show()
        plt.savefig(osp.join(output_dir, "obb.png"))

    return obb_coords, obb_polygon

if __name__ == '__main__':
    polygon_coords = [
        [1960.2086065298297, 788.6438090123423],
        [2332.7453506990687, 871.5613673329996],
        [2332.7453506990687, 871.5613673329996],
        [2227.913866250809, 1355.4445469614068],
        [1861.2998048187603, 1276.6728665567823]
    ]
    
    # polygon2dota_by_rotate(Polygon(np.array(polygon_coords)), 1, output_dir='/HDD/_projects/nothing')
    polygon2dota_by_convex(Polygon(np.array(polygon_coords)), output_dir='/HDD/_projects/nothing')