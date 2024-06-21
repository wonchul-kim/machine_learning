import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from scipy.spatial import ConvexHull

# Provided polygon coordinates
polygon_coords = [
    [1960.2086065298297, 788.6438090123423],
    [2332.7453506990687, 871.5613673329996],
    [2332.7453506990687, 871.5613673329996],
    [2227.913866250809, 1355.4445469614068],
    [1861.2998048187603, 1276.6728665567823]
]

# Convert to numpy array for easier manipulation
polygon_coords = np.array(polygon_coords)

# Create a shapely polygon
polygon = Polygon(polygon_coords)

def get_obb_coord(polygon):
    def get_obb(polygon):
        min_area = float('inf')
        best_obb = None
        for angle in range(0, 180):
            rotated_polygon = rotate(polygon, angle, origin='centroid', use_radians=False)
            minx, miny, maxx, maxy = rotated_polygon.bounds
            area = (maxx - minx) * (maxy - miny)
            if area < min_area:
                min_area = area
                best_obb = (minx, miny, maxx, maxy, angle)
        return best_obb

    # Get the oriented bounding box
    minx, miny, maxx, maxy, angle = get_obb(polygon)

    # Create the bounding box polygon
    obb_polygon = Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy)
    ])

    # Rotate the bounding box back to the original orientation
    obb_polygon = rotate(obb_polygon, -angle, origin='centroid', use_radians=False)

    # Translate the bounding box to the original position
    obb_polygon = translate(obb_polygon, xoff=polygon.centroid.x - obb_polygon.centroid.x, yoff=polygon.centroid.y - obb_polygon.centroid.y)

    # Extract the coordinates of the bounding box
    obb_coords = np.array(list(obb_polygon.exterior.coords)[:-1])

    # DOTA format: x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
    category = 'unknown'
    difficult = 0
    dota_format = obb_coords.flatten().tolist() + [category, difficult]
    
    return obb_coords, obb_polygon

obb_coords, obb_polygon = get_obb_coord(polygon)

# Plot the original polygon and the OBB
plt.figure(figsize=(10, 10))
plt.plot(*polygon.exterior.xy, color='blue', label='Original Polygon')
plt.plot(*obb_polygon.exterior.xy, color='red', linestyle='--', label='Oriented Bounding Box')
plt.scatter(polygon_coords[:, 0], polygon_coords[:, 1], color='blue')
plt.scatter(obb_coords[:, 0], obb_coords[:, 1], color='red')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polygon and its Oriented Bounding Box')
plt.show()
plt.savefig("obb.png")

