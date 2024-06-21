import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def calculate_oriented_bounding_box_convexhull(vertices):
    # Convert vertices to numpy array
    vertices = np.array(vertices)
    
    # Step 1: Compute Convex Hull
    hull = ConvexHull(vertices)
    
    # Get the vertices of the convex hull in counterclockwise order
    hull_vertices = vertices[hull.vertices]
    
    # Step 2: Find the direction of the bounding box
    # We will find the longest edge of the convex hull, which represents the bounding box direction
    num_vertices = len(hull_vertices)
    max_edge_length = 0
    max_edge_index = 0
    
    for i in range(num_vertices):
        # Compute edge length (squared to avoid square root)
        edge_length_sq = np.sum((hull_vertices[i] - hull_vertices[(i + 1) % num_vertices]) ** 2)
        
        if edge_length_sq > max_edge_length:
            max_edge_length = edge_length_sq
            max_edge_index = i
    
    # Direction vector along the longest edge
    direction = hull_vertices[(max_edge_index + 1) % num_vertices] - hull_vertices[max_edge_index]
    
    # Step 3: Rotate vertices according to the direction found
    angle = np.arctan2(direction[1], direction[0])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    
    rotated_vertices = np.dot(vertices - np.mean(vertices, axis=0), rotation_matrix.T)
    
    # Step 4: Find bounding box coordinates
    min_x = np.min(rotated_vertices[:, 0])
    max_x = np.max(rotated_vertices[:, 0])
    min_y = np.min(rotated_vertices[:, 1])
    max_y = np.max(rotated_vertices[:, 1])
    
    # Bounding box corners
    bbox_vertices = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    
    # Rotate back the bounding box corners to the original coordinates
    inv_rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                                    [np.sin(-angle), np.cos(-angle)]])
    
    bbox_vertices = np.dot(bbox_vertices, inv_rotation_matrix.T)
    bbox_vertices += np.mean(vertices, axis=0)  # Translate back to original coordinates
    
    return hull_vertices, bbox_vertices

# Example vertices
vertices = np.array([
    [1960.2086065298297, 788.6438090123423],
    [2332.7453506990687, 871.5613673329996],
    [2332.7453506990687, 871.5613673329996],
    [2227.913866250809, 1355.4445469614068],
    [1861.2998048187603, 1276.6728665567823]
])

# Calculate oriented bounding box and convex hull vertices
hull_vertices, bbox_vertices = calculate_oriented_bounding_box_convexhull(vertices)

# Visualization
plt.figure(figsize=(8, 6))

# Plot original vertices
plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Original Vertices')

# Plot convex hull
plt.fill(hull_vertices[:, 0], hull_vertices[:, 1], edgecolor='b', fill=False, linewidth=2, label='Convex Hull')

# Plot bounding box
bbox_x = np.append(bbox_vertices[:, 0], bbox_vertices[0, 0])  # Closing the polygon
bbox_y = np.append(bbox_vertices[:, 1], bbox_vertices[0, 1])  # Closing the polygon
plt.plot(bbox_x, bbox_y, 'r-', linewidth=2, label='Oriented Bounding Box')

# Set plot attributes
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison: Original Vertices, Convex Hull, and Oriented Bounding Box')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()
