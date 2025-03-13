"""
Object Detection Utilities

This module provides functions for detecting objects in LiDAR point clouds.
Implementations include clustering algorithms, bounding box computation, and object classification.
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import sklearn.preprocessing
from scipy.spatial import ConvexHull


def euclidean_clustering(points, eps=0.5, min_points=10, max_points=10000):
    """
    Perform Euclidean clustering on point cloud using DBSCAN algorithm.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points (x, y, z)
        eps (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_points (int): Number of samples in a neighborhood for a point to be considered as a core point
        max_points (int): Maximum number of points for a valid cluster
        
    Returns:
        list: List of numpy arrays, each containing the points belonging to a cluster
        numpy.ndarray: Cluster labels for each point (-1 for noise)
    """
    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit(points[:, :3])
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    clusters = []
    
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        
        # Filter out clusters that are too large (may be ground residuals)
        if len(cluster_indices) <= max_points:
            cluster_points = points[cluster_indices]
            clusters.append(cluster_points)
    
    return clusters, labels


def open3d_clustering(points, eps=0.5, min_points=10, max_points=10000):
    """
    Perform point cloud clustering using Open3D's DBSCAN implementation.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points (x, y, z)
        eps (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_points (int): Number of samples in a neighborhood for a point to be considered as a core point
        max_points (int): Maximum number of points for a valid cluster
        
    Returns:
        list: List of Open3D point clouds, each representing a cluster
        list: List of indices for each cluster
    """
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Perform clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # Number of clusters in labels, ignoring noise if present
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    clusters = []
    cluster_indices = []
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        
        # Filter out clusters that are too large
        if len(indices) <= max_points:
            cluster_pcd = pcd.select_by_index(indices)
            clusters.append(cluster_pcd)
            cluster_indices.append(indices)
    
    return clusters, cluster_indices


def compute_bounding_boxes(clusters, oriented=True):
    """
    Compute axis-aligned or oriented bounding boxes for clusters.
    
    Args:
        clusters (list): List of numpy arrays, each containing points of a cluster
        oriented (bool): If True, compute oriented bounding boxes, otherwise axis-aligned
        
    Returns:
        list: List of dictionaries containing bounding box information
    """
    bounding_boxes = []
    
    for cluster in clusters:
        if len(cluster) < 4:  # Need at least 4 points for a 3D bounding box
            continue
            
        # Extract XYZ coordinates if additional features are present
        points = cluster[:, :3]
        
        if oriented:
            # Use PCA to find principal components (orientation)
            centered_points = points - np.mean(points, axis=0)
            cov = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Order eigenvectors by decreasing eigenvalues
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Project points onto principal components
            R = eigenvectors  # Rotation matrix
            projected_points = centered_points @ R
            
            # Find min/max along principal axes
            min_projected = np.min(projected_points, axis=0)
            max_projected = np.max(projected_points, axis=0)
            
            # Compute dimensions and center in principal component space
            dimensions = max_projected - min_projected
            center_projected = (min_projected + max_projected) / 2
            
            # Transform center back to original space
            center = center_projected @ R.T + np.mean(points, axis=0)
            
            # Create bounding box object
            bbox = {
                'center': center,
                'dimensions': dimensions,
                'rotation_matrix': R,
                'points': points
            }
        else:
            # Simple axis-aligned bounding box
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            center = (min_bound + max_bound) / 2
            dimensions = max_bound - min_bound
            
            bbox = {
                'center': center,
                'dimensions': dimensions,
                'min_bound': min_bound,
                'max_bound': max_bound,
                'points': points
            }
            
        bounding_boxes.append(bbox)
    
    return bounding_boxes


def compute_features(clusters):
    """
    Compute geometric features for each cluster for classification.
    
    Args:
        clusters (list): List of numpy arrays, each containing points of a cluster
        
    Returns:
        numpy.ndarray: Feature matrix with rows corresponding to clusters
        list: List of feature names
    """
    features = []
    
    for cluster in clusters:
        if len(cluster) < 4:  # Need at least 4 points for features
            continue
            
        # Extract points
        points = cluster[:, :3]
        
        # Compute basic statistics
        center = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        
        # Compute bounding box
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        dimensions = max_bound - min_bound
        volume = np.prod(dimensions)
        
        # Compute height (z-dimension)
        height = dimensions[2]
        
        # Compute aspect ratios
        width_length_ratio = dimensions[0] / max(dimensions[1], 1e-6)
        height_length_ratio = dimensions[2] / max(dimensions[1], 1e-6)
        
        # Compute point density
        density = len(points) / max(volume, 1e-6)
        
        # Try to compute 2D convex hull (in x-y plane)
        try:
            hull_2d = ConvexHull(points[:, :2])
            hull_area = hull_2d.volume  # In 2D, volume is area
            compactness = hull_area / (dimensions[0] * dimensions[1])
        except:
            hull_area = 0
            compactness = 0
        
        # Collect all features
        cluster_features = [
            len(points),  # Number of points
            dimensions[0], dimensions[1], dimensions[2],  # Width, length, height
            volume,
            width_length_ratio,
            height_length_ratio,
            density,
            hull_area,
            compactness
        ]
        
        features.append(cluster_features)
    
    # Convert to numpy array
    features = np.array(features)
    
    # Feature names for reference
    feature_names = [
        'num_points', 
        'width', 'length', 'height',
        'volume',
        'width_length_ratio',
        'height_length_ratio',
        'density',
        'hull_area',
        'compactness'
    ]
    
    return features, feature_names


def classify_objects(bounding_boxes, features):
    """
    Classify objects into categories based on geometric features.
    This is a simple rule-based classification for demonstration.
    
    Args:
        bounding_boxes (list): List of dictionaries with bounding box information
        features (numpy.ndarray): Feature matrix with rows corresponding to bounding boxes
        
    Returns:
        list: List of classification labels
    """
    if len(bounding_boxes) == 0 or features.shape[0] == 0:
        return []
    
    # Extract relevant features
    # Features are expected to be in the order defined in compute_features function
    heights = features[:, 3]  # Height is the 4th feature (0-indexed)
    width_length_ratios = features[:, 5]  # 6th feature
    densities = features[:, 7]  # 8th feature
    
    # Define simple rules for classification
    classifications = []
    
    for i in range(len(bounding_boxes)):
        height = heights[i]
        width_length_ratio = width_length_ratios[i]
        density = densities[i]
        dimensions = bounding_boxes[i]['dimensions']
        
        # Simple rules for demonstration
        if 1.5 < height < 3.0 and abs(width_length_ratio - 1.0) < 0.5:
            # Potential car
            if dimensions[0] < 2.5 and dimensions[1] < 2.5:
                classification = 'small_vehicle'
            elif dimensions[0] < 6.0 and dimensions[1] < 3.0:
                classification = 'car'
            else:
                classification = 'large_vehicle'
        elif 0.2 < height < 1.0:
            # Potential small object
            classification = 'small_object'
        elif 1.0 < height < 2.5 and max(dimensions[0], dimensions[1]) < 1.0:
            # Potential pedestrian
            classification = 'pedestrian'
        elif height > 3.0 and max(dimensions[0], dimensions[1]) < 2.0:
            # Potential pole or tree
            classification = 'pole'
        else:
            # Unknown
            classification = 'unknown'
            
        classifications.append(classification)
    
    return classifications


def detect_objects(points, ground_mask, eps=0.5, min_points=10, max_points=10000, oriented_boxes=True):
    """
    Complete pipeline for detecting objects in a point cloud.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of point cloud points
        ground_mask (numpy.ndarray): Boolean mask of ground points
        eps (float): Clustering distance parameter
        min_points (int): Minimum cluster size
        max_points (int): Maximum cluster size
        oriented_boxes (bool): Whether to compute oriented or axis-aligned bounding boxes
        
    Returns:
        list: List of dictionaries containing object information
        list: List of clusters
    """
    # Remove ground points
    non_ground_mask = ~ground_mask
    non_ground_points = points[non_ground_mask]
    
    # Perform clustering
    clusters, labels = euclidean_clustering(
        non_ground_points, 
        eps=eps, 
        min_points=min_points,
        max_points=max_points
    )
    
    # Compute bounding boxes
    bounding_boxes = compute_bounding_boxes(clusters, oriented=oriented_boxes)
    
    # Compute features
    features, feature_names = compute_features(clusters)
    
    # Classify objects
    classifications = classify_objects(bounding_boxes, features)
    
    # Create final object list
    objects = []
    for i, (bbox, label) in enumerate(zip(bounding_boxes, classifications)):
        obj = {
            'id': i,
            'center': bbox['center'],
            'dimensions': bbox['dimensions'],
            'class': label,
            'bbox': bbox
        }
        objects.append(obj)
    
    return objects, clusters