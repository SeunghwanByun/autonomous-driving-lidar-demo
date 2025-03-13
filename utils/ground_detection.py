"""
Ground Detection Utilities

This module provides functions for detecting and removing ground points from LiDAR point clouds.
Various ground plane detection methods are implemented, including RANSAC-based and height-based approaches.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor
import open3d as o3d


def ransac_ground_detection(points, distance_thresh=0.15, max_iterations=100, return_model=False):
    """
    Detect ground plane using RANSAC algorithm.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points (x, y, z)
        distance_thresh (float): Maximum distance for a point to be considered an inlier
        max_iterations (int): Maximum number of iterations for RANSAC
        return_model (bool): Whether to return the fitted model
        
    Returns:
        numpy.ndarray: Boolean mask of ground points
        dict (optional): Fitted ground plane model parameters (if return_model=True)
    """
    # Extract x and y coordinates as features and z as target
    X = points[:, :2]  # x, y as features
    y = points[:, 2]   # z as target
    
    # Create RANSAC regressor for plane fitting
    ransac = RANSACRegressor(
        max_trials=max_iterations, 
        residual_threshold=distance_thresh,
        random_state=42
    )
    
    # Fit the model
    ransac.fit(X, y)
    
    # Get inliers (ground points)
    inlier_mask = ransac.inlier_mask_
    
    if return_model:
        # Get model parameters (a, b, c) where z = a*x + b*y + c
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # Convert to normal form: ax + by + cz + d = 0 where c = -1
        # So: a'x + b'y - z + d' = 0
        # Where a'=a, b'=b, d'=c
        model = {
            'normal': np.array([a, b, -1.0]),
            'intercept': c,
            'coefficients': np.array([a, b, c])
        }
        
        # Normalize the normal vector
        norm = np.linalg.norm(model['normal'])
        model['normal'] = model['normal'] / norm
        
        return inlier_mask, model
    
    return inlier_mask


def height_threshold_ground_detection(points, height_thresh=0.3, percentile=1.0):
    """
    Detect ground points using a simple height threshold approach.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points (x, y, z)
        height_thresh (float): Maximum height from the lowest point to be considered ground
        percentile (float): Percentile of lowest points to determine ground reference (0-100)
        
    Returns:
        numpy.ndarray: Boolean mask of ground points
    """
    # Find the height reference (lowest point or percentile)
    if percentile <= 0:
        height_ref = np.min(points[:, 2])
    else:
        height_ref = np.percentile(points[:, 2], percentile)
    
    # Create mask for points below height threshold
    ground_mask = points[:, 2] <= (height_ref + height_thresh)
    
    return ground_mask


def adaptive_height_threshold(points, grid_size=1.0, height_thresh=0.2, min_points_per_cell=5):
    """
    Detect ground using an adaptive height thresholding approach.
    Creates a grid in the x-y plane and applies height thresholding in each cell.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points (x, y, z)
        grid_size (float): Size of grid cells in meters
        height_thresh (float): Maximum height from the lowest point in a cell to be considered ground
        min_points_per_cell (int): Minimum number of points for a valid cell
        
    Returns:
        numpy.ndarray: Boolean mask of ground points
    """
    # Find min and max for x and y to define grid boundaries
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # Calculate grid dimensions
    x_cells = int(np.ceil((max_x - min_x) / grid_size))
    y_cells = int(np.ceil((max_y - min_y) / grid_size))
    
    # Initialize ground mask with all False
    ground_mask = np.zeros(points.shape[0], dtype=bool)
    
    # Process each grid cell
    for i in range(x_cells):
        for j in range(y_cells):
            # Calculate cell boundaries
            cell_min_x = min_x + i * grid_size
            cell_max_x = min_x + (i + 1) * grid_size
            cell_min_y = min_y + j * grid_size
            cell_max_y = min_y + (j + 1) * grid_size
            
            # Find points in the cell
            cell_mask = (
                (points[:, 0] >= cell_min_x) & (points[:, 0] < cell_max_x) &
                (points[:, 1] >= cell_min_y) & (points[:, 1] < cell_max_y)
            )
            
            cell_points_indices = np.where(cell_mask)[0]
            
            # If not enough points in the cell, skip
            if len(cell_points_indices) < min_points_per_cell:
                continue
            
            # Find the lowest point in the cell
            cell_heights = points[cell_points_indices, 2]
            min_height = np.min(cell_heights)
            
            # Mark points as ground if they're within the height threshold
            ground_indices = cell_points_indices[
                cell_heights <= (min_height + height_thresh)
            ]
            
            ground_mask[ground_indices] = True
    
    return ground_mask


def remove_ground_points(points, ground_mask):
    """
    Remove ground points from the point cloud.
    
    Args:
        points (numpy.ndarray): [N, 3] or [N, 4] array of point cloud points
        ground_mask (numpy.ndarray): Boolean mask of ground points
        
    Returns:
        numpy.ndarray: Point cloud with ground points removed
    """
    # Invert the ground mask to get non-ground points
    non_ground_mask = ~ground_mask
    
    # Return non-ground points
    return points[non_ground_mask]


def extract_ground_points(points, ground_mask):
    """
    Extract only ground points from the point cloud.
    
    Args:
        points (numpy.ndarray): [N, 3] or [N, 4] array of point cloud points
        ground_mask (numpy.ndarray): Boolean mask of ground points
        
    Returns:
        numpy.ndarray: Only ground points
    """
    return points[ground_mask]


def estimate_ground_plane(points, method='ransac', **kwargs):
    """
    Estimate the ground plane parameters from point cloud.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points
        method (str): Method to use - 'ransac' (default) or 'pca'
        **kwargs: Additional parameters to pass to the ground detection method
        
    Returns:
        dict: Ground plane model parameters
        numpy.ndarray: Boolean mask of ground points
    """
    if method == 'ransac':
        # Get ground points and model using RANSAC
        ground_mask, model = ransac_ground_detection(
            points, 
            return_model=True, 
            **kwargs
        )
        return model, ground_mask
        
    elif method == 'pca':
        # Use PCA to find the ground plane
        # First identify potential ground points using height-based method
        height_thresh = kwargs.get('height_thresh', 0.3)
        ground_mask = height_threshold_ground_detection(
            points, 
            height_thresh=height_thresh
        )
        
        # Extract ground points
        ground_points = points[ground_mask]
        
        # Center the points
        centroid = np.mean(ground_points, axis=0)
        centered_points = ground_points - centroid
        
        # Compute covariance matrix
        cov = np.cov(centered_points.T)
        
        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # The smallest eigenvalue's eigenvector is the normal to the plane
        normal = eigenvectors[:, 0]
        
        # Ensure normal points upward (positive z)
        if normal[2] < 0:
            normal = -normal
        
        # Calculate d in ax + by + cz + d = 0
        d = -np.dot(normal, centroid)
        
        model = {
            'normal': normal,
            'intercept': d,
            'centroid': centroid
        }
        
        return model, ground_mask
    
    else:
        raise ValueError(f"Unknown method: {method}")


def fit_ground_to_open3d_plane(points):
    """
    Fit a ground plane using Open3D's plane segmentation.
    
    Args:
        points (numpy.ndarray): [N, 3] array of point cloud points
        
    Returns:
        tuple: (a, b, c, d) coefficients of plane equation ax + by + cz + d = 0
        numpy.ndarray: Boolean mask of inlier points
    """
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Segment plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.2,
        ransac_n=3,
        num_iterations=100
    )
    
    # Create inlier mask
    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inliers] = True
    
    return plane_model, inlier_mask