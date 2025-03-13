"""
LiDAR-Camera Calibration Utilities

This module provides functions for handling calibration data and coordinate
transformations between LiDAR, camera, and image coordinate systems.
Primarily designed for use with KITTI dataset format.
"""

import numpy as np
import os


def load_calib_data(calib_filepath):
    """
    Load and parse calibration data from KITTI calibration file.
    
    Args:
        calib_filepath (str): Path to the calibration file
        
    Returns:
        dict: Dictionary containing calibration matrices
    """
    calibration = {}
    
    with open(calib_filepath, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            key, value = line.split(':', 1)
            key = key.strip()
            value = np.array([float(x) for x in value.split()])
            
            if key == 'R0_rect':
                calibration[key] = np.reshape(value, (3, 3))
            elif key in ['Tr_velo_to_cam', 'Tr_imu_to_velo']:
                # Convert 3x4 matrix to 4x4 by adding [0,0,0,1] as the last row
                calibration[key] = np.vstack((np.reshape(value, (3, 4)), [0, 0, 0, 1]))
            elif key in ['P0', 'P1', 'P2', 'P3']:
                calibration[key] = np.reshape(value, (3, 4))
                
    # Compute additional useful matrices
    if 'R0_rect' in calibration and 'Tr_velo_to_cam' in calibration:
        # Rotation matrix for rectification
        rect_mat = np.eye(4)
        rect_mat[:3, :3] = calibration['R0_rect']
        
        # Complete transformation from LiDAR to rectified camera
        calibration['velo_to_cam_rect'] = rect_mat @ calibration['Tr_velo_to_cam']
        
    return calibration


def lidar_to_camera(points, calib_data):
    """
    Transform points from LiDAR coordinate system to camera coordinate system.
    
    Args:
        points (numpy.ndarray): [N, 3] or [N, 4] array of points in LiDAR coordinate system
        calib_data (dict): Calibration data dictionary from load_calib_data
        
    Returns:
        numpy.ndarray: [N, 3] array of points in camera coordinate system
    """
    # Ensure points is homogeneous 
    if points.shape[1] == 3:
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    else:
        points_hom = points.copy()

    # Transform from LiDAR to camera coordinate system
    points_cam = points_hom @ calib_data['Tr_velo_to_cam'].T
    
    # Return only x, y, z (discard homogeneous component)
    return points_cam[:, :3]


def camera_to_image(points, calib_data, cam_idx=2):
    """
    Project points from camera coordinate system to image plane.
    
    Args:
        points (numpy.ndarray): [N, 3] array of points in camera coordinate system
        calib_data (dict): Calibration data dictionary from load_calib_data
        cam_idx (int): Camera index (0-3, default is 2 for left color camera in KITTI)
        
    Returns:
        numpy.ndarray: [N, 2] array of points in image coordinates (u, v)
    """
    # Get camera projection matrix
    proj_mat = calib_data[f'P{cam_idx}']
    
    # Convert to homogeneous coordinates
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Project to image plane
    pts_img_hom = points_hom @ proj_mat.T
    
    # Normalize by the third coordinate
    pts_img = pts_img_hom[:, :2] / pts_img_hom[:, 2:3]
    
    return pts_img


def lidar_to_image(points, calib_data, cam_idx=2):
    """
    Transform points from LiDAR coordinate system directly to image plane.
    
    Args:
        points (numpy.ndarray): [N, 3] or [N, 4] array of points in LiDAR coordinate system
        calib_data (dict): Calibration data dictionary from load_calib_data
        cam_idx (int): Camera index (0-3, default is 2 for left color camera in KITTI)
        
    Returns:
        numpy.ndarray: [N, 2] array of points in image coordinates (u, v)
        numpy.ndarray: [N, 3] array of points in camera coordinate system (for depth information)
    """
    # First transform to camera coordinate system
    points_cam = lidar_to_camera(points, calib_data)
    
    # Apply rectification if available
    if 'R0_rect' in calib_data:
        rect_mat = calib_data['R0_rect']
        points_cam = points_cam @ rect_mat.T
    
    # Then project to image plane
    points_img = camera_to_image(points_cam, calib_data, cam_idx)
    
    return points_img, points_cam


def get_fov_mask(points_img, points_cam, img_shape):
    """
    Create a mask for points that are within the camera's field of view.
    
    Args:
        points_img (numpy.ndarray): [N, 2] array of points in image coordinates (u, v)
        points_cam (numpy.ndarray): [N, 3] array of points in camera coordinate system
        img_shape (tuple): Image shape as (height, width)
        
    Returns:
        numpy.ndarray: Boolean mask of points that are within the FOV
    """
    height, width = img_shape
    
    # Points should be in front of the camera (positive Z)
    mask_z = points_cam[:, 2] > 0
    
    # Points should be within image boundaries
    mask_h = (points_img[:, 1] >= 0) & (points_img[:, 1] < height)
    mask_w = (points_img[:, 0] >= 0) & (points_img[:, 0] < width)
    
    # Combine all masks
    mask = mask_z & mask_h & mask_w
    
    return mask


def load_transforms_from_file(transform_filepath):
    """
    Load transformation matrices from a custom file format.
    
    Args:
        transform_filepath (str): Path to the transform file
        
    Returns:
        dict: Dictionary containing transformation matrices
    """
    transforms = {}
    
    if os.path.exists(transform_filepath):
        with open(transform_filepath, 'r') as f:
            lines = f.readlines()
            
            current_matrix = None
            matrix_data = []
            
            for line in lines:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                    
                if ':' in line:  # Matrix name line
                    if current_matrix and matrix_data:
                        transforms[current_matrix] = np.array(matrix_data)
                        matrix_data = []
                    
                    current_matrix = line.split(':')[0].strip()
                else:  # Matrix data line
                    matrix_data.append([float(x) for x in line.split()])
            
            # Don't forget the last matrix
            if current_matrix and matrix_data:
                transforms[current_matrix] = np.array(matrix_data)
    
    return transforms


def save_calibration_to_file(calib_data, output_filepath):
    """
    Save calibration data to a file in KITTI format.
    
    Args:
        calib_data (dict): Calibration data dictionary
        output_filepath (str): Path to save the calibration file
    """
    with open(output_filepath, 'w') as f:
        for key, value in calib_data.items():
            if key in ['R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo', 'P0', 'P1', 'P2', 'P3']:
                if value.shape == (3, 3):  # R0_rect
                    value_str = ' '.join([str(v) for v in value.flatten()])
                    f.write(f"{key}: {value_str}\n")
                elif value.shape == (4, 4):  # Tr_velo_to_cam, Tr_imu_to_velo
                    value_str = ' '.join([str(v) for v in value[:3].flatten()])
                    f.write(f"{key}: {value_str}\n")
                elif value.shape == (3, 4):  # P0, P1, P2, P3
                    value_str = ' '.join([str(v) for v in value.flatten()])
                    f.write(f"{key}: {value_str}\n")