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
    
    try:
        with open(calib_filepath, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # 빈 줄이나 주석 건너뛰기
                    continue
                    
                if ':' in line:
                    key, value = line.split(':', 1)  # 최대 1번만 분할
                    key = key.strip()
                    values = [float(x) for x in value.strip().split()]
                    
                    if key == 'R0_rect':
                        calibration[key] = np.reshape(values, (3, 3))
                    elif key in ['Tr_velo_to_cam', 'Tr_imu_to_velo']:
                        # Convert 3x4 matrix to 4x4 by adding [0,0,0,1] as the last row
                        matrix = np.reshape(values, (3, 4))
                        matrix_4x4 = np.vstack((matrix, [0, 0, 0, 1]))
                        calibration[key] = matrix_4x4
                    elif key in ['P0', 'P1', 'P2', 'P3']:
                        calibration[key] = np.reshape(values, (3, 4))
                else:
                    # ':' 구분자가 없는 경우, 다른 형식 처리
                    parts = line.split()
                    if len(parts) > 1:  # 최소 키와 값이 있어야 함
                        key = parts[0]
                        values = [float(x) for x in parts[1:]]
                        # 적절한 형태로 변환 (여기서는 단순 벡터로)
                        calibration[key] = np.array(values)
    except Exception as e:
        print(f"캘리브레이션 파일 로드 중 오류: {e}")
        # 예시 캘리브레이션 데이터로 대체
        calibration = {
            'P2': np.array([
                [721.5377, 0.0, 609.5593, 44.85728],
                [0.0, 721.5377, 172.854, 0.2163791],
                [0.0, 0.0, 1.0, 0.002745884]
            ]),
            'R0_rect': np.eye(3),
            'Tr_velo_to_cam': np.array([
                [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                [0.0, 0.0, 0.0, 1.0]
            ])
        }
    
    # 필요한 변환 행렬이 없는 경우, 기본값 제공
    if 'R0_rect' not in calibration:
        calibration['R0_rect'] = np.eye(3)
    if 'Tr_velo_to_cam' not in calibration:
        calibration['Tr_velo_to_cam'] = np.eye(4)
    if 'P2' not in calibration:
        calibration['P2'] = np.array([
            [721.5377, 0.0, 609.5593, 44.85728],
            [0.0, 721.5377, 172.854, 0.2163791],
            [0.0, 0.0, 1.0, 0.002745884]
        ])
    
    # 필요한 추가 변환 행렬 계산
    if 'R0_rect' in calibration and 'Tr_velo_to_cam' in calibration:
        rect_mat = np.eye(4)
        rect_mat[:3, :3] = calibration['R0_rect']
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