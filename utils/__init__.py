"""
LiDAR Perception Utilities Package

This package provides utility functions for LiDAR perception tasks in autonomous driving,
including visualization, calibration, ground detection, object detection, and tracking.
"""

# Import utility modules to make them available when importing the utils package
from .visualization import (
    visualize_point_cloud, 
    visualize_ground, 
    visualize_objects,
    visualize_lanes,
    plot_bev,
    draw_boxes_3d
)

from .calibration import (
    load_calib_data,
    lidar_to_camera,
    camera_to_image,
    lidar_to_image
)

from .ground_detection import (
    ransac_ground_detection,
    height_threshold_ground_detection,
    remove_ground_points,
    estimate_ground_plane
)

from .object_detection import (
    euclidean_clustering,
    detect_objects,
    compute_bounding_boxes,
    classify_objects
)

from .tracking import (
    associate_detections_to_tracks,
    update_tracks,
    predict_new_locations,
    kalman_filter_update
)

# Package version
__version__ = '0.1.0'

# Package information
__author__ = 'Seunghwan'
__email__ = 'rapping44@naver.com'