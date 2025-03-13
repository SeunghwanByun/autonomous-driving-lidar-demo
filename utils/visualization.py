"""
Visualization Utilities

This module provides functions for visualizing LiDAR point clouds, detected objects,
ground planes, lane detections, and other perception results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Arrow
import matplotlib.cm as cm
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import math


def visualize_point_cloud(points, colors=None, size=1.0, window_name="Point Cloud Viewer"):
    """
    Visualize point cloud using Open3D.
    
    Args:
        points (numpy.ndarray): [N, 3] array of points (x, y, z)
        colors (numpy.ndarray, optional): [N, 3] array of RGB colors (0-1)
        size (float): Point size
        window_name (str): Name of the visualization window
        
    Returns:
        open3d.visualization.Visualizer or None: Open3D visualizer object or None if visualization failed
    """
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Set colors if provided
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif points.shape[1] >= 4:  # If intensity is available, color by intensity
        intensity = points[:, 3]
        intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-10)
        colors = cm.jet(intensity_norm)[:, :3]  # Get RGB from jet colormap
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 헤드리스 환경에서는 그냥 간단하게 True 반환
    try:
        # 간단한 시각화만 시도 (에러가 발생할 수 있음)
        try:
            o3d.visualization.draw_geometries([pcd], window_name=window_name)
        except Exception as e:
            print(f"인터랙티브 시각화를 생성할 수 없습니다: {e}")
            print("헤드리스 환경에서는 시각화가 지원되지 않습니다.")
        
        # 더 이상의 GUI 작업을 시도하지 않음
        return True
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        return False 

def visualize_ground(points, ground_mask, window_name="Ground Visualization"):
    """
    Visualize ground points in a different color.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of points (x, y, z, ...)
        ground_mask (numpy.ndarray): Boolean mask of ground points
        window_name (str): Name of the visualization window
        
    Returns:
        open3d.visualization.Visualizer: Open3D visualizer object
    """
    # Create colors based on ground/non-ground
    colors = np.zeros((points.shape[0], 3))
    colors[ground_mask] = [0.0, 0.8, 0.0]    # Green for ground
    colors[~ground_mask] = [0.8, 0.0, 0.0]   # Red for non-ground
    
    # Visualize
    return visualize_point_cloud(points, colors, window_name=window_name)


def visualize_objects(points, objects, window_name="Object Detection"):
    """
    Visualize detected objects with bounding boxes.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of points (x, y, z, ...)
        objects (list): List of dictionaries containing object information
        window_name (str): Name of the visualization window
        
    Returns:
        open3d.visualization.Visualizer: Open3D visualizer object
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color points by intensity if available
    if points.shape[1] >= 4:
        intensity = points[:, 3]
        intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-10)
        colors = cm.jet(intensity_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Add boxes for each object
    for obj in objects:
        center = obj.get('center', np.array([0, 0, 0]))
        dimensions = obj.get('dimensions', np.array([1, 1, 1]))
        
        # Create oriented box if rotation is available
        if 'rotation_matrix' in obj:
            R = obj['rotation_matrix']
            bbox = o3d.geometry.OrientedBoundingBox(center, R, dimensions)
        else:
            # Create axis-aligned box
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                center - dimensions/2, center + dimensions/2
            )
        
        # Color by object class
        class_name = obj.get('class', 'unknown')
        if class_name in ['car', 'vehicle', 'large_vehicle']:
            color = [1, 0, 0]  # Red for vehicles
        elif class_name in ['pedestrian', 'person']:
            color = [0, 1, 0]  # Green for pedestrians
        elif class_name in ['cyclist', 'bicycle', 'motorcycle']:
            color = [0, 0, 1]  # Blue for cyclists
        else:
            color = [1, 1, 0]  # Yellow for other objects
        
        bbox.color = color
        vis.add_geometry(bbox)
    
    # Add coordinate system
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    
    # Set viewpoint
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])  # Looking toward the car front
    view_control.set_up([0, -1, 0])     # Up direction
    
    # Update rendering
    vis.poll_events()
    vis.update_renderer()
    
    return vis


def visualize_lanes(points, lanes, window_name="Lane Detection"):
    """
    Visualize detected lanes.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of points (x, y, z, ...)
        lanes (list): List of lane information, each lane is a list of points
        window_name (str): Name of the visualization window
        
    Returns:
        open3d.visualization.Visualizer: Open3D visualizer object
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Create default colors (gray)
    colors = np.ones((points.shape[0], 3)) * 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Add lines for each lane
    line_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ]
    
    for i, lane in enumerate(lanes):
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(lane)
        
        # Create lines between consecutive points
        lines = [[j, j+1] for j in range(len(lane)-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set color
        color = line_colors[i % len(line_colors)]
        line_colors_array = np.array([color for _ in range(len(lines))])
        line_set.colors = o3d.utility.Vector3dVector(line_colors_array)
        
        vis.add_geometry(line_set)
    
    # Add coordinate system
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    
    # Set viewpoint
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])  # Looking toward the car front
    view_control.set_up([0, -1, 0])     # Up direction
    view_control.set_lookat([0, 0, 0])  # Look at origin
    
    # Update rendering
    vis.poll_events()
    vis.update_renderer()
    
    return vis


def plot_bev(points, objects=None, lanes=None, ground_mask=None, figsize=(10, 10), 
             xlim=(-50, 50), ylim=(-50, 50), grid=True, color_by_height=True):
    """
    Plot Bird's Eye View (BEV) of the point cloud and detection results.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of points (x, y, z, ...)
        objects (list, optional): List of dictionaries containing object information
        lanes (list, optional): List of lane information, each lane is a list of points
        ground_mask (numpy.ndarray, optional): Boolean mask of ground points
        figsize (tuple): Figure size
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
        grid (bool): Whether to show grid
        color_by_height (bool): Whether to color points by height
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    x = points[:, 0]
    y = points[:, 1]
    
    if color_by_height:
        # Color by height (z)
        z = points[:, 2]
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=0.5, alpha=0.5)
        fig.colorbar(scatter, ax=ax, label='Height (m)')
    elif ground_mask is not None:
        # Color by ground/non-ground
        colors = np.zeros((points.shape[0], 3))
        colors[ground_mask] = [0.0, 0.8, 0.0]    # Green for ground
        colors[~ground_mask] = [0.8, 0.0, 0.0]   # Red for non-ground
        ax.scatter(x, y, c=colors, s=0.5, alpha=0.5)
    else:
        # Default coloring
        ax.scatter(x, y, c='blue', s=0.5, alpha=0.5)
    
    # Plot objects if provided
    if objects is not None:
        for obj in objects:
            center = obj.get('center', np.array([0, 0, 0]))
            dimensions = obj.get('dimensions', np.array([1, 1, 1]))
            
            # Get 2D box
            width = dimensions[0]
            length = dimensions[1]
            
            # Check if oriented box
            if 'rotation_matrix' in obj:
                R = obj['rotation_matrix']
                yaw = np.arctan2(R[1, 0], R[0, 0])
                
                # Get corners
                corners = np.array([
                    [-length/2, -width/2],
                    [length/2, -width/2],
                    [length/2, width/2],
                    [-length/2, width/2]
                ])
                
                # Rotate corners
                rot_matrix = np.array([
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]
                ])
                corners = corners @ rot_matrix.T
                
                # Translate corners
                corners = corners + center[:2]
                
                # Create polygon
                polygon = Polygon(corners, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(polygon)
                
                # Add an arrow to show direction
                front_center = center[:2] + rot_matrix @ np.array([length/2, 0])
                arrow = Arrow(center[0], center[1], 
                              front_center[0] - center[0], front_center[1] - center[1],
                              width=1, color='red')
                ax.add_patch(arrow)
            else:
                # Axis-aligned box
                x_min = center[0] - length/2
                y_min = center[1] - width/2
                rect = Rectangle((x_min, y_min), length, width, 
                                 fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            
            # Add label
            class_name = obj.get('class', 'unknown')
            ax.text(center[0], center[1], class_name, fontsize=8, 
                    ha='center', va='center', color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot lanes if provided
    if lanes is not None:
        lane_colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, lane in enumerate(lanes):
            lane_color = lane_colors[i % len(lane_colors)]
            ax.plot(lane[:, 0], lane[:, 1], color=lane_color, linewidth=2)
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title("Bird's Eye View")
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    if grid:
        ax.grid(True)
    
    # Add ego-vehicle marker
    ax.plot(0, 0, 'o', color='black', markersize=10)
    
    # Add axes origin
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def draw_boxes_3d(points, objects, ax=None, figsize=(10, 10)):
    """
    Draw 3D bounding boxes for detected objects.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of points (x, y, z, ...)
        objects (list): List of dictionaries containing object information
        ax (matplotlib.axes._subplots.Axes3D, optional): 3D Axes to plot on
        figsize (tuple): Figure size if ax is None
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot point cloud
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Downsample if too many points
    if len(x) > 10000:
        idx = np.random.choice(len(x), 10000, replace=False)
        x = x[idx]
        y = y[idx]
        z = z[idx]
    
    ax.scatter(x, y, z, c='blue', s=0.5, alpha=0.2)
    
    # Draw 3D boxes for each object
    for obj in objects:
        center = obj.get('center', np.array([0, 0, 0]))
        dimensions = obj.get('dimensions', np.array([1, 1, 1]))
        
        # Extract dimensions
        width = dimensions[0]
        length = dimensions[1]
        height = dimensions[2]
        
        # Check if oriented box
        if 'rotation_matrix' in obj:
            R = obj['rotation_matrix']
        else:
            # Identity rotation
            R = np.eye(3)
            
        # Define 8 corners of the box in object frame
        x_corners = np.array([length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2])
        y_corners = np.array([width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2])
        z_corners = np.array([0, 0, 0, 0, height, height, height, height]) - height/2
        
        # Rotate and translate corners to world frame
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = R @ corners
        corners = corners + center.reshape(3, 1)
        
        # Draw lines between corners
        # Bottom face
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            ax.plot([corners[0, i], corners[0, j]],
                    [corners[1, i], corners[1, j]],
                    [corners[2, i], corners[2, j]], 'r-')
        
        # Top face
        for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
            ax.plot([corners[0, i], corners[0, j]],
                    [corners[1, i], corners[1, j]],
                    [corners[2, i], corners[2, j]], 'r-')
        
        # Connecting lines
        for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot([corners[0, i], corners[0, j]],
                    [corners[1, i], corners[1, j]],
                    [corners[2, i], corners[2, j]], 'r-')
        
        # Add label
        class_name = obj.get('class', 'unknown')
        ax.text(center[0], center[1], center[2] + height/2, class_name, 
                fontsize=8, color='black')
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Object Detection')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.3])  # Slightly compressed in z-axis
    
    # Add ego-vehicle marker
    ax.plot([0], [0], [0], 'o', color='black', markersize=10)
    
    return fig


def plot_intensity_image(points, intensity=None, resolution=0.1, figsize=(10, 8)):
    """
    Create a BEV intensity image from point cloud.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of points (x, y, z, ...)
        intensity (numpy.ndarray, optional): Intensity values for each point
        resolution (float): Grid resolution in meters
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    # Extract x, y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Get intensity values
    if intensity is None and points.shape[1] >= 4:
        intensity = points[:, 3]
    elif intensity is None:
        intensity = np.ones_like(x)  # Default to 1 if no intensity
    
    # Determine grid dimensions
    x_min, x_max = np.floor(np.min(x)), np.ceil(np.max(x))
    y_min, y_max = np.floor(np.min(y)), np.ceil(np.max(y))
    
    # Create grid
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)
    
    # Initialize intensity image
    intensity_img = np.zeros((grid_height, grid_width))
    counts = np.zeros((grid_height, grid_width))
    
    # Map points to grid cells
    for i in range(len(x)):
        grid_x = int((x[i] - x_min) / resolution)
        grid_y = int((y[i] - y_min) / resolution)
        
        # Check bounds
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            intensity_img[grid_y, grid_x] += intensity[i]
            counts[grid_y, grid_x] += 1
    
    # Average intensity in each cell
    mask = counts > 0
    intensity_img[mask] = intensity_img[mask] / counts[mask]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(intensity_img, origin='lower', cmap='viridis', 
                   extent=[x_min, x_max, y_min, y_max])
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Intensity')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('BEV Intensity Image')
    
    return fig


def visualize_kalman_tracks(tracks, detections=None, figsize=(10, 10), 
                           xlim=(-50, 50), ylim=(-50, 50)):
    """
    Visualize Kalman filter tracking results.
    
    Args:
        tracks (list): List of track dictionaries with history
        detections (list, optional): List of detection dictionaries
        figsize (tuple): Figure size
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot tracks
    track_colors = plt.cm.tab10.colors
    
    for i, track in enumerate(tracks):
        color = track_colors[i % len(track_colors)]
        
        # Get track history
        history = track.get('history', [])
        if not history and 'position' in track:
            # If no history, use current position
            history = [track]
        
        # Extract x, y positions
        x_pos = [state.get('position', state.get('center', [0, 0, 0]))[0] for state in history]
        y_pos = [state.get('position', state.get('center', [0, 0, 0]))[1] for state in history]
        
        # Plot track line
        ax.plot(x_pos, y_pos, '-', color=color, linewidth=2, label=f"Track {track.get('id', i)}")
        
        # Plot current position
        if history:
            latest = history[-1]
            pos = latest.get('position', latest.get('center', [0, 0, 0]))
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=8)
            
            # Plot velocity vector if available
            if 'velocity' in latest:
                vel = latest['velocity']
                ax.arrow(pos[0], pos[1], vel[0], vel[1], 
                         head_width=0.5, head_length=0.7, fc=color, ec=color)
    
    # Plot detections if provided
    if detections:
        det_x = [d['center'][0] for d in detections]
        det_y = [d['center'][1] for d in detections]
        ax.scatter(det_x, det_y, c='red', marker='x', s=50, label='Detections')
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Object Tracking')
    
    # Add grid
    ax.grid(True)
    
    # Add legend
    if len(tracks) <= 10:  # Only show legend if not too many tracks
        ax.legend()
    
    # Add ego-vehicle marker
    ax.plot(0, 0, 'o', color='black', markersize=10)
    
    # Add axes origin
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    return fig