"""
Multi-Object Tracking Utilities

This module provides functions for tracking multiple objects detected in LiDAR point clouds.
Implementations include Kalman filtering, data association, and track management.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import uuid


class Track:
    """
    Track class for representing a tracked object.
    
    Attributes:
        id (str): Unique identifier for the track
        kalman_filter (KalmanFilter): Kalman filter for state estimation
        class_name (str): Object class (e.g., car, pedestrian)
        age (int): Number of frames the track has existed
        hits (int): Number of successful detections
        time_since_update (int): Frames since last successful detection
        dimensions (numpy.ndarray): 3D dimensions of the object [width, length, height]
        history (list): History of track states
    """
    
    def __init__(self, detection, track_id=None):
        """
        Initialize a new track.
        
        Args:
            detection (dict): Detection information containing center, dimensions, class
            track_id (str): Unique track ID (default: None, will generate UUID)
        """
        self.id = track_id if track_id is not None else str(uuid.uuid4())
        self.class_name = detection.get('class', 'unknown')
        self.dimensions = detection.get('dimensions', np.array([1.0, 1.0, 1.0]))
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.history = []
        
        # Initialize Kalman filter
        self.kalman_filter = self._init_kalman_filter(detection['center'])
        
        # Store first state
        self.history.append(self.get_state())

    def _init_kalman_filter(self, center):
        """
        Initialize the Kalman filter for this track.
        
        Args:
            center (numpy.ndarray): Initial position [x, y, z]
            
        Returns:
            KalmanFilter: Initialized Kalman filter
        """
        # We use a constant velocity model
        # State: [x, y, z, vx, vy, vz]
        kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # Initial state
        kf.x = np.array([center[0], center[1], center[2], 0, 0, 0])
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1],  # z = z + vz
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1]   # vz = vz
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(3) * 0.01
        
        # Process uncertainty
        kf.Q = np.eye(6) * 0.1
        kf.Q[3:, 3:] = np.eye(3) * 1.0  # Higher uncertainty for velocity
        
        # Initial state uncertainty
        kf.P = np.eye(6) * 1.0
        kf.P[3:, 3:] = np.eye(3) * 10.0  # Higher uncertainty for velocity
        
        return kf
    
    def predict(self):
        """
        Predict the state of the track for the next frame.
        
        Returns:
            numpy.ndarray: Predicted position [x, y, z]
        """
        self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.get_position()
    
    def update(self, detection):
        """
        Update the track with a new detection.
        
        Args:
            detection (dict): Detection information containing center
            
        Returns:
            numpy.ndarray: Updated position [x, y, z]
        """
        # Update Kalman filter with new measurement
        self.kalman_filter.update(detection['center'])
        
        # Update track properties
        self.hits += 1
        self.time_since_update = 0
        
        # Adaptive update of dimensions
        alpha = 0.7  # Weight for existing dimensions
        self.dimensions = alpha * self.dimensions + (1 - alpha) * detection.get('dimensions', self.dimensions)
        
        # Add current state to history
        self.history.append(self.get_state())
        
        return self.get_position()
    
    def get_position(self):
        """
        Get the current position of the track.
        
        Returns:
            numpy.ndarray: Current position [x, y, z]
        """
        return self.kalman_filter.x[:3]
    
    def get_velocity(self):
        """
        Get the current velocity of the track.
        
        Returns:
            numpy.ndarray: Current velocity [vx, vy, vz]
        """
        return self.kalman_filter.x[3:6]
    
    def get_state(self):
        """
        Get the full state of the track.
        
        Returns:
            dict: Dictionary containing position, velocity, dimensions, etc.
        """
        return {
            'id': self.id,
            'position': self.get_position(),
            'velocity': self.get_velocity(),
            'dimensions': self.dimensions,
            'class': self.class_name,
            'age': self.age,
            'time_since_update': self.time_since_update
        }
    
    def get_predicted_bbox(self):
        """
        Get the predicted 3D bounding box of the track.
        
        Returns:
            dict: Bounding box information
        """
        center = self.get_position()
        dimensions = self.dimensions
        
        bbox = {
            'center': center,
            'dimensions': dimensions,
            'min_bound': center - dimensions / 2,
            'max_bound': center + dimensions / 2
        }
        
        return bbox


class MultiObjectTracker:
    """
    Multiple object tracker using Kalman filter and Hungarian algorithm for data association.
    """
    
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        """
        Initialize the multi-object tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep a track alive without matching
            min_hits (int): Minimum number of hits needed to consider a track confirmed
            iou_threshold (float): IoU threshold for considering a detection as a match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update the tracker with new detections.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            list: List of active track states
        """
        self.frame_count += 1
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        
        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            self._create_track(detections[idx])
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return active tracks
        return self.get_active_tracks()
    
    def _associate_detections_to_tracks(self, detections):
        """
        Associate detections to existing tracks using IoU and Hungarian algorithm.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            tuple: (matches, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute IoU between tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            track_bbox = track.get_predicted_bbox()
            
            for d, detection in enumerate(detections):
                # Create detection bbox
                det_center = detection['center']
                det_dim = detection.get('dimensions', np.array([1.0, 1.0, 1.0]))
                det_bbox = {
                    'center': det_center,
                    'dimensions': det_dim,
                    'min_bound': det_center - det_dim / 2,
                    'max_bound': det_center + det_dim / 2
                }
                
                iou_matrix[t, d] = self._iou_3d(track_bbox, det_bbox)
        
        # Apply Hungarian algorithm to find optimal assignments
        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter matches with low IoU
        matches = []
        for t, d in zip(track_indices, detection_indices):
            if iou_matrix[t, d] >= self.iou_threshold:
                matches.append((t, d))
            else:
                # If IoU is too low, treat as unmatched
                track_indices = np.append(track_indices, t)
                detection_indices = np.append(detection_indices, d)
        
        # Find unmatched tracks and detections
        all_tracks = set(range(len(self.tracks)))
        all_detections = set(range(len(detections)))
        matched_tracks = set([m[0] for m in matches])
        matched_detections = set([m[1] for m in matches])
        
        unmatched_tracks = list(all_tracks - matched_tracks)
        unmatched_detections = list(all_detections - matched_detections)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _iou_3d(self, bbox1, bbox2):
        """
        Compute 3D IoU between two bounding boxes.
        
        Args:
            bbox1 (dict): First bounding box with min_bound and max_bound
            bbox2 (dict): Second bounding box with min_bound and max_bound
            
        Returns:
            float: IoU score
        """
        # Compute intersection boundaries
        min_bound = np.maximum(bbox1['min_bound'], bbox2['min_bound'])
        max_bound = np.minimum(bbox1['max_bound'], bbox2['max_bound'])
        
        # Check if boxes overlap
        if np.any(min_bound > max_bound):
            return 0.0
        
        # Compute volumes
        intersection_volume = np.prod(max_bound - min_bound)
        volume1 = np.prod(bbox1['max_bound'] - bbox1['min_bound'])
        volume2 = np.prod(bbox2['max_bound'] - bbox2['min_bound'])
        
        # Compute IoU
        union_volume = volume1 + volume2 - intersection_volume
        iou = intersection_volume / union_volume
        
        return iou
    
    def _create_track(self, detection):
        """
        Create a new track from a detection.
        
        Args:
            detection (dict): Detection information
        """
        new_track = Track(detection)
        self.tracks.append(new_track)
    
    def get_active_tracks(self):
        """
        Get all active tracks (confirmed tracks).
        
        Returns:
            list: List of active track states
        """
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update <= 1:
                active_tracks.append(track.get_state())
        
        return active_tracks


def kalman_filter_update(track_states, detections, dt=0.1):
    """
    Update track states using Kalman filtering.
    Simplified function for demonstration purposes.
    
    Args:
        track_states (list): List of track states
        detections (list): List of detections
        dt (float): Time step
        
    Returns:
        list: Updated track states
    """
    # Create or update tracker
    tracker = MultiObjectTracker()
    
    # Convert track states to tracks if not first frame
    if len(track_states) > 0 and not hasattr(kalman_filter_update, 'tracker'):
        kalman_filter_update.tracker = tracker
        
        for state in track_states:
            detection = {
                'center': state['position'],
                'dimensions': state['dimensions'],
                'class': state['class']
            }
            kalman_filter_update.tracker._create_track(detection)
    
    # Update with new detections
    if hasattr(kalman_filter_update, 'tracker'):
        tracker = kalman_filter_update.tracker
    
    updated_states = tracker.update(detections)
    return updated_states


def associate_detections_to_tracks(detections, tracks, iou_threshold=0.3, use_class=True):
    """
    Associate detections to existing tracks using IoU and Hungarian algorithm.
    Standalone function (not requiring MultiObjectTracker class).
    
    Args:
        detections (list): List of detection dictionaries
        tracks (list): List of track dictionaries
        iou_threshold (float): IoU threshold for considering a detection as a match
        use_class (bool): Whether to consider class information in association
        
    Returns:
        tuple: (matches, unmatched_detections, unmatched_tracks)
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    
    # Compute cost matrix based on distance
    cost_matrix = np.zeros((len(tracks), len(detections)))
    
    for t, track in enumerate(tracks):
        for d, detection in enumerate(detections):
            # Skip if classes don't match (if using class info)
            if use_class and track.get('class', '') != detection.get('class', ''):
                cost_matrix[t, d] = float('inf')
                continue
                
            # Compute Euclidean distance between centers
            track_pos = track['position'] if 'position' in track else track['center']
            det_pos = detection['center']
            
            distance = np.linalg.norm(track_pos - det_pos)
            
            # Distances above a threshold are unlikely matches
            if distance > 5.0:  # 5 meters threshold
                cost_matrix[t, d] = float('inf')
            else:
                cost_matrix[t, d] = distance
    
    # Solve assignment problem
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    
    # Filter matches with high cost
    matches = []
    for t, d in zip(track_indices, detection_indices):
        if cost_matrix[t, d] != float('inf'):
            matches.append((t, d))
    
    # Find unmatched tracks and detections
    all_tracks = set(range(len(tracks)))
    all_detections = set(range(len(detections)))
    matched_tracks = set([m[0] for m in matches])
    matched_detections = set([m[1] for m in matches])
    
    unmatched_tracks = list(all_tracks - matched_tracks)
    unmatched_detections = list(all_detections - matched_detections)
    
    return matches, unmatched_detections, unmatched_tracks


def predict_new_locations(tracks, dt=0.1):
    """
    Predict new locations of tracks using constant velocity model.
    Standalone function (not requiring MultiObjectTracker class).
    
    Args:
        tracks (list): List of track dictionaries
        dt (float): Time step
        
    Returns:
        list: Updated track states with predicted locations
    """
    updated_tracks = []
    
    for track in tracks:
        # Extract position and velocity
        position = track['position']
        velocity = track.get('velocity', np.zeros(3))
        
        # Predict new position
        new_position = position + velocity * dt
        
        # Update track
        new_track = track.copy()
        new_track['position'] = new_position
        new_track['predicted'] = True
        
        updated_tracks.append(new_track)
    
    return updated_tracks


def update_tracks(tracks, detections, matched_pairs):
    """
    Update tracks with matched detections.
    Standalone function (not requiring MultiObjectTracker class).
    
    Args:
        tracks (list): List of track dictionaries
        detections (list): List of detection dictionaries
        matched_pairs (list): List of (track_idx, detection_idx) pairs
        
    Returns:
        list: Updated track states
    """
    updated_tracks = tracks.copy()
    
    for track_idx, detection_idx in matched_pairs:
        track = updated_tracks[track_idx]
        detection = detections[detection_idx]
        
        # Simple update (could be more sophisticated with Kalman filter)
        alpha = 0.7  # Smoothing factor
        
        track['position'] = alpha * track['position'] + (1 - alpha) * detection['center']
        
        # Update dimensions if available
        if 'dimensions' in detection:
            if 'dimensions' in track:
                track['dimensions'] = alpha * track['dimensions'] + (1 - alpha) * detection['dimensions']
            else:
                track['dimensions'] = detection['dimensions']
        
        # Reset age counter
        track['age'] = 0
        track['hits'] = track.get('hits', 0) + 1
        
        # Store velocity (if possible)
        if 'prev_position' in track:
            track['velocity'] = (track['position'] - track['prev_position']) / 0.1  # Assuming 10Hz
        
        track['prev_position'] = track['position'].copy()
    
    return updated_tracks