"""
Deep Learning Utilities for Point Cloud Processing

This module provides functions and classes for applying deep learning models to point cloud data.
Implementations include PointNet-based segmentation and classification models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import urllib.request
import warnings


class TNet(nn.Module):
    """
    T-Net for input and feature transformation in PointNet architecture
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        iden = torch.eye(self.k, dtype=torch.float32).view(1, self.k*self.k).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
            
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNetFeatures(nn.Module):
    """
    PointNet feature extractor
    """
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetFeatures, self).__init__()
        self.stn = TNet(k=channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = TNet(k=64)
            
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        
        # Input transformation
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # MLP on each point
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transformation if enabled
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
            
        pointfeat = x
        
        # Continue with MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        
        # Max pooling to get global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetClassification(nn.Module):
    """
    PointNet model for point cloud classification
    """
    def __init__(self, num_classes=40, feature_transform=False, channel=3):
        super(PointNetClassification, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeatures(global_feat=True, feature_transform=feature_transform, channel=channel)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetSegmentation(nn.Module):
    """
    PointNet model for semantic segmentation of point clouds
    """
    def __init__(self, num_classes=50, feature_transform=False, channel=3):
        super(PointNetSegmentation, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeatures(global_feat=False, feature_transform=feature_transform, channel=channel)
        
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1)
        return F.log_softmax(x, dim=-1), trans, trans_feat


def download_pretrained_model(model_url, model_path):
    """
    Download a pretrained model from a URL.
    
    Args:
        model_url (str): URL to download the model from
        model_path (str): Path to save the model to
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        if not os.path.exists(model_path):
            print(f"Downloading pretrained model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Download complete!")
        return True
    except Exception as e:
        warnings.warn(f"Could not download pretrained model: {str(e)}")
        return False


def load_pretrained_classification_model(num_classes=40, model_path=None):
    """
    Load a pretrained PointNet classification model.
    
    Args:
        num_classes (int): Number of classes for the model
        model_path (str): Path to the pretrained model weights
        
    Returns:
        PointNetClassification: The loaded model
    """
    model = PointNetClassification(num_classes=num_classes, feature_transform=True)
    
    if model_path is None:
        model_path = os.path.join("pretrained_models", "pointnet_classification.pth")
        
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            warnings.warn(f"Could not load pretrained model: {str(e)}")
    else:
        warnings.warn(f"Pretrained model not found at {model_path}. Using random weights.")
        
    return model


def load_pretrained_segmentation_model(num_classes=50, model_path=None):
    """
    Load a pretrained PointNet segmentation model.
    
    Args:
        num_classes (int): Number of classes for the model
        model_path (str): Path to the pretrained model weights
        
    Returns:
        PointNetSegmentation: The loaded model
    """
    model = PointNetSegmentation(num_classes=num_classes, feature_transform=True)
    
    if model_path is None:
        model_path = os.path.join("pretrained_models", "pointnet_segmentation.pth")
        
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            warnings.warn(f"Could not load pretrained model: {str(e)}")
    else:
        warnings.warn(f"Pretrained model not found at {model_path}. Using random weights.")
        
    return model


def preprocess_pointcloud(points, num_points=2048):
    """
    Preprocess point cloud for PointNet input.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of point cloud points
        num_points (int): Number of points to sample
        
    Returns:
        torch.Tensor: Preprocessed point cloud tensor of shape [1, 3, num_points]
    """
    # Extract xyz coordinates
    xyz = points[:, :3]
    
    # Center the point cloud
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    
    # Normalize to unit sphere
    max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / max_dist
    
    # Random sampling if necessary
    if len(xyz) >= num_points:
        idx = np.random.choice(len(xyz), num_points, replace=False)
    else:
        idx = np.random.choice(len(xyz), num_points, replace=True)
    
    xyz = xyz[idx, :]
    
    # Convert to torch tensor with batch dimension
    xyz_tensor = torch.from_numpy(xyz).float().transpose(0, 1).unsqueeze(0)
    
    return xyz_tensor


def classify_pointcloud(model, points, num_points=2048, class_names=None):
    """
    Classify a point cloud using a PointNet classification model.
    
    Args:
        model (PointNetClassification): PointNet classification model
        points (numpy.ndarray): [N, 3+] array of point cloud points
        num_points (int): Number of points to sample
        class_names (list): List of class names
        
    Returns:
        int: Predicted class index
        float: Confidence score
        str: Class name (if class_names is provided)
    """
    # Preprocess point cloud
    xyz_tensor = preprocess_pointcloud(points, num_points)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        logits, _, _ = model(xyz_tensor)
        
    # Get prediction
    pred_idx = torch.argmax(logits, dim=1).item()
    confidence = torch.exp(logits[0, pred_idx]).item()
    
    if class_names is not None and pred_idx < len(class_names):
        class_name = class_names[pred_idx]
    else:
        class_name = f"class_{pred_idx}"
        
    return pred_idx, confidence, class_name


def segment_pointcloud(model, points, num_points=2048, class_names=None):
    """
    Perform semantic segmentation on a point cloud using a PointNet segmentation model.
    
    Args:
        model (PointNetSegmentation): PointNet segmentation model
        points (numpy.ndarray): [N, 3+] array of point cloud points
        num_points (int): Number of points to sample
        class_names (list): List of class names
        
    Returns:
        numpy.ndarray: [N] array of class indices for each point
        list: List of class names (if class_names is provided)
    """
    # Original points
    original_xyz = points[:, :3]
    
    # Preprocess point cloud
    xyz_tensor = preprocess_pointcloud(points, num_points)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        logits, _, _ = model(xyz_tensor)
    
    # Get predictions for sampled points
    pred = torch.argmax(logits, dim=2).squeeze().cpu().numpy()
    
    # Map predictions back to original point cloud (nearest neighbor)
    processed_xyz = xyz_tensor.squeeze().transpose(0, 1).cpu().numpy()
    
    # For each original point, find nearest point in processed set
    original_labels = np.zeros(len(original_xyz), dtype=np.int32)
    
    # Simple nearest neighbor implementation
    # For large point clouds, a KD-tree would be more efficient
    for i, point in enumerate(original_xyz):
        # Calculate distances to all processed points
        dists = np.sum((processed_xyz - point)**2, axis=1)
        nearest_idx = np.argmin(dists)
        original_labels[i] = pred[nearest_idx]
        
    # Map class indices to class names if provided
    result_classes = None
    if class_names is not None:
        result_classes = [class_names[idx] if idx < len(class_names) else f"unknown_{idx}" 
                          for idx in original_labels]
        
    return original_labels, result_classes


def detect_objects_pointnet(points, segmentation_model, classification_model, 
                           seg_class_names=None, cls_class_names=None):
    """
    Detect and classify objects in a point cloud using PointNet models.
    
    Args:
        points (numpy.ndarray): [N, 3+] array of point cloud points
        segmentation_model (PointNetSegmentation): PointNet segmentation model
        classification_model (PointNetClassification): PointNet classification model
        seg_class_names (list): List of class names for segmentation
        cls_class_names (list): List of class names for classification
        
    Returns:
        list: List of dictionaries containing object information
        numpy.ndarray: Segmentation labels for each point
    """
    # First, segment the point cloud
    seg_labels, _ = segment_pointcloud(segmentation_model, points, class_names=seg_class_names)
    
    # Find unique object segments (ignoring background/road)
    unique_labels = np.unique(seg_labels)
    
    objects = []
    for label in unique_labels:
        # Skip background (usually label 0)
        if label == 0:
            continue
            
        # Get points for this segment
        mask = seg_labels == label
        object_points = points[mask]
        
        # Skip if too few points
        if len(object_points) < 10:
            continue
            
        # Classify the object
        class_idx, confidence, class_name = classify_pointcloud(
            classification_model, 
            object_points, 
            class_names=cls_class_names
        )
        
        # Compute bounding box
        min_bound = np.min(object_points[:, :3], axis=0)
        max_bound = np.max(object_points[:, :3], axis=0)
        center = (min_bound + max_bound) / 2
        dimensions = max_bound - min_bound
        
        # Create object dictionary
        obj = {
            'center': center,
            'dimensions': dimensions,
            'min_bound': min_bound,
            'max_bound': max_bound,
            'class': class_name,
            'confidence': confidence,
            'num_points': len(object_points)
        }
        
        objects.append(obj)
        
    return objects, seg_labels


class PointNetPlusPlus(nn.Module):
    """
    Simplified implementation of PointNet++ architecture.
    This is a placeholder for the actual implementation.
    """
    def __init__(self, num_classes=40):
        super(PointNetPlusPlus, self).__init__()
        warnings.warn("This is a simplified placeholder for PointNet++. "
                     "For actual implementation, please use official repository.")
        
        self.num_classes = num_classes
        # Actual implementation would include SA modules, FP modules, etc.
        
    def forward(self, xyz, features=None):
        # Placeholder for actual implementation
        batch_size = xyz.shape[0]
        return torch.zeros(batch_size, self.num_classes)


# Class names for ModelNet40 dataset
MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

# Class names for ShapeNet part segmentation (simplified)
SHAPENET_PART_CLASSES = [
    'airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
    'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table'
]

# Simplified semantic segmentation classes for autonomous driving
SEMANTIC_CLASSES = [
    'unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]