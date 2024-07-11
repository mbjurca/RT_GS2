import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

def compute_min_distances(point_cloud1, point_cloud2):
    """
    Compute minimum distances from each point in point_cloud1 to point_cloud2
    and vice versa, assuming both point clouds have the same shape.
    """
    # Compute squared distances
    dists_squared = torch.sum((point_cloud1[:, :, None] - point_cloud2[:, None, :]) ** 2, dim=-1)
    
    # Find the minimum distances
    min_dists1, _ = torch.min(dists_squared, dim=2)  # min dists for points in cloud1 to cloud2
    min_dists2, _ = torch.min(dists_squared, dim=1)  # min dists for points in cloud2 to cloud1

    return min_dists1, min_dists2

class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, point_cloud1, point_cloud2):
        """
        Computes the Chamfer Distance between two point clouds.
        
        Args:
            point_cloud1: Tensor of shape (1, N, D) where N is the number of points in the point cloud
                          and D is the dimension of each point.
            point_cloud2: Tensor of shape (1, N, D), assumed to have the same shape as point_cloud1.
            
        Returns:
            A scalar tensor with the Chamfer Distance.
        """        
        min_dists1, min_dists2 = compute_min_distances(point_cloud1, point_cloud2)

        # Compute the Chamfer Distance
        chamfer_distance = torch.mean(min_dists1) + torch.mean(min_dists2)
        
        return chamfer_distance

# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        loss = self.criterion(x, label)
        return loss
    
def safe_one_hot(tensor, num_classes, ignore_index=-1):
    # Replace ignore_index with a temporary valid class index
    # Choose a value that is outside your normal class range
    # Here, we use `num_classes` as the temporary index because it's guaranteed to be outside the normal range
    temp_index = num_classes
    modified_tensor = torch.where(tensor == ignore_index, temp_index, tensor)
    
    # Perform one_hot encoding
    one_hot_encoded = torch.nn.functional.one_hot(modified_tensor, num_classes + 1)  # +1 for the temporary class
    
    # If ignore_index was used, remove the temporary class dimension from the one_hot encoding
        # This slices off the last dimension, effectively ignoring the temporary class
    one_hot_encoded = one_hot_encoded[..., :-1].permute(0, 3, 1, 2)    
    
    return one_hot_encoded

def dice_loss(pred, targets, ignore_index, average, eps: float = 1e-8):
    # Input tensors will have shape (Batch, Class)
    # Dimension 0 = batch
    # Dimension 1 = class code or predicted logit
    
    # compute softmax over the classes axis to convert logits to probabilities
    pred_soft = torch.softmax(pred, dim=1)

    # create reference one hot tensors
    ref_one_hot = safe_one_hot(targets, num_classes = pred.shape[1], ignore_index=ignore_index)

    #Calculate the dice loss
    if average == "micro":
      #Use dim=1 to aggregate results across all classes
      intersection = torch.sum(pred_soft * ref_one_hot, dim=1)
      cardinality = torch.sum(pred_soft + ref_one_hot, dim=1)
    else:
      #With no dim argument, will be calculated separately for each class
      intersection = torch.sum(pred_soft * ref_one_hot)
      cardinality = torch.sum(pred_soft + ref_one_hot)

    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss = -dice_score + 1.0

    # reduce the loss across samples (and classes in case of `macro` averaging)
    dice_loss = torch.mean(dice_loss)

    return dice_loss


class DiceLoss(nn.Module):
    def __init__(self, ignore_index, average=None, eps: float = 1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, pred, ref):
        return dice_loss(pred, ref, self.ignore_index, self.average, self.eps)
    
class DiceCELoss(nn.Module):
    def __init__(self, ignore_index) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(ignore_index)
        self.ce_loss =  nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, ref):
        return self.dice_loss(pred, ref) + self.ce_loss(pred, ref)

    
class LossCR(nn.Module):
    def __init__(self, label_smoothing, num_classes, feat_dim, alpha=1.0):
        super(LossCR, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.register_buffer('W_star', self.init_etf_classifier(num_classes, feat_dim, alpha))
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    def init_etf_classifier(self, K, d, alpha):
        # Create an initial random (d x K) matrix
        W = torch.randn(d, K)
        
        # Perform QR decomposition on W to obtain orthonormal basis
        Q, R = torch.linalg.qr(W)
        
        # Use the first K columns of Q
        W_star = Q[:, :K]
        
        # Scale W_star to meet the ETF condition W W^T = alpha (I - 1/K J)
        scaling_factor = np.sqrt((alpha * (K - 1)) / K)
        W_star *= scaling_factor
        
        return W_star
    
    def forward(self, preds, labels, labels_depth, z):
        """
        Calculate the Center Collapse Regularization loss combined with semantic loss.
        
        Args:
        preds (torch.Tensor): Predicted logits of shape (N, C, H, W), where N is the batch size,
                              C is the number of classes, and H, W are the spatial dimensions.
        labels (torch.Tensor): True labels of shape (N, H, W).
        z (torch.Tensor): The feature map of shape (N, d, H, W).
        
        Returns:
        torch.Tensor: The computed total loss.
        """
        N, d, H, W = z.size()
        z = z.permute(0, 2, 3, 1).reshape(-1, d)  # Flatten spatial dimensions
        labels_flat = labels.view(-1)  # Flatten label tensor

        # Compute class centers Z_bar
        Z_bar = torch.zeros(self.num_classes, d, device=z.device)
        for k in range(self.num_classes):
            mask = (labels_flat == k)
            if torch.any(mask):
                Z_bar[k] = torch.mean(z[mask], dim=0)

        logits = torch.matmul(Z_bar, self.W_star.to(z.device))  # Transpose W_star to match dimensions
        target = torch.arange(self.num_classes, device=Z_bar.device)
        LCR_loss = self.semantic_loss(logits, target)

        semantic_loss = self.semantic_loss(preds, labels)

        # Here, you might want to check if lambda should be a fixed value or an adjustable hyperparameter.
        lambda_reg = 0.4
        total_loss = lambda_reg * LCR_loss + semantic_loss

        return total_loss

