from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import torch 
from torch_geometric.nn import fps
import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import warnings
from random import randint
import os
from sklearn.metrics import confusion_matrix
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np

def random_point_sampling(point_cloud, num_samples):
    """
    Sample a subset of points randomly from a point cloud.

    Parameters:
    - point_cloud: Tensor of shape (N, D), the initial point cloud.
    - num_samples: int, the number of points to sample.

    Returns:
    - sampled_points: Tensor of the sampled points.
    - indices: Tensor odf the sampled point indices
    """
    N, D = point_cloud.shape
    indices = torch.randperm(N)[:num_samples].to(point_cloud.device)
    sampled_points = point_cloud[indices]
    return sampled_points, indices

def compute_closest_indices_optimized(initial_points, selected_points, selected_indices):
    """
    Compute the indices tensor indicating -1 for selected points or the index of the closest selected point,
    optimized to avoid explicit for-loops.

    Parameters:
    - initial_points: Tensor of shape (N, D), the initial point cloud.
    - selected_points: Tensor of the selected points.
    - selected_indices: Tensor of indices of the selected points in the initial point cloud.

    Returns:
    - indices_tensor: Tensor of shape (N,) with -1 for selected points or the index of the closest selected point.
    """
    # Compute all pairwise distances between initial points and selected points
    initial_expanded = initial_points[:3, :].unsqueeze(1)  # Shape: (N, 1, D)
    selected_expanded = selected_points[:3, :].unsqueeze(0)  # Shape: (1, M, D)
    distances = torch.sum((initial_expanded - selected_expanded) ** 2, dim=2)  # Shape: (N, M)

    # Find the index of the closest selected point for each initial point
    min_distances_indices = torch.argmin(distances, dim=1)  # Shape: (N,)

    # Create the indices tensor, initially setting all values to their closest selected point index
    indices_tensor = selected_indices[min_distances_indices]

    # Set the indices of the selected points to -1
    for idx in selected_indices:
        indices_tensor[idx] = -1

    return indices_tensor

def custom_collate(batch):
    # Extract all items (assuming each item in the batch is a tuple as per your return statement)
    renderings, features, semantic_mask, mask, view, gaussians, pipeline, background = zip(*batch)
    
    # Collate each item as usual, except for the views (Camera objects)
    rendering = torch.stack(renderings)
    features = torch.cat(features)  # Assuming features is already batched properly
    semantic_mask = torch.stack(semantic_mask)
    mask = torch.stack(mask)
    # Return a tuple or dict as needed by your model
    return rendering, features, semantic_mask, mask, view, gaussians, pipeline, background

# Create a custom collate in order to deal with the fact that the number of nodes per instance
# is different from one graph to another
def graph_collate(batch):
    renderings, features, semantic_masks, instance_masks = zip(*batch)
    
    # Concatenate features along dimension 1
    # First, calculate the total length of the concatenated features and create an index array
    total_length = sum(f.size(0) for f in features)  # Total number of features across all items
    concatenated_features = torch.cat(features, dim=0)  # Concatenate all features into a single 1D array
    
    # Create an index array
    indices = []
    for i, f in enumerate(features):
        indices.extend([i] * f.size(0))  # Repeat the index for each feature in the item
    indices = torch.tensor(indices, dtype=torch.long)  # Convert to a torch tensor
    
    # Use default_collate for other data types
    renderings = default_collate(renderings)
    semantic_masks = default_collate(semantic_masks)
    instance_masks = default_collate(instance_masks)
    # Return the concatenated features and the indices array along with other collated data
    return renderings, concatenated_features, indices, semantic_masks, instance_masks

# def cluster_gaussians(pos, eps=0.5, min_samples=5):
#     """
#     Cluster points using DBSCAN based on the first 3 features and select representative points for each cluster.
#     Returns the full feature set for the representative points.
    
#     Parameters:
#     - pos: Tensor of shape [n, d], where n is the number of points and d is the dimensionality.
#     - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
#     - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    
#     Returns:
#     - clusters: Tensor of cluster labels for each point.
#     - representative_points: Tensor of the full features of the representative points for each cluster.
#     """
#     # Use only the first 3 features for clustering
#     pos_for_clustering = pos[:, :3].numpy()
    
#     # Apply DBSCAN
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_for_clustering)
#     labels = clustering.labels_
    
#     # Find unique clusters, ignoring noise if present (-1 label)
#     unique_clusters = np.unique(labels[labels >= 0])
    
#     # Initialize a list to store the full features of the representative points
#     representative_points = []

#     for cluster in unique_clusters:
#         # Indices of points in the current cluster
#         cluster_indices = np.where(labels == cluster)[0]
        
#         # Strategy: Choose the point closest to the centroid as representative
#         cluster_points = pos_for_clustering[cluster_indices]
#         centroid = np.mean(cluster_points, axis=0)
#         representative_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))]
        
#         # Append the full features of the representative point
#         representative_points.append(pos[representative_idx])

#     # Convert the list of tensors to a single tensor
#     representative_points = torch.stack(representative_points)
    
#     # Convert cluster labels to a tensor
#     clusters = torch.tensor(labels, dtype=torch.long)
    
#     return clusters, representative_points


def cluster_gaussians(pos, min_cluster_size=5, min_samples=None):
    """
    Cluster points using HDBSCAN based on the first 3 features and select representative points for each cluster.
    Parameters:
    - pos: Tensor of shape [n, d], where n is the number of points and d is the dimensionality.
    - min_cluster_size: The minimum size of clusters; smaller clusters will be considered as noise.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - clusters: Tensor of cluster labels for each point.
    - representative_points: Tensor of the full features of the representative points for each cluster.
    """
    # Convert PyTorch tensor to numpy for HDBSCAN
    pos_np = pos.numpy()
    
    # Use only the first 3 features for clustering
    pos_for_clustering = pos_np[:, :3]
    
    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(pos_for_clustering)
    labels = clusterer.labels_
    
    # Find unique clusters, ignoring noise if present (-1 label)
    unique_clusters = np.unique(labels[labels >= 0])
    
    # Initialize a list to store the full features of the representative points
    representative_points = []

    for cluster in unique_clusters:
        # Indices of points in the current cluster
        cluster_indices = np.where(labels == cluster)[0]
        
        # Strategy: Choose the point closest to the centroid as representative
        cluster_points = pos_for_clustering[cluster_indices]
        centroid = np.mean(cluster_points, axis=0)
        representative_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))]
        
        # Append the full features of the representative point
        representative_points.append(pos[representative_idx])

    # Convert the list of numpy arrays to a single PyTorch tensor
    representative_points = torch.stack(representative_points)
    
    # Convert cluster labels to a PyTorch tensor
    clusters = torch.tensor(labels, dtype=torch.long)
    
    return clusters, representative_points


def compute_binary_masks(gaussians_labels, mask, total_no_gaussians):
    # takes as input the labeled gaussians returnes the colors for each mask
    # output: C x mask_colors
    # the mask_colors will have value 1 on every channel if the gaussian was assigend to that specific class if not

    mask_colors = torch.zeros(101, total_no_gaussians, 3).to('cuda')

    unique_vals, inverse_indices = torch.unique(gaussians_labels, return_inverse=True)

    one_hot_encoded = torch.zeros(101, gaussians_labels.shape[0]).to('cuda')

    one_hot_encoded.scatter_(0, inverse_indices.unsqueeze(0), 1)

    mask = mask.squeeze(0)
    mask_colors[:, mask, ...] = one_hot_encoded.unsqueeze(-1)

    return mask_colors

def calculate_miou_and_log_confusion_matrix(logits, semantic_mask):

    logits, semantic_mask = logits.squeeze(), semantic_mask.squeeze()

    # Convert logits to predicted labels
    predicted_labels = torch.argmax(logits, 0)

    # Flatten the tensors for confusion matrix computation
    predicted_labels_flat = predicted_labels.view(-1).clone().detach().cpu().numpy()
    semantic_mask_flat = semantic_mask.view(-1).clone().detach().cpu().numpy()
    
    # Compute confusion matrix
    num_classes = logits.shape[0]
    cm = confusion_matrix(semantic_mask_flat, predicted_labels_flat, labels=list(range(num_classes)))
    
    # Calculate IoU for each class
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm)
    iou = intersection / (union + np.finfo(float).eps)  # Adding a small epsilon to avoid division by zero
    # Identify classes present in the semantic mask
    present_classes = np.unique(semantic_mask_flat)
    present_classes = present_classes[present_classes != 0]
    
    # Filter IoU for present classes and compute mean
    present_iou = iou[present_classes]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        miou = np.nanmean(present_iou[present_iou > 0]) 

    return miou

def miou(logits, labels, num_classes):
    # One-hot encode logits and labels
    logits_one_hot = F.one_hot(logits, num_classes).permute(0, 3, 1, 2)
    labels_one_hot = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
    
    # Compute intersection and union
    intersection = logits_one_hot & labels_one_hot
    union = logits_one_hot | labels_one_hot
    
    # Sum over all dimensions except the classes dimension to get the shape of [num_classes]
    intersection_sum = intersection.sum(dim=[1, 2, 3])
    union_sum = union.sum(dim=[1, 2, 3])
    
    # Compute IoU score for each class, then take mean
    iou = intersection_sum / (union_sum + 1e-6)
    miou = torch.mean(iou)
    
    return miou

def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()

def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label=-1):
    if (true_labels == ignore_label).all():
        return [0]*4
    
    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))

    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious

def map_to_color(mask, color_mapping):
    """Map semantic mask to color image using color_mapping."""
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        color_image[mask == label] = color
    return color_image

def create_visualization(semantic_mask, predicted_mask, image, color_mapping, semantic_mapping, output_dir):
    # Ensure tensors are moved to CPU, cloned, and detached
    semantic_mask = semantic_mask.clone().detach().cpu()
    predicted_mask = predicted_mask.clone().detach().cpu()
    image = image.clone().detach().cpu()
    
    # Map masks to color images
    ground_truth_color = map_to_color(semantic_mask.numpy(), color_mapping)
    predicted_color = map_to_color(predicted_mask.numpy(), color_mapping)
    
    # Determine unique labels in both masks for the legend
    unique_labels = np.unique(np.concatenate((semantic_mask.numpy().flatten(), predicted_mask.numpy().flatten())))
    relevant_color_mapping = {label: color_mapping[label] for label in unique_labels if label in color_mapping}
    
    # Prepare the images for plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust figsize as needed
    axes[0].imshow(image.numpy().transpose(1, 2, 0))
    axes[0].set_title('Original')
    axes[1].imshow(ground_truth_color)
    axes[1].set_title('Ground Truth')
    axes[2].imshow(predicted_color)
    axes[2].set_title('Prediction')
    
    # Hide axis for image plots
    for ax in axes[:3]:
        ax.axis('off')
    
    # Create legend on the 4th subplot
    axes[3].axis('off')
    handles = [plt.Rectangle((0,0),1,1, color=np.array(color)/255.0) for color in relevant_color_mapping.values()]
    labels = [key for key in relevant_color_mapping.keys()]
    semantic_labels = [next((k for k, v in semantic_mapping.items() if v == label), 'Unknown') for label in labels]
    axes[3].legend(handles, semantic_labels, loc='upper left', title="Legend")
    
    plt.tight_layout()
    

    plt.savefig(f'{output_dir}/{randint(10, 10000)}.png')


def save_visualization(model, train_loader, test_loader, semantic_mapping, color_mapping, output_dir):

    output_dir_train = os.path.join(output_dir, 'plots', 'train')
    output_dir_test = os.path.join(output_dir, 'plots', 'test')
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)

    for rendering, features, semantic_mask, mask, view, gaussians, pipeline, background in train_loader:

        features = features.unsqueeze(0)

        pred = model(features, rendering, view, gaussians, pipeline, background, mask=None).squeeze().argmax(0)

        create_visualization(semantic_mask=semantic_mask.squeeze(),
                            predicted_mask=pred,
                            image=rendering.squeeze(),
                            color_mapping=color_mapping,
                            semantic_mapping=semantic_mapping, 
                            output_dir=output_dir_train)
        
    for rendering, features, semantic_mask, mask, view, gaussians, pipeline, background in test_loader:

        features = features.unsqueeze(0)

        pred = model(features, rendering, view, gaussians, pipeline, background, mask=None).squeeze().argmax(0)

        create_visualization(semantic_mask=semantic_mask.squeeze(),
                            predicted_mask=pred,
                            image=rendering.squeeze(),
                            color_mapping=color_mapping,
                            semantic_mapping=semantic_mapping, 
                            output_dir=output_dir_test)
        
# from AsynFormer
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=4,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def image_similarity_metrics(image1, image2):
    # Validate input dimensions
    if image1.shape != image2.shape or image1.shape[2] != 3:
        raise ValueError("Both images must have the shape [H, W, 3]")
    
    # Compute PSNR
    psnr_value = psnr(image1, image2, data_range=255)
    
    # Compute SSIM
    ssim_value = ssim(image1, image2, multichannel=True, data_range=255)
    
    # Prepare images for LPIPS computation
    # Convert images to torch tensors
    tensor1 = torch.from_numpy(image1).float().permute(2, 0, 1).unsqueeze(0) / 255
    tensor2 = torch.from_numpy(image2).float().permute(2, 0, 1).unsqueeze(0) / 255
    
    # Normalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor1 = (tensor1 - mean) / std
    tensor2 = (tensor2 - mean) / std
    
    # Initialize the LPIPS model
    lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet as the feature extractor

    # Compute LPIPS
    lpips_value = lpips_model(tensor1, tensor2)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'LPIPS': lpips_value.item()  # Convert tensor to Python scalar
    }

#from atlas
def compute_depth_metrics(depth_pred, depth_trgt):
    """ Computes 2d metrics between two depth maps
    
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth

    Returns:
        Dict of metrics
    """
    mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt<10) * (depth_trgt>0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred-depth_trgt)
    abs_rel = abs_diff/depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff/depth_trgt
    sq_log_diff = (np.log(depth_pred)-np.log(depth_trgt))**2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25**2).astype('float')
    r3 = (thresh < 1.25**3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics
