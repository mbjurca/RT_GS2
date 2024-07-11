import torch
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms

def compute_epsilon_from_clusters(points, dbscan_eps=0.05, min_samples=10):
        """
        Clusters points using DBSCAN and computes an efficient approximation of the average 
        distance within clusters by using distances to cluster centroids. Returns this mean
        distance as epsilon.
        
        Parameters:
        - points: PyTorch tensor of shape (N, 3), points in 3D space.
        - dbscan_eps: The maximum distance between two samples for one to be considered
          as in the neighborhood of the other.
        - min_samples: The number of samples in a neighborhood for a point to be considered
          as a core point.
        
        Returns:
        - epsilon: The mean of the average distances within each cluster to their centroids.
        """
        # Ensure points are on CPU and in NumPy format for clustering
        points_np = points.detach().cpu().numpy()

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=dbscan_eps, min_samples=min_samples).fit(points_np)
        labels = clustering.labels_

        # Filter out noise (-1 labels)
        unique_labels = set(labels) - {-1}

        cluster_distances = []

        for label in unique_labels:
            # Extract points in the current cluster
            cluster_points = points_np[labels == label]
            
            # Compute the centroid of the cluster
            centroid = np.mean(cluster_points, axis=0)
            
            # Compute distances from each point in the cluster to the centroid
            distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
            
            # Compute average distance for the cluster
            avg_distance = np.mean(distances)
            cluster_distances.append(avg_distance)

        # Compute the mean of the average distances as epsilon
        if cluster_distances:
            epsilon = np.mean(cluster_distances)
        else:
            epsilon = 0  # Fallback value in case no clusters were found

        return epsilon

def save_gaussians_to_ply(filename, gaussians):
        """
        Saves gaussian centers to a PLY file.
        
        Parameters:
        - filename: Path to the output PLY file.
        - gaussians: Numpy array of shape (N, 3) containing the gaussian centers to save.
        """
        header = """ply
            format ascii 1.0
            element vertex {}
            property float x
            property float y
            property float z
            end_header
            """.format(gaussians.shape[0])

        with open(filename, 'w') as f:
            f.write(header)
            np.savetxt(f, gaussians, fmt='%f %f %f')

def save_gaussians_to_ply(filename, gaussians):
        """
        Saves gaussian centers to a PLY file.
        
        Parameters:
        - filename: Path to the output PLY file.
        - gaussians: Numpy array of shape (N, 3) containing the gaussian centers to save.
        """
        header = """ply
            format ascii 1.0
            element vertex {}
            property float x
            property float y
            property float z
            end_header
            """.format(gaussians.shape[0])

        with open(filename, 'w') as f:
            f.write(header)
            np.savetxt(f, gaussians, fmt='%f %f %f')

def vizualize_tuple(rendering, semantic, instance):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize to fit your needs

    axs[0].imshow(rendering.permute(1, 2, 0).cpu().numpy())  # Or rendering_rgb if conversion is needed
    axs[0].set_title('Rendering')
    axs[0].axis('off')  # Turn off axis

    axs[1].imshow(semantic)
    axs[1].set_title('Semantic')
    axs[1].axis('off')  # Turn off axis

    axs[2].imshow(instance)
    axs[2].set_title('Instance')
    axs[2].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()

def transform_point_4x4(points, matrix):
            # Append a column of ones to the points to handle homogeneous coordinates
            ones = torch.ones(points.shape[0], 1, device=points.device)
            points_hom = torch.cat([points, ones], dim=1)
            
            # Transform the points using the 4x4 matrix
            transformed_points_hom = torch.mm(matrix.t(), points_hom.t())
            
            return transformed_points_hom.t()

def transform_point_4x3(points, matrix):
    # Directly use the 4x3 matrix to transform points
    ones = torch.ones(points.shape[0], 1, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    
    # Transform the points using the 4x3 matrix
    transformed_points = torch.mm(matrix.t(), points_hom.t())
    
    return transformed_points.t()[:, :3]

def rearrange_cov3d(cov3D):
    """
    Rearranges the flattened covariance matrix to a proper 3x3 matrix.
    cov3D: Tensor of shape (no_gaussians, 6)
    """
    no_gaussians = cov3D.shape[0]
    cov_matrix = torch.zeros((no_gaussians, 3, 3), device=cov3D.device, dtype=cov3D.dtype)
    
    # Fill in the upper triangular part
    cov_matrix[:, 0, 0] = cov3D[:, 0]  # Cxx
    cov_matrix[:, 0, 1] = cov3D[:, 1]  # Cxy
    cov_matrix[:, 0, 2] = cov3D[:, 2]  # Cxz
    cov_matrix[:, 1, 1] = cov3D[:, 3]  # Cyy
    cov_matrix[:, 1, 2] = cov3D[:, 4]  # Cyz
    cov_matrix[:, 2, 2] = cov3D[:, 5]  # Czz
    
    # Since the covariance matrix is symmetric, fill in the lower triangular part
    cov_matrix[:, 1, 0] = cov3D[:, 1]  # Cxy
    cov_matrix[:, 2, 0] = cov3D[:, 2]  # Cxz
    cov_matrix[:, 2, 1] = cov3D[:, 4]  # Cyz
    
    return cov_matrix

def compute_cov2d(means, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    """
    Compute the 2D covariance for a batch of 3D points.
    """
    t = transform_point_4x3(means, viewmatrix)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    t[:, 0] = torch.clamp(t[:, 0] / t[:, 2], -limx, limx) * t[:, 2]
    t[:, 1] = torch.clamp(t[:, 1] / t[:, 2], -limy, limy) * t[:, 2]

    J = torch.zeros((means.shape[0], 3, 3), device=means.device)
    J[:, 0, 0] = focal_x / t[:, 2]
    J[:, 1, 1] = focal_y / t[:, 2]
    J[:, 0, 2] = -(focal_x * t[:, 0]) / (t[:, 2] ** 2)
    J[:, 1, 2] = -(focal_y * t[:, 1]) / (t[:, 2] ** 2)

    W = viewmatrix[:3, :3]  # Only rotational part for W
    T = torch.matmul(W, J)

    cov3D=rearrange_cov3d(cov3D)
    cov = torch.matmul(T, torch.matmul(cov3D, T.transpose(1, 2)))

    # Discard 3rd row and column and apply low-pass filter
    cov2d = torch.zeros((means.shape[0], 2, 2), device=means.device)
    cov2d[:, 0, 0] = cov[:, 0, 0] + 0.3
    cov2d[:, 0, 1] = cov[:, 0, 1]
    cov2d[:, 1, 0] = cov[:, 1, 0]
    cov2d[:, 1, 1] = cov[:, 1, 1] + 0.3
    return cov2d

def visualize_point_cloud(points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def visualize_colored_point_cloud(points, voxel_indices):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a discrete colormap with a set of distinct colors
    cmap = plt.get_cmap('tab20')  # 'tab20' provides 20 distinct colors
    max_index = float(max(voxel_indices)+1)
    colors = cmap((voxel_indices.flatten() % 20) / 20)  # Ensures more distinct colors
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def visualize_shuffled_point_cloud(new_coordinates, original_voxel_indices):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use the same colors based on the original voxel assignment
    colors = plt.cm.jet(original_voxel_indices.flatten() / float(max(original_voxel_indices)+1))
    ax.scatter(new_coordinates[:, 0], new_coordinates[:, 1], new_coordinates[:, 2], color=colors)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

class ToTensor(object):

    def __call__(self, sample):
        rendering = sample['rendering']
        rendered_features = sample['rendered_features']
        depth = sample['depth']
        semantic_mask = sample['semantic_mask']

        rendering = torch.tensor(rendering).permute(2, 0, 1) 
        rendered_features = torch.tensor(rendered_features)
        rendered_features = transforms.functional.resize(rendered_features, (rendering.shape[1], rendering.shape[2]))

        if not isinstance(depth, list):
            depth = torch.tensor(depth.astype(np.float32), dtype=torch.float32)

        semantic_mask = torch.tensor(semantic_mask, dtype=torch.long)

        return {'rendering': rendering, 'rendered_features': rendered_features, 'depth': depth, 'semantic_mask': semantic_mask}
    
class Normalize(object):    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        rendering = sample['rendering']
        rendering = rendering / 255.

        return {'rendering': rendering, 'rendered_features': sample['rendered_features'], 'depth': sample['depth'], 'semantic_mask': sample['semantic_mask']}
    
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, sample):
        rendering = sample['rendering']

        if np.random.rand() < self.p:
            rendering = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(rendering)

        return {'rendering': rendering, 'rendered_features': sample['rendered_features'], 'depth': sample['depth'], 'semantic_mask': sample['semantic_mask']}
    
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        rendering = sample['rendering']
        rendered_features = sample['rendered_features']
        semantic_mask = sample['semantic_mask']
        depth = sample['depth']

        if np.random.rand() < self.p:
            rendering = transforms.functional.hflip(rendering)
            rendered_features = transforms.functional.hflip(rendered_features)
            if not isinstance(depth, list):
                depth = transforms.functional.hflip(depth)
            semantic_mask = transforms.functional.hflip(semantic_mask)

        return {'rendering': rendering, 'rendered_features': rendered_features, 'depth': depth, 'semantic_mask': semantic_mask}
    
class RandomZoomIn(object):
    def __init__(self, p=0.5, max_scale_factor=1.5):
        self.p = p
        self.scale_factor = np.random.uniform(1, max_scale_factor)

    def __call__(self, sample):
        rendering = sample['rendering']
        rendered_features = sample['rendered_features']
        semantic_mask = sample['semantic_mask']
        depth = sample['depth']

        if np.random.rand() < self.p:
            _, H, W = rendering.shape
            new_h, new_w = int(H * self.scale_factor), int(W * self.scale_factor)
            top_h = np.random.randint(0, new_h - H - 1)
            top_w = np.random.randint(0, new_w - W - 1)

            rendering = transforms.functional.resize(rendering, (new_h, new_w))[:, top_h:top_h+H, top_w:top_w+W]
            rendered_features = transforms.functional.resize(rendered_features, (new_h, new_w))[:, top_h:top_h+H, top_w:top_w+W]
            if not isinstance(depth, list):
                depth = transforms.functional.resize(depth.unsqueeze(0), (new_h, new_w))[:, top_h:top_h+H, top_w:top_w+W].squeeze(0)
            semantic_mask = transforms.functional.resize(semantic_mask.unsqueeze(0), (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)[:, top_h:top_h+H, top_w:top_w+W].squeeze(0)
            
        return {'rendering': rendering, 'rendered_features': rendered_features, 'depth': depth, 'semantic_mask': semantic_mask}
    
class RandomPerspective(object):
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale
    
    # https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomPerspective
    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __call__(self, sample):
        rendering = sample['rendering']
        rendered_features = sample['rendered_features']
        semantic_mask = sample['semantic_mask']
        depth = sample['depth']

        if np.random.rand() < self.p:
            _, H, W = rendering.shape
            start_points, end_points = self.get_params(W, H, self.distortion_scale)
            rendering = transforms.functional.perspective(rendering, start_points, end_points)
            rendered_features = transforms.functional.perspective(rendered_features, start_points, end_points)
            if not isinstance(depth, list):
                depth = transforms.functional.perspective(depth.unsqueeze(0), start_points, end_points).squeeze(0)
            semantic_mask = transforms.functional.perspective(semantic_mask.unsqueeze(0), start_points, end_points, interpolation=transforms.InterpolationMode.NEAREST, fill=-1).squeeze(0)

        return {'rendering': rendering, 'rendered_features': rendered_features, 'depth': depth, 'semantic_mask': semantic_mask}

class RandomCutOut(object):
     
    def __init__(self, p=0.5, patch_size=0.1):
        self.p = p
        self.patch_size = patch_size

    def __call__(self, sample):
        rendering = sample['rendering']
        rendered_features = sample['rendered_features']

        if np.random.rand() < self.p:
            _, H, W = rendering.shape
            patch_h = int(H * self.patch_size)
            patch_w = int(W * self.patch_size)

            patch_rendering_x = np.random.randint(0, W - patch_w)
            patch_rendering_y = np.random.randint(0, H - patch_h)

            patch_features_x = np.random.randint(0, W - patch_w)
            patch_features_y = np.random.randint(0, H - patch_h)

            rendering[:, patch_rendering_y:patch_rendering_y+patch_h, patch_rendering_x:patch_rendering_x+patch_w] = 0
            rendered_features[:, patch_features_y:patch_features_y+patch_h, patch_features_x:patch_features_x+patch_w] = 0

        return {'rendering': rendering, 'rendered_features': rendered_features, 'depth': sample['depth'], 'semantic_mask': sample['semantic_mask']}

    

    