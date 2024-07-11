import torch
import sys


sys.path.append('/home/mihnea/mihnea/Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops')
from pointnet2_utils import furthest_point_sample

class Sampler:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, point_cloud):
        raise NotImplementedError("Sample method needs to be implemented by subclasses.")

class FarthestPointSampler(Sampler):
    def sample(self, point_cloud, max_chunk_size=50000):

        N, _ = point_cloud.shape
        sampled_points_list = []
        for start_idx in range(0, N, max_chunk_size):
            end_idx = min(start_idx + max_chunk_size, N)
            chunk = point_cloud[start_idx:end_idx, ...]
            sampled_chunk = furthest_point_sample(chunk, self.num_samples)
            sampled_points_list.append(sampled_chunk)
        # Combine sampled points from each chunk
        sampled_points = torch.cat(sampled_points_list, dim=1)
        # Further processing can be added here if necessary
        return sampled_points

class RandomSampler(Sampler):
    def sample(self, point_cloud):
        N, D = point_cloud.shape
        indices = torch.randperm(N)[:self.num_samples].to(point_cloud.device)
        sampled_points = point_cloud[indices]
        
        # Compute the mask tensor
        mask = torch.full((N,), fill_value=-1, dtype=torch.long, device=point_cloud.device)
        for i in range(N):
            if i not in indices:
                dists = torch.norm(point_cloud[i] - sampled_points, dim=1)
                mask[i] = indices[torch.argmin(dists)]
        
        return sampled_points, mask

def compute_closest_indices_optimized(initial_points, selected_points, selected_indices):
    # Function to compute mask tensor without explicit loops, for use in FPS implementation
    pass
