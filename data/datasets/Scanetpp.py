import torch 
import os
import sys
from tqdm import tqdm 
import math
from argparse import ArgumentParser
import torchvision
from torch.utils import data
import numpy as np
import cv2
import json 
import os
import torch.nn.functional as F
import torchvision.transforms as T
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from models.gaussian_splatting.gaussian_renderer import render
from models.gaussian_splatting.gaussian_renderer import GaussianModel
from models.gaussian_splatting.scene import Scene
from scipy.spatial import KDTree

from utils.dataset_utils import compute_epsilon_from_clusters, transform_point_4x4, transform_point_4x3#, #compute_cov2d, assign_semantic_labels
from utils.sh_utils import eval_sh



class ModelParams():

    def __init__(self, scene_dir):

        self.sh_degree = 3
        self.source_path = scene_dir
        self.model_path = scene_dir
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = True

class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = True

class ScanNetpp_dataset(data.Dataset):

    def __init__(self, 
            scene_dir,
            device,
            img_size, 
            eval_mode='train', 
            save_randerings=False,
            color_mapping_file='/media/mihnea/0C722848722838BA/ScanNet++2/metadata/top_100_color_mapping.json', 
            semantic_mapping='/media/mihnea/0C722848722838BA/ScanNet++2/metadata/top_100_semantic_map.json', 
            sh2color=True):
        super().__init__()

        self.scene_dir = scene_dir
        self.device = device
        # self.num_sem_cls = 101
        self.semantic_mapping = semantic_mapping
        self.img_size = img_size
        self.sh2color = sh2color



        dataset = ModelParams(scene_dir)
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, load_iteration=30000, shuffle=False)
        self.pipeline = PipelineParams()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        # Compute the adjacency lists
        # self.epsilon = compute_epsilon_from_clusters(self.gaussians.get_xyz)

        # Load coresponding views    
        if eval_mode == 'train':
            self.views = self.scene.getTrainCameras()
        elif eval_mode == 'test':
            self.views = self.scene.getTestCameras()
        

        if save_randerings:
            self.save_randerings()

        # Create color mapping such that semantic/instance randering is possible
        with open(color_mapping_file, 'r') as file:
            color_mapping = json.load(file)

        self.color_mapping = {int(k): np.array(v, dtype=np.uint8) for k, v in color_mapping.items()}

        self.transform_dino = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


    def __getitem__(self, index):

        
        view = self.views[index]

        mask, means_2d = self.frastrum_culling_gaussians(points=self.gaussians.get_xyz, 
                                            view_matrix=view.world_view_transform, 
                                            projection_matrix=view.full_proj_transform, 
                                            image_height=view.image_height, 
                                            image_width=view.image_width)
                        
        means_3d = self.gaussians.get_xyz
        cov_3d = self.gaussians.get_covariance()

        if self.sh2color:
            color_info = self.compute_sh2color(viewpoint_camera=view, 
                                            pc=self.gaussians, 
                                            mask=mask)
        else:
            color_info = self.gaussians.get_features.reshape(means_3d.shape[0], -1)

        opacity = self.gaussians.get_opacity      

        # read semantic mask and convert to torch long
        semantic_mask = cv2.imread(os.path.join(self.scene_dir, 'semantic', view.image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        semantic_mask =cv2.resize(semantic_mask, self.img_size, interpolation = cv2.INTER_NEAREST)
        semantic_mask = torch.tensor(semantic_mask, dtype=torch.long, device=self.device)


        # gaussians_semantic_labels = self.compute_gaussians_label(means_2d=means_2d,
        #                                             semantic_mask=semantic_mask)        

        # gaussian_colors_rasterize = self.compute_colors(gaussian_labels=gaussians_semantic_labels,
        #                                             color_mapping=self.color_mapping,
        #                                             gaussians_mask=mask)
        
        # mask_colors = self.compute_binary_masks(gaussians_semantic_labels, mask, self.gaussians.get_xyz.shape[0])
        
        # for cls in range(self.num_sem_cls):
        #      randering = self.resterize_image(view, mask_colors[cls, ...])
        #      cv_render = randering.permute(1, 2, 0).clone().detach().cpu().numpy()
        #      cv2.imshow(f'{cls}', cv_render)

        # cv2.waitKey()

        # randering = self.resterize_image(view, gaussian_colors_rasterize)

        """-------------UNCOMENT THIS FOR VISALIZATION-------------

        cv_render = randering.permute(1, 2, 0).clone().detach().cpu().numpy()

        color_image = self.create_color_semantic_mask(semantic_mask=semantic_mask.clone().detach().cpu().numpy(), 
                                                    color_mapping=self.color_mapping)
        # Save or display the resulting color image
        cv2.imshow('original mask', color_image)  # Save the color image
        cv2.imshow('rendered mask', cv_render)  # Save the color image

        cv2.waitKey(20000)
        """
    
        # Then, concatenate them into a single one-dimensional array
        features = torch.cat((means_3d, opacity, cov_3d, color_info), dim=1)
    
        randering = self.resterize_image(view, None).unsqueeze(0)
        randering = F.interpolate(randering, size=(self.img_size[1], self.img_size[0]), mode='bilinear', align_corners=False).squeeze(0)
        randering = self.transform_dino(randering)

        return randering, features, semantic_mask, mask, view, self.gaussians, self.pipeline, self.background


    def __len__(self):
        return len(self.views)
    
    def resterize_image(self, view, override_color):
        #rasterize image given a specific view
        return render(view, self.gaussians, self.pipeline, self.background, override_color=override_color)["render"]

    def save_randerings(self):
            render_path = os.path.join(self.scene_dir, "dataset", "renders")
            os.makedirs(render_path, exist_ok=True)

            for idx, view in tqdm(self.views, desc="Saving randerings"):
                pass
    

    def frastrum_culling_gaussians(self, points, view_matrix, projection_matrix, image_height, image_width):

        # Convert inputs to PyTorch tensors
        # points_tensor = torch.tensor(points, dtype=torch.float32)
        # view_matrix_tensor = torch.tensor(view_matrix, dtype=torch.float32)
        # projection_matrix_tensor = torch.tensor(projection_matrix, dtype=torch.float32)
        
        # Transform points to homogeneous clip space
        p_hom = transform_point_4x4(points, projection_matrix)
        p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
        p_proj = p_hom[:, :3] * p_w.unsqueeze(-1)

        # Transform points using the view matrix
        p_view = transform_point_4x3(points, view_matrix)

        # Determine visibility based on z values and projected coordinates
        visible = (p_view[:, 2] >= 0) & (p_proj[:, 0] >= -1.3) & (p_proj[:, 0] <= 1.3) & \
                (p_proj[:, 1] >= -1.3) & (p_proj[:, 1] <= 1.3)
        
        # Convert NDC to pixel coordinates
        visible_points_ndc = p_proj[visible]
        image_coords = torch.zeros_like(visible_points_ndc[:, :2], dtype=torch.long)
        image_coords[:, 1] = ((visible_points_ndc[:, 0] + 1) * 0.5) * image_width
        image_coords[:, 0] = ((visible_points_ndc[:, 1] + 1) * 0.5) * image_height  # Flip y-axis for image coordinates

        return visible, image_coords
    
    def compute_gaussians_label(self, means_2d, semantic_mask):
        # Given the visible gaussians 2d position attribute each one a semantic class
        # Takes as input the means_2d (image position) and the original semantic mask 

        # Ensure the x and y coordinates are within the image dimensions
        clamped_y = torch.clamp(means_2d[:, 0], 0, semantic_mask.shape[0] - 1)
        clamped_x = torch.clamp(means_2d[:, 1], 0, semantic_mask.shape[1] - 1)
        
        # Compute flat indices using clamped values
        flat_indices = clamped_y.long() * semantic_mask.shape[1] + clamped_x.long()
        
        # Use the flat indices to get the labels from the semantic mask
        gaussian_labels = semantic_mask.view(-1)[flat_indices]

        return gaussian_labels
    
    def create_color_semantic_mask(self, semantic_mask, color_mapping):
         # Prepare an empty array for the color image
        height, width = semantic_mask.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert class IDs to an index array
        class_ids = np.unique(semantic_mask)
        indexes = np.digitize(semantic_mask, class_ids, right=True)

        # Vectorized assignment of colors
        for idx, class_id in enumerate(class_ids):
            color_image[indexes == idx] = color_mapping[class_id]
            
        return color_image
        
    
    def compute_colors(self, gaussian_labels, color_mapping, gaussians_mask):

        max_label = max(color_mapping.keys())
        color_mapping_tensor = torch.zeros((max_label + 1, 3), dtype=torch.uint8, device=self.device)
        for k, v in color_mapping.items():
            color_mapping_tensor[k] = torch.tensor(v, dtype=torch.uint8)
        gaussian_colors = color_mapping_tensor[gaussian_labels] / 255.

        gaussian_colors_rasterize = torch.zeros((self.gaussians.get_xyz.shape[0], 3), dtype=torch.float, device=self.device)
        gaussian_colors_rasterize[gaussians_mask] = gaussian_colors

        return gaussian_colors_rasterize


    def compute_adjacency_lists(self, points, epsilon):
            points =  points.detach().cpu().numpy()
            tree = KDTree(points)
            adjacency_lists = {}
            for i, point in enumerate(points):
                # Find all points within epsilon distance, including the point itself
                indices = tree.query_ball_point(point, epsilon)
                # Remove self-connections
                indices = [idx for idx in indices if idx != i]
                # Store in adjacency list
                adjacency_lists[i] = indices
            return adjacency_lists
    
    def compute_binary_masks(self, gaussians_labels, mask, total_no_gaussians):
        # takes as input the labeled gaussians returnes the colors for each mask
        # output: C x mask_colors
        # the mask_colors will have value 1 on every channel if the gaussian was assigend to that specific class if not
        mask_colors = torch.zeros(self.num_sem_cls, total_no_gaussians, 3).to(self.device)

        unique_vals, inverse_indices = torch.unique(gaussians_labels, return_inverse=True)

        one_hot_encoded = torch.zeros(self.num_sem_cls, gaussians_labels.shape[0]).to(self.device)

        one_hot_encoded.scatter_(0, inverse_indices.unsqueeze(0), 1).to(self.device)

        mask_colors[:, mask, ...] = one_hot_encoded.unsqueeze(-1)

        return mask_colors
    
    def compute_sh2color(self, pc, viewpoint_camera, mask):
        shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz[mask] - viewpoint_camera.camera_center.repeat(pc.get_features[mask].shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        return colors_precomp
    
    def get_semantic_map(self):

        with open(self.semantic_mapping, 'r') as file:
            data = json.load(file)

        return data
    
    def get_color_map(self):

        return self.color_mapping


if __name__ == '__main__':
    test_dataset = ScanNetpp_dataset('/media/mihnea/0C722848722838BA/ScanNet++2/undistorted_dataset/56a0ec536c', device='cuda:0', eval_mode='test')    
    test_dataset.__getitem__(0)    
    

