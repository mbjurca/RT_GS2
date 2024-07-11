import numpy as np
import torch
import sys, os
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import gc
import time 
sys.path.append(f'{sys.path[0]}/../..')
from utils.dataset_utils import transform_point_4x4, transform_point_4x3#, #compute_cov2d, assign_semantic_labels

from models.gaussian_splatting.gaussian_renderer import GaussianModel
from models.gaussian_splatting.scene import Scene


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

def construct_list_of_attributes(features_dc, features_rest, scaling, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

def save_ply(gaussians, mask, path):

    xyz = gaussians.get_xyz[mask].detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    f_dc = gaussians.get_features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians.get_features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians.get_opacity_no_activation[mask].detach().cpu().numpy()
    scale = gaussians.get_scaling_no_activation[mask].detach().cpu().numpy()
    rotation = gaussians.get_rotation_no_activation[mask].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc=f_dc, features_rest=f_rest, scaling=scale, rotation=rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = torch.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = torch.tensor(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = torch.tensor(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = torch.tensor(plydata.elements[0]["f_dc_2"]) 

    return torch.tensor(xyz), features_dc

def frastrum_culling_gaussians(points, view_matrix, projection_matrix, image_height, image_width):

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

def hash_point(point, precision=5):
    """Hash a 3D point based on its spatial location with a given precision."""
    return tuple(np.round(point, decimals=precision))

def build_hash_table(points):
    """Build a hash table for a set of 3D points."""
    hash_table = {}
    for index, point in enumerate(points):
        point_hash = hash_point(point)
        # Assuming unique points or only caring about one of the duplicates
        hash_table[point_hash] = index
    return hash_table

def extract_contrastive_data(train_scene_dir, test_scene_dir, root_dir):

    output_path_train = os.path.join(root_dir, 'train_contrastive')
    output_path_test = os.path.join(root_dir, 'test_contrastive')

    os.makedirs(output_path_train, exist_ok=True)
    os.makedirs(output_path_test, exist_ok=True)

    for stage in ['Test', 'Train']:

        list_models = os.listdir(train_scene_dir) if stage == 'Train' else os.listdir(test_scene_dir)
        path = output_path_train if stage == 'Train' else output_path_test

        for scene_idx, scene_name in tqdm(enumerate(list_models), desc=f'Scene processing in stage {stage}'):
            scene_dir = os.path.join(root_dir, stage, scene_name)
            dataset = ModelParams(scene_dir)
            gaussians = GaussianModel(3)
            scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
            views = scene.getTrainCameras() + scene.getTestCameras()

            for idx_view_1 in range(0, len(views) - 1, 80): # -1 ensures the last view is not compared with itself (as it would have no subsequent views)
                view_1 = views[idx_view_1]
                mask_1, _ = frastrum_culling_gaussians(points=gaussians.get_xyz, 
                                                view_matrix=view_1.world_view_transform, 
                                                projection_matrix=view_1.full_proj_transform, 
                                                image_height=view_1.image_height, 
                                                image_width=view_1.image_width)
                last_saved_idx = 0
                for idx_view_2 in range(idx_view_1 + 1, len(views)):
                    view_2 = views[idx_view_2]

                    mask_2, _ = frastrum_culling_gaussians(points=gaussians.get_xyz, 
                                                    view_matrix=view_2.world_view_transform, 
                                                    projection_matrix=view_2.full_proj_transform, 
                                                    image_height=view_2.image_height, 
                                                    image_width=view_2.image_width)
                    
                    if (mask_1 & mask_2).sum() / (mask_1 | mask_2).sum() > 0.3 and (mask_1 & mask_2).sum() / (mask_1 | mask_2).sum() < 0.8:
                        if last_saved_idx + 25 <= idx_view_2:   
                            # Build hash tables for xyz1 and xyz2
                            xyz1 = gaussians.get_xyz[mask_1].detach().cpu().numpy()
                            xyz2 = gaussians.get_xyz[mask_2].detach().cpu().numpy()
                            hash_table_xyz1 = build_hash_table(xyz1)
                            hash_table_xyz2 = build_hash_table(xyz2)

                            mask_intersection = mask_1 & mask_2
                            xyz_intersection = gaussians.get_xyz[mask_intersection].detach().cpu().numpy()

                            # Find correspondences by looking up each point in xyz_intersection in the hash tables
                            correspondences = []
                            for point in xyz_intersection:
                                point_hash = hash_point(point)
                                if point_hash in hash_table_xyz1 and point_hash in hash_table_xyz2:
                                    index_1 = hash_table_xyz1[point_hash]
                                    index_2 = hash_table_xyz2[point_hash]
                                    correspondences.append((index_1, index_2))
                            if not os.path.exists(os.path.join(path, f'scene_{scene_idx}_view_{idx_view_1}_view_{idx_view_2}.npz')):
                                np.savez(os.path.join(path, f'scene_{scene_idx}_view_{idx_view_1}_view_{idx_view_2}.npz'), 
                                    mask_1=mask_1.detach().cpu().numpy(), 
                                    mask_2=mask_2.detach().cpu().numpy(), 
                                    correspondences=np.array(correspondences))
                            last_saved_idx = idx_view_2

            del scene, views, gaussians, dataset
            gc.collect()
            torch.cuda.empty_cache()
                    
def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys
                    
def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):

    coord = coord.detach().cpu().numpy()
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count
    
#save ply file with the voxelized point clouds
def save_ply_voxelized(xyz, path):
    structured_xyz = np.array([tuple(point) for point in xyz], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # Create a PlyElement object from the structured array
    el = PlyElement.describe(structured_xyz, 'vertex')
    
    # Write to file
    PlyData([el]).write(path)

    

def voxelize_point_clouds(xyz, mask_1, mask2, voxel_size=0.07):

    pc_1 = xyz[mask_1]
    pc_2 = xyz[mask_2]

    save_ply_voxelized(pc_1, 'initial_view_0.ply')
    save_ply_voxelized(pc_2, 'initial_view_1.ply')
     
    idx_1 = voxelize(pc_1, voxel_size=voxel_size)
    idx_2 = voxelize(pc_2, voxel_size=voxel_size)
    print(np.unique(idx_1).shape, np.unique(idx_2).shape)

    save_ply_voxelized(pc_1[idx_1], 'voxelized_view_0.ply')
    save_ply_voxelized(pc_2[idx_2], 'voxelized_view_1.ply')
     
     
def convert_dc_to_color(features_dc):
    # Normalize and scale features_dc from -1 to 1 range to 0 to 255 range
    features_dc = features_dc * 0.28209479177387814 + 0.5
    features_min = features_dc.min()
    features_max = features_dc.max()
    colors = (features_dc - features_min) / (features_max - features_min) * 255.0
    colors = colors.int()  # Convert to integer RGB values
    return colors

def save_ply_with_colors(xyz, colors, filename='point_cloud_with_colors.ply'):

    print(colors)
    # Create a structured numpy array to hold xyz and color data
    vertices = np.array([(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(xyz, colors)],
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Create the PlyElement object and write to file
    ply_element = PlyElement.describe(vertices, 'vertex')
    PlyData([ply_element], text=True).write(filename)

    print(f"Saved point cloud to '{filename}'")

   
if __name__ == '__main__':

    train_scene_dir = '/media/mihnea/Elements/mihnea/ScanNet/Train'
    test_scene_list = '/media/mihnea/Elements/mihnea/ScanNet/Test'
    root_dir = '/media/mihnea/Elements/mihnea/ScanNet'

    extract_contrastive_data(train_scene_dir=train_scene_dir,
                        test_scene_dir=test_scene_list, 
                        root_dir=root_dir)
    


    